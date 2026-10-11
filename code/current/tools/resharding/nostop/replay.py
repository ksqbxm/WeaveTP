#!/usr/bin/env python3
"""模拟请求时间线 + 实测切换轨迹. CPU-only projection, not measured serving."""

import argparse
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from . import artifacts, simulator, trace_metrics
else:
    import artifacts
    import simulator
    import trace_metrics


LABEL = "模拟请求时间线 + 实测切换轨迹"


def replay_launch(sim, requests, launch, b_max=256):
    offsets = launch["completion_offsets_s"]
    if (len(offsets) < 2 or offsets[0] != 0 or not all(math.isfinite(t) for t in offsets)
            or any(b <= a for a, b in zip(offsets, offsets[1:]))):
        raise ValueError("need strictly increasing measured completion offsets starting at zero")
    first_long = next(q for q in requests if q.kind == "long")

    class Replay(sim.Sim):
        def __init__(self):
            super().__init__(requests, "weavetp", b_max=b_max,
                             plan=[("before_admit", first_long.rid, "2->4", first_long.seg)])
            self.last = {}
            self.maximum = {}
            self.counts = {}
            self.affected = set()

        def observe(self, count=1, interval=0):
            # Compressed decode blocks have constant cadence and no interior departures.
            first = self.T - (count - 1) * interval
            for replica in self.reps:
                for q in replica.reqs.values():
                    previous = self.last.get(q.rid)
                    gap = first - previous if previous is not None else None
                    if gap is not None and gap <= 0:
                        raise ValueError("non-increasing simulated token completions")
                    candidates = [v for v in (self.maximum.get(q.rid), gap,
                                              interval if count > 1 else None) if v is not None]
                    self.maximum[q.rid] = max(candidates, default=None)
                    self.last[q.rid] = self.T
                    self.counts[q.rid] = self.counts.get(q.rid, 0) + count

        def _complete(self, step):
            self.observe()
            super()._complete(step)

        def decode_until(self, last_step):
            start = self.T
            while self.s <= last_step:
                finishes = [r.next_finish() for r in self.reps if r.next_finish() is not None]
                if not finishes:
                    self.s = last_step + 1
                    break
                end = min(min(finishes), last_step)
                count = end - self.s + 1
                interval = sim.t_dec(max(r.n for r in self.reps), self.tp, max(r.used(self.s) for r in self.reps))
                self._add(self._decode_cat(), count * interval, count * sum(r.n for r in self.reps))
                self.T += count * interval
                self.observe(count, interval)
                self.s = end
                sim.Sim._complete(self, end)
                self.s += 1
            return self.T - start

        def admit_step(self, stop_rid):
            pf, admitted, reason = super().admit_step(stop_rid)
            # Finish the prefill step when the same call admits requests AND reaches
            # the anchor. Otherwise the simulator skips their first token entirely.
            return pf, admitted, None if admitted and reason == "anchor" else reason

        def feasible(self, to_tp):
            # Common pre-window feasibility for every trajectory; no historical
            # sw duration/overlap constants or method-specific pre-window drain.
            return self.peak_ok(to_tp, 0) and all(r.resv() <= self.cap(to_tp) for r in sim.regroup(self.reps, to_tp))

        def do_switch(self, to_tp, anchor):
            if self.tp != 2 or to_tp != 4 or self.res.switches:
                raise ValueError("replay supports exactly one expansion")
            start, initial_step = self.T, self.s
            self.affected = {q.rid for r in self.reps for q in r.reqs.values()}
            for i, offset in enumerate(offsets[1:], 1):
                if i == len(offsets) - 1:
                    self.reps, self.tp = sim.regroup(self.reps, 4), 4
                self._add("measured_switch_window", start + offset - self.T, sum(r.n for r in self.reps))
                self.T = start + offset
                self._complete(self.s)
                self.s += 1
            self.res.switches.append({"dir": "2->4", "at_s": start, "step": initial_step,
                                      "wall": offsets[-1], "anchor": anchor,
                                      "completion_times_s": [start + v for v in offsets]})

    runner = Replay()
    result = runner.run()
    if len(result.switches) != 1 or any(runner.counts.get(q.rid) != q.out for q in result.reqs):
        raise ValueError("request token counts or single-expansion invariant failed")
    rows = [{"rid": q.rid, "kind": q.kind, "out_tokens": q.out,
             "completed_tokens": runner.counts[q.rid], "max_tbt_s": runner.maximum[q.rid],
             "active_at_switch": q.rid in runner.affected,
             "first_token_s": q.t_first, "last_token_s": q.t_done} for q in result.reqs]
    values = [r["max_tbt_s"] for r in rows if r["max_tbt_s"] is not None]
    return {**{k: launch[k] for k in (*trace_metrics.GROUP, "launch_id")}, "label": LABEL,
            "requests": rows, "request_count": len(rows), "eligible_requests": len(values),
            "affected_requests": len(runner.affected), "request_max_tbt_s": trace_metrics.distribution(values),
            "request_slo": trace_metrics.slo(values), "switch": result.switches[0], "time_s": result.time_s}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--simulator", default=simulator.SIMULATOR, type=Path)
    parser.add_argument("--lengths", default=simulator.PROFILE_DATA / "lengths.csv", type=Path)
    parser.add_argument("--longout", default=simulator.PROFILE_DATA / "lengths_longout.csv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = {k: getattr(args, k) for k in ("summary", "simulator", "lengths", "longout")}
    missing = {k: str(p) for k, p in inputs.items() if not p.is_file()}
    if missing:
        print(json.dumps({"status": "missing_inputs", "required_paths": missing, "label": LABEL}, ensure_ascii=False))
        return 2
    sim = simulator.load_simulator(args.simulator)
    short, longs = sim.load_pools(str(args.lengths), str(args.longout))
    requests = sim.build_workload(short, longs["sky_t1_17k"], cycles=1, n_short=4000, n_long=600, seed=7)
    summary = artifacts.read_json(args.summary)
    report = {"label": LABEL, "world": 8, "b_max": 256,
              "workload": {"cycles": 1, "n_short": 4000, "n_long": 600, "seed": 7, "source": "sky_t1_17k"},
              "sources": {k: {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in inputs.items()},
              "assumptions": ["all requests arrive at zero; simulator FCFS/prefill/decode outside the window",
                              "one expansion before first long request; remain TP4 thereafter",
                              "unscaled measured cadence applied to active simulated requests; no admissions in window",
                              "common pre-window capacity/peak check at zero growth; not a GPU capacity prediction",
                              "SLO denominator: all requests with >=2 output tokens; TTFT excluded"],
              "input_errors": summary["errors"], "launches": []}
    args.output.mkdir(parents=True, exist_ok=True)
    for launch in summary["launches"]:
        row = replay_launch(sim, requests, launch)
        requests_path = args.output / f"requests_{launch['launch_id']}.csv"
        artifacts.write_csv(requests_path, row.pop("requests"),
                  ["rid", "kind", "out_tokens", "completed_tokens", "max_tbt_s", "active_at_switch", "first_token_s", "last_token_s"])
        row["requests_csv"] = str(requests_path.resolve())
        report["launches"].append(row)
    artifacts.write_json(args.output / "replay.json", report)
    flat = [{**{k: r[k] for k in (*trace_metrics.GROUP, "launch_id", "label", "eligible_requests", "affected_requests")},
             "p50_s": r["request_max_tbt_s"]["p50"], "p99_s": r["request_max_tbt_s"]["p99"],
             **{f"slo_{t}s": v for t, v in r["request_slo"].items()}} for r in report["launches"]]
    artifacts.write_csv(args.output / "replay.csv", flat, [*trace_metrics.GROUP, "launch_id", "label", "eligible_requests", "affected_requests",
                                                "p50_s", "p99_s", *[f"slo_{t}s" for t in trace_metrics.slo([])]])
    print(json.dumps({"launches": len(flat), "label": LABEL, "output": str(args.output)}, ensure_ascii=False))
    return 1 if summary["errors"] or not flat else 0


if __name__ == "__main__":
    raise SystemExit(main())
