#!/usr/bin/env python3
"""CPU-only Restart reference: measured/recorded load + estimated KV rebuild + measured TP4 first step."""

import argparse
import json
import math
import re
from pathlib import Path

try:
    from .replay import SIMULATOR, load_simulator
    from .summarize import GROUP, read_json, write_csv, write_json
except ImportError:
    from replay import SIMULATOR, load_simulator
    from summarize import GROUP, read_json, write_csv, write_json


def loading_time(log):
    """Parse only explicit Megatron load-checkpoint timers (milliseconds)."""
    if log is None:
        return 35.0, {"kind": "recorded_reference", "seconds": 35.0, "note": "no measured load-checkpoint timer available"}
    content = Path(log).read_text(encoding="utf-8")
    # Timers.log emits either max or (min,max) across ranks in ms. Never
    # guess which checkpoint load to use if a log contains multiple timers.
    pattern = r"load-checkpoint\s*\.*\s*:\s*(?:\(([\d.eE+-]+),\s*([\d.eE+-]+)\)|([\d.eE+-]+))"
    entries = list(re.finditer(pattern, content))
    if not entries:
        if re.search(r"load-checkpoint\s*\.*\s*:", content):
            raise ValueError(f"invalid checkpoint timer in {log}")
        return None
    if len(entries) != 1 or "time across ranks (ms):" not in content:
        raise ValueError(f"need exactly one explicitly millisecond load-checkpoint timer in {log}")
    match = entries[0]
    value = float(match.group(2) or match.group(3)) / 1000
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid checkpoint timer in {log}")
    return value, {"kind": "measured_timer", "path": str(Path(log).resolve()),
                   "line": content[:match.start()].count("\n") + 1, "text": match.group(0),
                   "selection": "single checkpoint timer; max rank, ms / 1000; loading layout as recorded in source log"}


def estimate(launch, sim, load):
    load_s, source = load
    kv_tokens = int(launch["config"]["CYCLE_KV_TOKENS"])
    batch = int(launch["config"]["MICRO_BATCH_SIZE"])
    rebuild = batch * kv_tokens / sim.REPREFILL_RATE[4]
    raw = launch["pre_trigger_s"] + load_s + rebuild + launch["tp4_first_step_s"]
    return {**{k: launch[k] for k in (*GROUP, "launch_id")}, "reference": "Restart estimate",
            "load_s": load_s, "load_source": source, "kv_rebuild_s": rebuild,
            "kv_source": {"rate_tokens_per_replica_s": sim.REPREFILL_RATE[4],
                          "kind": "simulator REPREFILL_RATE[4] estimate", "batch": batch, "kv_tokens": kv_tokens},
            "tp4_first_step_s": launch["tp4_first_step_s"],
            "tp4_source": "trace max-rank target_ready to first TP4 completion (host wall, not Event latency)",
            "pre_trigger_s": launch["pre_trigger_s"], "normal_tbt_s": launch["normal_tbt_s"],
            "raw_max_gap_s": raw, "excess_downtime_s": max(0, raw - launch["normal_tbt_s"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--simulator", default=SIMULATOR, type=Path)
    parser.add_argument("--load-log", type=Path, help="explicit target TP4 load log; missing/malformed timer is an error")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    sim = load_simulator(args.simulator)
    report = read_json(args.summary)
    rows = []
    for launch in report["launches"]:
        log = args.load_log or Path(launch["source"]).parent / f"{launch['case']}.log"
        load = loading_time(log) if log.is_file() else None
        if args.load_log and load is None:
            raise ValueError(f"no load-checkpoint timer in explicit log {args.load_log}")
        rows.append(estimate(launch, sim, load or loading_time(None)))
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "restart.json", {"simulator": str(args.simulator.resolve()), "launches": rows,
                                             "input_errors": report["errors"], "label": "estimated reference; not a GPU Restart run"})
    write_csv(args.output / "restart.csv", rows,
              [*GROUP, "launch_id", "reference", "load_s", "kv_rebuild_s", "tp4_first_step_s", "pre_trigger_s",
               "raw_max_gap_s", "normal_tbt_s", "excess_downtime_s", "load_source", "kv_source", "tp4_source"])
    print(json.dumps({"launches": len(rows), "output": str(args.output)}))
    return 1 if report["errors"] or not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
