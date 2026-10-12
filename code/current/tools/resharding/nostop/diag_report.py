#!/usr/bin/env python3
"""Exactly two diagnostic decisions; no formal summaries or request replay."""

import argparse
import json
import math
import shlex
import statistics
from pathlib import Path

if __package__:
    from . import common, memory_budget, trace_metrics
else:
    import common
    import memory_budget
    import trace_metrics


def cold_check(traces):
    if len(traces) != 8 or {r["rank"] for r in traces} != set(range(8)):
        return {"status": "inconclusive", "reason": "missing rank traces"}
    if any(t["clock"] != traces[0]["clock"] for t in traces) or not traces[0]["clock"].get("monotonic"):
        return {"status": "inconclusive", "reason": "incomparable rank clocks"}
    measured = traces[0]["measured_switch"]
    if any(t["measured_switch"] != measured for t in traces):
        return {"status": "inconclusive", "reason": "different measured switches"}
    by_step, by_wave = {}, {}
    for trace in traces:
        for row in trace["decodes"]:
            # Validation replays source tokens, so it is not a serving completion.
            # Keep post/normal steps as predecessors, including for warmup shrink.
            if row["phase"] != "validation":
                by_step.setdefault(row["step"], []).append(row)
        for wave in trace["waves"]:
            by_wave.setdefault((wave["switch"], wave["wave"]), []).append((trace["rank"], wave))
    if not by_step or any(len(rows) != 8 or {r["rank"] for r in rows} != set(range(8)) for rows in by_step.values()):
        return {"status": "inconclusive", "reason": "incomplete logical steps"}
    previous, gaps = None, {}
    for _, rows in sorted(by_step.items()):
        if len({(r["switch"], r["phase"], r["wave"], r["kv_offset"]) for r in rows}) != 1:
            return {"status": "inconclusive", "reason": "rank logical steps disagree"}
        completion = max(r["complete_host_ns"] for r in rows)
        if previous is not None:
            if completion <= previous:
                return {"status": "inconclusive", "reason": "non-increasing completion times"}
            if rows[0]["phase"] == "base":
                gaps.setdefault((rows[0]["switch"], rows[0]["wave"]), []).append(completion - previous)
        previous = completion
    blocking = {}
    for key, entries in by_wave.items():
        durations = []
        for rank, entry in entries:
            starts = [r["host_start_ns"] for rows in by_step.values() for r in rows
                      if r["rank"] == rank and r["phase"] == "base" and (r["switch"], r["wave"]) == key]
            if starts:
                durations.append(min(starts) - entry["submit_host_ns"])
        if len(durations) == 8 and {rank for rank, _ in entries} == set(range(8)) and min(durations) >= 0:
            blocking[key] = max(durations)
    if not gaps or gaps.keys() != blocking.keys() or gaps.keys() != by_wave.keys():
        return {"status": "inconclusive", "reason": "need complete wave/decode evidence on all ranks"}
    switches = {}
    for (switch, wave), values in sorted(gaps.items()):
        row = switches.setdefault(switch, {"wave_decode_gaps_s": {}, "wave_max_gap_s": {}, "wave_submit_to_decode_host_s": {}})
        row["wave_decode_gaps_s"][wave] = [v / 1e9 for v in values]
        row["wave_max_gap_s"][wave] = max(values) / 1e9
        row["wave_submit_to_decode_host_s"][wave] = blocking[(switch, wave)] / 1e9
    result = {"switches": switches, "criterion": "cycle switch 2 first-wave maximum gap <= switch 0",
              "fixed_switch_overhead": "retained in raw decode gaps; no subtraction or comparison against later waves"}
    if measured == 0:
        return {**result, "status": "reference", "reason": "independent mode is reference only; frozen mode is cycle"}
    if measured != 2 or set(switches) != {0, 1, 2} or (0, 0) not in gaps or (2, 0) not in gaps:
        return {**result, "status": "inconclusive", "reason": "need cold switch 0 and measured switch 2 in a complete cycle"}
    cold, warm = max(gaps[(0, 0)]), max(gaps[(2, 0)])
    return {**result, "status": "passed" if warm <= cold else "failed",
            "cold_first_wave_gap_s": cold / 1e9, "measured_first_wave_gap_s": warm / 1e9,
            "cold_start_reduction_s": (cold - warm) / 1e9}


def pressure_check(low, high, default_rows, candidate_rows):
    if not low.get("complete") or not high.get("complete"):
        return {"status": "inconclusive", "reason": "incomplete bandwidth probe"}
    maps = [{(r["src"], r["dst"]): dict(r) for r in p["links"]} for p in (low, high)]
    details = []
    union = {(r["src"], r["dst"]) for r in default_rows + candidate_rows if r["src"] != r["dst"]}
    for pair in sorted(union):
        if any(pair not in m for m in maps):
            return {"status": "inconclusive", "reason": f"unprobed migration link {pair}"}
        a, b = (m[pair] for m in maps)
        for row in (a, b):
            samples = [s["gbps"] for s in row.get("samples", [])]
            if len(samples) != 5 or not all(math.isfinite(v) and v > 0 for v in samples):
                return {"status": "inconclusive", "reason": "need all five valid raw probe samples"}
            row["median_gbps"] = statistics.median(samples)
            row["cv"] = statistics.pstdev(samples) / statistics.mean(samples)
            row["end_to_start_change"] = statistics.mean(samples[-2:]) / statistics.mean(samples[:2]) - 1
        if not all(math.isfinite(v) and v > 0 for v in (a["median_gbps"], b["median_gbps"])):
            return {"status": "inconclusive", "reason": "invalid bandwidth"}
        details.append({"src": pair[0], "dst": pair[1], "l0_gbps": a["median_gbps"],
                        "l3_gbps": b["median_gbps"], "slowdown": a["median_gbps"] / b["median_gbps"],
                        "l0_cv": a["cv"], "l3_cv": b["cv"],
                        "l0_end_to_start_change": a["end_to_start_change"],
                        "l3_end_to_start_change": b["end_to_start_change"],
                        "variability_flag": any(r["cv"] > 0.15 or abs(r["end_to_start_change"]) > 0.20 for r in (a, b))})
    selected = common.main_links(default_rows)
    costs = [sum(r["bytes"] / m[(r["src"], r["dst"])]["median_gbps"] for r in selected) for m in maps]
    ratio = costs[1] / costs[0]
    total_bytes = sum(r["bytes"] for r in selected)
    return {"status": "passed" if ratio >= 2.5 else "failed", "required_slowdown": 2.5,
            "l0_effective_gbps": total_bytes / costs[0], "l3_effective_gbps": total_bytes / costs[1],
            "b0_weighted_slowdown": ratio, "b0_primary_links": selected, "links": details,
            "aggregation": "fixed B0 byte-weighted harmonic bandwidth, >=80% of remote Base bytes",
            "note": "W2 candidate links are reported, not required to meet 2.5x individually"}


def make_report(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks = {}
    smoke_errors = {}
    for method in ("W2", "B0"):
        case = root / f"smoke_L0_{method}"
        if not (case / "exit_code.txt").exists() or (case / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
            smoke_errors[method] = "smoke launch incomplete"
        else:
            try:
                traces = trace_metrics.load_traces(case / "run")
                reason = trace_metrics.launch_gate(traces, "cycle", method == "B0")
                if method == "W2" and not reason:
                    checks["smoke_L0_W2"] = cold_check(traces)
            except (ValueError, KeyError) as error:
                reason = f"invalid smoke evidence: {error}"
            if reason:
                smoke_errors[method] = reason
    for level in ("L0", "L3"):
        for mode in ("cycle", "independent"):
            name = f"{level}_W2_{mode}"
            case = root / name
            if mode == "independent" and not case.exists():
                continue  # Independent evidence is optional, never rerun to freeze cycle.
            if not (case / "exit_code.txt").exists() or (case / "exit_code.txt").read_text(encoding="utf-8").strip() != "0":
                checks[name] = {"status": "inconclusive", "reason": "launch did not complete successfully"}
                continue
            try:
                traces = trace_metrics.load_traces(case / "run")
                checks[name] = cold_check(traces)
                reason = trace_metrics.launch_gate(traces, mode, False)
            except (ValueError, KeyError) as error:
                reason = f"invalid launch evidence: {error}"
            if reason:
                checks[name] = {"status": "inconclusive", "reason": reason}
    cycle_ok = all(checks[f"{level}_W2_cycle"]["status"] == "passed" for level in ("L0", "L3"))
    recommended = "cycle" if cycle_ok and not smoke_errors else None
    try:
        routes = json.loads((root / "probe_routes.json").read_text(encoding="utf-8"))
        pressure = pressure_check(json.loads((root / "probe_L0.json").read_text(encoding="utf-8")),
                                  json.loads((root / "probe_L3.json").read_text(encoding="utf-8")),
                                  routes["default"], routes["candidate"])
    except (FileNotFoundError, ValueError, KeyError) as error:
        pressure = {"status": "inconclusive", "reason": f"missing/invalid bandwidth evidence: {error}"}
    if not manifest["execution_complete"]:
        pressure["status"] = "inconclusive"
        pressure["reason"] = "diagnostic execution, pressure/telemetry or final memory recovery did not complete successfully; see status.json"
    budget = manifest["memory_budget"]
    pressure["memory_budget"] = budget
    if budget is None:
        pressure.update(status="inconclusive", reason="new L0 torch memory measurements not yet available")
    elif not budget["feasible"]:
        pressure.update(status="failed", reason="measured model peak leaves no positive L3 count with 2 GiB headroom")
    cold_status = "passed" if recommended else "not_isolated" if all(
        checks[f"{level}_W2_cycle"]["status"] != "inconclusive" for level in ("L0", "L3")) and not smoke_errors else "inconclusive"
    report = {"cold_start": {"status": cold_status,
                              "recommended_mode": "cycle", "switches": 3, "checks": checks, "smoke_errors": smoke_errors},
              "pressure": pressure}
    (root / "diag_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    env = {"DIAGNOSTICS_READY": int(recommended is not None and pressure["status"] == "passed"),
           "NOSTOP_COLD_MODE": "cycle", "SWITCHES": 3,
           "MAX_WAVES": common.MAX_WAVES, "SEQ_LENGTH": common.capacity_budget()["max_position_embeddings"],
           "MAX_POSITION_EMBEDDINGS": common.capacity_budget()["max_position_embeddings"],
           "STRESS_LINKS": manifest["stress_links"],
           "STRESS_BYTES": manifest["stress_bytes"], "STRESS_SYNC_EVERY": manifest["stress_sync_every"],
           "PRESSURE_REQUIRED_SLOWDOWN": 2.5, "PROBE_CV_FLAG_THRESHOLD": 0.15,
           "PROBE_SHIFT_FLAG_THRESHOLD": 0.20,
           "PROFILE": manifest["profile"], "DIAGNOSTIC_EVIDENCE": str(root.resolve())}
    if budget is not None:
        env.update(memory_budget.frozen_values(budget))
        count = manifest["l3_processes"] if manifest["l3_processes"] is not None else budget["counts"]["L3"]
        env.update({f"{level}_PROCESSES_PER_PAIR": n for level, n in memory_budget.scaled_counts(count).items()})
    lines = ["# Recommendation only. Does not launch formal experiments.",
             "# NCCL_RUNTIME_CONNECT is intentionally not modified."]
    lines += [f"export {k}={shlex.quote(str(v))}" for k, v in env.items()]
    (root / "frozen.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(make_report(args.directory), indent=2))


if __name__ == "__main__":
    main()
