#!/usr/bin/env python3
"""Exactly two diagnostic decisions; no formal summaries or request replay."""

import argparse
import json
import math
import shlex
import statistics
from pathlib import Path

try:
    from .common import MAX_WAVES, capacity_budget, main_links
except ImportError:
    from common import MAX_WAVES, capacity_budget, main_links


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
            if row["switch"] == measured and row["phase"] in ("normal", "base"):
                by_step.setdefault(row["step"], []).append(row)
        for wave in trace["waves"]:
            if wave["switch"] == measured:
                by_wave.setdefault(wave["wave"], []).append((trace["rank"], wave))
    if not by_step or any(len(rows) != 8 or {r["rank"] for r in rows} != set(range(8)) for rows in by_step.values()):
        return {"status": "inconclusive", "reason": "incomplete logical steps"}
    previous, gaps = None, {}
    for _, rows in sorted(by_step.items()):
        if len({(r["phase"], r["wave"], r["kv_offset"]) for r in rows}) != 1:
            return {"status": "inconclusive", "reason": "rank logical steps disagree"}
        completion = max(r["complete_host_ns"] for r in rows)
        if previous is not None:
            if completion <= previous:
                return {"status": "inconclusive", "reason": "non-increasing completion times"}
            if rows[0]["phase"] == "base":
                gaps.setdefault(rows[0]["wave"], []).append(completion - previous)
        previous = completion
    blocking = {}
    for wave, entries in by_wave.items():
        durations = []
        for rank, entry in entries:
            starts = [r["host_start_ns"] for rows in by_step.values() for r in rows
                      if r["rank"] == rank and r["phase"] == "base" and r["wave"] == wave]
            if starts:
                durations.append(min(starts) - entry["submit_host_ns"])
        if len(durations) == 8 and {rank for rank, _ in entries} == set(range(8)) and min(durations) >= 0:
            blocking[wave] = max(durations)
    if (0 not in gaps or 0 not in blocking or len(gaps) < 2
            or gaps.keys() != blocking.keys() or gaps.keys() != by_wave.keys()):
        return {"status": "inconclusive", "reason": "need first and later waves with decode on all ranks"}
    wave_gaps = {wave: max(values) for wave, values in gaps.items()}
    gap_excess = wave_gaps[0] - statistics.median(v for k, v in wave_gaps.items() if k != 0)
    blocking_excess = blocking[0] - statistics.median(v for k, v in blocking.items() if k != 0)
    return {"status": "failed" if gap_excess > 100_000_000 or blocking_excess > 50_000_000 else "passed",
            "first_wave_gap_excess_s": gap_excess / 1e9, "first_wave_host_blocking_excess_s": blocking_excess / 1e9,
            "wave_max_gap_s": {k: v / 1e9 for k, v in wave_gaps.items()},
            "wave_submit_to_decode_host_s": {k: v / 1e9 for k, v in blocking.items()},
            "thresholds_s": {"gap_excess": 0.100, "host_blocking_excess": 0.050}}


def pressure_check(low, high, default_rows, candidate_rows):
    if not low.get("complete") or not high.get("complete"):
        return {"status": "inconclusive", "reason": "incomplete bandwidth probe"}
    maps = [{(r["src"], r["dst"]): r for r in p["links"]} for p in (low, high)]
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
    selected = main_links(default_rows)
    costs = [sum(r["bytes"] / m[(r["src"], r["dst"])]["median_gbps"] for r in selected) for m in maps]
    ratio = costs[1] / costs[0]
    total_bytes = sum(r["bytes"] for r in selected)
    return {"status": "passed" if ratio >= 2.5 else "failed", "required_slowdown": 2.5,
            "l0_effective_gbps": total_bytes / costs[0], "l3_effective_gbps": total_bytes / costs[1],
            "b0_weighted_slowdown": ratio, "b0_primary_links": selected, "links": details,
            "aggregation": "fixed B0 byte-weighted harmonic bandwidth, >=80% of remote Base bytes",
            "note": "W2 candidate links are reported, not required to meet 2.5x individually"}


def launch_gate(traces, mode, stall_pause):
    """Validate diagnostic evidence without imposing any candidate-plan win."""
    expected = 2 if mode == "cycle" else 0
    if len(traces) != 8 or {t["rank"] for t in traces} != set(range(8)):
        return "missing rank traces"
    for t in traces:
        if t["measured_switch"] != expected or t["stall_pause"] != stall_pause or not any(
                s["measured"] and s["kv_tokens"] == 2048 and s["target_weight_storage_bytes"] == 0
                and s["target_weight_bytes_required"] > 0 for s in t["state_checks"]):
            return "KV or standby state gate failed"
        if mode == "independent" and not t.get("independent_warmup"):
            return "missing independent warmup evidence"
        migration = [r for r in t["decodes"] if r["switch"] == expected
                     and r["phase"] in ("base", "weight_check", "delta", "prepare")]
        if (stall_pause and migration) or (not stall_pause and not migration):
            return "STW/no-stop decode invariant failed"
        if len([r for r in t["decodes"] if r["switch"] == expected and r["phase"] == "post"]) != 10:
            return "missing ten completed target steps"
    return None


def load_traces(directory):
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("trace_rank*.json"))]


def make_report(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    checks = {}
    smoke_errors = {}
    for method in ("W2", "B0"):
        case = root / f"smoke_L0_{method}"
        if not (case / "exit_code.txt").exists() or (case / "exit_code.txt").read_text().strip() != "0":
            smoke_errors[method] = "smoke launch incomplete"
        else:
            try:
                reason = launch_gate(load_traces(case / "run"), "cycle", method == "B0")
            except (ValueError, KeyError) as error:
                reason = f"invalid smoke evidence: {error}"
            if reason:
                smoke_errors[method] = reason
    for level in ("L0", "L3"):
        for mode in ("cycle", "independent"):
            name = f"{level}_W2_{mode}"
            case = root / name
            if not (case / "exit_code.txt").exists() or (case / "exit_code.txt").read_text().strip() != "0":
                checks[name] = {"status": "inconclusive", "reason": "launch did not complete successfully"}
                continue
            try:
                traces = load_traces(case / "run")
                checks[name] = cold_check(traces)
                reason = launch_gate(traces, mode, False)
            except (ValueError, KeyError) as error:
                reason = f"invalid launch evidence: {error}"
            if reason:
                checks[name] = {"status": "inconclusive", "reason": reason}
    cycle_ok = all(checks[f"{level}_W2_cycle"]["status"] == "passed" for level in ("L0", "L3"))
    independent_ok = all(checks[f"{level}_W2_independent"]["status"] == "passed" for level in ("L0", "L3"))
    recommended = None if smoke_errors else "cycle" if cycle_ok else "independent" if independent_ok else None
    try:
        routes = json.loads((root / "probe_routes.json").read_text())
        pressure = pressure_check(json.loads((root / "probe_L0.json").read_text()),
                                  json.loads((root / "probe_L3.json").read_text()),
                                  routes["default"], routes["candidate"])
    except (FileNotFoundError, ValueError, KeyError) as error:
        pressure = {"status": "inconclusive", "reason": f"missing/invalid bandwidth evidence: {error}"}
    if not manifest["execution_complete"]:
        pressure["status"] = "inconclusive"
        pressure["reason"] = "diagnostic execution, pressure/telemetry or final memory recovery did not complete successfully; see status.json"
    cold_status = "passed" if recommended else "not_isolated" if all(
        c["status"] != "inconclusive" for c in checks.values()) and not smoke_errors else "inconclusive"
    report = {"cold_start": {"status": cold_status,
                              "recommended_mode": recommended, "checks": checks, "smoke_errors": smoke_errors},
              "pressure": pressure}
    (root / "diag_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    env = {"DIAGNOSTICS_READY": int(recommended is not None and pressure["status"] == "passed"),
           "NOSTOP_COLD_MODE": recommended or "cycle", "SWITCHES": 1 if recommended == "independent" else 3,
           "MAX_WAVES": MAX_WAVES, "SEQ_LENGTH": capacity_budget()["max_position_embeddings"],
           "MAX_POSITION_EMBEDDINGS": capacity_budget()["max_position_embeddings"],
           "STRESS_LINKS": manifest["stress_links"], "L3_PROCESSES_PER_PAIR": manifest["l3_processes"],
           "STRESS_BYTES": manifest["stress_bytes"], "STRESS_SYNC_EVERY": manifest["stress_sync_every"],
           "PRESSURE_REQUIRED_SLOWDOWN": 2.5, "PROBE_CV_FLAG_THRESHOLD": 0.15,
           "PROBE_SHIFT_FLAG_THRESHOLD": 0.20,
           "PROFILE": manifest["profile"], "DIAGNOSTIC_EVIDENCE": str(root.resolve())}
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
