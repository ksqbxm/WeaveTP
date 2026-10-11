#!/usr/bin/env python3
"""CPU-only, launch-first summaries of the measured expansion (never GPU execution)."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

try:
    from .common import diagnostic_cases
    from .diag_report import launch_gate
except ImportError:
    from common import diagnostic_cases
    from diag_report import launch_gate

THRESHOLDS = (1.5, 2, 3, 5, 10, 30)
MIGRATION = {"base", "prepare", "weight_check", "delta"}
GROUP = ("method", "level", "cold_mode", "cohort")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items() if k in fields})


def percentile(values, p):
    """Linear interpolation at (n - 1) * p; no pooling across launches."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def distribution(values):
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, .5), "p99": percentile(values, .99)}


def slo(values):
    return {str(t): sum(v <= t for v in values) / len(values) if values else None for t in THRESHOLDS}


def aligned_steps(traces):
    if len(traces) != 8 or {t["rank"] for t in traces} != set(range(8)):
        raise ValueError("need exactly ranks 0..7")
    clock = traces[0]["clock"]
    if (not all(clock.get(k) for k in ("host", "boot_id", "time_namespace", "implementation", "monotonic"))
            or any(t["clock"] != clock for t in traces)):
        raise ValueError("rank perf_counter clocks are not comparable")
    by_step = {}
    for trace in traces:
        seen = set()
        for row in trace["decodes"]:
            if row["step"] in seen or row["rank"] != trace["rank"]:
                raise ValueError("duplicate step or wrong rank")
            seen.add(row["step"])
            if row["phase"] != "validation":
                by_step.setdefault(row["step"], []).append(row)
    result = []
    for step, rows in sorted(by_step.items()):
        if len(rows) != 8 or len({tuple(r[k] for k in ("switch", "tp", "phase", "wave", "kv_offset")) for r in rows}) != 1:
            raise ValueError(f"incomplete or inconsistent logical step {step}")
        if any(r["complete_host_ns"] <= r["host_start_ns"] or not math.isfinite(r["event_ms"])
               or r["event_ms"] <= 0 for r in rows):
            raise ValueError(f"invalid completion/Event duration at step {step}")
        row = {k: rows[0][k] for k in ("step", "switch", "tp", "phase", "wave", "kv_offset")}
        row.update(complete_ns=max(r["complete_host_ns"] for r in rows),
                   start_ns=min(r["host_start_ns"] for r in rows),
                   event_ms=max(r["event_ms"] for r in rows))
        if result and row["complete_ns"] <= result[-1]["complete_ns"]:
            raise ValueError("non-increasing logical completion times")
        result.append(row)
    return result


def summarize_trace(traces, method, mode, result=None):
    error = launch_gate(traces, mode, method == "B0")
    if error:
        raise ValueError(error)
    measured = traces[0]["measured_switch"]
    steps = [r for r in aligned_steps(traces) if r["switch"] == measured]
    normal = [r for r in steps if r["phase"] == "normal"]
    post = [r for r in steps if r["phase"] == "post"]
    if len(normal) < 2 or not post or any(r["tp"] != 2 for r in normal) or post[0]["tp"] != 4:
        raise ValueError("need normal TP2 completions and target TP4 completion")
    marks = {}
    for name in ("switch_trigger", "base_done", "delta_done", "target_ready", "target_first_decode_complete"):
        values = [[r["host_ns"] for r in t["milestones"] if r["switch"] == measured and r["name"] == name] for t in traces]
        if any(len(v) != 1 for v in values):
            raise ValueError(f"missing/duplicate milestone {name}")
        marks[name] = max(v[0] for v in values)
    bounds = [normal[-1]["complete_ns"], *marks.values()]
    if bounds != sorted(bounds) or marks["target_first_decode_complete"] != post[0]["complete_ns"]:
        raise ValueError("invalid measured window boundaries")
    window = [r for r in steps if normal[-1]["step"] <= r["step"] <= post[0]["step"]]
    if any(b["step"] != a["step"] + 1 for a, b in zip(normal + window[1:], (normal + window[1:])[1:])):
        raise ValueError("missing logical step inside normal or migration segment")
    migration = window[1:-1]
    if any(r["phase"] not in MIGRATION or r["tp"] != 2 for r in migration):
        raise ValueError("unexpected step in migration window")
    gaps = [(b["complete_ns"] - a["complete_ns"]) / 1e9 for a, b in zip(window, window[1:])]
    normal_gaps = [(b["complete_ns"] - a["complete_ns"]) / 1e9 for a, b in zip(normal, normal[1:])]
    normal_tbt = statistics.median(normal_gaps)
    plans = []
    for trace in traces:
        records = [r for r in trace["plan_records"] if r["switch"] == measured]
        if len(records) != 1:
            raise ValueError("need measured plan record on every rank")
        plans.append({"rank": trace["rank"], **records[0]})
    for key in ("plan_variant", "execution_mode", "waves", "candidate_fallback_reason"):
        if any(r[key] != plans[0][key] for r in plans):
            raise ValueError(f"inconsistent {key} across ranks")
    waves = []
    for index in range(plans[0]["waves"]):
        entries = [[r for r in t["waves"] if r["switch"] == measured and r["wave"] == index] for t in traces]
        if any(len(e) != 1 for e in entries):
            raise ValueError(f"incomplete wave {index}")
        rows = [e[0] for e in entries]
        if any(r["commit_done_host_ns"] < r["submit_host_ns"] for r in rows):
            raise ValueError(f"invalid wave {index} bounds")
        wave_gaps = [gap for step, gap in zip(window[1:], gaps) if step["phase"] == "base" and step["wave"] == index]
        waves.append({"wave": index, "decode_gaps_s": wave_gaps, "max_gap_s": max(wave_gaps, default=None),
                      "host_span_s": (max(r["commit_done_host_ns"] for r in rows) - min(r["submit_host_ns"] for r in rows)) / 1e9})
    byte_counts = {}
    for stage in ("base", "delta"):
        rows = [r for p in plans for r in p[f"{stage}_receiver_rows"]]
        if any(r["dst"] != p["rank"] or r["bytes"] < 0 for p in plans for r in p[f"{stage}_receiver_rows"]):
            raise ValueError("invalid receiver byte accounting")
        byte_counts[f"{stage}_bytes"] = sum(r["bytes"] for r in rows)
        byte_counts[f"{stage}_remote_bytes"] = sum(r["bytes"] for r in rows if r["src"] != r["dst"])
    transport = None
    if result is not None:
        record = next(r for r in result["switches"] if r["index"] == measured)
        if record["direction"] != "2->4":
            raise ValueError("measured switch is not expansion")
        transport = sum(r["transport_s"] for r in record["base"]["wave_records"])
    return {"measured_switch": measured, "raw_max_gap_s": max(gaps), "normal_tbt_s": normal_tbt,
            "excess_downtime_s": max(0, max(gaps) - normal_tbt), "window_tbt_s": gaps,
            "normal_tbt_samples_s": normal_gaps, "tbt": distribution(gaps), "interval_slo": slo(gaps),
            "step_latency_ms": {phase: {**distribution([r["event_ms"] for r in rows]),
                                        "samples": [r["event_ms"] for r in rows]}
                                for phase, rows in (("normal", normal), ("migration", migration))},
            "migration_decode_steps": len(migration), "waves": waves, "wave_count": len(waves),
            "plan_variant": plans[0]["plan_variant"], "execution_mode": plans[0]["execution_mode"],
            "scheduler": "FIFO" if plans[0]["execution_mode"] in ("baseline", "fifo") else plans[0]["execution_mode"],
            "adaptive_reasons_by_rank": {p["rank"]: p["adaptive_hybrid"].get("reason") if p["adaptive_hybrid"] else None for p in plans},
            "fallback_reason": plans[0]["candidate_fallback_reason"], "rank_plan_records": plans,
            "transport_s": transport, "switch_wall_s": (marks["target_ready"] - marks["switch_trigger"]) / 1e9,
            "pre_trigger_s": (marks["switch_trigger"] - normal[-1]["complete_ns"]) / 1e9,
            "tp4_first_step_s": (post[0]["complete_ns"] - marks["target_ready"]) / 1e9,
            "completion_offsets_s": [(r["complete_ns"] - window[0]["complete_ns"]) / 1e9 for r in window],
            "window_steps": window, "milestones_ns": marks, "verified_execution_overlap_steps": None,
            "peak_allocated_bytes": max(t["peak_allocated"] for t in traces),
            "peak_reserved_bytes": max(t["peak_reserved"] for t in traces), **byte_counts}


def collect(runs, cases=None):
    launches, errors, duplicates, seen = [], [], [], set()
    available = set()
    for root in map(Path, runs):
        manifest = read_json(root / "manifest.json")
        catalog = {name: (level, method, mode) for name, level, method, mode in manifest["cases"]}
        known = {c[0]: c[1:] for c in diagnostic_cases()}
        catalog.update({name: known[name] for name in manifest.get("reused_cases", {})})
        available.update(catalog)
        for name, (level, method, mode) in catalog.items():
            if cases is not None and name not in cases:
                continue
            directory = root / name
            try:
                if (directory / "exit_code.txt").read_text().strip() != "0":
                    raise ValueError("launch did not exit successfully")
                paths = sorted((directory / "run").glob("trace_rank*.json"))
                digest = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
                config = read_json(directory / "config.json")
                result_path = directory / "run/result.json"
                metrics = summarize_trace([read_json(p) for p in paths], method, mode,
                                          read_json(result_path) if result_path.exists() else None)
                if digest in seen:
                    duplicates.append(str(directory.resolve()))
                    continue
                # Keep different machines, profiles, pressure settings and cache/batch sizes separate.
                cohort = {"host": read_json(paths[0])["clock"]["host"],
                          "profile_sha256": manifest["profile_sha256"], "checkpoint": manifest["checkpoint"],
                          "config": {k: v for k, v in config.items() if k not in
                                     ("PROFILE", "OUT_DIR", "ENTRYPOINT_OVERRIDE", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE")},
                          "pressure": {k: manifest[k] for k in ("stress_links", "l3_processes", "stress_bytes", "stress_sync_every")} if level != "L0" else None}
                launches.append({"launch_id": digest, "case": name, "source": str(directory.resolve()),
                                 "method": method, "level": level, "cold_mode": mode,
                                 "cohort": hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest()[:16],
                                 "cohort_inputs": cohort, "config": config, **metrics})
                seen.add(digest)
            except (OSError, ValueError, KeyError, StopIteration, TypeError) as error:
                errors.append({"source": str(directory.resolve()), "error": str(error)})
    if cases is not None and set(cases) - available:
        errors.append({"error": "selected cases not recorded", "cases": sorted(set(cases) - available)})
    if not launches and not errors:
        errors.append({"error": "no selected launches found"})
    return {"launches": launches, "errors": errors, "duplicate_sources": duplicates}


def metric_points(launch):
    for key in ("raw_max_gap_s", "normal_tbt_s", "excess_downtime_s", "switch_wall_s", "transport_s",
                "migration_decode_steps", "wave_count", "base_bytes", "delta_bytes", "base_remote_bytes", "delta_remote_bytes"):
        yield key, "", launch[key]
    for stat in ("mean", "p50", "p99"):
        yield f"tbt_{stat}_s", "", launch["tbt"][stat]
    for threshold, value in launch["interval_slo"].items():
        yield f"interval_slo_{threshold}s", "", value
    for phase, values in launch["step_latency_ms"].items():
        for stat in ("mean", "p50", "p99"):
            yield f"step_latency_{stat}_ms", phase, values[stat]


def aggregate(launches):
    groups = {}
    for launch in launches:
        for metric, phase, value in metric_points(launch):
            key = (*[launch[k] for k in GROUP], metric, phase)
            group = groups.setdefault(key, [])
            group.append({"launch_id": launch["launch_id"], "value": value})
    rows = []
    for key, points in sorted(groups.items()):
        values = [p["value"] for p in points if p["value"] is not None]
        rows.append({**dict(zip((*GROUP, "metric", "phase"), key)), "n": len(values),
                     "launch_count": len(points), "mean": statistics.mean(values) if values else None,
                     "std": statistics.stdev(values) if len(values) > 1 else None, "points": points})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--cases", help="comma-separated case names; default: all recorded cases")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = collect(args.runs, args.cases.split(",") if args.cases else None)
    report["semantics"] = {"completion": "max rank perf_counter_ns after existing Event synchronization",
                           "event_latency": "maximum same-step rank Event duration, never cross-GPU subtraction",
                           "window": "last normal completion through first TP4 completion, inclusive",
                           "statistics": "linear quantiles; sample std; n=1 std=null; no cross-launch step pooling",
                           "wave_gap": "gap ending at a decode in this wave; STW has no wave decode gaps",
                           "bytes": "receiver-only; total includes local copies, remote excludes them"}
    rows = aggregate(report["launches"])
    report["aggregates"] = rows
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "summary.json", report)
    fields = [*GROUP, "phase", "metric", "n", "launch_count", "mean", "std", "points"]
    write_csv(args.output / "aggregates.csv", rows, fields)
    write_csv(args.output / "downtime.csv", [r for r in rows if r["metric"] in
              ("excess_downtime_s", "raw_max_gap_s", "normal_tbt_s")], fields)
    # The figure compares W2 migration against its own same-launch normal segment.
    # B0 normal samples remain in aggregates.csv and launches.csv.
    write_csv(args.output / "step_latency.csv", [r for r in rows if r["phase"] and r["method"] == "W2"], fields)
    flat = [{**{k: launch[k] for k in (*GROUP, "launch_id", "case", "plan_variant", "execution_mode", "scheduler", "fallback_reason", "adaptive_reasons_by_rank")},
             "phase": phase, "metric": metric, "value": value}
            for launch in report["launches"] for metric, phase, value in metric_points(launch)]
    write_csv(args.output / "launches.csv", flat,
              [*GROUP, "launch_id", "case", "phase", "metric", "value", "plan_variant", "execution_mode", "scheduler", "fallback_reason", "adaptive_reasons_by_rank"])
    print(json.dumps({"launches": len(report["launches"]), "errors": report["errors"], "output": str(args.output)}, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
