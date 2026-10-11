#!/usr/bin/env python3
"""CPU-only collection and launch-first aggregation of measured expansions."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

if __package__:
    from . import artifacts, common, trace_metrics
else:
    import artifacts
    import common
    import trace_metrics


def collect(runs, cases=None):
    launches, errors, duplicates, seen = [], [], [], set()
    available = set()
    for root in map(Path, runs):
        manifest = artifacts.read_json(root / "manifest.json")
        catalog = {name: (level, method, mode) for name, level, method, mode in manifest["cases"]}
        known = {c[0]: c[1:] for c in common.diagnostic_cases()}
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
                config = artifacts.read_json(directory / "config.json")
                result_path = directory / "run/result.json"
                metrics = trace_metrics.summarize_trace([artifacts.read_json(p) for p in paths], method, mode,
                                          artifacts.read_json(result_path) if result_path.exists() else None)
                if digest in seen:
                    duplicates.append(str(directory.resolve()))
                    continue
                # Keep different machines, profiles, pressure settings and cache/batch sizes separate.
                cohort = {"host": artifacts.read_json(paths[0])["clock"]["host"],
                          "profile_sha256": manifest["profile_sha256"], "checkpoint": manifest["checkpoint"],
                          "config": {k: v for k, v in config.items() if k not in
                                     ("PROFILE", "OUT_DIR", "ENTRYPOINT_OVERRIDE", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE")},
                          "pressure": {**{k: manifest[k] for k in ("stress_links", "stress_bytes", "stress_sync_every")},
                                       "processes_per_pair": manifest["processes_per_pair"][level] if "processes_per_pair" in manifest else manifest["l3_processes"]} if level != "L0" else None}
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
            key = (*[launch[k] for k in trace_metrics.GROUP], metric, phase)
            group = groups.setdefault(key, [])
            group.append({"launch_id": launch["launch_id"], "value": value})
    rows = []
    for key, points in sorted(groups.items()):
        values = [p["value"] for p in points if p["value"] is not None]
        rows.append({**dict(zip((*trace_metrics.GROUP, "metric", "phase"), key)), "n": len(values),
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
    artifacts.write_json(args.output / "summary.json", report)
    fields = [*trace_metrics.GROUP, "phase", "metric", "n", "launch_count", "mean", "std", "points"]
    artifacts.write_csv(args.output / "aggregates.csv", rows, fields)
    artifacts.write_csv(args.output / "downtime.csv", [r for r in rows if r["metric"] in
              ("excess_downtime_s", "raw_max_gap_s", "normal_tbt_s")], fields)
    # The figure compares W2 migration against its own same-launch normal segment.
    # B0 normal samples remain in aggregates.csv and launches.csv.
    artifacts.write_csv(args.output / "step_latency.csv", [r for r in rows if r["phase"] and r["method"] == "W2"], fields)
    flat = [{**{k: launch[k] for k in (*trace_metrics.GROUP, "launch_id", "case", "plan_variant", "execution_mode", "scheduler", "fallback_reason", "adaptive_reasons_by_rank")},
             "phase": phase, "metric": metric, "value": value}
            for launch in report["launches"] for metric, phase, value in metric_points(launch)]
    artifacts.write_csv(args.output / "launches.csv", flat,
              [*trace_metrics.GROUP, "launch_id", "case", "phase", "metric", "value", "plan_variant", "execution_mode", "scheduler", "fallback_reason", "adaptive_reasons_by_rank"])
    print(json.dumps({"launches": len(report["launches"]), "errors": report["errors"], "output": str(args.output)}, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
