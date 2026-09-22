#!/usr/bin/env python3
"""Summarize the four-way live MoE bandwidth/packing ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

VARIANTS = ("original", "pack_only", "route_only", "combined")
T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _mean_ci(values: list[float]) -> tuple[float, float | None]:
    mean = statistics.fmean(values) if values else 0.0
    if len(values) < 2:
        return mean, None
    critical = T95.get(len(values), 2.228 if len(values) <= 20 else 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def _switch_row(row: dict) -> dict[str, float]:
    waves = row["base"]["wave_records"]
    stages = {}
    for wave in waves:
        for name, seconds in wave.get("transport_stage_s", {}).items():
            stages[name] = stages.get(name, 0.0) + float(seconds)
    route = row.get("source_route_stats", {})
    return {
        "transport_ms": 1000.0 * sum(float(wave["transport_s"]) for wave in waves),
        "exposed_ms": 1000.0 * float(row["base"]["exposed_wait_s"]),
        "wall_ms": 1000.0 * float(row["base"]["wall_s"]),
        "tpot_ms": float(row["base"]["tpot_mean_ms"]),
        "pack_ms": 1000.0 * stages.get("pack", 0.0),
        "nccl_ms": 1000.0 * stages.get("nccl", 0.0),
        "unpack_ms": 1000.0 * stages.get("unpack", 0.0),
        "accepted": float(route.get("accepted", route.get("changed_from_default", 0))),
        "rerouted_mib": float(route.get("rerouted_bytes", 0)) / (1 << 20),
        "multi_source": float(route.get("multi_source", 0)),
        "proposed": float(route.get("proposed", 0)),
        "exclude_local_fallbacks": float(route.get("exclude_local_fallbacks", 0)),
    }


def _load(root: Path, warmup_switches: int):
    by_variant = {variant: {} for variant in VARIANTS}
    for rep_dir in sorted(root.glob("rep_[0-9][0-9]")):
        rep = int(rep_dir.name.split("_")[1])
        for variant in VARIANTS:
            result_path = rep_dir / variant / "result.json"
            if not result_path.is_file():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            rows = [
                _switch_row(row)
                for row in payload["switches"][warmup_switches:]
                if row["direction"] == "2->4"
            ]
            if rows:
                by_variant[variant][rep] = rows
    return by_variant


def _run_means(rows_by_rep, metric: str) -> dict[int, float]:
    return {
        rep: statistics.fmean(row[metric] for row in rows)
        for rep, rows in rows_by_rep.items()
    }


def _print_variant(name: str, rows_by_rep) -> None:
    rows = [row for rows in rows_by_rep.values() for row in rows]
    run_transport = list(_run_means(rows_by_rep, "transport_ms").values())
    mean, half = _mean_ci(run_transport)
    ci = "n/a" if half is None else f"[{mean-half:.3f}, {mean+half:.3f}]"
    print(
        f"{name:10s} runs={len(rows_by_rep)} expansions={len(rows)} "
        f"transport_mean={mean:.3f} ms 95%CI={ci} "
        f"p50={_percentile([row['transport_ms'] for row in rows], 0.50):.3f} "
        f"p95={_percentile([row['transport_ms'] for row in rows], 0.95):.3f} "
        f"p99={_percentile([row['transport_ms'] for row in rows], 0.99):.3f} "
        f"max={max((row['transport_ms'] for row in rows), default=0.0):.3f}"
    )
    print(
        " " * 12
        + f"stage_mean pack={statistics.fmean([row['pack_ms'] for row in rows]) if rows else 0.0:.3f} "
        f"nccl={statistics.fmean([row['nccl_ms'] for row in rows]) if rows else 0.0:.3f} "
        f"unpack={statistics.fmean([row['unpack_ms'] for row in rows]) if rows else 0.0:.3f} ms "
        f"unpack_p99={_percentile([row['unpack_ms'] for row in rows], 0.99):.3f} ms"
    )
    print(
        " " * 12
        + f"multi_source={statistics.fmean([row['multi_source'] for row in rows]) if rows else 0.0:.1f} "
        f"proposed={statistics.fmean([row['proposed'] for row in rows]) if rows else 0.0:.1f} "
        f"accepted={statistics.fmean([row['accepted'] for row in rows]) if rows else 0.0:.1f} "
        f"rerouted={statistics.fmean([row['rerouted_mib'] for row in rows]) if rows else 0.0:.2f} MiB "
        f"exclude_local_fallbacks="
        f"{statistics.fmean([row['exclude_local_fallbacks'] for row in rows]) if rows else 0.0:.1f}"
    )


def _print_comparison(label: str, baseline, candidate) -> None:
    baseline_means = _run_means(baseline, "transport_ms")
    candidate_means = _run_means(candidate, "transport_ms")
    reps = sorted(set(baseline_means).intersection(candidate_means))
    savings = [baseline_means[rep] - candidate_means[rep] for rep in reps]
    mean, half = _mean_ci(savings)
    baseline_mean = statistics.fmean(baseline_means[rep] for rep in reps) if reps else 0.0
    pct = 100.0 * mean / baseline_mean if baseline_mean else 0.0
    ci = "n/a" if half is None else f"[{mean-half:+.3f}, {mean+half:+.3f}]"
    wins = sum(value > 0.0 for value in savings)
    print(
        f"{label:29s} saving={mean:+.3f} ms ({pct:+.2f}%) "
        f"95%CI={ci} wins={wins}/{len(savings)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup-switches", type=int, default=4)
    args = parser.parse_args()
    by_variant = _load(args.root, args.warmup_switches)

    print(f"Ablation root: {args.root}")
    print("Statistical unit: one paired run-level expansion mean")
    for variant in VARIANTS:
        _print_variant(variant, by_variant[variant])

    print("\nPaired Transport effects (positive means faster):")
    _print_comparison("packing: original -> pack", by_variant["original"], by_variant["pack_only"])
    _print_comparison("routing: original -> route", by_variant["original"], by_variant["route_only"])
    _print_comparison("routing under pack: pack -> both", by_variant["pack_only"], by_variant["combined"])
    _print_comparison("combined: original -> both", by_variant["original"], by_variant["combined"])


if __name__ == "__main__":
    main()
