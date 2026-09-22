#!/usr/bin/env python3
"""Summarize repeated baseline versus bandwidth-aware MoE+TP runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "base_migration_s",
    "base_wall_s",
    "base_exposed_wait_s",
    "dirty_refresh_s",
    "training_disruption_s",
    "transition_wall_s",
    "standalone_train_step_s",
    "overlap_train_step_s",
    "training_slowdown_pct",
)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_path"] = str(path)
        missing = [metric for metric in METRICS if metric not in row]
        if missing:
            raise ValueError(f"{path} is missing metrics: {', '.join(missing)}")
        rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        mode = str(row["scheduler"])
        if mode == "auto":
            mode = "bandwidth-aware"
        grouped.setdefault(mode, []).append(row)

    required = {"baseline", "bandwidth-aware"}
    if not required.issubset(grouped):
        raise ValueError("Both baseline and bandwidth-aware metrics are required")

    summary: dict[str, Any] = {
        "runs": {mode: len(items) for mode, items in grouped.items()},
        "groups": {},
        "comparison": {},
    }
    for mode, items in grouped.items():
        summary["groups"][mode] = {
            metric: _stats([float(item[metric]) for item in items]) for metric in METRICS
        }

    aware_items = grouped["bandwidth-aware"]
    aware_enabled = sum(
        bool(item.get("scheduler_enabled", item.get("source_routing_enabled", False)))
        for item in aware_items
    )
    summary["scheduler_activation"] = {
        "aware_enabled_runs": aware_enabled,
        "aware_total_runs": len(aware_items),
        "all_aware_runs_enabled": aware_enabled == len(aware_items),
        "any_aware_run_enabled": aware_enabled > 0,
        "predicted_gain_pct": _stats(
            [
                float(item.get("scheduler_gate", {}).get("predicted_gain_pct", 0.0))
                for item in aware_items
            ]
        ),
        "prediction_error_pct": _stats(
            [
                float(item.get("scheduler_gate", {}).get("prediction_error_pct", 0.0))
                for item in aware_items
            ]
        ),
        "selected_wave_counts": [
            int(item.get("scheduler_gate", {}).get("selected_wave_count", 0))
            for item in aware_items
        ],
    }

    for metric in METRICS:
        baseline = summary["groups"]["baseline"][metric]["mean"]
        candidate = summary["groups"]["bandwidth-aware"][metric]["mean"]
        saved = baseline - candidate
        summary["comparison"][metric] = {
            "baseline_mean": baseline,
            "bandwidth_aware_mean": candidate,
            "saved": saved,
            "reduction_pct": 100.0 * saved / baseline if baseline else 0.0,
            "speedup": baseline / candidate if candidate else float("inf"),
        }

    summary["correctness"] = {
        "max_weight_diff": max(float(row["weight_max_diff"]) for row in rows),
        "all_payload_audits_disabled": all(
            not bool(row.get("audit_migration_payloads", True)) for row in rows
        ),
    }
    if all("online_effective_bandwidth_cv" in row for row in rows):
        summary["real_link_evidence"] = {}
        for mode, items in grouped.items():
            summary["real_link_evidence"][mode] = {
                "idle_bandwidth_cv": _stats(
                    [float(item["idle_bandwidth_cv"]) for item in items]
                ),
                "online_bandwidth_cv": _stats(
                    [float(item["online_effective_bandwidth_cv"]) for item in items]
                ),
                "contention_cv": _stats(
                    [float(item.get("online_contention_cv", 0.0)) for item in items]
                ),
                "max_link_degradation_pct": _stats(
                    [
                        float(item["online_link_change"]["max_degradation_pct"])
                        for item in items
                    ]
                ),
                "expert_tokens_by_rank": [
                    item["expert_activity"]["expert_tokens_by_rank"] for item in items
                ],
            }
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("MoE+TP scheduler benchmark")
    print(f"runs={summary['runs']}")
    activation = summary["scheduler_activation"]
    print(
        "scheduler_enabled="
        f"{activation['aware_enabled_runs']}/{activation['aware_total_runs']}"
    )
    print(
        f"predicted_gain_mean={activation['predicted_gain_pct']['mean']:.2f}% "
        f"prediction_error_mean={activation['prediction_error_pct']['mean']:.2f}% "
        f"selected_waves={activation['selected_wave_counts']}"
    )
    print(
        f"{'metric':30s} {'baseline(s)':>13s} {'aware(s)':>13s} "
        f"{'saved(s)':>12s} {'reduction':>11s} {'speedup':>9s}"
    )
    for metric in METRICS:
        row = summary["comparison"][metric]
        unit = "%" if metric == "training_slowdown_pct" else "s"
        print(
            f"{metric:30s} {row['baseline_mean']:13.6f} "
            f"{row['bandwidth_aware_mean']:13.6f} {row['saved']:12.6f} "
            f"{row['reduction_pct']:10.2f}% {row['speedup']:9.3f}x [{unit}]"
        )
    correctness = summary["correctness"]
    print(f"max_weight_diff={correctness['max_weight_diff']:.8g}")
    print(
        "Primary training metric: training_disruption_s = "
        "base_exposed_wait_s + dirty_refresh_s"
    )
    if "real_link_evidence" in summary:
        print("Real-link evidence (mean across repeats):")
        for mode, evidence in summary["real_link_evidence"].items():
            print(
                f"  {mode}: idle_cv={evidence['idle_bandwidth_cv']['mean']:.4f} "
                f"online_cv={evidence['online_bandwidth_cv']['mean']:.4f} "
                f"contention_cv={evidence['contention_cv']['mean']:.4f} "
                f"max_degradation={evidence['max_link_degradation_pct']['mean']:.2f}% "
                f"expert_tokens_by_rank={evidence['expert_tokens_by_rank'][0]}"
            )
        baseline_cv = summary["real_link_evidence"]["baseline"]["contention_cv"]["mean"]
        aware_cv = summary["real_link_evidence"]["bandwidth-aware"]["contention_cv"]["mean"]
        if abs(baseline_cv - aware_cv) > 0.05:
            print(
                "WARNING: baseline and aware observed different contention; "
                "the timing delta is not a clean scheduler effect."
            )
    if not activation["any_aware_run_enabled"]:
        print(
            "ATTRIBUTION: bandwidth-aware execution never activated; timing differences "
            "are run-to-run variance, not scheduler speedup."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    rows = _load(args.metrics)
    summary = _summarize(rows)
    _print_summary(summary)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary JSON to {args.json_output}")


if __name__ == "__main__":
    main()
