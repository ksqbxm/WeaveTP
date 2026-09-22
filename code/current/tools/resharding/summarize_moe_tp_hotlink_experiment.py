#!/usr/bin/env python3
"""Summarize the causal chain in the real MoE hot-link experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(summary: dict[str, Any], scheduler: str, metric: str) -> float:
    return float(summary["groups"][scheduler][metric]["mean"])


def _evidence(summary: dict[str, Any], scheduler: str, metric: str) -> float:
    return float(summary["real_link_evidence"][scheduler][metric]["mean"])


def _active_rank_count(summary: dict[str, Any], scheduler: str) -> float:
    token_maps = summary["real_link_evidence"][scheduler]["expert_tokens_by_rank"]
    counts = [sum(int(tokens) > 0 for tokens in token_map.values()) for token_map in token_maps]
    return sum(counts) / max(len(counts), 1)


def _reduction_pct(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline if baseline else 0.0


def _causality_evidence(
    balanced: dict[str, Any], target: dict[str, Any], scheduler: str
) -> dict[str, Any]:
    balanced_active = _active_rank_count(balanced, scheduler)
    target_active = _active_rank_count(target, scheduler)
    balanced_cv = _evidence(balanced, scheduler, "online_bandwidth_cv")
    target_cv = _evidence(target, scheduler, "online_bandwidth_cv")
    balanced_contention_cv = _evidence(balanced, scheduler, "contention_cv")
    target_contention_cv = _evidence(target, scheduler, "contention_cv")
    balanced_degradation = _evidence(
        balanced, scheduler, "max_link_degradation_pct"
    )
    target_degradation = _evidence(target, scheduler, "max_link_degradation_pct")
    cv_ratio = (
        target_contention_cv / balanced_contention_cv
        if balanced_contention_cv > 0.0
        else float("inf")
    )
    return {
        "balanced_active_expert_ranks": balanced_active,
        "target_active_expert_ranks": target_active,
        "balanced_online_bandwidth_cv": balanced_cv,
        "target_online_bandwidth_cv": target_cv,
        "balanced_contention_cv": balanced_contention_cv,
        "target_contention_cv": target_contention_cv,
        "online_bandwidth_cv_increase": target_cv - balanced_cv,
        "online_bandwidth_cv_ratio": cv_ratio,
        "contention_cv_ratio": cv_ratio,
        "balanced_max_link_degradation_pct": balanced_degradation,
        "target_max_link_degradation_pct": target_degradation,
        "max_link_degradation_increase_pct_points": (
            target_degradation - balanced_degradation
        ),
        "hotspot_confirmed": (
            target_active < balanced_active
            and target_contention_cv >= 0.10
            and target_contention_cv >= balanced_contention_cv + 0.05
            and cv_ratio >= 1.5
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", required=True, type=Path)
    parser.add_argument("--fixed-hot", required=True, type=Path)
    parser.add_argument("--controlled-hot", type=Path)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    balanced = _load(args.balanced)
    hot = _load(args.fixed_hot)
    controlled_hot = _load(args.controlled_hot) if args.controlled_hot else None
    evaluation = controlled_hot or hot
    baseline_mode = "baseline"
    natural_causality = _causality_evidence(balanced, hot, baseline_mode)
    controlled_causality = (
        _causality_evidence(balanced, controlled_hot, baseline_mode)
        if controlled_hot is not None
        else None
    )
    activation = evaluation.get("scheduler_activation", {})
    aware_enabled_runs = int(activation.get("aware_enabled_runs", 0))
    aware_total_runs = int(activation.get("aware_total_runs", 0))
    scheduler_effect_attributable = (
        aware_total_runs > 0
        and aware_enabled_runs == aware_total_runs
        and abs(
            _evidence(evaluation, "baseline", "contention_cv")
            - _evidence(evaluation, "bandwidth-aware", "contention_cv")
        ) <= 0.05
    )

    scheduler_metrics = {}
    for metric in (
        "base_migration_s",
        "base_exposed_wait_s",
        "dirty_refresh_s",
        "training_disruption_s",
        "transition_wall_s",
    ):
        baseline_value = _mean(evaluation, "baseline", metric)
        aware_value = _mean(evaluation, "bandwidth-aware", metric)
        scheduler_metrics[metric] = {
            "baseline": baseline_value,
            "bandwidth_aware": aware_value,
            "saved_s": baseline_value - aware_value,
            "reduction_pct": _reduction_pct(baseline_value, aware_value),
        }

    result = {
        "natural_moe_causality": natural_causality,
        "controlled_heterogeneity_causality": controlled_causality,
        "scheduler_evaluation_scenario": (
            "fixed-hot-pressure" if controlled_hot is not None else "fixed-hot"
        ),
        "fixed_hot_scheduler_effect": scheduler_metrics,
        "scheduler_effect_attributable": scheduler_effect_attributable,
        "scheduler_improves_primary_metric": (
            scheduler_effect_attributable
            and scheduler_metrics["training_disruption_s"]["saved_s"] > 0.0
        ),
    }

    print("Real MoE hot-link experiment")
    print(
        "natural MoE skew: "
        f"active_ranks={natural_causality['target_active_expert_ranks']:.2f}; "
        f"online_cv={natural_causality['target_online_bandwidth_cv']:.4f}; "
        f"contention_cv={natural_causality['target_contention_cv']:.4f}; "
        f"contention_ratio={natural_causality['contention_cv_ratio']:.2f}x; "
        f"hotspot_confirmed={natural_causality['hotspot_confirmed']}"
    )
    if controlled_causality is not None:
        print(
            "controlled endpoint pressure: "
            f"online_cv={controlled_causality['target_online_bandwidth_cv']:.4f}; "
            f"contention_cv={controlled_causality['target_contention_cv']:.4f}; "
            f"contention_ratio={controlled_causality['contention_cv_ratio']:.2f}x; "
            f"hotspot_confirmed={controlled_causality['hotspot_confirmed']}"
        )
    print(f"{result['scheduler_evaluation_scenario']} scheduler effect:")
    print(
        f"  attribution_valid={scheduler_effect_attributable} "
        f"scheduler_enabled={aware_enabled_runs}/{aware_total_runs}"
    )
    for metric, values in scheduler_metrics.items():
        print(
            f"  {metric}: baseline={values['baseline']:.6f}s "
            f"aware={values['bandwidth_aware']:.6f}s "
            f"saved={values['saved_s']:.6f}s "
            f"reduction={values['reduction_pct']:.2f}%"
        )
    print(
        "scheduler_improves_primary_metric="
        f"{result['scheduler_improves_primary_metric']}"
    )

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved experiment conclusion to {args.json_output}")


if __name__ == "__main__":
    main()
