#!/usr/bin/env python3
"""Summarize paired baseline/bandwidth-aware live MoE scale experiments."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup-switches", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--minimum-detectable-pct", type=float, default=5.0)
    return parser.parse_args()


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def direction_metrics(data: dict, direction: str, warmup_switches: int) -> dict[str, float]:
    rows = [
        row
        for row in data["switches"][warmup_switches:]
        if row["direction"] == direction
    ]
    if not rows:
        raise ValueError(f"no measured {direction} switches")

    def transport_ms(row: dict) -> float:
        return 1000.0 * sum(
            wave["transport_s"] for wave in row["base"]["wave_records"]
        )

    def exposed_ms(row: dict) -> float:
        return 1000.0 * (
            row["base"]["exposed_wait_s"]
            + row["delta_and_commit_s"]
            + row["pointer_switch_s"]
        )

    def effective_transport_ms(row: dict) -> float:
        return transport_ms(row) + 1000.0 * row.get(
            "online_replan_wait_s", row.get("online_replan_s", 0.0)
        )

    return {
        "transport": statistics.fmean(transport_ms(row) for row in rows),
        "effective_transport": statistics.fmean(
            effective_transport_ms(row) for row in rows
        ),
        "exposed": statistics.fmean(exposed_ms(row) for row in rows),
        "wall": statistics.fmean(
            1000.0
            * (
                row["base"]["wall_s"]
                + row["delta_and_commit_s"]
                + row["pointer_switch_s"]
                + row.get(
                    "online_replan_wait_s", row.get("online_replan_s", 0.0)
                )
            )
            for row in rows
        ),
        "tpot": statistics.fmean(row["base"]["tpot_mean_ms"] for row in rows),
        "tpot_p95": statistics.fmean(
            row["base"].get("tpot_p95_ms", row["base"]["tpot_mean_ms"])
            for row in rows
        ),
        "tpot_p99": statistics.fmean(
            row["base"].get(
                "tpot_p99_ms",
                row["base"].get("tpot_p95_ms", row["base"]["tpot_mean_ms"]),
            )
            for row in rows
        ),
        "tpot_cv": statistics.fmean(
            row["base"].get("tpot_cv", 0.0) for row in rows
        ),
        "bytes": statistics.fmean(
            sum(wave["bytes"] for wave in row["base"]["wave_records"])
            for row in rows
        ),
        "max_diff": max(row["validation_max_diff"] for row in rows),
        "replan": statistics.fmean(
            1000.0 * row.get("online_replan_s", 0.0) for row in rows
        ),
        "replan_wait": statistics.fmean(
            1000.0
            * row.get("online_replan_wait_s", row.get("online_replan_s", 0.0))
            for row in rows
        ),
        "replans": sum(bool(row.get("replanned", False)) for row in rows),
        "foreground_pressure": statistics.fmean(
            row["base"].get("foreground_pressure_mean", 0.0) for row in rows
        ),
    }


def aware_diagnostics(data: dict, warmup_switches: int) -> dict[str, object]:
    rows = [
        row
        for row in data["switches"][warmup_switches:]
        if row["direction"] == "2->4"
    ]
    variants: dict[str, int] = {}
    cache_hits = 0
    reused_send_bytes = 0
    for row in rows:
        variant = row.get("plan_variant", "candidate")
        variants[variant] = variants.get(variant, 0) + 1
        for wave in row["base"]["wave_records"]:
            stats = wave.get("copy_service", {})
            cache_hits += int(stats.get("send_pack_cache_hits", 0))
            reused_send_bytes += int(stats.get("reused_send_bytes", 0))
    all_rank_stats = data.get("copy_service_by_rank", [])
    if all_rank_stats:
        cache_hits = 0
        reused_send_bytes = 0
        for rank_stats in all_rank_stats:
            for switch in rank_stats["switches"]:
                if (
                    switch["direction"] != "2->4"
                    or switch.get("plan_variant") != "candidate"
                ):
                    continue
                for stats in switch["waves"]:
                    cache_hits += int(stats.get("send_pack_cache_hits", 0))
                    reused_send_bytes += int(stats.get("reused_send_bytes", 0))
    return {
        "variants": variants,
        "send_pack_cache_hits": cache_hits,
        "reused_send_bytes": reused_send_bytes,
        "guard": data.get(
            "online_migration_first_guard",
            data.get("online_dual_objective_guard"),
        ),
        "route_stats": data.get("source_route_stats", {}),
        "plan_controller": data.get("online_plan_controller"),
    }


def bootstrap_ci(
    values: list[float], samples: int, rng: random.Random
) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    means = sorted(
        statistics.fmean(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def required_paired_runs(values: list[float], minimum_effect_ms: float) -> int | None:
    if len(values) < 2 or minimum_effect_ms <= 0.0:
        return None
    paired_sd = statistics.stdev(values)
    # Normal approximation for a two-sided alpha=0.05 paired test with 80% power.
    return max(2, math.ceil(((1.96 + 0.8416) * paired_sd / minimum_effect_ms) ** 2))


def main() -> None:
    args = parse_args()
    rng = random.Random(1234)
    print("Statistical unit: one independent AB/BA run-level paired mean")
    print(f"Discarded warmup switches per run: {args.warmup_switches}")

    expert_dirs = sorted(
        args.root.glob("e*"),
        key=lambda path: int(path.name[1:]),
    )
    for expert_dir in expert_dirs:
        experts = int(expert_dir.name[1:])
        pairs = []
        for repetition in sorted(expert_dir.glob("rep_*")):
            baseline_path = repetition / "baseline" / "result.json"
            aware_path = repetition / "residual" / "result.json"
            if not baseline_path.exists() or not aware_path.exists():
                continue
            baseline = load_result(baseline_path)
            aware = load_result(aware_path)
            pairs.append(
                {
                    "baseline_expand": direction_metrics(
                        baseline, "2->4", args.warmup_switches
                    ),
                    "aware_expand": direction_metrics(
                        aware, "2->4", args.warmup_switches
                    ),
                    "baseline_shrink": direction_metrics(
                        baseline, "4->2", args.warmup_switches
                    ),
                    "aware_diagnostics": aware_diagnostics(
                        aware, args.warmup_switches
                    ),
                }
            )

        if not pairs:
            continue

        print(f"\nExperts={experts}, paired runs={len(pairs)}")
        migrated_mib = statistics.fmean(
            pair["baseline_expand"]["bytes"] for pair in pairs
        ) / (1 << 20)
        print(f"Mean migrated bytes/expansion={migrated_mib:.2f} MiB")

        transport_savings = []
        for metric in (
            "transport",
            "effective_transport",
            "exposed",
            "wall",
            "tpot",
            "tpot_p95",
            "tpot_p99",
            "tpot_cv",
        ):
            baseline_mean = statistics.fmean(
                pair["baseline_expand"][metric] for pair in pairs
            )
            aware_mean = statistics.fmean(
                pair["aware_expand"][metric] for pair in pairs
            )
            savings = [
                pair["baseline_expand"][metric] - pair["aware_expand"][metric]
                for pair in pairs
            ]
            if metric == "effective_transport":
                transport_savings = savings
            mean_saving = statistics.fmean(savings)
            confidence_interval = bootstrap_ci(savings, args.bootstrap_samples, rng)
            wins = sum(value > 0.0 for value in savings)
            percent = 100.0 * mean_saving / baseline_mean if baseline_mean else 0.0
            if confidence_interval is None:
                confidence_text = "95%CI=n/a"
            else:
                low, high = confidence_interval
                confidence_text = f"95%CI=[{low:+.4f},{high:+.4f}]"
            unit = "" if metric == "tpot_cv" else " ms"
            print(
                f"  expansion {metric:9s}: baseline={baseline_mean:9.4f} "
                f"aware={aware_mean:9.4f} saving={mean_saving:+8.4f}{unit} "
                f"({percent:+7.2f}%) {confidence_text} "
                f"wins={wins}/{len(pairs)}"
            )

        guard_accepts = sum(
            bool((pair["aware_diagnostics"]["guard"] or {}).get("candidate_accepted"))
            for pair in pairs
        )
        route_changes = [
            int(pair["aware_diagnostics"]["route_stats"].get("accepted", 0))
            for pair in pairs
        ]
        cache_hits = sum(
            int(pair["aware_diagnostics"]["send_pack_cache_hits"])
            for pair in pairs
        )
        reused_mib = sum(
            int(pair["aware_diagnostics"]["reused_send_bytes"])
            for pair in pairs
        ) / (1 << 20)
        variants: dict[str, int] = {}
        for pair in pairs:
            for name, count in pair["aware_diagnostics"]["variants"].items():
                variants[name] = variants.get(name, 0) + int(count)
        print(
            f"  migration-first guard accepted={guard_accepts}/{len(pairs)} "
            f"formal_variants={variants}"
        )
        print(
            f"  source reroutes mean={statistics.fmean(route_changes):.1f} "
            f"candidate_bucket_cache_hits={cache_hits} "
            f"candidate_reused_send={reused_mib:.2f} MiB"
        )
        print(
            "  online loop: "
            f"replans={sum(pair['aware_expand']['replans'] for pair in pairs)} "
            f"useful_replans={sum(int((pair['aware_diagnostics']['plan_controller'] or {}).get('useful_replans', 0)) for pair in pairs)} "
            f"noop_replans={sum(int((pair['aware_diagnostics']['plan_controller'] or {}).get('no_op_replans', 0)) for pair in pairs)} "
            f"mean_planning={statistics.fmean(pair['aware_expand']['replan'] for pair in pairs):.4f} ms "
            f"mean_exposed_wait={statistics.fmean(pair['aware_expand']['replan_wait'] for pair in pairs):.4f} ms "
            f"mean_foreground_pressure="
            f"{100.0 * statistics.fmean(pair['aware_expand']['foreground_pressure'] for pair in pairs):.2f}%"
        )

        effective_transport_saving = statistics.fmean(
            pair["baseline_expand"]["effective_transport"]
            - pair["aware_expand"]["effective_transport"]
            for pair in pairs
        )
        cumulative = " ".join(
            f"N={count}:{count * effective_transport_saving:+.2f}ms"
            for count in (10, 50, 100)
        )
        print(f"  projected cumulative effective-transport saving: {cumulative}")

        baseline_cycle = statistics.fmean(
            pair["baseline_expand"]["effective_transport"]
            + pair["baseline_shrink"]["transport"]
            for pair in pairs
        )
        hybrid_cycle = statistics.fmean(
            pair["aware_expand"]["effective_transport"]
            + pair["baseline_shrink"]["transport"]
            for pair in pairs
        )
        cycle_saving = baseline_cycle - hybrid_cycle
        print(
            f"  hybrid cycle effective transport: baseline={baseline_cycle:.4f} ms "
            f"hybrid={hybrid_cycle:.4f} ms saving={cycle_saving:+.4f} ms "
            f"({100.0 * cycle_saving / baseline_cycle:+.2f}%)"
        )

        baseline_expand = statistics.fmean(
            pair["baseline_expand"]["effective_transport"] for pair in pairs
        )
        minimum_effect = baseline_expand * args.minimum_detectable_pct / 100.0
        required = required_paired_runs(transport_savings, minimum_effect)
        if required is not None:
            print(
                f"  paired runs needed to detect {args.minimum_detectable_pct:g}% "
                f"transport saving at 80% power (pilot estimate): {required}"
            )

        max_diff = max(
            max(
                pair["baseline_expand"]["max_diff"],
                pair["aware_expand"]["max_diff"],
                pair["baseline_shrink"]["max_diff"],
            )
            for pair in pairs
        )
        print(f"  maximum validation difference: {max_diff:.3e}")


if __name__ == "__main__":
    main()
