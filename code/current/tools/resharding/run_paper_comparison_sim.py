#!/usr/bin/env python3
"""Controlled strategy-level comparison for dynamic TP migration.

This experiment is intentionally CPU-side.  It uses the repository's actual
bandwidth model and scheduler gate, but replaces NCCL execution with a
deterministic critical-path model so it can run on a workstation without four
or eight GPUs.  The three paper-inspired modes are proxies, not re-runs of the
authors' implementations:

* ``moebius-live`` keeps a fixed source layout but overlaps migration with a
  foreground decode window;
* ``pat-gated`` chooses a route from an idle profile and applies a predicted
  benefit gate before one-shot migration;
* ``moetp`` chooses equivalent sources from the current residual profile and
  applies the repository's wave/critical-path gate.

The output is suitable for checking relative trends and for deciding which
multi-GPU experiments to run next.  It must not be reported as GPU/NCCL
throughput.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

# Allow direct execution via ``python tools/resharding/...py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.resharding.moe_tp import (
    EffectiveBandwidthModel,
    _estimate_rows_critical_seconds,
    _select_scheduler_execution,
)

WORLD_SIZE = 4
SOURCE_RANKS = (0, 1)
DESTINATION_RANKS = (2, 3)
PAYLOAD_CHOICES = (4, 8, 12, 16)  # MiB per equivalent migration slice


def _rows(seed: int, tasks_per_destination: int = 12) -> list[dict[str, int | bool]]:
    rng = random.Random(seed)
    rows: list[dict[str, int | bool]] = []
    task_id = 0
    for dst in DESTINATION_RANKS:
        for _ in range(tasks_per_destination):
            rows.append(
                {
                    "task_id": task_id,
                    # Static TP ownership: each destination keeps a fixed
                    # source replica.  The deliberately crossed mapping makes
                    # link heterogeneity observable; the candidate may use
                    # either equivalent source, which is the key condition for
                    # MOETP.
                    "src_rank": 1 if dst == 2 else 0,
                    "dst_rank": dst,
                    "bytes": rng.choice(PAYLOAD_CHOICES) << 20,
                    "remote": True,
                    "param": f"weight::experts.local_experts.{task_id % 8}.fc1",
                }
            )
            task_id += 1
    return rows


def _matrices(seed: int) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    """Return idle and loaded link profiles in Gbps."""
    rng = random.Random(seed * 7919 + 17)
    # The two profiles intentionally disagree.  PAT sees the idle profile,
    # whereas MOETP receives the loaded/residual profile after foreground
    # traffic and hotspot pressure are present.
    idle = {
        (0, 2): 95.0 + rng.uniform(-8, 8),
        (0, 3): 72.0 + rng.uniform(-8, 8),
        (1, 2): 78.0 + rng.uniform(-8, 8),
        (1, 3): 108.0 + rng.uniform(-8, 8),
    }
    loaded = dict(idle)
    stressed_pair = rng.choice(tuple(idle))
    loaded[stressed_pair] *= rng.uniform(0.28, 0.58)
    # A smaller unrelated drift models changing foreground collectives.
    for pair in loaded:
        if pair != stressed_pair:
            loaded[pair] *= rng.uniform(0.84, 1.02)
    return idle, loaded


def _critical(rows: Iterable[dict], matrix: dict[tuple[int, int], float]) -> float:
    return _estimate_rows_critical_seconds(rows, EffectiveBandwidthModel(matrix, latency_us=5.0))


def _greedy_route(
    baseline_rows: list[dict[str, int | bool]],
    matrix: dict[tuple[int, int], float],
) -> list[dict[str, int | bool]]:
    """Assign each equivalent slice to the source minimizing global load."""
    model = EffectiveBandwidthModel(matrix, latency_us=5.0)
    send_load: dict[int, float] = {}
    recv_load: dict[int, float] = {}
    link_load: dict[tuple[int, int], float] = {}
    routed: list[dict[str, int | bool]] = []
    for row in sorted(baseline_rows, key=lambda item: int(item["bytes"]), reverse=True):
        dst = int(row["dst_rank"])
        best_source = None
        best_score = None
        for src in SOURCE_RANKS:
            candidate = {**row, "src_rank": src}
            cost = model.estimate_seconds(candidate)
            score = max(
                send_load.get(src, 0.0) + cost,
                recv_load.get(dst, 0.0) + cost,
                link_load.get((src, dst), 0.0) + cost,
            )
            if best_score is None or score < best_score:
                best_source, best_score = src, score
        assert best_source is not None
        routed_row = {**row, "src_rank": best_source}
        cost = model.estimate_seconds(routed_row)
        send_load[best_source] = send_load.get(best_source, 0.0) + cost
        recv_load[dst] = recv_load.get(dst, 0.0) + cost
        link_load[(best_source, dst)] = link_load.get((best_source, dst), 0.0) + cost
        routed.append(routed_row)
    return sorted(routed, key=lambda item: int(item["task_id"]))


def _args(*, max_waves: int, min_benefit_pct: float, uncertainty_pct: float = 0.0):
    return SimpleNamespace(
        max_waves=max_waves,
        max_wave_tasks=10_000,
        max_wave_bytes=0,
        scheduler_wave_overhead_us=100.0,
        scheduler_switch_overhead_us=50.0,
        scheduler_prediction_uncertainty_pct=uncertainty_pct,
        scheduler_min_benefit_pct=min_benefit_pct,
    )


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def run(repeats: int, seed: int, tasks_per_destination: int) -> dict[str, object]:
    records: list[dict[str, float | int | bool | str]] = []
    for iteration in range(repeats):
        case_seed = seed + iteration
        baseline = _rows(case_seed, tasks_per_destination)
        idle, loaded = _matrices(case_seed)
        idle_model = EffectiveBandwidthModel(idle, latency_us=5.0)
        loaded_model = EffectiveBandwidthModel(loaded, latency_us=5.0)

        baseline_transport = _estimate_rows_critical_seconds(baseline, loaded_model)

        # Moebius proxy: keep the fixed source ownership but overlap the
        # migration with a finite foreground decode interval.
        decode_window_s = 0.004 + random.Random(case_seed * 13).uniform(0.002, 0.010)
        cutover_s = 0.0005
        moebius_exposed = max(0.0, baseline_transport - decode_window_s) + cutover_s

        # PAT proxy: choose a plan from the idle profile and accept it only if
        # its predicted one-shot benefit clears the cost gate.
        pat_candidate = _greedy_route(baseline, idle)
        pat_gate = _select_scheduler_execution(
            baseline,
            pat_candidate,
            idle_model,
            _args(max_waves=1, min_benefit_pct=5.0),
            world_size=WORLD_SIZE,
        )
        pat_rows = pat_candidate if bool(pat_gate["accepted"]) else baseline
        pat_transport = _estimate_rows_critical_seconds(pat_rows, loaded_model)

        # MOETP proxy: route with the current residual profile and use the
        # repository's global critical-path/wave selection and benefit guard.
        moetp_candidate = _greedy_route(baseline, loaded)
        moetp_gate = _select_scheduler_execution(
            baseline,
            moetp_candidate,
            loaded_model,
            _args(max_waves=4, min_benefit_pct=5.0, uncertainty_pct=1.0),
            world_size=WORLD_SIZE,
        )
        moetp_rows = moetp_candidate if bool(moetp_gate["accepted"]) else baseline
        moetp_transport = _estimate_rows_critical_seconds(moetp_rows, loaded_model)
        if bool(moetp_gate["accepted"]):
            waves = moetp_gate["waves"]
            by_id = {int(row["task_id"]): row for row in moetp_rows}
            moetp_transport = sum(
                _estimate_rows_critical_seconds(
                    [by_id[int(task_id)] for task_id in wave], loaded_model
                )
                for wave in waves
            ) + max(0, len(waves) - 1) * 100.0e-6
            moetp_transport += 50.0e-6

        def live_exposed(transport_s: float) -> float:
            return max(0.0, transport_s - decode_window_s) + cutover_s

        for mode, transport, exposed, accepted in (
            ("fixed-blocking", baseline_transport, baseline_transport, True),
            ("moebius-live", baseline_transport, moebius_exposed, True),
            ("pat-gated", pat_transport, live_exposed(pat_transport), bool(pat_gate["accepted"])),
            ("moetp", moetp_transport, live_exposed(moetp_transport), bool(moetp_gate["accepted"])),
        ):
            records.append(
                {
                    "iteration": iteration,
                    "mode": mode,
                    "baseline_transport_ms": baseline_transport * 1000.0,
                    "transport_ms": transport * 1000.0,
                    "exposed_wait_ms": exposed * 1000.0,
                    "accepted": accepted,
                    "loaded_remote_cv": EffectiveBandwidthModel(loaded).remote_cv(),
                }
            )

    summary: dict[str, object] = {"repeats": repeats, "seed": seed, "records": records, "modes": {}}
    fixed_transport = statistics.fmean(
        float(row["transport_ms"]) for row in records if row["mode"] == "fixed-blocking"
    )
    fixed_exposed = statistics.fmean(
        float(row["exposed_wait_ms"]) for row in records if row["mode"] == "fixed-blocking"
    )
    for mode in ("fixed-blocking", "moebius-live", "pat-gated", "moetp"):
        rows = [row for row in records if row["mode"] == mode]
        transports = [float(row["transport_ms"]) for row in rows]
        exposed = [float(row["exposed_wait_ms"]) for row in rows]
        summary["modes"][mode] = {
            "n": len(rows),
            "accept_rate_pct": 100.0 * statistics.fmean(bool(row["accepted"]) for row in rows),
            "win_rate_pct": 100.0
            * statistics.fmean(
                float(row["transport_ms"]) <= float(row["baseline_transport_ms"]) * 0.999
                for row in rows
            ),
            "regression_rate_pct": 100.0
            * statistics.fmean(
                float(row["transport_ms"]) > float(row["baseline_transport_ms"]) * 1.001
                for row in rows
            ),
            "transport_mean_ms": statistics.fmean(transports),
            "transport_median_ms": statistics.median(transports),
            "transport_p95_ms": _percentile(transports, 0.95),
            "exposed_wait_mean_ms": statistics.fmean(exposed),
            "exposed_wait_p95_ms": _percentile(exposed, 0.95),
            "transport_reduction_vs_fixed_pct": 100.0 * (fixed_transport - statistics.fmean(transports)) / fixed_transport,
            "exposed_wait_reduction_vs_fixed_pct": 100.0 * (fixed_exposed - statistics.fmean(exposed)) / fixed_exposed,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--tasks-per-destination", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = run(args.repeats, args.seed, args.tasks_per_destination)
    print("strategy-level paper comparison (CPU critical-path model)")
    print(f"repeats={summary['repeats']} seed={summary['seed']}")
    print(
        f"{'mode':18s} {'accept%':>8s} {'win%':>7s} {'regress%':>9s} "
        f"{'Transport ms':>15s} {'exposed wait ms':>17s} {'Transport red%':>16s}"
    )
    for mode, values in summary["modes"].items():
        print(
            f"{mode:18s} {values['accept_rate_pct']:8.1f} "
            f"{values['win_rate_pct']:7.1f} {values['regression_rate_pct']:9.1f} "
            f"{values['transport_mean_ms']:15.3f} "
            f"{values['exposed_wait_mean_ms']:17.3f} "
            f"{values['transport_reduction_vs_fixed_pct']:16.2f}"
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
