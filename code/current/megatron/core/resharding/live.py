# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Optional

import torch
import torch.distributed as dist

from .utils import ReshardPlan, TransferOp


@dataclass(frozen=True)
class TransferTask:
    """Serializable receiver-side view of one reshard transfer."""

    task_id: int
    src_rank: int
    dst_rank: int
    num_bytes: int
    param_name: str
    hotness: float = 0.0

    @property
    def link(self) -> tuple[int, int]:
        return (self.src_rank, self.dst_rank)


@dataclass(frozen=True)
class AdaptiveHybridDecision:
    """End-to-end gate and wave policy for a live migration switch."""

    plan_variant: str
    scheduler_mode: str
    reason: str
    projected_transport_gain_pct: float
    predicted_end_to_end_gain_pct: float
    extra_wave_penalty_pct: float
    baseline_waves: int
    candidate_waves: int
    foreground_pressure_fraction: float

    def snapshot(self) -> dict[str, object]:
        return {
            "plan_variant": self.plan_variant,
            "scheduler_mode": self.scheduler_mode,
            "reason": self.reason,
            "projected_transport_gain_pct": self.projected_transport_gain_pct,
            "predicted_end_to_end_gain_pct": self.predicted_end_to_end_gain_pct,
            "extra_wave_penalty_pct": self.extra_wave_penalty_pct,
            "baseline_waves": self.baseline_waves,
            "candidate_waves": self.candidate_waves,
            "foreground_pressure_fraction": self.foreground_pressure_fraction,
        }


def _bounded_wave_count(task_count: int, max_wave_tasks: int) -> int:
    if task_count <= 0:
        return 0
    if max_wave_tasks <= 0:
        return 1
    return math.ceil(task_count / max_wave_tasks)


def select_adaptive_hybrid_policy(
    *,
    direction: str,
    requested_scheduler_mode: str,
    candidate_route_available: bool,
    projected_transport_gain_pct: float,
    baseline_task_count: int,
    candidate_task_count: int,
    max_wave_tasks: int,
    min_end_to_end_gain_pct: float,
    residual_max_waves: int,
    foreground_pressure_fraction: float,
    max_foreground_pressure: float,
) -> AdaptiveHybridDecision:
    """Select a semantic-preserving route and bound online scheduling overhead.

    The planner's route gain is discounted by extra decode/barrier waves. Large
    migrations use bounded FIFO waves because repeatedly rebuilding a residual
    schedule is exposed CPU work; smaller plans retain residual feedback.
    """

    if requested_scheduler_mode not in {"baseline", "residual"}:
        raise ValueError(f"unknown scheduler mode: {requested_scheduler_mode}")
    if baseline_task_count < 0 or candidate_task_count < 0:
        raise ValueError("adaptive task counts must be non-negative")
    if max_wave_tasks < 0:
        raise ValueError("adaptive max wave tasks must be non-negative")
    if min_end_to_end_gain_pct < 0.0:
        raise ValueError("adaptive minimum end-to-end gain must be non-negative")
    if residual_max_waves < 1:
        raise ValueError("adaptive residual wave limit must be positive")
    if not 0.0 <= foreground_pressure_fraction <= 1.0:
        raise ValueError("adaptive foreground pressure must be in [0, 1]")
    if not 0.0 <= max_foreground_pressure <= 1.0:
        raise ValueError("adaptive maximum foreground pressure must be in [0, 1]")

    baseline_waves = _bounded_wave_count(baseline_task_count, max_wave_tasks)
    candidate_waves = _bounded_wave_count(candidate_task_count, max_wave_tasks)
    extra_wave_penalty_pct = 100.0 * max(0, candidate_waves - baseline_waves) / max(
        baseline_waves, 1
    )
    predicted_end_to_end_gain_pct = (
        float(projected_transport_gain_pct) - extra_wave_penalty_pct
    )

    checks = [
        ("baseline_scheduler", requested_scheduler_mode == "baseline"),
        ("direction_without_reroute", direction != "2->4"),
        ("candidate_route_unavailable", not candidate_route_available),
        (
            "foreground_pressure_too_high",
            foreground_pressure_fraction > max_foreground_pressure,
        ),
        (
            "predicted_end_to_end_gain_below_threshold",
            predicted_end_to_end_gain_pct < min_end_to_end_gain_pct,
        ),
    ]
    fallback_reason = next((name for name, failed in checks if failed), None)
    if fallback_reason is not None:
        return AdaptiveHybridDecision(
            plan_variant="baseline",
            scheduler_mode="fifo",
            reason=fallback_reason,
            projected_transport_gain_pct=float(projected_transport_gain_pct),
            predicted_end_to_end_gain_pct=predicted_end_to_end_gain_pct,
            extra_wave_penalty_pct=extra_wave_penalty_pct,
            baseline_waves=baseline_waves,
            candidate_waves=candidate_waves,
            foreground_pressure_fraction=float(foreground_pressure_fraction),
        )

    scheduler_mode = (
        "residual" if candidate_waves <= residual_max_waves else "fifo"
    )
    reason = (
        "candidate_residual"
        if scheduler_mode == "residual"
        else "candidate_fifo_to_bound_scheduler_overhead"
    )
    return AdaptiveHybridDecision(
        plan_variant="candidate",
        scheduler_mode=scheduler_mode,
        reason=reason,
        projected_transport_gain_pct=float(projected_transport_gain_pct),
        predicted_end_to_end_gain_pct=predicted_end_to_end_gain_pct,
        extra_wave_penalty_pct=extra_wave_penalty_pct,
        baseline_waves=baseline_waves,
        candidate_waves=candidate_waves,
        foreground_pressure_fraction=float(foreground_pressure_fraction),
    )


class MigrationFirstGuard:
    """Select a migration-first plan while bounding foreground regression.

    The first expansion calibrates the default plan. Candidate expansions then
    warm persistent communication buffers before contributing measurements.
    Once both sides are measured, the candidate is enabled only when transport
    reaches the requested gain after amortizing exposed planning cost. TPOT is
    a safety budget, not an optimization requirement.
    """

    def __init__(
        self,
        *,
        min_transport_gain_pct: float = 1.0,
        max_tpot_regression_pct: float = 5.0,
        ewma_alpha: float = 0.5,
        robust_window: int = 3,
        candidate_warmup_samples: int = 1,
        min_samples: int = 3,
        min_candidate_win_rate: float = 2.0 / 3.0,
        max_transport_regression_pct: float = 2.0,
        reevaluate_interval: int = 0,
        decision_hysteresis_pct: float = 0.0,
    ) -> None:
        if min_transport_gain_pct < 0.0:
            raise ValueError("min_transport_gain_pct must be non-negative")
        if max_tpot_regression_pct < 0.0:
            raise ValueError("max_tpot_regression_pct must be non-negative")
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if robust_window < 1:
            raise ValueError("robust_window must be positive")
        if candidate_warmup_samples < 0 or min_samples < 1:
            raise ValueError("guard sample counts must be non-negative and min_samples >= 1")
        if not 0.0 <= min_candidate_win_rate <= 1.0:
            raise ValueError("min_candidate_win_rate must be in [0, 1]")
        if max_transport_regression_pct < 0.0:
            raise ValueError("max_transport_regression_pct must be non-negative")
        if reevaluate_interval < 0:
            raise ValueError("reevaluate_interval must be non-negative")
        if decision_hysteresis_pct < 0.0:
            raise ValueError("decision_hysteresis_pct must be non-negative")
        self.min_transport_gain_pct = float(min_transport_gain_pct)
        self.max_tpot_regression_pct = float(max_tpot_regression_pct)
        self.ewma_alpha = float(ewma_alpha)
        self.robust_window = int(robust_window)
        self.candidate_warmup_samples = int(candidate_warmup_samples)
        self.min_samples = int(min_samples)
        self.min_candidate_win_rate = float(min_candidate_win_rate)
        self.max_transport_regression_pct = float(max_transport_regression_pct)
        self.reevaluate_interval = int(reevaluate_interval)
        self.decision_hysteresis_pct = float(decision_hysteresis_pct)
        self._attempts = {"baseline": 0, "candidate": 0}
        self._samples = {"baseline": 0, "candidate": 0}
        self._transport_s: dict[str, float] = {}
        self._amortized_overhead_s: dict[str, float] = {}
        self._tpot_ms: dict[str, float] = {}
        self._transport_history_s = {"baseline": [], "candidate": []}
        self._overhead_history_s = {"baseline": [], "candidate": []}
        self._tpot_history_ms = {"baseline": [], "candidate": []}
        self._paired_transport_gain_pct: list[float] = []
        self._latest_candidate_gain_pct: float | None = None
        self._decision: bool | None = None
        self._incumbent_observations = 0
        self._decision_updates = 0
        self._runtime_fallbacks = 0
        self._forced_baseline_reason: str | None = None
        self._phase_resets = 0
        self._last_reset_reason: str | None = None

    def select_variant(self) -> str:
        # A candidate may disappear after an online replan. Do not keep asking
        # for it merely because calibration has not completed yet.
        if (
            not self.calibrated
            and self._decision is False
            and self._forced_baseline_reason is not None
        ):
            return "baseline"
        if self._samples["baseline"] < 1:
            return "baseline"
        if self._attempts["candidate"] < self.candidate_warmup_samples:
            return "candidate"
        if not self.calibrated:
            # Alternate hot baseline/candidate measurements. A candidate warmup
            # invalidates the earlier baseline, so refresh it before the first
            # measured candidate. Every candidate is then adjacent to a baseline.
            required_baseline_samples = self._samples["candidate"] + 1 + int(
                self.candidate_warmup_samples > 0
            )
            if self._samples["baseline"] < required_baseline_samples:
                return "baseline"
            return "candidate"
        incumbent = "candidate" if self.candidate_accepted else "baseline"
        if (
            self.reevaluate_interval > 0
            and self._incumbent_observations >= self.reevaluate_interval
        ):
            return "baseline" if incumbent == "candidate" else "candidate"
        return incumbent

    def observe(
        self,
        variant: str,
        *,
        transport_s: float,
        tpot_ms: float,
        scheduler_exposed_s: float = 0.0,
        amortization_horizon: int = 1,
    ) -> None:
        if variant not in self._attempts:
            raise ValueError(f"unknown migration variant: {variant}")
        if transport_s < 0.0 or tpot_ms < 0.0 or scheduler_exposed_s < 0.0:
            raise ValueError("migration observations must be non-negative")
        if amortization_horizon < 1:
            raise ValueError("amortization_horizon must be positive")
        incumbent = None
        if self._decision is not None:
            incumbent = "candidate" if self._decision else "baseline"
        self._attempts[variant] += 1
        if (
            variant == "candidate"
            and self._attempts[variant] <= self.candidate_warmup_samples
        ):
            return
        self._samples[variant] += 1
        amortized_overhead_s = float(scheduler_exposed_s) / amortization_horizon
        if variant == "candidate" and self._transport_history_s["baseline"]:
            baseline_transport_s = self._transport_history_s["baseline"][-1]
            candidate_effective_s = float(transport_s) + amortized_overhead_s
            self._latest_candidate_gain_pct = 100.0 * (
                baseline_transport_s - candidate_effective_s
            ) / max(baseline_transport_s, 1.0e-12)
            self._record_history(
                self._paired_transport_gain_pct,
                self._latest_candidate_gain_pct,
            )
        self._record_history(
            self._transport_history_s[variant], float(transport_s)
        )
        self._record_history(
            self._overhead_history_s[variant], amortized_overhead_s
        )
        robust_tpot_ms = self._record_robust_sample(
            self._tpot_history_ms[variant], float(tpot_ms)
        )
        previous_transport = self._transport_s.get(variant, float(transport_s))
        previous_overhead = self._amortized_overhead_s.get(
            variant, amortized_overhead_s
        )
        previous_tpot = self._tpot_ms.get(variant, robust_tpot_ms)
        self._transport_s[variant] = (
            self.ewma_alpha * float(transport_s)
            + (1.0 - self.ewma_alpha) * previous_transport
        )
        self._amortized_overhead_s[variant] = (
            self.ewma_alpha * amortized_overhead_s
            + (1.0 - self.ewma_alpha) * previous_overhead
        )
        self._tpot_ms[variant] = (
            self.ewma_alpha * robust_tpot_ms
            + (1.0 - self.ewma_alpha) * previous_tpot
        )
        if not self.calibrated:
            return
        if self._decision is None:
            self._decision = self._candidate_passes_thresholds()
            self._decision_updates += 1
            return
        if variant == incumbent:
            self._incumbent_observations += 1
            immediate_transport_regression = bool(
                incumbent == "candidate"
                and self._latest_candidate_gain_pct is not None
                and self._latest_candidate_gain_pct
                < -self.max_transport_regression_pct
            )
            if incumbent == "candidate" and (
                immediate_transport_regression
                or not self._candidate_transport_passes(apply_hysteresis=True)
            ):
                self._decision = False
                self._decision_updates += 1
                self._runtime_fallbacks += 1
                self._incumbent_observations = 0
                self._forced_baseline_reason = (
                    "measured_transport_regression"
                    if immediate_transport_regression
                    else "measured_transport_gain_below_threshold"
                )
            return
        updated_decision = self._candidate_passes_thresholds(
            apply_hysteresis=True
        )
        if updated_decision != self._decision:
            self._decision_updates += 1
        self._decision = updated_decision
        self._incumbent_observations = 0
        if self._decision:
            self._forced_baseline_reason = None

    def _record_history(self, history: list[float], value: float) -> None:
        history.append(float(value))
        del history[: -self.robust_window]

    def _record_robust_sample(self, history: list[float], value: float) -> float:
        self._record_history(history, value)
        # TPOT is a secondary safety metric, so one high spike should not
        # overturn a transport win. The primary transport metric remains
        # unsmoothed and can trigger an immediate fallback.
        return float(statistics.median_low(history))

    def force_baseline(self, reason: str) -> None:
        """Immediately disable a candidate that cannot alter the migration path."""

        if not reason:
            raise ValueError("forced baseline reason must be non-empty")
        if self._decision is not False:
            self._decision_updates += 1
        self._decision = False
        self._incumbent_observations = 0
        self._forced_baseline_reason = str(reason)

    def reset_for_phase(self, reason: str) -> None:
        """Discard measurements that no longer describe the foreground phase."""

        if not reason:
            raise ValueError("phase reset reason must be non-empty")
        self._attempts = {"baseline": 0, "candidate": 0}
        self._samples = {"baseline": 0, "candidate": 0}
        self._transport_s.clear()
        self._amortized_overhead_s.clear()
        self._tpot_ms.clear()
        self._transport_history_s = {"baseline": [], "candidate": []}
        self._overhead_history_s = {"baseline": [], "candidate": []}
        self._tpot_history_ms = {"baseline": [], "candidate": []}
        self._paired_transport_gain_pct.clear()
        self._latest_candidate_gain_pct = None
        self._decision = None
        self._incumbent_observations = 0
        self._forced_baseline_reason = None
        self._phase_resets += 1
        self._last_reset_reason = str(reason)

    @property
    def calibrated(self) -> bool:
        required_baseline_samples = self.min_samples + int(
            self.candidate_warmup_samples > 0
        )
        return (
            self._samples["baseline"] >= required_baseline_samples
            and self._samples["candidate"] >= self.min_samples
            and len(self._paired_transport_gain_pct) >= min(
                self.min_samples, self.robust_window
            )
        )

    @property
    def candidate_accepted(self) -> bool:
        if self._decision is not None:
            return self._decision
        return self._candidate_passes_thresholds()

    def _candidate_passes_thresholds(self, *, apply_hysteresis: bool = False) -> bool:
        if not self.calibrated:
            return False
        baseline_tpot = self._tpot_ms["baseline"]
        max_tpot_regression_pct = self.max_tpot_regression_pct
        if apply_hysteresis and self._decision is not None:
            if self._decision:
                max_tpot_regression_pct += self.decision_hysteresis_pct
            else:
                max_tpot_regression_pct -= self.decision_hysteresis_pct
        tpot_limit = baseline_tpot * (1.0 + max_tpot_regression_pct / 100.0)
        return (
            self._candidate_transport_passes(
                apply_hysteresis=apply_hysteresis
            )
            and self._tpot_ms["candidate"] <= tpot_limit
        )

    def _candidate_transport_passes(
        self, *, apply_hysteresis: bool = False
    ) -> bool:
        if not self.calibrated:
            return False
        required_gain_pct = self.min_transport_gain_pct
        if apply_hysteresis and self._decision is not None:
            if self._decision:
                required_gain_pct = max(
                    0.0, required_gain_pct - self.decision_hysteresis_pct
                )
            else:
                required_gain_pct += self.decision_hysteresis_pct
        baseline_transport = self._transport_s["baseline"]
        candidate_effective_transport = self._transport_s["candidate"] + (
            self._amortized_overhead_s.get("candidate", 0.0)
        )
        transport_limit = baseline_transport * (
            1.0 - required_gain_pct / 100.0
        )
        aggregate_passes = candidate_effective_transport <= transport_limit
        paired_gains = self._paired_transport_gain_pct[-self.robust_window :]
        paired_median_gain_pct = float(statistics.median(paired_gains))
        paired_win_rate = sum(
            gain_pct >= required_gain_pct for gain_pct in paired_gains
        ) / len(paired_gains)
        worst_paired_gain_pct = min(paired_gains)
        return (
            aggregate_passes
            and paired_median_gain_pct >= required_gain_pct
            # CLI and shell values commonly spell 2/3 as 0.6666667. Treat
            # that representation as the same threshold as the exact ratio.
            and paired_win_rate + 1.0e-6 >= self.min_candidate_win_rate
            # A majority can hide one severe long-tail stall. Candidates are
            # admitted only when every calibration pair stays in the same
            # regression budget used by the runtime fallback.
            and worst_paired_gain_pct >= -self.max_transport_regression_pct
        )

    def snapshot(self) -> dict[str, object]:
        transport_gain_pct = None
        effective_transport_gain_pct = None
        tpot_gain_pct = None
        paired_transport_gain_median_pct = None
        paired_transport_gain_worst_pct = None
        candidate_win_rate = None
        if self.calibrated:
            baseline_transport = self._transport_s["baseline"]
            baseline_tpot = self._tpot_ms["baseline"]
            transport_gain_pct = 100.0 * (
                baseline_transport - self._transport_s["candidate"]
            ) / max(baseline_transport, 1.0e-12)
            candidate_effective_transport = self._transport_s["candidate"] + (
                self._amortized_overhead_s.get("candidate", 0.0)
            )
            effective_transport_gain_pct = 100.0 * (
                baseline_transport - candidate_effective_transport
            ) / max(baseline_transport, 1.0e-12)
            tpot_gain_pct = 100.0 * (
                baseline_tpot - self._tpot_ms["candidate"]
            ) / max(baseline_tpot, 1.0e-12)
            paired_transport_gain_median_pct = float(
                statistics.median(self._paired_transport_gain_pct)
            )
            paired_transport_gain_worst_pct = min(
                self._paired_transport_gain_pct
            )
            candidate_win_rate = sum(
                gain_pct >= self.min_transport_gain_pct
                for gain_pct in self._paired_transport_gain_pct
            ) / len(self._paired_transport_gain_pct)
        return {
            "calibrated": self.calibrated,
            "candidate_accepted": self.candidate_accepted,
            "next_variant": self.select_variant(),
            "min_transport_gain_pct": self.min_transport_gain_pct,
            "max_tpot_regression_pct": self.max_tpot_regression_pct,
            "robust_window": self.robust_window,
            "candidate_warmup_samples": self.candidate_warmup_samples,
            "min_samples": self.min_samples,
            "min_candidate_win_rate": self.min_candidate_win_rate,
            "max_transport_regression_pct": self.max_transport_regression_pct,
            "reevaluate_interval": self.reevaluate_interval,
            "decision_hysteresis_pct": self.decision_hysteresis_pct,
            "incumbent_observations": self._incumbent_observations,
            "decision_updates": self._decision_updates,
            "runtime_fallbacks": self._runtime_fallbacks,
            "phase_resets": self._phase_resets,
            "last_reset_reason": self._last_reset_reason,
            "attempts": dict(self._attempts),
            "samples": dict(self._samples),
            "transport_s": dict(self._transport_s),
            "amortized_scheduler_overhead_s": dict(self._amortized_overhead_s),
            "tpot_ms": dict(self._tpot_ms),
            "transport_history_s": {
                name: list(values)
                for name, values in self._transport_history_s.items()
            },
            "amortized_overhead_history_s": {
                name: list(values)
                for name, values in self._overhead_history_s.items()
            },
            "tpot_history_ms": {
                name: list(values)
                for name, values in self._tpot_history_ms.items()
            },
            "paired_transport_gain_pct": list(self._paired_transport_gain_pct),
            "paired_transport_gain_median_pct": paired_transport_gain_median_pct,
            "paired_transport_gain_worst_pct": paired_transport_gain_worst_pct,
            "candidate_win_rate": candidate_win_rate,
            "latest_candidate_gain_pct": self._latest_candidate_gain_pct,
            "transport_gain_pct": transport_gain_pct,
            "effective_transport_gain_pct": effective_transport_gain_pct,
            "tpot_gain_pct": tpot_gain_pct,
            "forced_baseline_reason": self._forced_baseline_reason,
        }


# Backward-compatible import for existing benchmark commands and result readers.
DualObjectiveMigrationGuard = MigrationFirstGuard


class ResidualBandwidthTracker:
    """EWMA estimate of bandwidth left to migration while inference is active."""

    def __init__(
        self,
        capacity_gbps: Mapping[tuple[int, int], float],
        *,
        ewma_alpha: float = 0.5,
        floor_fraction: float = 0.05,
    ) -> None:
        if not 0.0 < ewma_alpha <= 1.0:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if not 0.0 < floor_fraction <= 1.0:
            raise ValueError("floor_fraction must be in (0, 1]")
        self.capacity_gbps = {
            (int(src), int(dst)): float(value)
            for (src, dst), value in capacity_gbps.items()
        }
        if not self.capacity_gbps or any(value <= 0.0 for value in self.capacity_gbps.values()):
            raise ValueError("capacity_gbps must contain positive link rates")
        self.ewma_alpha = float(ewma_alpha)
        self.floor_fraction = float(floor_fraction)
        self.foreground_gbps: dict[tuple[int, int], float] = {}
        self.service_gbps: dict[tuple[int, int], float] = {}
        self.samples = 0
        self.foreground_samples = 0
        self.wave_samples = 0

    def _update(
        self,
        target: dict[tuple[int, int], float],
        samples: Mapping[tuple[int, int], float],
    ) -> None:
        for pair, value in samples.items():
            normalized_pair = (int(pair[0]), int(pair[1]))
            observed = max(float(value), 0.0)
            previous = target.get(normalized_pair, observed)
            target[normalized_pair] = (
                self.ewma_alpha * observed + (1.0 - self.ewma_alpha) * previous
            )

    def observe_foreground(
        self,
        samples_gbps: Mapping[tuple[int, int], float],
        *,
        decay_missing: bool = False,
    ) -> None:
        """Update measured foreground TP/A2A occupancy for selected links."""
        samples = dict(samples_gbps)
        if decay_missing:
            samples = {
                pair: float(samples.get(pair, 0.0))
                for pair in self.capacity_gbps
                if pair[0] != pair[1]
            }
        self._update(self.foreground_gbps, samples)
        self.samples += 1
        self.foreground_samples += 1

    def observe_tpot_pressure(
        self,
        active_links: Iterable[tuple[int, int]],
        *,
        baseline_tpot_ms: float,
        observed_tpot_ms: float,
        max_fraction: float = 0.95,
    ) -> float:
        """Map online decode slowdown to shared-link foreground pressure.

        The estimate is deliberately conservative: it only assigns pressure to
        links used by the active TP/EP groups, while actual migration service
        feedback remains the upper bound used by :meth:`residual_gbps`.
        """
        if baseline_tpot_ms <= 0.0 or observed_tpot_ms <= 0.0:
            return 0.0
        if not 0.0 <= max_fraction < 1.0:
            raise ValueError("max_fraction must be in [0, 1)")
        pressure_fraction = min(
            max(1.0 - baseline_tpot_ms / observed_tpot_ms, 0.0),
            max_fraction,
        )
        active = {
            (int(src), int(dst))
            for src, dst in active_links
            if int(src) != int(dst)
        }
        samples = {
            pair: (
                self.capacity_gbps[pair] * pressure_fraction
                if pair in active
                else 0.0
            )
            for pair in self.capacity_gbps
            if pair[0] != pair[1]
        }
        self.observe_foreground(samples, decay_missing=True)
        return pressure_fraction

    def observe_wave(
        self, tasks: Iterable[TransferTask], elapsed_s: float
    ) -> dict[tuple[int, int], float]:
        """Update effective migration service rates observed under foreground load."""
        if elapsed_s <= 0.0:
            return {}
        bytes_by_link: dict[tuple[int, int], int] = {}
        for task in tasks:
            bytes_by_link[task.link] = bytes_by_link.get(task.link, 0) + task.num_bytes
        observed = {
            pair: num_bytes * 8.0 / elapsed_s / 1.0e9
            for pair, num_bytes in bytes_by_link.items()
            if pair[0] != pair[1]
        }
        self._update(self.service_gbps, observed)
        self.samples += 1
        self.wave_samples += 1
        return observed

    def residual_gbps(self, src_rank: int, dst_rank: int) -> float:
        pair = (int(src_rank), int(dst_rank))
        capacity = self.capacity_gbps.get(pair)
        if capacity is None:
            capacity = max(self.capacity_gbps.values()) if src_rank == dst_rank else 1.0
        floor = capacity * self.floor_fraction
        residual = max(capacity - self.foreground_gbps.get(pair, 0.0), floor)
        if pair in self.service_gbps:
            residual = min(residual, max(self.service_gbps[pair], floor))
        return max(residual, 1.0e-9)

    def remote_cv(self) -> float:
        values = [
            self.residual_gbps(src, dst)
            for src, dst in self.capacity_gbps
            if src != dst
        ]
        if len(values) < 2:
            return 0.0
        mean = statistics.fmean(values)
        return statistics.pstdev(values) / max(mean, 1.0e-9)

    def residual_matrix(self) -> dict[tuple[int, int], float]:
        return {
            pair: self.residual_gbps(*pair) for pair in self.capacity_gbps
        }

    @staticmethod
    def matrix_change_pct(
        previous: Mapping[tuple[int, int], float],
        current: Mapping[tuple[int, int], float],
    ) -> float:
        pairs = set(previous).intersection(current)
        if not pairs:
            return float("inf")
        return max(
            100.0
            * abs(float(current[pair]) - float(previous[pair]))
            / max(abs(float(previous[pair])), 1.0e-9)
            for pair in pairs
        )

    def snapshot(self) -> dict[str, object]:
        def rows(values: Mapping[tuple[int, int], float]) -> list[dict[str, float | int]]:
            return [
                {"src": src, "dst": dst, "gbps": value}
                for (src, dst), value in sorted(values.items())
            ]

        return {
            "samples": self.samples,
            "foreground_samples": self.foreground_samples,
            "wave_samples": self.wave_samples,
            "capacity": rows(self.capacity_gbps),
            "foreground": rows(self.foreground_gbps),
            "service": rows(self.service_gbps),
            "residual": rows(
                {
                    pair: self.residual_gbps(*pair)
                    for pair in self.capacity_gbps
                }
            ),
        }


class OnlineResidualPlanController:
    """Trigger source replanning when measured residual capacity changes."""

    def __init__(
        self,
        tracker: ResidualBandwidthTracker,
        *,
        min_change_pct: float = 5.0,
        max_stale_expansions: int = 4,
        min_wave_samples: int = 1,
        no_op_cooldown_expansions: int = 8,
    ) -> None:
        if min_change_pct < 0.0:
            raise ValueError("min_change_pct must be non-negative")
        if (
            max_stale_expansions < 0
            or min_wave_samples < 1
            or no_op_cooldown_expansions < 0
        ):
            raise ValueError(
                "replan intervals must be non-negative and min_wave_samples positive"
            )
        self.tracker = tracker
        self.min_change_pct = float(min_change_pct)
        self.max_stale_expansions = int(max_stale_expansions)
        self.min_wave_samples = int(min_wave_samples)
        self.no_op_cooldown_expansions = int(no_op_cooldown_expansions)
        self._last_matrix = dict(tracker.capacity_gbps)
        self._last_replan_expansion = 0
        self._last_change_pct = 0.0
        self._replans = 0
        self._useful_replans = 0
        self._no_op_replans = 0
        self._not_before_expansion = 0

    def should_replan(self, expansion_index: int) -> bool:
        if expansion_index < 0:
            raise ValueError("expansion_index must be non-negative")
        if expansion_index < self._not_before_expansion:
            self._last_change_pct = 0.0
            return False
        if self.tracker.wave_samples < self.min_wave_samples:
            self._last_change_pct = 0.0
            return False
        current = self.tracker.residual_matrix()
        self._last_change_pct = self.tracker.matrix_change_pct(
            self._last_matrix, current
        )
        stale = (
            self.max_stale_expansions > 0
            and expansion_index - self._last_replan_expansion
            >= self.max_stale_expansions
        )
        return self._last_change_pct >= self.min_change_pct or stale

    def mark_replanned(self, expansion_index: int, *, useful: bool = True) -> None:
        self._last_matrix = self.tracker.residual_matrix()
        self._last_replan_expansion = int(expansion_index)
        self._replans += 1
        if useful:
            self._useful_replans += 1
            self._not_before_expansion = int(expansion_index) + 1
        else:
            self._no_op_replans += 1
            self._not_before_expansion = (
                int(expansion_index) + self.no_op_cooldown_expansions
            )

    def snapshot(self) -> dict[str, object]:
        return {
            "replans": self._replans,
            "min_change_pct": self.min_change_pct,
            "max_stale_expansions": self.max_stale_expansions,
            "min_wave_samples": self.min_wave_samples,
            "no_op_cooldown_expansions": self.no_op_cooldown_expansions,
            "last_change_pct": self._last_change_pct,
            "last_replan_expansion": self._last_replan_expansion,
            "useful_replans": self._useful_replans,
            "no_op_replans": self._no_op_replans,
            "not_before_expansion": self._not_before_expansion,
        }


class ResidualBandwidthWaveScheduler:
    """Pack transfers into a bounded number of online residual-aware waves."""

    def __init__(
        self,
        tracker: ResidualBandwidthTracker,
        *,
        max_waves: int = 4,
        max_wave_bytes: int = 0,
        max_wave_tasks: int = 0,
        latency_us: float = 5.0,
        hotness_weight: float = 0.0,
    ) -> None:
        if max_waves < 1:
            raise ValueError("max_waves must be positive")
        self.tracker = tracker
        self.max_waves = int(max_waves)
        self.max_wave_bytes = max(int(max_wave_bytes), 0)
        self.max_wave_tasks = max(int(max_wave_tasks), 0)
        self.latency_s = max(float(latency_us), 0.0) * 1.0e-6
        self.hotness_weight = max(float(hotness_weight), 0.0)

    def task_seconds(self, task: TransferTask) -> float:
        gbps = self.tracker.residual_gbps(task.src_rank, task.dst_rank)
        return task.num_bytes * 8.0 / (gbps * 1.0e9) + self.latency_s

    def _wave_count(self, tasks: list[TransferTask]) -> int:
        if not tasks:
            return 0
        by_bytes = 1
        if self.max_wave_bytes:
            by_bytes = math.ceil(sum(task.num_bytes for task in tasks) / self.max_wave_bytes)
        by_tasks = 1
        if self.max_wave_tasks:
            by_tasks = math.ceil(len(tasks) / self.max_wave_tasks)
        # Heterogeneous residual rates benefit from at least two feedback points.
        by_variation = 2 if self.tracker.remote_cv() >= 0.10 and len(tasks) > 1 else 1
        return min(self.max_waves, max(1, by_bytes, by_tasks, by_variation))

    def schedule(self, tasks: Iterable[TransferTask]) -> list[list[int]]:
        task_list = list(tasks)
        if not task_list:
            return []
        if any(task.task_id < 0 for task in task_list):
            return [[task.task_id for task in task_list]]

        wave_count = self._wave_count(task_list)
        waves = [
            {
                "task_ids": [],
                "bytes": 0,
                "send": {},
                "recv": {},
                "link": {},
                "critical_s": 0.0,
            }
            for _ in range(wave_count)
        ]

        def priority(task: TransferTask) -> tuple[float, int, int]:
            hotness_factor = 1.0 + self.hotness_weight * max(task.hotness, 0.0)
            return (self.task_seconds(task) * hotness_factor, task.num_bytes, -task.task_id)

        for task in sorted(task_list, key=priority, reverse=True):
            cost = self.task_seconds(task)

            def score(index: int) -> tuple[float, int, int, int]:
                wave = waves[index]
                send = wave["send"].get(task.src_rank, 0.0) + cost
                recv = wave["recv"].get(task.dst_rank, 0.0) + cost
                link = wave["link"].get(task.link, 0.0) + cost
                critical = max(wave["critical_s"], send, recv, link)
                byte_overflow = (
                    max(0, wave["bytes"] + task.num_bytes - self.max_wave_bytes)
                    if self.max_wave_bytes
                    else 0
                )
                task_overflow = (
                    max(0, len(wave["task_ids"]) + 1 - self.max_wave_tasks)
                    if self.max_wave_tasks
                    else 0
                )
                return (
                    float(byte_overflow > 0 or task_overflow > 0),
                    critical,
                    wave["bytes"],
                    index,
                )

            selected = min(range(wave_count), key=score)
            wave = waves[selected]
            wave["task_ids"].append(task.task_id)
            wave["bytes"] += task.num_bytes
            wave["send"][task.src_rank] = wave["send"].get(task.src_rank, 0.0) + cost
            wave["recv"][task.dst_rank] = wave["recv"].get(task.dst_rank, 0.0) + cost
            wave["link"][task.link] = wave["link"].get(task.link, 0.0) + cost
            wave["critical_s"] = max(
                wave["critical_s"],
                wave["send"][task.src_rank],
                wave["recv"][task.dst_rank],
                wave["link"][task.link],
            )

        waves.sort(key=lambda wave: (wave["critical_s"], wave["bytes"]), reverse=True)
        return [list(wave["task_ids"]) for wave in waves if wave["task_ids"]]

    def next_wave(
        self,
        tasks: Iterable[TransferTask],
        *,
        completed_waves: int,
    ) -> list[int]:
        """Select one online wave while enforcing ``max_waves`` as a total cap."""
        if completed_waves < 0:
            raise ValueError("completed_waves must be non-negative")
        task_list = list(tasks)
        if not task_list or completed_waves >= self.max_waves:
            return []
        if self.max_waves == 1:
            # With no wave boundary to optimize, residual sorting only mutates
            # the peer FIFO submitted to NCCL. Preserve the planner's order so
            # a path-only policy cannot regress transport through reordering.
            return [task.task_id for task in task_list]
        if completed_waves == self.max_waves - 1:
            scheduled = self.schedule(task_list)
            return [task_id for wave in scheduled for task_id in wave]
        scheduled = self.schedule(task_list)
        return scheduled[0] if scheduled else []


def _slice_numel(shape: tuple[int, ...], index: tuple[slice, ...]) -> int:
    count = 1
    for dim, item in zip(shape, index):
        if not isinstance(item, slice):
            continue
        start, stop, step = item.indices(dim)
        count *= max(0, math.ceil((stop - start) / step))
    return count


def collect_transfer_tasks(
    plan: ReshardPlan,
    dst_module: Optional[torch.nn.Module],
    *,
    group=None,
    hotness: Optional[Mapping[str, float]] = None,
) -> list[TransferTask]:
    """Collect the unique receiver-side task table on every participating rank."""
    rank = group.rank() if group is not None else dist.get_rank()
    world_size = group.size() if group is not None else dist.get_world_size()
    dst_params = dict(dst_module.named_parameters()) if dst_module is not None else {}
    local_tasks = []
    for op in plan.recv_ops:
        parameter = dst_params.get(op.param_name)
        if parameter is None:
            continue
        local_tasks.append(
            TransferTask(
                task_id=-1 if op.task_id is None else int(op.task_id),
                src_rank=int(op.peer_rank),
                dst_rank=int(rank),
                num_bytes=_slice_numel(tuple(parameter.shape), op.my_slice)
                * parameter.element_size(),
                param_name=op.param_name,
                hotness=float((hotness or {}).get(op.param_name, 0.0)),
            )
        )
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_tasks, group=group)
    tasks = [task for rank_tasks in gathered for task in rank_tasks]
    tasks.sort(key=lambda task: task.task_id)
    return tasks


def filter_plan_by_task_ids(plan: ReshardPlan, task_ids: Iterable[int]) -> ReshardPlan:
    ordered_task_ids = [int(task_id) for task_id in task_ids]
    selected = set(ordered_task_ids)
    position = {task_id: index for index, task_id in enumerate(ordered_task_ids)}

    def selected_ops(ops: Iterable[TransferOp]) -> list[TransferOp]:
        retained = [op for op in ops if op.task_id in selected]
        retained.sort(key=lambda op: position[int(op.task_id)])
        return retained

    rerouted_task_ids = plan.rerouted_task_ids
    if rerouted_task_ids is not None:
        rerouted_task_ids = frozenset(rerouted_task_ids.intersection(selected))
    return ReshardPlan(
        send_ops=selected_ops(plan.send_ops),
        recv_ops=selected_ops(plan.recv_ops),
        rerouted_task_ids=rerouted_task_ids,
    )


def restrict_plan_sequence(
    plan: ReshardPlan,
    *,
    start: int,
    end: int,
    kv_prefix: str = "kv::",
    include_non_kv: bool = True,
) -> ReshardPlan:
    """Restrict KV-cache transfers to a stable token interval.

    The full KV plan is topology-dependent but sequence-range independent, so
    callers can cache it once and derive prefix/delta plans without collective
    replanning as generation advances.
    """
    if start < 0 or end < start:
        raise ValueError(f"invalid sequence interval [{start}, {end})")

    def update(op: TransferOp) -> Optional[TransferOp]:
        is_kv = op.param_name.startswith(kv_prefix)
        if not is_kv:
            return op if include_non_kv else None
        if end == start:
            return None
        if not op.my_slice or not op.peer_slice:
            raise ValueError(f"KV task {op.task_id} has no sequence dimension")
        my_slice = list(op.my_slice)
        peer_slice = list(op.peer_slice)
        my_slice[0] = slice(start, end)
        peer_slice[0] = slice(start, end)
        return replace(op, my_slice=tuple(my_slice), peer_slice=tuple(peer_slice))

    def updated(ops: Iterable[TransferOp]) -> list[TransferOp]:
        result = []
        for op in ops:
            restricted = update(op)
            if restricted is not None:
                result.append(restricted)
        return result

    send_ops = updated(plan.send_ops)
    recv_ops = updated(plan.recv_ops)
    retained_task_ids = {
        op.task_id for op in send_ops + recv_ops if op.task_id is not None
    }
    rerouted_task_ids = plan.rerouted_task_ids
    if rerouted_task_ids is not None:
        rerouted_task_ids = frozenset(rerouted_task_ids.intersection(retained_task_ids))
    return ReshardPlan(
        send_ops=send_ops,
        recv_ops=recv_ops,
        rerouted_task_ids=rerouted_task_ids,
    )


def remap_task_hotness(
    tasks: Iterable[TransferTask],
    hotness_fn: Callable[[str], float],
) -> list[TransferTask]:
    return [replace(task, hotness=float(hotness_fn(task.param_name))) for task in tasks]
