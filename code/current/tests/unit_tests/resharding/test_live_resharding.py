from __future__ import annotations

import torch

from megatron.core.resharding.execution import launch_reshard_plan
from megatron.core.resharding.live import (
    MigrationFirstGuard,
    OnlineResidualPlanController,
    ResidualBandwidthTracker,
    ResidualBandwidthWaveScheduler,
    TransferTask,
    filter_plan_by_task_ids,
    restrict_plan_sequence,
    select_adaptive_hybrid_policy,
)
from megatron.core.resharding.utils import ReshardPlan, TransferOp


class _DeferredHandle:
    def __init__(self, complete):
        self._complete = complete
        self._done = False

    def done(self):
        return self._done

    def wait(self):
        if not self._done:
            self._complete()
            self._done = True

    def elapsed_seconds(self):
        return 0.125 if self._done else None


class _DeferredCopyService:
    def __init__(self):
        self.sends = []
        self.recvs = []

    def submit_send(self, tensor, dest_rank, task_id=None):
        self.sends.append((tensor, task_id))

    def submit_recv(self, tensor, src_rank, task_id=None):
        self.recvs.append((tensor, task_id))

    def launch(self):
        def complete():
            sends = {task_id: tensor for tensor, task_id in self.sends}
            for tensor, task_id in self.recvs:
                tensor.copy_(sends[task_id])

        return _DeferredHandle(complete)


def _module(name, tensor):
    module = torch.nn.Module()
    module.register_parameter(name, torch.nn.Parameter(tensor))
    return module


def test_launch_wait_commit_defers_noncontiguous_writeback():
    src = _module("weight", torch.arange(16, dtype=torch.float32).reshape(4, 4))
    dst = _module("weight", torch.zeros(4, 4))
    src_slice = (slice(None), slice(0, 2))
    dst_slice = (slice(None), slice(0, 2))
    plan = ReshardPlan(
        send_ops=[TransferOp("weight", 0, True, src_slice, dst_slice, 7)],
        recv_ops=[TransferOp("weight", 0, False, dst_slice, src_slice, 7)],
    )

    transaction = launch_reshard_plan(
        plan,
        src,
        dst,
        _DeferredCopyService(),
        synchronize_group=False,
        synchronize_device=False,
    )

    assert not transaction.done()
    assert torch.count_nonzero(dst.weight) == 0
    transaction.wait()
    assert transaction.transport_elapsed_s == 0.125
    assert torch.count_nonzero(dst.weight) == 0
    transaction.commit()
    assert torch.equal(dst.weight[:, :2], src.weight[:, :2])
    assert torch.count_nonzero(dst.weight[:, 2:]) == 0


def test_residual_tracker_uses_foreground_and_wave_feedback():
    tracker = ResidualBandwidthTracker(
        {(0, 1): 100.0, (1, 0): 80.0},
        ewma_alpha=1.0,
        floor_fraction=0.1,
    )
    tracker.observe_foreground({(0, 1): 60.0})
    assert tracker.residual_gbps(0, 1) == 40.0

    task = TransferTask(0, 0, 1, 1_000_000_000, "weight")
    tracker.observe_wave([task], elapsed_s=0.4)
    assert tracker.residual_gbps(0, 1) == 20.0


def test_tpot_pressure_updates_only_active_collective_links():
    tracker = ResidualBandwidthTracker(
        {(0, 1): 100.0, (1, 0): 100.0, (0, 2): 80.0},
        ewma_alpha=1.0,
    )

    pressure = tracker.observe_tpot_pressure(
        {(0, 1), (1, 0)},
        baseline_tpot_ms=10.0,
        observed_tpot_ms=20.0,
        max_fraction=0.9,
    )

    assert pressure == 0.5
    assert tracker.residual_gbps(0, 1) == 50.0
    assert tracker.residual_gbps(1, 0) == 50.0
    assert tracker.residual_gbps(0, 2) == 80.0
    assert tracker.foreground_samples == 1


def test_online_plan_controller_replans_on_drift_and_staleness():
    tracker = ResidualBandwidthTracker({(0, 1): 100.0}, ewma_alpha=1.0)
    controller = OnlineResidualPlanController(
        tracker,
        min_change_pct=10.0,
        max_stale_expansions=2,
    )
    task = TransferTask(0, 0, 1, 1_000_000_000, "weight")

    assert not controller.should_replan(1)
    tracker.observe_wave([task], elapsed_s=0.16)
    assert controller.should_replan(1)
    controller.mark_replanned(1)
    assert not controller.should_replan(2)
    assert controller.should_replan(3)
    assert controller.snapshot()["replans"] == 1


def test_online_plan_controller_cools_down_after_noop_replan():
    tracker = ResidualBandwidthTracker({(0, 1): 100.0}, ewma_alpha=1.0)
    controller = OnlineResidualPlanController(
        tracker,
        min_change_pct=1.0,
        max_stale_expansions=2,
        no_op_cooldown_expansions=4,
    )
    task = TransferTask(0, 0, 1, 1_000_000_000, "weight")
    tracker.observe_wave([task], elapsed_s=0.16)
    assert controller.should_replan(1)

    controller.mark_replanned(1, useful=False)
    tracker.observe_wave([task], elapsed_s=1.0)
    assert not controller.should_replan(4)
    assert controller.should_replan(5)
    snapshot = controller.snapshot()
    assert snapshot["no_op_replans"] == 1
    assert snapshot["useful_replans"] == 0


def test_wave_scheduler_covers_every_task_once():
    tracker = ResidualBandwidthTracker(
        {(0, 1): 10.0, (2, 3): 100.0, (0, 3): 50.0},
        ewma_alpha=1.0,
    )
    scheduler = ResidualBandwidthWaveScheduler(
        tracker,
        max_waves=3,
        max_wave_bytes=8_000,
    )
    tasks = [
        TransferTask(0, 0, 1, 8_000, "slow"),
        TransferTask(1, 2, 3, 8_000, "fast"),
        TransferTask(2, 0, 3, 8_000, "middle"),
    ]
    waves = scheduler.schedule(tasks)

    assert 1 < len(waves) <= 3
    assert sorted(task_id for wave in waves for task_id in wave) == [0, 1, 2]


def test_online_wave_selection_enforces_total_cap():
    tracker = ResidualBandwidthTracker({(0, 1): 10.0, (2, 3): 100.0})
    scheduler = ResidualBandwidthWaveScheduler(tracker, max_waves=2)
    remaining = {
        task.task_id: task
        for task in (
            TransferTask(0, 0, 1, 8_000, "a"),
            TransferTask(1, 2, 3, 8_000, "b"),
            TransferTask(2, 0, 1, 4_000, "c"),
        )
    }
    selected = []
    for completed_waves in range(2):
        task_ids = scheduler.next_wave(
            remaining.values(), completed_waves=completed_waves
        )
        selected.append(task_ids)
        for task_id in task_ids:
            remaining.pop(task_id)

    assert not remaining
    assert len(selected) == 2
    assert sorted(task_id for wave in selected for task_id in wave) == [0, 1, 2]


def test_single_wave_preserves_planner_fifo_in_filtered_plan():
    tracker = ResidualBandwidthTracker(
        {(0, 1): 10.0, (2, 3): 100.0},
        ewma_alpha=1.0,
    )
    scheduler = ResidualBandwidthWaveScheduler(tracker, max_waves=1)
    tasks = [
        TransferTask(0, 2, 3, 8_000, "fast"),
        TransferTask(1, 0, 1, 8_000, "slow"),
    ]
    task_ids = scheduler.next_wave(tasks, completed_waves=0)
    full = (slice(None),)
    plan = ReshardPlan(
        send_ops=[
            TransferOp("fast", 3, True, full, full, 0),
            TransferOp("slow", 1, True, full, full, 1),
        ],
        recv_ops=[],
    )

    filtered = filter_plan_by_task_ids(plan, task_ids)

    assert task_ids == [0, 1]
    assert [op.task_id for op in filtered.send_ops] == [0, 1]


def test_adaptive_hybrid_uses_fifo_for_large_profitable_expansion():
    decision = select_adaptive_hybrid_policy(
        direction="2->4",
        requested_scheduler_mode="residual",
        candidate_route_available=True,
        projected_transport_gain_pct=19.5,
        baseline_task_count=22_792,
        candidate_task_count=22_792,
        max_wave_tasks=512,
        min_end_to_end_gain_pct=5.0,
        residual_max_waves=8,
        foreground_pressure_fraction=0.0,
        max_foreground_pressure=0.2,
    )

    assert decision.plan_variant == "candidate"
    assert decision.scheduler_mode == "fifo"
    assert decision.baseline_waves == decision.candidate_waves == 45
    assert decision.predicted_end_to_end_gain_pct == 19.5


def test_adaptive_hybrid_falls_back_for_reverse_direction():
    decision = select_adaptive_hybrid_policy(
        direction="4->2",
        requested_scheduler_mode="residual",
        candidate_route_available=True,
        projected_transport_gain_pct=20.0,
        baseline_task_count=1_024,
        candidate_task_count=1_024,
        max_wave_tasks=512,
        min_end_to_end_gain_pct=5.0,
        residual_max_waves=8,
        foreground_pressure_fraction=0.0,
        max_foreground_pressure=0.2,
    )

    assert decision.plan_variant == "baseline"
    assert decision.scheduler_mode == "fifo"
    assert decision.reason == "direction_without_reroute"


def test_adaptive_hybrid_rejects_extra_wave_penalty():
    decision = select_adaptive_hybrid_policy(
        direction="2->4",
        requested_scheduler_mode="residual",
        candidate_route_available=True,
        projected_transport_gain_pct=20.0,
        baseline_task_count=1_024,
        candidate_task_count=2_048,
        max_wave_tasks=512,
        min_end_to_end_gain_pct=5.0,
        residual_max_waves=8,
        foreground_pressure_fraction=0.0,
        max_foreground_pressure=0.2,
    )

    assert decision.plan_variant == "baseline"
    assert decision.predicted_end_to_end_gain_pct < 0.0
    assert decision.reason == "predicted_end_to_end_gain_below_threshold"


def test_adaptive_hybrid_keeps_residual_feedback_for_small_plan():
    decision = select_adaptive_hybrid_policy(
        direction="2->4",
        requested_scheduler_mode="residual",
        candidate_route_available=True,
        projected_transport_gain_pct=12.0,
        baseline_task_count=2_000,
        candidate_task_count=2_000,
        max_wave_tasks=512,
        min_end_to_end_gain_pct=5.0,
        residual_max_waves=8,
        foreground_pressure_fraction=0.1,
        max_foreground_pressure=0.2,
    )

    assert decision.plan_variant == "candidate"
    assert decision.scheduler_mode == "residual"
    assert decision.reason == "candidate_residual"


def test_restrict_plan_sequence_keeps_weights_and_slices_kv():
    full = (slice(None), slice(None), slice(None))
    plan = ReshardPlan(
        send_ops=[
            TransferOp("weight::w", 1, True, full, full, 0),
            TransferOp("kv::layer.key", 1, True, full, full, 1),
        ],
        recv_ops=[
            TransferOp("weight::w", 0, False, full, full, 0),
            TransferOp("kv::layer.key", 0, False, full, full, 1),
        ],
        rerouted_task_ids=frozenset({0, 1}),
    )

    prefix = restrict_plan_sequence(plan, start=0, end=9, include_non_kv=True)
    delta = restrict_plan_sequence(plan, start=9, end=12, include_non_kv=False)

    assert [op.param_name for op in prefix.send_ops] == ["weight::w", "kv::layer.key"]
    assert prefix.send_ops[1].my_slice[0] == slice(0, 9)
    assert [op.param_name for op in delta.send_ops] == ["kv::layer.key"]
    assert delta.send_ops[0].my_slice[0] == slice(9, 12)
    assert prefix.rerouted_task_ids == frozenset({0, 1})
    assert delta.rerouted_task_ids == frozenset({1})


def test_migration_first_guard_rejects_transport_regression_despite_tpot_gain():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=1,
        min_samples=1,
    )

    assert guard.select_variant() == "baseline"
    guard.observe("baseline", transport_s=0.050, tpot_ms=290.0)
    assert guard.select_variant() == "candidate"
    guard.observe("candidate", transport_s=0.080, tpot_ms=210.0)
    assert not guard.calibrated
    assert guard.select_variant() == "baseline"
    guard.observe("baseline", transport_s=0.051, tpot_ms=292.0)
    assert guard.select_variant() == "candidate"
    guard.observe("candidate", transport_s=0.060, tpot_ms=205.0)

    assert guard.calibrated
    assert not guard.candidate_accepted
    assert guard.select_variant() == "baseline"
    previous_samples = guard.snapshot()["samples"]["baseline"]
    guard.observe("baseline", transport_s=0.040, tpot_ms=180.0)
    assert guard.snapshot()["samples"]["baseline"] == previous_samples + 1
    assert not guard.candidate_accepted


def test_migration_first_guard_accepts_transport_gain_with_bounded_tpot_regression():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
    )
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    guard.observe("candidate", transport_s=0.049, tpot_ms=208.0)

    assert guard.candidate_accepted
    assert guard.select_variant() == "candidate"
    snapshot = guard.snapshot()
    assert snapshot["transport_gain_pct"] > 0.0
    assert snapshot["tpot_gain_pct"] < 0.0


def test_migration_first_guard_rejects_excessive_tpot_regression():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
    )
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    guard.observe("candidate", transport_s=0.040, tpot_ms=220.0)

    assert not guard.candidate_accepted


def test_migration_first_guard_forces_baseline_for_noop_candidate():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
    )
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    guard.observe("candidate", transport_s=0.040, tpot_ms=200.0)
    assert guard.candidate_accepted

    guard.force_baseline("online_replan_no_reroutes")

    assert not guard.candidate_accepted
    assert guard.select_variant() == "baseline"
    assert guard.snapshot()["forced_baseline_reason"] == "online_replan_no_reroutes"


def test_migration_first_guard_stays_on_baseline_when_candidate_is_unavailable():
    guard = MigrationFirstGuard(candidate_warmup_samples=1, min_samples=3)
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    assert guard.select_variant() == "candidate"

    guard.force_baseline("candidate_plan_unavailable")

    assert guard.select_variant() == "baseline"
    assert guard.snapshot()["attempts"] == {"baseline": 1, "candidate": 0}


def test_migration_first_guard_resets_stale_measurements_for_new_phase():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
    )
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    guard.observe("candidate", transport_s=0.040, tpot_ms=200.0)
    assert guard.candidate_accepted

    guard.reset_for_phase("foreground_phase_changed")

    snapshot = guard.snapshot()
    assert not snapshot["calibrated"]
    assert not snapshot["candidate_accepted"]
    assert snapshot["next_variant"] == "baseline"
    assert snapshot["attempts"] == {"baseline": 0, "candidate": 0}
    assert snapshot["phase_resets"] == 1
    assert snapshot["last_reset_reason"] == "foreground_phase_changed"


def test_migration_first_guard_amortizes_exposed_planning_over_repeated_switches():
    repeated = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
    )
    repeated.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    repeated.observe(
        "candidate",
        transport_s=0.048,
        tpot_ms=200.0,
        scheduler_exposed_s=0.010,
        amortization_horizon=10,
    )
    assert repeated.candidate_accepted

    one_shot = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
    )
    one_shot.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    one_shot.observe(
        "candidate",
        transport_s=0.048,
        tpot_ms=200.0,
        scheduler_exposed_s=0.010,
        amortization_horizon=1,
    )
    assert not one_shot.candidate_accepted


def test_migration_first_guard_rolling_probe_reacts_to_phase_change():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=0.0,
        max_tpot_regression_pct=0.0,
        candidate_warmup_samples=0,
        min_samples=1,
        reevaluate_interval=2,
        decision_hysteresis_pct=1.0,
        ewma_alpha=1.0,
        robust_window=1,
    )
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    guard.observe("candidate", transport_s=0.045, tpot_ms=180.0)
    assert guard.candidate_accepted

    guard.observe("candidate", transport_s=0.060, tpot_ms=230.0)
    guard.observe("candidate", transport_s=0.060, tpot_ms=230.0)
    assert guard.select_variant() == "baseline"
    guard.observe("baseline", transport_s=0.040, tpot_ms=170.0)
    assert not guard.candidate_accepted

    guard.observe("baseline", transport_s=0.040, tpot_ms=170.0)
    guard.observe("baseline", transport_s=0.040, tpot_ms=170.0)
    assert guard.select_variant() == "candidate"
    guard.observe("candidate", transport_s=0.030, tpot_ms=140.0)

    assert guard.candidate_accepted
    assert guard.snapshot()["decision_updates"] == 3


def test_migration_first_guard_ignores_one_tpot_spike_but_not_two():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
        reevaluate_interval=1,
        ewma_alpha=1.0,
        robust_window=3,
    )
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    guard.observe("candidate", transport_s=0.045, tpot_ms=190.0)
    assert guard.candidate_accepted

    guard.observe("candidate", transport_s=0.046, tpot_ms=450.0)
    assert guard.select_variant() == "baseline"
    guard.observe("baseline", transport_s=0.051, tpot_ms=201.0)
    assert guard.candidate_accepted

    guard.observe("candidate", transport_s=0.046, tpot_ms=440.0)
    assert guard.select_variant() == "baseline"
    guard.observe("baseline", transport_s=0.051, tpot_ms=201.0)
    assert not guard.candidate_accepted


def test_migration_first_guard_immediately_rejects_transport_spike():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=1.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=1,
        reevaluate_interval=4,
        decision_hysteresis_pct=1.0,
        ewma_alpha=1.0,
        robust_window=3,
    )
    guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
    guard.observe("candidate", transport_s=0.045, tpot_ms=190.0)
    assert guard.candidate_accepted

    guard.observe("candidate", transport_s=0.080, tpot_ms=190.0)

    assert not guard.candidate_accepted
    assert guard.select_variant() == "baseline"


def test_migration_first_guard_uses_adjacent_pairs_and_rejects_one_lucky_sample():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=5.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=1,
        min_samples=3,
        min_candidate_win_rate=0.6666667,
        ewma_alpha=1.0,
        robust_window=3,
    )

    observed = []
    transports = {
        "baseline": iter((0.050, 0.050, 0.050, 0.050)),
        "candidate": iter((0.090, 0.045, 0.060, 0.060)),
    }
    while not guard.calibrated:
        variant = guard.select_variant()
        observed.append(variant)
        guard.observe(
            variant,
            transport_s=next(transports[variant]),
            tpot_ms=200.0,
        )

    assert observed == [
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
        "baseline",
        "candidate",
    ]
    assert not guard.candidate_accepted
    snapshot = guard.snapshot()
    assert abs(snapshot["candidate_win_rate"] - 1.0 / 3.0) < 1.0e-12
    assert snapshot["paired_transport_gain_median_pct"] < 0.0


def test_migration_first_guard_accepts_only_repeated_transport_wins():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=5.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=3,
        min_candidate_win_rate=0.6666667,
        ewma_alpha=0.5,
        robust_window=3,
    )

    for baseline_s, candidate_s in (
        (0.050, 0.045),
        (0.050, 0.047),
        (0.050, 0.049),
    ):
        assert guard.select_variant() == "baseline"
        guard.observe("baseline", transport_s=baseline_s, tpot_ms=200.0)
        assert guard.select_variant() == "candidate"
        guard.observe("candidate", transport_s=candidate_s, tpot_ms=200.0)

    assert guard.calibrated
    assert guard.candidate_accepted
    snapshot = guard.snapshot()
    assert abs(snapshot["candidate_win_rate"] - 2.0 / 3.0) < 1.0e-12
    assert abs(snapshot["paired_transport_gain_median_pct"] - 6.0) < 1.0e-12


def test_migration_first_guard_rejects_majority_with_one_long_tail_stall():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=5.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=3,
        min_candidate_win_rate=0.6666667,
        max_transport_regression_pct=2.0,
        ewma_alpha=0.5,
        robust_window=3,
    )

    for candidate_s in (0.055, 0.040, 0.040):
        guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
        guard.observe("candidate", transport_s=candidate_s, tpot_ms=200.0)

    snapshot = guard.snapshot()
    assert snapshot["candidate_win_rate"] == 2.0 / 3.0
    assert abs(snapshot["paired_transport_gain_median_pct"] - 20.0) < 1.0e-12
    assert abs(snapshot["paired_transport_gain_worst_pct"] + 10.0) < 1.0e-12
    assert not guard.candidate_accepted


def test_migration_first_guard_falls_back_after_accepted_candidate_regresses():
    guard = MigrationFirstGuard(
        min_transport_gain_pct=5.0,
        max_tpot_regression_pct=5.0,
        candidate_warmup_samples=0,
        min_samples=3,
        min_candidate_win_rate=2.0 / 3.0,
        max_transport_regression_pct=2.0,
        reevaluate_interval=4,
        ewma_alpha=0.5,
        robust_window=3,
    )
    for candidate_s in (0.044, 0.045, 0.046):
        guard.observe("baseline", transport_s=0.050, tpot_ms=200.0)
        guard.observe("candidate", transport_s=candidate_s, tpot_ms=200.0)
    assert guard.candidate_accepted

    guard.observe("candidate", transport_s=0.060, tpot_ms=190.0)

    assert not guard.candidate_accepted
    assert guard.select_variant() == "baseline"
    snapshot = guard.snapshot()
    assert snapshot["runtime_fallbacks"] == 1
    assert snapshot["forced_baseline_reason"] == "measured_transport_regression"
