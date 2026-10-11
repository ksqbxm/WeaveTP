"""CPU regressions for bounded startup synchronization; no CUDA or server access."""

import argparse
import ast
import copy
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import test_dp_sweep as shells
import test_dp_sweep_coordinator as coordinator
import test_dp_sweep_tp as tp_tests
import test_live_defaults as defaults
import test_live_storage_lifecycle as lifecycle
import torch

from megatron.core.resharding import live
from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.refit import BandwidthAwareRefitPolicy
from megatron.core.transformer.module import Float16Module
from tools.resharding.correctness.harness import Mailbox, MailService, centralized_plans, param
from tools.resharding.correctness.reference import Layout, exact_chunks, markers

BASELINE = "397ae279143e1705f05257988b7b8aa0bdac439d"


@pytest.mark.parametrize("src_tp,dst_tp,world,limit,weights", [
    (src, dst, world, limit, 1)
    for src, dst, world in ((1, 2, 2), (2, 4, 8)) for limit in (1, 3, 2048)
] + [(2, 4, 8, 2048, 513)])
def test_real_plans_copy_every_task_once_and_commit_before_global_barrier(src_tp, dst_tp, world, limit, weights):
    full = markers((3, 16), dtype=torch.bfloat16)
    sources = tp_tests._bank(full, src_tp, world)
    targets = tp_tests._bank(full, dst_tp, world, empty=True)
    # Exercise deferred writeback too, not only contiguous direct receives.
    for bank in (sources, targets):
        for model in bank.values():
            model.config = SimpleNamespace(num_moe_experts=None, fp16=False, bf16=True,
                                           virtual_pipeline_model_parallel_size=None)
            original = next(model.named_parameters())[1]
            model.entries = {f"weight::w{index}": param(original.detach().clone()) for index in range(weights)}
            for parameter in model.entries.values():
                parameter.__dict__.update(original.__dict__)
    for model in targets.values():
        for p in model.entries.values():
            storage = torch.full((p.shape[0], p.shape[1] * 2), float("nan"), dtype=p.dtype)
            p.data = storage[:, ::2]
    policy = BandwidthAwareRefitPolicy(bandwidth_gbps={(a, b): 100.0 for a in range(world) for b in range(world)})
    plans = centralized_plans(sources, targets, **policy.planner_kwargs())
    task_ids = sorted(op.task_id for plan in plans.values() for op in plan.recv_ops)
    assert len(set(task_ids)) == len(task_ids)
    batches = [task_ids[start:start + limit] for start in range(0, len(task_ids), limit)]
    if weights == 513:
        assert list(map(len, batches)) == [2048, 2048, 8]
    mailboxes = [Mailbox() for _ in batches]
    ready = [threading.Barrier(world, action=box.flush, timeout=15) for box in mailboxes]
    committed = [set() for _ in batches]
    barriers = threading.Barrier(world, timeout=15)
    gathered = [None] * world
    gather_barrier = threading.Barrier(world, timeout=15)
    launches, completions = [[] for _ in range(world)], [[] for _ in range(world)]

    def gather(output, value, group):
        gathered[group.rank()] = value
        gather_barrier.wait()
        output[:] = gathered

    def worker(rank):
        namespace = defaults.load(defaults.CURRENT, ["_synchronize_initial_weights"])
        group = SimpleNamespace(rank=lambda: rank, size=lambda: world)
        control = object()
        service = SimpleNamespace(producer_stream=None)
        factory = Mock(return_value=service)
        build = Mock(return_value=plans[rank])

        def launch(plan, source, target, actual_service, **kwargs):
            assert actual_service is service and service.producer_stream is stream
            assert kwargs == dict(group=group, synchronize_group=False,
                                  synchronize_device=True, release_cache=False)
            index = len(launches[rank])
            selected = batches[index]
            assert len(selected) <= limit
            assert all(op.task_id in selected for op in plan.send_ops + plan.recv_ops)
            launches[rank].append(([op.task_id for op in plan.send_ops], [op.task_id for op in plan.recv_ops]))
            txn = launch_reshard_plan(plan, source, target, MailService(mailboxes[index], rank), **kwargs)
            original_wait, original_commit = txn.wait, txn.commit

            def wait():
                if not txn.waited:
                    ready[index].wait()  # All sends/receives posted before mailbox transfer.
                return original_wait()

            def commit():
                result = original_commit()
                committed[index].add(rank)
                completions[rank].append(("commit", index))
                return result

            txn.wait, txn.commit = wait, commit
            return txn

        def barrier(*, group):
            assert group is control
            index = len(completions[rank]) // 2
            assert rank in committed[index]
            completions[rank].append(("barrier", index))
            barriers.wait()
            assert committed[index] == set(range(world))

        stream = object()
        namespace.update(build_centralized_reshard_plan=build, collect_transfer_tasks=live.collect_transfer_tasks,
                         filter_plan_by_task_ids=live.filter_plan_by_task_ids, launch_reshard_plan=launch,
                         NCCLCopyService=factory, dist=SimpleNamespace(barrier=barrier),
                         torch=SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: stream)))
        args = SimpleNamespace(live_p2p_order="peer-size-desc")
        namespace["_synchronize_initial_weights"](
            Float16Module(sources[rank].config, sources[rank]),
            Float16Module(targets[rank].config, targets[rank]), args=args, policy=policy,
            reshard_group=group, control_group=control, max_tasks=limit,
        )
        build.assert_called_once_with(sources[rank], targets[rank], num_experts=None,
                                      group=group, **policy.planner_kwargs())
        factory.assert_called_once_with(group=group, p2p_order="peer-size-desc",
                                        ensure_all_ranks_participate=True)

    with patch("torch.distributed.all_gather_object", side_effect=gather), \
            patch("torch.cuda.synchronize"):
        with ThreadPoolExecutor(max_workers=world) as pool:
            list(pool.map(worker, range(world)))
    for rank in range(world):
        sends = [tid for sent, _ in launches[rank] for tid in sent]
        recvs = [tid for _, received in launches[rank] for tid in received]
        assert Counter(sends) == Counter(op.task_id for op in plans[rank].send_ops)
        assert Counter(recvs) == Counter(op.task_id for op in plans[rank].recv_ops)
        assert completions[rank] == [(event, i) for i in range(len(batches)) for event in ("commit", "barrier")]
        expected = Layout(tuple(range(rank // dst_tp * dst_tp, (rank // dst_tp + 1) * dst_tp)), 1).shard(full, rank)
        for parameter in targets[rank].entries.values():
            exact_chunks(parameter, expected)
    if limit == 1:
        assert any(not sent and not recv for per_rank in launches for sent, recv in per_rank)
    assert sum(box.local + box.remote for box in mailboxes) == len(task_ids)


def test_zero_keeps_original_swap_and_default_output(tmp_path):
    baseline = ast.parse(subprocess.check_output(
        ["git", "show", BASELINE + ":" + defaults.LIVE_PATH], cwd=defaults.ROOT))
    for release in (False, True):
        expected = coordinator.capture(baseline, tmp_path, release=release)
        for options in ({}, {"live_init_sync_max_tasks": 0}):
            swap = Mock()
            forbidden = Mock(side_effect=AssertionError("disabled batching executed"))
            actual = coordinator.capture(
                defaults.CURRENT, tmp_path, release=release, live_overrides=options,
                namespace_overrides={"swap_model_weights": swap, "_synchronize_initial_weights": forbidden},
            )
            assert actual == expected
            swap.assert_called_once()
            forbidden.assert_not_called()
    parser = argparse.ArgumentParser()
    defaults.load(defaults.CURRENT, ["add_live_args"])["add_live_args"](parser)
    assert "live_init_sync_max_tasks" not in vars(parser.parse_args([]))
    assert parser.parse_args(["--live-init-sync-max-tasks", "2048"]).live_init_sync_max_tasks == 2048


def test_negative_limit_fails_before_model_construction():
    model = Mock(side_effect=AssertionError("model constructed"))
    with pytest.raises(ValueError, match="init-sync-max-tasks"):
        lifecycle.run_coordinator(defaults.CURRENT, live_overrides={"live_init_sync_max_tasks": -1},
                                  namespace_overrides={"get_training_model": model})
    model.assert_not_called()


def test_enabled_startup_sync_precedes_unchanged_validation_and_standby_release():
    events = []
    original_load = defaults.load

    def load(tree, names):
        namespace = original_load(tree, names)
        if "run_live_benchmark" not in names:
            return namespace
        helper = original_load(tree, ["_synchronize_initial_weights"])["_synchronize_initial_weights"]
        # Run the helper with the coordinator's same CPU model/transport doubles.
        def sync(*args, **kwargs):
            helper.__globals__.update(namespace)
            helper(*args, **kwargs)
            events.append("sync")
        validation = namespace["_validate_cutover"]
        def validate(*args, **kwargs):
            events.append("validate")
            return validation(*args, **kwargs)
        namespace.update(_synchronize_initial_weights=sync, _validate_cutover=validate)
        return namespace

    def initialize(model, index):
        model.config.num_moe_experts = 4
        if index == 1:
            model.w.data.fill_(float("nan"))

    with patch.object(defaults, "load", side_effect=load), patch("torch.cuda.synchronize"):
        result, models, releases = lifecycle.run_coordinator(
            defaults.CURRENT, release=True, model_initializer=initialize,
            live_overrides={"live_init_sync_max_tasks": 1},
            namespace_overrides={"swap_model_weights": Mock(side_effect=AssertionError("unbounded startup"))},
        )
    assert events == ["sync"] + ["validate"] * 5  # Initial calibration plus four switches.
    assert len(releases) == 5
    assert all(row["validation_max_diff"] == 0 for row in result["switches"])
    exact_chunks(models[0].w, torch.arange(12.).reshape(3, 4))


def test_all_other_benchmark_logic_is_baseline_ast():
    baseline = ast.parse(subprocess.check_output(
        ["git", "show", BASELINE + ":" + defaults.LIVE_PATH], cwd=defaults.ROOT))
    for node in baseline.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name not in {"add_live_args", "run_live_benchmark"}:
            assert ast.dump(node) == ast.dump(defaults.function(defaults.CURRENT, node.name))
    current = copy.deepcopy(defaults.function(defaults.CURRENT, "run_live_benchmark"))
    # Only the limit read/check and the initial sync call may differ.
    current.body = [node for node in current.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "init_sync_max_tasks" for t in node.targets)
        or isinstance(node, ast.If) and ast.unparse(node.test) == "init_sync_max_tasks < 0"
    )]
    index = next(i for i, node in enumerate(current.body)
                 if isinstance(node, ast.If) and ast.unparse(node.test) == "init_sync_max_tasks")
    current.body[index:index + 1] = current.body[index].orelse
    assert ast.dump(current) == ast.dump(defaults.function(baseline, "run_live_benchmark"))


@pytest.mark.parametrize("deepseek", [False, True])
def test_launcher_passes_positive_limit_and_omits_zero(tmp_path, deepseek):
    with patch.object(shells, "BASELINE", BASELINE):
        before, _, _ = shells.shell_capture(tmp_path, baseline=True, deepseek=deepseek)
        for extra in ({}, {"INIT_SYNC_MAX_TASKS": "0"}):
            actual, calls, _ = shells.shell_capture(tmp_path, deepseek=deepseek, extra=extra)
            assert (actual.returncode, actual.stdout, actual.stderr) == (before.returncode, before.stdout, before.stderr)
            assert b"--live-init-sync-max-tasks" not in calls[-1]
    run, calls, _ = shells.shell_capture(tmp_path, deepseek=deepseek, extra={"INIT_SYNC_MAX_TASKS": "2048"})
    assert run.returncode == 0
    assert calls[-1][calls[-1].index(b"--live-init-sync-max-tasks") + 1] == b"2048"
    for bad in ("-1", "1.5", "abc", "01"):
        run, calls, _ = shells.shell_capture(tmp_path, deepseek=deepseek, extra={"INIT_SYNC_MAX_TASKS": bad})
        assert run.returncode == 2 and not calls
