"""Execute the actual benchmark coordinator with CPU models and transport doubles.

Real planner and transport geometry are covered separately. This tests default
result parity and the actual lifecycle insertion points, including KV offsets.
"""

import argparse
import itertools
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import test_live_defaults as defaults
import torch

from megatron.core.resharding import live
from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.refit import BandwidthAwareRefitPolicy
from megatron.core.resharding.utils import ReshardPlan, TransferOp
from tools.resharding.standby_weights import (
    StandbyWeights,
    release_standby,
    storage_bytes,
    validate_release_mode,
)


def run_coordinator(tree, *, release=False, identity=False, repeat=False, fault=False, check_polls=1, capacity=128):
    names = ["run_live_benchmark", "add_live_args", "StaticKVCacheModule", "LiveStateBundle",
             "_run_async_waves", "_parse_rank_phases", "_pressure_ranks_for_switch",
             "_active_experts_for_switch", "_set_active_experts", "_poison_kv", "_percentile",
             "_tasks_for_ids", "_validate_cutover", "_expert_hotness"]
    ns = defaults.load(tree, names)
    parser = argparse.ArgumentParser()
    ns["add_live_args"](parser)
    args = parser.parse_args([])
    vars(args).update(live_release_standby_weights=release, live_kv_request_identity=identity,
                      tensor_model_parallel_size=2, expert_tensor_parallel_size=2,
                      expert_model_parallel_size=1, num_experts=4, micro_batch_size=1,
                      use_tp_pp_dp_mapping=False, seed=7, max_position_embeddings=capacity,
                      live_active_experts_tuple=(0, 1), live_active_expert_phases_tuple=((0, 1),),
                      live_switches=4, live_repeat_forward=repeat, live_scheduler_mode="baseline",
                      live_max_waves=1, live_max_overlap_steps=2)
    groups = SimpleNamespace(rank=lambda: 0, size=lambda: 1)
    models, releases, written = [], [], []
    pg = SimpleNamespace(tp=groups, dp=groups, ep=groups)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.arange(12.).reshape(3, 4), requires_grad=False)
            self.config, self.pg_collection = SimpleNamespace(), pg
        def cuda(self):
            return self

    def model_builder(*_a, **_kw):
        model = Model()
        models.append(model)
        return [model]

    def context(**_kw):
        return SimpleNamespace(key_value_memory_dict={1: (torch.zeros(128, 1, 2, 2), torch.zeros(128, 1, 2, 2))},
                               sequence_len_offset=0, enable_decode_mode=lambda: None)

    def prefill(_model, ctx, _args):
        ctx.sequence_len_offset = args.live_prompt_tokens
        return torch.zeros(1)

    def decode(model, ctx, _args, *, token_value):
        # Reading freed storage must fail before touching its contents.
        assert model.w.untyped_storage().nbytes() == 48
        assert torch.isfinite(model.w).all()
        pos = ctx.sequence_len_offset
        history = torch.tensor(0.)
        for k, v in ctx.key_value_memory_dict.values():
            assert torch.isfinite(k[:pos]).all() and torch.isfinite(v[:pos]).all()
            history += k[:pos].sum() + v[:pos].sum()
            k[pos].fill_(token_value)
            v[pos].fill_(token_value + 1)
        ctx.sequence_len_offset += 1
        return (model.w.sum() + history).reshape(1), 1.0

    def plan(_source, target, **_kw):
        sends, recvs = [], []
        for tid, (name, parameter) in enumerate(target.named_parameters()):
            if fault and name.startswith("weight::"):
                continue
            index = (slice(None),) * parameter.ndim
            sends.append(TransferOp(name, 0, True, index, index, tid))
            recvs.append(TransferOp(name, 0, False, index, index, tid))
        return ReshardPlan(sends, recvs)

    class Handle:
        def __init__(self, sends, recvs):
            self.sends, self.recvs, self.polls = sends, recvs, 0
        def done(self):
            self.polls += 1
            return self.polls >= 2
        def wait(self):
            with torch.no_grad():
                for tensor, tid in self.recvs:
                    tensor.copy_(dict(self.sends)[tid])
            self.sends.clear()
            self.recvs.clear()
        def elapsed_seconds(self):
            return 0.001

    class Service:
        def __init__(self, **_):
            self.producer_stream = None
            self.sends, self.recvs = [], []
        def submit_send(self, tensor, _peer, task_id=None):
            self.sends.append((task_id, tensor))
        def submit_recv(self, tensor, _peer, task_id=None):
            self.recvs.append((tensor, task_id))
        def launch(self):
            result = Handle(self.sends, self.recvs)
            self.sends, self.recvs = [], []
            return result
        def invalidate_persistent_pack_cache(self):
            pass

    class Event:
        def __init__(self, **_):
            self.polls = 0
        def record(self, *_):
            pass
        def query(self):
            self.polls += 1
            return self.polls > check_polls
        def synchronize(self):
            pass

    def tasks(plan, bundle, **_kw):
        params = dict(bundle.named_parameters())
        return [live.TransferTask(op.task_id, 0, 0, live._slice_numel(tuple(params[op.param_name].shape), op.my_slice)
                                  * params[op.param_name].element_size(), op.param_name) for op in plan.recv_ops]

    def release_hook(weights, services):
        record = release_standby(weights, services)
        releases.append(weights)
        return record

    def request_domains(*_):
        args.live_request_domain_id = 1
        return {"domain_count": 2}

    for name in ("ResidualBandwidthTracker", "ResidualBandwidthWaveScheduler", "restrict_plan_sequence", "filter_plan_by_task_ids"):
        ns[name] = getattr(live, name)
    clock = itertools.count()
    ns.update(get_args=lambda: args, dist=SimpleNamespace(
        get_rank=lambda: 0, get_world_size=lambda: 4, new_group=lambda **_: groups,
        barrier=lambda **_: None, broadcast_object_list=lambda *_a, **_kw: None,
        all_gather_object=lambda out, value, **_: out.__setitem__(slice(None), [value] * len(out))),
        time=SimpleNamespace(perf_counter=lambda: next(clock) / 1000), statistics=__import__("statistics"),
        math=__import__("math"), re=__import__("re"),
        nullcontext=nullcontext, print_rank_0=lambda *_: None, _validate_model_preset=lambda *_: None,
        get_training_model=model_builder, build_inference_pg_collection=lambda *_a, **_kw: pg,
        core_transformer_config_from_args=lambda _: SimpleNamespace(),
        swap_model_weights=lambda *_a, **_kw: None, model_parallel_cuda_manual_seed=lambda *_: None,
        StaticInferenceContext=context, _prefill=prefill, _decode=decode, _remove_hooks=lambda *_: None,
        _load_bandwidth_matrix=lambda *_: {(0, 0): 100}, BandwidthAwareRefitPolicy=BandwidthAwareRefitPolicy,
        build_centralized_reshard_plan=plan, NCCLCopyService=Service,
        _all_max=lambda value, _: value, _all_min=lambda value, _: value, _all_ready=lambda value, _: value,
        _collect_active_collective_links=lambda *_a, **_kw: set(), collect_transfer_tasks=tasks,
        launch_reshard_plan=launch_reshard_plan, _hybrid_fast_path_decision=lambda *_a, **_kw: {"enabled": False, "reason": "disabled"},
        _configure_request_domains=request_domains, StandbyWeights=StandbyWeights,
        validate_release_mode=validate_release_mode, storage_bytes=storage_bytes,
        release_standby=release_hook, cuda_memory=lambda: {"allocated": 0, "reserved": 0},
        _write_results=lambda _, result: written.append(result), json=SimpleNamespace(dumps=lambda *_a, **_kw: ""))
    with patch("torch.cuda.Stream", return_value=Mock()), patch("torch.cuda.Event", Event), \
            patch("torch.cuda.stream", return_value=nullcontext()), patch("torch.cuda.current_stream", return_value=Mock()), \
            patch("tools.resharding.standby_weights.cuda_memory", return_value={"allocated": 0, "reserved": 0}):
        ns["run_live_benchmark"]()
    return written[0], models, releases


def test_actual_default_coordinator_result_matches_frozen_main():
    before, _, _ = run_coordinator(defaults.BASE)
    after, _, releases = run_coordinator(defaults.CURRENT)
    assert before == after
    assert releases == []


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("repeat", [False, True])
def test_actual_coordinator_releases_after_validation_and_migrates_check_delta(identity, repeat):
    result, models, releases = run_coordinator(defaults.CURRENT, release=True, identity=identity, repeat=repeat)
    assert len(releases) == 5  # Startup + four successful switches.
    assert models[0].w.untyped_storage().nbytes() == 48
    assert models[1].w.untyped_storage().nbytes() == 0
    assert ("kv_request_domains" in result) == identity
    for row, memory in zip(result["switches"], result["standby_weight_storage"]["ranks"][0]["switches"]):
        assert row["delta_tokens"] == row["base"]["overlap_steps"] + memory["weight_check_overlap_steps"]
        assert memory["all_weights_finite"] and memory["standby_kv_bytes"] > 0
        assert memory["inactive_tp"] == (4 if repeat or row["direction"] == "4->2" else 2)


def test_actual_coordinator_rejects_missing_weight_before_target_decode():
    with pytest.raises(AssertionError, match="cutover rejected"):
        run_coordinator(defaults.CURRENT, release=True, fault=True)


def test_pending_weight_check_stops_decode_at_cap_and_includes_all_kv_delta():
    result, _, _ = run_coordinator(defaults.CURRENT, release=True, check_polls=100)
    for row, memory in zip(result["switches"], result["standby_weight_storage"]["ranks"][0]["switches"]):
        assert memory["weight_check_overlap_steps"] == 2
        assert row["delta_tokens"] == row["base"]["overlap_steps"] + 2


def test_capacity_includes_additional_check_decode_only_when_enabled():
    # Legacy worst case is 8 + 1 + 4 * (1 * 2 + 2) = 25; release adds 8.
    run_coordinator(defaults.CURRENT, capacity=26)
    with pytest.raises(ValueError, match="max-position-embeddings"):
        run_coordinator(defaults.CURRENT, release=True, capacity=33)
    run_coordinator(defaults.CURRENT, release=True, capacity=34)


def test_identity_can_be_enabled_without_storage_release():
    result, _, releases = run_coordinator(defaults.CURRENT, identity=True)
    assert "kv_request_domains" in result
    assert "standby_weight_storage" not in result
    assert releases == []
