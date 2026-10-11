"""CPU TP-pair, layer-structure and real storage/reshard lifecycle regressions."""

import argparse
import ast
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_dp_sweep as fixtures
import test_live_defaults as defaults
import test_live_storage_lifecycle as lifecycle
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.resharding.live import select_adaptive_hybrid_policy
from megatron.core.transformer.transformer_config import TransformerConfig
from tools.resharding.correctness.harness import (
    Bundle,
    centralized_plans,
    finish_all,
    launch_all,
    param,
)
from tools.resharding.correctness.reference import Layout, exact_chunks, markers
from tools.resharding.standby_weights import StandbyWeights

PAIRS = [(1, 2, world) for world in range(2, 17, 2)] + [(2, 4, world) for world in range(4, 17, 4)]


@pytest.mark.parametrize("src,dst,world,overrides", [
    (1, 4, 4, {}), (2, 2, 4, {}), (3, 6, 6, {}),
    (1, 2, 3, {}), (2, 4, 6, {}),
    (1, 2, 2, {"tensor_model_parallel_size": 2}),
    (1, 2, 2, {"expert_tensor_parallel_size": 2}),
    (1, 2, 6, {"expert_model_parallel_size": 2}),
])
def test_coordinator_rejects_invalid_tp_pair_model_mismatch_and_world_divisibility(src, dst, world, overrides):
    builder = Mock(side_effect=AssertionError("invalid layout must fail before model construction"))
    with pytest.raises(ValueError):
        lifecycle.run_coordinator(defaults.CURRENT, src_tp=src, dst_tp=dst, world_size=world,
                                  live_overrides=overrides, namespace_overrides={"get_training_model": builder})
    builder.assert_not_called()


@pytest.mark.parametrize("direction,legacy", [("1->2", "2->4"), ("2->1", "4->2")])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("tasks", [3, 30])
def test_direction_parameterization_preserves_adaptive_and_fast_path_decisions(direction, legacy, candidate, tasks):
    kwargs = dict(requested_scheduler_mode="residual", candidate_route_available=candidate,
                  projected_transport_gain_pct=20.0, baseline_task_count=tasks,
                  candidate_task_count=tasks, max_wave_tasks=2, min_end_to_end_gain_pct=5.0,
                  residual_max_waves=4, foreground_pressure_fraction=0.1, max_foreground_pressure=0.5)
    assert select_adaptive_hybrid_policy(direction=direction, **kwargs) == select_adaptive_hybrid_policy(
        direction=legacy, **kwargs)
    namespace = defaults.load(defaults.CURRENT, ["_hybrid_fast_path_decision"])
    namespace["_foreground_pressure_fraction"] = lambda _: 0.1
    args = SimpleNamespace(live_hybrid_fast_path=True, live_hybrid_min_predicted_gain_pct=5.0,
                           live_hybrid_max_remote_cv=0.5, live_hybrid_max_foreground_pressure=0.5)
    plan = SimpleNamespace(source_route_stats={"projected_global_gain_pct": 20.0})
    tracker = SimpleNamespace(remote_cv=lambda: 0.1)
    decide = namespace["_hybrid_fast_path_decision"]
    assert decide(plan, tracker, args, direction=direction, plan_variant="candidate",
                  candidate_route_available=candidate) == decide(
        plan, tracker, args, direction=legacy, plan_variant="candidate", candidate_route_available=candidate)


@pytest.mark.parametrize("src,dst,world", PAIRS)
@pytest.mark.parametrize("limits", [(7, 11, 5), (7, 0, 0)])
def test_actual_coordinator_directions_caps_and_tp_keys(src, dst, world, limits):
    common, expansion, shrink = limits
    result, models, released = lifecycle.run_coordinator(
        defaults.CURRENT, release=True, src_tp=src, dst_tp=dst, world_size=world,
        live_overrides=dict(live_switches=2, live_max_wave_tasks=common,
                            live_expansion_max_wave_tasks=expansion, live_shrink_max_wave_tasks=shrink),
    )
    assert [row["direction"] for row in result["switches"]] == [f"{src}->{dst}", f"{dst}->{src}"]
    assert [row["base"]["max_wave_tasks"] for row in result["switches"]] == [
        expansion or common, shrink or common,
    ]
    assert set(result["final_residual_bandwidth_by_tp"]) == {str(src), str(dst)}
    assert result["final_active_tp"] == src
    assert [storage.parameters[0] is models[1].w for storage in released] == [True, False, True]
    assert models[1].w.untyped_storage().nbytes() == 0
    exact_chunks(models[0].w, torch.arange(12.).reshape(3, 4))
    memory = result["standby_weight_storage"]["ranks"][0]["switches"]
    assert [row["inactive_tp"] for row in memory] == [src, dst]
    assert all(row["all_weights_finite"] for row in memory)


def test_tp1_target_is_reallocated_poisoned_and_validated_before_old_source_release():
    events = []

    class TracedWeights(StandbyWeights):
        def __init__(self, model, **kwargs):
            super().__init__(model, **kwargs)
            self.tp = 1 if not events else 2
            events.append(("construct", self.tp))
        def release(self):
            assert self.finite_flag()
            super().release()
            events.append(("release", self.tp))
        def allocate_and_poison(self, stream=None):
            assert all(p.untyped_storage().nbytes() == 0 for p in self.parameters)
            super().allocate_and_poison(stream)
            assert all(torch.isnan(p).all() for p in self.parameters)
            events.append(("poison", self.tp))
        def finite_flag(self):
            result = super().finite_flag()
            if bool(result):
                events.append(("finite", self.tp))
            return result

    result, models, _ = lifecycle.run_coordinator(
        defaults.CURRENT, release=True, src_tp=1, dst_tp=2, world_size=2,
        live_overrides={"live_switches": 2}, namespace_overrides={"StandbyWeights": TracedWeights},
    )
    assert [event for event in events if event[0] in ("release", "poison")] == [
        ("release", 2), ("poison", 2), ("release", 1), ("poison", 1), ("release", 2),
    ]
    restore = events.index(("poison", 1))
    assert events.index(("finite", 1), restore) < len(events) - 1
    assert result["switches"][1]["validation_max_diff"] == 0
    exact_chunks(models[0].w, torch.arange(12.).reshape(3, 4))


def _bank(full, tp, world, *, empty=False):
    bank = {}
    for rank in range(world):
        ranks = tuple(range(rank // tp * tp, (rank // tp + 1) * tp))
        layout = Layout(ranks, 1)
        tensor = layout.shard(full, rank)
        weight = param(torch.full_like(tensor, float("nan")) if empty else tensor)
        weight.tensor_model_parallel = tp > 1
        weight.partition_dim, weight.partition_stride = 1, 1
        module = Bundle({"weight::w": weight})
        module.pg_collection = SimpleNamespace(
            tp=list(ranks), dp=list(range(rank % tp, world, tp)),
            ep=[rank], expt_tp=list(ranks), pp=[rank],
        )
        bank[rank] = module
    return bank


@pytest.mark.parametrize("world", range(2, 17, 2))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_tp1_tp2_planner_and_transactions_restore_released_tp1(world, dtype):
    # Real central planner, metadata extraction, transactions and storage. Only
    # collectives and transport are CPU doubles; the oracle uses global values.
    full = markers((3, 16), dtype=dtype)
    banks = {1: _bank(full, 1, world), 2: _bank(full, 2, world, empty=True)}
    plans = centralized_plans(banks[1], banks[2])
    transactions, _ = launch_all(plans, banks[1], banks[2])
    finish_all(transactions)  # Initial source -> standby synchronization.
    weights = {tp: {rank: StandbyWeights(model, protected_tensors=[])
                    for rank, model in bank.items()} for tp, bank in banks.items()}
    for storage in weights[2].values():
        storage.release()
    for source, target in ((1, 2), (2, 1)):
        for storage in weights[target].values():
            assert all(p.untyped_storage().nbytes() == 0 for p in storage.parameters)
            storage.allocate_and_poison()
            assert all(torch.isnan(p).all() for p in storage.parameters)
        plans = centralized_plans(banks[source], banks[target])
        remote_receives = sum(op.peer_rank != rank for rank, plan in plans.items() for op in plan.recv_ops)
        assert (remote_receives == 0) if source == 1 else (remote_receives >= world)
        transactions, mailbox = launch_all(plans, banks[source], banks[target])
        assert not mailbox.finished
        finish_all(transactions)
        assert mailbox.remote == remote_receives
        assert mailbox.local > 0
        for rank, model in banks[target].items():
            expected = Layout(tuple(range(rank // target * target, (rank // target + 1) * target)), 1).shard(full, rank)
            assert weights[target][rank].finite_flag()
            exact_chunks(next(model.parameters()), expected)
        for storage in weights[source].values():
            storage.release()
    assert all(next(model.parameters()).untyped_storage().nbytes() == 0 for model in banks[2].values())
    for model in banks[1].values():
        exact_chunks(next(model.parameters()), full)


def test_missing_remote_shrink_slice_is_rejected_after_tp1_reallocation():
    full = markers((3, 16))
    source, target = _bank(full, 2, 2), _bank(full, 1, 2)
    storage = StandbyWeights(target[0], protected_tensors=[])
    storage.release()
    storage.allocate_and_poison()
    plans = centralized_plans(source, target)
    omitted = next(op.task_id for op in plans[0].recv_ops if op.peer_rank != 0)
    for plan in plans.values():
        plan.send_ops = [op for op in plan.send_ops if op.task_id != omitted]
        plan.recv_ops = [op for op in plan.recv_ops if op.task_id != omitted]
    transactions, _ = launch_all(plans, source, target)
    finish_all(transactions)
    assert not storage.finite_flag()


def _preset(layers=27, skip=False):
    return SimpleNamespace(
        live_model_preset="deepseek-v2-lite", live_skip_checkpoint=skip,
        load=None if skip else "/converted/checkpoint", num_layers=layers,
        hidden_size=2048, ffn_hidden_size=10944, num_attention_heads=16, kv_channels=16,
        kv_lora_rank=512, q_lora_rank=None, qk_head_dim=128, qk_pos_emb_head_dim=64,
        v_head_dim=128, num_experts=64, moe_ffn_hidden_size=1408, moe_router_topk=6,
        moe_shared_expert_intermediate_size=2816, normalization="RMSNorm",
        layernorm_epsilon=1.0e-6, moe_layer_freq=[0] + [1] * (layers - 1),
        multi_latent_attention=True, qk_layernorm=True, swiglu=True, bf16=True,
        untie_embeddings_and_output_weights=True,
    )


@pytest.mark.parametrize("layers,skip", [(9, True), (27, False)])
def test_deepseek_preset_and_real_decoder_specs_preserve_dense_then_moe(layers, skip):
    args = _preset(layers, skip)
    defaults.load(defaults.CURRENT, ["_validate_model_preset"])["_validate_model_preset"](args)
    # Construct actual decoder specs with the DeepSeek dimensions; this does not
    # allocate model weights or invoke CUDA.
    config = TransformerConfig(
        num_layers=args.num_layers, hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size, num_attention_heads=args.num_attention_heads,
        num_moe_experts=args.num_experts, moe_layer_freq=args.moe_layer_freq,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size, moe_router_topk=args.moe_router_topk,
        moe_shared_expert_intermediate_size=args.moe_shared_expert_intermediate_size,
        multi_latent_attention=True, qk_layernorm=True, normalization="RMSNorm",
        use_cpu_initialization=True,
    )
    specs = get_gpt_decoder_block_spec(config, False, normalization="RMSNorm", pp_rank=0).layer_specs
    assert [spec.submodules.mlp.module.__name__ for spec in specs] == ["MLP"] + ["MoELayer"] * (layers - 1)


@pytest.mark.parametrize("change", [
    {"load": None, "live_skip_checkpoint": False},
    {"num_layers": 0}, {"num_layers": 28},
    {"moe_layer_freq": [1] * 9}, {"moe_layer_freq": [0, 0] + [1] * 7},
    {"moe_layer_freq": [0] + [1] * 26}, {"num_experts": 32},
])
def test_deepseek_rejects_missing_checkpoint_or_wrong_prefix(change):
    args = _preset(9, True)
    vars(args).update(change)
    with pytest.raises(ValueError):
        defaults.load(defaults.CURRENT, ["_validate_model_preset"])["_validate_model_preset"](args)


@pytest.mark.parametrize("tp,ep", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_metadata_counts_one_logical_model_and_actual_layer_types(tp, ep):
    world = 8
    counts = []
    args = _preset(9, True)
    models = []
    for rank in range(world):
        model = torch.nn.Module()
        for name, count, expert, sharded in (
            ("dense_weight", 12 // tp, False, True), ("dense_bias", 2, False, False),
            ("expert_weight", 24 // (tp * ep), True, True), ("expert_bias", 6 // ep, True, False),
        ):
            parameter = torch.nn.Parameter(torch.zeros(count))
            parameter.allreduce = not expert
            parameter.tensor_model_parallel = sharded and tp > 1
            model.register_parameter(name, parameter)
        model.pg_collection = SimpleNamespace(
            tp=list(range(rank // tp * tp, rank // tp * tp + tp)),
            expt_tp=list(range(rank // tp * tp, rank // tp * tp + tp)),
            dp=list(range(rank % tp, world, tp)),
            expt_dp=list(range(rank % (tp * ep), world, tp * ep)),
        )
        model.decoder = SimpleNamespace(layers=[
            SimpleNamespace(mlp=SimpleNamespace(**({"experts": object()} if index else {})))
            for index in range(9)
        ])
        models.append(model)
        namespace = defaults.load(defaults.CURRENT, ["_model_metadata"])
        def gather(output, value, **_):
            counts.append(value)
            output[:] = [value] * world
        namespace["dist"] = SimpleNamespace(get_rank=lambda group: group.index(rank),
                                           get_world_size=lambda _: world, all_gather_object=gather)
        namespace["_model_metadata"](args, model, None)
    assert sum(counts) == 44
    namespace["dist"].all_gather_object = lambda output, value, **_: output.__setitem__(slice(None), counts)
    result = namespace["_model_metadata"](args, SimpleNamespace(module=models[-1]), None)
    assert result == dict(weights_random_init=True, num_layers=9,
                          layer_types=["dense"] + ["moe"] * 8, parameter_count=44)


def test_skip_checkpoint_keeps_initial_sync_and_records_nine_layer_metadata():
    checkpoint = Mock(side_effect=AssertionError("random initialization must not load"))
    synced = []
    source_values = torch.randn((3, 4), generator=torch.Generator().manual_seed(123))
    def initialize(model, index):
        with torch.no_grad():
            model.w.copy_(source_values if index == 0 else torch.full_like(source_values, float("nan")))
    def sync(source, target, **_):
        exact_chunks(source.w, source_values)
        assert torch.isnan(target.w).all()
        with torch.no_grad():
            target.w.copy_(source.w)
        synced.append((source, target))
    result, models, _ = lifecycle.run_coordinator(
        defaults.CURRENT, release=True, src_tp=1, dst_tp=2, world_size=2,
        live_overrides=dict(live_model_preset="deepseek-v2-lite", live_skip_checkpoint=True,
                            num_layers=9, live_switches=2),
        namespace_overrides=dict(load_checkpoint=checkpoint, swap_model_weights=sync),
        model_initializer=initialize,
    )
    checkpoint.assert_not_called()
    assert synced == [(models[0], models[1])]
    assert not result["checkpoint_loaded"]
    assert result["weights_random_init"]
    assert result["num_layers"] == 9
    assert result["layer_types"] == ["dense"] + ["moe"] * 8
    assert result["switches"][1]["direction"] == "2->1"
    exact_chunks(models[0].w, source_values)


def test_default_deepseek_still_loads_checkpoint_and_matches_step1_result():
    baseline = ast.parse(fixtures.baseline_bytes("examples/rl/benchmark_live_moe_tp.py"))
    results = []
    for tree in (baseline, defaults.CURRENT):
        checkpoint = Mock(return_value=(17, None))
        console = []
        result, models, _ = lifecycle.run_coordinator(
            tree, live_overrides=dict(live_model_preset="deepseek-v2-lite", num_layers=27),
            namespace_overrides={"load_checkpoint": checkpoint, "print_rank_0": console.append},
        )
        checkpoint.assert_called_once_with([models[0]], None, None)
        assert result["checkpoint_loaded"] and result["checkpoint_iteration"] == 17
        assert "weights_random_init" not in result
        results.append((result, "\n".join(console).encode("utf-8")))
    assert results[0] == results[1]


def test_new_parser_defaults_do_not_change_printed_namespace():
    namespaces = []
    for tree in (ast.parse(fixtures.baseline_bytes("examples/rl/benchmark_live_moe_tp.py")), defaults.CURRENT):
        parser = argparse.ArgumentParser()
        defaults.load(tree, ["add_live_args"])["add_live_args"](parser)
        namespaces.append(vars(parser.parse_args([])))
    assert namespaces[0] == namespaces[1]
    assert "live_src_tp" not in namespaces[1] and "live_skip_checkpoint" not in namespaces[1]
