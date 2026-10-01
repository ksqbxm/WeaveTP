#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Live MoE TP2<->TP4 reshard benchmark with state-preserving KV cutover.

The active model keeps executing real MoE decode steps while a standby layout
receives model weights and the stable KV prefix through non-blocking NCCL. At a
token boundary, only KV entries appended during the overlap are refreshed, then
the active model pointer changes. Residual link service rates observed after
each wave are fed back into the scheduler before it packs the remaining tasks.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.resharding import weavetp_observations as observations

from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.resharding import (
    BandwidthAwareRefitPolicy,
    MigrationFirstGuard,
    OnlineResidualPlanController,
    ResidualBandwidthTracker,
    ResidualBandwidthWaveScheduler,
    build_centralized_reshard_plan,
    collect_transfer_tasks,
    filter_plan_by_task_ids,
    launch_reshard_plan,
    restrict_plan_sequence,
    select_adaptive_hybrid_policy,
    swap_model_weights,
)
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.rl.parallel_utils import build_inference_pg_collection
from megatron.training import get_args
from megatron.training import get_model as get_training_model
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args, parse_and_validate_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

_HOTSPOT_PRESSURE_BUFFERS: dict[
    tuple[int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


class SequenceParallelTorchLayerNorm(torch.nn.LayerNorm):
    """Torch LayerNorm carrying Megatron sequence-parallel metadata."""

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1.0e-5,
        **_kwargs,
    ) -> None:
        if config.normalization != "LayerNorm":
            raise ValueError("live benchmark currently requires LayerNorm")
        if config.layernorm_zero_centered_gamma:
            raise ValueError("Torch LayerNorm does not support zero-centered gamma")
        super().__init__(hidden_size, eps=eps)
        self.sequence_parallel = bool(config.sequence_parallel)
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        setattr(self.bias, "sequence_parallel", self.sequence_parallel)


class LiveTopologyInvariantTopKRouter(TopKRouter):
    """Deterministic MoE routing that remains stable across TP layouts."""

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        logits = super().gating(input)
        mode = getattr(self.config, "live_router_mode", "balanced")
        if mode == "native":
            return logits
        if mode == "balanced":
            token_ids = torch.arange(
                logits.numel() // self.num_experts,
                device=logits.device,
            ).reshape(logits.shape[:-1])
            targets = token_ids.remainder(self.num_experts)
            routed = torch.full_like(logits, -1.0e4)
            routed.scatter_(-1, targets.unsqueeze(-1), 0.0)
            if self.config.moe_router_topk > 1:
                second = (targets + 1).remainder(self.num_experts)
                routed.scatter_(-1, second.unsqueeze(-1), -1.0)
            return logits * 0.0 + routed

        active_experts = tuple(getattr(self.config, "live_active_experts", (0, 1)))
        mask = torch.zeros(self.num_experts, device=logits.device, dtype=torch.bool)
        mask[list(active_experts)] = True
        if mode == "fixed-hot":
            return logits.masked_fill(~mask.unsqueeze(0), float("-inf"))

        if mode == "shifting-hot":
            layer = max((self.layer_number or 1) - 1, 0)
            hot = layer % self.num_experts
            expert_ids = torch.arange(self.num_experts, device=logits.device)
            bias = -abs(float(getattr(self.config, "live_router_bias", 8.0))) * (
                (expert_ids - hot) % self.num_experts
            ).to(logits.dtype)
            return logits + bias
        raise ValueError(f"unsupported live router mode: {mode}")


class StaticKVCacheModule(torch.nn.Module):
    """Expose a StaticInferenceContext KV cache to the reshard planner."""

    def __init__(self, context: StaticInferenceContext, pg_collection) -> None:
        super().__init__()
        self.pg_collection = pg_collection
        self._entries: list[tuple[str, torch.nn.Parameter]] = []
        for layer_number in sorted(context.key_value_memory_dict):
            tensors = context.key_value_memory_dict[layer_number]
            if not isinstance(tensors, tuple) or len(tensors) != 2:
                continue
            for kind, tensor in zip(("key", "value"), tensors):
                parameter = torch.nn.Parameter(tensor, requires_grad=False)
                if parameter.data_ptr() != tensor.data_ptr():
                    raise RuntimeError("KV parameter wrapper must alias inference storage")
                parameter.tensor_model_parallel = True
                parameter.partition_dim = 2
                parameter.partition_stride = 1
                parameter.allreduce = True
                name = f"layer_{int(layer_number):04d}.{kind}"
                self._entries.append((name, parameter))
        if not self._entries:
            raise RuntimeError("real model forward did not allocate a static KV cache")

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        name_prefix = f"{prefix}." if prefix else ""
        for name, parameter in self._entries:
            yield name_prefix + name, parameter


class LiveStateBundle(torch.nn.Module):
    """Expose model weights and a full preallocated KV cache as one plan."""

    def __init__(self, model: GPTModel, kv_cache: StaticKVCacheModule) -> None:
        super().__init__()
        self.config = model.config
        self.pg_collection = model.pg_collection
        self._entries = [
            (f"weight::{name}", parameter) for name, parameter in model.named_parameters()
        ]
        self._entries.extend(
            (f"kv::{name}", parameter) for name, parameter in kv_cache.named_parameters()
        )

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        name_prefix = f"{prefix}." if prefix else ""
        for name, parameter in self._entries:
            yield name_prefix + name, parameter


def add_live_args(parser):
    group = parser.add_argument_group(title="live MoE TP benchmark")
    group.add_argument(
        "--live-model-preset",
        choices=("synthetic", "deepseek-v2-lite"),
        default="synthetic",
    )
    group.add_argument("--live-dst-tp", type=int, default=4)
    group.add_argument("--live-switches", type=int, default=4)
    group.add_argument(
        "--live-repeat-forward",
        action="store_true",
        help="Repeat 2->4 expansions in one process instead of alternating with 4->2.",
    )
    group.add_argument(
        "--live-method-variant",
        choices=(
            "baseline",
            "moetp++",
            "moetp++-hybrid",
            "dance-moe",
            "harmoeny-proxy",
            "expertflow-proxy",
            "firm-moe-proxy",
            "flying-serving-proxy",
            "anchortp-proxy",
            "llumnix-proxy",
        ),
        default="moetp++",
        help="Label the tested policy variant for reproducible comparisons.",
    )
    group.add_argument("--live-prompt-tokens", type=int, default=8)
    group.add_argument("--live-min-overlap-steps", type=int, default=1)
    group.add_argument("--live-max-overlap-steps", type=int, default=64)
    group.add_argument("--live-max-waves", type=int, default=4)
    group.add_argument("--live-max-wave-bytes", type=int, default=0)
    group.add_argument("--live-max-wave-tasks", type=int, default=0)
    group.add_argument(
        "--live-expansion-max-wave-tasks",
        type=int,
        default=0,
        help=(
            "Optional TP2->TP4 task bound; zero reuses --live-max-wave-tasks. "
            "This also bounds the post-overlap KV delta."
        ),
    )
    group.add_argument(
        "--live-shrink-max-wave-tasks",
        type=int,
        default=0,
        help=(
            "Optional TP4->TP2 task bound; zero reuses --live-max-wave-tasks. "
            "This also bounds the post-overlap KV delta."
        ),
    )
    group.add_argument(
        "--live-hybrid-fast-path",
        action="store_true",
        help=(
            "Use a guarded single-wave candidate path when residual bandwidth "
            "and foreground pressure are favorable."
        ),
    )
    group.add_argument(
        "--live-hybrid-min-predicted-gain-pct", type=float, default=10.0
    )
    group.add_argument("--live-hybrid-max-remote-cv", type=float, default=0.22)
    group.add_argument(
        "--live-hybrid-max-foreground-pressure", type=float, default=0.20
    )
    group.add_argument(
        "--live-adaptive-hybrid",
        action="store_true",
        help=(
            "Keep MOETP source rerouting while selecting the route and wave "
            "scheduler with an end-to-end overhead gate."
        ),
    )
    group.add_argument(
        "--live-adaptive-min-predicted-e2e-gain-pct", type=float, default=5.0
    )
    group.add_argument(
        "--live-adaptive-residual-max-waves",
        type=int,
        default=8,
        help=(
            "Use residual repacking only up to this predicted wave count; "
            "larger plans use bounded FIFO waves."
        ),
    )
    group.add_argument(
        "--live-scheduler-mode",
        choices=("baseline", "residual"),
        default="residual",
    )
    group.add_argument("--live-bandwidth-ewma", type=float, default=0.5)
    group.add_argument("--live-bandwidth-floor-fraction", type=float, default=0.05)
    group.add_argument("--live-bandwidth-profile", type=str, default=None)
    group.add_argument("--live-default-remote-gbps", type=float, default=100.0)
    group.add_argument("--live-default-local-gbps", type=float, default=1000.0)
    group.add_argument("--live-hotness-weight", type=float, default=0.0)
    group.add_argument(
        "--live-router-mode",
        choices=("native", "balanced", "fixed-hot", "shifting-hot"),
        default="fixed-hot",
    )
    group.add_argument("--live-active-experts", type=str, default="0,1")
    group.add_argument(
        "--live-active-expert-phases",
        type=str,
        default="",
        help="Semicolon-separated expert sets, for example '0,1;512,513'.",
    )
    group.add_argument("--live-hotspot-phase-expansions", type=int, default=1)
    group.add_argument("--live-hotspot-pressure-bytes", type=int, default=0)
    group.add_argument("--live-hotspot-pressure-iters", type=int, default=1)
    group.add_argument(
        "--live-pressure-rank-phases",
        type=str,
        default="",
        help="Optional semicolon-separated rank sets, for example '0,1;4,5'.",
    )
    group.add_argument("--live-router-bias", type=float, default=8.0)
    group.add_argument("--live-allow-nonlocal-reroute", action="store_true")
    group.add_argument("--live-disable-source-reroute", action="store_true")
    group.add_argument("--live-emulate-noncollocated-sources", action="store_true")
    group.add_argument("--live-allow-aware-shrink", action="store_true")
    group.add_argument("--live-force-bandwidth-routing", action="store_true")
    group.add_argument("--live-reroute-min-gain-pct", type=float, default=10.0)
    group.add_argument("--live-reroute-min-contention-gain-pct", type=float, default=0.0)
    group.add_argument("--live-reroute-min-global-gain-pct", type=float, default=5.0)
    group.add_argument("--live-reroute-penalty-us", type=float, default=20.0)
    group.add_argument("--live-reroute-min-bytes", type=int, default=1 << 20)
    group.add_argument("--live-pack-target-bytes", type=int, default=4 << 20)
    group.add_argument("--live-pack-max-item-bytes", type=int, default=1 << 20)
    group.add_argument("--live-unpack-chunk-bytes", type=int, default=64 << 20)
    group.add_argument(
        "--live-p2p-order",
        choices=(
            "send-recv",
            "nccl-round",
            "task-round",
            "small-first",
            "large-first",
            "peer-size-asc",
            "peer-size-desc",
        ),
        default="nccl-round",
        help="Ordering used for NCCL point-to-point send/recv submission.",
    )
    group.add_argument("--live-persistent-pack-buffers", action="store_true")
    group.add_argument("--live-pack-rerouted-only", action="store_true")
    group.add_argument(
        "--live-online-migration-first-guard",
        "--live-online-dual-objective-guard",
        dest="live_online_migration_first_guard",
        action="store_true",
    )
    group.add_argument("--live-min-transport-gain-pct", type=float, default=1.0)
    group.add_argument("--live-max-tpot-regression-pct", type=float, default=5.0)
    group.add_argument(
        "--live-guard-tpot-metric", choices=("mean", "p95", "p99"), default="p95"
    )
    group.add_argument("--live-guard-ewma", type=float, default=0.5)
    group.add_argument("--live-guard-robust-window", type=int, default=3)
    group.add_argument("--live-candidate-warmup-expansions", type=int, default=1)
    group.add_argument("--live-guard-min-samples", type=int, default=3)
    group.add_argument(
        "--live-guard-min-candidate-win-rate", type=float, default=2.0 / 3.0
    )
    group.add_argument(
        "--live-guard-max-transport-regression-pct", type=float, default=2.0
    )
    group.add_argument("--live-online-replan", action="store_true")
    group.add_argument("--live-replan-min-change-pct", type=float, default=5.0)
    group.add_argument("--live-replan-max-stale-expansions", type=int, default=4)
    group.add_argument("--live-replan-noop-cooldown-expansions", type=int, default=8)
    group.add_argument("--live-amortization-horizon-expansions", type=int, default=0)
    group.add_argument("--live-foreground-pressure-cap", type=float, default=0.90)
    group.add_argument("--live-foreground-baseline-ewma", type=float, default=0.25)
    group.add_argument("--live-guard-reevaluate-expansions", type=int, default=0)
    group.add_argument("--live-guard-hysteresis-pct", type=float, default=0.0)
    group.add_argument("--live-logit-atol", type=float, default=5.0e-3)
    group.add_argument("--live-logit-rtol", type=float, default=5.0e-3)
    group.add_argument(
        "--live-logit-validation-mode",
        choices=("allclose", "bf16-relative"),
        default="allclose",
    )
    group.add_argument("--live-logit-max-nrmse", type=float, default=0.35)
    group.add_argument("--live-logit-min-cosine", type=float, default=0.97)
    group.add_argument("--live-logit-min-top1-agreement", type=float, default=0.80)
    group.add_argument("--live-diagnose-equivalence", action="store_true")
    group.add_argument("--live-json-output", type=str, default=None)
    return parser


def _parse_experts(spec: str, num_experts: int, topk: int) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in spec.split(",") if value.strip()))
    if len(values) < topk:
        raise ValueError("--live-active-experts must contain at least moe-router-topk experts")
    if any(value < 0 or value >= num_experts for value in values):
        raise ValueError(f"active experts must be in [0, {num_experts})")
    return values


def _parse_expert_phases(
    spec: str,
    *,
    fallback: tuple[int, ...],
    num_experts: int,
    topk: int,
) -> tuple[tuple[int, ...], ...]:
    if not spec.strip():
        return (fallback,)
    phases = tuple(
        _parse_experts(phase, num_experts, topk)
        for phase in spec.split(";")
        if phase.strip()
    )
    if not phases:
        raise ValueError("--live-active-expert-phases must contain an expert set")
    return phases


def _active_experts_for_switch(args, switch_index: int) -> tuple[int, ...]:
    expansion_index = max(int(switch_index), 0) // 2
    phase_index = (
        expansion_index // max(int(args.live_hotspot_phase_expansions), 1)
    ) % len(args.live_active_expert_phases_tuple)
    return args.live_active_expert_phases_tuple[phase_index]


def _parse_rank_phases(spec: str, world_size: int) -> tuple[tuple[int, ...], ...]:
    if not spec.strip():
        return ((),)
    phases = tuple(
        tuple(
            dict.fromkeys(
                int(value.strip())
                for value in phase.split(",")
                if value.strip()
            )
        )
        for phase in spec.split(";")
        if phase.strip()
    )
    if not phases or any(not phase for phase in phases):
        raise ValueError("live pressure rank phases must contain non-empty rank sets")
    if any(rank < 0 or rank >= world_size for phase in phases for rank in phase):
        raise ValueError(f"live pressure ranks must be in [0, {world_size})")
    return phases


def _pressure_ranks_for_switch(args, switch_index: int) -> tuple[int, ...]:
    expansion_index = max(int(switch_index), 0) // 2
    phase_index = (
        expansion_index // max(int(args.live_hotspot_phase_expansions), 1)
    ) % len(args.live_pressure_rank_phases_tuple)
    return args.live_pressure_rank_phases_tuple[phase_index]


def _set_active_experts(models: Iterable[GPTModel], active_experts: tuple[int, ...]) -> None:
    for model in models:
        model.config.live_active_experts = active_experts


def _local_rank_owns_hot_expert(
    model: GPTModel, active_experts: tuple[int, ...], num_experts: int
) -> bool:
    ep_group = getattr(model.pg_collection, "ep", None)
    if ep_group is None or ep_group.size() <= 1:
        return False
    ranks = tuple(dist.get_process_group_ranks(ep_group))
    if num_experts % len(ranks) != 0:
        return False
    experts_per_rank = num_experts // len(ranks)
    owner_ranks = {
        ranks[min(expert // experts_per_rank, len(ranks) - 1)]
        for expert in active_experts
    }
    return dist.get_rank() in owner_ranks


def _apply_hotspot_pressure(model: GPTModel, args) -> None:
    num_bytes = int(args.live_hotspot_pressure_bytes)
    explicit_pressure = bool(args.live_pressure_ranks_tuple)
    if (
        num_bytes <= 0
        or (
            dist.get_rank() not in args.live_pressure_ranks_tuple
            if explicit_pressure
            else not _local_rank_owns_hot_expert(
                model, args.live_active_experts_tuple, args.num_experts
            )
        )
    ):
        return
    key = (torch.cuda.current_device(), num_bytes)
    buffers = _HOTSPOT_PRESSURE_BUFFERS.get(key)
    if buffers is None:
        device = torch.device("cuda", torch.cuda.current_device())
        buffers = (
            torch.empty(num_bytes, dtype=torch.uint8, device=device),
            torch.empty(num_bytes, dtype=torch.uint8, device=device),
        )
        _HOTSPOT_PRESSURE_BUFFERS[key] = buffers
    source, destination = buffers
    for _ in range(args.live_hotspot_pressure_iters):
        destination.copy_(source, non_blocking=True)
        source, destination = destination, source


def _configure_live_model(config: TransformerConfig, args) -> None:
    # Autoregressive decode advances one token at a time, so its sequence
    # dimension is not divisible by TP2/TP4 as sequence parallel requires.
    config.sequence_parallel = False
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.moe_router_load_balancing_type = "none"
    config.moe_aux_loss_coeff = 0.0
    config.moe_router_force_load_balancing = False
    config.live_router_mode = args.live_router_mode
    config.live_active_experts = args.live_active_experts_tuple
    config.live_router_bias = args.live_router_bias


def _validate_model_preset(args) -> None:
    if args.live_model_preset == "synthetic":
        return
    if not args.load:
        raise ValueError("deepseek-v2-lite requires --load with a converted Megatron checkpoint")

    expected = {
        "num_layers": 27,
        "hidden_size": 2048,
        "ffn_hidden_size": 10944,
        "num_attention_heads": 16,
        "kv_channels": 16,
        "kv_lora_rank": 512,
        "q_lora_rank": None,
        "qk_head_dim": 128,
        "qk_pos_emb_head_dim": 64,
        "v_head_dim": 128,
        "num_experts": 64,
        "moe_ffn_hidden_size": 1408,
        "moe_router_topk": 6,
        "moe_shared_expert_intermediate_size": 2816,
        "normalization": "RMSNorm",
        "layernorm_epsilon": 1.0e-6,
    }
    mismatches = [
        f"{name}={getattr(args, name, None)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name, None) != value
    ]
    expected_moe_pattern = [0] + [1] * 26
    if args.moe_layer_freq != expected_moe_pattern:
        mismatches.append(
            f"moe_layer_freq={args.moe_layer_freq!r} (expected {expected_moe_pattern!r})"
        )
    required_flags = {
        "multi_latent_attention": True,
        "qk_layernorm": True,
        "swiglu": True,
        "bf16": True,
        "untie_embeddings_and_output_weights": True,
    }
    mismatches.extend(
        f"{name}={getattr(args, name, None)!r} (expected {value!r})"
        for name, value in required_flags.items()
        if getattr(args, name, None) is not value
    )
    if mismatches:
        raise ValueError(
            "DeepSeek-V2-Lite preset does not match the official architecture: "
            + "; ".join(mismatches)
        )


def _install_live_router(block_spec: TransformerBlockSubmodules) -> None:
    for layer_spec in block_spec.layer_specs:
        mlp_submodules = getattr(layer_spec.submodules.mlp, "submodules", None)
        if mlp_submodules is not None and hasattr(mlp_submodules, "router"):
            mlp_submodules.router = LiveTopologyInvariantTopKRouter


def model_provider(
    pre_process=True,
    post_process=True,
    parallel_output=False,
    pg_collection=None,
    config=None,
):
    args = get_args()
    if config is None:
        config = core_transformer_config_from_args(args)
    _configure_live_model(config, args)
    if args.live_model_preset == "deepseek-v2-lite":
        block_spec = get_gpt_decoder_block_spec(
            config,
            use_transformer_engine=False,
            normalization=args.normalization,
        )
        if args.live_router_mode != "native":
            _install_live_router(block_spec)
        return GPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=False,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=False,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            pg_collection=pg_collection,
        )

    layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=False,
    )
    layer_spec.submodules.mlp.submodules.router = LiveTopologyInvariantTopKRouter
    layer_spec.submodules.input_layernorm = SequenceParallelTorchLayerNorm
    layer_spec.submodules.pre_mlp_layernorm = SequenceParallelTorchLayerNorm
    block_spec = TransformerBlockSubmodules(
        layer_specs=[layer_spec] * config.num_layers,
        layer_norm=SequenceParallelTorchLayerNorm,
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=block_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=False,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        pg_collection=pg_collection,
    )


def _load_bandwidth_matrix(args, world_size: int) -> dict[tuple[int, int], float]:
    if args.live_bandwidth_profile:
        payload = json.loads(Path(args.live_bandwidth_profile).read_text(encoding="utf-8"))
        matrix = payload.get("matrix_gbps")
        if (
            matrix is None
            or len(matrix) != world_size
            or any(len(row) != world_size for row in matrix)
        ):
            raise ValueError("live bandwidth profile must match WORLD_SIZE")
        return {
            (src, dst): float(value)
            for src, row in enumerate(matrix)
            for dst, value in enumerate(row)
        }
    return {
        (src, dst): (
            args.live_default_local_gbps if src == dst else args.live_default_remote_gbps
        )
        for src in range(world_size)
        for dst in range(world_size)
    }


def _make_tokens(args, length: int, *, offset: int = 0) -> torch.Tensor:
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.arange(offset, offset + length, device=device, dtype=torch.long)
    values = (values + 17).remainder(args.padded_vocab_size)
    return values.unsqueeze(0).expand(args.micro_batch_size, -1).contiguous()


@torch.inference_mode()
def _prefill(model: GPTModel, context: StaticInferenceContext, args) -> torch.Tensor:
    tokens = _make_tokens(args, args.live_prompt_tokens)
    positions = torch.arange(
        args.live_prompt_tokens,
        device=tokens.device,
        dtype=torch.long,
    ).unsqueeze(0).expand_as(tokens)
    context.sequence_len_offset = 0
    context.batch_size_offset = 0
    context.enable_prefill_mode()
    logits = model(
        tokens,
        positions,
        None,
        inference_context=context,
        runtime_gather_output=True,
    )
    torch.cuda.current_stream().synchronize()
    context.sequence_len_offset = args.live_prompt_tokens
    context.enable_decode_mode()
    return logits


@torch.inference_mode()
def _decode(
    model: GPTModel,
    context: StaticInferenceContext,
    args,
    *,
    token_value: int,
) -> tuple[torch.Tensor, float]:
    device = torch.device("cuda", torch.cuda.current_device())
    tokens = torch.full(
        (args.micro_batch_size, 1),
        token_value % args.padded_vocab_size,
        device=device,
        dtype=torch.long,
    )
    positions = torch.full_like(tokens, context.sequence_len_offset)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _apply_hotspot_pressure(model, args)
    logits = model(
        tokens,
        positions,
        None,
        inference_context=context,
        runtime_gather_output=True,
    )
    end.record()
    end.synchronize()
    context.sequence_len_offset += 1
    context.enable_decode_mode()
    return logits, float(start.elapsed_time(end))


@torch.inference_mode()
def _poison_kv(context: StaticInferenceContext, end: int) -> None:
    for key, value in context.key_value_memory_dict.values():
        key[:end].fill_(float("nan"))
        value[:end].fill_(float("nan"))


def _all_max(value: float, control_group) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=control_group)
    return float(tensor.item())


def _all_min(value: float, control_group) -> float:
    tensor = torch.tensor([value], dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=control_group)
    return float(tensor.item())


def _all_ready(ready: bool, control_group) -> bool:
    tensor = torch.tensor([int(ready)], dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=control_group)
    return bool(tensor.item())


def _comparison_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    control_group,
) -> dict[str, float]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"equivalence shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    difference = reference - candidate
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
    difference_flat = difference.reshape(-1)
    reference_rms = float(reference_flat.square().mean().sqrt().item())
    difference_rms = float(difference_flat.square().mean().sqrt().item())
    denominator = float(reference_flat.norm().item() * candidate_flat.norm().item())
    cosine = (
        float(torch.dot(reference_flat, candidate_flat).item()) / denominator
        if denominator > 0.0
        else 1.0
    )
    if reference.ndim >= 1 and reference.shape[-1] > 1:
        top1_agreement = float(
            (reference.argmax(dim=-1) == candidate.argmax(dim=-1)).float().mean().item()
        )
    else:
        top1_agreement = 1.0
    return {
        "max_abs": _all_max(float(difference_flat.abs().max().item()), control_group),
        "mean_abs": _all_max(float(difference_flat.abs().mean().item()), control_group),
        "nrmse": _all_max(difference_rms / max(reference_rms, 1.0e-12), control_group),
        "cosine": _all_min(cosine, control_group),
        "top1_agreement": _all_min(top1_agreement, control_group),
        "reference_max_abs": _all_max(
            float(reference_flat.abs().max().item()), control_group
        ),
    }


def _primary_tensor(output) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            tensor = _primary_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for value in output.values():
            tensor = _primary_tensor(value)
            if tensor is not None:
                return tensor
    return None


def _install_layer_capture(model: GPTModel):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    layers = [
        (int(module.layer_number) - 1, module)
        for _name, module in model.named_modules()
        if module.__class__.__name__ == "TransformerLayer"
        and isinstance(getattr(module, "layer_number", None), int)
    ]
    if not layers:
        raise AssertionError("equivalence diagnostics found no Transformer layers")
    for layer_index, layer in sorted(layers, key=lambda item: item[0]):
        def capture(_module, _inputs, output, *, index=layer_index):
            tensor = _primary_tensor(output)
            if tensor is not None:
                captured[index] = tensor.detach().clone()

        handles.append(layer.register_forward_hook(capture))
    return captured, handles


def _remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def _print_equivalence_diagnostics(
    *,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    reference_layers: dict[int, torch.Tensor],
    candidate_layers: dict[int, torch.Tensor],
    control_group,
) -> None:
    logits_metrics = _comparison_metrics(
        reference_logits, candidate_logits, control_group
    )
    print_rank_0(f"Initial TP2/TP4 prefill equivalence: {logits_metrics}")
    common_layers = sorted(set(reference_layers) & set(candidate_layers))
    if len(common_layers) != len(reference_layers) or len(common_layers) != len(candidate_layers):
        raise AssertionError(
            "equivalence diagnostics did not capture the same layers for TP2 and TP4"
        )
    for layer_index in common_layers:
        metrics = _comparison_metrics(
            reference_layers[layer_index],
            candidate_layers[layer_index],
            control_group,
        )
        print_rank_0(f"Initial layer {layer_index:02d} equivalence: {metrics}")


def _collect_active_collective_links(
    model: GPTModel,
    control_group,
    *,
    active_experts: tuple[int, ...],
    num_experts: int,
    pressure_ranks: tuple[int, ...] = (),
) -> set[tuple[int, int]]:
    """Return directed links used by the active TP and hot-expert collectives."""
    local_groups = set()
    for name in ("tp", "expt_tp", "ep"):
        group = getattr(model.pg_collection, name, None)
        if group is None or group.size() <= 1:
            continue
        local_groups.add(
            (name, tuple(dist.get_process_group_ranks(group)))
        )
    gathered = [None] * dist.get_world_size(group=control_group)
    dist.all_gather_object(gathered, sorted(local_groups), group=control_group)
    groups = {
        (str(name), tuple(ranks))
        for rank_groups in gathered
        for name, ranks in rank_groups
    }
    links = set()
    for name, ranks in groups:
        if name != "ep" or num_experts % len(ranks) != 0:
            links.update(
                (src, dst) for src in ranks for dst in ranks if src != dst
            )
            continue

        experts_per_rank = num_experts // len(ranks)
        owner_ranks = {
            ranks[min(expert // experts_per_rank, len(ranks) - 1)]
            for expert in active_experts
        }
        for owner in owner_ranks:
            for peer in ranks:
                if peer != owner:
                    links.add((peer, owner))
                    links.add((owner, peer))
    world_size = dist.get_world_size(group=control_group)
    for pressure_rank in pressure_ranks:
        for peer in range(world_size):
            if peer != pressure_rank:
                links.add((pressure_rank, peer))
                links.add((peer, pressure_rank))
    return links


def _expert_hotness(name: str, active_experts: tuple[int, ...]) -> float:
    match = re.search(r"(?:local_experts\.|weight)(\d+)(?:\.|$)", name)
    if match is None:
        return 0.0
    return 1.0 if int(match.group(1)) in active_experts else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _tasks_for_ids(tasks, task_ids: Iterable[int]):
    selected = set(task_ids)
    return [task for task in tasks if task.task_id in selected]


def _global_plan_task_count(plan, control_group) -> int:
    count = torch.tensor([len(plan.recv_ops)], dtype=torch.int64)
    dist.all_reduce(count, op=dist.ReduceOp.SUM, group=control_group)
    return int(count.item())


def _foreground_pressure_fraction(tracker: ResidualBandwidthTracker) -> float:
    return max(
        (
            float(tracker.foreground_gbps.get(pair, 0.0))
            / max(float(tracker.capacity_gbps[pair]), 1.0e-9)
            for pair in tracker.capacity_gbps
            if pair[0] != pair[1]
        ),
        default=0.0,
    )


def _hybrid_fast_path_decision(
    plan,
    tracker: ResidualBandwidthTracker,
    args,
    *,
    direction: str,
    plan_variant: str,
    candidate_route_available: bool,
) -> dict[str, object]:
    """Gate the PAT/Moebius-inspired one-wave path with MOETP signals."""

    route_stats = getattr(plan, "source_route_stats", {}) or {}
    predicted_gain_pct = float(route_stats.get("projected_global_gain_pct", 0.0))
    remote_cv = float(tracker.remote_cv())
    pressure_fraction = _foreground_pressure_fraction(tracker)

    checks = [
        ("disabled", not args.live_hybrid_fast_path),
        ("direction_not_supported", direction != "2->4"),
        ("candidate_plan_unavailable", not candidate_route_available),
        ("baseline_variant", plan_variant != "candidate"),
        (
            "predicted_gain_below_threshold",
            predicted_gain_pct < args.live_hybrid_min_predicted_gain_pct,
        ),
        ("remote_cv_too_high", remote_cv > args.live_hybrid_max_remote_cv),
        (
            "foreground_pressure_too_high",
            pressure_fraction > args.live_hybrid_max_foreground_pressure,
        ),
    ]
    reason = next((name for name, failed in checks if failed), "accepted")
    return {
        "enabled": reason == "accepted",
        "reason": reason,
        "predicted_global_gain_pct": predicted_gain_pct,
        "remote_cv": remote_cv,
        "foreground_pressure_fraction": pressure_fraction,
        "min_predicted_gain_pct": args.live_hybrid_min_predicted_gain_pct,
        "max_remote_cv": args.live_hybrid_max_remote_cv,
        "max_foreground_pressure": args.live_hybrid_max_foreground_pressure,
    }


@torch.inference_mode()
def _run_async_waves(
    *,
    plan,
    src_bundle: LiveStateBundle,
    dst_bundle: LiveStateBundle,
    active_model: GPTModel,
    active_context: StaticInferenceContext,
    service: NCCLCopyService,
    tracker: ResidualBandwidthTracker,
    scheduler: ResidualBandwidthWaveScheduler,
    reshard_group,
    control_group,
    args,
    token_counter: int,
    foreground_baseline_tpot_ms: float,
    foreground_links: set[tuple[int, int]],
    wave_scheduler_mode: str | None = None,
    max_wave_tasks: int | None = None,
    hybrid_fast_path: bool = False,
    hybrid_fast_path_reason: dict[str, object] | None = None,
) -> tuple[dict[str, object], int]:
    effective_scheduler_mode = wave_scheduler_mode or args.live_scheduler_mode
    wave_task_limit = args.live_max_wave_tasks if max_wave_tasks is None else max_wave_tasks
    if effective_scheduler_mode not in {"baseline", "fifo", "residual"}:
        raise ValueError(f"unknown wave scheduler mode: {effective_scheduler_mode}")
    if wave_task_limit < 0:
        raise ValueError("wave task limit must be non-negative")
    hotness = {
        name: _expert_hotness(name, args.live_active_experts_tuple)
        for name, _parameter in dst_bundle.named_parameters()
    }
    tasks = collect_transfer_tasks(plan, dst_bundle, group=reshard_group, hotness=hotness)
    remaining = {task.task_id: task for task in tasks}
    wave_records = []
    tpot_ms = []
    foreground_pressure = []
    total_exposed_wait_s = 0.0
    wall_start = time.perf_counter()
    completed_waves = 0

    while remaining:
        if hybrid_fast_path and completed_waves == 0:
            # The candidate route has already passed the residual-bandwidth
            # and foreground-pressure gate. Keep the fast path bounded by the
            # same NCCL FIFO limit; it still avoids residual re-packing on the
            # first wave without submitting an unbounded coalesced batch.
            remaining_ids = list(remaining)
            task_ids = (
                remaining_ids[:wave_task_limit]
                if wave_task_limit
                else remaining_ids
            )
        elif effective_scheduler_mode in {"baseline", "fifo"}:
            # Keep the reference path deterministic, but bound each NCCL P2P
            # batch when requested.  Large reverse TP plans can contain many
            # thousands of small sends/recvs; a single coalesced batch may
            # exceed NCCL's reliable FIFO window even when every rank submits
            # the same operation count.
            remaining_ids = list(remaining)
            if completed_waves >= args.live_max_waves - 1:
                task_ids = remaining_ids
            elif wave_task_limit:
                task_ids = remaining_ids[:wave_task_limit]
            else:
                task_ids = remaining_ids
        else:
            task_ids = scheduler.next_wave(
                remaining.values(), completed_waves=completed_waves
            )
        # Residual bandwidth observations are rank-local, so independent
        # schedulers can otherwise select different task sets and later enter
        # model collectives out of order.  Rank 0 owns the global wave choice;
        # broadcast it before every rank builds its filtered plan.
        scheduled_ids = [list(task_ids) if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(scheduled_ids, src=0, group=control_group)
        task_ids = [int(task_id) for task_id in (scheduled_ids[0] or [])]
        if not task_ids:
            break
        wave_tasks = _tasks_for_ids(tasks, task_ids)
        wave_plan = filter_plan_by_task_ids(plan, task_ids)
        wave_start = time.perf_counter()
        pressure_start = len(foreground_pressure)
        transaction = launch_reshard_plan(
            wave_plan,
            src_bundle,
            dst_bundle,
            service,
            group=reshard_group,
            synchronize_group=False,
            synchronize_device=False,
            release_cache=False,
        )
        # All ranks must enqueue migration P2P before any rank enters active
        # model collectives on another communicator. Keep this ordering barrier
        # on Gloo so it cannot serialize the NCCL migration stream.
        dist.barrier(group=control_group)
        copy_stats = dict(transaction.metadata.get("copy_service", {}))
        overlap_steps = 0
        while True:
            _, step_ms = _decode(
                active_model,
                active_context,
                args,
                token_value=token_counter,
            )
            token_counter += 1
            overlap_steps += 1
            step_tpot_ms = _all_max(step_ms, control_group)
            tpot_ms.append(step_tpot_ms)
            foreground_pressure.append(
                tracker.observe_tpot_pressure(
                    foreground_links,
                    baseline_tpot_ms=foreground_baseline_tpot_ms,
                    observed_tpot_ms=step_tpot_ms,
                    max_fraction=args.live_foreground_pressure_cap,
                )
            )
            if overlap_steps < args.live_min_overlap_steps:
                continue
            globally_ready = _all_ready(transaction.done(), control_group)
            if globally_ready or overlap_steps >= args.live_max_overlap_steps:
                break

        wait_start = time.perf_counter()
        transaction.wait()
        exposed_wait_s = time.perf_counter() - wait_start
        local_transport_s = transaction.transport_elapsed_s or 0.0
        transport_stage_s = transaction.transport_stage_elapsed_s or {}
        copy_stats["stage_elapsed_s"] = transport_stage_s
        commit_start = time.perf_counter()
        transaction.commit()
        torch.cuda.current_stream().synchronize()
        commit_s = time.perf_counter() - commit_start
        wave_s = _all_max(time.perf_counter() - wave_start, control_group)
        exposed_wait_s = _all_max(exposed_wait_s, control_group)
        commit_s = _all_max(commit_s, control_group)
        transport_s = _all_max(local_transport_s, control_group)
        if transport_s <= 0.0:
            transport_s = wave_s
        observed = tracker.observe_wave(wave_tasks, transport_s)
        total_exposed_wait_s += exposed_wait_s + commit_s
        wave_records.append(
            {
                "tasks": len(task_ids),
                "bytes": sum(task.num_bytes for task in wave_tasks),
                "elapsed_s": wave_s,
                "transport_s": transport_s,
                "transport_stage_s": transport_stage_s,
                "overlap_steps": overlap_steps,
                "exposed_wait_s": exposed_wait_s,
                "commit_s": commit_s,
                "observed_gbps": [
                    {"src": src, "dst": dst, "gbps": value}
                    for (src, dst), value in sorted(observed.items())
                ],
                "copy_service": copy_stats,
                "copy_service_rank": dist.get_rank(),
                "foreground_pressure_mean": (
                    statistics.fmean(foreground_pressure[pressure_start:])
                    if len(foreground_pressure) > pressure_start
                    else 0.0
                ),
            }
        )
        for task_id in task_ids:
            remaining.pop(task_id, None)
        completed_waves += 1

    wall_s = _all_max(time.perf_counter() - wall_start, control_group)
    tpot_mean_ms = sum(tpot_ms) / len(tpot_ms) if tpot_ms else 0.0
    tpot_std_ms = statistics.pstdev(tpot_ms) if len(tpot_ms) > 1 else 0.0
    return (
        {
            "waves": len(wave_records),
            "wall_s": wall_s,
            "exposed_wait_s": total_exposed_wait_s,
            "overlap_steps": len(tpot_ms),
            "tpot_mean_ms": tpot_mean_ms,
            "tpot_p95_ms": _percentile(tpot_ms, 0.95),
            "tpot_p99_ms": _percentile(tpot_ms, 0.99),
            "tpot_std_ms": tpot_std_ms,
            "tpot_cv": tpot_std_ms / tpot_mean_ms if tpot_mean_ms else 0.0,
            "foreground_pressure_mean": (
                statistics.fmean(foreground_pressure)
                if foreground_pressure
                else 0.0
            ),
            "foreground_pressure_p95": _percentile(foreground_pressure, 0.95),
            "wave_records": wave_records,
            "execution_mode": (
                "hybrid-fast" if hybrid_fast_path else effective_scheduler_mode
            ),
            "max_wave_tasks": wave_task_limit,
            "hybrid_fast_path": bool(hybrid_fast_path),
            "hybrid_fast_path_reason": dict(hybrid_fast_path_reason or {}),
        },
        token_counter,
    )


@torch.inference_mode()
def _validate_cutover(
    src_model: GPTModel,
    src_context: StaticInferenceContext,
    dst_model: GPTModel,
    dst_context: StaticInferenceContext,
    args,
    control_group,
    token_value: int,
) -> tuple[float, int, float, float]:
    src_logits, src_tpot_ms = _decode(
        src_model, src_context, args, token_value=token_value
    )
    dst_logits, dst_tpot_ms = _decode(
        dst_model, dst_context, args, token_value=token_value
    )
    local_diff = float((src_logits.float() - dst_logits.float()).abs().max().item())
    max_diff = _all_max(local_diff, control_group)
    local_close = torch.allclose(
        src_logits.float(),
        dst_logits.float(),
        atol=args.live_logit_atol,
        rtol=args.live_logit_rtol,
    )
    metrics = None
    if args.live_logit_validation_mode == "bf16-relative":
        metrics = _comparison_metrics(src_logits, dst_logits, control_group)
        local_close = (
            metrics["nrmse"] <= args.live_logit_max_nrmse
            and metrics["cosine"] >= args.live_logit_min_cosine
            and metrics["top1_agreement"] >= args.live_logit_min_top1_agreement
        )
    globally_close = _all_ready(bool(local_close), control_group)
    if not globally_close:
        if metrics is None:
            metrics = _comparison_metrics(src_logits, dst_logits, control_group)
        raise AssertionError(
            f"post-cutover logits mismatch: max_diff={max_diff}, metrics={metrics}"
        )
    return (
        max_diff,
        token_value + 1,
        _all_max(src_tpot_ms, control_group),
        _all_max(dst_tpot_ms, control_group),
    )


def _write_results(args, result: dict[str, object]) -> None:
    if dist.get_rank() != 0 or not args.live_json_output:
        return
    path = Path(args.live_json_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_rank_0(f"Saved live benchmark results to {path}")


def run_live_benchmark() -> None:
    args = get_args()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    _validate_model_preset(args)
    args.live_pressure_rank_phases_tuple = _parse_rank_phases(
        args.live_pressure_rank_phases, world_size
    )
    args.live_pressure_ranks_tuple = args.live_pressure_rank_phases_tuple[0]
    src_tp = args.tensor_model_parallel_size
    dst_tp = args.live_dst_tp
    if src_tp != 2 or dst_tp != 4:
        raise ValueError("this benchmark currently implements the requested TP2<->TP4 cycle")
    if world_size % dst_tp != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must be divisible by TP={dst_tp}")
    if args.num_experts is None or args.num_experts < 2:
        raise ValueError("live MoE benchmark requires --num-experts >= 2")
    ep_size = args.expert_model_parallel_size
    if ep_size < 1 or args.num_experts % ep_size != 0:
        raise ValueError("num-experts must be divisible by expert-model-parallel-size")
    if world_size % (src_tp * ep_size) != 0:
        raise ValueError("WORLD_SIZE must be divisible by source TP*EP")
    if world_size % (dst_tp * ep_size) != 0:
        raise ValueError("WORLD_SIZE must be divisible by destination TP*EP")
    if args.expert_tensor_parallel_size != src_tp:
        raise ValueError("source expert tensor parallel size must match source TP")
    if args.live_switches < 1 or args.live_max_waves < 1:
        raise ValueError("live-switches and live-max-waves must be positive")
    if (
        args.live_max_wave_tasks < 0
        or args.live_expansion_max_wave_tasks < 0
        or args.live_shrink_max_wave_tasks < 0
    ):
        raise ValueError("live wave task limits must be non-negative")
    if args.live_hybrid_min_predicted_gain_pct < 0.0:
        raise ValueError("live hybrid minimum predicted gain must be non-negative")
    if args.live_hybrid_max_remote_cv < 0.0:
        raise ValueError("live hybrid maximum remote CV must be non-negative")
    if not 0.0 <= args.live_hybrid_max_foreground_pressure < 1.0:
        raise ValueError("live hybrid foreground pressure must be in [0, 1)")
    if args.live_adaptive_min_predicted_e2e_gain_pct < 0.0:
        raise ValueError("live adaptive minimum end-to-end gain must be non-negative")
    if args.live_adaptive_residual_max_waves < 1:
        raise ValueError("live adaptive residual wave limit must be positive")
    if args.live_reroute_penalty_us < 0.0:
        raise ValueError("live-reroute-penalty-us must be non-negative")
    if args.live_pack_target_bytes < 0 or args.live_pack_max_item_bytes < 0:
        raise ValueError("live packing byte thresholds must be non-negative")
    if args.live_unpack_chunk_bytes < 0:
        raise ValueError("live unpack chunk bytes must be non-negative")
    if (
        args.live_pack_target_bytes
        and args.live_pack_max_item_bytes > args.live_pack_target_bytes
    ):
        raise ValueError("live-pack-max-item-bytes cannot exceed live-pack-target-bytes")
    if not 1 <= args.live_min_overlap_steps <= args.live_max_overlap_steps:
        raise ValueError("overlap steps must satisfy 1 <= min <= max")
    if args.live_min_transport_gain_pct < 0.0:
        raise ValueError("live minimum transport gain must be non-negative")
    if args.live_max_tpot_regression_pct < 0.0:
        raise ValueError("live maximum TPOT regression must be non-negative")
    if not 0.0 < args.live_guard_ewma <= 1.0:
        raise ValueError("live guard EWMA must be in (0, 1]")
    if args.live_guard_robust_window < 1:
        raise ValueError("live guard robust window must be positive")
    if args.live_candidate_warmup_expansions < 0:
        raise ValueError("live candidate warmup expansions must be non-negative")
    if args.live_guard_min_samples < 1:
        raise ValueError("live guard minimum samples must be positive")
    if not 0.0 <= args.live_guard_min_candidate_win_rate <= 1.0:
        raise ValueError("live guard candidate win rate must be in [0, 1]")
    if args.live_guard_max_transport_regression_pct < 0.0:
        raise ValueError("live guard maximum transport regression must be non-negative")
    if args.live_replan_min_change_pct < 0.0:
        raise ValueError("live replan minimum change must be non-negative")
    if args.live_replan_max_stale_expansions < 0:
        raise ValueError("live replan maximum staleness must be non-negative")
    if args.live_replan_noop_cooldown_expansions < 0:
        raise ValueError("live no-op replan cooldown must be non-negative")
    if args.live_amortization_horizon_expansions < 0:
        raise ValueError("live amortization horizon must be non-negative")
    if not 0.0 <= args.live_foreground_pressure_cap < 1.0:
        raise ValueError("live foreground pressure cap must be in [0, 1)")
    if not 0.0 < args.live_foreground_baseline_ewma <= 1.0:
        raise ValueError("live foreground baseline EWMA must be in (0, 1]")
    if args.live_guard_reevaluate_expansions < 0:
        raise ValueError("live guard reevaluation interval must be non-negative")
    if args.live_guard_hysteresis_pct < 0.0:
        raise ValueError("live guard hysteresis must be non-negative")
    if args.live_hotspot_phase_expansions < 1:
        raise ValueError("live hotspot phase expansions must be positive")
    if args.live_hotspot_pressure_bytes < 0:
        raise ValueError("live hotspot pressure bytes must be non-negative")
    if args.live_hotspot_pressure_iters < 1:
        raise ValueError("live hotspot pressure iterations must be positive")
    if (
        args.live_hotspot_pressure_bytes
        and ep_size <= 1
        and not args.live_pressure_ranks_tuple
    ):
        raise ValueError(
            "EP=1 hotspot pressure requires explicit live pressure rank phases"
        )
    if len(args.live_active_expert_phases_tuple) > 1 and args.live_router_mode != "fixed-hot":
        raise ValueError("active expert phases currently require live-router-mode=fixed-hot")
    worst_case_tokens = (
        args.live_prompt_tokens
        + 1
        + args.live_switches * (args.live_max_waves * args.live_max_overlap_steps + 2)
    )
    if worst_case_tokens >= args.max_position_embeddings:
        raise ValueError("max-position-embeddings is too small for the requested live run")

    expansion_max_wave_tasks = (
        args.live_expansion_max_wave_tasks or args.live_max_wave_tasks
    )
    shrink_max_wave_tasks = args.live_shrink_max_wave_tasks or args.live_max_wave_tasks
    directional_max_wave_tasks = {
        "2->4": expansion_max_wave_tasks,
        "4->2": shrink_max_wave_tasks,
    }

    print_rank_0("Building TP2 active MoE model...")
    src_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=False,
        ),
        wrap_with_ddp=False,
    )[0].cuda()
    src_model.eval()
    loaded_checkpoint_iteration = None
    if args.live_model_preset == "deepseek-v2-lite":
        loaded_checkpoint_iteration, _ = load_checkpoint([src_model], None, None)
        print_rank_0(
            "Loaded DeepSeek-V2-Lite checkpoint before constructing the TP4 standby "
            f"layout (iteration={loaded_checkpoint_iteration})."
        )

    print_rank_0("Building TP4 standby MoE model...")
    dst_pg = build_inference_pg_collection(
        world_size,
        tp_size=dst_tp,
        pp_size=1,
        ep_size=ep_size,
        expt_tp_size=dst_tp,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping,
    )
    dst_config = core_transformer_config_from_args(args)
    dst_config.tensor_model_parallel_size = dst_tp
    dst_config.expert_tensor_parallel_size = dst_tp
    dst_config.expert_model_parallel_size = ep_size
    dst_model = get_training_model(
        lambda pre_process, post_process, **kwargs: model_provider(
            pre_process=pre_process,
            post_process=post_process,
            parallel_output=False,
            pg_collection=dst_pg,
            config=dst_config,
        ),
        wrap_with_ddp=False,
    )[0].cuda()
    dst_model.eval()

    all_ranks = list(range(world_size))
    reshard_group = dist.new_group(ranks=all_ranks, backend="nccl")
    control_group = dist.new_group(ranks=all_ranks, backend="gloo")
    parallel_groups = (
        observations.observe_parallel_groups(
            torch, dist, {src_tp: src_model.pg_collection, dst_tp: dst_model.pg_collection},
            control_group,
        )
        if world_size == 16 else None
    )
    source_reroute_enabled = bool(
        args.live_scheduler_mode != "baseline"
        and not args.live_disable_source_reroute
    )
    planner_group = (
        dist.new_group(ranks=all_ranks, backend="gloo")
        if args.live_online_replan and source_reroute_enabled
        else None
    )
    capacity = _load_bandwidth_matrix(args, world_size)
    policy = BandwidthAwareRefitPolicy(
        bandwidth_gbps=capacity,
        reference_bandwidth_gbps=capacity,
        reroute_penalty_us=args.live_reroute_penalty_us,
        reroute_min_gain_pct=args.live_reroute_min_gain_pct,
        reroute_min_contention_gain_pct=args.live_reroute_min_contention_gain_pct,
        reroute_min_global_gain_pct=args.live_reroute_min_global_gain_pct,
        reroute_min_bytes=args.live_reroute_min_bytes,
        prefer_local_source=not args.live_allow_nonlocal_reroute,
        force=args.live_force_bandwidth_routing,
        p2p_order=args.live_p2p_order,
        pack_target_bytes=args.live_pack_target_bytes,
        pack_max_item_bytes=args.live_pack_max_item_bytes,
        persistent_pack_buffers=args.live_persistent_pack_buffers,
        pack_rerouted_only=args.live_pack_rerouted_only,
    )

    print_rank_0("Synchronizing initial TP2 weights into TP4...")
    swap_model_weights(
        src_model,
        dst_model,
        refit_method=NCCLCopyService(
            group=reshard_group, p2p_order=args.live_p2p_order
        ),
        group=reshard_group,
        bandwidth_policy=policy,
        release_cache=False,
    )

    model_parallel_cuda_manual_seed(args.seed + 101)
    src_context = StaticInferenceContext(
        max_batch_size=args.micro_batch_size,
        max_sequence_length=args.max_position_embeddings,
    )
    dst_context = StaticInferenceContext(
        max_batch_size=args.micro_batch_size,
        max_sequence_length=args.max_position_embeddings,
    )
    src_layer_outputs = {}
    dst_layer_outputs = {}
    src_hooks = []
    dst_hooks = []
    if args.live_diagnose_equivalence:
        src_layer_outputs, src_hooks = _install_layer_capture(src_model)
        dst_layer_outputs, dst_hooks = _install_layer_capture(dst_model)
    src_prefill_logits = _prefill(src_model, src_context, args)
    dst_prefill_logits = _prefill(dst_model, dst_context, args)
    _remove_hooks(src_hooks)
    _remove_hooks(dst_hooks)
    if args.live_diagnose_equivalence:
        _print_equivalence_diagnostics(
            reference_logits=src_prefill_logits,
            candidate_logits=dst_prefill_logits,
            reference_layers=src_layer_outputs,
            candidate_layers=dst_layer_outputs,
            control_group=control_group,
        )
    src_kv = StaticKVCacheModule(src_context, src_model.pg_collection)
    dst_kv = StaticKVCacheModule(dst_context, dst_model.pg_collection)
    src_bundle = LiveStateBundle(src_model, src_kv)
    dst_bundle = LiveStateBundle(dst_model, dst_kv)

    calibration_token = args.live_prompt_tokens + 31
    initial_validation_diff, token_counter, src_clean_tpot_ms, dst_clean_tpot_ms = (
        _validate_cutover(
            src_model,
            src_context,
            dst_model,
            dst_context,
            args,
            control_group,
            calibration_token,
        )
    )
    initial_active_experts = args.live_active_experts_tuple
    initial_pressure_ranks = args.live_pressure_ranks_tuple
    foreground_tpot_ms = {
        (src_tp, initial_active_experts, initial_pressure_ranks): src_clean_tpot_ms,
        (dst_tp, initial_active_experts, initial_pressure_ranks): dst_clean_tpot_ms,
    }
    foreground_link_history = []

    # The baseline is deliberately bandwidth-unaware. Residual runs with one
    # wave isolate source-path selection; additional waves add online feedback.
    planner_kwargs = policy.planner_kwargs() if source_reroute_enabled else {}
    planner_kwargs["source_exclude_local"] = args.live_emulate_noncollocated_sources
    print_rank_0("Building cached TP2->TP4 live-state plan...")
    plan_2_to_4 = build_centralized_reshard_plan(
        src_bundle,
        dst_bundle,
        num_experts=args.num_experts,
        group=reshard_group,
        **planner_kwargs,
    )
    print_rank_0("Building cached TP4->TP2 live-state plan...")
    reverse_planner_kwargs = planner_kwargs if args.live_allow_aware_shrink else {}
    plan_4_to_2 = build_centralized_reshard_plan(
        dst_bundle,
        src_bundle,
        num_experts=args.num_experts,
        group=reshard_group,
        **reverse_planner_kwargs,
    )

    trackers = {
        tp: ResidualBandwidthTracker(
            capacity,
            ewma_alpha=args.live_bandwidth_ewma,
            floor_fraction=args.live_bandwidth_floor_fraction,
        )
        for tp in (src_tp, dst_tp)
    }
    schedulers = {
        tp: ResidualBandwidthWaveScheduler(
            trackers[tp],
            max_waves=args.live_max_waves,
            max_wave_bytes=args.live_max_wave_bytes,
            max_wave_tasks=(
                expansion_max_wave_tasks if tp == src_tp else shrink_max_wave_tasks
            ),
            hotness_weight=args.live_hotness_weight,
        )
        for tp in (src_tp, dst_tp)
    }
    plan_controller = (
        OnlineResidualPlanController(
            trackers[src_tp],
            min_change_pct=args.live_replan_min_change_pct,
            max_stale_expansions=args.live_replan_max_stale_expansions,
            no_op_cooldown_expansions=(
                args.live_replan_noop_cooldown_expansions
            ),
        )
        if args.live_online_replan and source_reroute_enabled
        else None
    )
    planner_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-reshard-planner")
        if plan_controller is not None
        else None
    )
    pending_replan: Future | None = None
    pending_replan_trigger = -1

    def build_online_plan(online_policy):
        planning_start = time.perf_counter()
        online_planner_kwargs = online_policy.planner_kwargs()
        online_planner_kwargs["source_exclude_local"] = (
            args.live_emulate_noncollocated_sources
        )
        online_plan = build_centralized_reshard_plan(
            src_bundle,
            dst_bundle,
            num_experts=args.num_experts,
            group=planner_group,
            **online_planner_kwargs,
        )
        planning_s = _all_max(
            time.perf_counter() - planning_start, planner_group
        )
        return (
            online_plan,
            getattr(online_plan, "baseline_plan", online_plan),
            planning_s,
        )

    def schedule_online_replan(trigger_expansion: int) -> bool:
        nonlocal pending_replan, pending_replan_trigger
        if (
            plan_controller is None
            or planner_executor is None
            or pending_replan is not None
            or not plan_controller.should_replan(trigger_expansion)
        ):
            return False
        online_policy = replace(
            policy,
            bandwidth_gbps=dict(trackers[src_tp].residual_matrix()),
            reference_bandwidth_gbps=capacity,
        )
        pending_replan = planner_executor.submit(build_online_plan, online_policy)
        pending_replan_trigger = int(trigger_expansion)
        return True
    pack_target_bytes = (
        0 if args.live_scheduler_mode == "baseline" else args.live_pack_target_bytes
    )
    pack_max_item_bytes = (
        0 if args.live_scheduler_mode == "baseline" else args.live_pack_max_item_bytes
    )
    ensure_wave_participation = bool(
        args.live_scheduler_mode != "baseline" and args.live_max_waves > 1
    )
    services = {
        "2->4-candidate": NCCLCopyService(
            group=reshard_group,
            p2p_order=args.live_p2p_order,
            pack_target_bytes=pack_target_bytes,
            pack_max_item_bytes=pack_max_item_bytes,
            persistent_pack_buffers=args.live_persistent_pack_buffers,
            pack_rerouted_only=args.live_pack_rerouted_only,
            unpack_chunk_bytes=args.live_unpack_chunk_bytes,
            ensure_all_ranks_participate=ensure_wave_participation,
        ),
        "2->4-baseline": NCCLCopyService(
            group=reshard_group,
            p2p_order=args.live_p2p_order,
            unpack_chunk_bytes=args.live_unpack_chunk_bytes,
            ensure_all_ranks_participate=(
                ensure_wave_participation and not args.live_adaptive_hybrid
            ),
        ),
        "4->2": NCCLCopyService(
            group=reshard_group,
            p2p_order=args.live_p2p_order,
            pack_target_bytes=(pack_target_bytes if args.live_allow_aware_shrink else 0),
            pack_max_item_bytes=(
                pack_max_item_bytes if args.live_allow_aware_shrink else 0
            ),
            persistent_pack_buffers=(
                args.live_persistent_pack_buffers and args.live_allow_aware_shrink
            ),
            pack_rerouted_only=(
                args.live_pack_rerouted_only and args.live_allow_aware_shrink
            ),
            unpack_chunk_bytes=args.live_unpack_chunk_bytes,
            ensure_all_ranks_participate=(
                ensure_wave_participation
                and (not args.live_adaptive_hybrid or args.live_allow_aware_shrink)
            ),
        ),
    }
    baseline_plan_2_to_4 = getattr(plan_2_to_4, "baseline_plan", plan_2_to_4)

    def reroute_count(plan) -> int:
        route_stats = getattr(plan, "source_route_stats", {})
        for key in ("accepted", "changed_from_default"):
            if key in route_stats:
                return int(route_stats[key])
        rerouted_task_ids = getattr(plan, "rerouted_task_ids", None)
        return len(rerouted_task_ids) if rerouted_task_ids is not None else 0

    candidate_route_available = reroute_count(plan_2_to_4) > 0
    packing_candidate_available = bool(
        pack_target_bytes > 0 and not args.live_pack_rerouted_only
    )
    candidate_plan_available = (
        candidate_route_available or packing_candidate_available
    )
    guard = None
    if args.live_online_migration_first_guard:
        if args.live_scheduler_mode == "baseline":
            raise ValueError("the migration-first guard requires residual scheduler mode")
        guard = MigrationFirstGuard(
            min_transport_gain_pct=args.live_min_transport_gain_pct,
            max_tpot_regression_pct=args.live_max_tpot_regression_pct,
            ewma_alpha=args.live_guard_ewma,
            robust_window=args.live_guard_robust_window,
            candidate_warmup_samples=args.live_candidate_warmup_expansions,
            min_samples=args.live_guard_min_samples,
            min_candidate_win_rate=args.live_guard_min_candidate_win_rate,
            max_transport_regression_pct=(
                args.live_guard_max_transport_regression_pct
            ),
            reevaluate_interval=args.live_guard_reevaluate_expansions,
            decision_hysteresis_pct=args.live_guard_hysteresis_pct,
        )

    active_tp = 2
    expansion_index = 0
    guard_phase_key = (initial_active_experts, initial_pressure_ranks)
    records = []
    observation_inputs = []
    for switch_index in range(args.live_switches):
        active_experts = _active_experts_for_switch(args, switch_index)
        pressure_ranks = _pressure_ranks_for_switch(args, switch_index)
        phase_changed = (
            active_experts != args.live_active_experts_tuple
            or pressure_ranks != args.live_pressure_ranks_tuple
        )
        args.live_active_experts_tuple = active_experts
        args.live_pressure_ranks_tuple = pressure_ranks
        _set_active_experts((src_model, dst_model), active_experts)
        phase_calibration_diff = None
        if (
            (src_tp, active_experts, pressure_ranks) not in foreground_tpot_ms
            or (dst_tp, active_experts, pressure_ranks) not in foreground_tpot_ms
        ):
            (
                phase_calibration_diff,
                token_counter,
                phase_src_tpot_ms,
                phase_dst_tpot_ms,
            ) = _validate_cutover(
                src_model,
                src_context,
                dst_model,
                dst_context,
                args,
                control_group,
                token_counter,
            )
            foreground_tpot_ms[(src_tp, active_experts, pressure_ranks)] = (
                phase_src_tpot_ms
            )
            foreground_tpot_ms[(dst_tp, active_experts, pressure_ranks)] = (
                phase_dst_tpot_ms
            )
        switch_start = time.perf_counter()
        replan_s = 0.0
        replan_wait_s = 0.0
        replanned = False
        replan_scheduled = False
        replan_trigger_expansion = -1
        replan_useful = None
        candidate_fallback_reason = None
        adaptive_decision = None
        wave_scheduler_mode = None
        if active_tp == 2:
            direction = "2->4"
            source_model, target_model = src_model, dst_model
            source_context, target_context = src_context, dst_context
            source_bundle, target_bundle = src_bundle, dst_bundle
            current_guard_phase_key = (active_experts, pressure_ranks)
            if guard is not None and current_guard_phase_key != guard_phase_key:
                guard.reset_for_phase("foreground_phase_changed")
            guard_phase_key = current_guard_phase_key
            requested_plan_variant = (
                guard.select_variant()
                if guard is not None
                else (
                    "baseline"
                    if args.live_scheduler_mode == "baseline"
                    else "candidate"
                )
            )
            guard_snapshot = guard.snapshot() if guard is not None else None
            freeze_calibration_plan = bool(
                guard_snapshot is not None
                and not guard_snapshot["calibrated"]
                and guard_snapshot["attempts"]["candidate"] > 0
            )
            plan_variant = requested_plan_variant
            if requested_plan_variant == "candidate" and not freeze_calibration_plan:
                if pending_replan is None:
                    replan_scheduled = schedule_online_replan(expansion_index)
                if pending_replan is not None:
                    replan_wait_start = time.perf_counter()
                    (
                        plan_2_to_4,
                        baseline_plan_2_to_4,
                        replan_s,
                    ) = pending_replan.result()
                    replan_wait_s = _all_max(
                        time.perf_counter() - replan_wait_start, control_group
                    )
                    replan_trigger_expansion = pending_replan_trigger
                    pending_replan = None
                    pending_replan_trigger = -1
                    replan_useful = reroute_count(plan_2_to_4) > 0
                    plan_controller.mark_replanned(
                        expansion_index, useful=replan_useful
                    )
                    replanned = True
                    candidate_route_available = replan_useful
                    candidate_plan_available = (
                        candidate_route_available
                        or packing_candidate_available
                    )
                    if not candidate_plan_available:
                        candidate_fallback_reason = "online_replan_no_reroutes"
                        if guard is not None:
                            guard.force_baseline(candidate_fallback_reason)
            if requested_plan_variant == "candidate" and not candidate_plan_available:
                plan_variant = "baseline"
                candidate_fallback_reason = (
                    candidate_fallback_reason or "candidate_plan_unavailable"
                )
                if guard is not None:
                    guard.force_baseline(candidate_fallback_reason)
            full_plan = (
                baseline_plan_2_to_4
                if plan_variant == "baseline"
                else plan_2_to_4
            )
            candidate_full_plan = plan_2_to_4
            baseline_full_plan = baseline_plan_2_to_4
            service = services[f"2->4-{plan_variant}"]
            next_tp = 4
            source_tp = 2
        else:
            direction = "4->2"
            source_model, target_model = dst_model, src_model
            source_context, target_context = dst_context, src_context
            source_bundle, target_bundle = dst_bundle, src_bundle
            full_plan = plan_4_to_2
            candidate_full_plan = plan_4_to_2
            baseline_full_plan = plan_4_to_2
            plan_variant = (
                "candidate"
                if args.live_scheduler_mode != "baseline"
                and args.live_allow_aware_shrink
                else "baseline"
            )
            service = services["4->2"]
            next_tp = 2
            source_tp = 4
            requested_plan_variant = plan_variant

        tracker = trackers[source_tp]
        scheduler = schedulers[source_tp]
        foreground_key = (source_tp, active_experts, pressure_ranks)
        foreground_baseline_used_ms = foreground_tpot_ms[foreground_key]
        active_foreground_links = _collect_active_collective_links(
            source_model,
            control_group,
            active_experts=active_experts,
            num_experts=args.num_experts,
            pressure_ranks=pressure_ranks,
        )
        foreground_link_history.append(
            {
                "switch_index": switch_index,
                "source_tp": source_tp,
                "active_experts": list(active_experts),
                "pressure_ranks": list(pressure_ranks),
                "links": [
                    {"src": src, "dst": dst}
                    for src, dst in sorted(active_foreground_links)
                ],
            }
        )

        snapshot_end = source_context.sequence_len_offset
        if target_context.sequence_len_offset != snapshot_end:
            raise RuntimeError("source and standby KV offsets diverged before migration")
        if args.live_adaptive_hybrid:
            adaptive_start = time.perf_counter()
            baseline_preview = restrict_plan_sequence(
                baseline_full_plan,
                start=0,
                end=snapshot_end,
                include_non_kv=True,
            )
            candidate_preview = restrict_plan_sequence(
                candidate_full_plan,
                start=0,
                end=snapshot_end,
                include_non_kv=True,
            )
            route_stats = getattr(candidate_full_plan, "source_route_stats", {}) or {}
            adaptive_decision = select_adaptive_hybrid_policy(
                direction=direction,
                requested_scheduler_mode=args.live_scheduler_mode,
                candidate_route_available=(
                    candidate_route_available and requested_plan_variant == "candidate"
                ),
                projected_transport_gain_pct=float(
                    route_stats.get("projected_global_gain_pct", 0.0)
                ),
                baseline_task_count=_global_plan_task_count(
                    baseline_preview, control_group
                ),
                candidate_task_count=_global_plan_task_count(
                    candidate_preview, control_group
                ),
                max_wave_tasks=directional_max_wave_tasks[direction],
                min_end_to_end_gain_pct=(
                    args.live_adaptive_min_predicted_e2e_gain_pct
                ),
                residual_max_waves=args.live_adaptive_residual_max_waves,
                foreground_pressure_fraction=_foreground_pressure_fraction(tracker),
                max_foreground_pressure=args.live_hybrid_max_foreground_pressure,
            )
            plan_variant = adaptive_decision.plan_variant
            wave_scheduler_mode = adaptive_decision.scheduler_mode
            if direction == "2->4":
                full_plan = (
                    candidate_full_plan
                    if plan_variant == "candidate"
                    else baseline_full_plan
                )
                service = services[f"2->4-{plan_variant}"]
                if plan_variant == "baseline" and requested_plan_variant == "candidate":
                    candidate_fallback_reason = adaptive_decision.reason
            else:
                full_plan = baseline_full_plan
                service = services["4->2"]
            adaptive_decision_s = _all_max(
                time.perf_counter() - adaptive_start, control_group
            )
        else:
            adaptive_decision_s = 0.0
        flying_shared_state = args.live_method_variant == "flying-serving-proxy"
        kv_only_migration = args.live_method_variant == "llumnix-proxy"
        if not flying_shared_state:
            _poison_kv(target_context, snapshot_end)
        base_plan = restrict_plan_sequence(
            full_plan,
            start=0,
            end=0 if flying_shared_state else snapshot_end,
            include_non_kv=not (flying_shared_state or kv_only_migration),
        )
        hybrid_decision = _hybrid_fast_path_decision(
            full_plan,
            tracker,
            args,
            direction=direction,
            plan_variant=plan_variant,
            candidate_route_available=candidate_route_available,
        )
        print_rank_0(
            f"Switch {switch_index + 1}/{args.live_switches} {direction}: "
            f"stable_kv_tokens={snapshot_end} variant={plan_variant} "
            f"wave_scheduler={wave_scheduler_mode or args.live_scheduler_mode} "
            f"adaptive_reason={adaptive_decision.reason if adaptive_decision else 'disabled'} "
            f"hybrid_fast={hybrid_decision['enabled']} "
            f"reason={hybrid_decision['reason']}"
        )
        base_metrics, token_counter = _run_async_waves(
            plan=base_plan,
            src_bundle=source_bundle,
            dst_bundle=target_bundle,
            active_model=source_model,
            active_context=source_context,
            service=service,
            tracker=tracker,
            scheduler=scheduler,
            reshard_group=reshard_group,
            control_group=control_group,
            args=args,
            token_counter=token_counter,
            foreground_baseline_tpot_ms=foreground_baseline_used_ms,
            foreground_links=active_foreground_links,
            wave_scheduler_mode=wave_scheduler_mode,
            max_wave_tasks=directional_max_wave_tasks[direction],
            hybrid_fast_path=bool(hybrid_decision["enabled"]),
            hybrid_fast_path_reason=hybrid_decision,
        )

        delta_end = source_context.sequence_len_offset
        delta_plan = restrict_plan_sequence(
            full_plan,
            start=snapshot_end,
            end=delta_end,
            include_non_kv=False,
        )
        cutover_start = time.perf_counter()
        with torch.inference_mode():
            # The post-overlap KV delta contains one operation per KV shard and
            # can still be several thousand P2P items on a reverse TP switch.
            # Keep it under the same NCCL FIFO bound as the main migration.
            delta_tasks = collect_transfer_tasks(
                delta_plan, target_bundle, group=reshard_group
            )
            delta_chunk_size = directional_max_wave_tasks[direction] or max(
                len(delta_tasks), 1
            )
            for delta_start in range(0, len(delta_tasks), delta_chunk_size):
                delta_ids = [
                    task.task_id
                    for task in delta_tasks[delta_start : delta_start + delta_chunk_size]
                ]
                delta_chunk_plan = filter_plan_by_task_ids(delta_plan, delta_ids)
                delta_transaction = launch_reshard_plan(
                    delta_chunk_plan,
                    source_bundle,
                    target_bundle,
                    service,
                    group=reshard_group,
                    synchronize_group=False,
                    synchronize_device=False,
                    release_cache=False,
                )
                delta_transaction.wait().commit()
            torch.cuda.current_stream().synchronize()
        target_context.sequence_len_offset = delta_end
        target_context.enable_decode_mode()
        delta_s = _all_max(time.perf_counter() - cutover_start, control_group)

        if guard is not None and direction == "2->4":
            remaining_expansions = (args.live_switches - switch_index + 1) // 2
            configured_horizon = args.live_amortization_horizon_expansions
            amortization_horizon = (
                min(configured_horizon, remaining_expansions)
                if configured_horizon > 0
                else remaining_expansions
            )
            guard.observe(
                plan_variant,
                transport_s=sum(
                    wave["transport_s"] for wave in base_metrics["wave_records"]
                ),
                tpot_ms=base_metrics[
                    f"tpot_{args.live_guard_tpot_metric}_ms"
                ],
                scheduler_exposed_s=(
                    replan_wait_s if plan_variant == "candidate" else 0.0
                ),
                amortization_horizon=max(amortization_horizon, 1),
            )
        if (
            direction == "2->4"
            and plan_controller is not None
            and pending_replan is None
            and switch_index + 2 < args.live_switches
        ):
            guard_after_observe = guard.snapshot() if guard is not None else None
            ready_to_replan = bool(
                guard_after_observe is None or guard_after_observe["calibrated"]
                or not candidate_plan_available
            )
            if ready_to_replan:
                scheduled_now = schedule_online_replan(expansion_index + 1)
                replan_scheduled = replan_scheduled or scheduled_now
                if scheduled_now:
                    replan_trigger_expansion = pending_replan_trigger

        pointer_start = time.perf_counter()
        active_tp = 2 if args.live_repeat_forward and direction == "2->4" else next_tp
        pointer_switch_s = _all_max(time.perf_counter() - pointer_start, control_group)
        (
            validation_diff,
            token_counter,
            source_clean_tpot_ms,
            target_clean_tpot_ms,
        ) = _validate_cutover(
            source_model,
            source_context,
            target_model,
            target_context,
            args,
            control_group,
            token_counter,
        )
        baseline_alpha = args.live_foreground_baseline_ewma
        foreground_tpot_ms[(source_tp, active_experts, pressure_ranks)] = (
            baseline_alpha * source_clean_tpot_ms
            + (1.0 - baseline_alpha)
            * foreground_tpot_ms[(source_tp, active_experts, pressure_ranks)]
        )
        foreground_tpot_ms[(next_tp, active_experts, pressure_ranks)] = (
            baseline_alpha * target_clean_tpot_ms
            + (1.0 - baseline_alpha)
            * foreground_tpot_ms[(next_tp, active_experts, pressure_ranks)]
        )
        if direction == "2->4":
            expansion_index += 1
        switch_wall_s = _all_max(time.perf_counter() - switch_start, control_group)
        record = {
            "index": switch_index,
            "direction": direction,
            "active_experts": list(active_experts),
            "pressure_ranks": list(pressure_ranks),
            "hotspot_phase_changed": phase_changed,
            "phase_calibration_max_diff": phase_calibration_diff,
            "plan_variant": plan_variant,
            "requested_plan_variant": requested_plan_variant,
            "candidate_fallback_reason": candidate_fallback_reason,
            "snapshot_tokens": snapshot_end,
            "delta_tokens": delta_end - snapshot_end,
            "delta_and_commit_s": delta_s,
            "pointer_switch_s": pointer_switch_s,
            "validation_max_diff": validation_diff,
            "replanned": replanned,
            "online_replan_s": replan_s,
            "online_replan_wait_s": replan_wait_s,
            "online_replan_scheduled": replan_scheduled,
            "online_replan_trigger_expansion": replan_trigger_expansion,
            "online_replan_useful": replan_useful,
            "hybrid_fast_path": hybrid_decision,
            "adaptive_hybrid": (
                adaptive_decision.snapshot() if adaptive_decision is not None else None
            ),
            "adaptive_decision_s": adaptive_decision_s,
            "switch_wall_s": switch_wall_s,
            "foreground_baseline_used_tpot_ms": foreground_baseline_used_ms,
            "foreground_baseline_updated_tpot_ms": foreground_tpot_ms[
                (source_tp, active_experts, pressure_ranks)
            ],
            "source_route_stats": dict(
                getattr(full_plan, "source_route_stats", {})
            ),
            "candidate_route_available": candidate_route_available,
            "packing_candidate_available": packing_candidate_available,
            "base": base_metrics,
            "residual_bandwidth": tracker.snapshot(),
            "migration_first_guard": guard.snapshot() if guard is not None else None,
            "dual_objective_guard": guard.snapshot() if guard is not None else None,
            "online_plan_controller": (
                plan_controller.snapshot() if plan_controller is not None else None
            ),
        }
        records.append(record)
        if parallel_groups is not None:
            # Save existing immutable plan references only, after switch_wall_s.
            # All metadata scans and communication happen after the final switch.
            observation_inputs.append({
                "record": record,
                "default_plan": getattr(baseline_full_plan, "baseline_plan", baseline_full_plan),
                "cached_plan": candidate_full_plan, "adopted_plan": full_plan,
                "base_plan": base_plan, "delta_plan": delta_plan, "bundle": target_bundle,
            })
        print_rank_0(
            f"Completed {direction}: waves={base_metrics['waves']} "
            f"overlap_steps={base_metrics['overlap_steps']} "
            f"base_wall={base_metrics['wall_s']:.6f}s "
            f"cutover_delta={delta_s:.6f}s max_diff={validation_diff:.6g}"
        )

    pending_replan_discarded = pending_replan is not None
    if planner_executor is not None:
        planner_executor.shutdown(wait=True, cancel_futures=False)

    local_copy_service_stats = {
        "rank": rank,
        "switches": [
            {
                "index": record["index"],
                "direction": record["direction"],
                "plan_variant": record["plan_variant"],
                "waves": [
                    dict(wave.get("copy_service", {}))
                    for wave in record["base"]["wave_records"]
                ],
            }
            for record in records
        ],
    }
    copy_service_by_rank = [None] * world_size
    dist.all_gather_object(
        copy_service_by_rank, local_copy_service_stats, group=control_group
    )

    if parallel_groups is not None:
        local_observations = [
            observations.local_switch_observation(
                **inputs, rank=rank, restrict_sequence=restrict_plan_sequence,
                enabled=source_reroute_enabled, allow_aware_shrink=args.live_allow_aware_shrink,
                threshold=args.live_reroute_min_global_gain_pct,
            )
            for inputs in observation_inputs
        ]
        observations_by_rank = [None] * world_size
        dist.all_gather_object(observations_by_rank, local_observations, group=control_group)
        rank_hosts = {row["rank"]: row["hostname"] for row in parallel_groups}
        for index, record in enumerate(records):
            record["plan_observation"] = observations.merge_switch_observations(
                [rows[index] for rows in observations_by_rank], rank_hosts,
            )

    result = {
        "benchmark": "live-moe-tp2-tp4",
        "method_variant": args.live_method_variant,
        "proxy_semantics": {
            "flying-serving-proxy": "resident weights and shared-KV cutover lower bound",
            "anchortp-proxy": "bandwidth-aware full-state minimal-source migration",
            "llumnix-proxy": "request/KV-only live migration with resident target weights",
        }.get(args.live_method_variant, "full weight and KV migration"),
        "model_preset": args.live_model_preset,
        "checkpoint_loaded": loaded_checkpoint_iteration is not None,
        "checkpoint_iteration": loaded_checkpoint_iteration,
        "world_size": world_size,
        "parallel_groups": parallel_groups,
        "num_experts": args.num_experts,
        "expert_parallel_size": ep_size,
        "router_mode": args.live_router_mode,
        "logit_validation_mode": args.live_logit_validation_mode,
        "logit_max_nrmse": args.live_logit_max_nrmse,
        "logit_min_cosine": args.live_logit_min_cosine,
        "logit_min_top1_agreement": args.live_logit_min_top1_agreement,
        "active_expert_phases": [
            list(phase) for phase in args.live_active_expert_phases_tuple
        ],
        "hotspot_phase_expansions": args.live_hotspot_phase_expansions,
        "hotspot_pressure_bytes": args.live_hotspot_pressure_bytes,
        "hotspot_pressure_iters": args.live_hotspot_pressure_iters,
        "pressure_rank_phases": [
            list(phase) for phase in args.live_pressure_rank_phases_tuple
        ],
        "scheduler_mode": args.live_scheduler_mode,
        "hybrid_fast_path": args.live_hybrid_fast_path,
        "hybrid_min_predicted_gain_pct": args.live_hybrid_min_predicted_gain_pct,
        "hybrid_max_remote_cv": args.live_hybrid_max_remote_cv,
        "hybrid_max_foreground_pressure": args.live_hybrid_max_foreground_pressure,
        "adaptive_hybrid": args.live_adaptive_hybrid,
        "adaptive_min_predicted_e2e_gain_pct": (
            args.live_adaptive_min_predicted_e2e_gain_pct
        ),
        "adaptive_residual_max_waves": args.live_adaptive_residual_max_waves,
        "bandwidth_aware_plan": source_reroute_enabled,
        "source_reroute_enabled": source_reroute_enabled,
        "emulate_noncollocated_sources": args.live_emulate_noncollocated_sources,
        "bandwidth_aware_shrink": (
            args.live_scheduler_mode != "baseline" and args.live_allow_aware_shrink
        ),
        "source_route_stats": dict(
            getattr(plan_2_to_4, "source_route_stats", {})
        ),
        "initial_validation_max_diff": initial_validation_diff,
        "online_replan": plan_controller is not None,
        "online_plan_controller": (
            plan_controller.snapshot() if plan_controller is not None else None
        ),
        "pending_replan_discarded": pending_replan_discarded,
        "foreground_tpot_ms": [
            {
                "tp": tp,
                "active_experts": list(active_experts),
                "pressure_ranks": list(pressure_ranks),
                "tpot_ms": tpot_ms,
            }
            for (tp, active_experts, pressure_ranks), tpot_ms in sorted(
                foreground_tpot_ms.items(),
                key=lambda item: (item[0][0], item[0][1], item[0][2]),
            )
        ],
        "foreground_link_history": foreground_link_history,
        "max_waves": args.live_max_waves,
        "max_wave_tasks": args.live_max_wave_tasks,
        "expansion_max_wave_tasks": expansion_max_wave_tasks,
        "shrink_max_wave_tasks": shrink_max_wave_tasks,
        "pack_target_bytes": pack_target_bytes,
        "pack_max_item_bytes": pack_max_item_bytes,
        "unpack_chunk_bytes": args.live_unpack_chunk_bytes,
        "p2p_order": args.live_p2p_order,
        "persistent_pack_buffers": args.live_persistent_pack_buffers,
        "pack_rerouted_only": args.live_pack_rerouted_only,
        "candidate_route_available": candidate_route_available,
        "packing_candidate_available": packing_candidate_available,
        "online_migration_first_guard": (
            guard.snapshot() if guard is not None else None
        ),
        "online_dual_objective_guard": (
            guard.snapshot() if guard is not None else None
        ),
        "guard_tpot_metric": args.live_guard_tpot_metric,
        "switches": records,
        "copy_service_by_rank": copy_service_by_rank,
        "final_active_tp": active_tp,
        "final_residual_bandwidth": trackers[src_tp].snapshot(),
        "final_residual_bandwidth_by_tp": {
            str(tp): tracker.snapshot() for tp, tracker in trackers.items()
        },
    }
    _write_results(args, result)
    if rank == 0:
        print(json.dumps(result, indent=2))


def main() -> None:
    try:
        parse_and_validate_args(
            extra_args_provider=add_live_args,
            args_defaults={
                "tokenizer_type": "NullTokenizer",
                "no_load_optim": True,
                "no_load_rng": True,
                "no_save_optim": True,
                "no_save_rng": True,
                "transformer_impl": "local",
            },
            ignore_unknown_args=False,
        )
        # Keep the inference benchmark compatible with the Apex/TE-free local
        # Torch backend and with Megatron's ETP > 1 constraints.
        args = get_args()
        args.apply_rope_fusion = False
        args.add_bias_linear = False
        args.add_qkv_bias = False
        args.bias_gelu_fusion = False
        args.gradient_accumulation_fusion = False
        args.sequence_parallel = False
        initialize_megatron()
        args = get_args()
        args.live_active_experts_tuple = _parse_experts(
            args.live_active_experts,
            args.num_experts,
            args.moe_router_topk,
        )
        args.live_active_expert_phases_tuple = _parse_expert_phases(
            args.live_active_expert_phases,
            fallback=args.live_active_experts_tuple,
            num_experts=args.num_experts,
            topk=args.moe_router_topk,
        )
        args.live_active_experts_tuple = args.live_active_expert_phases_tuple[0]
        run_live_benchmark()
    finally:
        # torchrun terminates sibling ranks after the first uncaught error. Tear
        # down every derived NCCL/Gloo group first so in-flight work is not left
        # behind in the driver when that termination arrives.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
