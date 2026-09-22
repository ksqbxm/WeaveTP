#!/usr/bin/env python3
"""Online TP resharding while a Megatron MoE model keeps training.

The prototype targets one scale-out transition:

    source:      src_tp x expert_parallel_size
    destination: dst_tp x expert_parallel_size

The destination world spans every launched rank while the source world uses a
prefix of the ranks.  Model weights and AdamW tensor state are copied into a
shadow model over a dedicated NCCL process group.  Source ranks run real MoE
forward/backward/optimizer steps at the same time, so token All-to-All, expert
compute, TP collectives, and migration compete for the same GPU/interconnect
resources.

The bandwidth-aware executor predicts the original plan and bounded candidate
wave counts (1, 2, 4, ...), enables the candidate only when its conservative
gain reaches the configured threshold, and preserves same-peer FIFO in an
NCCL-style cyclic order. After each wave it records effective service rates for
later calibration. Parameters changed by the optimizer during the base copy
are refreshed with the untouched one-batch baseline at the consistent cutover.

Typical 8-GPU run:

  torchrun --standalone --nproc_per_node=8 tools/resharding/moe_tp.py \
    --src-tp 2 --dst-tp 4 --expert-parallel-size 2 --num-experts 8 \
    --scheduler auto --profile-p2p-bandwidth

Use ``--planner-self-test`` without torchrun to test the pure scheduling logic.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.resharding import build_centralized_reshard_plan
from megatron.core.resharding.utils import ReshardPlan
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tools.resharding import tp_build_src_dst_demo as tp_demo


DEMO_REVISION = "moe-tp-routing-v29-benefit-gated-nccl-waves"
MIGRATION_KINDS = ("weight", "exp_avg", "exp_avg_sq")
EXPERT_NAME_RE = re.compile(r"(?:^|\.)local_experts\.(\d+)(?:\.|$)")
_ENDPOINT_PRESSURE_BUFFERS: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}


class SequenceParallelTorchLayerNorm(torch.nn.LayerNorm):
    """PyTorch LayerNorm with Megatron sequence-parallel parameter metadata."""

    def __init__(
        self,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1.0e-5,
        **_kwargs,
    ) -> None:
        if config.normalization != "LayerNorm":
            raise ValueError("SequenceParallelTorchLayerNorm only supports LayerNorm")
        if config.layernorm_zero_centered_gamma:
            raise ValueError("Torch LayerNorm does not support zero-centered gamma")
        super().__init__(hidden_size, eps=eps)
        self.sequence_parallel = bool(config.sequence_parallel)
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)
        setattr(self.bias, "sequence_parallel", self.sequence_parallel)


def _expert_routing_bias(
    num_experts: int,
    mode: str,
    magnitude: float,
    layer_number: Optional[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a topology-independent expert preference vector."""
    if mode == "balanced":
        return torch.zeros(num_experts, device=device, dtype=dtype)

    hot_expert = 0
    if mode == "shifting":
        hot_expert = max((layer_number or 1) - 1, 0) % num_experts
    expert_ids = torch.arange(num_experts, device=device)
    priority = ((expert_ids - hot_expert) % num_experts).to(dtype=dtype)
    return -abs(float(magnitude)) * priority


class TopologyInvariantTopKRouter(TopKRouter):
    """Top-k router whose synthetic skew is stable across TP layouts."""

    def gating(self, input: torch.Tensor) -> torch.Tensor:
        logits = super().gating(input)
        mode = getattr(self.config, "demo_routing_mode", "balanced")
        if mode == "balanced":
            # A random, untrained router can be heavily skewed even when no bias is
            # applied. Use deterministic round-robin targets for the control case so
            # "balanced" measures the network rather than router initialization noise.
            token_ids = torch.arange(
                logits.numel() // self.num_experts,
                device=logits.device,
            ).reshape(logits.shape[:-1])
            target_experts = token_ids.remainder(self.num_experts)
            balanced_logits = torch.full_like(logits, -1.0e4)
            balanced_logits.scatter_(-1, target_experts.unsqueeze(-1), 0.0)
            return logits * 0.0 + balanced_logits
        if mode == "fixed-hot":
            active_experts = tuple(getattr(self.config, "demo_active_experts", (0,)))
            active_mask = torch.zeros(
                self.num_experts,
                device=logits.device,
                dtype=torch.bool,
            )
            active_mask[list(active_experts)] = True
            return logits.masked_fill(~active_mask.unsqueeze(0), float("-inf"))

        magnitude = getattr(self.config, "demo_router_bias_magnitude", 0.0)
        bias = _expert_routing_bias(
            self.num_experts,
            mode,
            magnitude,
            self.layer_number,
            device=logits.device,
            dtype=logits.dtype,
        )
        return logits + bias


def _migration_name(kind: str, parameter_name: str) -> str:
    return f"{kind}::{parameter_name}"


def _base_parameter_name(migration_name: str) -> str:
    kind, separator, parameter_name = migration_name.partition("::")
    if not separator or kind not in MIGRATION_KINDS:
        return migration_name
    return parameter_name


def _is_expert_parameter(name: str) -> bool:
    return EXPERT_NAME_RE.search(name) is not None


def _expert_module_name(name: str) -> Optional[str]:
    match = EXPERT_NAME_RE.search(name)
    if match is None:
        return None
    return name[: match.end()].rstrip(".")


@dataclass
class DirtyTracker:
    """Track optimizer-visible parameter updates and expert token activity."""

    update_counts: dict[str, int] = field(default_factory=dict)
    expert_tokens: dict[str, int] = field(default_factory=dict)
    _step_expert_tokens: dict[str, int] = field(default_factory=dict)
    _stateful_expert_modules: set[str] = field(default_factory=set)
    _expert_activity_enabled: bool = False
    steps: int = 0
    _hooks: list = field(default_factory=list)

    def install_expert_hooks(
        self,
        model: torch.nn.Module,
        hidden_size: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self._expert_activity_enabled = True
        if optimizer is not None:
            for parameter_name, parameter in model.named_parameters():
                expert_module = _expert_module_name(parameter_name)
                if expert_module is None:
                    continue
                state = optimizer.state.get(parameter, {})
                if any(
                    torch.is_tensor(state.get(key))
                    and bool(torch.count_nonzero(state[key]).item())
                    for key in ("exp_avg", "exp_avg_sq")
                ):
                    self._stateful_expert_modules.add(expert_module)
        for module_name, module in model.named_modules():
            if _expert_module_name(module_name) != module_name:
                continue

            def count_tokens(_module, inputs, name=module_name):
                if not inputs or not torch.is_tensor(inputs[0]):
                    return
                tensor = inputs[0]
                tokens = tensor.numel() // max(hidden_size, 1)
                self.expert_tokens[name] = self.expert_tokens.get(name, 0) + int(tokens)
                self._step_expert_tokens[name] = (
                    self._step_expert_tokens.get(name, 0) + int(tokens)
                )

            self._hooks.append(module.register_forward_pre_hook(count_tokens))

    def observe_gradients(self, model: torch.nn.Module) -> set[str]:
        dirty = set()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            expert_module = _expert_module_name(name)
            if expert_module is not None and self._step_expert_tokens.get(expert_module, 0) > 0:
                self._stateful_expert_modules.add(expert_module)
            if (
                self._expert_activity_enabled
                and expert_module is not None
                and self._step_expert_tokens.get(expert_module, 0) == 0
                and expert_module not in self._stateful_expert_modules
            ):
                # Megatron may materialize zero gradients for experts that received no
                # tokens. AdamW must skip them so their weights and state stay clean.
                parameter.grad = None
                continue
            dirty.add(name)
        for name in dirty:
            self.update_counts[name] = self.update_counts.get(name, 0) + 1
        self._step_expert_tokens.clear()
        self.steps += 1
        return dirty

    @property
    def dirty_names(self) -> set[str]:
        return set(self.update_counts)

    def rates(self) -> dict[str, float]:
        denominator = max(self.steps, 1)
        return {name: count / denominator for name, count in self.update_counts.items()}

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


class MigrationBundle(torch.nn.Module):
    """Expose model weights and AdamW state as one reshardable module."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        pg_collection,
    ) -> None:
        super().__init__()
        self.config = model.config
        self.pg_collection = pg_collection
        self._entries: list[tuple[str, torch.nn.Parameter]] = []

        for name, parameter in model.named_parameters():
            self._entries.append((_migration_name("weight", name), parameter))
            state = optimizer.state[parameter]
            for state_key in ("exp_avg", "exp_avg_sq"):
                state_parameter = torch.nn.Parameter(state[state_key], requires_grad=False)
                if state_parameter.data_ptr() != state[state_key].data_ptr():
                    raise RuntimeError(f"{name}.{state_key} wrapper does not alias optimizer state")
                tp_demo._copy_tp_metadata(parameter, state_parameter)
                self._entries.append((_migration_name(state_key, name), state_parameter))

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        name_prefix = f"{prefix}." if prefix else ""
        for name, parameter in self._entries:
            yield name_prefix + name, parameter


def _tensor_slice_digest(tensor: torch.Tensor, index: tuple[slice, ...]) -> str:
    payload = tensor.detach()[index].contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _audit_plan_payloads(
    plan: ReshardPlan,
    src_bundle: Optional[MigrationBundle],
    dst_bundle: MigrationBundle,
) -> int:
    """Verify every migrated tensor slice byte-for-byte using task IDs."""
    src_params = dict(src_bundle.named_parameters()) if src_bundle is not None else {}
    dst_params = dict(dst_bundle.named_parameters())
    local_sends = {}
    local_recvs = {}
    with torch.no_grad():
        for op in plan.send_ops:
            if op.task_id is None:
                continue
            local_sends[int(op.task_id)] = (
                op.param_name,
                _tensor_slice_digest(src_params[op.param_name], op.my_slice),
            )
        for op in plan.recv_ops:
            if op.task_id is None:
                continue
            local_recvs[int(op.task_id)] = (
                op.param_name,
                _tensor_slice_digest(dst_params[op.param_name], op.my_slice),
            )

    gathered_sends = [None] * dist.get_world_size()
    gathered_recvs = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_sends, local_sends)
    dist.all_gather_object(gathered_recvs, local_recvs)

    mismatch_count = 0
    if dist.get_rank() == 0:
        sends = {task_id: value for payload in gathered_sends for task_id, value in payload.items()}
        recvs = {task_id: value for payload in gathered_recvs for task_id, value in payload.items()}
        all_task_ids = sorted(set(sends) | set(recvs))
        mismatches = [
            (task_id, sends.get(task_id), recvs.get(task_id))
            for task_id in all_task_ids
            if sends.get(task_id, (None, None))[1] != recvs.get(task_id, (None, None))[1]
        ]
        mismatch_count = len(mismatches)
        by_kind = {kind: 0 for kind in MIGRATION_KINDS}
        for _task_id, send_value, recv_value in mismatches:
            name = (send_value or recv_value)[0]
            kind = name.partition("::")[0]
            if kind in by_kind:
                by_kind[kind] += 1
        print(
            "Migration payload audit: "
            f"tasks={len(all_task_ids)} mismatches={mismatch_count} "
            + " ".join(f"{kind}_mismatches={count}" for kind, count in by_kind.items()),
            flush=True,
        )
        for task_id, send_value, recv_value in mismatches[:5]:
            print(
                f"  mismatch task={task_id} send={send_value} recv={recv_value}",
                flush=True,
            )
    mismatch_count = _broadcast_object(mismatch_count)
    return int(mismatch_count)


@dataclass
class EffectiveBandwidthModel:
    """EWMA model of effective pairwise bandwidth under the current workload."""

    bandwidth_gbps: dict[tuple[int, int], float]
    ewma: float = 0.5
    latency_us: float = 5.0
    samples: int = 0

    def estimate_seconds(self, row: dict) -> float:
        pair = (int(row["src_rank"]), int(row["dst_rank"]))
        gbps = max(float(self.bandwidth_gbps.get(pair, 1.0)), 1.0e-9)
        return float(row["bytes"]) / (gbps * 1.0e9 / 8.0) + self.latency_us * 1.0e-6

    def remote_cv(self) -> float:
        values = [
            value
            for (src, dst), value in self.bandwidth_gbps.items()
            if src != dst and value > 0.0
        ]
        if len(values) < 2:
            return 0.0
        mean = statistics.fmean(values)
        return statistics.pstdev(values) / max(mean, 1.0e-9)

    def update_from_wave(self, rows: list[dict], elapsed_s: float) -> list[dict]:
        if elapsed_s <= 0.0:
            return []
        bytes_by_pair: dict[tuple[int, int], int] = {}
        for row in rows:
            pair = (int(row["src_rank"]), int(row["dst_rank"]))
            bytes_by_pair[pair] = bytes_by_pair.get(pair, 0) + int(row["bytes"])

        updates = []
        for pair, num_bytes in bytes_by_pair.items():
            observed = max(num_bytes * 8.0 / elapsed_s / 1.0e9, 1.0e-9)
            previous = max(self.bandwidth_gbps.get(pair, observed), 1.0e-9)
            # A wave shares its elapsed time with training and other disjoint pairs.
            # Treat the result as effective service rate, not physical link capacity.
            updated = self.ewma * observed + (1.0 - self.ewma) * previous
            self.bandwidth_gbps[pair] = updated
            updates.append(
                {
                    "src": pair[0],
                    "dst": pair[1],
                    "bytes": num_bytes,
                    "observed_gbps": observed,
                    "ewma_gbps": updated,
                }
            )
        self.samples += 1
        return updates


def _adaptive_wave_budget(remote_cv: float, min_cv: float, max_waves: int) -> int:
    if min_cv <= 0.0:
        raise ValueError("min_cv must be positive")
    if remote_cv < min_cv:
        return 1
    return min(max_waves, max(2, math.ceil(remote_cv / min_cv)))


def _candidate_wave_counts(max_waves: int) -> list[int]:
    """Return a small, bounded search space without assuming one fixed wave count."""
    counts = [1]
    candidate = 2
    while candidate < max_waves:
        counts.append(candidate)
        candidate *= 2
    if max_waves > 1 and counts[-1] != max_waves:
        counts.append(max_waves)
    return counts


def _nccl_round_task_order(rows: list[dict], world_size: int) -> list[int]:
    """Preserve peer FIFO while interleaving queues in NCCL-style cyclic rounds."""
    local_rows: list[dict] = []
    peer_queues: dict[tuple[int, int], list[dict]] = {}
    for row in sorted(rows, key=lambda item: int(item["task_id"])):
        src = int(row["src_rank"])
        dst = int(row["dst_rank"])
        if src == dst:
            local_rows.append(row)
        else:
            peer_queues.setdefault((src, dst), []).append(row)

    ordered = [int(row["task_id"]) for row in local_rows]
    while peer_queues:
        progressed = False
        for distance in range(1, max(world_size, 1)):
            active_pairs = sorted(
                pair
                for pair in peer_queues
                if (pair[1] - pair[0]) % max(world_size, 1) == distance
            )
            for pair in active_pairs:
                queue = peer_queues[pair]
                ordered.append(int(queue.pop(0)["task_id"]))
                progressed = True
                if not queue:
                    del peer_queues[pair]
        if progressed:
            continue
        pair = min(peer_queues)
        ordered.append(int(peer_queues[pair].pop(0)["task_id"]))
        if not peer_queues[pair]:
            del peer_queues[pair]
    return ordered


def _estimate_rows_critical_seconds(
    rows: list[dict], bandwidth: EffectiveBandwidthModel
) -> float:
    """Estimate the rank/link critical path of one NCCL batch."""
    send_load: dict[int, float] = {}
    recv_load: dict[int, float] = {}
    link_load: dict[tuple[int, int], float] = {}
    local_load: dict[int, float] = {}
    for row in rows:
        src = int(row["src_rank"])
        dst = int(row["dst_rank"])
        cost = bandwidth.estimate_seconds(row)
        if src == dst:
            local_load[dst] = local_load.get(dst, 0.0) + cost
            continue
        send_load[src] = send_load.get(src, 0.0) + cost
        recv_load[dst] = recv_load.get(dst, 0.0) + cost
        link_load[(src, dst)] = link_load.get((src, dst), 0.0) + cost
    return max(
        [0.0]
        + list(send_load.values())
        + list(recv_load.values())
        + list(link_load.values())
        + list(local_load.values())
    )


def _build_nccl_compatible_waves(
    rows: list[dict],
    *,
    wave_count: int,
    bandwidth: EffectiveBandwidthModel,
    max_tasks: int,
    max_wave_bytes: int,
    world_size: int,
) -> list[list[int]]:
    """Pack bounded waves, then order each wave like NCCL's per-peer queues."""
    if not rows:
        return []
    planner = OnlineWavePlanner(
        bandwidth,
        dirty_weight=0.0,
        max_tasks=max_tasks,
        max_wave_bytes=max_wave_bytes,
    )
    remaining = list(rows)
    waves: list[list[int]] = []
    for wave_index in range(max(1, wave_count)):
        if not remaining:
            break
        waves_left = max(1, wave_count - wave_index)
        selected = planner.choose_wave(remaining, waves_left=waves_left)
        selected_set = set(selected)
        selected_rows = [
            row for row in remaining if int(row["task_id"]) in selected_set
        ]
        waves.append(_nccl_round_task_order(selected_rows, world_size))
        remaining = [
            row for row in remaining if int(row["task_id"]) not in selected_set
        ]
    if remaining:
        waves[-1].extend(_nccl_round_task_order(remaining, world_size))
    return waves


def _select_scheduler_execution(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    bandwidth: EffectiveBandwidthModel,
    args: argparse.Namespace,
    *,
    world_size: int,
    force: bool = False,
) -> dict[str, object]:
    """Choose whether to schedule and how many waves using a conservative model."""
    baseline_s = _estimate_rows_critical_seconds(baseline_rows, bandwidth)
    if not candidate_rows:
        return {
            "accepted": False,
            "baseline_s": baseline_s,
            "candidate_s": baseline_s,
            "predicted_gain_pct": 0.0,
            "prediction_lower_bound_pct": 0.0,
            "min_benefit_pct": args.scheduler_min_benefit_pct,
            "uncertainty_pct": args.scheduler_prediction_uncertainty_pct,
            "selected_wave_count": 0,
            "waves": [],
            "candidates": [],
        }
    candidates: list[dict[str, object]] = []
    for wave_count in _candidate_wave_counts(args.max_waves):
        waves = _build_nccl_compatible_waves(
            candidate_rows,
            wave_count=wave_count,
            bandwidth=bandwidth,
            max_tasks=args.max_wave_tasks,
            max_wave_bytes=args.max_wave_bytes,
            world_size=world_size,
        )
        predicted_s = sum(
            _estimate_rows_critical_seconds(
                [
                    row
                    for row in candidate_rows
                    if int(row["task_id"]) in set(task_ids)
                ],
                bandwidth,
            )
            for task_ids in waves
        )
        predicted_s += max(0, len(waves) - 1) * args.scheduler_wave_overhead_us * 1.0e-6
        predicted_s += args.scheduler_switch_overhead_us * 1.0e-6
        gain_pct = 100.0 * (baseline_s - predicted_s) / max(baseline_s, 1.0e-12)
        candidates.append(
            {
                "wave_count": len(waves),
                "waves": waves,
                "predicted_s": predicted_s,
                "predicted_gain_pct": gain_pct,
            }
        )

    best = min(
        candidates,
        key=lambda item: (float(item["predicted_s"]), int(item["wave_count"])),
    )
    lower_bound_pct = (
        float(best["predicted_gain_pct"])
        - args.scheduler_prediction_uncertainty_pct
    )
    accepted = force or lower_bound_pct >= args.scheduler_min_benefit_pct
    return {
        "accepted": accepted,
        "baseline_s": baseline_s,
        "candidate_s": float(best["predicted_s"]),
        "predicted_gain_pct": float(best["predicted_gain_pct"]),
        "prediction_lower_bound_pct": lower_bound_pct,
        "min_benefit_pct": args.scheduler_min_benefit_pct,
        "uncertainty_pct": args.scheduler_prediction_uncertainty_pct,
        "selected_wave_count": int(best["wave_count"]),
        "waves": best["waves"],
        "candidates": [
            {
                key: value
                for key, value in item.items()
                if key != "waves"
            }
            for item in candidates
        ],
    }


def _bandwidth_rows(
    matrix: dict[tuple[int, int], float],
) -> list[dict[str, float | int]]:
    return [
        {"src": src, "dst": dst, "gbps": float(gbps)}
        for (src, dst), gbps in sorted(matrix.items())
        if src != dst
    ]


def _bandwidth_change_summary(
    idle_matrix: dict[tuple[int, int], float],
    online_matrix: dict[tuple[int, int], float],
) -> dict[str, object]:
    rows = []
    for pair, idle_gbps in sorted(idle_matrix.items()):
        if pair[0] == pair[1] or idle_gbps <= 0.0 or pair not in online_matrix:
            continue
        online_gbps = max(float(online_matrix[pair]), 0.0)
        degradation = 1.0 - online_gbps / float(idle_gbps)
        rows.append(
            {
                "src": pair[0],
                "dst": pair[1],
                "idle_gbps": float(idle_gbps),
                "online_gbps": online_gbps,
                "degradation_pct": 100.0 * degradation,
            }
        )
    positive = [max(float(row["degradation_pct"]), 0.0) for row in rows]
    relative_capacities = [
        float(row["online_gbps"]) / max(float(row["idle_gbps"]), 1.0e-12)
        for row in rows
    ]
    relative_mean = (
        statistics.fmean(relative_capacities) if relative_capacities else 1.0
    )
    relative_cv = (
        statistics.pstdev(relative_capacities) / max(relative_mean, 1.0e-12)
        if len(relative_capacities) > 1
        else 0.0
    )
    worst = max(rows, key=lambda row: float(row["degradation_pct"]), default=None)
    return {
        "mean_degradation_pct": statistics.fmean(positive) if positive else 0.0,
        "max_degradation_pct": max(positive, default=0.0),
        "relative_capacity_cv": relative_cv,
        "relative_capacity_min": min(relative_capacities, default=1.0),
        "relative_capacity_mean": relative_mean,
        "worst_link": worst,
        "links": rows,
    }


class OnlineWavePlanner:
    """Select high-cost, low-dirty-rate tasks while avoiding endpoint contention."""

    def __init__(
        self,
        bandwidth: EffectiveBandwidthModel,
        *,
        dirty_weight: float,
        max_tasks: int,
        max_wave_bytes: int,
    ) -> None:
        self.bandwidth = bandwidth
        self.dirty_weight = max(dirty_weight, 0.0)
        self.max_tasks = max(max_tasks, 1)
        self.max_wave_bytes = max(max_wave_bytes, 0)
        self.dirty_rates: dict[tuple[int, str], float] = {}

    def set_dirty_rates(self, rates: dict[tuple[int, str], float]) -> None:
        self.dirty_rates = rates

    def _score(self, row: dict) -> tuple[float, int, int]:
        parameter_name = _base_parameter_name(str(row["param"]))
        dirty_rate = self.dirty_rates.get((int(row["src_rank"]), parameter_name), 0.0)
        cold_priority = 1.0 / (1.0 + self.dirty_weight * dirty_rate)
        estimated_s = self.bandwidth.estimate_seconds(row)
        return (estimated_s * cold_priority, int(row["bytes"]), -int(row["task_id"]))

    def choose_wave(
        self,
        rows: list[dict],
        *,
        waves_left: int,
    ) -> list[int]:
        if not rows:
            return []
        if waves_left <= 1:
            return [int(row["task_id"]) for row in rows]

        total_bytes = sum(int(row["bytes"]) for row in rows)
        byte_target = self.max_wave_bytes or max(1, math.ceil(total_bytes / waves_left))
        task_target = min(self.max_tasks, max(1, math.ceil(len(rows) / waves_left)))
        ordered = sorted(rows, key=self._score, reverse=True)
        selected: list[dict] = []
        selected_ids: set[int] = set()
        selected_pairs: set[tuple[int, int]] = set()
        used_sources: set[int] = set()
        used_destinations: set[int] = set()
        selected_bytes = 0

        def target_reached() -> bool:
            return len(selected) >= task_target or selected_bytes >= byte_target

        # Prefer a matching of independent endpoints first. If that matching
        # cannot fill an even share of this wave, admit endpoint conflicts in a
        # second pass. This avoids leaving a very large final wave while still
        # reducing head-of-line contention where the task graph permits it.
        for allow_endpoint_conflicts in (False, True):
            for row in ordered:
                task_id = int(row["task_id"])
                if task_id in selected_ids:
                    continue
                if len(selected) >= self.max_tasks or target_reached():
                    break
                src = int(row["src_rank"])
                dst = int(row["dst_rank"])
                pair = (src, dst)
                same_peer_stream = pair in selected_pairs
                endpoint_conflict = src in used_sources or dst in used_destinations
                if endpoint_conflict and not same_peer_stream and not allow_endpoint_conflicts:
                    continue
                selected.append(row)
                selected_ids.add(task_id)
                selected_pairs.add(pair)
                used_sources.add(src)
                used_destinations.add(dst)
                selected_bytes += int(row["bytes"])
            if target_reached() or len(selected) >= self.max_tasks:
                break

        if not selected:
            selected.append(ordered[0])
        return [int(row["task_id"]) for row in selected]


def _initialize_adamw_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    for parameter in model.parameters():
        state = optimizer.state[parameter]
        state["step"] = torch.zeros((), device=parameter.device, dtype=torch.float32)
        state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)


def _init_distributed(src_tp: int, ep_size: int) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    mpu.destroy_model_parallel()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=src_tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=src_tp,
        order="tp-cp-ep-dp-pp",
        create_gloo_process_groups=False,
    )


def _warmup_process_group(group) -> None:
    """Initialize a NCCL communicator before migration and training overlap."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    token = torch.ones(1, device=device)
    dist.all_reduce(token, group=group)
    torch.cuda.synchronize(device)
    dist.barrier()


def _make_config(tp_size: int, args: argparse.Namespace) -> TransformerConfig:
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        ffn_hidden_size=args.ffn_hidden_size,
        tensor_model_parallel_size=tp_size,
        sequence_parallel=tp_size > 1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=args.expert_parallel_size,
        expert_tensor_parallel_size=tp_size,
        num_moe_experts=args.num_experts,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        moe_router_topk=args.moe_router_topk,
        moe_router_pre_softmax=args.moe_router_topk == 1,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        # Megatron's benchmark-only forced routing replaces router logits with
        # random values.  Random tensor shapes differ under sequence parallel,
        # so TP1 and TP2 route identical tokens to different experts.  The demo
        # instead adds a deterministic expert bias in TopologyInvariantTopKRouter.
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        add_bias_linear=False,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    config.demo_routing_mode = args.routing_mode
    config.demo_router_bias_magnitude = abs(args.router_bias_std)
    config.demo_active_experts = tuple(args.active_expert_ids)
    return config


def _build_model(config, pg_collection, args: argparse.Namespace) -> GPTModel:
    pre_process, post_process = tp_demo.pp_flags(pg_collection)
    layer_spec = get_gpt_layer_local_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=False,
    )
    layer_spec.submodules.mlp.submodules.router = TopologyInvariantTopKRouter
    layer_spec.submodules.input_layernorm = SequenceParallelTorchLayerNorm
    layer_spec.submodules.pre_mlp_layernorm = SequenceParallelTorchLayerNorm
    block_spec = TransformerBlockSubmodules(
        layer_specs=[layer_spec] * config.num_layers,
        layer_norm=SequenceParallelTorchLayerNorm,
    )
    return GPTModel(
        config=config,
        transformer_layer_spec=block_spec,
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_len,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=False,
        parallel_output=False,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        pg_collection=pg_collection,
    )


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tracker: DirtyTracker,
    args: argparse.Namespace,
) -> tuple[float, set[str]]:
    optimizer.zero_grad(set_to_none=True)
    # Queue controlled endpoint traffic before the forward pass. The dedicated
    # P2P/migration stream can now overlap this traffic immediately; placing it
    # after optimizer.step() let short probes finish before pressure began.
    _apply_endpoint_pressure(args)
    logits = tp_demo.run_forward(
        model,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.micro_batch_size,
    )
    loss = logits.float().square().mean()
    loss.backward()
    dirty = tracker.observe_gradients(model)
    optimizer.step()
    return float(loss.detach().item()), dirty


def _apply_endpoint_pressure(args: argparse.Namespace) -> None:
    """Create real GPU-memory pressure on selected hot ranks during training.

    The copies run on the training stream while migration/probe NCCL kernels use
    independent streams and communicators.  This models a memory-heavy hot expert
    competing with communication at one endpoint without falsifying B[src][dst].
    """
    num_bytes = int(args.endpoint_pressure_bytes)
    if num_bytes <= 0 or dist.get_rank() not in args.endpoint_pressure_rank_ids:
        return
    device_index = torch.cuda.current_device()
    key = (device_index, num_bytes)
    buffers = _ENDPOINT_PRESSURE_BUFFERS.get(key)
    if buffers is None:
        device = torch.device(f"cuda:{device_index}")
        buffers = (
            torch.empty(num_bytes, dtype=torch.uint8, device=device),
            torch.empty(num_bytes, dtype=torch.uint8, device=device),
        )
        _ENDPOINT_PRESSURE_BUFFERS[key] = buffers
    source, destination = buffers
    for _ in range(args.endpoint_pressure_iters):
        destination.copy_(source, non_blocking=True)
        source, destination = destination, source


def _profile_p2p_bandwidth_under_training(
    args: argparse.Namespace,
    *,
    probe_group,
    coordination_group,
    src_model: Optional[torch.nn.Module],
    src_optimizer: Optional[torch.optim.Optimizer],
    tracker: Optional[DirtyTracker],
) -> dict[tuple[int, int], float]:
    """Measure P2P completion time while one synchronized MoE step runs concurrently."""
    world_size = dist.get_world_size(probe_group)
    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    num_bytes = args.loaded_profile_p2p_bytes or args.profile_p2p_bytes
    warmup_iters = args.loaded_profile_p2p_warmup_iters
    timed_iters = args.loaded_profile_p2p_iters
    timed_passes = args.loaded_profile_p2p_passes
    max_iters = max(warmup_iters, timed_iters)
    send_buffers = [
        torch.empty(num_bytes, device=device, dtype=torch.uint8) for _ in range(max_iters)
    ]
    recv_buffers = [
        torch.empty(num_bytes, device=device, dtype=torch.uint8) for _ in range(max_iters)
    ]
    probe_stream = torch.cuda.Stream(device=device)
    matrix: dict[tuple[int, int], float] = {}

    def run_probe(src: int, dst: int, iters: int) -> float:
        dist.barrier(group=coordination_group)
        if rank in (src, dst):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        p2p_ops = []
        with torch.cuda.stream(probe_stream):
            if rank == src:
                p2p_ops = [
                    dist.P2POp(dist.isend, send_buffers[index], dst, group=probe_group)
                    for index in range(iters)
                ]
            elif rank == dst:
                p2p_ops = [
                    dist.P2POp(dist.irecv, recv_buffers[index], src, group=probe_group)
                    for index in range(iters)
                ]
            requests = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []

        completion_elapsed = [0.0]

        def wait_for_probe() -> None:
            torch.cuda.set_device(device)
            # Work.wait() inserts the NCCL completion dependency into the
            # calling thread's current CUDA stream. Keep that stream identical
            # to the launch stream so its synchronization cannot finish before
            # the P2P transfer itself.
            with torch.cuda.stream(probe_stream):
                for request in requests:
                    request.wait()
            probe_stream.synchronize()
            completion_elapsed[0] = time.perf_counter() - start

        waiter = None
        if rank in (src, dst):
            waiter = threading.Thread(target=wait_for_probe, daemon=True)
            waiter.start()

        # This CPU-only handshake guarantees that every P2P launch precedes the
        # source ranks' MoE collectives, avoiding cross-communicator launch races.
        dist.barrier(group=coordination_group)
        if src_model is not None and src_optimizer is not None and tracker is not None:
            _train_step(src_model, src_optimizer, tracker, args)

        if waiter is not None:
            waiter.join()
        elapsed_s = completion_elapsed[0] if rank in (src, dst) else 0.0
        elapsed_tensor = torch.tensor(elapsed_s, dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=coordination_group)
        dist.barrier(group=coordination_group)
        return max(float(elapsed_tensor.item()), 1.0e-9)

    if src_model is not None:
        src_model.train()

    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                matrix[(src, dst)] = args.scheduler_local_bandwidth_gbps
                continue

            if warmup_iters:
                run_probe(src, dst, warmup_iters)
            elapsed_samples = [
                run_probe(src, dst, timed_iters) for _ in range(timed_passes)
            ]
            elapsed_s = statistics.median(elapsed_samples)
            matrix[(src, dst)] = max(
                num_bytes * timed_iters * 8.0 / elapsed_s / 1.0e9,
                1.0e-9,
            )

    dist.barrier(group=coordination_group)
    if rank == 0:
        model = EffectiveBandwidthModel(matrix)
        print(
            "\nProfiled P2P bandwidth under real MoE training: "
            f"payload={num_bytes} iters={timed_iters} passes={timed_passes} "
            f"remote_cv={model.remote_cv():.4f}",
            flush=True,
        )
        for src in range(world_size):
            cells = " ".join(f"{matrix[(src, dst)]:7.1f}" for dst in range(world_size))
            print(f"  src={src}: {cells}", flush=True)
    return matrix


def _save_loaded_bandwidth_profile(
    path: Optional[str],
    matrix: dict[tuple[int, int], float],
    args: argparse.Namespace,
) -> None:
    if not path or dist.get_rank() != 0:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    world_size = dist.get_world_size()
    payload = {
        "world_size": world_size,
        "unit": "Gbps",
        "condition": "real-moe-training",
        "routing_mode": args.routing_mode,
        "active_experts": list(args.active_expert_ids),
        "payload_bytes": args.loaded_profile_p2p_bytes or args.profile_p2p_bytes,
        "iters": args.loaded_profile_p2p_iters,
        "passes": args.loaded_profile_p2p_passes,
        "matrix_gbps": [
            [matrix[(src, dst)] for dst in range(world_size)]
            for src in range(world_size)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved loaded P2P bandwidth profile to {output_path}", flush=True)


def _measure_bandwidth_during_moe_training(
    src_model: Optional[torch.nn.Module],
    src_optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
) -> tuple[dict[tuple[int, int], float], dict[str, object]]:
    rank = dist.get_rank()
    all_ranks = list(range(dist.get_world_size()))
    probe_group = dist.new_group(ranks=all_ranks)
    coordination_group = dist.new_group(ranks=all_ranks, backend="gloo")
    tracker = DirtyTracker()
    try:
        if src_model is not None:
            tracker.install_expert_hooks(src_model, args.hidden_size, src_optimizer)

        # The first operation on a batch-P2P process group must involve all ranks.
        dist.barrier(group=coordination_group)
        warmup_token = torch.ones(1, device=f"cuda:{torch.cuda.current_device()}")
        dist.all_reduce(warmup_token, group=probe_group)
        torch.cuda.synchronize()
        dist.barrier(group=coordination_group)

        matrix = _profile_p2p_bandwidth_under_training(
            args,
            probe_group=probe_group,
            coordination_group=coordination_group,
            src_model=src_model,
            src_optimizer=src_optimizer,
            tracker=tracker if src_model is not None else None,
        )
        torch.cuda.synchronize()
        dist.barrier(group=coordination_group)

        local = {
            "rank": rank,
            "steps": tracker.steps,
            "expert_tokens": sum(tracker.expert_tokens.values()),
            "expert_modules": dict(sorted(tracker.expert_tokens.items())),
        }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        active = [item for item in gathered if int(item["steps"]) > 0]
        summary = {
            "training_steps_by_rank": {
                str(item["rank"]): int(item["steps"]) for item in active
            },
            "expert_tokens_by_rank": {
                str(item["rank"]): int(item["expert_tokens"]) for item in active
            },
            "expert_modules_by_rank": {
                str(item["rank"]): item["expert_modules"] for item in active
            },
        }
        if rank == 0:
            print(
                "Loaded-profile MoE activity: "
                f"steps_by_rank={summary['training_steps_by_rank']} "
                f"expert_tokens_by_rank={summary['expert_tokens_by_rank']}",
                flush=True,
            )
        _save_loaded_bandwidth_profile(args.loaded_profile_p2p_output, matrix, args)
        return matrix, summary
    finally:
        tracker.close()
        dist.destroy_process_group(coordination_group)
        dist.destroy_process_group(probe_group)


def _gather_rank_dirty_rates(
    tracker: Optional[DirtyTracker],
    initial_rates: dict[str, float],
) -> dict[tuple[int, str], float]:
    rank = dist.get_rank()
    local_rates = tracker.rates() if tracker is not None and tracker.steps else initial_rates
    local_payload = {name: float(rate) for name, rate in local_rates.items()}
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_payload)
    merged: dict[tuple[int, str], float] = {}
    for source_rank, rates in enumerate(gathered):
        for name, rate in rates.items():
            merged[(source_rank, name)] = float(rate)
    if rank != 0:
        return {}
    return merged


def _dirty_task_ids(plan: ReshardPlan, dirty_names: set[str]) -> set[int]:
    local_ids = {
        int(op.task_id)
        for op in plan.send_ops
        if op.task_id is not None and _base_parameter_name(op.param_name) in dirty_names
    }
    gathered: list[set[int]] = [set() for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_ids)
    return set().union(*gathered)


def _copy_optimizer_steps(
    src_model: Optional[torch.nn.Module],
    dst_model: torch.nn.Module,
    src_optimizer: Optional[torch.optim.Optimizer],
    dst_optimizer: torch.optim.Optimizer,
    src_pgs,
    dst_pgs,
) -> None:
    """Copy scalar Adam steps by EP identity and replicate them over destination TP."""
    local_payload: dict[tuple[int, str], float] = {}
    if src_model is not None and src_optimizer is not None:
        src_tp_rank = dist.get_rank(src_pgs.tp)
        src_ep_rank = dist.get_rank(src_pgs.ep)
        if src_tp_rank == 0:
            for name, parameter in src_model.named_parameters():
                local_payload[(src_ep_rank, name)] = float(
                    src_optimizer.state[parameter]["step"].item()
                )

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_payload)
    source_steps: dict[tuple[int, str], float] = {}
    for payload in gathered:
        source_steps.update(payload)

    dst_ep_rank = dist.get_rank(dst_pgs.ep)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    local_max_diff = torch.zeros((), device=device)
    with torch.no_grad():
        for name, parameter in dst_model.named_parameters():
            key = (dst_ep_rank, name)
            if key not in source_steps:
                # Dense parameters are replicated across EP. Their optimizer step is
                # identical, so any source EP replica is a valid control-state source.
                key = next(
                    (candidate for candidate in source_steps if candidate[1] == name),
                    None,
                )
            if key is None:
                raise RuntimeError(f"No source Adam step found for destination parameter {name}")
            dst_optimizer.state[parameter]["step"].fill_(source_steps[key])
            step_diff = (
                dst_optimizer.state[parameter]["step"]
                - float(source_steps[key])
            ).abs()
            local_max_diff = torch.maximum(local_max_diff, step_diff)
    dist.all_reduce(local_max_diff, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f"Optimizer step audit: max_diff={local_max_diff.item():.8f}", flush=True)


def _max_elapsed(local_elapsed: float, *, group=None) -> float:
    backend = str(dist.get_backend(group)).lower()
    device = (
        torch.device("cpu")
        if "gloo" in backend
        else torch.device(f"cuda:{torch.cuda.current_device()}")
    )
    tensor = torch.tensor(
        local_elapsed,
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


def _broadcast_object(value, src: int = 0):
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def _print_task_summary(rows: list[dict], label: str) -> None:
    if dist.get_rank() != 0:
        return
    total_bytes = sum(int(row["bytes"]) for row in rows)
    remote_bytes = sum(int(row["bytes"]) for row in rows if row["remote"])
    expert_bytes = sum(
        int(row["bytes"])
        for row in rows
        if _is_expert_parameter(_base_parameter_name(str(row["param"])))
    )
    state_bytes = sum(
        int(row["bytes"])
        for row in rows
        if not str(row["param"]).startswith("weight::")
    )
    print(
        f"\nMigration task summary for {label}: tasks={len(rows)} "
        f"total_bytes={total_bytes} remote_bytes={remote_bytes} "
        f"expert_bytes={expert_bytes} optimizer_state_bytes={state_bytes}",
        flush=True,
    )


def _execute_online_plan(
    plan: ReshardPlan,
    src_bundle: Optional[MigrationBundle],
    dst_bundle: MigrationBundle,
    src_model: Optional[torch.nn.Module],
    src_optimizer: Optional[torch.optim.Optimizer],
    tracker: Optional[DirtyTracker],
    args: argparse.Namespace,
    bandwidth_matrix: dict[tuple[int, int], float],
    *,
    reshard_group,
    control_group,
    label: str,
    live_steps: int,
    scheduler_mode: Optional[str] = None,
) -> tuple[dict[tuple[int, int], float], dict[str, object]]:
    rank = dist.get_rank()
    rows = tp_demo._collect_recv_task_rows(plan, dst_bundle)
    baseline_plan = getattr(plan, "baseline_plan", plan)
    baseline_rows = tp_demo._collect_recv_task_rows(baseline_plan, dst_bundle)
    if rank == 0:
        _print_task_summary(rows, label)
        bandwidth = EffectiveBandwidthModel(
            dict(bandwidth_matrix),
            ewma=args.bandwidth_ewma,
            latency_us=args.scheduler_latency_us,
        )
        requested_mode = scheduler_mode or args.scheduler
        if live_steps == 0 or requested_mode == "baseline":
            gate = {
                "accepted": False,
                "reason": "cutover-baseline" if live_steps == 0 else "baseline-requested",
                "baseline_s": _estimate_rows_critical_seconds(baseline_rows, bandwidth),
                "candidate_s": _estimate_rows_critical_seconds(baseline_rows, bandwidth),
                "predicted_gain_pct": 0.0,
                "prediction_lower_bound_pct": 0.0,
                "min_benefit_pct": args.scheduler_min_benefit_pct,
                "uncertainty_pct": args.scheduler_prediction_uncertainty_pct,
                "selected_wave_count": 1 if baseline_rows else 0,
                "waves": (
                    [[int(row["task_id"]) for row in baseline_rows]]
                    if baseline_rows
                    else []
                ),
                "candidates": [],
            }
        else:
            gate = _select_scheduler_execution(
                baseline_rows,
                rows,
                bandwidth,
                args,
                world_size=dist.get_world_size(),
                force=args.force_bandwidth_routing,
            )
            gate["reason"] = "forced" if args.force_bandwidth_routing else (
                "predicted-benefit" if gate["accepted"] else "benefit-below-threshold"
            )
        effective_mode = "bandwidth-aware" if gate["accepted"] else "baseline"
        effective_max_waves = (
            int(gate["selected_wave_count"])
            if gate["accepted"]
            else (1 if baseline_rows else 0)
        )
        active_rows = list(rows if gate["accepted"] else baseline_rows)
        planned_waves = (
            list(gate["waves"])
            if gate["accepted"]
            else (
                [[int(row["task_id"]) for row in baseline_rows]]
                if baseline_rows
                else []
            )
        )
        print(
            "Bandwidth-aware benefit gate: "
            f"accept={gate['accepted']} reason={gate['reason']} "
            f"baseline_s={gate['baseline_s']:.6f} candidate_s={gate['candidate_s']:.6f} "
            f"predicted_gain_pct={gate['predicted_gain_pct']:.3f} "
            f"lower_bound_pct={gate['prediction_lower_bound_pct']:.3f} "
            f"required_pct={gate['min_benefit_pct']:.3f} "
            f"selected_waves={effective_max_waves} "
            f"p2p_order={args.scheduler_p2p_order if gate['accepted'] else 'send-recv'}",
            flush=True,
        )
        if gate["candidates"]:
            print(
                "  wave candidates: "
                + ", ".join(
                    f"k={item['wave_count']}:t={item['predicted_s']:.6f}s:"
                    f"gain={item['predicted_gain_pct']:.2f}%"
                    for item in gate["candidates"]
                ),
                flush=True,
            )
    else:
        bandwidth = None
        gate = None
        effective_mode = None
        effective_max_waves = None
        active_rows = None
        planned_waves = None

    gate = _broadcast_object(gate)
    effective_mode = _broadcast_object(effective_mode)
    effective_max_waves = int(_broadcast_object(effective_max_waves))
    planned_waves = _broadcast_object(planned_waves)
    execution_plan = plan if bool(gate["accepted"]) else baseline_plan
    if rank == 0:
        workload_summary = {
            "tasks": len(active_rows),
            "bytes": sum(int(row["bytes"]) for row in active_rows),
            "remote_bytes": sum(int(row["bytes"]) for row in active_rows if row["remote"]),
        }
    else:
        workload_summary = None
    workload_summary = _broadcast_object(workload_summary)
    remaining_live_steps = int(live_steps)
    total_migration_s = 0.0
    total_training_s = 0.0
    exposed_wait_s = 0.0
    completed_tasks = 0
    wave_index = 0
    wave_records: list[dict[str, object]] = []
    wall_start = time.perf_counter()

    while wave_index < len(planned_waves):
        if rank == 0:
            task_ids = list(planned_waves[wave_index])
            waves_left = max(1, len(planned_waves) - wave_index)
            steps_this_wave = min(
                remaining_live_steps,
                max(1, math.ceil(remaining_live_steps / waves_left))
                if remaining_live_steps
                else 0,
            )
            completed_ids = {
                task_id
                for completed_wave in planned_waves[:wave_index]
                for task_id in completed_wave
            }
            decision = (task_ids, steps_this_wave, len(active_rows) - len(completed_ids))
        else:
            decision = None
        decision = _broadcast_object(decision)

        task_ids, steps_this_wave, remaining_before = decision
        wave_plan = tp_demo._filter_reshard_plan_by_task_ids(
            execution_plan, set(task_ids)
        )
        if effective_mode == "bandwidth-aware":
            wave_plan = tp_demo._reorder_reshard_plan_by_task_ids(
                wave_plan, list(task_ids)
            )
        if rank == 0:
            wave_rows = [
                row for row in active_rows if int(row["task_id"]) in set(task_ids)
            ]
            wave_bytes = sum(int(row["bytes"]) for row in wave_rows)
            wave_remote_bytes = sum(int(row["bytes"]) for row in wave_rows if row["remote"])
            print(
                f"\nOnline migration wave={wave_index} mode={effective_mode} "
                f"tasks={len(task_ids)} bytes={wave_bytes} remote_bytes={wave_remote_bytes} "
                f"remaining_before={remaining_before} train_steps={steps_this_wave}",
                flush=True,
            )
        else:
            wave_rows = None

        migration_args = argparse.Namespace(**vars(args))
        migration_args.nccl_copy_p2p_order = (
            args.scheduler_p2p_order
            if effective_mode == "bandwidth-aware"
            else "send-recv"
        )
        migration = tp_demo.LiveMigrationThread(
            label=f"{label} wave {wave_index}",
            plan=wave_plan,
            wave_plans=[wave_plan],
            src_module=src_bundle,
            dst_module=dst_bundle,
            args=migration_args,
            group=reshard_group,
            synchronize_group=False,
            print_summary=False,
        )
        dist.barrier(group=control_group)
        migration.start()
        migration.wait_until_launched()
        # Every rank submits migration P2P before any rank enters a training collective.
        # Keep this coordination off NCCL: a default-group NCCL barrier racing
        # the reshard communicator can create inconsistent cross-communicator
        # launch order and deadlock.
        dist.barrier(group=control_group)

        training_start = time.perf_counter()
        last_loss = 0.0
        if src_model is not None and src_optimizer is not None and tracker is not None:
            src_model.train()
            for _ in range(steps_this_wave):
                last_loss, _ = _train_step(src_model, src_optimizer, tracker, args)
        training_elapsed = time.perf_counter() - training_start

        wait_start = time.perf_counter()
        migration.join()
        local_wait = time.perf_counter() - wait_start
        dist.barrier(group=control_group)
        wave_elapsed = _max_elapsed(migration.elapsed_s, group=control_group)
        training_elapsed = _max_elapsed(training_elapsed, group=control_group)
        local_wait = _max_elapsed(local_wait, group=control_group)
        total_migration_s += wave_elapsed
        total_training_s += training_elapsed
        exposed_wait_s += local_wait
        remaining_live_steps -= steps_this_wave
        completed_tasks += len(task_ids)

        if rank == 0:
            updates = bandwidth.update_from_wave(wave_rows, wave_elapsed)
            wave_records.append(
                {
                    "wave": wave_index,
                    "tasks": len(task_ids),
                    "bytes": wave_bytes,
                    "remote_bytes": wave_remote_bytes,
                    "migration_s": wave_elapsed,
                    "training_s": training_elapsed,
                    "wait_s": local_wait,
                    "links": updates,
                }
            )
            slowest_updates = sorted(updates, key=lambda item: item["ewma_gbps"])[:4]
            update_text = ", ".join(
                f"{item['src']}->{item['dst']}:{item['ewma_gbps']:.1f}Gbps"
                for item in slowest_updates
            )
            print(
                f"Online wave result: migration_s={wave_elapsed:.6f} "
                f"training_s={training_elapsed:.6f} wait_s={local_wait:.6f} "
                f"last_loss={last_loss:.6f} effective_links=[{update_text}]",
                flush=True,
            )
        wave_index += 1

    if remaining_live_steps > 0:
        training_start = time.perf_counter()
        if src_model is not None and src_optimizer is not None and tracker is not None:
            for _ in range(remaining_live_steps):
                _train_step(src_model, src_optimizer, tracker, args)
        total_training_s += _max_elapsed(
            time.perf_counter() - training_start,
            group=control_group,
        )

    wall_s = _max_elapsed(time.perf_counter() - wall_start, group=control_group)

    if rank == 0:
        final_matrix = bandwidth.bandwidth_gbps
    else:
        final_matrix = None
    final_matrix = _broadcast_object(final_matrix)
    predicted_execution_s = float(
        gate["candidate_s"] if gate["accepted"] else gate["baseline_s"]
    )
    gate_metrics = {key: value for key, value in gate.items() if key != "waves"}
    gate_metrics["observed_migration_s"] = total_migration_s
    gate_metrics["prediction_error_pct"] = 100.0 * (
        total_migration_s - predicted_execution_s
    ) / max(total_migration_s, 1.0e-12)
    metrics = {
        "waves": float(wave_index),
        "effective_mode": effective_mode,
        "effective_max_waves": effective_max_waves,
        "tasks": float(completed_tasks),
        "migration_s": total_migration_s,
        "training_s": total_training_s,
        "exposed_wait_s": exposed_wait_s,
        "wall_s": wall_s,
        "live_steps": float(live_steps),
        "bytes": float(workload_summary["bytes"]),
        "remote_bytes": float(workload_summary["remote_bytes"]),
        "wave_records": wave_records,
        "scheduler_enabled": bool(gate["accepted"]),
        "scheduler_gate": gate_metrics,
        "p2p_order": args.scheduler_p2p_order if gate["accepted"] else "send-recv",
    }
    return final_matrix, metrics


def _validate_migration(
    src_model: Optional[torch.nn.Module],
    dst_model: torch.nn.Module,
    src_optimizer: Optional[torch.optim.Optimizer],
    dst_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    *,
    src_ranks: list[int],
) -> tuple[float, Optional[float]]:
    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if src_model is not None:
        src_model.eval()
    dst_model.eval()
    with torch.no_grad():
        torch.manual_seed(args.seed + 1000)
        torch.cuda.manual_seed_all(args.seed + 1000)
        reference = tp_demo._broadcast_active_source_logits(
            src_model,
            src_root_rank=src_ranks[0],
            args=args,
        )
        torch.manual_seed(args.seed + 1000)
        torch.cuda.manual_seed_all(args.seed + 1000)
        destination = tp_demo.run_forward(
            dst_model,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.micro_batch_size,
        )
        max_diff = (reference - destination).abs().max()
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)

    next_step_diff = None
    next_step_relative_diff = None
    if args.validate_next_optimizer_step:
        source_validation_tracker = DirtyTracker()
        destination_validation_tracker = DirtyTracker()
        if src_model is not None and src_optimizer is not None:
            torch.manual_seed(args.seed + 2000)
            torch.cuda.manual_seed_all(args.seed + 2000)
            _train_step(src_model, src_optimizer, source_validation_tracker, args)
        dist.barrier()
        torch.manual_seed(args.seed + 2000)
        torch.cuda.manual_seed_all(args.seed + 2000)
        _train_step(dst_model, dst_optimizer, destination_validation_tracker, args)
        dist.barrier()
        if src_model is not None:
            src_model.eval()
        dst_model.eval()
        with torch.no_grad():
            torch.manual_seed(args.seed + 3000)
            torch.cuda.manual_seed_all(args.seed + 3000)
            reference_after_step = tp_demo._broadcast_active_source_logits(
                src_model,
                src_root_rank=src_ranks[0],
                args=args,
            )
            torch.manual_seed(args.seed + 3000)
            torch.cuda.manual_seed_all(args.seed + 3000)
            destination_after_step = tp_demo.run_forward(
                dst_model,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                batch_size=args.micro_batch_size,
            )
            next_step_tensor = (reference_after_step - destination_after_step).abs().max()
            reference_scale = reference_after_step.abs().max()
        dist.all_reduce(next_step_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(reference_scale, op=dist.ReduceOp.MAX)
        next_step_diff = float(next_step_tensor.item())
        next_step_relative_diff = next_step_diff / max(float(reference_scale.item()), 1.0e-12)

    if rank == 0:
        print(
            f"\nMoE+TP migration validation: weight_max_diff={max_diff.item():.8f} "
            f"weight_passed={max_diff.item() <= args.validation_tolerance}",
            flush=True,
        )
        if next_step_diff is not None:
            print(
                f"Optimizer continuation validation: next_step_max_diff={next_step_diff:.8f} "
                f"next_step_relative_diff={next_step_relative_diff:.8f} "
                f"atol={args.optimizer_validation_tolerance:.8f} "
                f"passed={next_step_diff <= args.optimizer_validation_tolerance}",
                flush=True,
            )

    weight_ok = float(max_diff.item()) <= args.validation_tolerance
    optimizer_ok = (
        next_step_diff is None or next_step_diff <= args.optimizer_validation_tolerance
    )
    status = torch.tensor(
        int(weight_ok and optimizer_ok),
        device=device,
        dtype=torch.int32,
    )
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if not bool(status.item()):
        raise AssertionError(
            f"MoE+TP migration validation failed: weight_diff={max_diff.item()}, "
            f"next_step_diff={next_step_diff}"
        )
    return float(max_diff.item()), next_step_diff


def _collect_dirty_summary(
    tracker: Optional[DirtyTracker],
    dirty_task_rows: list[dict],
) -> dict[str, object]:
    local_payload = {
        "rank": dist.get_rank(),
        "steps": tracker.steps if tracker is not None else 0,
        "dirty": len(tracker.dirty_names) if tracker is not None else 0,
        "dirty_expert": (
            sum(_is_expert_parameter(name) for name in tracker.dirty_names)
            if tracker is not None
            else 0
        ),
        "expert_tokens": sum(tracker.expert_tokens.values()) if tracker is not None else 0,
        "expert_modules": dict(sorted(tracker.expert_tokens.items())) if tracker is not None else {},
    }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_payload)
    if dist.get_rank() == 0:
        active = [item for item in gathered if item["steps"] > 0]
        dirty_bytes = sum(int(row["bytes"]) for row in dirty_task_rows)
        summary = {
            "active_training_ranks": [int(item["rank"]) for item in active],
            "expert_active_ranks": [
                int(item["rank"]) for item in active if int(item["expert_tokens"]) > 0
            ],
            "max_dirty_params": max((int(item["dirty"]) for item in active), default=0),
            "max_dirty_expert_params": max(
                (int(item["dirty_expert"]) for item in active), default=0
            ),
            "expert_tokens": sum(int(item["expert_tokens"]) for item in active),
            "expert_tokens_by_rank": {
                str(item["rank"]): int(item["expert_tokens"]) for item in active
            },
            "expert_modules_by_rank": {
                str(item["rank"]): item["expert_modules"] for item in active
            },
            "dirty_refresh_tasks": len(dirty_task_rows),
            "dirty_refresh_bytes": dirty_bytes,
        }
        token_text = ",".join(
            f"r{rank}:{tokens}"
            for rank, tokens in summary["expert_tokens_by_rank"].items()
        )
        print(
            "\nDirty-state summary: "
            f"active_ranks={len(active)} "
            f"expert_active_ranks={summary['expert_active_ranks']} "
            f"max_dirty_params={summary['max_dirty_params']} "
            f"max_dirty_expert_params={summary['max_dirty_expert_params']} "
            f"expert_tokens={summary['expert_tokens']} "
            f"expert_tokens_by_rank=[{token_text}] "
            f"dirty_refresh_tasks={len(dirty_task_rows)} dirty_refresh_bytes={dirty_bytes}",
            flush=True,
        )
    else:
        summary = None
    return _broadcast_object(summary)


def _run(args: argparse.Namespace) -> None:
    _init_distributed(args.src_tp, args.expert_parallel_size)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world = args.dst_tp * args.expert_parallel_size
    source_world = args.src_tp * args.expert_parallel_size
    invalid_pressure_ranks = [
        pressure_rank
        for pressure_rank in args.endpoint_pressure_rank_ids
        if pressure_rank < 0 or pressure_rank >= world_size
    ]
    if invalid_pressure_ranks:
        raise ValueError(
            f"Endpoint pressure ranks outside WORLD_SIZE={world_size}: "
            f"{invalid_pressure_ranks}"
        )
    if rank == 0:
        print(
            f"MoE+TP demo revision={DEMO_REVISION} "
            f"routing={args.routing_mode} router_bias={abs(args.router_bias_std):.3f} "
            f"active_experts={args.active_expert_ids} "
            f"endpoint_pressure_ranks={args.endpoint_pressure_rank_ids} "
            f"endpoint_pressure_bytes={args.endpoint_pressure_bytes} "
            f"endpoint_pressure_iters={args.endpoint_pressure_iters}",
            flush=True,
        )
    if world_size != expected_world:
        raise ValueError(
            f"WORLD_SIZE must equal dst_tp*EP={expected_world}, got {world_size}"
        )
    if source_world >= world_size:
        raise ValueError(
            f"This prototype requires scale-out: src_tp*EP={source_world} < WORLD_SIZE={world_size}"
        )

    src_ranks = list(range(source_world))
    dst_ranks = list(range(world_size))
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    src_pgs = None
    dst_pgs = None
    reshard_group = None
    control_group = None
    profile_tracker = None
    live_tracker = None
    try:
        if args.profile_p2p_bandwidth:
            bandwidth_matrix = tp_demo.profile_p2p_bandwidth_matrix(args)
        elif args.scheduler_bandwidth_profile:
            bandwidth_matrix = tp_demo.load_bandwidth_profile(
                args.scheduler_bandwidth_profile,
                expected_world_size=world_size,
            )
        else:
            bandwidth_matrix = tp_demo._default_bandwidth_matrix(args, world_size)
        bandwidth_matrix = tp_demo.apply_scheduler_slow_links(
            bandwidth_matrix,
            args,
            world_size,
        )
        idle_bandwidth_matrix = dict(bandwidth_matrix)
        loaded_profile_activity: dict[str, object] = {}

        model_parallel_cuda_manual_seed(args.seed)
        torch.manual_seed(args.seed)
        src_pgs = tp_demo.build_pg_collection(
            tp_size=args.src_tp,
            ep_size=args.expert_parallel_size,
            rank_count=source_world,
        )
        dst_pgs = tp_demo.build_pg_collection(
            tp_size=args.dst_tp,
            ep_size=args.expert_parallel_size,
            rank_count=world_size,
        )
        src_config = _make_config(args.src_tp, args)
        dst_config = _make_config(args.dst_tp, args)
        model_parallel_cuda_manual_seed(args.seed)
        torch.manual_seed(args.seed)
        src_model = (
            _build_model(src_config, src_pgs, args).to(device)
            if rank in src_ranks
            else None
        )
        model_parallel_cuda_manual_seed(args.seed)
        torch.manual_seed(args.seed)
        dst_model = _build_model(dst_config, dst_pgs, args).to(device)

        src_optimizer = (
            torch.optim.AdamW(
                src_model.parameters(),
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=0.0,
            )
            if src_model is not None
            else None
        )
        dst_optimizer = torch.optim.AdamW(
            dst_model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=0.0,
        )
        if src_model is not None:
            _initialize_adamw_state(src_model, src_optimizer)
        _initialize_adamw_state(dst_model, dst_optimizer)

        # Exclude first-use CUDA/NCCL initialization from the standalone step baseline.
        warmup_tracker = DirtyTracker()
        if src_model is not None:
            _train_step(src_model, src_optimizer, warmup_tracker, args)
            torch.cuda.synchronize(device)
        dist.barrier()

        profile_tracker = DirtyTracker()
        if src_model is not None:
            profile_tracker.install_expert_hooks(src_model, args.hidden_size, src_optimizer)
        dist.barrier()
        profile_training_start = time.perf_counter()
        if src_model is not None:
            for _ in range(args.dirty_profile_steps):
                _train_step(src_model, src_optimizer, profile_tracker, args)
        standalone_training_s = _max_elapsed(time.perf_counter() - profile_training_start)
        standalone_train_step_s = standalone_training_s / max(args.dirty_profile_steps, 1)
        dist.barrier()

        if args.profile_p2p_under_training:
            bandwidth_matrix, loaded_profile_activity = _measure_bandwidth_during_moe_training(
                src_model,
                src_optimizer,
                args,
            )
            dist.barrier()

        src_bundle = (
            MigrationBundle(src_model, src_optimizer, src_pgs)
            if src_model is not None
            else None
        )
        dst_bundle = MigrationBundle(dst_model, dst_optimizer, dst_pgs)
        plan_start = time.perf_counter()
        measured_remote_cv = EffectiveBandwidthModel(bandwidth_matrix).remote_cv()
        preplan_link_change = _bandwidth_change_summary(
            idle_bandwidth_matrix, bandwidth_matrix
        )
        measured_contention_cv = float(preplan_link_change["relative_capacity_cv"])
        measured_max_degradation_pct = float(
            preplan_link_change["max_degradation_pct"]
        )
        # Always build a bandwidth-aware candidate for aware/auto mode. The
        # unified benefit gate below compares that candidate with the untouched
        # baseline plan and is the only authority that enables execution.
        route_by_bandwidth = args.scheduler in ("bandwidth-aware", "auto")
        if rank == 0:
            print(
                "Bandwidth-aware source routing: "
                f"candidate_built={route_by_bandwidth} forced={args.force_bandwidth_routing} "
                f"scheduler={args.scheduler} "
                f"remote_cv={measured_remote_cv:.4f} "
                f"contention_cv={measured_contention_cv:.4f} "
                f"max_degradation_pct={measured_max_degradation_pct:.2f} "
                f"execution_benefit_threshold_pct={args.scheduler_min_benefit_pct:.2f}",
                flush=True,
            )
        plan = build_centralized_reshard_plan(
            src_bundle,
            dst_bundle,
            num_experts=args.num_experts,
            prefer_local_source=not args.force_remote_transfer,
            source_bandwidth_gbps=bandwidth_matrix if route_by_bandwidth else None,
            source_reference_bandwidth_gbps=(
                idle_bandwidth_matrix if route_by_bandwidth else None
            ),
            source_latency_us=args.scheduler_latency_us,
            source_reroute_min_gain_pct=(
                float("-inf")
                if args.force_bandwidth_routing
                else args.source_reroute_min_gain_pct
            ),
            source_reroute_min_contention_gain_pct=(
                float("-inf")
                if args.force_bandwidth_routing
                else args.source_reroute_min_contention_gain_pct
            ),
            source_reroute_min_global_gain_pct=(
                float("-inf")
                if args.force_bandwidth_routing
                else args.source_reroute_min_global_gain_pct
            ),
            source_reroute_min_bytes=(
                0 if args.force_bandwidth_routing else args.source_reroute_min_bytes
            ),
        )
        plan_build_s = tp_demo.reduce_max_seconds(time.perf_counter() - plan_start)
        source_route_stats = getattr(plan, "source_route_stats", {})
        source_routing_enabled = (
            route_by_bandwidth and int(source_route_stats.get("accepted", 0)) > 0
        )
        if rank == 0 and route_by_bandwidth and not source_routing_enabled:
            print(
                "Bandwidth-aware no-regret gate: fallback=True "
                "reason=no-high-confidence-reroutes",
                flush=True,
            )
        if args.print_intersection_plan:
            tp_demo.print_intersection_plan(
                plan,
                dst_bundle,
                label="MoE+TP weights and AdamW state",
                limit=args.intersection_plan_limit,
            )

        reshard_group = dist.new_group(ranks=dst_ranks)
        _warmup_process_group(reshard_group)
        control_group = dist.new_group(ranks=dst_ranks, backend="gloo")
        dist.barrier(group=control_group)
        live_tracker = DirtyTracker()
        if src_model is not None:
            live_tracker.install_expert_hooks(src_model, args.hidden_size, src_optimizer)

        transition_start = time.perf_counter()
        base_matrix, base_metrics = _execute_online_plan(
            plan,
            src_bundle,
            dst_bundle,
            src_model,
            src_optimizer,
            live_tracker if src_model is not None else None,
            args,
            bandwidth_matrix,
            reshard_group=reshard_group,
            control_group=control_group,
            label="MoE+TP base migration",
            live_steps=args.live_train_steps,
            scheduler_mode=args.scheduler,
        )

        local_dirty_names = live_tracker.dirty_names if live_tracker is not None else set()
        refresh_source_plan = getattr(plan, "baseline_plan", plan)
        dirty_ids = _dirty_task_ids(refresh_source_plan, local_dirty_names)
        refresh_plan = tp_demo._filter_reshard_plan_by_task_ids(
            refresh_source_plan, dirty_ids
        )
        refresh_rows = tp_demo._collect_recv_task_rows(refresh_plan, dst_bundle)
        dirty_summary = _collect_dirty_summary(
            live_tracker if src_model is not None else None,
            refresh_rows,
        )

        cutover_start = time.perf_counter()
        refresh_matrix, refresh_metrics = _execute_online_plan(
            refresh_plan,
            src_bundle,
            dst_bundle,
            None,
            None,
            None,
            args,
            base_matrix,
            reshard_group=reshard_group,
            control_group=control_group,
            label="MoE+TP dirty refresh",
            live_steps=0,
            scheduler_mode="baseline",
        )
        if args.audit_migration_payloads:
            _audit_plan_payloads(refresh_plan, src_bundle, dst_bundle)
        _copy_optimizer_steps(
            src_model,
            dst_model,
            src_optimizer,
            dst_optimizer,
            src_pgs,
            dst_pgs,
        )
        dist.barrier()
        cutover_refresh_s = tp_demo.reduce_max_seconds(time.perf_counter() - cutover_start)
        transition_wall_s = _max_elapsed(time.perf_counter() - transition_start)

        validation_start = time.perf_counter()
        weight_diff, next_step_diff = _validate_migration(
            src_model,
            dst_model,
            src_optimizer,
            dst_optimizer,
            args,
            src_ranks=src_ranks,
        )
        validation_s = tp_demo.reduce_max_seconds(time.perf_counter() - validation_start)

        if rank == 0:
            final_bandwidth = EffectiveBandwidthModel(refresh_matrix)
            idle_bandwidth = EffectiveBandwidthModel(idle_bandwidth_matrix)
            loaded_bandwidth = EffectiveBandwidthModel(bandwidth_matrix)
            post_migration_bandwidth = EffectiveBandwidthModel(base_matrix)
            link_change = _bandwidth_change_summary(
                idle_bandwidth_matrix,
                bandwidth_matrix,
            )
            overlap_train_step_s = base_metrics["training_s"] / max(
                base_metrics["live_steps"], 1.0
            )
            training_slowdown_pct = 100.0 * (
                overlap_train_step_s / max(standalone_train_step_s, 1.0e-12) - 1.0
            )
            training_disruption_s = base_metrics["exposed_wait_s"] + cutover_refresh_s
            result = {
                "revision": DEMO_REVISION,
                "scheduler": args.scheduler,
                "routing_mode": args.routing_mode,
                "active_experts": list(args.active_expert_ids),
                "endpoint_pressure_ranks": list(args.endpoint_pressure_rank_ids),
                "endpoint_pressure_bytes": args.endpoint_pressure_bytes,
                "endpoint_pressure_iters": args.endpoint_pressure_iters,
                "force_bandwidth_routing": args.force_bandwidth_routing,
                "source_reroute_min_gain_pct": args.source_reroute_min_gain_pct,
                "source_reroute_min_contention_gain_pct": (
                    args.source_reroute_min_contention_gain_pct
                ),
                "source_reroute_min_global_gain_pct": (
                    args.source_reroute_min_global_gain_pct
                ),
                "source_reroute_min_bytes": args.source_reroute_min_bytes,
                "min_link_degradation_pct": args.min_link_degradation_pct,
                "source_routing_proposed": source_routing_enabled,
                "source_routing_enabled": (
                    source_routing_enabled and base_metrics["scheduler_enabled"]
                ),
                "source_route_stats": source_route_stats,
                "scheduler_enabled": base_metrics["scheduler_enabled"],
                "scheduler_gate": base_metrics["scheduler_gate"],
                "scheduler_p2p_order": base_metrics["p2p_order"],
                "scheduler_min_benefit_pct": args.scheduler_min_benefit_pct,
                "scheduler_prediction_uncertainty_pct": (
                    args.scheduler_prediction_uncertainty_pct
                ),
                "src_tp": args.src_tp,
                "dst_tp": args.dst_tp,
                "expert_parallel_size": args.expert_parallel_size,
                "num_experts": args.num_experts,
                "migration_bytes": int(base_metrics["bytes"]),
                "remote_migration_bytes": int(base_metrics["remote_bytes"]),
                "plan_build_s": plan_build_s,
                "base_migration_s": base_metrics["migration_s"],
                "base_wall_s": base_metrics["wall_s"],
                "base_exposed_wait_s": base_metrics["exposed_wait_s"],
                "training_disruption_s": training_disruption_s,
                "standalone_train_step_s": standalone_train_step_s,
                "overlap_train_step_s": overlap_train_step_s,
                "training_slowdown_pct": training_slowdown_pct,
                "dirty_refresh_s": cutover_refresh_s,
                "transition_wall_s": transition_wall_s,
                "validation_s": validation_s,
                "weight_max_diff": weight_diff,
                "next_step_max_diff": next_step_diff,
                "base_waves": int(base_metrics["waves"]),
                "refresh_waves": int(refresh_metrics["waves"]),
                "base_effective_mode": base_metrics["effective_mode"],
                "refresh_effective_mode": refresh_metrics["effective_mode"],
                "dirty_refresh_uses_baseline_plan": True,
                "base_wave_records": base_metrics["wave_records"],
                "refresh_wave_records": refresh_metrics["wave_records"],
                "expert_activity": dirty_summary,
                "loaded_profile_activity": loaded_profile_activity,
                "idle_bandwidth_cv": idle_bandwidth.remote_cv(),
                "online_effective_bandwidth_cv": loaded_bandwidth.remote_cv(),
                "online_contention_cv": float(link_change["relative_capacity_cv"]),
                "idle_bandwidth_gbps": _bandwidth_rows(idle_bandwidth_matrix),
                "online_effective_bandwidth_gbps": _bandwidth_rows(bandwidth_matrix),
                "online_link_change": link_change,
                "post_migration_estimated_bandwidth_cv": (
                    post_migration_bandwidth.remote_cv()
                ),
                "post_migration_estimated_bandwidth_gbps": _bandwidth_rows(base_matrix),
                "final_effective_bandwidth_cv": final_bandwidth.remote_cv(),
                "audit_migration_payloads": args.audit_migration_payloads,
                "validate_next_optimizer_step": args.validate_next_optimizer_step,
            }
            print(
                "\nTiming summary for online MoE+TP scale-out: "
                f"plan_build_s={plan_build_s:.6f} "
                f"base_migration_s={base_metrics['migration_s']:.6f} "
                f"base_wall_s={base_metrics['wall_s']:.6f} "
                f"training_overlap_s={base_metrics['training_s']:.6f} "
                f"base_exposed_wait_s={base_metrics['exposed_wait_s']:.6f} "
                f"training_disruption_s={training_disruption_s:.6f} "
                f"standalone_train_step_s={standalone_train_step_s:.6f} "
                f"overlap_train_step_s={overlap_train_step_s:.6f} "
                f"training_slowdown_pct={training_slowdown_pct:.3f} "
                f"dirty_refresh_s={cutover_refresh_s:.6f} "
                f"validation_s={validation_s:.6f} "
                f"cutover_pause_s={cutover_refresh_s:.6f} "
                f"transition_wall_s={transition_wall_s:.6f} "
                f"base_waves={int(base_metrics['waves'])} "
                f"refresh_waves={int(refresh_metrics['waves'])} "
                f"idle_bandwidth_cv={idle_bandwidth.remote_cv():.4f} "
                f"online_bandwidth_cv={loaded_bandwidth.remote_cv():.4f} "
                f"max_link_degradation_pct={link_change['max_degradation_pct']:.2f} "
                f"final_effective_bandwidth_cv={final_bandwidth.remote_cv():.4f} "
                f"weight_max_diff={weight_diff:.8f} "
                f"next_step_max_diff={next_step_diff}",
                flush=True,
            )
            if args.metrics_output:
                output_path = Path(args.metrics_output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(f"Saved benchmark metrics to {output_path}", flush=True)
    finally:
        if profile_tracker is not None:
            profile_tracker.close()
        if live_tracker is not None:
            live_tracker.close()
        if control_group is not None:
            dist.destroy_process_group(control_group)
        if reshard_group is not None:
            dist.destroy_process_group(reshard_group)
        if src_pgs is not None:
            tp_demo.destroy_pg_collection(src_pgs)
        if dst_pgs is not None:
            tp_demo.destroy_pg_collection(dst_pgs)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mpu.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


def _planner_self_test() -> None:
    if _candidate_wave_counts(8) != [1, 2, 4, 8]:
        raise AssertionError("wave candidate search should use powers of two")
    if _adaptive_wave_budget(0.09, 0.10, 4) != 1:
        raise AssertionError("low-CV workload should fall back to one wave")
    if _adaptive_wave_budget(0.21, 0.10, 4) != 3:
        raise AssertionError("wave budget should scale with bandwidth CV")
    matrix = {
        (0, 0): 1000.0,
        (1, 1): 1000.0,
        (2, 2): 1000.0,
        (3, 3): 1000.0,
        (0, 2): 12.0,
        (0, 3): 100.0,
        (1, 2): 90.0,
        (1, 3): 80.0,
    }
    rows = [
        {
            "task_id": 0,
            "src_rank": 0,
            "dst_rank": 2,
            "bytes": 64 << 20,
            "remote": True,
            "param": "weight::dense",
        },
        {
            "task_id": 1,
            "src_rank": 0,
            "dst_rank": 3,
            "bytes": 32 << 20,
            "remote": True,
            "param": "weight::experts.local_experts.0.fc1",
        },
        {
            "task_id": 2,
            "src_rank": 1,
            "dst_rank": 2,
            "bytes": 32 << 20,
            "remote": True,
            "param": "weight::experts.local_experts.1.fc1",
        },
        {
            "task_id": 3,
            "src_rank": 1,
            "dst_rank": 3,
            "bytes": 16 << 20,
            "remote": True,
            "param": "exp_avg::experts.local_experts.1.fc1",
        },
    ]
    bandwidth = EffectiveBandwidthModel(matrix, ewma=0.5)
    planner = OnlineWavePlanner(
        bandwidth,
        dirty_weight=4.0,
        max_tasks=4,
        max_wave_bytes=64 << 20,
    )
    planner.set_dirty_rates(
        {
            (0, "dense"): 1.0,
            (0, "experts.local_experts.0.fc1"): 0.0,
            (1, "experts.local_experts.1.fc1"): 0.25,
        }
    )
    wave = planner.choose_wave(rows, waves_left=3)
    if not wave or any(task_id not in {0, 1, 2, 3} for task_id in wave):
        raise AssertionError(f"invalid wave: {wave}")
    bandwidth.update_from_wave([rows[0]], elapsed_s=0.10)
    if bandwidth.bandwidth_gbps[(0, 2)] >= 12.0:
        raise AssertionError("EWMA did not lower the observed slow-link bandwidth")
    fifo_rows = [
        {**rows[0], "task_id": 10},
        {**rows[0], "task_id": 11},
        {**rows[2], "task_id": 12},
    ]
    fifo_order = _nccl_round_task_order(fifo_rows, world_size=4)
    if fifo_order.index(10) > fifo_order.index(11):
        raise AssertionError("NCCL-compatible ordering broke same-peer FIFO")

    gate_args = argparse.Namespace(
        max_waves=8,
        max_wave_tasks=64,
        max_wave_bytes=0,
        scheduler_wave_overhead_us=1000.0,
        scheduler_switch_overhead_us=0.0,
        scheduler_prediction_uncertainty_pct=0.0,
        scheduler_min_benefit_pct=5.0,
    )
    baseline_rows = [{**rows[0], "bytes": 32 << 20}]
    candidate_rows = [
        {
            **baseline_rows[0],
            "src_rank": 1,
            "task_id": 4,
        }
    ]
    gate_matrix = dict(matrix)
    gate_matrix[(1, 2)] = 120.0
    gate = _select_scheduler_execution(
        baseline_rows,
        candidate_rows,
        EffectiveBandwidthModel(gate_matrix),
        gate_args,
        world_size=4,
    )
    if not gate["accepted"] or gate["selected_wave_count"] != 1:
        raise AssertionError(f"expected profitable one-wave schedule, got {gate}")
    gate_args.scheduler_min_benefit_pct = 95.0
    rejected = _select_scheduler_execution(
        baseline_rows,
        candidate_rows,
        EffectiveBandwidthModel(gate_matrix),
        gate_args,
        world_size=4,
    )
    if rejected["accepted"]:
        raise AssertionError("benefit gate accepted a candidate below its threshold")
    print(
        "planner self-test passed:",
        f"wave={wave}",
        f"remote_cv={bandwidth.remote_cv():.4f}",
        f"gate_gain_pct={gate['predicted_gain_pct']:.2f}",
        flush=True,
    )


def _parse_active_experts(value: str, num_experts: int) -> tuple[int, ...]:
    try:
        expert_ids = tuple(sorted({int(item.strip()) for item in value.split(",") if item.strip()}))
    except ValueError as exc:
        raise ValueError("--active-experts must be a comma-separated list of integers") from exc
    if not expert_ids:
        raise ValueError("--active-experts must contain at least one expert id")
    invalid = [expert_id for expert_id in expert_ids if not 0 <= expert_id < num_experts]
    if invalid:
        raise ValueError(
            f"--active-experts contains ids outside [0, {num_experts - 1}]: {invalid}"
        )
    return expert_ids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoE training-aware online TP scale-out with adaptive bandwidth scheduling"
    )
    parser.add_argument("--planner-self-test", action="store_true")
    parser.add_argument("--src-tp", type=int, default=2)
    parser.add_argument("--dst-tp", type=int, default=4)
    parser.add_argument("--expert-parallel-size", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--moe-router-topk", type=int, default=2)
    parser.add_argument(
        "--routing-mode",
        choices=("balanced", "skewed", "shifting", "fixed-hot"),
        default="skewed",
    )
    parser.add_argument("--router-bias-std", type=float, default=2.0)
    parser.add_argument(
        "--active-experts",
        type=str,
        default="0",
        help=(
            "Comma-separated global expert ids allowed by fixed-hot routing. "
            "All other router logits are masked to -inf."
        ),
    )
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-query-groups", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=1024)
    parser.add_argument("--moe-ffn-hidden-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dirty-profile-steps", type=int, default=2)
    parser.add_argument("--live-train-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)

    parser.add_argument(
        "--scheduler",
        choices=("baseline", "bandwidth-aware", "auto"),
        default="auto",
    )
    parser.add_argument("--max-waves", type=int, default=8)
    parser.add_argument("--max-wave-tasks", type=int, default=64)
    parser.add_argument("--max-wave-bytes", type=int, default=0)
    parser.add_argument("--dirty-rate-weight", type=float, default=4.0)
    parser.add_argument("--bandwidth-ewma", type=float, default=0.5)
    parser.add_argument(
        "--scheduler-min-benefit-pct",
        type=float,
        default=5.0,
        help=(
            "Enable the bandwidth-aware plan only when its predicted migration "
            "critical-path reduction is at least this percentage."
        ),
    )
    parser.add_argument(
        "--scheduler-prediction-uncertainty-pct",
        type=float,
        default=0.0,
        help=(
            "Conservative percentage-point margin subtracted from predicted gain "
            "before applying --scheduler-min-benefit-pct."
        ),
    )
    parser.add_argument(
        "--scheduler-switch-overhead-us",
        type=float,
        default=0.0,
        help="Fixed planning/dispatch overhead included in every aware candidate.",
    )
    parser.add_argument(
        "--scheduler-wave-overhead-us",
        type=float,
        default=1000.0,
        help="Estimated barrier/thread overhead for each additional explicit wave.",
    )
    parser.add_argument(
        "--scheduler-release-cache",
        action="store_true",
        help="Release CUDA allocator cache after the final scheduled wave.",
    )
    parser.add_argument("--min-bandwidth-cv", type=float, default=0.10)
    parser.add_argument(
        "--min-link-degradation-pct",
        type=float,
        default=50.0,
        help=(
            "Require at least one loaded link to degrade by this percentage in "
            "addition to the normalized contention-CV gate."
        ),
    )
    parser.add_argument("--scheduler-latency-us", type=float, default=5.0)
    parser.add_argument(
        "--scheduler-p2p-order",
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
        help="P2P submission order for accepted bandwidth-aware plans.",
    )
    parser.add_argument(
        "--force-bandwidth-routing",
        action="store_true",
        help=(
            "Ablation only: bypass heterogeneity, local-gain, task-size, global-gain, "
            "and cutover fallback gates so the candidate schedule always executes."
        ),
    )
    parser.add_argument("--source-reroute-min-gain-pct", type=float, default=5.0)
    parser.add_argument(
        "--source-reroute-min-contention-gain-pct", type=float, default=0.0
    )
    parser.add_argument(
        "--source-reroute-min-global-gain-pct",
        type=float,
        default=5.0,
        help=(
            "Minimum predicted reduction of the complete plan critical path; "
            "otherwise all tentative source changes are discarded."
        ),
    )
    parser.add_argument("--source-reroute-min-bytes", type=int, default=0)
    parser.add_argument("--scheduler-local-bandwidth-gbps", type=float, default=1000.0)
    parser.add_argument("--scheduler-remote-bandwidth-gbps", type=float, default=100.0)
    parser.add_argument("--scheduler-bandwidth-profile", type=str, default=None)
    parser.add_argument("--scheduler-slow-links", type=str, default="")
    parser.add_argument("--scheduler-slow-link-simulate-delay", action="store_true")

    parser.add_argument(
        "--profile-p2p-bandwidth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--profile-p2p-bytes", type=int, default=8 << 20)
    parser.add_argument("--profile-p2p-iters", type=int, default=2)
    parser.add_argument(
        "--profile-p2p-passes",
        type=int,
        default=3,
        help="Independent idle-profile passes per directed pair; median is used.",
    )
    parser.add_argument("--profile-p2p-warmup-iters", type=int, default=1)
    parser.add_argument("--profile-p2p-output", type=str, default=None)
    parser.add_argument(
        "--profile-p2p-under-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Profile real P2P links on a dedicated NCCL group while source ranks "
            "continuously execute MoE training steps."
        ),
    )
    parser.add_argument("--loaded-profile-p2p-bytes", type=int, default=8 << 20)
    parser.add_argument("--loaded-profile-p2p-iters", type=int, default=2)
    parser.add_argument(
        "--loaded-profile-p2p-passes",
        type=int,
        default=3,
        help="Independent timed passes per directed pair; the median is used.",
    )
    parser.add_argument("--loaded-profile-p2p-warmup-iters", type=int, default=1)
    parser.add_argument("--loaded-profile-p2p-output", type=str, default=None)
    parser.add_argument(
        "--endpoint-pressure-ranks",
        type=str,
        default="",
        help="Comma-separated global ranks receiving real memory/copy pressure.",
    )
    parser.add_argument(
        "--endpoint-pressure-bytes",
        type=int,
        default=0,
        help="Bytes per pressure buffer; zero disables controlled heterogeneity.",
    )
    parser.add_argument("--endpoint-pressure-iters", type=int, default=1)

    parser.add_argument("--refit-backend", choices=("nccl", "gloo", "nvshmem"), default="nccl")
    parser.add_argument("--trace-transfer-devices", action="store_true")
    parser.add_argument("--require-gpu-direct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-remote-transfer", action="store_true")
    parser.add_argument("--reshard-fusion", choices=("none",), default="none")

    parser.add_argument("--print-intersection-plan", action="store_true")
    parser.add_argument("--intersection-plan-limit", type=int, default=40)
    parser.add_argument("--metrics-output", type=str, default=None)
    parser.add_argument(
        "--audit-migration-payloads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--validation-tolerance", type=float, default=2.0e-5)
    parser.add_argument(
        "--optimizer-validation-tolerance",
        type=float,
        default=3.0e-3,
        help=(
            "Absolute logit tolerance after one optimizer step across different TP layouts. "
            "Migration payloads remain protected by the byte-exact state audit."
        ),
    )
    parser.add_argument(
        "--validate-next-optimizer-step",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.planner_self_test:
        args.active_expert_ids = (0,)
        return
    for name in ("src_tp", "dst_tp", "expert_parallel_size", "num_experts"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.dst_tp <= args.src_tp:
        raise ValueError("This prototype currently supports TP scale-out only")
    if args.num_experts % args.expert_parallel_size != 0:
        raise ValueError("--num-experts must be divisible by --expert-parallel-size")
    if not 1 <= args.moe_router_topk <= args.num_experts:
        raise ValueError("--moe-router-topk must be in [1, num_experts]")
    args.active_expert_ids = _parse_active_experts(args.active_experts, args.num_experts)
    args.endpoint_pressure_rank_ids = tuple(
        sorted(
            {
                int(value.strip())
                for value in args.endpoint_pressure_ranks.split(",")
                if value.strip()
            }
        )
    )
    if args.routing_mode == "fixed-hot" and args.moe_router_topk > len(args.active_expert_ids):
        raise ValueError(
            "--moe-router-topk cannot exceed the number of --active-experts "
            "when --routing-mode=fixed-hot"
        )
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError("--hidden-size must be divisible by --num-attention-heads")
    for tp_size in (args.src_tp, args.dst_tp):
        if args.num_attention_heads % tp_size != 0:
            raise ValueError(f"--num-attention-heads must be divisible by TP={tp_size}")
        if args.num_query_groups % tp_size != 0 and tp_size % args.num_query_groups != 0:
            raise ValueError(f"--num-query-groups must be a multiple or divisor of TP={tp_size}")
    if args.max_waves <= 0 or args.max_wave_tasks <= 0:
        raise ValueError("--max-waves and --max-wave-tasks must be positive")
    if args.scheduler_min_benefit_pct < 0.0:
        raise ValueError("--scheduler-min-benefit-pct must be non-negative")
    if args.scheduler_prediction_uncertainty_pct < 0.0:
        raise ValueError("--scheduler-prediction-uncertainty-pct must be non-negative")
    if args.scheduler_switch_overhead_us < 0.0:
        raise ValueError("--scheduler-switch-overhead-us must be non-negative")
    if args.scheduler_wave_overhead_us < 0.0:
        raise ValueError("--scheduler-wave-overhead-us must be non-negative")
    if args.min_bandwidth_cv <= 0.0:
        raise ValueError("--min-bandwidth-cv must be positive")
    if not 0.0 < args.bandwidth_ewma <= 1.0:
        raise ValueError("--bandwidth-ewma must be in (0, 1]")
    if args.loaded_profile_p2p_bytes <= 0:
        raise ValueError("--loaded-profile-p2p-bytes must be positive")
    if args.loaded_profile_p2p_iters <= 0:
        raise ValueError("--loaded-profile-p2p-iters must be positive")
    if args.loaded_profile_p2p_passes <= 0:
        raise ValueError("--loaded-profile-p2p-passes must be positive")
    if args.loaded_profile_p2p_warmup_iters < 0:
        raise ValueError("--loaded-profile-p2p-warmup-iters must be non-negative")
    if args.profile_p2p_passes <= 0:
        raise ValueError("--profile-p2p-passes must be positive")
    if args.endpoint_pressure_bytes < 0:
        raise ValueError("--endpoint-pressure-bytes must be non-negative")
    if args.endpoint_pressure_iters <= 0:
        raise ValueError("--endpoint-pressure-iters must be positive")
    if args.endpoint_pressure_bytes > 0 and not args.endpoint_pressure_rank_ids:
        raise ValueError("--endpoint-pressure-ranks is required when pressure is enabled")
    if args.source_reroute_min_gain_pct < 0.0:
        raise ValueError("--source-reroute-min-gain-pct must be non-negative")
    if args.source_reroute_min_contention_gain_pct < 0.0:
        raise ValueError(
            "--source-reroute-min-contention-gain-pct must be non-negative"
        )
    if args.source_reroute_min_global_gain_pct < 0.0:
        raise ValueError("--source-reroute-min-global-gain-pct must be non-negative")
    if args.min_link_degradation_pct < 0.0:
        raise ValueError("--min-link-degradation-pct must be non-negative")
    if args.source_reroute_min_bytes < 0:
        raise ValueError("--source-reroute-min-bytes must be non-negative")
    if args.refit_backend != "nccl" and args.require_gpu_direct:
        raise ValueError("--require-gpu-direct currently requires --refit-backend nccl")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)
    if args.planner_self_test:
        _planner_self_test()
        return
    _run(args)


if __name__ == "__main__":
    main()
