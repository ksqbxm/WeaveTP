# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build and optionally swap GPT models with different TP sizes in one process group.

Run examples:
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 1 --dst-tp 4
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 4 --dst-tp 1 --swap
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 1 --dst-tp 4 --optimizer-state
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 1 --dst-tp 4 --distributed-optimizer-state
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 4 --dst-tp 1 --swap --require-gpu-direct
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 1 --dst-tp 4 --swap --zero-copy-gpu
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 1 --dst-tp 4 --swap --zero-copy-gpu --force-remote-transfer
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 1 --dst-tp 4 --swap --reuse-overlap-weights
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --swap --zero-copy-gpu --print-intersection-plan
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-scaleout --print-intersection-plan
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 4 --dst-tp 2 --fluidscale-scalein --print-intersection-plan
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-elastic
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 4 --dst-tp 2 --fluidscale-elastic
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-scaleout --num-layers 32 --hidden-size 2048 --num-attention-heads 32 --num-query-groups 8 --ffn-hidden-size 8192 --seq-len 1024 --vocab-size 32000 --micro-batch-size 1
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-scaleout --live-training-migration --live-train-steps 8 --num-layers 32 --hidden-size 2048 --num-attention-heads 32 --num-query-groups 8 --ffn-hidden-size 8192 --seq-len 1024 --vocab-size 32000 --micro-batch-size 1
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-scaleout --live-training-migration --flexible-tp-plan --flexible-dst-layout 2,1,1,2
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-scaleout --shadow-prep-boundary-transfer
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-scaleout --shadow-prep-boundary-transfer --profile-p2p-bandwidth --bandwidth-aware-rank-placement
  torchrun --nproc_per_node=4 tools/resharding/tp_build_src_dst_demo.py --src-tp 2 --dst-tp 4 --fluidscale-scaleout --live-training-migration --reshard-scheduler bandwidth-aware --scheduler-max-waves 8

This is a development demo for in-process TP resharding. It creates a source model and a
destination model with different ProcessGroupCollection objects, prints their TP parameter
metadata, can verify swap_model_weights by comparing source/destination logits, and can
verify AdamW and Megatron DistributedOptimizer tensor-state resharding.

The --fluidscale-scaleout mode demonstrates checkpoint-free TP scale-out in the style of
FluidScale: the source model lives only on the old active ranks, the destination shadow
model spans the expanded rank set, and parameter migration is executed from source/dest
view intersections over GPU copy services.

The --fluidscale-scalein mode demonstrates the reverse transition: the old active model
spans all launched ranks, the destination shadow model lives only on surviving ranks, and
source-only ranks send their owned parameter slices before being removed.

The --fluidscale-elastic mode chooses scale-out or scale-in automatically from
src_tp/dst_tp, so the demo can run with any launched GPU count that matches the selected
destination TP for scale-out or source TP for scale-in.

Elastic modes also simulate FluidScale's atomic switch: after resharding finishes at an
iteration boundary, the demo updates the current model/process-group reference from the
active world to the shadow world and verifies the next forward pass on the new world.
"""

import argparse
import copy
import gc
import itertools
import json
import math
import os
import statistics
import threading
import time
import traceback
from dataclasses import fields
from typing import Optional

import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import get_megatron_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.resharding import (
    build_centralized_reshard_plan,
    execute_reshard_plan,
    get_or_create_service,
)
from megatron.core.resharding.copy_services import (
    GlooCopyService,
    NCCLCopyService,
    NVSHMEMCopyService,
)
from megatron.core.resharding.copy_services.base import CopyService
from megatron.core.resharding.refit import clear_all_caches, swap_model_weights
from megatron.core.resharding.utils import ReshardPlan
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

GPU_DIRECT_BACKENDS = ("nccl", "nvshmem")
OPTIMIZER_TENSOR_STATE_KEYS = ("exp_avg", "exp_avg_sq")
DISTRIBUTED_OPTIMIZER_TENSOR_STATE_KEYS = ("param", "exp_avg", "exp_avg_sq")
TP_METADATA_ATTRS = (
    "tensor_model_parallel",
    "partition_dim",
    "partition_stride",
    "partition_sizes",
    "allreduce",
)


class _RankPermutedHyperCommGrid(HyperCommGrid):
    """HyperCommGrid whose logical grid positions map to arbitrary global ranks."""

    def __init__(self, shape: list[int], dim_names: list[str], rank_order: list[int]) -> None:
        if len(rank_order) != math.prod(shape):
            raise ValueError(
                f"rank_order has {len(rank_order)} ranks, expected {math.prod(shape)} for {shape}"
            )
        if len(set(rank_order)) != len(rank_order):
            raise ValueError(f"rank_order must not contain duplicates, got {rank_order}")
        world_size = (
            int(os.environ["WORLD_SIZE"])
            if "WORLD_SIZE" in os.environ
            else dist.get_world_size()
        )
        if any(rank < 0 or rank >= world_size for rank in rank_order):
            raise ValueError(
                f"rank_order entries must be in [0, {world_size}), got {rank_order}"
            )
        super().__init__(shape, dim_names, rank_offset=0)
        self.rank_order = list(rank_order)
        self._rank_set = set(rank_order)

    def _gen_rank_enum(self, dims: list[str]) -> list[list[int]]:
        logical_groups = super()._gen_rank_enum(dims)
        return [[self.rank_order[rank] for rank in group] for group in logical_groups]

    def is_current_rank_in_grid(self) -> bool:
        return dist.get_rank() in self._rank_set


class TracingCopyService(CopyService):
    """Wrap a CopyService and verify that submitted tensors stay on CUDA."""

    def __init__(
        self,
        backend: str,
        group=None,
        require_cuda: bool = True,
        require_remote: bool = False,
        p2p_order: Optional[str] = None,
    ) -> None:
        if require_cuda and backend not in GPU_DIRECT_BACKENDS:
            raise ValueError(
                f"--require-gpu-direct needs one of {GPU_DIRECT_BACKENDS}; got {backend!r}"
            )

        self.backend = backend
        self.require_cuda = require_cuda
        self.require_remote = require_remote
        self.group = group
        self.service = make_uncached_copy_service(
            backend, group=group, p2p_order=p2p_order
        )
        self.rank = group.rank() if group is not None else dist.get_rank()

        self.send_count = 0
        self.recv_count = 0
        self.local_bytes = 0
        self.remote_bytes = 0
        self.cpu_tensor_count = 0

    def _record(self, tensor: torch.Tensor, peer_rank: int, is_send: bool) -> None:
        if not tensor.is_cuda:
            self.cpu_tensor_count += 1
            if self.require_cuda:
                direction = "send" if is_send else "recv"
                raise RuntimeError(
                    f"Expected CUDA tensor for {direction}, got device={tensor.device} "
                    f"on rank={self.rank}, peer_rank={peer_rank}"
                )

        num_bytes = tensor.numel() * tensor.element_size()
        if peer_rank == self.rank:
            self.local_bytes += num_bytes
        else:
            self.remote_bytes += num_bytes

    def submit_send(
        self, src_tensor: torch.Tensor, dest_rank: int, task_id: Optional[int] = None
    ) -> None:
        self.send_count += 1
        self._record(src_tensor, dest_rank, is_send=True)
        self.service.submit_send(src_tensor, dest_rank, task_id=task_id)

    def submit_recv(
        self, dest_tensor: torch.Tensor, src_rank: int, task_id: Optional[int] = None
    ) -> None:
        self.recv_count += 1
        self._record(dest_tensor, src_rank, is_send=False)
        self.service.submit_recv(dest_tensor, src_rank, task_id=task_id)

    def run(self) -> None:
        self.service.run()

    def set_launch_callback(self, callback) -> None:
        if hasattr(self.service, "set_launch_callback"):
            self.service.set_launch_callback(callback)
        else:
            callback()

    def print_summary(self, label: str) -> None:
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        stats = torch.tensor(
            [
                self.send_count,
                self.recv_count,
                self.local_bytes,
                self.remote_bytes,
                self.cpu_tensor_count,
            ],
            device=device,
            dtype=torch.long,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=self.group)

        if self.rank == 0:
            transport = "cpu-staged" if self.backend == "gloo" else "gpu-to-gpu"
            print(
                f"\nTransfer trace for {label} "
                f"(backend={self.backend}, transport={transport}, "
                f"require_cuda={self.require_cuda}):",
                flush=True,
            )
            print(
                f"  sends={stats[0].item()} recvs={stats[1].item()} "
                f"local_bytes={stats[2].item()} remote_bytes={stats[3].item()} "
                f"cpu_tensors_seen={stats[4].item()}",
                flush=True,
            )
            if self.require_cuda:
                print("  zero-copy GPU path check: passed (no CPU tensors submitted)", flush=True)
            if self.require_remote:
                if stats[3].item() <= 0:
                    raise RuntimeError(
                        f"{label}: expected at least one remote GPU-GPU transfer, "
                        "but remote_bytes=0"
                    )
                print("  remote GPU-GPU transfer check: passed", flush=True)


def make_uncached_copy_service(
    backend: str, group=None, p2p_order: Optional[str] = None
) -> CopyService:
    if backend == "nccl":
        return NCCLCopyService(group=group, p2p_order=p2p_order)
    if backend == "gloo":
        return GlooCopyService(group=group)
    if backend == "nvshmem":
        return NVSHMEMCopyService(group=group)
    raise ValueError(f"Unknown backend '{backend}'")


def make_refit_method(args: argparse.Namespace, group=None) -> str | CopyService:
    if args.trace_transfer_devices or args.require_gpu_direct:
        return TracingCopyService(
            args.refit_backend,
            group=group,
            require_cuda=args.require_gpu_direct,
            require_remote=args.force_remote_transfer,
            p2p_order=getattr(args, "nccl_copy_p2p_order", None),
        )
    if args.refit_backend == "nccl" and getattr(args, "nccl_copy_p2p_order", None):
        return make_uncached_copy_service(
            args.refit_backend,
            group=group,
            p2p_order=args.nccl_copy_p2p_order,
        )
    return args.refit_backend


def make_plan_copy_service(
    args: argparse.Namespace, group=None, *, uncached: bool = False
) -> CopyService:
    if uncached and not (args.trace_transfer_devices or args.require_gpu_direct):
        return make_uncached_copy_service(
            args.refit_backend,
            group=group,
            p2p_order=getattr(args, "nccl_copy_p2p_order", None),
        )
    refit_method = make_refit_method(args, group=group)
    if isinstance(refit_method, str):
        return get_or_create_service(refit_method, group=group)
    return refit_method


def enable_zero_copy_gpu_mode(args: argparse.Namespace) -> None:
    """Turn on the stricter CPU-free GPU transfer checks used by this demo."""
    if args.force_remote_transfer:
        args.zero_copy_gpu = True
    if not args.zero_copy_gpu:
        return
    if args.refit_backend not in GPU_DIRECT_BACKENDS:
        raise ValueError(
            f"--zero-copy-gpu requires one of {GPU_DIRECT_BACKENDS}; "
            f"got --refit-backend={args.refit_backend!r}"
        )
    args.trace_transfer_devices = True
    args.require_gpu_direct = True

    if dist.get_rank() == 0:
        print(
            "\nZero-copy GPU mode enabled: parameter/state payloads must stay on CUDA "
            f"and use backend={args.refit_backend}.",
            flush=True,
        )
        if args.force_remote_transfer:
            print(
                "Remote transfer check enabled: planner will avoid same-rank source "
                "selection when another source replica is available.",
                flush=True,
            )


def prefer_local_source(args: argparse.Namespace) -> bool:
    return not args.force_remote_transfer


def reduce_max_seconds(value: float) -> float:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def print_path_timing(label: str, **metrics: float) -> None:
    if dist.get_rank() != 0:
        return
    metric_text = " ".join(f"{name}={value:.6f}" for name, value in metrics.items())
    print(f"\nTiming summary for {label}: {metric_text}", flush=True)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    index = (len(values) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    return values[lower] * (upper - index) + values[upper] * (index - lower)


def _bandwidth_matrix_stats(
    matrix: dict[tuple[int, int], float],
    world_size: int,
) -> dict[str, float]:
    remote_values = [
        matrix[(src, dst)]
        for src in range(world_size)
        for dst in range(world_size)
        if src != dst and (src, dst) in matrix
    ]
    if not remote_values:
        return {
            "remote_min_gbps": 0.0,
            "remote_max_gbps": 0.0,
            "remote_mean_gbps": 0.0,
            "remote_p50_gbps": 0.0,
            "remote_p95_gbps": 0.0,
            "remote_cv": 0.0,
            "slow_link_ratio_2x": 0.0,
        }
    mean = sum(remote_values) / len(remote_values)
    variance = sum((value - mean) ** 2 for value in remote_values) / len(remote_values)
    min_value = min(remote_values)
    max_value = max(remote_values)
    slow_cutoff = max_value / 2.0
    slow_ratio = sum(1 for value in remote_values if value <= slow_cutoff) / len(remote_values)
    return {
        "remote_min_gbps": min_value,
        "remote_max_gbps": max_value,
        "remote_mean_gbps": mean,
        "remote_p50_gbps": _percentile(remote_values, 50.0),
        "remote_p95_gbps": _percentile(remote_values, 95.0),
        "remote_cv": math.sqrt(variance) / mean if mean > 0.0 else 0.0,
        "slow_link_ratio_2x": slow_ratio,
    }


def _rank_metadata() -> list[dict]:
    local = {
        "rank": dist.get_rank(),
        "hostname": os.uname().nodename,
        "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", ""),
    }
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    return [item for item in gathered if item is not None]


def _print_bandwidth_matrix_summary(
    matrix: dict[tuple[int, int], float],
    world_size: int,
    *,
    prefix: str,
) -> None:
    if dist.get_rank() != 0:
        return
    stats = _bandwidth_matrix_stats(matrix, world_size)
    print(
        f"\n{prefix} bandwidth summary: "
        f"remote_min={stats['remote_min_gbps']:.2f}Gbps "
        f"remote_p50={stats['remote_p50_gbps']:.2f}Gbps "
        f"remote_p95={stats['remote_p95_gbps']:.2f}Gbps "
        f"remote_max={stats['remote_max_gbps']:.2f}Gbps "
        f"remote_mean={stats['remote_mean_gbps']:.2f}Gbps "
        f"remote_cv={stats['remote_cv']:.4f} "
        f"slow_link_ratio_2x={stats['slow_link_ratio_2x']:.4f}",
        flush=True,
    )


def _save_bandwidth_profile(
    matrix: dict[tuple[int, int], float],
    args: argparse.Namespace,
) -> None:
    if not args.profile_p2p_output:
        return
    rank_metadata = _rank_metadata()
    if dist.get_rank() != 0:
        return
    world_size = dist.get_world_size()
    payload = {
        "world_size": world_size,
        "unit": "Gbps",
        "payload_bytes": args.profile_p2p_bytes,
        "iters": args.profile_p2p_iters,
        "passes": int(getattr(args, "profile_p2p_passes", 1)),
        "warmup_iters": args.profile_p2p_warmup_iters,
        "ranks": rank_metadata,
        "stats": _bandwidth_matrix_stats(matrix, world_size),
        "matrix_gbps": [
            [matrix.get((src, dst), 0.0) for dst in range(world_size)]
            for src in range(world_size)
        ],
    }
    output_dir = os.path.dirname(os.path.abspath(args.profile_p2p_output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.profile_p2p_output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved P2P bandwidth profile to {args.profile_p2p_output}", flush=True)


def load_bandwidth_profile(
    path: str,
    *,
    expected_world_size: Optional[int] = None,
) -> dict[tuple[int, int], float]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    matrix_payload = payload.get("matrix_gbps", [])
    profile_world_size = int(payload.get("world_size", len(matrix_payload)))
    if expected_world_size is not None and profile_world_size != expected_world_size:
        raise ValueError(
            f"Bandwidth profile world_size={profile_world_size} does not match "
            f"current WORLD_SIZE={expected_world_size}: {path}"
        )
    matrix: dict[tuple[int, int], float] = {}
    for src, row in enumerate(matrix_payload):
        for dst, value in enumerate(row):
            matrix[(src, dst)] = float(value)
    return matrix


def _default_bandwidth_matrix(args: argparse.Namespace, world_size: int) -> dict[tuple[int, int], float]:
    matrix: dict[tuple[int, int], float] = {}
    for src in range(world_size):
        for dst in range(world_size):
            matrix[(src, dst)] = (
                args.scheduler_local_bandwidth_gbps
                if src == dst
                else args.scheduler_remote_bandwidth_gbps
            )
    return matrix


def _parse_slow_link_spec(spec: str) -> list[tuple[int, int, str, float]]:
    """Parse src:dst=gbps or src:dst*factor entries."""
    entries: list[tuple[int, int, str, float]] = []
    if not spec:
        return entries
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            pair_text, value_text = item.split("=", 1)
            mode = "set"
        elif "*" in item:
            pair_text, value_text = item.split("*", 1)
            mode = "mul"
        else:
            raise ValueError(
                "--scheduler-slow-links entries must look like src:dst=gbps "
                f"or src:dst*factor; got {item!r}"
            )
        pair_text = pair_text.replace("->", ":")
        parts = pair_text.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid slow-link pair {pair_text!r}; expected src:dst")
        src = int(parts[0])
        dst = int(parts[1])
        value = float(value_text)
        if value <= 0.0:
            raise ValueError(f"Slow-link value must be positive in {item!r}")
        entries.append((src, dst, mode, value))
    return entries


def apply_scheduler_slow_links(
    matrix: Optional[dict[tuple[int, int], float]],
    args: argparse.Namespace,
    world_size: int,
) -> Optional[dict[tuple[int, int], float]]:
    """Inject artificial heterogeneity into the scheduler bandwidth matrix.

    This supports Experiment 2: compare static ordering/packing against a
    scheduler that has a real B[src][dst] cost model when selected links are
    intentionally slow.  The same parsed map is reused by the optional runtime
    delay injector so the measured transfer path also sees the synthetic slow
    links.
    """
    entries = _parse_slow_link_spec(args.scheduler_slow_links)
    args.scheduler_slow_link_gbps = {}
    if not entries:
        return matrix
    if matrix is None:
        matrix = _default_bandwidth_matrix(args, world_size)
    else:
        matrix = dict(matrix)

    updates = []
    for src, dst, mode, value in entries:
        if src < 0 or src >= world_size or dst < 0 or dst >= world_size:
            raise ValueError(
                f"Slow-link pair {src}:{dst} is outside WORLD_SIZE={world_size}"
            )
        old = float(matrix.get((src, dst), args.scheduler_remote_bandwidth_gbps))
        new = value if mode == "set" else old * value
        matrix[(src, dst)] = max(new, 1.0e-9)
        args.scheduler_slow_link_gbps[(src, dst)] = matrix[(src, dst)]
        updates.append((src, dst, old, matrix[(src, dst)], mode, value))

    if dist.get_rank() == 0:
        print("\nInjected scheduler slow links:", flush=True)
        for src, dst, old, new, mode, value in updates:
            detail = f"set {value:.6g}Gbps" if mode == "set" else f"multiply by {value:.6g}"
            print(
                f"  link {src}->{dst}: {old:.3f}Gbps -> {new:.3f}Gbps ({detail})",
                flush=True,
            )
    _print_bandwidth_matrix_summary(matrix, world_size, prefix="Effective scheduler")
    return matrix


def profile_p2p_bandwidth_matrix(args: argparse.Namespace) -> dict[tuple[int, int], float]:
    """Measure pairwise GPU P2P bandwidth for scheduler cost estimation."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    num_bytes = args.profile_p2p_bytes
    timed_passes = int(getattr(args, "profile_p2p_passes", 1))
    max_iters = max(args.profile_p2p_warmup_iters, args.profile_p2p_iters)
    send_buffers = [
        torch.empty(num_bytes, device=device, dtype=torch.uint8) for _ in range(max_iters)
    ]
    recv_buffers = [
        torch.empty(num_bytes, device=device, dtype=torch.uint8) for _ in range(max_iters)
    ]
    probe_stream = torch.cuda.Stream(device=device)
    matrix: dict[tuple[int, int], float] = {}

    # Initialize the full communicator before pairwise batch-P2P calls. NCCL
    # requires every rank to participate when a process group is first used.
    warmup_token = torch.ones(1, device=device)
    dist.all_reduce(warmup_token)
    torch.cuda.synchronize(device)
    dist.barrier()

    def run_probe(src: int, dst: int, iters: int) -> float:
        dist.barrier()
        if rank in (src, dst):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        p2p_ops = []
        with torch.cuda.stream(probe_stream):
            if rank == src:
                p2p_ops = [
                    dist.P2POp(dist.isend, send_buffers[index], dst)
                    for index in range(iters)
                ]
            elif rank == dst:
                p2p_ops = [
                    dist.P2POp(dist.irecv, recv_buffers[index], src)
                    for index in range(iters)
                ]
            requests = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []
        for request in requests:
            request.wait()
        if rank in (src, dst):
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start if rank in (src, dst) else 0.0
        elapsed_tensor = torch.tensor(elapsed, device=device, dtype=torch.float64)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        return max(float(elapsed_tensor.item()), 1.0e-9)

    for src in range(world_size):
        for dst in range(world_size):
            if src == dst:
                matrix[(src, dst)] = args.scheduler_local_bandwidth_gbps
                continue

            if args.profile_p2p_warmup_iters:
                run_probe(src, dst, args.profile_p2p_warmup_iters)
            elapsed_samples = [
                run_probe(src, dst, args.profile_p2p_iters)
                for _ in range(timed_passes)
            ]
            elapsed_s = statistics.median(elapsed_samples)
            gbps = (num_bytes * args.profile_p2p_iters * 8.0) / elapsed_s / 1.0e9
            matrix[(src, dst)] = max(gbps, 1.0e-9)

    dist.barrier()
    if rank == 0:
        print(
            "\nProfiled P2P bandwidth matrix for scheduler "
            f"(payload={num_bytes} bytes, iters={args.profile_p2p_iters}, "
            f"passes={timed_passes}, unit=Gbps):",
            flush=True,
        )
        header = "src\\dst".ljust(10) + "".join(f"{dst:>10d}" for dst in range(world_size))
        print("  " + header, flush=True)
        for src in range(world_size):
            cells = "".join(f"{matrix[(src, dst)]:>10.1f}" for dst in range(world_size))
            print("  " + f"{src:<10d}" + cells, flush=True)
    _print_bandwidth_matrix_summary(matrix, world_size, prefix="Profiled P2P")
    _save_bandwidth_profile(matrix, args)

    return matrix


class OptimizerStateTensorModule(torch.nn.Module):
    """Expose optimizer state tensors through named_parameters for reshard planning."""

    def __init__(self, pg_collection: ProcessGroupCollection, named_state_params):
        super().__init__()
        self.pg_collection = pg_collection
        self._named_state_params = named_state_params

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
        name_prefix = f"{prefix}." if prefix else ""
        for name, param in self._named_state_params:
            yield name_prefix + name, param


def init_distributed_and_mpu(src_tp: int) -> None:
    """Initialize distributed state once.

    The global MPU is initialized with src_tp to satisfy Megatron global state. The source and
    destination models below use their own ProcessGroupCollection objects.
    """
    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))

    mpu.destroy_model_parallel()
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=src_tp,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )


def build_pg_collection(
    tp_size: int,
    pp_size: int = 1,
    ep_size: int = 1,
    *,
    rank_count: Optional[int] = None,
    rank_offset: int = 0,
    rank_order: Optional[list[int]] = None,
) -> ProcessGroupCollection:
    """Build a model-local process-group collection for a chosen TP size.

    rank_count/rank_offset let the demo model an old active world that covers
    only a prefix of the launched ranks. rank_order maps logical grid positions
    to arbitrary physical global ranks for bandwidth-aware destination placement.
    """
    cp_size = 1
    if rank_order is not None:
        if rank_count is not None and rank_count != len(rank_order):
            raise ValueError(
                f"rank_count={rank_count} does not match rank_order size={len(rank_order)}"
            )
        if rank_offset != 0:
            raise ValueError("rank_offset cannot be combined with an explicit rank_order")
        grid_world_size = len(rank_order)
    else:
        grid_world_size = dist.get_world_size() if rank_count is None else rank_count
    dp_size = grid_world_size // (tp_size * cp_size * ep_size * pp_size)

    assert dp_size >= 1, f"Invalid dp_size={dp_size}"
    assert tp_size * cp_size * ep_size * pp_size * dp_size == grid_world_size, (
        f"rank_count={grid_world_size} must be divisible by "
        f"tp*cp*ep*pp={tp_size}*{cp_size}*{ep_size}*{pp_size}"
    )

    grid_shape = [tp_size, cp_size, ep_size, pp_size, dp_size]
    grid_dims = ["tp", "cp", "ep", "pp", "dp"]
    if rank_order is None:
        grid = HyperCommGrid(grid_shape, grid_dims, rank_offset=rank_offset)
    else:
        grid = _RankPermutedHyperCommGrid(grid_shape, grid_dims, rank_order)

    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pp_group = grid.create_pg("pp")
    ep_group = grid.create_pg("ep")
    dp_group = grid.create_pg("dp")

    tp_cp_group = grid.create_pg(["tp", "cp"])
    mp_group = grid.create_pg(["tp", "cp", "ep", "pp"])
    tp_ep_group = grid.create_pg(["tp", "ep"])
    tp_ep_pp_group = grid.create_pg(["tp", "ep", "pp"])
    dp_cp_group = grid.create_pg(["cp", "dp"])
    tp_dp_cp_group = grid.create_pg(["tp", "cp", "dp"])

    if grid.is_current_rank_in_grid() and pp_group != dist.GroupMember.NON_GROUP_MEMBER:
        pp_ranks = dist.get_process_group_ranks(pp_group)
    else:
        pp_ranks = grid.get_rank_enum("pp")[0]
    embd_group = dist.new_group(ranks=mpu.default_embedding_ranks(pp_ranks))
    pos_embd_group = dist.new_group(ranks=mpu.default_position_embedding_ranks(pp_ranks))

    return ProcessGroupCollection(
        tp=tp_group,
        cp=cp_group,
        pp=pp_group,
        ep=ep_group,
        embd=embd_group,
        pos_embd=pos_embd_group,
        dp=dp_group,
        tp_cp=tp_cp_group,
        mp=mp_group,
        expt_tp=tp_group,
        expt_dp=dp_group,
        tp_ep=tp_ep_group,
        tp_ep_pp=tp_ep_pp_group,
        dp_cp=dp_cp_group,
        tp_dp_cp=tp_dp_cp_group,
        dp_cp_ag=None,
        expt_dp_ag=None,
        intra_dp_cp=dp_cp_group,
        intra_expt_dp=dp_group,
        inter_dist_opt=None,
        intra_dist_opt=dp_cp_group,
    )


def destroy_pg_collection(pgc: ProcessGroupCollection) -> None:
    """Destroy custom process groups created for a model."""
    destroyed = set()
    for field in fields(pgc):
        if not hasattr(pgc, field.name):
            continue
        pg = getattr(pgc, field.name)
        if pg is None or pg == dist.GroupMember.NON_GROUP_MEMBER or id(pg) in destroyed:
            continue
        destroyed.add(id(pg))
        dist.destroy_process_group(pg)


def pp_flags(pg_collection: ProcessGroupCollection) -> tuple[bool, bool]:
    pp_rank = dist.get_rank(pg_collection.pp)
    pp_size = dist.get_world_size(pg_collection.pp)
    return pp_rank == 0, pp_rank == pp_size - 1


def make_config(tp_size: int, args: argparse.Namespace) -> TransformerConfig:
    """Build the GPT config used by the demo.

    Defaults intentionally keep the original small model, while CLI options allow
    larger models for transfer-throughput and memory-pressure testing.
    """
    return TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=args.num_query_groups,
        ffn_hidden_size=args.ffn_hidden_size,
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )


def build_gpt_model(
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
    args: argparse.Namespace,
) -> GPTModel:
    pre_process, post_process = pp_flags(pg_collection)

    return GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
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


def print_param_summary(
    label: str,
    model: Optional[torch.nn.Module],
    pg_collection: Optional[ProcessGroupCollection],
    participating_ranks: Optional[set[int]] = None,
) -> None:
    rank = dist.get_rank()
    should_print = model is not None and (
        participating_ranks is None or rank in participating_ranks
    )
    if should_print:
        tp_rank = dist.get_rank(pg_collection.tp)
        tp_size = dist.get_world_size(pg_collection.tp)
    else:
        tp_rank = -1
        tp_size = 0

    dist.barrier()
    for global_rank in range(dist.get_world_size()):
        if rank == global_rank and should_print:
            print(
                f"\n===== {label} | global_rank={rank} tp_rank={tp_rank}/{tp_size} =====",
                flush=True,
            )
            for name, param in model.named_parameters():
                is_tp = bool(getattr(param, "tensor_model_parallel", False))
                partition_dim = getattr(param, "partition_dim", None)
                partition_stride = getattr(param, "partition_stride", None)

                interesting = (
                    "linear_qkv.weight" in name
                    or "linear_qkv.bias" in name
                    or "linear_proj.weight" in name
                    or "linear_fc1.weight" in name
                    or "linear_fc1.bias" in name
                    or "linear_fc2.weight" in name
                    or "embedding" in name
                    or "layernorm" in name.lower()
                    or "norm" in name.lower()
                )
                if not interesting:
                    continue

                print(
                    f"{name:80s} shape={tuple(param.shape)!s:18s} "
                    f"is_tp={is_tp:<5} dim={partition_dim} stride={partition_stride}",
                    flush=True,
                )

        dist.barrier()


def _stable_name_offset(name: str, state_key: str) -> float:
    value = 0
    for index, byte in enumerate(f"{name}:{state_key}".encode("utf-8")):
        value += (index + 1) * byte
    return float(value % 10000) / 1000.0


def _copy_tp_metadata(src_param: torch.nn.Parameter, dst_param: torch.nn.Parameter) -> None:
    for attr in TP_METADATA_ATTRS:
        if hasattr(src_param, attr):
            setattr(dst_param, attr, getattr(src_param, attr))


def _is_full_slice(index: tuple[slice, ...], shape: tuple[int, ...]) -> bool:
    if len(index) != len(shape):
        return False
    for item, dim_size in zip(index, shape):
        if not isinstance(item, slice):
            return False
        start = 0 if item.start is None else item.start
        stop = dim_size if item.stop is None else item.stop
        step = 1 if item.step is None else item.step
        if start != 0 or stop != dim_size or step != 1:
            return False
    return True


def _get_parent_module(root: torch.nn.Module, param_name: str) -> tuple[torch.nn.Module, str]:
    parts = param_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _copy_parameter_attrs(src_param: torch.nn.Parameter, dst_param: torch.nn.Parameter) -> None:
    for attr, value in getattr(src_param, "__dict__", {}).items():
        setattr(dst_param, attr, value)


def _replace_parameter_with_view(
    module: torch.nn.Module,
    param_name: str,
    src_view: torch.Tensor,
    old_dst_param: torch.nn.Parameter,
) -> torch.nn.Parameter:
    parent, leaf_name = _get_parent_module(module, param_name)
    new_param = torch.nn.Parameter(src_view, requires_grad=old_dst_param.requires_grad)
    _copy_parameter_attrs(old_dst_param, new_param)
    setattr(parent, leaf_name, new_param)
    return new_param


def _filter_reshard_plan(plan: ReshardPlan, skipped_task_ids: set[int]) -> ReshardPlan:
    if not skipped_task_ids:
        return plan
    return ReshardPlan(
        send_ops=[op for op in plan.send_ops if op.task_id not in skipped_task_ids],
        recv_ops=[op for op in plan.recv_ops if op.task_id not in skipped_task_ids],
    )


def _filter_reshard_plan_by_param_names(plan: ReshardPlan, param_names: set[str]) -> ReshardPlan:
    if not param_names:
        return ReshardPlan(send_ops=[], recv_ops=[])
    return ReshardPlan(
        send_ops=[op for op in plan.send_ops if op.param_name in param_names],
        recv_ops=[op for op in plan.recv_ops if op.param_name in param_names],
    )


def _filter_reshard_plan_by_task_ids(plan: ReshardPlan, task_ids: set[int]) -> ReshardPlan:
    if not task_ids:
        return ReshardPlan(send_ops=[], recv_ops=[])
    return ReshardPlan(
        send_ops=[op for op in plan.send_ops if op.task_id in task_ids],
        recv_ops=[op for op in plan.recv_ops if op.task_id in task_ids],
    )


def _reorder_reshard_plan_by_task_ids(
    plan: ReshardPlan,
    ordered_task_ids: list[int],
) -> ReshardPlan:
    if not ordered_task_ids:
        return plan

    task_order = {task_id: index for index, task_id in enumerate(ordered_task_ids)}

    def _sort_key(indexed_op):
        index, op = indexed_op
        return (task_order.get(op.task_id, len(task_order)), index)

    return ReshardPlan(
        send_ops=[op for _, op in sorted(enumerate(plan.send_ops), key=_sort_key)],
        recv_ops=[op for _, op in sorted(enumerate(plan.recv_ops), key=_sort_key)],
    )


def _format_slice(index: tuple[slice, ...]) -> str:
    parts = []
    for item in index:
        if not isinstance(item, slice):
            parts.append(str(item))
            continue
        start = "" if item.start is None else str(item.start)
        stop = "" if item.stop is None else str(item.stop)
        step = "" if item.step is None else str(item.step)
        parts.append(f"{start}:{stop}" if not step else f"{start}:{stop}:{step}")
    return "[" + ", ".join(parts) + "]"


def _slice_numel(shape: tuple[int, ...], index: tuple[slice, ...]) -> int:
    numel = 1
    for dim_size, item in zip(shape, index):
        if not isinstance(item, slice):
            continue
        start, stop, step = item.indices(dim_size)
        if step <= 0:
            raise RuntimeError(f"Only positive-step slices are supported, got {item}")
        length = max(0, (stop - start + step - 1) // step)
        numel *= length
    return numel


def print_intersection_plan(
    plan: ReshardPlan,
    dst_model: Optional[torch.nn.Module],
    *,
    label: str,
    limit: int,
) -> None:
    """Print the actual view-intersection transfer tasks used by the reshard plan."""
    rank = dist.get_rank()
    dst_params = dict(dst_model.named_parameters()) if dst_model is not None else {}
    local_rows = []

    for op in plan.recv_ops:
        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        bytes_count = _slice_numel(tuple(dst_param.shape), op.my_slice) * dst_param.element_size()
        local_rows.append(
            {
                "param": op.param_name,
                "src_rank": op.peer_rank,
                "dst_rank": rank,
                "src_slice": _format_slice(op.peer_slice),
                "dst_slice": _format_slice(op.my_slice),
                "bytes": bytes_count,
                "remote": op.peer_rank != rank,
                "task_id": op.task_id,
            }
        )

    gathered_rows = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_rows, gathered_rows, group_dst=0)

    if rank != 0:
        return

    rows = [row for rank_rows in gathered_rows for row in rank_rows]
    rows.sort(key=lambda row: (row["param"], row["dst_rank"], row["src_rank"], row["dst_slice"]))
    total_bytes = sum(row["bytes"] for row in rows)
    remote_bytes = sum(row["bytes"] for row in rows if row["remote"])
    local_bytes = total_bytes - remote_bytes

    print(f"\nIntersection transfer plan for {label}:", flush=True)
    print(
        f"  tasks={len(rows)} total_bytes={total_bytes} "
        f"local_bytes={local_bytes} remote_bytes={remote_bytes}",
        flush=True,
    )

    shown_rows = rows if limit <= 0 else rows[:limit]
    for row in shown_rows:
        locality = "remote" if row["remote"] else "local"
        print(
            f"  task={row['task_id']} dst_rank={row['dst_rank']} <- "
            f"src_rank={row['src_rank']} {locality} bytes={row['bytes']} "
            f"param={row['param']} src={row['src_slice']} dst={row['dst_slice']}",
            flush=True,
        )
    if limit > 0 and len(rows) > limit:
        print(f"  ... truncated {len(rows) - limit} tasks; use --intersection-plan-limit 0", flush=True)


def print_view_overlap_matrix(
    plan: ReshardPlan,
    dst_model: Optional[torch.nn.Module],
    *,
    label: str,
    src_ranks: list[int],
    dst_ranks: list[int],
) -> None:
    """Print aggregate bytes for the source-view/destination-view intersections."""
    rank = dist.get_rank()
    dst_params = dict(dst_model.named_parameters()) if dst_model is not None else {}
    local_cells: dict[tuple[int, int], int] = {}
    for op in plan.recv_ops:
        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        bytes_count = _slice_numel(tuple(dst_param.shape), op.my_slice) * dst_param.element_size()
        key = (op.peer_rank, rank)
        local_cells[key] = local_cells.get(key, 0) + bytes_count

    gathered_cells = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_cells, gathered_cells, group_dst=0)
    if rank != 0:
        return

    matrix: dict[tuple[int, int], int] = {}
    for rank_cells in gathered_cells:
        for key, value in rank_cells.items():
            matrix[key] = matrix.get(key, 0) + value

    total_bytes = sum(matrix.values())
    remote_bytes = sum(value for (src, dst), value in matrix.items() if src != dst)
    print(f"\nView-overlap transfer matrix for {label}:", flush=True)
    print(
        f"  source_ranks={src_ranks} destination_ranks={dst_ranks} "
        f"total_bytes={total_bytes} remote_bytes={remote_bytes}",
        flush=True,
    )
    header = "src\\dst".ljust(10) + "".join(f"{dst:>12d}" for dst in dst_ranks)
    print("  " + header, flush=True)
    for src in src_ranks:
        cells = "".join(f"{matrix.get((src, dst), 0):>12d}" for dst in dst_ranks)
        print("  " + f"{src:<10d}" + cells, flush=True)


def _intervals_from_weights(total_size: int, ranks: list[int], weights: list[float]) -> dict[int, tuple[int, int]]:
    if len(ranks) != len(weights):
        raise ValueError(f"Expected {len(ranks)} layout weights, got {len(weights)}")
    if total_size < 0:
        raise ValueError(f"total_size must be non-negative, got {total_size}")
    if any(weight < 0 for weight in weights):
        raise ValueError(f"Flexible TP layout weights must be non-negative, got {weights}")
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("Flexible TP layout weights must sum to a positive value")

    raw_bounds = [0]
    running = 0.0
    for weight in weights[:-1]:
        running += weight
        raw_bounds.append(int(round(total_size * running / weight_sum)))
    raw_bounds.append(total_size)

    bounds = [0]
    for bound in raw_bounds[1:-1]:
        bounds.append(min(max(bound, bounds[-1]), total_size))
    bounds.append(total_size)

    return {rank: (bounds[i], bounds[i + 1]) for i, rank in enumerate(ranks)}


def _uniform_layout_weights(rank_count: int) -> list[float]:
    return [1.0] * rank_count


def _parse_flexible_layout_weights(spec: Optional[str], rank_count: int) -> list[float]:
    if spec is None:
        return _uniform_layout_weights(rank_count)
    items = [item.strip() for item in spec.split(",") if item.strip()]
    if len(items) != rank_count:
        raise ValueError(
            f"Flexible TP layout expects {rank_count} comma-separated weights, got {len(items)}: "
            f"{spec!r}"
        )
    return [float(item) for item in items]


def _rank_interval_intersection(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[int, int] | None:
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    return (start, end) if start < end else None


def _plan_transfer_matrix(
    plan: ReshardPlan,
    dst_model: Optional[torch.nn.Module],
    *,
    tp_only: bool = False,
) -> dict[tuple[int, int], int]:
    rank = dist.get_rank()
    dst_params = dict(dst_model.named_parameters()) if dst_model is not None else {}
    local_cells: dict[tuple[int, int], int] = {}
    for op in plan.recv_ops:
        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        if tp_only and not bool(getattr(dst_param, "tensor_model_parallel", False)):
            continue
        bytes_count = _slice_numel(tuple(dst_param.shape), op.my_slice) * dst_param.element_size()
        key = (op.peer_rank, rank)
        local_cells[key] = local_cells.get(key, 0) + bytes_count

    gathered_cells = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_cells, gathered_cells, group_dst=0)
    if rank != 0:
        return {}

    matrix: dict[tuple[int, int], int] = {}
    for rank_cells in gathered_cells:
        for key, value in rank_cells.items():
            matrix[key] = matrix.get(key, 0) + value
    return matrix


def _estimate_flexible_tp_param_bytes(
    param: torch.nn.Parameter,
    *,
    src_ranks: list[int],
    dst_ranks: list[int],
    src_weights: list[float],
    dst_weights: list[float],
) -> tuple[dict[tuple[int, int], int], int]:
    if getattr(param, "partition_sizes", None) is not None:
        return {}, 0
    if not bool(getattr(param, "tensor_model_parallel", False)):
        return {}, param.numel() * param.element_size()

    partition_dim = int(getattr(param, "partition_dim", 0))
    local_dim = int(param.shape[partition_dim])
    src_total_dim = local_dim * len(src_ranks)
    other_numel = param.numel() // max(local_dim, 1)
    elem_size = param.element_size()
    src_intervals = _intervals_from_weights(src_total_dim, src_ranks, src_weights)
    dst_intervals = _intervals_from_weights(src_total_dim, dst_ranks, dst_weights)

    cells: dict[tuple[int, int], int] = {}
    for src_rank, src_interval in src_intervals.items():
        for dst_rank, dst_interval in dst_intervals.items():
            overlap = _rank_interval_intersection(src_interval, dst_interval)
            if overlap is None:
                continue
            bytes_count = (overlap[1] - overlap[0]) * other_numel * elem_size
            cells[(src_rank, dst_rank)] = cells.get((src_rank, dst_rank), 0) + bytes_count
    return cells, 0


def _add_cells(dst: dict[tuple[int, int], int], src: dict[tuple[int, int], int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + value


def print_flexible_tp_plan_analysis(
    uniform_plan: ReshardPlan,
    src_model: Optional[torch.nn.Module],
    dst_model: Optional[torch.nn.Module],
    args: argparse.Namespace,
    *,
    src_ranks: list[int],
    dst_ranks: list[int],
    label: str,
) -> None:
    """Compare current uniform TP reshard bytes with a Flexible TP interval layout.

    This is analysis-only. It estimates how much movement a variable-width TP
    target layout would require, but does not alter Megatron parameter shapes or
    execute the flexible layout.
    """
    rank = dist.get_rank()
    uniform_matrix = _plan_transfer_matrix(uniform_plan, dst_model, tp_only=True)

    local_flexible_cells: dict[tuple[int, int], int] = {}
    local_skipped_bytes = 0
    model_for_params = src_model
    src_weights = _parse_flexible_layout_weights(args.flexible_src_layout, len(src_ranks))
    dst_weights = _parse_flexible_layout_weights(args.flexible_dst_layout, len(dst_ranks))
    if model_for_params is not None:
        for _, param in model_for_params.named_parameters():
            cells, skipped_bytes = _estimate_flexible_tp_param_bytes(
                param,
                src_ranks=src_ranks,
                dst_ranks=dst_ranks,
                src_weights=src_weights,
                dst_weights=dst_weights,
            )
            _add_cells(local_flexible_cells, cells)
            local_skipped_bytes += skipped_bytes

    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object((local_flexible_cells, local_skipped_bytes), gathered, group_dst=0)
    if rank != 0:
        return

    flexible_matrix: dict[tuple[int, int], int] = {}
    skipped_bytes = 0
    for cells, skipped in gathered:
        _add_cells(flexible_matrix, cells)
        skipped_bytes += skipped

    def _summarize(matrix: dict[tuple[int, int], int]) -> tuple[int, int, int]:
        total = sum(matrix.values())
        remote = sum(value for (src, dst), value in matrix.items() if src != dst)
        return total, total - remote, remote

    uniform_total, uniform_local, uniform_remote = _summarize(uniform_matrix)
    flexible_total, flexible_local, flexible_remote = _summarize(flexible_matrix)
    remote_reduction = (
        0.0
        if uniform_remote == 0
        else 100.0 * (uniform_remote - flexible_remote) / uniform_remote
    )
    total_reduction = (
        0.0
        if uniform_total == 0
        else 100.0 * (uniform_total - flexible_total) / uniform_total
    )

    print(f"\nFlexible TP planner analysis for {label}:", flush=True)
    print(
        "  This is TP-parameter-only and analysis-only: it estimates variable-width "
        "TP intervals but does not execute uneven TP kernels.",
        flush=True,
    )
    print(
        f"  src_ranks={src_ranks} src_weights={src_weights} "
        f"dst_ranks={dst_ranks} dst_weights={dst_weights}",
        flush=True,
    )
    print(
        f"  uniform_total_bytes={uniform_total} uniform_local_bytes={uniform_local} "
        f"uniform_remote_bytes={uniform_remote}",
        flush=True,
    )
    print(
        f"  flexible_total_bytes={flexible_total} flexible_local_bytes={flexible_local} "
        f"flexible_remote_bytes={flexible_remote} skipped_bytes={skipped_bytes}",
        flush=True,
    )
    print(
        f"  estimated_total_reduction={total_reduction:.2f}% "
        f"estimated_remote_reduction={remote_reduction:.2f}%",
        flush=True,
    )
    header = "src\\dst".ljust(10) + "".join(f"{dst:>12d}" for dst in dst_ranks)
    print("  flexible matrix:", flush=True)
    print("  " + header, flush=True)
    for src_rank in src_ranks:
        cells = "".join(f"{flexible_matrix.get((src_rank, dst), 0):>12d}" for dst in dst_ranks)
        print("  " + f"{src_rank:<10d}" + cells, flush=True)


def _tp_overlap_units(
    src_tp_rank: int,
    dst_tp_rank: int,
    src_tp: int,
    dst_tp: int,
) -> tuple[int, int]:
    """Return overlap units and source-shard units on an integer LCM grid."""
    grid_units = math.lcm(src_tp, dst_tp)
    src_units = grid_units // src_tp
    dst_units = grid_units // dst_tp
    src_start = src_tp_rank * src_units
    dst_start = dst_tp_rank * dst_units
    overlap = max(
        0,
        min(src_start + src_units, dst_start + dst_units) - max(src_start, dst_start),
    )
    return overlap, src_units


def _build_rank_placement_rows(
    rank_order: list[Optional[int]],
    *,
    source_candidates: dict[int, list[int]],
    tp_bytes_by_src_rank: dict[int, int],
    tp_param_count_by_src_rank: dict[int, int],
    args: argparse.Namespace,
) -> list[dict]:
    """Estimate transfers for a logical-destination to physical-rank mapping."""
    src_tp = len(source_candidates)
    dst_tp = len(rank_order)
    rows: list[dict] = []
    task_id = 0
    for dst_tp_rank, dst_physical_rank in enumerate(rank_order):
        if dst_physical_rank is None:
            continue
        for src_tp_rank in range(src_tp):
            overlap, src_units = _tp_overlap_units(
                src_tp_rank,
                dst_tp_rank,
                src_tp,
                dst_tp,
            )
            if overlap == 0:
                continue
            bytes_count = int(
                round(tp_bytes_by_src_rank[src_tp_rank] * overlap / max(src_units, 1))
            )
            if bytes_count <= 0:
                continue
            op_count = max(1, tp_param_count_by_src_rank.get(src_tp_rank, 1))
            best_row = None
            best_cost = None
            for src_physical_rank in source_candidates[src_tp_rank]:
                row = {
                    "task_id": task_id,
                    "src_rank": int(src_physical_rank),
                    "dst_rank": int(dst_physical_rank),
                    "bytes": bytes_count,
                    "remote": bool(src_physical_rank != dst_physical_rank),
                    "op_count": op_count,
                    "param": f"aggregate_tp_{src_tp_rank}_to_{dst_tp_rank}",
                }
                cost = _task_estimated_seconds(row, args)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_row = row
            if best_row is None:
                raise RuntimeError(f"No physical source candidates for TP rank {src_tp_rank}")
            rows.append(best_row)
            task_id += 1
    return rows


def _rank_placement_objective(rows: list[dict], args: argparse.Namespace) -> tuple[float, int]:
    critical_s = _estimate_rows_critical_seconds(rows, args)
    remote_bytes = sum(int(row["bytes"]) for row in rows if row["remote"])
    return critical_s, remote_bytes


def _greedy_rank_placement(
    dst_physical_ranks: list[int],
    *,
    source_candidates: dict[int, list[int]],
    tp_bytes_by_src_rank: dict[int, int],
    tp_param_count_by_src_rank: dict[int, int],
    args: argparse.Namespace,
) -> list[int]:
    dst_tp = len(dst_physical_ranks)
    logical_bytes = {}
    for dst_tp_rank in range(dst_tp):
        total = 0
        for src_tp_rank, src_bytes in tp_bytes_by_src_rank.items():
            overlap, src_units = _tp_overlap_units(
                src_tp_rank,
                dst_tp_rank,
                len(source_candidates),
                dst_tp,
            )
            total += int(round(src_bytes * overlap / max(src_units, 1)))
        logical_bytes[dst_tp_rank] = total

    rank_order: list[Optional[int]] = [None] * dst_tp
    remaining = set(dst_physical_ranks)
    for dst_tp_rank in sorted(range(dst_tp), key=lambda item: (-logical_bytes[item], item)):
        best_physical_rank = None
        best_objective = None
        for physical_rank in sorted(remaining):
            candidate = list(rank_order)
            candidate[dst_tp_rank] = physical_rank
            rows = _build_rank_placement_rows(
                candidate,
                source_candidates=source_candidates,
                tp_bytes_by_src_rank=tp_bytes_by_src_rank,
                tp_param_count_by_src_rank=tp_param_count_by_src_rank,
                args=args,
            )
            objective = _rank_placement_objective(rows, args)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_physical_rank = physical_rank
        rank_order[dst_tp_rank] = best_physical_rank
        remaining.remove(best_physical_rank)

    resolved_order = [int(rank) for rank in rank_order]
    for _ in range(4):
        improved = False
        current_rows = _build_rank_placement_rows(
            resolved_order,
            source_candidates=source_candidates,
            tp_bytes_by_src_rank=tp_bytes_by_src_rank,
            tp_param_count_by_src_rank=tp_param_count_by_src_rank,
            args=args,
        )
        current_objective = _rank_placement_objective(current_rows, args)
        for left in range(dst_tp):
            for right in range(left + 1, dst_tp):
                candidate = list(resolved_order)
                candidate[left], candidate[right] = candidate[right], candidate[left]
                rows = _build_rank_placement_rows(
                    candidate,
                    source_candidates=source_candidates,
                    tp_bytes_by_src_rank=tp_bytes_by_src_rank,
                    tp_param_count_by_src_rank=tp_param_count_by_src_rank,
                    args=args,
                )
                objective = _rank_placement_objective(rows, args)
                if objective < current_objective:
                    resolved_order = candidate
                    current_objective = objective
                    improved = True
        if not improved:
            break
    return resolved_order


def _search_bandwidth_aware_rank_order(
    dst_physical_ranks: list[int],
    *,
    source_candidates: dict[int, list[int]],
    tp_bytes_by_src_rank: dict[int, int],
    tp_param_count_by_src_rank: dict[int, int],
    args: argparse.Namespace,
) -> tuple[list[int], str, dict[str, float | int | bool]]:
    identity_order = list(dst_physical_ranks)
    baseline_rows = _build_rank_placement_rows(
        identity_order,
        source_candidates=source_candidates,
        tp_bytes_by_src_rank=tp_bytes_by_src_rank,
        tp_param_count_by_src_rank=tp_param_count_by_src_rank,
        args=args,
    )
    baseline_s, baseline_remote_bytes = _rank_placement_objective(baseline_rows, args)

    search = args.rank_placement_search
    if search == "auto":
        search = (
            "exhaustive"
            if len(dst_physical_ranks) <= args.rank_placement_max_exhaustive_ranks
            else "greedy"
        )

    if search == "exhaustive":
        if len(dst_physical_ranks) > args.rank_placement_max_exhaustive_ranks:
            raise ValueError(
                "Exhaustive rank placement is disabled above "
                f"{args.rank_placement_max_exhaustive_ranks} ranks; got "
                f"{len(dst_physical_ranks)}"
            )
        best_order = identity_order
        best_objective = (baseline_s, baseline_remote_bytes)
        for permutation in itertools.permutations(dst_physical_ranks):
            candidate_order = list(permutation)
            rows = _build_rank_placement_rows(
                candidate_order,
                source_candidates=source_candidates,
                tp_bytes_by_src_rank=tp_bytes_by_src_rank,
                tp_param_count_by_src_rank=tp_param_count_by_src_rank,
                args=args,
            )
            objective = _rank_placement_objective(rows, args)
            if objective < best_objective:
                best_objective = objective
                best_order = candidate_order
    else:
        best_order = _greedy_rank_placement(
            dst_physical_ranks,
            source_candidates=source_candidates,
            tp_bytes_by_src_rank=tp_bytes_by_src_rank,
            tp_param_count_by_src_rank=tp_param_count_by_src_rank,
            args=args,
        )

    candidate_rows = _build_rank_placement_rows(
        best_order,
        source_candidates=source_candidates,
        tp_bytes_by_src_rank=tp_bytes_by_src_rank,
        tp_param_count_by_src_rank=tp_param_count_by_src_rank,
        args=args,
    )
    candidate_s, candidate_remote_bytes = _rank_placement_objective(candidate_rows, args)
    gain_s = baseline_s - candidate_s
    gain_pct = 0.0 if baseline_s <= 0.0 else 100.0 * gain_s / baseline_s
    accept = gain_s > 0.0 and gain_pct >= args.rank_placement_min_benefit_pct
    selected_order = best_order if accept else identity_order
    metrics: dict[str, float | int | bool] = {
        "accepted": accept,
        "baseline_s": baseline_s,
        "candidate_s": candidate_s,
        "predicted_gain_s": gain_s,
        "predicted_gain_pct": gain_pct,
        "baseline_remote_bytes": baseline_remote_bytes,
        "candidate_remote_bytes": candidate_remote_bytes,
        "baseline_slow_link_bytes": _rows_slow_link_bytes(baseline_rows, args),
        "candidate_slow_link_bytes": _rows_slow_link_bytes(candidate_rows, args),
    }
    return selected_order, search, metrics


def choose_bandwidth_aware_dst_rank_order(
    src_model: Optional[torch.nn.Module],
    args: argparse.Namespace,
    *,
    src_ranks: list[int],
    dst_ranks: list[int],
) -> list[int]:
    """Choose physical GPUs for destination logical TP ranks before model construction."""
    if not args.bandwidth_aware_rank_placement:
        return list(dst_ranks)

    rank = dist.get_rank()
    payload = None
    if src_model is not None and rank in src_ranks:
        tp_rank = dist.get_rank(src_model.pg_collection.tp)
        tp_params = [
            param
            for _, param in src_model.named_parameters()
            if bool(getattr(param, "tensor_model_parallel", False))
        ]
        payload = {
            "physical_rank": rank,
            "tp_rank": int(tp_rank),
            "tp_bytes": sum(param.numel() * param.element_size() for param in tp_params),
            "tp_param_count": len(tp_params),
        }

    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(payload, gathered, group_dst=0)
    result = [None]
    if rank == 0:
        source_candidates: dict[int, list[int]] = {tp_rank: [] for tp_rank in range(args.src_tp)}
        bytes_samples: dict[int, list[int]] = {tp_rank: [] for tp_rank in range(args.src_tp)}
        count_samples: dict[int, list[int]] = {tp_rank: [] for tp_rank in range(args.src_tp)}
        for item in gathered:
            if item is None:
                continue
            tp_rank = int(item["tp_rank"])
            source_candidates[tp_rank].append(int(item["physical_rank"]))
            bytes_samples[tp_rank].append(int(item["tp_bytes"]))
            count_samples[tp_rank].append(int(item["tp_param_count"]))
        missing = [tp_rank for tp_rank, ranks in source_candidates.items() if not ranks]
        if missing:
            raise RuntimeError(f"No active source ranks found for logical TP ranks {missing}")
        tp_bytes_by_src_rank = {
            tp_rank: int(round(sum(samples) / len(samples)))
            for tp_rank, samples in bytes_samples.items()
        }
        tp_param_count_by_src_rank = {
            tp_rank: max(1, int(round(sum(samples) / len(samples))))
            for tp_rank, samples in count_samples.items()
        }
        selected_order, search, metrics = _search_bandwidth_aware_rank_order(
            dst_ranks,
            source_candidates=source_candidates,
            tp_bytes_by_src_rank=tp_bytes_by_src_rank,
            tp_param_count_by_src_rank=tp_param_count_by_src_rank,
            args=args,
        )
        result[0] = {
            "rank_order": selected_order,
            "search": search,
            "metrics": metrics,
            "source_candidates": source_candidates,
        }
    dist.broadcast_object_list(result, src=0)
    placement = result[0]
    args.dst_rank_order = list(placement["rank_order"])
    if rank == 0:
        metrics = placement["metrics"]
        mapping = ", ".join(
            f"tp{logical_rank}->gpu{physical_rank}"
            for logical_rank, physical_rank in enumerate(placement["rank_order"])
        )
        bandwidth_source = (
            "profiled-matrix"
            if getattr(args, "scheduler_bandwidth_matrix", None) is not None
            else "local-remote-defaults"
        )
        print("\nBandwidth-aware destination rank placement:", flush=True)
        print(
            f"  accepted={metrics['accepted']} search={placement['search']} "
            f"bandwidth={bandwidth_source} mapping=[{mapping}]",
            flush=True,
        )
        print(
            f"  baseline_s={metrics['baseline_s']:.6f} "
            f"candidate_s={metrics['candidate_s']:.6f} "
            f"predicted_gain_s={metrics['predicted_gain_s']:.6f} "
            f"predicted_gain_pct={metrics['predicted_gain_pct']:.2f} "
            f"min_benefit_pct={args.rank_placement_min_benefit_pct:.2f}",
            flush=True,
        )
        print(
            f"  baseline_remote_bytes={metrics['baseline_remote_bytes']} "
            f"candidate_remote_bytes={metrics['candidate_remote_bytes']} "
            f"baseline_slow_link_bytes={metrics['baseline_slow_link_bytes']} "
            f"candidate_slow_link_bytes={metrics['candidate_slow_link_bytes']}",
            flush=True,
        )
    return list(placement["rank_order"])


def _collect_recv_task_rows(
    plan: ReshardPlan,
    dst_module: Optional[torch.nn.Module],
) -> list[dict]:
    rank = dist.get_rank()
    dst_params = dict(dst_module.named_parameters()) if dst_module is not None else {}
    local_rows = []
    for op in plan.recv_ops:
        if op.task_id is None:
            continue
        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        bytes_count = _slice_numel(tuple(dst_param.shape), op.my_slice) * dst_param.element_size()
        local_rows.append(
            {
                "task_id": int(op.task_id),
                "src_rank": int(op.peer_rank),
                "dst_rank": int(rank),
                "bytes": int(bytes_count),
                "remote": bool(op.peer_rank != rank),
                "param": op.param_name,
            }
        )

    gathered = [None] * dist.get_world_size() if rank == 0 else None
    dist.gather_object(local_rows, gathered, group_dst=0)
    if rank != 0:
        return []

    rows_by_task: dict[int, dict] = {}
    for rank_rows in gathered:
        for row in rank_rows:
            rows_by_task[row["task_id"]] = row
    return list(rows_by_task.values())


def _task_estimated_seconds(row: dict, args: argparse.Namespace) -> float:
    bandwidth_matrix = getattr(args, "scheduler_bandwidth_matrix", None)
    if bandwidth_matrix is not None:
        bandwidth_gbps = bandwidth_matrix.get((row["src_rank"], row["dst_rank"]))
    else:
        bandwidth_gbps = None
    if bandwidth_gbps is None:
        bandwidth_gbps = (
            args.scheduler_remote_bandwidth_gbps
            if row["remote"]
            else args.scheduler_local_bandwidth_gbps
        )
    bandwidth_bytes_per_s = max(bandwidth_gbps, 1.0e-9) * 1.0e9 / 8.0
    op_count = max(1, int(row.get("op_count", 1)))
    return (
        float(row["bytes"]) / bandwidth_bytes_per_s
        + op_count * args.scheduler_latency_us * 1.0e-6
    )


def _rows_slow_link_bytes(rows: list[dict], args: argparse.Namespace) -> int:
    slow_links = getattr(args, "scheduler_slow_link_gbps", {}) or {}
    if not slow_links:
        return 0
    return sum(
        int(row["bytes"])
        for row in rows
        if (int(row["src_rank"]), int(row["dst_rank"])) in slow_links
    )


def _plan_slow_link_delay_seconds(
    plan: ReshardPlan,
    src_module: Optional[torch.nn.Module],
    dst_module: Optional[torch.nn.Module],
    args: argparse.Namespace,
) -> tuple[float, int]:
    slow_links = getattr(args, "scheduler_slow_link_gbps", {}) or {}
    if not slow_links:
        return 0.0, 0
    rank = dist.get_rank()
    src_params = dict(src_module.named_parameters()) if src_module is not None else {}
    dst_params = dict(dst_module.named_parameters()) if dst_module is not None else {}
    bytes_by_link: dict[tuple[int, int], int] = {}

    for op in plan.send_ops:
        pair = (rank, int(op.peer_rank))
        if pair not in slow_links:
            continue
        src_param = src_params.get(op.param_name)
        if src_param is None:
            continue
        num_bytes = _slice_numel(tuple(src_param.shape), op.my_slice) * src_param.element_size()
        bytes_by_link[pair] = bytes_by_link.get(pair, 0) + int(num_bytes)

    for op in plan.recv_ops:
        pair = (int(op.peer_rank), rank)
        if pair not in slow_links:
            continue
        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        num_bytes = _slice_numel(tuple(dst_param.shape), op.my_slice) * dst_param.element_size()
        bytes_by_link[pair] = bytes_by_link.get(pair, 0) + int(num_bytes)

    delay_s = 0.0
    slow_bytes = 0
    for pair, num_bytes in bytes_by_link.items():
        gbps = max(float(slow_links[pair]), 1.0e-9)
        delay_s += num_bytes / (gbps * 1.0e9 / 8.0)
        slow_bytes += num_bytes
    return delay_s, slow_bytes


def maybe_inject_slow_link_delay(
    plan: ReshardPlan,
    src_module: Optional[torch.nn.Module],
    dst_module: Optional[torch.nn.Module],
    args: Optional[argparse.Namespace],
    *,
    label: str,
    group=None,
) -> None:
    if args is None or not getattr(args, "scheduler_slow_link_simulate_delay", False):
        return
    delay_s, slow_bytes = _plan_slow_link_delay_seconds(plan, src_module, dst_module, args)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    stats = torch.tensor([delay_s, float(slow_bytes)], device=device, dtype=torch.float64)
    dist.all_reduce(stats[0:1], op=dist.ReduceOp.MAX)
    dist.all_reduce(stats[1:2], op=dist.ReduceOp.SUM)
    if dist.get_rank() == 0 and stats[0].item() > 0.0:
        print(
            f"Simulated slow-link delay for {label}: "
            f"max_rank_delay_s={stats[0].item():.6f} total_rank_slow_bytes={int(stats[1].item())}",
            flush=True,
        )
    dist.barrier(group=group)
    if delay_s > 0.0:
        time.sleep(delay_s)
    dist.barrier(group=group)


def _pack_bandwidth_greedy_waves(rows: list[dict], args: argparse.Namespace) -> list[list[int]]:
    """Pack transfer tasks by the estimated seconds on rank and pairwise links.

    This is the explicit bandwidth-aware path: every task is modeled as
    (src_rank, dst_rank, bytes), costed by bytes / B[src][dst] + latency, and
    greedily assigned to the wave that minimizes the predicted critical path.
    """
    total_bytes = sum(row["bytes"] for row in rows)
    max_waves = max(1, args.scheduler_max_waves)
    max_wave_bytes = (
        args.scheduler_max_wave_bytes
        if args.scheduler_max_wave_bytes > 0
        else max(1, math.ceil(total_bytes / max_waves))
    )

    enriched_rows = [
        {**row, "estimated_s": _task_estimated_seconds(row, args)}
        for row in rows
    ]
    enriched_rows.sort(
        key=lambda row: (
            row["estimated_s"],
            row["remote"],
            row["bytes"],
            -row["task_id"],
        ),
        reverse=True,
    )

    waves: list[dict] = []

    def _candidate_score(wave: dict, row: dict) -> tuple[float, int, int]:
        src = int(row["src_rank"])
        dst = int(row["dst_rank"])
        cost = float(row["estimated_s"])
        critical_s = float(wave["critical_s"])
        if row["remote"]:
            send_s = wave["send_load_s"].get(src, 0.0) + cost
            recv_s = wave["recv_load_s"].get(dst, 0.0) + cost
            link_key = (src, dst)
            link_s = wave["link_load_s"].get(link_key, 0.0) + cost
            critical_s = max(critical_s, send_s, recv_s, link_s)
        else:
            local_s = wave["local_load_s"].get(dst, 0.0) + cost
            critical_s = max(critical_s, local_s)
        return (critical_s, wave["bytes"] + row["bytes"], len(wave["tasks"]))

    for row in enriched_rows:
        best_index = None
        best_score = None
        for index, wave in enumerate(waves):
            if wave["bytes"] + row["bytes"] > max_wave_bytes and wave["tasks"]:
                continue
            score = _candidate_score(wave, row)
            if best_score is None or score < best_score:
                best_index = index
                best_score = score

        if best_index is None:
            if len(waves) < max_waves:
                waves.append(
                    {
                        "tasks": [],
                        "bytes": 0,
                        "send_load_s": {},
                        "recv_load_s": {},
                        "link_load_s": {},
                        "local_load_s": {},
                        "critical_s": 0.0,
                    }
                )
                best_index = len(waves) - 1
            else:
                best_index = min(
                    range(len(waves)),
                    key=lambda index: _candidate_score(waves[index], row),
                )

        wave = waves[best_index]
        src = int(row["src_rank"])
        dst = int(row["dst_rank"])
        cost = float(row["estimated_s"])
        wave["tasks"].append(row["task_id"])
        wave["bytes"] += row["bytes"]
        if row["remote"]:
            wave["send_load_s"][src] = wave["send_load_s"].get(src, 0.0) + cost
            wave["recv_load_s"][dst] = wave["recv_load_s"].get(dst, 0.0) + cost
            link_key = (src, dst)
            wave["link_load_s"][link_key] = wave["link_load_s"].get(link_key, 0.0) + cost
        else:
            wave["local_load_s"][dst] = wave["local_load_s"].get(dst, 0.0) + cost
        wave["critical_s"] = _candidate_score(wave, {**row, "bytes": 0, "estimated_s": 0.0})[0]

    waves.sort(key=lambda wave: (wave["critical_s"], wave["bytes"]), reverse=True)
    return [wave["tasks"] for wave in waves if wave["tasks"]]


def _pack_scheduled_waves(rows: list[dict], args: argparse.Namespace) -> list[list[int]]:
    if not rows:
        return []

    if args.reshard_scheduler == "nccl-round":
        return _pack_nccl_round_waves(rows, args)
    if args.reshard_scheduler == "bandwidth-greedy":
        return _pack_bandwidth_greedy_waves(rows, args)

    total_bytes = sum(row["bytes"] for row in rows)
    max_waves = max(1, args.scheduler_max_waves)
    if args.scheduler_max_wave_bytes > 0:
        max_wave_bytes = args.scheduler_max_wave_bytes
    else:
        max_wave_bytes = max(1, math.ceil(total_bytes / max_waves))

    if args.reshard_scheduler == "bytes-desc":
        rows = sorted(rows, key=lambda row: (row["remote"], row["bytes"]), reverse=True)
    else:
        rows = sorted(
            rows,
            key=lambda row: (
                _task_estimated_seconds(row, args),
                row["remote"],
                row["bytes"],
            ),
            reverse=True,
        )

    waves = []
    for row in rows:
        best_index = None
        best_score = None
        for index, wave in enumerate(waves):
            if wave["bytes"] + row["bytes"] > max_wave_bytes and wave["tasks"]:
                continue
            src_load = wave["send_load"].get(row["src_rank"], 0) + row["bytes"]
            dst_load = wave["recv_load"].get(row["dst_rank"], 0) + row["bytes"]
            link_key = (row["src_rank"], row["dst_rank"])
            link_load = wave["link_load"].get(link_key, 0) + row["bytes"]
            score = max(src_load, dst_load, link_load, wave["bytes"] + row["bytes"])
            if best_score is None or score < best_score:
                best_index = index
                best_score = score

        if best_index is None:
            if len(waves) < max_waves:
                waves.append(
                    {
                        "tasks": [],
                        "bytes": 0,
                        "send_load": {},
                        "recv_load": {},
                        "link_load": {},
                    }
                )
                best_index = len(waves) - 1
            else:
                best_index = min(range(len(waves)), key=lambda index: waves[index]["bytes"])

        wave = waves[best_index]
        wave["tasks"].append(row["task_id"])
        wave["bytes"] += row["bytes"]
        wave["send_load"][row["src_rank"]] = wave["send_load"].get(row["src_rank"], 0) + row["bytes"]
        wave["recv_load"][row["dst_rank"]] = wave["recv_load"].get(row["dst_rank"], 0) + row["bytes"]
        link_key = (row["src_rank"], row["dst_rank"])
        wave["link_load"][link_key] = wave["link_load"].get(link_key, 0) + row["bytes"]

    return [wave["tasks"] for wave in waves if wave["tasks"]]


def _pack_nccl_round_waves(rows: list[dict], args: argparse.Namespace) -> list[list[int]]:
    """Order P2P tasks in the same spirit as NCCL's p2p round scheduler.

    NCCL stores submitted P2P operations in per-peer send/recv queues, then
    builds device work by repeatedly walking a per-rank P2P round schedule and
    popping one task from each active peer queue.  A single-batch reshard
    scheduler should therefore avoid destroying same-peer FIFO order; the NCCL
    planner already handles cross-peer fairness.

    This mode keeps the original FIFO order for each (src, dst) queue and only
    interleaves peer queues by a deterministic ring-style round.  It is a
    topology-agnostic approximation of NCCL's internal p2pSchedule, but it
    preserves the key invariant that repeated transfers over the same peer link
    remain in planner order.
    """
    world_size = dist.get_world_size()
    max_waves = max(1, args.scheduler_max_waves)
    queues: dict[tuple[int, int], list[dict]] = {}
    local_rows: list[dict] = []

    for row in sorted(rows, key=lambda item: item["task_id"]):
        src = int(row["src_rank"])
        dst = int(row["dst_rank"])
        if src == dst:
            local_rows.append(row)
            continue
        queues.setdefault((src, dst), []).append(row)

    def _round_index(src: int, dst: int) -> int:
        # Cyclic send/recv distance. NCCL's real p2pSchedule is topology-aware,
        # but this matches the schedule's one-peer-per-round structure.
        return (dst - src) % max(world_size, 1)

    ordered_remote: list[int] = []
    while queues:
        progressed = False
        for round_index in range(world_size):
            active_pairs = [
                pair
                for pair in queues
                if _round_index(pair[0], pair[1]) == round_index
            ]
            active_pairs.sort()
            for pair in active_pairs:
                queue = queues.get(pair)
                if not queue:
                    continue
                ordered_remote.append(queue.pop(0)["task_id"])
                progressed = True
                if not queue:
                    del queues[pair]
        if not progressed:
            break

    ordered_task_ids = [row["task_id"] for row in local_rows] + ordered_remote
    if args.scheduler_execution_mode == "single-batch":
        return [ordered_task_ids]

    # For explicit wave execution, keep the same NCCL-like order but split the
    # flattened stream into byte-balanced chunks without re-sorting tasks.
    rows_by_id = {row["task_id"]: row for row in rows}
    total_bytes = sum(row["bytes"] for row in rows)
    max_wave_bytes = (
        args.scheduler_max_wave_bytes
        if args.scheduler_max_wave_bytes > 0
        else max(1, math.ceil(total_bytes / max_waves))
    )
    waves: list[list[int]] = []
    current_wave: list[int] = []
    current_bytes = 0
    for task_id in ordered_task_ids:
        task_bytes = rows_by_id[task_id]["bytes"]
        if current_wave and len(waves) + 1 < max_waves and current_bytes + task_bytes > max_wave_bytes:
            waves.append(current_wave)
            current_wave = []
            current_bytes = 0
        current_wave.append(task_id)
        current_bytes += task_bytes
    if current_wave:
        waves.append(current_wave)
    return waves


def _estimate_rows_critical_seconds(rows: list[dict], args: argparse.Namespace) -> float:
    """Estimate the rank/link critical path for a set of transfer tasks."""
    send_load: dict[int, float] = {}
    recv_load: dict[int, float] = {}
    link_load: dict[tuple[int, int], float] = {}
    local_load: dict[int, float] = {}
    for row in rows:
        cost = _task_estimated_seconds(row, args)
        src = int(row["src_rank"])
        dst = int(row["dst_rank"])
        if row["remote"]:
            send_load[src] = send_load.get(src, 0.0) + cost
            recv_load[dst] = recv_load.get(dst, 0.0) + cost
            link_key = (src, dst)
            link_load[link_key] = link_load.get(link_key, 0.0) + cost
        else:
            local_load[dst] = local_load.get(dst, 0.0) + cost
    return max(
        [0.0]
        + list(send_load.values())
        + list(recv_load.values())
        + list(link_load.values())
        + list(local_load.values())
    )


def _estimate_scheduled_rows_seconds(
    rows: list[dict],
    waves_task_ids: list[list[int]],
    args: argparse.Namespace,
) -> float:
    if not waves_task_ids:
        return 0.0
    rows_by_id = {row["task_id"]: row for row in rows}
    if args.scheduler_execution_mode == "single-batch":
        # Single-batch scheduling only changes submission order. It does not
        # reduce the byte critical path, so the safe predictor should not assume
        # a benefit from reordering alone.
        return _estimate_rows_critical_seconds(rows, args)

    wave_barrier_s = args.scheduler_wave_barrier_us * 1.0e-6
    total = 0.0
    for index, task_ids in enumerate(waves_task_ids):
        wave_rows = [rows_by_id[task_id] for task_id in task_ids]
        total += _estimate_rows_critical_seconds(wave_rows, args)
        if index + 1 < len(waves_task_ids):
            total += wave_barrier_s
    return total


def _scheduler_auto_gate_accepts(
    rows: list[dict],
    waves_task_ids: list[list[int]],
    args: argparse.Namespace,
) -> tuple[bool, dict[str, float]]:
    baseline_s = _estimate_rows_critical_seconds(rows, args)
    candidate_s = _estimate_scheduled_rows_seconds(rows, waves_task_ids, args)
    switch_overhead_s = args.scheduler_switch_overhead_us * 1.0e-6
    threshold = max(0.0, args.scheduler_min_benefit_pct) * 0.01
    required_s = baseline_s * threshold + switch_overhead_s
    predicted_gain_s = baseline_s - candidate_s
    accept = predicted_gain_s > required_s
    return accept, {
        "baseline_s": baseline_s,
        "candidate_s": candidate_s,
        "predicted_gain_s": predicted_gain_s,
        "required_gain_s": required_s,
        "min_benefit_pct": threshold * 100.0,
        "switch_overhead_s": switch_overhead_s,
    }


def schedule_reshard_plan_waves(
    plan: ReshardPlan,
    dst_module: Optional[torch.nn.Module],
    args: argparse.Namespace,
    *,
    label: str,
) -> list[ReshardPlan]:
    if args.reshard_scheduler == "none":
        return [plan]

    rank = dist.get_rank()
    rows = _collect_recv_task_rows(plan, dst_module)
    if rank == 0:
        missing_task_ids = any(op.task_id is None for op in plan.recv_ops)
        if missing_task_ids:
            print(
                f"\nReshard scheduler disabled for {label}: plan has recv ops without task_id.",
                flush=True,
            )
            waves_task_ids = None
        else:
            waves_task_ids = _pack_scheduled_waves(rows, args)
            total_bytes = sum(row["bytes"] for row in rows)
            remote_bytes = sum(row["bytes"] for row in rows if row["remote"])
            if args.scheduler_auto_gate:
                gate_accepts, gate = _scheduler_auto_gate_accepts(rows, waves_task_ids, args)
                print(
                    f"\nReshard scheduler auto-gate for {label}: "
                    f"accept={gate_accepts} baseline_s={gate['baseline_s']:.6f} "
                    f"candidate_s={gate['candidate_s']:.6f} "
                    f"predicted_gain_s={gate['predicted_gain_s']:.6f} "
                    f"required_gain_s={gate['required_gain_s']:.6f} "
                    f"min_benefit_pct={gate['min_benefit_pct']:.3f} "
                    f"switch_overhead_s={gate['switch_overhead_s']:.6f}",
                    flush=True,
                )
                if not gate_accepts:
                    print(
                        f"Reshard scheduler disabled by auto-gate for {label}; "
                        "falling back to original transfer order.",
                        flush=True,
                    )
                    waves_task_ids = None
            bandwidth_source = (
                "profiled-matrix"
                if getattr(args, "scheduler_bandwidth_matrix", None) is not None
                else "local-remote-defaults"
            )
            if waves_task_ids:
                print(
                    f"\nReshard scheduler for {label}: "
                    f"mode={args.reshard_scheduler} execution={args.scheduler_execution_mode} "
                    f"waves={len(waves_task_ids)} "
                    f"tasks={len(rows)} total_bytes={total_bytes} remote_bytes={remote_bytes} "
                    f"max_waves={args.scheduler_max_waves} bandwidth={bandwidth_source}",
                    flush=True,
                )
                for index, task_ids in enumerate(waves_task_ids[: args.scheduler_print_waves]):
                    wave_rows = [row for row in rows if row["task_id"] in set(task_ids)]
                    wave_bytes = sum(row["bytes"] for row in wave_rows)
                    wave_remote = sum(row["bytes"] for row in wave_rows if row["remote"])
                    wave_slow = _rows_slow_link_bytes(wave_rows, args)
                    print(
                        f"  wave={index} tasks={len(task_ids)} "
                        f"bytes={wave_bytes} remote_bytes={wave_remote} "
                        f"slow_link_bytes={wave_slow}",
                        flush=True,
                    )
            if waves_task_ids and len(waves_task_ids) > args.scheduler_print_waves:
                print(
                    f"  ... {len(waves_task_ids) - args.scheduler_print_waves} more waves",
                    flush=True,
                )
    else:
        waves_task_ids = None

    waves_task_ids = [waves_task_ids]
    dist.broadcast_object_list(waves_task_ids, src=0)
    waves_task_ids = waves_task_ids[0]
    if not waves_task_ids:
        return [plan]

    if args.scheduler_execution_mode == "waves":
        return [_filter_reshard_plan_by_task_ids(plan, set(task_ids)) for task_ids in waves_task_ids]

    ordered_task_ids = [task_id for task_ids in waves_task_ids for task_id in task_ids]
    return [_reorder_reshard_plan_by_task_ids(plan, ordered_task_ids)]


def execute_reshard_plan_with_scheduler(
    plan: ReshardPlan,
    src_module: Optional[torch.nn.Module],
    dst_module: Optional[torch.nn.Module],
    service: CopyService,
    args: argparse.Namespace,
    *,
    label: str,
    group=None,
) -> None:
    wave_plans = schedule_reshard_plan_waves(plan, dst_module, args, label=label)
    if args.reshard_fusion in ("peer", "adaptive-peer"):
        execute_scheduled_fused_peer_reshard_plans(
            wave_plans,
            src_module,
            dst_module,
            service,
            args,
            label=label,
            group=group,
        )
    else:
        execute_scheduled_reshard_waves(
            wave_plans,
            src_module,
            dst_module,
            service,
            args=args,
            label=label,
            group=group,
        )


def execute_scheduled_fused_peer_reshard_plans(
    wave_plans: list[ReshardPlan],
    src_module: Optional[torch.nn.Module],
    dst_module: Optional[torch.nn.Module],
    service: CopyService,
    args: argparse.Namespace,
    *,
    label: str,
    group=None,
) -> None:
    for wave_index, wave_plan in enumerate(wave_plans):
        wave_label = label
        if len(wave_plans) > 1:
            wave_label = f"{label} wave {wave_index + 1}/{len(wave_plans)}"
            if dist.get_rank() == 0:
                print(
                    f"Executing fused scheduled wave {wave_index + 1}/{len(wave_plans)} "
                    f"for {label}: sends={len(wave_plan.send_ops)} recvs={len(wave_plan.recv_ops)}",
                    flush=True,
                )
        maybe_inject_slow_link_delay(
            wave_plan,
            src_module,
            dst_module,
            args,
            label=wave_label,
            group=group,
        )
        execute_fused_peer_reshard_plan(
            wave_plan,
            src_module,
            dst_module,
            service,
            args,
            label=wave_label,
            group=group,
        )


def _adaptive_fusion_chunk_bytes(
    remote_bytes: int,
    remote_peer_groups: int,
    args: argparse.Namespace,
) -> int:
    if args.reshard_fusion == "peer":
        return 0
    if args.reshard_fusion_chunk_bytes > 0:
        return args.reshard_fusion_chunk_bytes

    world_size = dist.get_world_size()
    active_groups = max(1, remote_peer_groups)
    min_chunk = max(1, args.reshard_fusion_min_chunk_bytes)
    max_chunk = max(min_chunk, args.reshard_fusion_max_chunk_bytes)
    target_chunks_per_group = max(1, args.reshard_fusion_target_chunks_per_peer)
    target_rank_chunks = min(32, max(8, world_size // 4))
    target_total_chunks = max(active_groups * target_chunks_per_group, target_rank_chunks)
    dynamic_chunk = max(min_chunk, math.ceil(max(1, remote_bytes) / target_total_chunks))
    return min(max_chunk, dynamic_chunk)


def _pack_fusion_send_chunks(
    entries: list[tuple[int, torch.Tensor, int]],
    max_chunk_bytes: int,
) -> list[tuple[list[int], torch.Tensor]]:
    if not entries:
        return []
    if max_chunk_bytes <= 0:
        task_ids = [task_id for task_id, _, _ in entries]
        tensors = [tensor for _, tensor, _ in entries]
        return [(task_ids, tensors[0] if len(tensors) == 1 else torch.cat(tensors))]

    chunks: list[tuple[list[int], torch.Tensor]] = []
    current_task_ids: list[int] = []
    current_tensors: list[torch.Tensor] = []
    current_bytes = 0
    for task_id, tensor, num_bytes in entries:
        if current_tensors and current_bytes + num_bytes > max_chunk_bytes:
            chunks.append(
                (
                    current_task_ids,
                    current_tensors[0] if len(current_tensors) == 1 else torch.cat(current_tensors),
                )
            )
            current_task_ids = []
            current_tensors = []
            current_bytes = 0
        current_task_ids.append(task_id)
        current_tensors.append(tensor)
        current_bytes += num_bytes
    if current_tensors:
        chunks.append(
            (
                current_task_ids,
                current_tensors[0] if len(current_tensors) == 1 else torch.cat(current_tensors),
            )
        )
    return chunks


def _pack_fusion_recv_chunks(
    entries: list[tuple[int, torch.nn.Parameter, tuple[slice, ...], int]],
    max_chunk_bytes: int,
) -> list[tuple[list[int], list[tuple[torch.nn.Parameter, tuple[slice, ...], int]]]]:
    if not entries:
        return []
    if max_chunk_bytes <= 0:
        task_ids = [task_id for task_id, _, _, _ in entries]
        chunk_entries = [(dst_param, dst_slice, numel) for _, dst_param, dst_slice, numel in entries]
        return [(task_ids, chunk_entries)]

    chunks: list[tuple[list[int], list[tuple[torch.nn.Parameter, tuple[slice, ...], int]]]] = []
    current_task_ids: list[int] = []
    current: list[tuple[torch.nn.Parameter, tuple[slice, ...], int]] = []
    current_bytes = 0
    for task_id, dst_param, dst_slice, numel in entries:
        num_bytes = numel * dst_param.element_size()
        if current and current_bytes + num_bytes > max_chunk_bytes:
            chunks.append((current_task_ids, current))
            current_task_ids = []
            current = []
            current_bytes = 0
        current_task_ids.append(task_id)
        current.append((dst_param, dst_slice, numel))
        current_bytes += num_bytes
    if current:
        chunks.append((current_task_ids, current))
    return chunks


def _global_adaptive_chunk_bytes(local_chunk_bytes: int, device: torch.device, group=None) -> int:
    chunk_tensor = torch.tensor(local_chunk_bytes, device=device, dtype=torch.long)
    dist.all_reduce(chunk_tensor, op=dist.ReduceOp.MAX, group=group)
    return int(chunk_tensor.item())


def _validate_fused_chunk_metadata(
    send_metadata: list[tuple],
    recv_metadata: list[tuple],
    *,
    label: str,
    group=None,
) -> None:
    world_size = dist.get_world_size(group=group)
    gathered_sends = [None for _ in range(world_size)]
    gathered_recvs = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_sends, send_metadata, group=group)
    dist.all_gather_object(gathered_recvs, recv_metadata, group=group)

    all_sends = [item for rank_items in gathered_sends for item in (rank_items or [])]
    all_recvs = [item for rank_items in gathered_recvs for item in (rank_items or [])]
    if sorted(all_sends) == sorted(all_recvs):
        send_order: dict[tuple, list[tuple]] = {}
        recv_order: dict[tuple, list[tuple]] = {}
        for src_rank, dst_rank, dtype_name, task_ids, num_bytes, mode in all_sends:
            send_order.setdefault((src_rank, dst_rank, dtype_name), []).append(
                (task_ids, num_bytes, mode)
            )
        for src_rank, dst_rank, dtype_name, task_ids, num_bytes, mode in all_recvs:
            recv_order.setdefault((src_rank, dst_rank, dtype_name), []).append(
                (task_ids, num_bytes, mode)
            )
        if send_order == recv_order:
            return
        mismatched_keys = sorted(
            set(send_order.keys()) | set(recv_order.keys()),
            key=lambda item: (item[0], item[1], item[2]),
        )
        for key in mismatched_keys:
            if send_order.get(key) != recv_order.get(key):
                raise RuntimeError(
                    f"{label}: fused chunk order mismatch before NCCL run; "
                    f"key={key} send_sample={send_order.get(key, [])[:3]} "
                    f"recv_sample={recv_order.get(key, [])[:3]}"
                )

    send_only = sorted(set(all_sends) - set(all_recvs))[:3]
    recv_only = sorted(set(all_recvs) - set(all_sends))[:3]
    raise RuntimeError(
        f"{label}: fused chunk metadata mismatch before NCCL run; "
        f"send_chunks={len(all_sends)} recv_chunks={len(all_recvs)} "
        f"send_only_sample={send_only} recv_only_sample={recv_only}"
    )


def _estimate_fusion_group_chunks(
    group_bytes: dict[tuple[int, str], list[int]],
    args: argparse.Namespace,
) -> tuple[int, int]:
    if not group_bytes:
        return 0, 0
    total_bytes = sum(sum(values) for values in group_bytes.values())
    chunk_bytes = _adaptive_fusion_chunk_bytes(total_bytes, len(group_bytes), args)
    chunk_count = 0
    for values in group_bytes.values():
        if chunk_bytes <= 0:
            chunk_count += 1
            continue
        current = 0
        for num_bytes in values:
            if current and current + num_bytes > chunk_bytes:
                chunk_count += 1
                current = 0
            current += num_bytes
        if current:
            chunk_count += 1
    return chunk_count, total_bytes


def _fusion_auto_gate_accepts(
    plan: ReshardPlan,
    src_module: Optional[torch.nn.Module],
    dst_module: Optional[torch.nn.Module],
    args: argparse.Namespace,
    *,
    label: str,
    group=None,
) -> bool:
    rank = dist.get_rank()
    src_params = dict(src_module.named_parameters()) if src_module is not None else {}
    dst_params = dict(dst_module.named_parameters()) if dst_module is not None else {}

    direct_ops = 0
    direct_bytes = 0
    candidate_direct_ops = 0
    candidate_direct_bytes = 0
    send_group_bytes: dict[tuple[int, str], list[int]] = {}
    recv_group_bytes: dict[tuple[int, str], list[int]] = {}
    small_op_bytes = (
        args.reshard_fusion_small_op_bytes if args.reshard_fusion == "adaptive-peer" else 0
    )

    def _use_direct(num_bytes: int) -> bool:
        return args.reshard_fusion == "adaptive-peer" and small_op_bytes > 0 and (
            num_bytes > small_op_bytes
        )

    for op in plan.send_ops:
        if op.peer_rank == rank:
            continue
        src_param = src_params.get(op.param_name)
        if src_param is None:
            continue
        src_view = src_param.data[op.my_slice]
        num_bytes = src_view.numel() * src_view.element_size()
        direct_ops += 1
        direct_bytes += num_bytes
        if _use_direct(num_bytes):
            candidate_direct_ops += 1
            candidate_direct_bytes += num_bytes
        else:
            key = (int(op.peer_rank), str(src_view.dtype))
            send_group_bytes.setdefault(key, []).append(num_bytes)

    for op in plan.recv_ops:
        if op.peer_rank == rank:
            continue
        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        dst_view = dst_param.data[op.my_slice]
        num_bytes = dst_view.numel() * dst_view.element_size()
        direct_ops += 1
        direct_bytes += num_bytes
        if _use_direct(num_bytes):
            candidate_direct_ops += 1
            candidate_direct_bytes += num_bytes
        else:
            key = (int(op.peer_rank), str(dst_view.dtype))
            recv_group_bytes.setdefault(key, []).append(num_bytes)

    fused_send_chunks, fused_send_bytes = _estimate_fusion_group_chunks(send_group_bytes, args)
    fused_recv_chunks, fused_recv_bytes = _estimate_fusion_group_chunks(recv_group_bytes, args)
    local_stats = torch.tensor(
        [
            direct_ops,
            direct_bytes,
            candidate_direct_ops,
            candidate_direct_bytes,
            fused_send_chunks + fused_recv_chunks,
            fused_send_bytes + fused_recv_bytes,
        ],
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        dtype=torch.float64,
    )
    dist.all_reduce(local_stats, op=dist.ReduceOp.SUM, group=group)

    remote_bandwidth_bytes_s = max(args.scheduler_remote_bandwidth_gbps, 1.0e-9) * 1.0e9 / 8.0
    pack_bandwidth_bytes_s = max(args.reshard_fusion_pack_bandwidth_gbps, 1.0e-9) * 1.0e9 / 8.0
    latency_s = args.scheduler_latency_us * 1.0e-6
    direct_ops_g = float(local_stats[0].item())
    direct_bytes_g = float(local_stats[1].item())
    candidate_direct_ops_g = float(local_stats[2].item())
    candidate_direct_bytes_g = float(local_stats[3].item())
    fused_ops_g = float(local_stats[4].item())
    fused_bytes_g = float(local_stats[5].item())

    direct_s = direct_bytes_g / remote_bandwidth_bytes_s + direct_ops_g * latency_s
    fused_transfer_bytes = (
        candidate_direct_bytes_g
        + fused_bytes_g * max(1.0, args.reshard_fusion_parallelism_penalty)
    )
    fused_s = (
        fused_transfer_bytes / remote_bandwidth_bytes_s
        + (candidate_direct_ops_g + fused_ops_g) * latency_s
        + (2.0 * fused_bytes_g) / pack_bandwidth_bytes_s
        + args.reshard_fusion_control_overhead_us * 1.0e-6
    )
    threshold = max(0.0, args.reshard_fusion_min_benefit_pct) * 0.01
    required_gain_s = direct_s * threshold
    predicted_gain_s = direct_s - fused_s
    accept = fused_bytes_g > 0.0 and predicted_gain_s > required_gain_s

    if rank == 0:
        print(
            f"\nFusion auto-gate for {label}: accept={accept} "
            f"direct_s={direct_s:.6f} fused_s={fused_s:.6f} "
            f"predicted_gain_s={predicted_gain_s:.6f} "
            f"required_gain_s={required_gain_s:.6f} "
            f"direct_ops={direct_ops_g:.0f} fused_ops={fused_ops_g:.0f} "
            f"direct_bytes={direct_bytes_g:.0f} fused_bytes={fused_bytes_g:.0f} "
            f"candidate_direct_ops={candidate_direct_ops_g:.0f} "
            f"candidate_direct_bytes={candidate_direct_bytes_g:.0f}",
            flush=True,
        )
        if not accept:
            print(
                f"Fusion disabled by auto-gate for {label}; "
                "falling back to direct NCCL P2P transfer.",
                flush=True,
            )
    return accept


def execute_fused_peer_reshard_plan(
    plan: ReshardPlan,
    src_module: Optional[torch.nn.Module],
    dst_module: Optional[torch.nn.Module],
    service: CopyService,
    args: argparse.Namespace,
    *,
    label: str,
    group=None,
) -> None:
    """Pack same-peer TransferOps into one dtype-homogeneous P2P buffer.

    The migration amount is still exactly the view-intersection plan. This only
    changes execution granularity from many parameter-slice P2P ops to fewer
    peer/dtype buffers, then unpacks received buffers into destination slices.
    """
    if args.reshard_fusion_auto_gate and not _fusion_auto_gate_accepts(
        plan,
        src_module,
        dst_module,
        args,
        label=label,
        group=group,
    ):
        execute_reshard_plan(
            plan,
            src_module,
            dst_module,
            service=service,
            group=group,
        )
        return

    rank = dist.get_rank()
    src_params = dict(src_module.named_parameters()) if src_module is not None else {}
    dst_params = dict(dst_module.named_parameters()) if dst_module is not None else {}

    local_sends: dict[int, torch.Tensor] = {}
    remote_send_groups: dict[tuple[int, str], list[tuple[int, torch.Tensor, int]]] = {}
    remote_recv_groups: dict[
        tuple[int, str], list[tuple[int, torch.nn.Parameter, tuple[slice, ...], int]]
    ] = {}
    direct_recv_writebacks: list[tuple[torch.Tensor, torch.nn.Parameter, tuple[slice, ...]]] = []

    local_bytes = 0
    remote_send_bytes = 0
    remote_recv_bytes = 0
    fused_remote_send_bytes = 0
    fused_remote_recv_bytes = 0
    direct_remote_send_bytes = 0
    direct_remote_recv_bytes = 0
    remote_send_count = 0
    remote_recv_count = 0
    direct_remote_send_count = 0
    direct_remote_recv_count = 0
    local_copy_count = 0
    send_chunk_metadata: list[tuple] = []
    recv_chunk_metadata: list[tuple] = []

    hybrid_small_op_bytes = (
        args.reshard_fusion_small_op_bytes if args.reshard_fusion == "adaptive-peer" else 0
    )

    def _use_direct_remote(num_bytes: int) -> bool:
        return args.reshard_fusion == "adaptive-peer" and hybrid_small_op_bytes > 0 and (
            num_bytes > hybrid_small_op_bytes
        )

    for op in plan.send_ops:
        src_param = src_params.get(op.param_name)
        if src_param is None:
            continue
        src_view = src_param.data[op.my_slice]
        numel = src_view.numel()
        num_bytes = numel * src_view.element_size()
        if op.peer_rank == rank:
            if op.task_id is None:
                raise RuntimeError(f"{label}: local fused send requires task_id")
            local_sends[int(op.task_id)] = src_view
            local_bytes += num_bytes
            continue
        if op.task_id is None:
            raise RuntimeError(f"{label}: remote fused send requires task_id")
        remote_send_count += 1
        remote_send_bytes += num_bytes
        if _use_direct_remote(num_bytes):
            send_tensor = src_view if src_view.is_contiguous() else src_view.contiguous()
            service.submit_send(send_tensor, op.peer_rank, task_id=op.task_id)
            direct_remote_send_count += 1
            direct_remote_send_bytes += num_bytes
            send_chunk_metadata.append(
                (
                    rank,
                    int(op.peer_rank),
                    str(send_tensor.dtype),
                    (int(op.task_id),),
                    num_bytes,
                    "direct",
                )
            )
            continue
        key = (int(op.peer_rank), str(src_view.dtype))
        remote_send_groups.setdefault(key, []).append(
            (int(op.task_id), src_view.contiguous().view(-1), num_bytes)
        )
        fused_remote_send_bytes += num_bytes

    local_recvs: list[tuple[int, torch.nn.Parameter, tuple[slice, ...]]] = []
    for op in plan.recv_ops:
        dst_param = dst_params.get(op.param_name)
        if dst_param is None:
            continue
        dst_view = dst_param.data[op.my_slice]
        numel = dst_view.numel()
        num_bytes = numel * dst_view.element_size()
        if op.peer_rank == rank:
            if op.task_id is None:
                raise RuntimeError(f"{label}: local fused recv requires task_id")
            local_recvs.append((int(op.task_id), dst_param, op.my_slice))
            local_bytes += num_bytes
            continue
        if op.task_id is None:
            raise RuntimeError(f"{label}: remote fused recv requires task_id")
        remote_recv_count += 1
        remote_recv_bytes += num_bytes
        if _use_direct_remote(num_bytes):
            recv_metadata = (
                int(op.peer_rank),
                rank,
                str(dst_view.dtype),
                (int(op.task_id),),
                num_bytes,
                "direct",
            )
            if dst_view.is_contiguous():
                service.submit_recv(dst_view, op.peer_rank, task_id=op.task_id)
            else:
                recv_buffer = torch.empty_like(dst_view.contiguous())
                service.submit_recv(recv_buffer, op.peer_rank, task_id=op.task_id)
                direct_recv_writebacks.append((recv_buffer, dst_param, op.my_slice))
            direct_remote_recv_count += 1
            direct_remote_recv_bytes += num_bytes
            recv_chunk_metadata.append(recv_metadata)
            continue
        key = (int(op.peer_rank), str(dst_view.dtype))
        remote_recv_groups.setdefault(key, []).append((int(op.task_id), dst_param, op.my_slice, numel))
        fused_remote_recv_bytes += num_bytes

    local_copy_start = time.perf_counter()
    with torch.no_grad():
        for task_id, dst_param, dst_slice in local_recvs:
            src_view = local_sends.get(task_id)
            if src_view is None:
                raise RuntimeError(f"{label}: missing local fused send for task_id={task_id}")
            dst_param.data[dst_slice].copy_(src_view)
            local_copy_count += 1
    local_copy_s = reduce_max_seconds(time.perf_counter() - local_copy_start)

    pack_start = time.perf_counter()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    chunk_bytes = _global_adaptive_chunk_bytes(
        (
            _adaptive_fusion_chunk_bytes(
                fused_remote_send_bytes,
                len(remote_send_groups),
                args,
            )
            if remote_send_groups
            else 0
        ),
        device,
        group=group,
    )

    send_buffers = []
    send_chunk_ids_by_key: dict[tuple[int, str], list[tuple[int, ...]]] = {}
    for (peer_rank, _dtype_name), entries in sorted(remote_send_groups.items()):
        key = (peer_rank, _dtype_name)
        for task_ids, send_buffer in _pack_fusion_send_chunks(entries, chunk_bytes):
            send_chunk_ids_by_key.setdefault(key, []).append(tuple(task_ids))
            send_buffers.append(send_buffer)
            num_bytes = send_buffer.numel() * send_buffer.element_size()
            send_chunk_metadata.append(
                (rank, peer_rank, _dtype_name, tuple(task_ids), num_bytes, "fused")
            )
            service.submit_send(send_buffer, peer_rank, task_id=task_ids[0])

    recv_writebacks = []
    recv_chunk_ids_by_key: dict[tuple[int, str], list[tuple[int, ...]]] = {}
    for (peer_rank, dtype_name), entries in sorted(remote_recv_groups.items()):
        key = (peer_rank, dtype_name)
        for task_ids, chunk_entries in _pack_fusion_recv_chunks(entries, chunk_bytes):
            recv_chunk_ids_by_key.setdefault(key, []).append(tuple(task_ids))
            first_param = chunk_entries[0][0]
            total_numel = sum(numel for _, _, numel in chunk_entries)
            recv_buffer = torch.empty(total_numel, device=first_param.device, dtype=first_param.dtype)
            recv_chunk_metadata.append(
                (
                    peer_rank,
                    rank,
                    dtype_name,
                    tuple(task_ids),
                    total_numel * first_param.element_size(),
                    "fused",
                )
            )
            service.submit_recv(recv_buffer, peer_rank, task_id=task_ids[0])
            recv_writebacks.append((dtype_name, recv_buffer, chunk_entries))
    pack_submit_s = reduce_max_seconds(time.perf_counter() - pack_start)

    mismatched_chunks = []
    for key, recv_chunks in recv_chunk_ids_by_key.items():
        send_chunks = send_chunk_ids_by_key.get(key)
        if send_chunks is not None and send_chunks != recv_chunks:
            mismatched_chunks.append((key, send_chunks[:3], recv_chunks[:3]))
    if mismatched_chunks:
        sample = "; ".join(
            f"key={key} send={send_chunks} recv={recv_chunks}"
            for key, send_chunks, recv_chunks in mismatched_chunks[:3]
        )
        raise RuntimeError(f"{label}: adaptive fused chunk task_id mismatch: {sample}")

    metadata_start = time.perf_counter()
    if args.reshard_fusion == "adaptive-peer":
        _validate_fused_chunk_metadata(
            send_chunk_metadata,
            recv_chunk_metadata,
            label=label,
            group=group,
        )
    metadata_s = reduce_max_seconds(time.perf_counter() - metadata_start)

    nccl_start = time.perf_counter()
    service.run()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    nccl_sync_s = reduce_max_seconds(time.perf_counter() - nccl_start)

    unpack_start = time.perf_counter()
    with torch.no_grad():
        for recv_buffer, dst_param, dst_slice in direct_recv_writebacks:
            dst_param.data[dst_slice].copy_(recv_buffer.view(dst_param.data[dst_slice].shape))
        for _dtype_name, recv_buffer, entries in recv_writebacks:
            offset = 0
            for dst_param, dst_slice, numel in entries:
                dst_view = dst_param.data[dst_slice]
                chunk = recv_buffer[offset : offset + numel].view(dst_view.shape)
                dst_view.copy_(chunk)
                offset += numel
    unpack_submit_s = reduce_max_seconds(time.perf_counter() - unpack_start)

    stats = torch.tensor(
        [
            len(plan.send_ops),
            len(plan.recv_ops),
            remote_send_count,
            remote_recv_count,
            len(remote_send_groups),
            len(remote_recv_groups),
            len(send_buffers),
            len(recv_writebacks),
            chunk_bytes,
            local_copy_count,
            local_bytes,
            remote_send_bytes,
            remote_recv_bytes,
            direct_remote_send_count,
            direct_remote_recv_count,
            direct_remote_send_bytes,
            direct_remote_recv_bytes,
            fused_remote_send_bytes,
            fused_remote_recv_bytes,
            hybrid_small_op_bytes,
        ],
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    chunk_stats = torch.tensor(
        [chunk_bytes, chunk_bytes],
        device=device,
        dtype=torch.long,
    )
    dist.all_reduce(chunk_stats[0:1], op=dist.ReduceOp.MIN)
    dist.all_reduce(chunk_stats[1:2], op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"\nFused peer transfer for {label}: "
            f"original_sends={stats[0].item()} original_recvs={stats[1].item()} "
            f"original_remote_sends={stats[2].item()} original_remote_recvs={stats[3].item()} "
            f"peer_groups_send={stats[4].item()} peer_groups_recv={stats[5].item()} "
            f"fused_remote_sends={stats[6].item()} fused_remote_recvs={stats[7].item()} "
            f"chunk_bytes_min={chunk_stats[0].item()} chunk_bytes_max={chunk_stats[1].item()} "
            f"direct_remote_sends={stats[13].item()} direct_remote_recvs={stats[14].item()} "
            f"direct_remote_bytes={stats[15].item() + stats[16].item()} "
            f"fused_input_remote_bytes={stats[17].item() + stats[18].item()} "
            f"hybrid_small_op_bytes={stats[19].item()} "
            f"local_copies={stats[9].item()} local_bytes={stats[10].item()} "
            f"remote_bytes={stats[11].item() + stats[12].item()}",
            flush=True,
        )
        print(
            f"Fused peer timing for {label}: "
            f"local_copy_s={local_copy_s:.6f} pack_submit_s={pack_submit_s:.6f} "
            f"metadata_s={metadata_s:.6f} nccl_sync_s={nccl_sync_s:.6f} "
            f"unpack_submit_s={unpack_submit_s:.6f}",
            flush=True,
        )


def execute_scheduled_reshard_waves(
    wave_plans: list[ReshardPlan],
    src_module: Optional[torch.nn.Module],
    dst_module: Optional[torch.nn.Module],
    service: CopyService,
    *,
    args: Optional[argparse.Namespace] = None,
    label: str,
    group=None,
    synchronize_group: bool = True,
) -> None:
    for wave_index, wave_plan in enumerate(wave_plans):
        is_last_wave = wave_index + 1 == len(wave_plans)
        if len(wave_plans) > 1 and dist.get_rank() == 0:
            print(
                f"Executing scheduled wave {wave_index + 1}/{len(wave_plans)} for {label}: "
                f"sends={len(wave_plan.send_ops)} recvs={len(wave_plan.recv_ops)}",
                flush=True,
            )
        maybe_inject_slow_link_delay(
            wave_plan,
            src_module,
            dst_module,
            args,
            label=label if len(wave_plans) == 1 else f"{label} wave {wave_index + 1}",
            group=group,
        )
        execute_reshard_plan(
            wave_plan,
            src_module,
            dst_module,
            service=service,
            group=group,
            # NCCL operations are submitted in the same wave order on every
            # rank. Keep intermediate waves on the stream and pay the global
            # completion cost only once at the end of the migration.
            synchronize_group=synchronize_group and is_last_wave,
            synchronize_device=is_last_wave,
            release_cache=(
                bool(args.scheduler_release_cache) and is_last_wave
                if args is not None
                else is_last_wave
            ),
        )


def reuse_overlapping_weight_storage(
    src_model: Optional[torch.nn.Module],
    dst_model: torch.nn.Module,
    plan: ReshardPlan,
    *,
    apply_aliases: bool,
    allow_noncontiguous: bool,
) -> ReshardPlan:
    """Alias destination parameters to local source views when the plan permits it.

    This is deliberately conservative: one destination parameter can be storage-reused only when
    its whole local shard comes from one same-rank source slice. Multi-source gathers and remote
    sends still use the normal reshard copy path.
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    src_params = dict(src_model.named_parameters()) if src_model is not None else {}
    dst_params = dict(dst_model.named_parameters())
    recv_ops_by_param: dict[str, list] = {}
    for op in plan.recv_ops:
        recv_ops_by_param.setdefault(op.param_name, []).append(op)

    skipped_task_ids: set[int] = set()
    local_stats = {
        "total_dst_bytes": 0,
        "aliasable_bytes": 0,
        "aliased_bytes": 0,
        "aliased_params": 0,
        "copy_bytes": 0,
        "blocked_multi_source_bytes": 0,
        "blocked_remote_bytes": 0,
        "blocked_partial_bytes": 0,
        "blocked_noncontiguous_bytes": 0,
        "blocked_other_bytes": 0,
    }

    def _nbytes(param: torch.Tensor) -> int:
        return param.numel() * param.element_size()

    for name, dst_param in dst_params.items():
        param_bytes = _nbytes(dst_param)
        local_stats["total_dst_bytes"] += param_bytes
        ops = recv_ops_by_param.get(name, [])
        if len(ops) != 1:
            local_stats["blocked_multi_source_bytes"] += param_bytes
            local_stats["copy_bytes"] += param_bytes
            continue

        op = ops[0]
        if op.peer_rank != rank:
            local_stats["blocked_remote_bytes"] += param_bytes
            local_stats["copy_bytes"] += param_bytes
            continue
        if not _is_full_slice(op.my_slice, tuple(dst_param.shape)):
            local_stats["blocked_partial_bytes"] += param_bytes
            local_stats["copy_bytes"] += param_bytes
            continue

        src_param = src_params.get(name)
        if src_param is None:
            local_stats["blocked_other_bytes"] += param_bytes
            local_stats["copy_bytes"] += param_bytes
            continue

        src_view = src_param.detach()[op.peer_slice]
        if (
            tuple(src_view.shape) != tuple(dst_param.shape)
            or src_view.dtype != dst_param.dtype
            or src_view.device != dst_param.device
        ):
            local_stats["blocked_other_bytes"] += param_bytes
            local_stats["copy_bytes"] += param_bytes
            continue

        if not src_view.is_contiguous() and not allow_noncontiguous:
            local_stats["blocked_noncontiguous_bytes"] += param_bytes
            local_stats["copy_bytes"] += param_bytes
            continue

        local_stats["aliasable_bytes"] += param_bytes
        if apply_aliases:
            new_param = _replace_parameter_with_view(dst_model, name, src_view, dst_param)
            if new_param.untyped_storage().data_ptr() != src_param.untyped_storage().data_ptr():
                raise RuntimeError(f"{name}: replacement parameter does not share source storage")
            skipped_task_ids.add(op.task_id)
            local_stats["aliased_bytes"] += param_bytes
            local_stats["aliased_params"] += 1
        else:
            local_stats["copy_bytes"] += param_bytes

    ordered_keys = list(local_stats)
    stats_tensor = torch.tensor([local_stats[key] for key in ordered_keys], device=device)
    dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        stats = {key: int(value.item()) for key, value in zip(ordered_keys, stats_tensor)}
        total = max(stats["total_dst_bytes"], 1)
        aliasable_pct = 100.0 * stats["aliasable_bytes"] / total
        aliased_pct = 100.0 * stats["aliased_bytes"] / total
        mode = "applied" if apply_aliases else "analysis"
        print(f"\nStorage reuse {mode} for model weights:", flush=True)
        print(
            f"  total_dst_bytes={stats['total_dst_bytes']} "
            f"aliasable_bytes={stats['aliasable_bytes']} ({aliasable_pct:.2f}%) "
            f"aliased_bytes={stats['aliased_bytes']} ({aliased_pct:.2f}%) "
            f"aliased_params={stats['aliased_params']}",
            flush=True,
        )
        print(
            f"  copy_bytes={stats['copy_bytes']} "
            f"blocked_remote_bytes={stats['blocked_remote_bytes']} "
            f"blocked_multi_source_bytes={stats['blocked_multi_source_bytes']} "
            f"blocked_noncontiguous_bytes={stats['blocked_noncontiguous_bytes']} "
            f"blocked_partial_bytes={stats['blocked_partial_bytes']} "
            f"blocked_other_bytes={stats['blocked_other_bytes']}",
            flush=True,
        )

    return _filter_reshard_plan(plan, skipped_task_ids)


def _expected_optimizer_state_tensor(
    name: str,
    model_param: torch.nn.Parameter,
    pg_collection: ProcessGroupCollection,
    state_key: str,
) -> torch.Tensor:
    """Build the logical expected local optimizer-state shard for one parameter."""
    if getattr(model_param, "partition_sizes", None) is not None:
        raise NotImplementedError("This demo does not cover partition_sizes-packed params yet")

    local_shape = tuple(model_param.shape)
    full_shape = list(local_shape)
    is_tp = bool(getattr(model_param, "tensor_model_parallel", False))
    partition_dim = int(getattr(model_param, "partition_dim", 0))
    partition_stride = int(getattr(model_param, "partition_stride", 1))
    tp_size = dist.get_world_size(pg_collection.tp) if is_tp else 1

    if is_tp:
        full_shape[partition_dim] *= tp_size

    full_numel = math.prod(full_shape)
    multiplier_by_key = {
        "exp_avg": 0.001,
        "exp_avg_sq": 0.002,
        "param": 0.003,
    }
    if state_key not in multiplier_by_key:
        raise ValueError(f"Unsupported optimizer state key: {state_key}")
    multiplier = multiplier_by_key[state_key]
    base = _stable_name_offset(name, state_key)
    full_tensor = (
        torch.arange(full_numel, device=model_param.device, dtype=torch.float32)
        .reshape(full_shape)
        .mul(multiplier)
        .add(base)
    )

    if not is_tp:
        return full_tensor.contiguous()

    tp_rank = dist.get_rank(pg_collection.tp)
    local_dim = local_shape[partition_dim]
    if local_dim % partition_stride != 0:
        raise RuntimeError(
            f"{name}: local dim {local_dim} is not divisible by stride {partition_stride}"
        )

    segment_len = local_dim // partition_stride
    chunks = []
    for stride_idx in range(partition_stride):
        global_segment = tp_rank + stride_idx * tp_size
        start = global_segment * segment_len
        chunks.append(full_tensor.narrow(partition_dim, start, segment_len))
    return torch.cat(chunks, dim=partition_dim).contiguous()


def _init_adamw_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pg_collection: ProcessGroupCollection,
    *,
    fill_expected: bool,
    step: int,
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            state = optimizer.state[param]
            state["step"] = torch.tensor(float(step), device=param.device)
            for state_key in OPTIMIZER_TENSOR_STATE_KEYS:
                expected = _expected_optimizer_state_tensor(name, param, pg_collection, state_key)
                state[state_key] = expected if fill_expected else torch.zeros_like(expected)


def _optimizer_state_module(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pg_collection: ProcessGroupCollection,
    state_key: str,
) -> OptimizerStateTensorModule:
    named_state_params = []
    for name, model_param in model.named_parameters():
        state_tensor = optimizer.state[model_param][state_key]
        state_param = torch.nn.Parameter(state_tensor, requires_grad=False)
        if state_param.data_ptr() != state_tensor.data_ptr():
            raise RuntimeError(f"{name}.{state_key}: optimizer state wrapper does not alias state")
        _copy_tp_metadata(model_param, state_param)
        named_state_params.append((name, state_param))
    return OptimizerStateTensorModule(pg_collection, named_state_params)


def _max_optimizer_state_diff(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    pg_collection: ProcessGroupCollection,
) -> float:
    local_max = torch.zeros((), device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    with torch.no_grad():
        for name, param in model.named_parameters():
            for state_key in OPTIMIZER_TENSOR_STATE_KEYS:
                expected = _expected_optimizer_state_tensor(name, param, pg_collection, state_key)
                actual = optimizer.state[param][state_key]
                diff = (actual - expected).abs().max()
                local_max = torch.maximum(local_max, diff)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    return local_max.item()


def _copy_optimizer_step_scalar(
    src_model: torch.nn.Module,
    src_optimizer: torch.optim.Optimizer,
    dst_model: torch.nn.Module,
    dst_optimizer: torch.optim.Optimizer,
) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    step_tensor = torch.zeros((), dtype=torch.float32, device=device)
    if dist.get_rank() == 0:
        first_src_param = next(src_model.parameters())
        step_tensor.copy_(src_optimizer.state[first_src_param]["step"].to(device))
    dist.broadcast(step_tensor, src=0)
    for dst_param in dst_model.parameters():
        dst_optimizer.state[dst_param]["step"] = step_tensor.detach().clone()


def verify_optimizer_state_reshard(
    src_model: torch.nn.Module,
    dst_model: torch.nn.Module,
    src_pg_collection: ProcessGroupCollection,
    dst_pg_collection: ProcessGroupCollection,
    args: argparse.Namespace,
) -> None:
    """Reshard AdamW tensor states and compare destination shards to expected values."""
    src_optimizer = torch.optim.AdamW(src_model.parameters(), lr=1e-3)
    dst_optimizer = torch.optim.AdamW(dst_model.parameters(), lr=1e-3)

    _init_adamw_state(
        src_model,
        src_optimizer,
        src_pg_collection,
        fill_expected=True,
        step=args.optimizer_step,
    )
    _init_adamw_state(
        dst_model,
        dst_optimizer,
        dst_pg_collection,
        fill_expected=False,
        step=0,
    )

    before_diff = _max_optimizer_state_diff(dst_model, dst_optimizer, dst_pg_collection)
    if dist.get_rank() == 0:
        print(f"\nBefore optimizer-state swap max diff: {before_diff:.6f}", flush=True)

    service = make_plan_copy_service(args)
    for state_key in OPTIMIZER_TENSOR_STATE_KEYS:
        src_state_module = _optimizer_state_module(
            src_model, src_optimizer, src_pg_collection, state_key
        )
        dst_state_module = _optimizer_state_module(
            dst_model, dst_optimizer, dst_pg_collection, state_key
        )
        plan = build_centralized_reshard_plan(
            src_state_module,
            dst_state_module,
            num_experts=None,
            prefer_local_source=prefer_local_source(args),
        )
        execute_reshard_plan(
            plan,
            src_state_module,
            dst_state_module,
            service=service,
        )
    if isinstance(service, TracingCopyService):
        service.print_summary("AdamW optimizer state")

    _copy_optimizer_step_scalar(src_model, src_optimizer, dst_model, dst_optimizer)
    dist.barrier()

    after_diff = _max_optimizer_state_diff(dst_model, dst_optimizer, dst_pg_collection)
    is_close = after_diff <= 1e-6
    if dist.get_rank() == 0:
        print(f"After optimizer-state swap max diff:  {after_diff:.6f}", flush=True)
        print(f"Optimizer state check passed:        {is_close}", flush=True)

    assert is_close, (
        f"optimizer state reshard failed: src_tp={args.src_tp}, dst_tp={args.dst_tp}, "
        f"backend={args.refit_backend}, max_diff={after_diff}"
    )


def _unwrap_single_distributed_optimizer(megatron_optimizer):
    optimizers = getattr(megatron_optimizer, "chained_optimizers", [megatron_optimizer])
    if len(optimizers) != 1:
        raise RuntimeError(f"Expected one optimizer shard, got {len(optimizers)}")
    return optimizers[0]


def _build_megatron_distributed_optimizer(
    model: torch.nn.Module,
    config: TransformerConfig,
    pg_collection: ProcessGroupCollection,
):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=True,
    )
    ddp_model = DistributedDataParallel(
        config,
        ddp_config,
        model,
        pg_collection=pg_collection,
    )
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=1e-3,
        use_distributed_optimizer=True,
    )
    megatron_optimizer = get_megatron_optimizer(
        optimizer_config,
        [ddp_model],
        use_gloo_process_groups=False,
        pg_collection=pg_collection,
    )
    dist_optimizer = _unwrap_single_distributed_optimizer(megatron_optimizer)
    dist_optimizer.init_state_fn(dist_optimizer.optimizer, dist_optimizer.config)
    return ddp_model, megatron_optimizer, dist_optimizer


def _set_distributed_optimizer_step(dist_optimizer, step: int) -> None:
    for param_group in dist_optimizer.optimizer.param_groups:
        for param in param_group["params"]:
            dist_optimizer.optimizer.state[param]["step"] = torch.tensor(
                float(step), device=param.device
            )


def _init_distributed_optimizer_state(
    ddp_model: DistributedDataParallel,
    dist_optimizer,
    pg_collection: ProcessGroupCollection,
    *,
    fill_expected: bool,
    step: int,
) -> None:
    param_to_name = {param: name for name, param in ddp_model.named_parameters()}
    _set_distributed_optimizer_step(dist_optimizer, step)

    with torch.no_grad():
        for gbuf_range_maps in dist_optimizer.gbuf_ranges:
            for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():
                        name = param_to_name[model_param]
                        tensors = dist_optimizer._get_main_param_and_optimizer_states(
                            model_param
                        )
                        param_range = param_range_map["param"]
                        for state_key in DISTRIBUTED_OPTIMIZER_TENSOR_STATE_KEYS:
                            expected = _expected_optimizer_state_tensor(
                                name, model_param, pg_collection, state_key
                            ).flatten()
                            shard = expected[param_range.start : param_range.end]
                            if fill_expected:
                                tensors[state_key].copy_(shard)
                            else:
                                tensors[state_key].zero_()


def _distributed_optimizer_model_space_state_module(
    ddp_model: DistributedDataParallel,
    dist_optimizer,
    pg_collection: ProcessGroupCollection,
    state_key: str,
) -> OptimizerStateTensorModule:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    param_to_name = {param: name for name, param in ddp_model.named_parameters()}
    dp_zero_state = dist_optimizer.get_parameter_state_dp_zero(
        use_gloo_comm=False,
        return_on_all_ranks=True,
    )

    named_state_params = []
    for gbuf_idx, buffer in enumerate(dist_optimizer.buffers):
        gbuf_range_maps = dist_optimizer.gbuf_ranges[gbuf_idx]
        if len(gbuf_range_maps) != 1:
            raise RuntimeError("This demo expects one dtype per distributed optimizer buffer")
        dtype = next(iter(gbuf_range_maps.keys()))
        world_tensors = dp_zero_state[gbuf_idx][dtype]
        for model_param, (start, end, _) in buffer.param_index_map.items():
            name = param_to_name[model_param]
            state_tensor = (
                world_tensors[state_key][start:end]
                .to(device=device, dtype=torch.float32)
                .reshape(model_param.shape)
                .contiguous()
            )
            state_param = torch.nn.Parameter(state_tensor, requires_grad=False)
            _copy_tp_metadata(model_param, state_param)
            named_state_params.append((name, state_param))
    return OptimizerStateTensorModule(pg_collection, named_state_params)


def _distributed_optimizer_direct_state_module(
    ddp_model: DistributedDataParallel,
    dist_optimizer,
    pg_collection: ProcessGroupCollection,
    state_key: str,
) -> OptimizerStateTensorModule:
    if dist_optimizer.data_parallel_group.size() != 1:
        raise NotImplementedError(
            "Direct destination writeback currently requires destination DP size 1. "
            "Use --dst-tp equal to WORLD_SIZE for this demo."
        )

    named_state_params = []
    for name, model_param in ddp_model.named_parameters():
        tensors = dist_optimizer._get_main_param_and_optimizer_states(model_param)
        state_tensor = tensors[state_key].view_as(model_param)
        state_param = torch.nn.Parameter(state_tensor, requires_grad=False)
        if state_param.data_ptr() != tensors[state_key].data_ptr():
            raise RuntimeError(f"{name}.{state_key}: wrapper does not alias dist optimizer state")
        _copy_tp_metadata(model_param, state_param)
        named_state_params.append((name, state_param))
    return OptimizerStateTensorModule(pg_collection, named_state_params)


def _max_distributed_optimizer_state_diff(
    ddp_model: DistributedDataParallel,
    dist_optimizer,
    pg_collection: ProcessGroupCollection,
) -> float:
    if dist_optimizer.data_parallel_group.size() != 1:
        raise NotImplementedError("Destination distributed optimizer diff requires DP size 1")

    local_max = torch.zeros((), device=torch.device(f"cuda:{torch.cuda.current_device()}"))
    with torch.no_grad():
        for name, model_param in ddp_model.named_parameters():
            tensors = dist_optimizer._get_main_param_and_optimizer_states(model_param)
            for state_key in DISTRIBUTED_OPTIMIZER_TENSOR_STATE_KEYS:
                expected = _expected_optimizer_state_tensor(
                    name, model_param, pg_collection, state_key
                ).flatten()
                actual = tensors[state_key].flatten()
                diff = (actual - expected).abs().max()
                local_max = torch.maximum(local_max, diff)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    return local_max.item()


def _copy_distributed_optimizer_step_scalar(src_dist_optimizer, dst_dist_optimizer) -> None:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    step_tensor = torch.zeros((), dtype=torch.float32, device=device)
    if dist.get_rank() == 0:
        first_group = src_dist_optimizer.optimizer.param_groups[0]
        first_param = first_group["params"][0]
        step_tensor.copy_(src_dist_optimizer.optimizer.state[first_param]["step"].to(device))
    dist.broadcast(step_tensor, src=0)
    _set_distributed_optimizer_step(dst_dist_optimizer, int(step_tensor.item()))


def verify_distributed_optimizer_state_reshard(
    src_model: torch.nn.Module,
    dst_model: torch.nn.Module,
    src_config: TransformerConfig,
    dst_config: TransformerConfig,
    src_pg_collection: ProcessGroupCollection,
    dst_pg_collection: ProcessGroupCollection,
    args: argparse.Namespace,
) -> None:
    """Verify real Megatron DistributedOptimizer main-param and Adam state resharding."""
    dst_dp_size = dist.get_world_size(dst_pg_collection.dp)
    if dst_dp_size != 1:
        raise NotImplementedError(
            f"This first DistributedOptimizer demo requires destination DP size 1, got {dst_dp_size}. "
            "Try --dst-tp equal to torchrun --nproc_per_node."
        )

    src_ddp_model, _, src_dist_optimizer = _build_megatron_distributed_optimizer(
        src_model, src_config, src_pg_collection
    )
    dst_ddp_model, _, dst_dist_optimizer = _build_megatron_distributed_optimizer(
        dst_model, dst_config, dst_pg_collection
    )

    _init_distributed_optimizer_state(
        src_ddp_model,
        src_dist_optimizer,
        src_pg_collection,
        fill_expected=True,
        step=args.optimizer_step,
    )
    _init_distributed_optimizer_state(
        dst_ddp_model,
        dst_dist_optimizer,
        dst_pg_collection,
        fill_expected=False,
        step=0,
    )

    before_diff = _max_distributed_optimizer_state_diff(
        dst_ddp_model, dst_dist_optimizer, dst_pg_collection
    )
    if dist.get_rank() == 0:
        print(f"\nBefore DistributedOptimizer state swap max diff: {before_diff:.6f}", flush=True)

    service = make_plan_copy_service(args)
    for state_key in DISTRIBUTED_OPTIMIZER_TENSOR_STATE_KEYS:
        src_state_module = _distributed_optimizer_model_space_state_module(
            src_ddp_model, src_dist_optimizer, src_pg_collection, state_key
        )
        dst_state_module = _distributed_optimizer_direct_state_module(
            dst_ddp_model, dst_dist_optimizer, dst_pg_collection, state_key
        )
        plan = build_centralized_reshard_plan(
            src_state_module,
            dst_state_module,
            num_experts=None,
            prefer_local_source=prefer_local_source(args),
        )
        execute_reshard_plan(
            plan,
            src_state_module,
            dst_state_module,
            service=service,
        )
    if isinstance(service, TracingCopyService):
        service.print_summary("DistributedOptimizer state")

    _copy_distributed_optimizer_step_scalar(src_dist_optimizer, dst_dist_optimizer)
    dist.barrier()

    after_diff = _max_distributed_optimizer_state_diff(
        dst_ddp_model, dst_dist_optimizer, dst_pg_collection
    )
    is_close = after_diff <= 1e-6
    if dist.get_rank() == 0:
        print(f"After DistributedOptimizer state swap max diff:  {after_diff:.6f}", flush=True)
        print(f"DistributedOptimizer state check passed:        {is_close}", flush=True)

    assert is_close, (
        f"DistributedOptimizer state reshard failed: src_tp={args.src_tp}, "
        f"dst_tp={args.dst_tp}, backend={args.refit_backend}, max_diff={after_diff}"
    )


def run_forward(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
) -> torch.Tensor:
    """Run a deterministic forward pass on every rank."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    generator = torch.Generator(device=device)
    generator.manual_seed(2026)
    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    position_ids = (
        torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )
    attention_mask = torch.ones(
        (batch_size, 1, seq_len, seq_len),
        device=device,
        dtype=torch.bool,
    )

    return model(tokens, position_ids, attention_mask)


def verify_swap(src_model: torch.nn.Module, dst_model: torch.nn.Module, args: argparse.Namespace) -> None:
    """Swap source weights into destination model and compare forward outputs."""
    dist.barrier()

    with torch.no_grad():
        src_logits = run_forward(
            src_model,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.micro_batch_size,
        )
        dst_logits_before = run_forward(
            dst_model,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.micro_batch_size,
        )

    before_diff = (src_logits - dst_logits_before).abs().max().item()
    if dist.get_rank() == 0:
        print(f"\nBefore swap max diff: {before_diff:.6f}", flush=True)

    refit_method = make_refit_method(args)
    if args.force_remote_transfer or args.reuse_overlap_weights or args.print_intersection_plan:
        service = (
            refit_method
            if isinstance(refit_method, CopyService)
            else get_or_create_service(refit_method)
        )
        plan = build_centralized_reshard_plan(
            src_model,
            dst_model,
            num_experts=src_model.config.num_moe_experts,
            prefer_local_source=prefer_local_source(args),
        )
        if args.print_intersection_plan:
            print_intersection_plan(
                plan,
                dst_model,
                label="model weights",
                limit=args.intersection_plan_limit,
            )
        if args.reuse_overlap_weights:
            plan = reuse_overlapping_weight_storage(
                src_model,
                dst_model,
                plan,
                apply_aliases=True,
                allow_noncontiguous=args.allow_noncontiguous_storage_reuse,
            )
        execute_reshard_plan_with_scheduler(
            plan,
            src_model,
            dst_model,
            service,
            args,
            label="model weights",
        )
        refit_method = service
    else:
        swap_model_weights([src_model], [dst_model], refit_method=refit_method)
    dist.barrier()
    if isinstance(refit_method, TracingCopyService):
        refit_method.print_summary("model weights")

    with torch.no_grad():
        dst_logits_after = run_forward(
            dst_model,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.micro_batch_size,
        )

    after_diff = (src_logits - dst_logits_after).abs().max().item()
    is_close = torch.allclose(dst_logits_after, src_logits, atol=5e-4, rtol=5e-4)

    if dist.get_rank() == 0:
        print(f"After swap max diff:  {after_diff:.6f}", flush=True)
        print(f"Swap check passed:    {is_close}", flush=True)

    assert is_close, (
        f"swap_model_weights failed: src_tp={args.src_tp}, dst_tp={args.dst_tp}, "
        f"backend={args.refit_backend}, max_diff={after_diff}"
    )


def _broadcast_active_source_logits(
    src_model: Optional[torch.nn.Module],
    *,
    src_root_rank: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Run source forward on active ranks and broadcast rank-root logits to joining ranks."""
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    if src_model is not None:
        src_logits = run_forward(
            src_model,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.micro_batch_size,
        )
    else:
        src_logits = torch.empty(
            (args.micro_batch_size, args.seq_len, args.vocab_size),
            device=device,
            dtype=torch.float32,
        )
    dist.broadcast(src_logits, src=src_root_rank)
    return src_logits


def _train_one_live_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    logits = run_forward(
        model,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.micro_batch_size,
    )
    loss = logits.float().square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


class LiveMigrationThread:
    """Run one reshard plan while the main thread keeps training."""

    def __init__(
        self,
        *,
        label: str,
        plan: ReshardPlan,
        src_module: Optional[torch.nn.Module],
        dst_module: Optional[torch.nn.Module],
        args: argparse.Namespace,
        group,
        wave_plans: Optional[list[ReshardPlan]] = None,
        synchronize_group: bool = True,
        print_summary: bool = True,
    ) -> None:
        self.label = label
        self.plan = plan
        self.wave_plans = wave_plans
        self.src_module = src_module
        self.dst_module = dst_module
        self.args = args
        self.group = group
        self.synchronize_group = synchronize_group
        self.should_print_summary = print_summary
        self.device_index = torch.cuda.current_device()
        self.done = threading.Event()
        self.launched = threading.Event()
        self.elapsed_s = 0.0
        self.error_trace: Optional[str] = None
        self.thread = threading.Thread(target=self._run, name=f"{label}-reshard", daemon=False)

    def start(self) -> None:
        self.thread.start()

    def join(self) -> None:
        self.thread.join()
        if self.error_trace is not None:
            raise RuntimeError(
                f"Background migration failed on rank {dist.get_rank()}:\n{self.error_trace}"
            )

    def wait_until_launched(self, timeout_s: float = 60.0) -> None:
        if not self.launched.wait(timeout=timeout_s):
            raise RuntimeError(
                f"Background migration did not submit NCCL operations within {timeout_s}s "
                f"on rank {dist.get_rank()}"
            )
        if self.error_trace is not None:
            self.join()

    def _run(self) -> None:
        torch.cuda.set_device(self.device_index)
        start = time.perf_counter()
        try:
            service = make_plan_copy_service(self.args, group=self.group, uncached=True)
            if hasattr(service, "set_launch_callback"):
                service.set_launch_callback(self.launched.set)
            else:
                self.launched.set()
            wave_plans = self.wave_plans
            if wave_plans is None:
                wave_plans = [self.plan]
            execute_scheduled_reshard_waves(
                wave_plans,
                self.src_module,
                self.dst_module,
                service,
                args=self.args,
                label=self.label,
                group=self.group,
                synchronize_group=self.synchronize_group,
            )
            if self.should_print_summary and isinstance(service, TracingCopyService):
                service.print_summary(self.label)
        except BaseException:
            self.error_trace = traceback.format_exc()
        finally:
            self.launched.set()
            self.elapsed_s = time.perf_counter() - start
            self.done.set()


def simulate_atomic_switch(
    *,
    active_model: Optional[torch.nn.Module],
    shadow_model: Optional[torch.nn.Module],
    active_ranks: list[int],
    shadow_ranks: list[int],
    reference_logits: torch.Tensor,
    label: str,
    args: argparse.Namespace,
) -> tuple[float, bool, dict[str, float]]:
    """Simulate FluidScale's P_current <- P_shadow switch at an iteration boundary.

    This is intentionally a pointer/reference update in the demo. The expensive
    work is the earlier shadow model construction and reshard transfer; the switch
    itself only changes which already-initialized model/process groups are used by
    the next iteration.
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    is_shadow_rank = rank in shadow_ranks
    if is_shadow_rank and shadow_model is None:
        raise RuntimeError(f"Rank {rank} is in shadow_ranks but has no shadow model")

    current_generation = 0
    current_model = active_model if rank in active_ranks else None
    current_ranks = active_ranks

    if rank == 0:
        print(f"\nAtomic switch simulation for {label}:", flush=True)
        print(
            f"  Stable(G{current_generation}): active_ranks={active_ranks}; "
            f"Shadow(G{current_generation + 1}): shadow_ranks={shadow_ranks}",
            flush=True,
        )
        print("  Consistent cut: waiting at iteration boundary after reshard.", flush=True)

    barrier_start = time.perf_counter()
    dist.barrier()
    switch_entry_barrier_s = reduce_max_seconds(time.perf_counter() - barrier_start)

    ready_start = time.perf_counter()
    shadow_ready = torch.tensor(
        1 if (not is_shadow_rank or shadow_model is not None) else 0,
        device=device,
        dtype=torch.int,
    )
    dist.all_reduce(shadow_ready, op=dist.ReduceOp.MIN)
    shadow_ready_reduce_s = reduce_max_seconds(time.perf_counter() - ready_start)
    if not bool(shadow_ready.item()):
        raise RuntimeError("Shadow world is not ready on all destination ranks")

    switch_start = time.perf_counter()
    current_model = shadow_model if is_shadow_rank else None
    current_ranks = shadow_ranks
    current_generation += 1
    switch_elapsed_us = (time.perf_counter() - switch_start) * 1.0e6
    elapsed_tensor = torch.tensor(switch_elapsed_us, device=device, dtype=torch.float64)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    pointer_update_s = float(elapsed_tensor.item()) * 1.0e-6

    post_pointer_barrier_start = time.perf_counter()
    dist.barrier()
    post_pointer_barrier_s = reduce_max_seconds(time.perf_counter() - post_pointer_barrier_start)

    if rank == 0:
        print(
            f"  Switch: P_current <- P_shadow; generation=G{current_generation}; "
            f"max_local_pointer_update_us={elapsed_tensor.item():.3f}",
            flush=True,
        )
        print(f"  Next iteration uses ranks={current_ranks}.", flush=True)

    forward_start = time.perf_counter()
    with torch.no_grad():
        if rank in current_ranks:
            switched_logits = run_forward(
                current_model,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                batch_size=args.micro_batch_size,
            )
            switch_diff = (reference_logits - switched_logits).abs().max()
            switch_close = torch.tensor(
                int(torch.allclose(switched_logits, reference_logits, atol=5e-4, rtol=5e-4)),
                device=device,
                dtype=torch.int,
            )
        else:
            switch_diff = torch.zeros((), device=device)
            switch_close = torch.ones((), device=device, dtype=torch.int)
    post_switch_forward_s = reduce_max_seconds(time.perf_counter() - forward_start)

    diff_reduce_start = time.perf_counter()
    dist.all_reduce(switch_diff, op=dist.ReduceOp.MAX)
    dist.all_reduce(switch_close, op=dist.ReduceOp.MIN)
    switch_diff_reduce_s = reduce_max_seconds(time.perf_counter() - diff_reduce_start)

    is_close = bool(switch_close.item())
    if rank == 0:
        print(f"  Post-switch forward max diff: {switch_diff.item():.6f}", flush=True)
        print(f"  Atomic switch check passed:  {is_close}", flush=True)

    assert is_close, (
        f"Atomic switch failed for {label}: max_diff={switch_diff.item()}, "
        f"active_ranks={active_ranks}, shadow_ranks={shadow_ranks}"
    )
    return (
        switch_diff.item(),
        is_close,
        {
            "switch_entry_barrier_s": switch_entry_barrier_s,
            "shadow_ready_reduce_s": shadow_ready_reduce_s,
            "pointer_update_s": pointer_update_s,
            "post_pointer_barrier_s": post_pointer_barrier_s,
            "post_switch_forward_s": post_switch_forward_s,
            "switch_diff_reduce_s": switch_diff_reduce_s,
        },
    )


def verify_live_training_tp_migration(
    src_model: Optional[torch.nn.Module],
    dst_model: Optional[torch.nn.Module],
    args: argparse.Namespace,
    *,
    src_ranks: list[int],
    dst_ranks: list[int],
    label: str,
) -> None:
    """Overlap TP parameter migration with live training on the active world.

    This is the B-path test harness for TP weights only:
    - source ranks keep running real forward/backward/SGD steps;
    - a background thread migrates a base snapshot into the shadow model;
    - parameters updated during the overlap are marked dirty;
    - at the cutover boundary, only dirty parameters are refreshed before the
      atomic pointer switch.
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    src_module = src_model if rank in src_ranks else None
    dst_module = dst_model if rank in dst_ranks else None
    src_root_rank = src_ranks[0]
    representative_module = src_module if src_module is not None else dst_module
    if representative_module is None:
        raise RuntimeError(f"Rank {rank} is neither source nor destination for {label}")

    if rank == 0:
        print(
            f"\nLive-training TP parameter migration for {label}:",
            f"active_ranks={src_ranks}",
            f"shadow_ranks={dst_ranks}",
            f"src_tp={args.src_tp}",
            f"dst_tp={args.dst_tp}",
            f"train_overlap_steps={args.live_train_steps}",
            f"backend={args.refit_backend}",
            flush=True,
        )
        print(
            "  Active ranks keep training while a background reshard copies TP parameter "
            "views into the shadow model. Dirty parameters are refreshed at cutover.",
            flush=True,
        )

    reshard_group_ranks = sorted(set(src_ranks) | set(dst_ranks))
    reshard_group = dist.new_group(ranks=reshard_group_ranks)
    if rank == 0:
        print(
            f"  Dedicated reshard process group ranks={reshard_group_ranks}; "
            "keeps background P2P separate from active training collectives.",
            flush=True,
        )

    plan_start = time.perf_counter()
    base_plan = build_centralized_reshard_plan(
        src_module,
        dst_module,
        num_experts=representative_module.config.num_moe_experts,
        prefer_local_source=prefer_local_source(args),
    )
    plan_build_s = reduce_max_seconds(time.perf_counter() - plan_start)
    print_view_overlap_matrix(
        base_plan,
        dst_module,
        label=f"{label} live base TP weights",
        src_ranks=src_ranks,
        dst_ranks=dst_ranks,
    )
    if args.flexible_tp_plan:
        print_flexible_tp_plan_analysis(
            base_plan,
            src_module,
            dst_module,
            args,
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            label=f"{label} live base TP weights",
        )
    if args.print_intersection_plan:
        print_intersection_plan(
            base_plan,
            dst_module,
            label=f"{label} live base TP weights",
            limit=args.intersection_plan_limit,
        )

    base_wave_plans = schedule_reshard_plan_waves(
        base_plan,
        dst_module,
        args,
        label=f"{label} live base TP weights",
    )
    background = LiveMigrationThread(
        label=f"{label} live base TP weights",
        plan=base_plan,
        src_module=src_module,
        dst_module=dst_module,
        args=args,
        group=reshard_group,
        wave_plans=base_wave_plans,
    )
    dist.barrier()
    background.start()

    dirty_param_names: set[str] = set()
    last_loss = 0.0
    train_start = time.perf_counter()
    optimizer = None
    if src_module is not None:
        src_module.train()
        optimizer = torch.optim.SGD(src_module.parameters(), lr=args.live_train_lr)
        for _ in range(args.live_train_steps):
            last_loss = _train_one_live_step(src_module, optimizer, args)
            dirty_param_names.update(name for name, _ in src_module.named_parameters())
    train_elapsed_s = time.perf_counter() - train_start if src_module is not None else 0.0

    exposed_wait_start = time.perf_counter()
    background.join()
    exposed_wait_s = time.perf_counter() - exposed_wait_start
    dist.barrier()

    if not dirty_param_names:
        dirty_param_names.update(name for name, _ in representative_module.named_parameters())
    refresh_plan = _filter_reshard_plan_by_param_names(base_plan, dirty_param_names)

    train_elapsed = torch.tensor(train_elapsed_s, device=device, dtype=torch.float64)
    exposed_wait = torch.tensor(exposed_wait_s, device=device, dtype=torch.float64)
    migration_elapsed = torch.tensor(background.elapsed_s, device=device, dtype=torch.float64)
    last_loss_tensor = torch.tensor(last_loss, device=device, dtype=torch.float64)
    dirty_count = torch.tensor(len(dirty_param_names), device=device, dtype=torch.long)
    dist.all_reduce(train_elapsed, op=dist.ReduceOp.MAX)
    dist.all_reduce(exposed_wait, op=dist.ReduceOp.MAX)
    dist.all_reduce(migration_elapsed, op=dist.ReduceOp.MAX)
    dist.all_reduce(last_loss_tensor, op=dist.ReduceOp.MAX)
    dist.all_reduce(dirty_count, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            f"\nLive migration overlap summary for {label}:",
            f"base_migration_max_s={migration_elapsed.item():.3f}",
            f"training_overlap_max_s={train_elapsed.item():.3f}",
            f"post_training_wait_max_s={exposed_wait.item():.3f}",
            f"dirty_params={dirty_count.item()}",
            f"last_loss={last_loss_tensor.item():.6f}",
            flush=True,
        )

    with torch.no_grad():
        reference_logits = _broadcast_active_source_logits(
            src_module,
            src_root_rank=src_root_rank,
            args=args,
        )
        if dst_module is not None:
            stale_logits = run_forward(
                dst_module,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                batch_size=args.micro_batch_size,
            )
            stale_diff = (reference_logits - stale_logits).abs().max()
        else:
            stale_diff = torch.zeros((), device=device)
    dist.all_reduce(stale_diff, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            f"Before dirty refresh shadow max diff: {stale_diff.item():.6f}",
            flush=True,
        )

    refresh_service = make_plan_copy_service(args, group=reshard_group, uncached=True)
    refresh_start = time.perf_counter()
    execute_reshard_plan_with_scheduler(
        refresh_plan,
        src_module,
        dst_module,
        refresh_service,
        args,
        label=f"{label} dirty TP refresh",
        group=reshard_group,
    )
    dirty_refresh_s = reduce_max_seconds(time.perf_counter() - refresh_start)
    if isinstance(refresh_service, TracingCopyService):
        refresh_service.print_summary(f"{label} dirty TP refresh")
    dist.barrier()

    switch_start = time.perf_counter()
    after_diff, is_close, switch_metrics = simulate_atomic_switch(
        active_model=src_module,
        shadow_model=dst_module,
        active_ranks=src_ranks,
        shadow_ranks=dst_ranks,
        reference_logits=reference_logits,
        label=f"{label} live training",
        args=args,
    )
    validation_switch_s = reduce_max_seconds(time.perf_counter() - switch_start)
    print_path_timing(
        f"B-live {label}",
        plan_build_s=plan_build_s,
        base_migration_s=migration_elapsed.item(),
        training_overlap_s=train_elapsed.item(),
        post_training_wait_s=exposed_wait.item(),
        dirty_refresh_s=dirty_refresh_s,
        validation_switch_s=validation_switch_s,
        **switch_metrics,
        exposed_pause_s=exposed_wait.item() + dirty_refresh_s + validation_switch_s,
    )

    if rank == 0:
        print(f"After live migration max diff:  {after_diff:.6f}", flush=True)
        print(f"Live migration check passed:   {is_close}", flush=True)

    assert is_close, (
        f"Live TP migration failed for {label}: src_tp={args.src_tp}, "
        f"dst_tp={args.dst_tp}, backend={args.refit_backend}, max_diff={after_diff}"
    )
    dist.destroy_process_group(reshard_group)


def verify_shadow_prep_boundary_transfer(
    src_model: Optional[torch.nn.Module],
    dst_model: Optional[torch.nn.Module],
    args: argparse.Namespace,
    *,
    src_ranks: list[int],
    dst_ranks: list[int],
    label: str,
) -> None:
    """C path: prepare shadow world/plan first, then pause at boundary for one transfer.

    This avoids dirty refresh by freezing active training during the actual TP
    parameter migration. It measures the visible cutover pause separately from
    the already-prepared shadow world and transfer plan.
    """
    rank = dist.get_rank()
    src_module = src_model if rank in src_ranks else None
    dst_module = dst_model if rank in dst_ranks else None
    src_root_rank = src_ranks[0]
    representative_module = src_module if src_module is not None else dst_module
    if representative_module is None:
        raise RuntimeError(f"Rank {rank} is neither source nor destination for {label}")

    if rank == 0:
        print(
            f"\nShadow-prep boundary TP transfer for {label}:",
            f"active_ranks={src_ranks}",
            f"shadow_ranks={dst_ranks}",
            f"src_tp={args.src_tp}",
            f"dst_tp={args.dst_tp}",
            f"backend={args.refit_backend}",
            flush=True,
        )
        print(
            "  C path: shadow model/process groups and transfer plan are prepared before "
            "the cutover; active training pauses only for one TP parameter transfer.",
            flush=True,
        )

    prep_start = time.perf_counter()
    reshard_group_ranks = sorted(set(src_ranks) | set(dst_ranks))
    reshard_group = dist.new_group(ranks=reshard_group_ranks)
    shadow_prepare_s = reduce_max_seconds(time.perf_counter() - prep_start)

    plan_start = time.perf_counter()
    plan = build_centralized_reshard_plan(
        src_module,
        dst_module,
        num_experts=representative_module.config.num_moe_experts,
        prefer_local_source=prefer_local_source(args),
    )
    plan_build_s = reduce_max_seconds(time.perf_counter() - plan_start)
    print_view_overlap_matrix(
        plan,
        dst_module,
        label=f"{label} boundary TP weights",
        src_ranks=src_ranks,
        dst_ranks=dst_ranks,
    )
    if args.flexible_tp_plan:
        print_flexible_tp_plan_analysis(
            plan,
            src_module,
            dst_module,
            args,
            src_ranks=src_ranks,
            dst_ranks=dst_ranks,
            label=f"{label} boundary TP weights",
        )
    if args.print_intersection_plan:
        print_intersection_plan(
            plan,
            dst_module,
            label=f"{label} boundary TP weights",
            limit=args.intersection_plan_limit,
        )

    shadow_warmup_s = 0.0
    if not args.skip_shadow_forward_warmup:
        warmup_start = time.perf_counter()
        with torch.no_grad():
            if dst_module is not None:
                run_forward(
                    dst_module,
                    vocab_size=args.vocab_size,
                    seq_len=args.seq_len,
                    batch_size=args.micro_batch_size,
                )
        torch.cuda.synchronize()
        shadow_warmup_s = reduce_max_seconds(time.perf_counter() - warmup_start)
        if rank == 0:
            print(
                f"Shadow forward warmup for {label}: {shadow_warmup_s:.6f}s "
                "(not counted in exposed pause)",
                flush=True,
            )

    with torch.no_grad():
        reference_logits = _broadcast_active_source_logits(
            src_module,
            src_root_rank=src_root_rank,
            args=args,
        )

    dist.barrier()
    boundary_start = time.perf_counter()
    service = make_plan_copy_service(args, group=reshard_group, uncached=True)
    execute_reshard_plan_with_scheduler(
        plan,
        src_module,
        dst_module,
        service,
        args,
        label=f"{label} boundary TP weights",
        group=reshard_group,
    )
    boundary_transfer_s = reduce_max_seconds(time.perf_counter() - boundary_start)
    if isinstance(service, TracingCopyService):
        service.print_summary(f"{label} boundary TP weights")

    switch_start = time.perf_counter()
    after_diff, is_close, switch_metrics = simulate_atomic_switch(
        active_model=src_module,
        shadow_model=dst_module,
        active_ranks=src_ranks,
        shadow_ranks=dst_ranks,
        reference_logits=reference_logits,
        label=f"{label} shadow-prep boundary",
        args=args,
    )
    validation_switch_s = reduce_max_seconds(time.perf_counter() - switch_start)
    print_path_timing(
        f"C-shadow-prep {label}",
        shadow_prepare_s=shadow_prepare_s,
        plan_build_s=plan_build_s,
        shadow_warmup_s=shadow_warmup_s,
        boundary_transfer_s=boundary_transfer_s,
        validation_switch_s=validation_switch_s,
        **switch_metrics,
        exposed_pause_s=boundary_transfer_s + validation_switch_s,
    )

    if rank == 0:
        print(f"After shadow-prep boundary max diff: {after_diff:.6f}", flush=True)
        print(f"Shadow-prep boundary check passed: {is_close}", flush=True)

    assert is_close, (
        f"Shadow-prep boundary transfer failed for {label}: src_tp={args.src_tp}, "
        f"dst_tp={args.dst_tp}, backend={args.refit_backend}, max_diff={after_diff}"
    )
    dist.destroy_process_group(reshard_group)


def verify_fluidscale_scaleout(
    src_model: Optional[torch.nn.Module],
    dst_model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    src_active_ranks: list[int],
    dst_ranks: list[int],
) -> None:
    """Scale out from an old active TP world to a shadow TP world by view intersections."""
    rank = dist.get_rank()
    src_root_rank = src_active_ranks[0]
    src_module = src_model if rank in src_active_ranks else None
    dist.barrier()

    if rank == 0:
        print(
            "\nFluidScale-style live TP scale-out:",
            f"old_active_ranks={src_active_ranks}",
            f"new_shadow_ranks={dst_ranks}",
            f"src_tp={args.src_tp}",
            f"dst_tp={args.dst_tp}",
            f"backend={args.refit_backend}",
            flush=True,
        )
        print(
            "  Migration path: source/destination view intersections -> GPU copy service; "
            "no checkpoint save/load and no full tensor materialization.",
            flush=True,
        )

    with torch.no_grad():
        src_logits = _broadcast_active_source_logits(
            src_module,
            src_root_rank=src_root_rank,
            args=args,
        )
        dst_logits_before = run_forward(
            dst_model,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            batch_size=args.micro_batch_size,
        )

    before_diff = (src_logits - dst_logits_before).abs().max().item()
    if rank == 0:
        print(f"\nBefore scale-out reshard max diff: {before_diff:.6f}", flush=True)

    plan_start = time.perf_counter()
    plan = build_centralized_reshard_plan(
        src_module,
        dst_model,
        num_experts=dst_model.config.num_moe_experts,
        prefer_local_source=prefer_local_source(args),
    )
    plan_build_s = reduce_max_seconds(time.perf_counter() - plan_start)
    print_view_overlap_matrix(
        plan,
        dst_model,
        label="TP scale-out model weights",
        src_ranks=src_active_ranks,
        dst_ranks=dst_ranks,
    )
    if args.flexible_tp_plan:
        print_flexible_tp_plan_analysis(
            plan,
            src_module,
            dst_model,
            args,
            src_ranks=src_active_ranks,
            dst_ranks=dst_ranks,
            label="TP scale-out model weights",
        )
    if args.print_intersection_plan:
        print_intersection_plan(
            plan,
            dst_model,
            label="TP scale-out model weights",
            limit=args.intersection_plan_limit,
        )
    if args.reuse_overlap_weights:
        plan = reuse_overlapping_weight_storage(
            src_module,
            dst_model,
            plan,
            apply_aliases=True,
            allow_noncontiguous=args.allow_noncontiguous_storage_reuse,
        )

    service = make_plan_copy_service(args)
    transfer_start = time.perf_counter()
    execute_reshard_plan_with_scheduler(
        plan,
        src_module,
        dst_model,
        service,
        args,
        label="TP scale-out model weights",
    )
    dist.barrier()
    transfer_s = reduce_max_seconds(time.perf_counter() - transfer_start)
    if isinstance(service, TracingCopyService):
        service.print_summary("TP scale-out model weights")

    switch_start = time.perf_counter()
    after_diff, is_close, switch_metrics = simulate_atomic_switch(
        active_model=src_module,
        shadow_model=dst_model,
        active_ranks=src_active_ranks,
        shadow_ranks=dst_ranks,
        reference_logits=src_logits,
        label="TP scale-out",
        args=args,
    )
    validation_switch_s = reduce_max_seconds(time.perf_counter() - switch_start)
    print_path_timing(
        "A-blocking TP scale-out",
        plan_build_s=plan_build_s,
        boundary_transfer_s=transfer_s,
        validation_switch_s=validation_switch_s,
        **switch_metrics,
        exposed_pause_s=transfer_s + validation_switch_s,
    )

    if rank == 0:
        print(f"After scale-out reshard max diff:  {after_diff:.6f}", flush=True)
        print(f"Scale-out check passed:           {is_close}", flush=True)

    assert is_close, (
        f"FluidScale scale-out reshard failed: src_tp={args.src_tp}, "
        f"dst_tp={args.dst_tp}, backend={args.refit_backend}, max_diff={after_diff}"
    )


def verify_fluidscale_scalein(
    src_model: torch.nn.Module,
    dst_model: Optional[torch.nn.Module],
    args: argparse.Namespace,
    *,
    src_ranks: list[int],
    dst_surviving_ranks: list[int],
) -> None:
    """Scale in from an old TP world to a surviving-rank shadow TP world."""
    rank = dist.get_rank()
    dst_module = dst_model if rank in dst_surviving_ranks else None
    src_root_rank = src_ranks[0]
    dist.barrier()

    if rank == 0:
        removed_ranks = [r for r in src_ranks if r not in dst_surviving_ranks]
        print(
            "\nFluidScale-style live TP scale-in:",
            f"old_active_ranks={src_ranks}",
            f"new_surviving_ranks={dst_surviving_ranks}",
            f"removed_ranks={removed_ranks}",
            f"src_tp={args.src_tp}",
            f"dst_tp={args.dst_tp}",
            f"backend={args.refit_backend}",
            flush=True,
        )
        print(
            "  Migration path: source/destination view intersections -> GPU copy service; "
            "source-only ranks send their shards before leaving.",
            flush=True,
        )

    with torch.no_grad():
        src_logits = _broadcast_active_source_logits(
            src_model,
            src_root_rank=src_root_rank,
            args=args,
        )
        if dst_module is not None:
            dst_logits_before = run_forward(
                dst_module,
                vocab_size=args.vocab_size,
                seq_len=args.seq_len,
                batch_size=args.micro_batch_size,
            )
            before_diff = (src_logits - dst_logits_before).abs().max()
        else:
            before_diff = torch.zeros((), device=src_logits.device)
    dist.all_reduce(before_diff, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"\nBefore scale-in reshard max diff: {before_diff.item():.6f}", flush=True)

    plan_start = time.perf_counter()
    plan = build_centralized_reshard_plan(
        src_model,
        dst_module,
        num_experts=src_model.config.num_moe_experts,
        prefer_local_source=prefer_local_source(args),
    )
    plan_build_s = reduce_max_seconds(time.perf_counter() - plan_start)
    print_view_overlap_matrix(
        plan,
        dst_module,
        label="TP scale-in model weights",
        src_ranks=src_ranks,
        dst_ranks=dst_surviving_ranks,
    )
    if args.flexible_tp_plan:
        print_flexible_tp_plan_analysis(
            plan,
            src_model,
            dst_module,
            args,
            src_ranks=src_ranks,
            dst_ranks=dst_surviving_ranks,
            label="TP scale-in model weights",
        )
    if args.print_intersection_plan:
        print_intersection_plan(
            plan,
            dst_module,
            label="TP scale-in model weights",
            limit=args.intersection_plan_limit,
        )

    service = make_plan_copy_service(args)
    transfer_start = time.perf_counter()
    execute_reshard_plan_with_scheduler(
        plan,
        src_model,
        dst_module,
        service,
        args,
        label="TP scale-in model weights",
    )
    dist.barrier()
    transfer_s = reduce_max_seconds(time.perf_counter() - transfer_start)
    if isinstance(service, TracingCopyService):
        service.print_summary("TP scale-in model weights")

    switch_start = time.perf_counter()
    after_diff, is_close, switch_metrics = simulate_atomic_switch(
        active_model=src_model,
        shadow_model=dst_module,
        active_ranks=src_ranks,
        shadow_ranks=dst_surviving_ranks,
        reference_logits=src_logits,
        label="TP scale-in",
        args=args,
    )
    validation_switch_s = reduce_max_seconds(time.perf_counter() - switch_start)
    print_path_timing(
        "A-blocking TP scale-in",
        plan_build_s=plan_build_s,
        boundary_transfer_s=transfer_s,
        validation_switch_s=validation_switch_s,
        **switch_metrics,
        exposed_pause_s=transfer_s + validation_switch_s,
    )

    if rank == 0:
        print(f"After scale-in reshard max diff:  {after_diff:.6f}", flush=True)
        print(f"Scale-in check passed:           {is_close}", flush=True)

    assert is_close, (
        f"FluidScale scale-in reshard failed: src_tp={args.src_tp}, "
        f"dst_tp={args.dst_tp}, backend={args.refit_backend}, max_diff={after_diff}"
    )


def validate_model_args(args: argparse.Namespace) -> None:
    """Validate user-provided model dimensions before distributed model construction."""
    if args.ffn_hidden_size == 0:
        args.ffn_hidden_size = None

    if args.num_layers <= 0:
        raise ValueError(f"--num-layers must be positive, got {args.num_layers}")
    if args.hidden_size <= 0:
        raise ValueError(f"--hidden-size must be positive, got {args.hidden_size}")
    if args.num_attention_heads <= 0:
        raise ValueError(
            f"--num-attention-heads must be positive, got {args.num_attention_heads}"
        )
    if args.num_query_groups <= 0:
        raise ValueError(f"--num-query-groups must be positive, got {args.num_query_groups}")
    if args.vocab_size <= 0:
        raise ValueError(f"--vocab-size must be positive, got {args.vocab_size}")
    if args.seq_len <= 0:
        raise ValueError(f"--seq-len must be positive, got {args.seq_len}")
    if args.micro_batch_size <= 0:
        raise ValueError(
            f"--micro-batch-size must be positive, got {args.micro_batch_size}"
        )
    if args.live_train_steps <= 0:
        raise ValueError(f"--live-train-steps must be positive, got {args.live_train_steps}")
    if args.live_train_lr <= 0.0:
        raise ValueError(f"--live-train-lr must be positive, got {args.live_train_lr}")
    if args.scheduler_max_waves <= 0:
        raise ValueError(f"--scheduler-max-waves must be positive, got {args.scheduler_max_waves}")
    if args.scheduler_max_wave_bytes < 0:
        raise ValueError(
            f"--scheduler-max-wave-bytes must be non-negative, got {args.scheduler_max_wave_bytes}"
        )
    if args.scheduler_local_bandwidth_gbps <= 0.0:
        raise ValueError(
            "--scheduler-local-bandwidth-gbps must be positive, "
            f"got {args.scheduler_local_bandwidth_gbps}"
        )
    if args.scheduler_remote_bandwidth_gbps <= 0.0:
        raise ValueError(
            "--scheduler-remote-bandwidth-gbps must be positive, "
            f"got {args.scheduler_remote_bandwidth_gbps}"
        )
    if args.scheduler_latency_us < 0.0:
        raise ValueError(f"--scheduler-latency-us must be non-negative, got {args.scheduler_latency_us}")
    if args.scheduler_print_waves < 0:
        raise ValueError(f"--scheduler-print-waves must be non-negative, got {args.scheduler_print_waves}")
    if args.reshard_fusion != "none" and args.refit_backend != "nccl":
        raise ValueError("--reshard-fusion currently requires --refit-backend nccl")
    if args.reshard_fusion_chunk_bytes < 0:
        raise ValueError(
            f"--reshard-fusion-chunk-bytes must be non-negative, got {args.reshard_fusion_chunk_bytes}"
        )
    if args.reshard_fusion_min_chunk_bytes <= 0:
        raise ValueError(
            "--reshard-fusion-min-chunk-bytes must be positive, "
            f"got {args.reshard_fusion_min_chunk_bytes}"
        )
    if args.reshard_fusion_max_chunk_bytes <= 0:
        raise ValueError(
            "--reshard-fusion-max-chunk-bytes must be positive, "
            f"got {args.reshard_fusion_max_chunk_bytes}"
        )
    if args.reshard_fusion_max_chunk_bytes < args.reshard_fusion_min_chunk_bytes:
        raise ValueError(
            "--reshard-fusion-max-chunk-bytes must be >= "
            "--reshard-fusion-min-chunk-bytes"
        )
    if args.reshard_fusion_target_chunks_per_peer <= 0:
        raise ValueError(
            "--reshard-fusion-target-chunks-per-peer must be positive, "
            f"got {args.reshard_fusion_target_chunks_per_peer}"
        )
    if args.reshard_fusion_small_op_bytes < 0:
        raise ValueError(
            "--reshard-fusion-small-op-bytes must be non-negative, "
            f"got {args.reshard_fusion_small_op_bytes}"
        )
    if args.reshard_fusion_min_benefit_pct < 0.0:
        raise ValueError(
            "--reshard-fusion-min-benefit-pct must be non-negative, "
            f"got {args.reshard_fusion_min_benefit_pct}"
        )
    if args.reshard_fusion_pack_bandwidth_gbps <= 0.0:
        raise ValueError(
            "--reshard-fusion-pack-bandwidth-gbps must be positive, "
            f"got {args.reshard_fusion_pack_bandwidth_gbps}"
        )
    if args.reshard_fusion_parallelism_penalty < 1.0:
        raise ValueError(
            "--reshard-fusion-parallelism-penalty must be >= 1.0, "
            f"got {args.reshard_fusion_parallelism_penalty}"
        )
    if args.reshard_fusion_control_overhead_us < 0.0:
        raise ValueError(
            "--reshard-fusion-control-overhead-us must be non-negative, "
            f"got {args.reshard_fusion_control_overhead_us}"
        )
    if args.scheduler_min_benefit_pct < 0.0:
        raise ValueError(
            "--scheduler-min-benefit-pct must be non-negative, "
            f"got {args.scheduler_min_benefit_pct}"
        )
    if args.scheduler_switch_overhead_us < 0.0:
        raise ValueError(
            "--scheduler-switch-overhead-us must be non-negative, "
            f"got {args.scheduler_switch_overhead_us}"
        )
    if args.scheduler_wave_barrier_us < 0.0:
        raise ValueError(
            "--scheduler-wave-barrier-us must be non-negative, "
            f"got {args.scheduler_wave_barrier_us}"
        )
    if args.profile_p2p_bytes <= 0:
        raise ValueError(f"--profile-p2p-bytes must be positive, got {args.profile_p2p_bytes}")
    if args.profile_p2p_iters <= 0:
        raise ValueError(f"--profile-p2p-iters must be positive, got {args.profile_p2p_iters}")
    if args.profile_p2p_passes <= 0:
        raise ValueError(
            f"--profile-p2p-passes must be positive, got {args.profile_p2p_passes}"
        )
    if args.profile_p2p_warmup_iters < 0:
        raise ValueError(
            f"--profile-p2p-warmup-iters must be non-negative, got {args.profile_p2p_warmup_iters}"
        )
    if args.rank_placement_max_exhaustive_ranks <= 0:
        raise ValueError(
            "--rank-placement-max-exhaustive-ranks must be positive, got "
            f"{args.rank_placement_max_exhaustive_ranks}"
        )
    if args.rank_placement_min_benefit_pct < 0.0:
        raise ValueError(
            "--rank-placement-min-benefit-pct must be non-negative, got "
            f"{args.rank_placement_min_benefit_pct}"
        )
    _parse_slow_link_spec(args.scheduler_slow_links)
    if args.hidden_size % args.num_attention_heads != 0:
        raise ValueError(
            f"--hidden-size ({args.hidden_size}) must be divisible by "
            f"--num-attention-heads ({args.num_attention_heads})"
        )
    if args.num_attention_heads % args.num_query_groups != 0:
        raise ValueError(
            f"--num-attention-heads ({args.num_attention_heads}) must be divisible by "
            f"--num-query-groups ({args.num_query_groups})"
        )
    if args.ffn_hidden_size is not None and args.ffn_hidden_size <= 0:
        raise ValueError(f"--ffn-hidden-size must be positive or 0, got {args.ffn_hidden_size}")
    for label, tp_size in (("src", args.src_tp), ("dst", args.dst_tp)):
        if args.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"--num-attention-heads ({args.num_attention_heads}) must be divisible by "
                f"{label}_tp ({tp_size})"
            )
        if args.num_query_groups % tp_size != 0 and tp_size % args.num_query_groups != 0:
            raise ValueError(
                f"--num-query-groups ({args.num_query_groups}) must be a multiple or divisor "
                f"of {label}_tp ({tp_size})"
            )


def print_model_config_summary(args: argparse.Namespace) -> None:
    if dist.get_rank() != 0:
        return
    ffn_hidden_size = args.ffn_hidden_size or 4 * args.hidden_size
    print(
        "\nDemo model config:",
        f"layers={args.num_layers}",
        f"hidden_size={args.hidden_size}",
        f"attention_heads={args.num_attention_heads}",
        f"query_groups={args.num_query_groups}",
        f"ffn_hidden_size={ffn_hidden_size}",
        f"vocab_size={args.vocab_size}",
        f"seq_len={args.seq_len}",
        f"micro_batch_size={args.micro_batch_size}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-tp", type=int, required=True)
    parser.add_argument("--dst-tp", type=int, required=True)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of GPT transformer layers. Defaults to the original tiny demo model.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="GPT hidden size. Increase this with --num-layers for larger migration tests.",
    )
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=8,
        help="Number of attention heads.",
    )
    parser.add_argument(
        "--num-query-groups",
        type=int,
        default=4,
        help="Number of query groups for GQA. Must be compatible with src/dst TP sizes.",
    )
    parser.add_argument(
        "--ffn-hidden-size",
        type=int,
        default=0,
        help="FFN hidden size. Use 0 to default to 4 * hidden_size.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128,
        help="Vocabulary size used to build embeddings and output logits.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=8,
        help="Sequence length used for model construction and deterministic forward checks.",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Batch size used by deterministic forward checks.",
    )
    parser.add_argument(
        "--live-training-migration",
        action="store_true",
        help=(
            "Run a TP-parameter-only live migration test: keep the active source model "
            "training while a background thread migrates base weights into the shadow "
            "TP model, then refresh dirty parameters and atomically switch."
        ),
    )
    parser.add_argument(
        "--shadow-prep-boundary-transfer",
        action="store_true",
        help=(
            "Run the C path: prepare the shadow TP world and transfer plan before the "
            "cutover, then pause active training at the boundary for one TP parameter "
            "transfer and pointer swap. This avoids dirty refresh."
        ),
    )
    parser.add_argument(
        "--skip-shadow-forward-warmup",
        action="store_true",
        help=(
            "Disable the C-path pre-cutover shadow forward warmup. By default the demo "
            "runs one shadow forward before boundary transfer to move first-use TP/NCCL/"
            "CUDA initialization cost out of the exposed pause."
        ),
    )
    parser.add_argument(
        "--live-train-steps",
        type=int,
        default=4,
        help=(
            "Number of real forward/backward/SGD steps to run on active ranks while "
            "--live-training-migration is copying the base TP weights."
        ),
    )
    parser.add_argument(
        "--live-train-lr",
        type=float,
        default=1e-4,
        help="SGD learning rate used by the live-training migration test.",
    )
    parser.add_argument(
        "--flexible-tp-plan",
        action="store_true",
        help=(
            "Print an analysis-only Flexible TP plan that compares current uniform TP "
            "reshard bytes with a variable-width interval layout. This does not execute "
            "uneven TP kernels."
        ),
    )
    parser.add_argument(
        "--flexible-src-layout",
        type=str,
        default=None,
        help=(
            "Comma-separated source TP interval weights for --flexible-tp-plan. "
            "Defaults to uniform weights over source ranks."
        ),
    )
    parser.add_argument(
        "--flexible-dst-layout",
        type=str,
        default=None,
        help=(
            "Comma-separated destination TP interval weights for --flexible-tp-plan. "
            "Defaults to uniform weights over destination ranks."
        ),
    )
    parser.add_argument(
        "--bandwidth-aware-rank-placement",
        action="store_true",
        help=(
            "Before TP scale-out, map destination logical TP ranks to physical GPUs "
            "using measured B[src][dst], estimated TP intersection bytes, and local "
            "reuse. The view intersection still defines tensor correctness."
        ),
    )
    parser.add_argument(
        "--rank-placement-search",
        choices=("auto", "exhaustive", "greedy"),
        default="auto",
        help=(
            "Search method for --bandwidth-aware-rank-placement. 'auto' exhaustively "
            "searches small worlds and uses greedy placement plus pair swaps otherwise."
        ),
    )
    parser.add_argument(
        "--rank-placement-max-exhaustive-ranks",
        type=int,
        default=8,
        help="Largest destination TP size for exhaustive physical-rank permutation search.",
    )
    parser.add_argument(
        "--rank-placement-min-benefit-pct",
        type=float,
        default=0.0,
        help=(
            "Minimum predicted migration critical-path improvement required before "
            "using a non-identity destination rank placement."
        ),
    )
    parser.add_argument(
        "--reshard-scheduler",
        choices=("none", "bytes-desc", "bandwidth-aware", "bandwidth-greedy", "nccl-round"),
        default="none",
        help=(
            "Schedule TP parameter transfer tasks before execution. 'none' preserves "
            "the original single-batch behavior; 'bandwidth-aware' packs tasks into "
            "rank/link-balanced waves using estimated link bandwidth; "
            "'bandwidth-greedy' uses the explicit bytes / B[src][dst] + latency "
            "cost model to minimize predicted rank/link critical path; "
            "'nccl-round' keeps per-peer FIFO order and interleaves peer queues "
            "using an NCCL-like P2P round order."
        ),
    )
    parser.add_argument(
        "--reshard-fusion",
        choices=("none", "peer", "adaptive-peer"),
        default="none",
        help=(
            "Fuse TP reshard TransferOps before execution. 'peer' packs all same "
            "src/dst/dtype slices into one contiguous buffer; 'adaptive-peer' keeps "
            "same-peer/dtype FIFO but splits each peer stream into dynamic chunks to "
            "balance lower op count with NCCL parallelism."
        ),
    )
    parser.add_argument(
        "--reshard-fusion-chunk-bytes",
        type=int,
        default=0,
        help=(
            "Fixed chunk size for --reshard-fusion adaptive-peer. 0 chooses a dynamic "
            "size from world size, active peer count, and remote bytes."
        ),
    )
    parser.add_argument(
        "--reshard-fusion-min-chunk-bytes",
        type=int,
        default=1 << 20,
        help="Minimum dynamic chunk size for --reshard-fusion adaptive-peer.",
    )
    parser.add_argument(
        "--reshard-fusion-max-chunk-bytes",
        type=int,
        default=16 << 20,
        help="Maximum dynamic chunk size for --reshard-fusion adaptive-peer.",
    )
    parser.add_argument(
        "--reshard-fusion-target-chunks-per-peer",
        type=int,
        default=4,
        help=(
            "Dynamic fusion target chunks per active peer/dtype group for "
            "--reshard-fusion adaptive-peer."
        ),
    )
    parser.add_argument(
        "--reshard-fusion-small-op-bytes",
        type=int,
        default=256 << 10,
        help=(
            "For --reshard-fusion adaptive-peer, only remote ops at or below this "
            "size are packed into fused buffers. Larger ops are submitted directly "
            "to preserve NCCL's fine-grained parallelism. Use 0 to fuse all ops."
        ),
    )
    parser.add_argument(
        "--reshard-fusion-auto-gate",
        action="store_true",
        help=(
            "Use a PAT-style cost gate before fused peer transfer. Fusion is used "
            "only when predicted communication savings exceed pack/unpack and "
            "control overhead; otherwise the plan falls back to direct NCCL P2P."
        ),
    )
    parser.add_argument(
        "--reshard-fusion-min-benefit-pct",
        type=float,
        default=5.0,
        help="Minimum predicted fusion speedup percentage required by auto-gate.",
    )
    parser.add_argument(
        "--reshard-fusion-pack-bandwidth-gbps",
        type=float,
        default=600.0,
        help="Estimated GPU pack/unpack bandwidth used by --reshard-fusion-auto-gate.",
    )
    parser.add_argument(
        "--reshard-fusion-parallelism-penalty",
        type=float,
        default=1.25,
        help=(
            "Penalty multiplier for fused bytes in --reshard-fusion-auto-gate. "
            "Values above 1 model lost NCCL P2P parallelism from larger chunks."
        ),
    )
    parser.add_argument(
        "--reshard-fusion-control-overhead-us",
        type=float,
        default=50.0,
        help="Fixed control/metadata overhead used by --reshard-fusion-auto-gate.",
    )
    parser.add_argument(
        "--scheduler-max-waves",
        type=int,
        default=8,
        help="Maximum number of waves used by --reshard-scheduler.",
    )
    parser.add_argument(
        "--scheduler-execution-mode",
        choices=("single-batch", "waves"),
        default="single-batch",
        help=(
            "How to execute scheduled tasks. 'single-batch' preserves NCCL's one-shot "
            "batch execution and only reorders submit order; 'waves' executes each "
            "scheduled wave as a separate reshard plan for debugging/experiments."
        ),
    )
    parser.add_argument(
        "--scheduler-max-wave-bytes",
        type=int,
        default=0,
        help=(
            "Maximum bytes per scheduled wave. Use 0 to derive it from total bytes "
            "and --scheduler-max-waves."
        ),
    )
    parser.add_argument(
        "--scheduler-local-bandwidth-gbps",
        type=float,
        default=1000.0,
        help="Estimated same-rank GPU copy bandwidth used by bandwidth-aware scheduling.",
    )
    parser.add_argument(
        "--scheduler-remote-bandwidth-gbps",
        type=float,
        default=200.0,
        help="Estimated cross-rank P2P bandwidth used by bandwidth-aware scheduling.",
    )
    parser.add_argument(
        "--scheduler-latency-us",
        type=float,
        default=5.0,
        help="Per-transfer latency term used by bandwidth-aware scheduling.",
    )
    parser.add_argument(
        "--scheduler-auto-gate",
        action="store_true",
        help=(
            "Use a PAT-style cost gate before applying task reordering/waves. "
            "The scheduler is enabled only when the predicted critical-path "
            "reduction exceeds the configured overhead and benefit threshold."
        ),
    )
    parser.add_argument(
        "--scheduler-min-benefit-pct",
        type=float,
        default=5.0,
        help="Minimum predicted scheduler speedup percentage required by auto-gate.",
    )
    parser.add_argument(
        "--scheduler-switch-overhead-us",
        type=float,
        default=0.0,
        help="Fixed scheduler transition overhead used by --scheduler-auto-gate.",
    )
    parser.add_argument(
        "--scheduler-wave-barrier-us",
        type=float,
        default=100.0,
        help="Estimated inter-wave barrier cost used when gating wave execution.",
    )
    parser.add_argument(
        "--scheduler-release-cache",
        action="store_true",
        help=(
            "Release CUDA allocator cache after the final scheduled wave. The default "
            "keeps cached buffers so repeated inference refits can reuse allocations."
        ),
    )
    parser.add_argument(
        "--profile-p2p-bandwidth",
        action="store_true",
        help=(
            "Measure pairwise GPU P2P bandwidth after distributed initialization and "
            "use the resulting bandwidth[src][dst] matrix in the scheduler cost model."
        ),
    )
    parser.add_argument(
        "--profile-p2p-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help="Payload size in bytes for each pairwise P2P bandwidth profiling transfer.",
    )
    parser.add_argument(
        "--profile-p2p-iters",
        type=int,
        default=3,
        help="Timed iterations per src/dst pair for --profile-p2p-bandwidth.",
    )
    parser.add_argument(
        "--profile-p2p-passes",
        type=int,
        default=3,
        help="Independent timed passes per directed pair; the median is used.",
    )
    parser.add_argument(
        "--profile-p2p-warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations per src/dst pair for --profile-p2p-bandwidth.",
    )
    parser.add_argument(
        "--profile-p2p-output",
        type=str,
        default=None,
        help="Optional JSON path used to save the measured P2P bandwidth matrix.",
    )
    parser.add_argument(
        "--scheduler-bandwidth-profile",
        type=str,
        default=None,
        help=(
            "Load a previously saved P2P bandwidth JSON profile and use it as "
            "B[src][dst] in bandwidth-aware scheduler cost models."
        ),
    )
    parser.add_argument(
        "--scheduler-slow-links",
        type=str,
        default="",
        help=(
            "Inject artificial heterogeneous links into the scheduler bandwidth "
            "matrix. CSV format: 'src:dst=gbps' sets an absolute bandwidth, "
            "and 'src:dst*factor' multiplies the existing B[src][dst]. Example: "
            "'1:2=8,1:3*0.2'."
        ),
    )
    parser.add_argument(
        "--scheduler-slow-link-simulate-delay",
        action="store_true",
        help=(
            "Also inject runtime sleep delays for tasks crossing --scheduler-slow-links. "
            "Use this only for controlled heterogeneity experiments; it is not a "
            "real transport throttle."
        ),
    )
    parser.add_argument(
        "--scheduler-print-waves",
        type=int,
        default=8,
        help="Number of scheduled waves to print for inspection.",
    )
    parser.add_argument(
        "--fluidscale-elastic",
        action="store_true",
        help=(
            "Infer the live elastic direction from --src-tp and --dst-tp. "
            "If dst_tp > src_tp, run scale-out; if dst_tp < src_tp, run scale-in."
        ),
    )
    parser.add_argument(
        "--fluidscale-scaleout",
        action="store_true",
        help=(
            "Emulate live elastic TP scale-out: only the old active ranks own the source "
            "TP model, all launched ranks build the destination shadow TP model, and "
            "weights migrate by source/destination view intersections."
        ),
    )
    parser.add_argument(
        "--scaleout-src-rank-count",
        type=int,
        default=None,
        help=(
            "Number of old active ranks in --fluidscale-scaleout. Defaults to --src-tp, "
            "matching the 2-GPU TP=2 -> 4-GPU TP=4 example."
        ),
    )
    parser.add_argument(
        "--scaleout-src-rank-offset",
        type=int,
        default=0,
        help="First global rank of the old active world in --fluidscale-scaleout.",
    )
    parser.add_argument(
        "--fluidscale-scalein",
        action="store_true",
        help=(
            "Emulate live elastic TP scale-in: all launched ranks own the old source TP "
            "model, only surviving ranks build the destination TP model, and removed "
            "ranks only send their source shards."
        ),
    )
    parser.add_argument(
        "--scalein-dst-rank-count",
        type=int,
        default=None,
        help=(
            "Number of surviving destination ranks in --fluidscale-scalein. Defaults to "
            "--dst-tp, matching the 4-GPU TP=4 -> 2-GPU TP=2 example."
        ),
    )
    parser.add_argument(
        "--scalein-dst-rank-offset",
        type=int,
        default=0,
        help="First global rank of the surviving destination world in --fluidscale-scalein.",
    )
    parser.add_argument("--swap", action="store_true", help="Run swap_model_weights verification")
    parser.add_argument(
        "--optimizer-state",
        action="store_true",
        help="Run AdamW exp_avg/exp_avg_sq optimizer-state reshard verification",
    )
    parser.add_argument(
        "--distributed-optimizer-state",
        action="store_true",
        help=(
            "Run Megatron DistributedOptimizer param/exp_avg/exp_avg_sq reshard verification. "
            "This first demo requires destination DP size 1."
        ),
    )
    parser.add_argument(
        "--optimizer-step",
        type=int,
        default=7,
        help="Synthetic AdamW step value copied as replicated scalar state",
    )
    parser.add_argument(
        "--refit-backend",
        choices=("nccl", "gloo", "nvshmem"),
        default="nccl",
        help="Backend used by weight and optimizer-state reshard verification",
    )
    parser.add_argument(
        "--trace-transfer-devices",
        action="store_true",
        help="Print per-reshard transfer counts and bytes, grouped by local vs remote peers",
    )
    parser.add_argument(
        "--require-gpu-direct",
        action="store_true",
        help=(
            "Fail if the selected backend is CPU-staged or if reshard execution submits any "
            "non-CUDA tensor to the copy service"
        ),
    )
    parser.add_argument(
        "--zero-copy-gpu",
        action="store_true",
        help=(
            "Convenience mode for paper-style storage-free, CPU-free payload migration. "
            "This enables transfer tracing and requires a GPU-direct backend such as nccl."
        ),
    )
    parser.add_argument(
        "--force-remote-transfer",
        action="store_true",
        help=(
            "Require the demo to produce cross-rank GPU-GPU traffic. This disables the "
            "planner's same-rank source preference when another source replica is available "
            "and fails if remote_bytes stays zero."
        ),
    )
    parser.add_argument(
        "--analyze-storage-reuse",
        action="store_true",
        help=(
            "Print how much destination model weight storage can be reused from same-rank "
            "source parameter views. This is analysis only and does not mutate parameters."
        ),
    )
    parser.add_argument(
        "--reuse-overlap-weights",
        action="store_true",
        help=(
            "For model weights only, alias destination parameters to same-rank source "
            "storage views when the reshard plan maps the full destination shard to one "
            "local source slice. Non-aliasable weights still use normal reshard copy."
        ),
    )
    parser.add_argument(
        "--allow-noncontiguous-storage-reuse",
        action="store_true",
        help=(
            "Allow destination parameters to alias non-contiguous source views. This can "
            "increase reuse for RowParallelLinear-style dim=1 shards but may not be valid "
            "for all kernels, so the default only aliases contiguous views."
        ),
    )
    parser.add_argument(
        "--print-intersection-plan",
        action="store_true",
        help=(
            "Print the actual source/destination view-intersection transfer tasks for "
            "model weights before executing the reshard."
        ),
    )
    parser.add_argument(
        "--intersection-plan-limit",
        type=int,
        default=80,
        help="Maximum number of intersection transfer tasks to print; use 0 for all tasks.",
    )
    args = parser.parse_args()
    validate_model_args(args)
    if args.fluidscale_elastic:
        if args.fluidscale_scaleout or args.fluidscale_scalein:
            raise ValueError(
                "--fluidscale-elastic chooses the direction automatically; do not combine it "
                "with --fluidscale-scaleout or --fluidscale-scalein."
            )
        if args.dst_tp > args.src_tp:
            args.fluidscale_scaleout = True
        elif args.dst_tp < args.src_tp:
            args.fluidscale_scalein = True
        else:
            raise ValueError(
                "--fluidscale-elastic needs a TP degree change. "
                f"Got src_tp=dst_tp={args.src_tp}."
            )
    if args.fluidscale_scaleout and args.fluidscale_scalein:
        raise ValueError("--fluidscale-scaleout and --fluidscale-scalein are mutually exclusive.")
    if args.bandwidth_aware_rank_placement and not args.fluidscale_scaleout:
        raise ValueError(
            "--bandwidth-aware-rank-placement currently supports FluidScale TP scale-out only."
        )
    if args.live_training_migration and not (
        args.fluidscale_scaleout or args.fluidscale_scalein
    ):
        raise ValueError(
            "--live-training-migration currently runs on the FluidScale TP scale-out/"
            "scale-in paths. Combine it with --fluidscale-scaleout, --fluidscale-scalein, "
            "or --fluidscale-elastic."
        )
    if args.shadow_prep_boundary_transfer and not (
        args.fluidscale_scaleout or args.fluidscale_scalein
    ):
        raise ValueError(
            "--shadow-prep-boundary-transfer currently runs on the FluidScale TP "
            "scale-out/scale-in paths. Combine it with --fluidscale-scaleout, "
            "--fluidscale-scalein, or --fluidscale-elastic."
        )
    if args.live_training_migration and args.shadow_prep_boundary_transfer:
        raise ValueError(
            "--live-training-migration and --shadow-prep-boundary-transfer are separate "
            "B and C path tests; run them in separate invocations."
        )
    if args.fluidscale_scaleout or args.fluidscale_scalein:
        args.zero_copy_gpu = True

    src_pgs = None
    dst_pgs = None
    try:
        init_distributed_and_mpu(args.src_tp)
        enable_zero_copy_gpu_mode(args)
        print_model_config_summary(args)
        if args.force_remote_transfer and args.reuse_overlap_weights:
            raise ValueError(
                "--force-remote-transfer conflicts with --reuse-overlap-weights: remote "
                "verification intentionally avoids local aliases, while storage reuse needs "
                "same-rank source views."
            )

        world_size = dist.get_world_size()
        assert world_size % args.dst_tp == 0, (
            f"WORLD_SIZE={world_size} not divisible by dst_tp={args.dst_tp}"
        )
        scaleout_src_rank_count = (
            args.src_tp if args.scaleout_src_rank_count is None else args.scaleout_src_rank_count
        )
        scalein_dst_rank_count = (
            args.dst_tp if args.scalein_dst_rank_count is None else args.scalein_dst_rank_count
        )
        if args.fluidscale_scaleout:
            if args.swap or args.optimizer_state or args.distributed_optimizer_state:
                raise ValueError(
                    "--fluidscale-scaleout is a model-weight migration demo; do not combine it "
                    "with --swap, --optimizer-state, or --distributed-optimizer-state."
                )
            if args.dst_tp != world_size:
                raise ValueError(
                    "--fluidscale-scaleout expects the destination TP group to span all launched "
                    f"ranks; got dst_tp={args.dst_tp}, WORLD_SIZE={world_size}."
                )
            if scaleout_src_rank_count % args.src_tp != 0:
                raise ValueError(
                    f"scaleout source rank count {scaleout_src_rank_count} must be divisible "
                    f"by src_tp={args.src_tp}."
                )
            if args.scaleout_src_rank_offset < 0:
                raise ValueError("--scaleout-src-rank-offset must be non-negative.")
            if args.scaleout_src_rank_offset + scaleout_src_rank_count > world_size:
                raise ValueError(
                    "Old active rank range exceeds WORLD_SIZE: "
                    f"offset={args.scaleout_src_rank_offset}, "
                    f"count={scaleout_src_rank_count}, WORLD_SIZE={world_size}."
                )
        elif args.fluidscale_scalein:
            if args.swap or args.optimizer_state or args.distributed_optimizer_state:
                raise ValueError(
                    "--fluidscale-scalein is a model-weight migration demo; do not combine it "
                    "with --swap, --optimizer-state, or --distributed-optimizer-state."
                )
            if args.src_tp != world_size:
                raise ValueError(
                    "--fluidscale-scalein expects the source TP group to span all launched "
                    f"ranks; got src_tp={args.src_tp}, WORLD_SIZE={world_size}."
                )
            if scalein_dst_rank_count % args.dst_tp != 0:
                raise ValueError(
                    f"scale-in destination rank count {scalein_dst_rank_count} must be "
                    f"divisible by dst_tp={args.dst_tp}."
                )
            if args.scalein_dst_rank_offset < 0:
                raise ValueError("--scalein-dst-rank-offset must be non-negative.")
            if args.scalein_dst_rank_offset + scalein_dst_rank_count > world_size:
                raise ValueError(
                    "Surviving destination rank range exceeds WORLD_SIZE: "
                    f"offset={args.scalein_dst_rank_offset}, "
                    f"count={scalein_dst_rank_count}, WORLD_SIZE={world_size}."
                )
        else:
            assert world_size % args.src_tp == 0, (
                f"WORLD_SIZE={world_size} not divisible by src_tp={args.src_tp}"
            )

        if args.profile_p2p_bandwidth:
            args.scheduler_bandwidth_matrix = profile_p2p_bandwidth_matrix(args)
        elif args.scheduler_bandwidth_profile:
            args.scheduler_bandwidth_matrix = load_bandwidth_profile(
                args.scheduler_bandwidth_profile,
                expected_world_size=world_size,
            )
            _print_bandwidth_matrix_summary(
                args.scheduler_bandwidth_matrix,
                world_size,
                prefix="Loaded P2P",
            )
        else:
            args.scheduler_bandwidth_matrix = None
        args.scheduler_bandwidth_matrix = apply_scheduler_slow_links(
            args.scheduler_bandwidth_matrix,
            args,
            world_size,
        )

        model_parallel_cuda_manual_seed(1234)
        torch.manual_seed(1234)

        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        base_cfg = make_config(args.src_tp, args)
        src_cfg = copy.deepcopy(base_cfg)
        dst_cfg = copy.deepcopy(base_cfg)
        src_cfg.tensor_model_parallel_size = args.src_tp
        dst_cfg.tensor_model_parallel_size = args.dst_tp

        if args.fluidscale_scaleout:
            src_pgs = build_pg_collection(
                tp_size=args.src_tp,
                rank_count=scaleout_src_rank_count,
                rank_offset=args.scaleout_src_rank_offset,
            )
        else:
            src_pgs = build_pg_collection(tp_size=args.src_tp)

        src_active_ranks = list(
            range(
                args.scaleout_src_rank_offset,
                args.scaleout_src_rank_offset + scaleout_src_rank_count,
            )
        )
        is_src_active_rank = dist.get_rank() in src_active_ranks
        src_model = (
            build_gpt_model(src_cfg, src_pgs, args).to(device).eval()
            if (not args.fluidscale_scaleout or is_src_active_rank)
            else None
        )
        dst_rank_order = list(range(world_size))
        if args.bandwidth_aware_rank_placement:
            dst_rank_order = choose_bandwidth_aware_dst_rank_order(
                src_model,
                args,
                src_ranks=src_active_ranks,
                dst_ranks=list(range(world_size)),
            )

        if args.fluidscale_scalein:
            dst_pgs = build_pg_collection(
                tp_size=args.dst_tp,
                rank_count=scalein_dst_rank_count,
                rank_offset=args.scalein_dst_rank_offset,
            )
        elif args.bandwidth_aware_rank_placement:
            dst_pgs = build_pg_collection(tp_size=args.dst_tp, rank_order=dst_rank_order)
        else:
            dst_pgs = build_pg_collection(tp_size=args.dst_tp)

        dst_surviving_ranks = list(
            range(
                args.scalein_dst_rank_offset,
                args.scalein_dst_rank_offset + scalein_dst_rank_count,
            )
        )
        is_dst_surviving_rank = dist.get_rank() in dst_surviving_ranks
        dst_model = (
            build_gpt_model(dst_cfg, dst_pgs, args).to(device).eval()
            if (not args.fluidscale_scalein or is_dst_surviving_rank)
            else None
        )

        print_param_summary(
            "SRC",
            src_model,
            src_pgs,
            participating_ranks=set(src_active_ranks) if args.fluidscale_scaleout else None,
        )
        print_param_summary(
            "DST",
            dst_model,
            dst_pgs,
            participating_ranks=set(dst_surviving_ranks) if args.fluidscale_scalein else None,
        )

        if args.fluidscale_scaleout:
            if args.live_training_migration:
                verify_live_training_tp_migration(
                    src_model,
                    dst_model,
                    args,
                    src_ranks=src_active_ranks,
                    dst_ranks=list(range(world_size)),
                    label="TP scale-out",
                )
            elif args.shadow_prep_boundary_transfer:
                verify_shadow_prep_boundary_transfer(
                    src_model,
                    dst_model,
                    args,
                    src_ranks=src_active_ranks,
                    dst_ranks=list(range(world_size)),
                    label="TP scale-out",
                )
            else:
                verify_fluidscale_scaleout(
                    src_model,
                    dst_model,
                    args,
                    src_active_ranks=src_active_ranks,
                    dst_ranks=list(range(world_size)),
                )
            dist.barrier()
            del src_model, dst_model
            gc.collect()
            torch.cuda.empty_cache()
            return
        if args.fluidscale_scalein:
            if args.live_training_migration:
                verify_live_training_tp_migration(
                    src_model,
                    dst_model,
                    args,
                    src_ranks=list(range(world_size)),
                    dst_ranks=dst_surviving_ranks,
                    label="TP scale-in",
                )
            elif args.shadow_prep_boundary_transfer:
                verify_shadow_prep_boundary_transfer(
                    src_model,
                    dst_model,
                    args,
                    src_ranks=list(range(world_size)),
                    dst_ranks=dst_surviving_ranks,
                    label="TP scale-in",
                )
            else:
                verify_fluidscale_scalein(
                    src_model,
                    dst_model,
                    args,
                    src_ranks=list(range(world_size)),
                    dst_surviving_ranks=dst_surviving_ranks,
                )
            dist.barrier()
            del src_model, dst_model
            gc.collect()
            torch.cuda.empty_cache()
            return

        if args.analyze_storage_reuse:
            reuse_plan = build_centralized_reshard_plan(
                src_model,
                dst_model,
                num_experts=src_model.config.num_moe_experts,
                prefer_local_source=True,
            )
            reuse_overlapping_weight_storage(
                src_model,
                dst_model,
                reuse_plan,
                apply_aliases=False,
                allow_noncontiguous=args.allow_noncontiguous_storage_reuse,
            )

        if args.flexible_tp_plan and not (
            args.swap or args.reuse_overlap_weights or args.print_intersection_plan
        ):
            flexible_base_plan = build_centralized_reshard_plan(
                src_model,
                dst_model,
                num_experts=src_model.config.num_moe_experts,
                prefer_local_source=prefer_local_source(args),
            )
            print_flexible_tp_plan_analysis(
                flexible_base_plan,
                src_model,
                dst_model,
                args,
                src_ranks=list(range(world_size)),
                dst_ranks=list(range(world_size)),
                label="model weights",
            )

        if args.print_intersection_plan and not (args.swap or args.reuse_overlap_weights):
            intersection_plan = build_centralized_reshard_plan(
                src_model,
                dst_model,
                num_experts=src_model.config.num_moe_experts,
                prefer_local_source=prefer_local_source(args),
            )
            print_intersection_plan(
                intersection_plan,
                dst_model,
                label="model weights",
                limit=args.intersection_plan_limit,
            )
            if args.flexible_tp_plan:
                print_flexible_tp_plan_analysis(
                    intersection_plan,
                    src_model,
                    dst_model,
                    args,
                    src_ranks=list(range(world_size)),
                    dst_ranks=list(range(world_size)),
                    label="model weights",
                )

        if args.swap or args.reuse_overlap_weights:
            verify_swap(src_model, dst_model, args)
        if args.optimizer_state:
            verify_optimizer_state_reshard(src_model, dst_model, src_pgs, dst_pgs, args)
        if args.distributed_optimizer_state:
            verify_distributed_optimizer_state_reshard(
                src_model,
                dst_model,
                src_cfg,
                dst_cfg,
                src_pgs,
                dst_pgs,
                args,
            )

        dist.barrier()

        del src_model, dst_model
        gc.collect()
        torch.cuda.empty_cache()
    finally:
        clear_all_caches()
        if src_pgs is not None:
            destroy_pg_collection(src_pgs)
        if dst_pgs is not None:
            destroy_pg_collection(dst_pgs)
        mpu.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
