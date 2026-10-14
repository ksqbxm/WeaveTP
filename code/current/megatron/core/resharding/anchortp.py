"""AnchorTP fixed-source, stop-the-world migration execution.

The intersection planner owns data placement. This module only orders its GPU
tasks by static link bandwidth and submits optional, explicit CPU reloads.
"""

import hashlib
import json
import math
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from .async_execution import launch_reshard_plan, prepare_reshard_receives
from .live import collect_transfer_tasks, filter_plan_by_task_ids

IMPLEMENTATION = "anchortp-fixed-paths-v1"
PATHS = ("local_copy", "gpu_pcie", "host_to_gpu", "gpu_nvlink", "gpu_xgmi", "network")


def configure(args):
    """Bind every AnchorTP entrypoint independently of inherited shell options."""
    from .planner import _source_search_config

    if _source_search_config() is not None:
        raise ValueError("AnchorTP fixed sources conflict with explicit source-search configuration")
    args.live_scheduler_mode = "anchortp"
    args.live_disable_source_reroute = True
    for name in (
        "allow_aware_shrink", "online_replan", "adaptive_hybrid", "hybrid_fast_path",
        "online_migration_first_guard", "persistent_pack_buffers", "pack_rerouted_only",
        "force_bandwidth_routing", "allow_nonlocal_reroute", "emulate_noncollocated_sources",
    ):
        setattr(args, "live_" + name, False)
    args.live_pack_target_bytes = args.live_pack_max_item_bytes = 0
    args.live_unpack_chunk_bytes = 0
    args.live_p2p_order = "task-round"
    args.live_max_waves = 0
    args.live_min_overlap_steps = args.live_max_overlap_steps = 0


def topology(control_group, profile=None):
    hosts = [None] * dist.get_world_size(group=control_group)
    dist.all_gather_object(hosts, socket.gethostname(), group=control_group)
    # Optional hardware topology, never inferred from measured bandwidth.
    payload = json.loads(Path(profile).read_text(encoding="utf-8")) if profile else {}
    links = payload.get("link_types")
    if links is not None:
        if len(links) != len(hosts) or any(len(row) != len(hosts) for row in links):
            raise ValueError("AnchorTP link_types must match WORLD_SIZE")
        for src, row in enumerate(links):
            for dst, kind in enumerate(row):
                if kind not in {"local", "pcie", "nvlink", "xgmi", "network"}:
                    raise ValueError(f"unknown hardware link type: {kind}")
                if kind in {"nvlink", "xgmi"}:
                    if src == dst or hosts[src] != hosts[dst]:
                        raise ValueError("high-speed GPU links must join different GPUs on one host")
                    if (kind == "xgmi") != bool(torch.version.hip):
                        raise ValueError("hardware link type conflicts with CUDA/ROCm backend")
    return hosts, links


def path_kind(task, hosts, links=None):
    src, dst = task.src_rank, task.dst_rank
    if src == dst:
        return "local_copy"
    if hosts[src] != hosts[dst]:
        return "network"
    kind = links[src][dst] if links else "pcie"
    if kind not in {"pcie", "nvlink", "xgmi"}:
        raise ValueError("same-host GPU tasks require a PCIe, NVLink or XGMI link")
    return "gpu_" + kind


def schedule(tasks, bandwidth, max_tasks=0, max_bytes=0):
    """Stable descending static bandwidth; caps never discard the final tasks.

    A single indivisible slice larger than max_bytes occupies its own batch.
    """
    if max_tasks < 0 or max_bytes < 0:
        raise ValueError("AnchorTP batch limits must be non-negative")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("AnchorTP requires globally unique task IDs")
    for task in tasks:
        if task.task_id < 0:
            raise ValueError("AnchorTP task IDs must be non-negative")
        value = bandwidth[task.src_rank, task.dst_rank]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("AnchorTP static bandwidth must be finite and positive")
    ordered = sorted(tasks, key=lambda task: (-bandwidth[task.src_rank, task.dst_rank], task.task_id))
    batches, batch, size = [], [], 0
    for task in ordered:
        if batch and ((max_tasks and len(batch) >= max_tasks)
                      or (max_bytes and size + task.num_bytes > max_bytes)):
            batches.append(batch)
            batch, size = [], 0
        batch.append(task.task_id)
        size += task.num_bytes
    if batch:
        batches.append(batch)
    return batches


@dataclass(frozen=True)
class HostTask:
    """A CPU tensor explicitly supplied for a slice on this receiving rank."""

    param_name: str
    dst_slice: tuple
    tensor: torch.Tensor


def _overlap(left, right, shape):
    for a, b, length in zip(left, right, shape):
        a0, a1, astep = a.indices(length)
        b0, b1, bstep = b.indices(length)
        if astep != 1 or bstep != 1:
            raise ValueError("Host reload slices require unit strides")
        if max(a0, b0) >= min(a1, b1):
            return False
    return True


def _prepare_host(tasks, plan, dst_module):
    parameters = dict(dst_module.named_parameters()) if dst_module is not None else {}
    destinations = [(op.param_name, op.my_slice) for op in plan.recv_ops]
    prepared = []
    for task in tasks:
        parameter = parameters[task.param_name]
        if len(task.dst_slice) != parameter.ndim:
            raise ValueError("Host reload must specify every destination dimension")
        target = parameter.data[task.dst_slice]
        source = task.tensor
        if source.device.type != "cpu" or source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError("Host reload requires a matching CPU tensor")
        if target.device.type != "cuda":
            raise ValueError("Host reload destination must be a GPU tensor")
        for name, slices in destinations:
            if name == task.param_name and _overlap(task.dst_slice, slices, parameter.shape):
                raise ValueError("Host reload overlaps another receiving task")
        destinations.append((task.param_name, task.dst_slice))
        # Retain both the caller's tensor and pinned staging until upload ends.
        staging = source if source.is_contiguous() and source.is_pinned() else source.contiguous().pin_memory()
        receive = target if target.is_contiguous() else torch.empty(target.shape, dtype=target.dtype, device=target.device)
        prepared.append((source, staging, receive, target))
    return prepared


def _maximum(value, group):
    tensor = torch.tensor(value, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return float(tensor.item())


@torch.inference_mode()
def execute(plan, src_module, dst_module, service, *, bandwidth, hosts, links=None,
            max_tasks=0, max_bytes=0, group=None, control_group=None, host_tasks=(),
            prealloc_started=None):
    """Preallocate the entire migration, rendezvous, upload/copy, then commit.

    Callers must freeze decode and offsets before entering, and pass the start
    of destination rebuilding/poisoning as prealloc_started when applicable.
    All ranks enter even if they have no receiving tasks. OOM is fatal; there is
    no per-batch allocation fallback. Resources are owned by this call only.
    """
    wall_start = time.perf_counter() if prealloc_started is None else prealloc_started
    tasks = collect_transfer_tasks(plan, dst_module, group=group)
    batches = schedule(tasks, bandwidth, max_tasks, max_bytes)
    agreed = [batches if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(agreed, src=0, group=control_group)
    batches = agreed[0]
    paths = {name: {"tasks": 0, "bytes": 0} for name in PATHS}
    for task in tasks:
        stats = paths[path_kind(task, hosts, links)]
        stats["tasks"] += 1
        stats["bytes"] += task.num_bytes

    error = None
    try:
        receives = prepare_reshard_receives(plan, dst_module)
        uploads = _prepare_host(host_tasks, plan, dst_module)
        host_stream = torch.cuda.Stream() if uploads else None
        # Includes destination poison on a separate preparation stream.
        torch.cuda.synchronize()
    except (RuntimeError, ValueError, KeyError) as exc:
        error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(group=control_group)
    dist.all_gather_object(errors, error, group=control_group)
    if any(errors):
        raise RuntimeError("AnchorTP preallocation failed: " + "; ".join(e for e in errors if e))
    # No model data transport may precede this all-rank readiness rendezvous.
    dist.barrier(group=control_group)
    prealloc_s = _maximum(time.perf_counter() - wall_start, control_group)

    if uploads:
        with torch.cuda.stream(host_stream):
            for _source, staging, receive, target in uploads:
                receive.copy_(staging, non_blocking=True)
                if receive is not target:
                    target.copy_(receive, non_blocking=True)
    service.p2p_order = "task-round"
    service.pack_target_bytes = service.pack_max_item_bytes = 0
    # Source slices become contiguous on this stream after preallocation.
    # Both local copies and NCCL must wait for that work in every batch.
    service.producer_stream = torch.cuda.current_stream()
    records = []
    by_id = {task.task_id: task for task in tasks}
    for ids in batches:
        start = time.perf_counter()
        service.set_task_order(ids)
        transaction = launch_reshard_plan(
            filter_plan_by_task_ids(plan, ids), src_module, dst_module, service,
            group=group, prepared_receives=receives,
        )
        transaction.wait().commit()
        service.producer_stream.synchronize()
        elapsed = _maximum(time.perf_counter() - start, control_group)
        transport = _maximum(transaction.transport_elapsed_s or 0.0, control_group)
        records.append({
            "tasks": len(ids), "bytes": sum(by_id[i].num_bytes for i in ids),
            "task_ids": ids, "elapsed_s": elapsed, "transport_s": transport,
            "overlap_steps": 0, "copy_service": dict(transaction.metadata.get("copy_service", {})),
        })
    if host_stream is not None:
        host_stream.synchronize()
    # Uploads and deferred noncontiguous writes are complete on every rank.
    dist.barrier(group=control_group)
    host_stats = torch.tensor([len(uploads), sum(t.numel() * t.element_size() for _, t, _, _ in uploads)], dtype=torch.int64)
    dist.all_reduce(host_stats, op=dist.ReduceOp.SUM, group=control_group)
    paths["host_to_gpu"] = dict(zip(("tasks", "bytes"), host_stats.tolist()))
    wall_s = _maximum(time.perf_counter() - wall_start, control_group)
    return {
        "execution_mode": "anchortp", "implementation": IMPLEMENTATION,
        "waves": len(records), "wave_records": records, "max_wave_tasks": max_tasks,
        "prealloc_s": prealloc_s, "wall_s": wall_s, "exposed_wait_s": wall_s,
        "overlap_steps": 0, "tpot_samples": 0, "paths": paths,
        "reload_bytes": paths["host_to_gpu"]["bytes"],
        **{f"tpot_{stat}": 0.0 for stat in ("mean_ms", "p95_ms", "p99_ms", "std_ms", "cv")},
    }


def provenance(root):
    files = ("examples/rl/benchmark_live_moe_tp.py", "megatron/core/resharding/anchortp.py",
             "megatron/core/resharding/async_execution.py", "megatron/core/resharding/copy_services/nccl_copy_service.py",
             "megatron/core/resharding/planner.py")
    exported_commit = os.environ.get("ANCHORTP_CODE_COMMIT")
    commit = exported_commit or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
    ).strip()
    return {
        "implementation": IMPLEMENTATION,
        "code_commit": commit,
        "code_dirty": None if exported_commit else bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        ),
        "code_sha256": {name: hashlib.sha256((Path(root) / name).read_bytes()).hexdigest() for name in files},
    }
