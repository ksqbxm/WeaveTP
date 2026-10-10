# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import heapq
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Mapping

import torch
import torch.distributed as dist

from .utils import (
    ParameterMetadata,
    ReshardPlan,
    ShardingDescriptor,
    TransferOp,
    _build_layer_module_prefix_map,
    _get_rank_in_group,
    extract_param_metadata,
    select_src_metadata_balanced,
)

logger = logging.getLogger(__name__)


def _source_search_config():
    algorithm = os.environ.get("WEAVETP_SOURCE_SEARCH")
    if algorithm is None:
        return None
    if algorithm not in {"greedy", "ls", "heap", "dp", "dfs", "dijkstra", "ilp"}:
        raise ValueError(f"Unknown WEAVETP_SOURCE_SEARCH: {algorithm!r}")
    budget = float(os.environ.get("WEAVETP_SEARCH_BUDGET_S", "300"))
    memory = float(os.environ.get("WEAVETP_SEARCH_MEM_GIB", "8"))
    gap = float(os.environ.get("WEAVETP_ILP_MIP_REL_GAP", "0"))
    if not math.isfinite(budget) or not 0 < budget <= 540:
        raise ValueError("WEAVETP_SEARCH_BUDGET_S must be in (0, 540]")
    if not math.isfinite(memory) or memory <= 0:
        raise ValueError("WEAVETP_SEARCH_MEM_GIB must be positive and finite")
    if gap not in (0.0, 1.0e-4):
        raise ValueError("WEAVETP_ILP_MIP_REL_GAP must be 0 or 1e-4")
    return dict(algorithm=algorithm, budget_s=budget, memory_gib=memory, mip_rel_gap=gap)


def _source_search_memory():
    """Current RSS for limits; the process high water is diagnostic only."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
    return info.rss / (1 << 30), peak / (1 << 30)


class _SourceSearchStopped(Exception):
    def __init__(self, status):
        self.status = status


class _SourceSearchLimits:
    def __init__(self, config):
        self.started = time.perf_counter()
        self.deadline = self.started + config["budget_s"]
        self.timed = config["algorithm"] not in ("greedy", "ls")
        self.memory_gib = config["memory_gib"]
        self.rss_start, self.peak_rss = _source_search_memory()
        self.rss_max = self.rss_start
        self.steps = 0
        self.best = None
        self.best_h = math.inf
        self.details = {}

    def check(self, force=False, enforce=True):
        self.steps += 1
        if not force and self.steps % 4096:
            return
        rss, peak = _source_search_memory()
        self.rss_max = max(self.rss_max, rss)
        self.peak_rss = max(self.peak_rss, peak)
        if not enforce:
            return
        if self.rss_max - self.rss_start >= self.memory_gib:
            raise _SourceSearchStopped("memory")
        if self.timed and time.perf_counter() >= self.deadline:
            raise _SourceSearchStopped("budget")

    def consider(self, inst, choice):
        h = inst.makespan(choice)
        if h < self.best_h:
            self.best, self.best_h = list(choice), h


@dataclass
class _SourceSearchInstance:
    base: list[float]
    inc: list[list[tuple[tuple[int, float], ...]]]
    default: list[int]
    keys: list[tuple[int, str]] = field(default_factory=list)
    candidates: list[list[ParameterMetadata]] = field(default_factory=list)

    def loads(self, choice):
        loads = list(self.base)
        for i, a in enumerate(choice):
            for r, seconds in self.inc[i][a]:
                loads[r] += seconds
        return loads

    def makespan(self, choice):
        return max(self.loads(choice), default=0.0)


def _build_source_search_instance(src_params, dst_params, bandwidth, latency, penalty,
                                  min_bytes, exclude_local, limits):
    resources, base, increments, defaults, keys, candidates = {}, [], [], [], [], []

    def increment(name, metadata, dst, rank, extra_latency):
        values = {}
        for (sender, receiver), seconds in _source_metadata_transfer_seconds(
            name, metadata, dst, rank, bandwidth, latency + extra_latency
        ).items():
            for key in (("link", sender, receiver), ("source", sender), ("destination", receiver)):
                if key not in resources:
                    resources[key] = len(base)
                    base.append(0.0)
                r = resources[key]
                values[r] = values.get(r, 0.0) + seconds
        return tuple(sorted(values.items()))

    for rank, params in dst_params.items():
        for name, dst in params.items():
            limits.check()
            source = src_params.get(name)
            if not source:
                raise RuntimeError(f"Destination parameter {name!r} on rank {rank} not found")
            if exclude_local:
                nonlocal_source = [m for m in source if all(
                    s != rank for s, _, _ in _determine_source_ranks_for_dst_param(name, m, dst, rank)
                )]
                source = nonlocal_source or source
            default = select_src_metadata_balanced(source, dst, rank, prefer_local_source=True)
            # Match select_src_metadata_balanced's EP filter before deduplicating routes.
            ep = dst.expert_parallel_group_ranks
            source_ep = source[0].expert_parallel_group_ranks
            if ep is not None and source_ep and len(ep) == len(source_ep):
                local = ep.index(rank)
                source = [m for m in source if m.expert_parallel_group_ranks
                          and m.expert_parallel_group_ranks.index(m.owner_rank) == local]
            routes = {}
            for m in sorted(source, key=lambda m: m.owner_rank):
                limits.check()
                ranks = tuple(s for s, _, _ in _determine_source_ranks_for_dst_param(name, m, dst, rank))
                routes.setdefault(ranks, m)
            transfers = _determine_source_ranks_for_dst_param(name, default, dst, rank)
            default_ranks = tuple(s for s, _, _ in transfers)
            routes[default_ranks] = default
            size = sum(_slice_numel(tuple(dst.shape), sl) * dst.element_size for _, _, sl in transfers)
            if len(routes) == 1 or size < min_bytes:
                for r, seconds in increment(name, default, dst, rank, 0.0):
                    base[r] += seconds
                continue
            choices = sorted(routes.values(), key=lambda m: m.owner_rank)
            default_index = choices.index(default)
            increments.append([increment(name, m, dst, rank, 0.0 if a == default_index else penalty)
                               for a, m in enumerate(choices)])
            defaults.append(default_index)
            keys.append((rank, name))
            candidates.append(choices)
    limits.check(force=True)
    return _SourceSearchInstance(base, increments, defaults, keys, candidates)


def _search_child(loads, inc):
    child = list(loads)
    for r, seconds in inc:
        child[r] += seconds
    return tuple(child)


def _search_path(path):
    choices = []
    while path is not None:
        a, path = path
        choices.append(a)
    return choices[::-1]


def _search_local_cost(loads, inc):
    return max((loads[r] + seconds for r, seconds in inc), default=0.0)


def _search_ls(inst, limits):
    loads, choice = list(inst.base), []
    try:
        for options in inst.inc:
            a = min(range(len(options)), key=lambda a: (_search_local_cost(loads, options[a]), a))
            choice.append(a)
            loads = list(_search_child(loads, options[a]))
            limits.check()
        limits.consider(inst, choice)
        for _ in range(50):
            changed = False
            for i, options in enumerate(inst.inc):
                for a, inc in enumerate(options):
                    limits.check()
                    if a == choice[i]:
                        continue
                    old_h = max(loads, default=0.0)
                    old_inc = options[choice[i]]
                    saved = {r: loads[r] for r, _ in old_inc + inc}
                    for r, seconds in old_inc:
                        loads[r] -= seconds
                    for r, seconds in inc:
                        loads[r] += seconds
                    if max(loads, default=0.0) < old_h:
                        choice[i] = a
                        changed = True
                    else:
                        for r, value in saved.items():
                            loads[r] = value
            if not changed:
                break
    finally:
        if len(choice) == len(inst.inc):
            limits.consider(inst, choice)
        limits.check(force=True, enforce=sys.exc_info()[0] is None)


def _search_heap(inst, limits):
    loads, versions = list(inst.base), [0] * len(inst.base)
    touched = [sorted({r for inc in options for r, _ in inc}) for options in inst.inc]
    heap, choice = [], list(inst.default)

    def entry(i):
        options = inst.inc[i]
        a = min(range(len(options)), key=lambda a: (_search_local_cost(loads, options[a]), a))
        return (_search_local_cost(loads, options[a]), i, a, tuple(versions[r] for r in touched[i]))

    try:
        for i in range(len(inst.inc)):
            heapq.heappush(heap, entry(i))
            limits.check()
        while heap:
            _, i, a, stamp = heapq.heappop(heap)
            if stamp != tuple(versions[r] for r in touched[i]):
                heapq.heappush(heap, entry(i))
            else:
                choice[i] = a
                for r, seconds in inst.inc[i][a]:
                    loads[r] += seconds
                    versions[r] += 1
                if not heap:
                    limits.consider(inst, choice)
            limits.check()
    finally:
        limits.check(force=True, enforce=sys.exc_info()[0] is None)


def _search_dfs(inst, limits):
    stack = [(0, tuple(inst.base), None)]
    try:
        while stack:
            depth, loads, path = stack.pop()
            limits.check()
            if max(loads, default=0.0) >= limits.best_h:
                continue
            if depth == len(inst.inc):
                limits.consider(inst, _search_path(path))
                continue
            children = []
            for a, inc in enumerate(inst.inc[depth]):
                child = _search_child(loads, inc)
                h = max(child, default=0.0)
                if h < limits.best_h:
                    children.append((h, a, child))
                limits.check()
            for _, a, child in sorted(children, reverse=True):
                stack.append((depth + 1, child, (a, path)))
    finally:
        limits.check(force=True, enforce=sys.exc_info()[0] is None)


def _search_dp(inst, limits):
    states = {tuple(inst.base): None}
    try:
        for depth, options in enumerate(inst.inc):
            following = {}
            for loads, path in states.items():
                for a, inc in enumerate(options):
                    child = _search_child(loads, inc)
                    if max(child, default=0.0) < limits.best_h:
                        parent = (a, path)
                        if depth + 1 == len(inst.inc):
                            limits.consider(inst, _search_path(parent))
                        else:
                            following.setdefault(child, parent)
                    limits.check()
            states = following
            if not states:
                break
    finally:
        limits.check(force=True, enforce=sys.exc_info()[0] is None)


def _search_dijkstra(inst, limits):
    start = tuple(inst.base)
    heap, seen, serial = [(max(start, default=0.0), 0, 0, start, None)], {(0, start)}, 0
    try:
        while heap:
            _, _, depth, loads, path = heapq.heappop(heap)
            if depth == len(inst.inc):
                limits.consider(inst, _search_path(path))
                return
            for a, inc in enumerate(inst.inc[depth]):
                child = _search_child(loads, inc)
                key = (depth + 1, child)
                if key not in seen:
                    seen.add(key)
                    parent = (a, path)
                    serial += 1
                    heapq.heappush(heap, (max(child, default=0.0), serial, depth + 1, child, parent))
                    if depth + 1 == len(inst.inc) and max(child, default=0.0) < limits.best_h:
                        limits.consider(inst, _search_path(parent))
                limits.check()
    finally:
        limits.check(force=True, enforce=sys.exc_info()[0] is None)


def _search_ilp(inst, limits, gap):
    if not inst.inc:
        limits.check(force=True)
        return
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    try:
        nx = sum(map(len, inst.inc))
        nres, nitems = len(inst.base), len(inst.inc)
        rows, cols, values = [], [], []
        j = 0
        for i, options in enumerate(inst.inc):
            for inc in options:
                for r, seconds in inc:
                    rows.append(r)
                    cols.append(j)
                    values.append(seconds)
                rows.append(nres + i)
                cols.append(j)
                values.append(1.0)
                j += 1
                limits.check()
        for r in range(nres):
            rows.append(r)
            cols.append(nx)
            values.append(-1.0)
        matrix = coo_matrix((values, (rows, cols)), shape=(nres + nitems, nx + 1)).tocsc()
        upper = np.r_[-np.asarray(inst.base), np.ones(nitems)]
        lower = np.r_[np.full(nres, -np.inf), np.ones(nitems)]
        limits.check(force=True)
        remaining = limits.deadline - time.perf_counter()
        if remaining <= 0:
            raise _SourceSearchStopped("budget")
        result = milp(
            np.r_[np.zeros(nx), 1.0], integrality=np.r_[np.ones(nx), 0],
            bounds=Bounds(np.zeros(nx + 1), np.r_[np.ones(nx), np.inf]),
            constraints=LinearConstraint(matrix, lower, upper),
            options={"time_limit": remaining, "mip_rel_gap": gap, "disp": False},
        )
        limits.details.update(ilp_status=int(result.status), ilp_message=str(result.message),
                              mip_gap=getattr(result, "mip_gap", None),
                              mip_dual_bound=getattr(result, "mip_dual_bound", None))
        if result.status not in (0, 1):
            raise RuntimeError(f"MILP failed: {result.message}")
        if result.x is not None:
            x = np.asarray(result.x)
            binary = x[:nx]
            if (not np.all(np.isfinite(x)) or np.any(binary < -1e-6)
                    or np.any(binary > 1 + 1e-6)
                    or np.any(np.abs(binary - np.rint(binary)) > 1e-6)):
                raise RuntimeError("MILP returned a non-integral incumbent")
            choice, offset = [], 0
            for options in inst.inc:
                block = binary[offset:offset + len(options)]
                if abs(float(block.sum()) - 1.0) > 1e-6:
                    raise RuntimeError("MILP incumbent violates one-choice constraint")
                choice.append(int(block.argmax()))
                offset += len(options)
            if inst.makespan(choice) > float(x[-1]) + 1e-6 * max(1.0, abs(float(x[-1]))):
                raise RuntimeError("MILP incumbent violates resource constraints")
            limits.consider(inst, choice)
        elif result.status == 0:
            raise RuntimeError("MILP reported success without an incumbent")
        if result.status == 1:
            raise _SourceSearchStopped("budget")
    finally:
        limits.check(force=True, enforce=sys.exc_info()[0] is None)


def _run_source_search(src_params, dst_params, config, bandwidth, latency, penalty,
                       min_bytes, exclude_local):
    limits = _SourceSearchLimits(config)
    inst, solver_start, solver_s, status = None, None, 0.0, "complete"
    try:
        inst = _build_source_search_instance(src_params, dst_params, bandwidth, latency,
                                            penalty, min_bytes, exclude_local, limits)
        limits.best = list(inst.default)
        limits.best_h = inst.makespan(inst.default)
        solver_start = time.perf_counter()
        if config["algorithm"] == "ilp":
            _search_ilp(inst, limits, config["mip_rel_gap"])
        else:
            solvers = dict(ls=_search_ls, heap=_search_heap, dp=_search_dp,
                           dfs=_search_dfs, dijkstra=_search_dijkstra)
            solvers[config["algorithm"]](inst, limits)
    except _SourceSearchStopped as stopped:
        status = stopped.status
    finally:
        if solver_start is not None:
            solver_s = time.perf_counter() - solver_start
    override = {}
    if inst is not None:
        for i, a in enumerate(limits.best):
            if a != inst.default[i]:
                override[inst.keys[i]] = inst.candidates[i][a]
    try:
        limits.check(force=True)
    except _SourceSearchStopped as stopped:
        if status == "complete":
            status = stopped.status
    stats = dict(
        algorithm=config["algorithm"], prepass_s=time.perf_counter() - limits.started,
        solver_s=solver_s, status=status, budget_s=config["budget_s"],
        memory_limit_gib=config["memory_gib"],
        decision_items=len(inst.inc) if inst is not None else None,
        log10_search_space=sum(math.log10(len(row)) for row in inst.inc) if inst is not None else None,
        overridden_entries=len(override), search_rss_delta_gib=limits.rss_max - limits.rss_start,
        rss_start_gib=limits.rss_start, peak_rss_gib=limits.peak_rss,
        rss_measurement_scope="prepass_and_solver_checkpoints",
        mip_rel_gap=config["mip_rel_gap"] if config["algorithm"] == "ilp" else None,
        **limits.details,
    )
    return override, stats


def _slice_numel(shape: tuple[int, ...], index: tuple[slice, ...]) -> int:
    """Return the number of elements selected by a basic positive-step slice."""
    count = 1
    for dim, item in zip(shape, index):
        if not isinstance(item, slice):
            continue
        start, stop, step = item.indices(dim)
        if step <= 0:
            raise ValueError("Reshard planner only supports positive slice steps")
        count *= max(0, (stop - start + step - 1) // step)
    return count


def _source_metadata_transfer_seconds(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    dst_rank: int,
    bandwidth_gbps: Mapping[tuple[int, int], float],
    latency_us: float,
) -> dict[tuple[int, int], float]:
    """Return estimated service time for every link used by one source replica."""
    transfers = _determine_source_ranks_for_dst_param(
        param_name, src_metadata, dst_metadata, dst_rank
    )
    seconds_by_pair: dict[tuple[int, int], float] = {}
    for src_rank, _src_slice, dst_slice in transfers:
        num_bytes = _slice_numel(tuple(dst_metadata.shape), dst_slice) * dst_metadata.element_size
        pair = (int(src_rank), int(dst_rank))
        gbps = max(float(bandwidth_gbps.get(pair, 1.0)), 1.0e-9)
        seconds_by_pair[pair] = seconds_by_pair.get(pair, 0.0) + (
            num_bytes * 8.0 / (gbps * 1.0e9) + latency_us * 1.0e-6
        )
    return seconds_by_pair


def _estimate_source_metadata_cost(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    dst_rank: int,
    bandwidth_gbps: Mapping[tuple[int, int], float],
    latency_us: float,
    link_load_s: Mapping[tuple[int, int], float] | None = None,
    source_load_s: Mapping[int, float] | None = None,
    destination_load_s: Mapping[int, float] | None = None,
) -> float:
    """Estimate projected makespan after assigning one equivalent source replica.

    A per-parameter fastest-link decision can funnel many replicated parameters onto
    the same source or PCIe path.  Include accumulated link, sender, and receiver load
    so source selection spreads work while still preferring genuinely faster paths.
    """
    link_load_s = link_load_s or {}
    source_load_s = source_load_s or {}
    destination_load_s = destination_load_s or {}
    seconds_by_pair = _source_metadata_transfer_seconds(
        param_name,
        src_metadata,
        dst_metadata,
        dst_rank,
        bandwidth_gbps,
        latency_us,
    )
    if not seconds_by_pair:
        return 0.0

    source_increments: dict[int, float] = {}
    for (src_rank, _), seconds in seconds_by_pair.items():
        source_increments[src_rank] = source_increments.get(src_rank, 0.0) + seconds

    projected_link = max(
        float(link_load_s.get(pair, 0.0)) + seconds
        for pair, seconds in seconds_by_pair.items()
    )
    projected_source = max(
        float(source_load_s.get(src_rank, 0.0)) + seconds
        for src_rank, seconds in source_increments.items()
    )
    projected_destination = float(destination_load_s.get(dst_rank, 0.0)) + sum(
        seconds_by_pair.values()
    )
    return max(projected_link, projected_source, projected_destination)


def _sort_ops_by_dst_offset(ops, dim):
    """Sort transfer ops by destination offset on the sharded dimension."""
    ops.sort(key=lambda op: op[2][dim].start if isinstance(op[2][dim], slice) else 0)


def _build_descriptors_for_param(
    src_metadata: ParameterMetadata, dst_metadata: ParameterMetadata
) -> list[ShardingDescriptor]:
    """Construct sharding descriptors (currently TP) for this parameter based on actual layout.
    Guard TP descriptor with size conservation so we don't mis-classify replicated tensors.
    """
    descriptors: list[ShardingDescriptor] = []

    # TP descriptor: allow when either side participates in TP
    if src_metadata.is_tp or dst_metadata.is_tp:
        # Prefer destination partition_dim, else source
        tp_dim = dst_metadata.partition_dim if dst_metadata.is_tp else src_metadata.partition_dim
        src_tp_ranks = src_metadata.tensor_parallel_group_ranks
        dst_tp_ranks = dst_metadata.tensor_parallel_group_ranks
        if src_tp_ranks is None or dst_tp_ranks is None:
            # Not enough context to build TP descriptor
            return descriptors
        src_stride = src_metadata.partition_stride if src_metadata.is_tp else 1
        dst_stride = dst_metadata.partition_stride if dst_metadata.is_tp else 1

        # Size conservation check on partition dim
        src_world = len(src_tp_ranks)
        dst_world = len(dst_tp_ranks)
        src_local = src_metadata.shape[tp_dim]
        dst_local = dst_metadata.shape[tp_dim]
        if src_world * src_local != dst_world * dst_local:
            raise RuntimeError(
                f"Cannot build TP descriptor for {dst_metadata.name} dim{tp_dim}: "
                f"src_world*src_local={src_world}*{src_local} != {dst_world}*{dst_local}. "
                "This usually means the param is marked TP but is effectively replicated on that "
                "dim or partition_dim/metadata is inconsistent between source and destination."
            )

        descriptors.append(
            ShardingDescriptor(
                name="tp",
                dim=tp_dim,
                src_stride=src_stride,
                dst_stride=dst_stride,
                src_dim_ranks=src_tp_ranks,
                dst_dim_ranks=dst_tp_ranks,
            )
        )
    return descriptors


def _plan_multi_dim_lcm(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    descriptors: list[ShardingDescriptor],
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """
    TP-only planner using LCM tiling to support strides on source/destination.
    - Requires exactly one TP descriptor
    - Supports arbitrary integer strides (contiguous micro-tiles)
    """
    if not descriptors:
        return []
    if len(descriptors) != 1:
        raise NotImplementedError(
            f"{param_name}: _plan_multi_dim_lcm supports TP-only (one descriptor)"
        )
    if descriptors[0].name != "tp":
        raise NotImplementedError(f"{param_name}: _plan_multi_dim_lcm expects TP descriptor")
    d = descriptors[0]
    if my_global_rank not in d.dst_dim_ranks:
        return []

    src_shape = tuple(src_metadata.shape)
    dst_shape = tuple(dst_metadata.shape)
    dim = d.dim
    src_world = len(d.src_dim_ranks)
    dst_world = len(d.dst_dim_ranks)
    src_local = src_shape[dim]
    dst_local = dst_shape[dim]
    if src_world * src_local != dst_world * dst_local:
        raise RuntimeError(
            f"{param_name}: size mismatch on TP dim{dim} "
            f"(src_world={src_world}, src_local={src_local}, "
            f"dst_world={dst_world}, dst_local={dst_local})"
        )
    # LCM tiling with strides
    Ns = src_world * max(1, d.src_stride)
    Nd = dst_world * max(1, d.dst_stride)
    full_len = dst_local * dst_world
    g = math.gcd(Ns, Nd)
    L = (Ns // g) * Nd
    if full_len % L != 0:
        raise RuntimeError(
            f"{param_name}: TP dim{dim} full_len {full_len} not divisible by LCM {L} "
            f"(Ns={Ns}, Nd={Nd})"
        )
    unit = full_len // L  # micro-tile length
    cps = L // Ns  # micro-tiles per source segment
    cpd = L // Nd  # micro-tiles per destination segment
    seg_src = cps * unit  # contiguous length per source segment
    seg_dst = cpd * unit  # contiguous length per destination segment
    dst_local_rank = _get_rank_in_group(my_global_rank, d.dst_dim_ranks)
    ops: list[tuple[int, tuple[slice, ...], tuple[slice, ...]]] = []
    # Sweep destination segments owned by this rank (handle destination stride)
    for k in range(max(1, d.dst_stride)):
        g_dst_seg = dst_local_rank + k * dst_world
        # Within this segment, enumerate the cpd micro-tiles
        for off in range(cpd):
            g_micro = g_dst_seg * cpd + off
            s_idx = g_micro // cps
            in_seg = g_micro % cps
            src_owner_in_dim = s_idx % src_world
            src_global_rank = d.src_dim_ranks[src_owner_in_dim]
            src_local_seg_idx = s_idx // src_world
            src_start = src_local_seg_idx * seg_src + in_seg * unit
            dst_start = k * seg_dst + off * unit
            # Build full N-D slices
            src_slice = [slice(None)] * len(src_shape)
            dst_slice = [slice(None)] * len(dst_shape)
            src_slice[dim] = slice(src_start, src_start + unit)
            dst_slice[dim] = slice(dst_start, dst_start + unit)
            ops.append((src_global_rank, tuple(src_slice), tuple(dst_slice)))

    _sort_ops_by_dst_offset(ops, dim)
    return ops


def _plan_block_interleaved(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    descriptors: list[ShardingDescriptor],
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """
    Block-interleaved TP planner for parameters with ``partition_sizes``.

    When a parameter packs multiple independently-sharded components of
    *different* sizes (e.g. Mamba in_proj packs z, x, B, C, dt), a simple
    contiguous concat produces the wrong layout.  This function treats each
    block independently: it gathers (or scatters) each block across TP ranks
    before moving to the next block.

    ``partition_sizes`` lists the per-TP-rank block sizes along the partition
    dim.  Block *i* occupies ``[sum(sizes[:i]), sum(sizes[:i+1]))`` in the
    local tensor on every TP rank.  In the *full* (TP-gathered) tensor, block
    *i* occupies ``[sum(full_sizes[:i]), sum(full_sizes[:i+1]))`` where
    ``full_sizes[i] = sizes[i] * src_tp_world``.
    """
    if not descriptors or descriptors[0].name != "tp":
        return []
    d = descriptors[0]
    if my_global_rank not in d.dst_dim_ranks:
        return []

    dim = d.dim
    src_shape = tuple(src_metadata.shape)
    dst_shape = tuple(dst_metadata.shape)
    src_world = len(d.src_dim_ranks)
    dst_world = len(d.dst_dim_ranks)
    dst_local_rank = _get_rank_in_group(my_global_rank, d.dst_dim_ranks)

    # Use partition_sizes from whichever side has it (prefer src)
    src_sizes = src_metadata.partition_sizes
    dst_sizes = dst_metadata.partition_sizes

    if src_sizes is None and dst_sizes is None:
        raise RuntimeError(f"{param_name}: _plan_block_interleaved called without partition_sizes")

    # Derive the full (un-sharded) block sizes
    if src_sizes is not None:
        num_blocks = len(src_sizes)
        full_sizes = [s * src_world for s in src_sizes]
    else:
        num_blocks = len(dst_sizes)
        full_sizes = [s * dst_world for s in dst_sizes]

    # Compute per-rank block sizes for both sides
    if src_sizes is None:
        src_sizes = [f // src_world for f in full_sizes]
    if dst_sizes is None:
        dst_sizes = [f // dst_world for f in full_sizes]

    # Validate conservation
    for i in range(num_blocks):
        if src_sizes[i] * src_world != dst_sizes[i] * dst_world:
            raise RuntimeError(
                f"{param_name}: block {i} size mismatch: "
                f"src_sizes[{i}]={src_sizes[i]}*{src_world} != "
                f"dst_sizes[{i}]={dst_sizes[i]}*{dst_world}"
            )

    ops: list[tuple[int, tuple[slice, ...], tuple[slice, ...]]] = []

    # For each block, compute the transfer ops independently
    src_block_offset = 0  # cumulative offset in source local tensor
    dst_block_offset = 0  # cumulative offset in destination local tensor

    for blk in range(num_blocks):
        src_blk_sz = src_sizes[blk]  # per-src-rank size of this block
        dst_blk_sz = dst_sizes[blk]  # per-dst-rank size of this block
        full_blk_sz = full_sizes[blk]

        # Within this block, use simple LCM tiling (stride=1)
        Ns = src_world
        Nd = dst_world
        g = math.gcd(Ns, Nd)
        L = (Ns // g) * Nd
        if full_blk_sz % L != 0:
            raise RuntimeError(
                f"{param_name}: block {blk} full_size {full_blk_sz} not divisible by LCM {L}"
            )
        unit = full_blk_sz // L
        cps = L // Ns
        cpd = L // Nd

        # This dst rank's segment within the block
        g_dst_seg = dst_local_rank
        for off in range(cpd):
            g_micro = g_dst_seg * cpd + off
            s_idx = g_micro // cps
            in_seg = g_micro % cps
            src_owner_in_dim = s_idx % src_world
            src_global_rank = d.src_dim_ranks[src_owner_in_dim]
            src_local_seg_idx = s_idx // src_world
            src_start = src_block_offset + src_local_seg_idx * (cps * unit) + in_seg * unit
            dst_start = dst_block_offset + off * unit

            src_slice = [slice(None)] * len(src_shape)
            dst_slice = [slice(None)] * len(dst_shape)
            src_slice[dim] = slice(src_start, src_start + unit)
            dst_slice[dim] = slice(dst_start, dst_start + unit)
            ops.append((src_global_rank, tuple(src_slice), tuple(dst_slice)))

        src_block_offset += src_blk_sz
        dst_block_offset += dst_blk_sz

    _sort_ops_by_dst_offset(ops, dim)
    return ops


def _finalize_dp_transfers(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Return receiver-side transfer for a parameter that is not TP-sharded.

    This is reached when we cannot build a TP sharding descriptor for the parameter
    (i.e., it is effectively replicated with respect to sharding).  We use this when the
    destination and source mode have no TP or the parameter is replicted on all ranks
    such as layernorm. If the source and destination DP groups match, we return a local
    full-tensor copy; otherwise we pick a source rank from the source DP group in a
    deterministic round-robin manner based on the receiver's global rank for better load
    distribution.
    """
    dst_dp_ranks = dst_metadata.data_parallel_group_ranks
    src_dp_ranks = src_metadata.data_parallel_group_ranks
    if my_global_rank not in dst_dp_ranks:
        return []

    dst_shape = dst_metadata.shape

    # Same DP layout - local copy (only if this rank has the source parameter)
    if src_dp_ranks == dst_dp_ranks and my_global_rank in src_dp_ranks:
        full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
        return [(my_global_rank, full_slice, full_slice)]

    # Different DP groups - use round-robin based on destination global rank for
    # better load balancing across source ranks. This ensures that destination
    # ranks are distributed across source ranks even when they have the same
    # position within their respective DP groups.
    #
    # In non-collocated mode, src_dp_ranks might include ranks that don't
    # have the source model (e.g., idle ranks or destination ranks). Filter to only
    # include the rank that provided this metadata (src_metadata.owner_rank).
    # src_metadata was selected by select_src_metadata_balanced, so owner_rank is the
    # actual source rank for this parameter.
    actual_src_rank = src_metadata.owner_rank
    src_global_rank = src_dp_ranks[my_global_rank % len(src_dp_ranks)]
    # Override with the actual source rank if the selected rank doesn't have the parameter
    if src_global_rank != actual_src_rank:
        src_global_rank = actual_src_rank
    full_slice = tuple(slice(None) for _ in range(len(dst_shape)))
    return [(src_global_rank, full_slice, full_slice)]


def _determine_source_ranks_for_dst_param(
    param_name: str,
    src_metadata: ParameterMetadata,
    dst_metadata: ParameterMetadata,
    my_global_rank: int,
) -> list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]:
    """Route to dimension-specific planner based on parameter sharding type."""

    # Regular TP/DP planning with EP-resolved metadata
    descriptors = _build_descriptors_for_param(src_metadata=src_metadata, dst_metadata=dst_metadata)
    if descriptors:
        # Use block-interleaved planner when partition_sizes is present
        # (e.g. Mamba in_proj packs components of different sizes)
        if src_metadata.partition_sizes is not None or dst_metadata.partition_sizes is not None:
            return _plan_block_interleaved(
                param_name=param_name,
                src_metadata=src_metadata,
                dst_metadata=dst_metadata,
                descriptors=descriptors,
                my_global_rank=my_global_rank,
            )
        return _plan_multi_dim_lcm(
            param_name=param_name,
            src_metadata=src_metadata,
            dst_metadata=dst_metadata,
            descriptors=descriptors,
            my_global_rank=my_global_rank,
        )
    # DP / replicated fallback
    return _finalize_dp_transfers(param_name, src_metadata, dst_metadata, my_global_rank)


def build_centralized_reshard_plan(
    src_module: torch.nn.Module,
    dst_module: torch.nn.Module,
    num_experts: int = None,
    group=None,
    src_rank_offset: int = 0,
    dst_rank_offset: int = 0,
    prefer_local_source: bool = True,
    source_bandwidth_gbps: Mapping[tuple[int, int], float] | None = None,
    source_reference_bandwidth_gbps: Mapping[tuple[int, int], float] | None = None,
    source_latency_us: float = 5.0,
    source_reroute_penalty_us: float = 0.0,
    source_reroute_min_gain_pct: float = 15.0,
    source_reroute_min_contention_gain_pct: float = 10.0,
    source_reroute_min_global_gain_pct: float = 10.0,
    source_reroute_min_bytes: int = 1 << 20,
    source_exclude_local: bool = False,
) -> ReshardPlan:
    """
    Centralized planning: Rank 0 builds complete plan for all ranks, then scatters.

    Supports None for src_module and/or dst_module to enable non-collocated mode:
    - src_module=None: Rank doesn't have source model (destination-only)
    - dst_module=None: Rank doesn't have destination model (source-only)
    - Both provided: Rank has both models (collocated mode)

    Each rank provides metadata only for the models it owns, including parallel group
    membership (tensor_parallel_group_ranks, expert_parallel_group_ranks, etc.).
    This metadata is sufficient for rank 0 to build correct transfer plans without
    requiring dummy models.
    """
    search_config = _source_search_config()
    use_search = (search_config is not None and search_config["algorithm"] != "greedy"
                  and source_bandwidth_gbps is not None)
    record_search = search_config is not None and source_bandwidth_gbps is not None

    # Use group.rank() instead of dist.get_rank(group) to support cross-cluster
    # ProcessGroups where members have independent default PGs (same default rank).
    my_global_rank = group.rank() if group is not None else dist.get_rank()
    world_size = group.size() if group is not None else dist.get_world_size()

    # Shared cache for deduplicating rank lists across all metadata on this
    # rank.  Params sharing the same TP/DP/EP/PP groups will reference one
    # list object, making pickle ~75% smaller for the gather.
    _rank_list_cache: dict = {}

    def _extract_metadata(module, rank_offset):
        """Extract per-parameter metadata from a module, or [] if module is None."""
        if module is None:
            return []
        pg = getattr(module, "pg_collection", None)
        if pg is None:
            raise ValueError("Module must have pg_collection")
        layer_prefix_map = _build_layer_module_prefix_map(module)
        return [
            extract_param_metadata(
                p,
                name,
                my_global_rank,
                pg,
                num_experts=num_experts,
                layer_module_prefix_map=layer_prefix_map,
                rank_offset=rank_offset,
                _rank_list_cache=_rank_list_cache,
            )
            for name, p in module.named_parameters(recurse=True)
        ]

    my_src_metadata = _extract_metadata(src_module, src_rank_offset)
    my_dst_metadata = _extract_metadata(dst_module, dst_rank_offset)

    # Gather metadata to rank 0 only (not all ranks) to save CPU memory.
    # Other ranks don't need the full metadata — they only need their own plan.
    all_src_metadata_by_rank = [None] * world_size if my_global_rank == 0 else None
    all_dst_metadata_by_rank = [None] * world_size if my_global_rank == 0 else None
    dist.gather_object(my_src_metadata, all_src_metadata_by_rank, group_dst=0, group=group)
    dist.gather_object(my_dst_metadata, all_dst_metadata_by_rank, group_dst=0, group=group)

    # Free local metadata — no longer needed after gather.
    del my_src_metadata, my_dst_metadata

    # Parameter to metadata maps keyed by resolved_name (only populated on rank 0)
    dst_param_metadata_by_rank = {}
    src_param_metadata: dict[str, list[ParameterMetadata]] = {}

    if my_global_rank == 0:
        for rank_id, rank_metadata_list in enumerate(all_dst_metadata_by_rank):
            dst_param_metadata_by_rank[rank_id] = {m.resolved_name: m for m in rank_metadata_list}
        for rank_metadata_list in all_src_metadata_by_rank:
            for metadata in rank_metadata_list:
                src_param_metadata.setdefault(metadata.resolved_name, []).append(metadata)

        # Free the raw gathered lists — data is now in the indexed dicts.
        del all_src_metadata_by_rank, all_dst_metadata_by_rank

    # Build the plan on global rank 0 and broadcast to all ranks
    if my_global_rank == 0:
        plans_for_all_ranks = {r: ReshardPlan([], []) for r in range(world_size)}
        baseline_plans_for_all_ranks = (
            {r: ReshardPlan([], []) for r in range(world_size)}
            if source_bandwidth_gbps is not None
            else None
        )
        rerouted_task_ids_by_rank = {r: set() for r in range(world_size)}
        # Global monotonically increasing ID for non-local transfers.
        # This is shared between the corresponding send/recv ops so that
        # NVSHMEM can build schedule.
        next_task_id = 0
        baseline_next_task_id = 0
        link_load_s: dict[tuple[int, int], float] = {}
        source_load_s: dict[int, float] = {}
        destination_load_s: dict[int, float] = {}
        baseline_link_load_s: dict[tuple[int, int], float] = {}
        baseline_source_load_s: dict[int, float] = {}
        baseline_destination_load_s: dict[int, float] = {}
        source_route_decisions = 0
        source_route_multi_source_decisions = 0
        source_route_proposals = 0
        source_route_changes = 0
        source_route_accepted_bytes = 0
        source_exclude_local_fallbacks = 0
        source_route_rejections = 0
        source_route_small_rejections = 0
        source_route_contention_rejections = 0
        source_route_global_rejections = 0
        accepted_predicted_gains: list[float] = []
        accepted_contention_gains: list[float] = []
        accepted_default_bytes_by_pair: dict[tuple[int, int], int] = {}
        accepted_candidate_bytes_by_pair: dict[tuple[int, int], int] = {}

        audit_path = os.environ.get("WEAVETP_SEARCH_AUDIT_PATH")
        if audit_path and source_bandwidth_gbps is not None:
            import pickle
            from pathlib import Path

            path = Path(audit_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                pickle.dump(dict(
                    src_params=src_param_metadata, dst_params=dst_param_metadata_by_rank,
                    kwargs=dict(source_bandwidth_gbps=source_bandwidth_gbps,
                                source_reference_bandwidth_gbps=source_reference_bandwidth_gbps,
                                source_latency_us=source_latency_us,
                                source_reroute_penalty_us=source_reroute_penalty_us,
                                source_reroute_min_gain_pct=source_reroute_min_gain_pct,
                                source_reroute_min_contention_gain_pct=source_reroute_min_contention_gain_pct,
                                source_reroute_min_global_gain_pct=source_reroute_min_global_gain_pct,
                                source_reroute_min_bytes=source_reroute_min_bytes,
                                source_exclude_local=source_exclude_local,
                                prefer_local_source=prefer_local_source),
                ), handle, protocol=pickle.HIGHEST_PROTOCOL)
        search_override, search_stats = {}, None
        if use_search:
            search_override, search_stats = _run_source_search(
                src_param_metadata, dst_param_metadata_by_rank, search_config,
                source_bandwidth_gbps, source_latency_us, source_reroute_penalty_us,
                source_reroute_min_bytes, source_exclude_local,
            )
        elif record_search:
            search_stats = dict(
                algorithm="greedy", prepass_s=0.0, solver_s=None, status="complete",
                budget_s=search_config["budget_s"], memory_limit_gib=search_config["memory_gib"],
                decision_items=None, log10_search_space=None, overridden_entries=0,
                search_rss_delta_gib=None, rss_start_gib=None, peak_rss_gib=None,
                rss_measurement_scope="planner_endpoints", mip_rel_gap=None,
            )
        assignment_start = time.perf_counter() if record_search else None

        # Pipeline-parallel (PP) "mapping" is handled implicitly.
        # Each rank contributes metadata only for the parameters it actually owns
        # (i.e., the module partitioning for its PP stage). When PP sizes differ
        # between source and destination, we don't compute an explicit stage-to-stage
        # mapping here; instead, we iterate destination ranks and plan copies for the
        # parameters present on those ranks. Any source rank that has the same logical
        # parameter (matched by resolved_name) can serve as a sender (with DP balancing),
        # and TP slicing is applied when applicable.
        for dst_rank in range(world_size):
            dst_rank_params = dst_param_metadata_by_rank.get(dst_rank, {})
            for resolved_name, dst_metadata in dst_rank_params.items():
                src_meta_list = src_param_metadata.get(resolved_name)
                if not src_meta_list:
                    raise RuntimeError(
                        f"Destination parameter '{resolved_name}' on rank {dst_rank} "
                        "not found in source model."
                    )
                if source_exclude_local:
                    nonlocal_src_meta = [
                        metadata
                        for metadata in src_meta_list
                        if all(
                            source_rank != dst_rank
                            for source_rank, _src_slice, _dst_slice in (
                                _determine_source_ranks_for_dst_param(
                                    resolved_name,
                                    metadata,
                                    dst_metadata,
                                    dst_rank,
                                )
                            )
                        )
                    ]
                    if nonlocal_src_meta:
                        src_meta_list = nonlocal_src_meta
                    else:
                        source_exclude_local_fallbacks += 1
                # Choose a representative source metadata. With a measured bandwidth
                # matrix, optimize projected global makespan instead of selecting the
                # independently fastest source for every parameter.
                source_cost_fn = None
                default_src_metadata = None
                if source_bandwidth_gbps is not None:
                    default_src_metadata = select_src_metadata_balanced(
                        src_meta_list,
                        dst_metadata,
                        dst_rank,
                        # Keep the comparison route identical to the normal
                        # baseline plan. The bandwidth-aware candidate may opt
                        # into evaluating non-local replicas below.
                        prefer_local_source=True,
                    )
                    source_cost_fn = lambda metadata: _estimate_source_metadata_cost(
                        resolved_name,
                        metadata,
                        dst_metadata,
                        dst_rank,
                        source_bandwidth_gbps,
                        source_latency_us,
                        link_load_s,
                        source_load_s,
                        destination_load_s,
                    )
                if use_search:
                    src_metadata = search_override.get((dst_rank, resolved_name), default_src_metadata)
                else:
                    src_metadata = select_src_metadata_balanced(
                        src_meta_list,
                        dst_metadata,
                        dst_rank,
                        prefer_local_source=prefer_local_source,
                        source_cost_fn=source_cost_fn,
                    )
                sources = _determine_source_ranks_for_dst_param(
                    resolved_name, src_metadata, dst_metadata, dst_rank
                )
                route_accepted = False
                if source_bandwidth_gbps is not None:
                    source_route_decisions += 1
                    if len(src_meta_list) > 1:
                        source_route_multi_source_decisions += 1
                    default_sources = _determine_source_ranks_for_dst_param(
                        resolved_name, default_src_metadata, dst_metadata, dst_rank
                    )
                    route_changed = [source[0] for source in sources] != [
                        source[0] for source in default_sources
                    ]
                    if route_changed and use_search:
                        source_route_proposals += 1
                        source_route_changes += 1
                        route_accepted = True
                        for transfers, counter in (
                            (default_sources, accepted_default_bytes_by_pair),
                            (sources, accepted_candidate_bytes_by_pair),
                        ):
                            for src_rank, _src_slice, dst_slice in transfers:
                                pair = (src_rank, dst_rank)
                                num_bytes = _slice_numel(tuple(dst_metadata.shape), dst_slice) * dst_metadata.element_size
                                counter[pair] = counter.get(pair, 0) + num_bytes
                                if counter is accepted_candidate_bytes_by_pair:
                                    source_route_accepted_bytes += num_bytes
                    if route_changed and not use_search:
                        source_route_proposals += 1
                        candidate_bytes = sum(
                            _slice_numel(tuple(dst_metadata.shape), dst_slice)
                            * dst_metadata.element_size
                            for _src_rank, _src_slice, dst_slice in sources
                        )
                        if candidate_bytes < source_reroute_min_bytes:
                            source_route_small_rejections += 1
                            source_route_rejections += 1
                            src_metadata = default_src_metadata
                            sources = default_sources
                        else:
                            contention_gain_pct = float("inf")
                            if source_reference_bandwidth_gbps is not None:
                                default_loaded_cost = _estimate_source_metadata_cost(
                                    resolved_name,
                                    default_src_metadata,
                                    dst_metadata,
                                    dst_rank,
                                    source_bandwidth_gbps,
                                    source_latency_us,
                                )
                                candidate_loaded_cost = _estimate_source_metadata_cost(
                                    resolved_name,
                                    src_metadata,
                                    dst_metadata,
                                    dst_rank,
                                    source_bandwidth_gbps,
                                    source_latency_us + source_reroute_penalty_us,
                                )
                                default_idle_cost = _estimate_source_metadata_cost(
                                    resolved_name,
                                    default_src_metadata,
                                    dst_metadata,
                                    dst_rank,
                                    source_reference_bandwidth_gbps,
                                    source_latency_us,
                                )
                                candidate_idle_cost = _estimate_source_metadata_cost(
                                    resolved_name,
                                    src_metadata,
                                    dst_metadata,
                                    dst_rank,
                                    source_reference_bandwidth_gbps,
                                    source_latency_us + source_reroute_penalty_us,
                                )
                                default_contention = default_loaded_cost / max(
                                    default_idle_cost, 1.0e-12
                                )
                                candidate_contention = candidate_loaded_cost / max(
                                    candidate_idle_cost, 1.0e-12
                                )
                                contention_gain_pct = 100.0 * (
                                    default_contention - candidate_contention
                                ) / max(default_contention, 1.0e-12)
                            default_cost = _estimate_source_metadata_cost(
                                resolved_name,
                                default_src_metadata,
                                dst_metadata,
                                dst_rank,
                                source_bandwidth_gbps,
                                source_latency_us,
                                link_load_s,
                                source_load_s,
                                destination_load_s,
                            )
                            candidate_cost = _estimate_source_metadata_cost(
                                resolved_name,
                                src_metadata,
                                dst_metadata,
                                dst_rank,
                                source_bandwidth_gbps,
                                source_latency_us + source_reroute_penalty_us,
                                link_load_s,
                                source_load_s,
                                destination_load_s,
                            )
                            predicted_gain_pct = 100.0 * (
                                default_cost - candidate_cost
                            ) / max(default_cost, 1.0e-12)
                            if (
                                contention_gain_pct
                                < source_reroute_min_contention_gain_pct
                            ):
                                source_route_contention_rejections += 1
                                source_route_rejections += 1
                                src_metadata = default_src_metadata
                                sources = default_sources
                            elif predicted_gain_pct >= source_reroute_min_gain_pct:
                                source_route_changes += 1
                                source_route_accepted_bytes += candidate_bytes
                                route_accepted = True
                                accepted_predicted_gains.append(predicted_gain_pct)
                                accepted_contention_gains.append(contention_gain_pct)
                                for src_rank, _src_slice, dst_slice in default_sources:
                                    pair = (src_rank, dst_rank)
                                    num_bytes = (
                                        _slice_numel(
                                            tuple(dst_metadata.shape), dst_slice
                                        )
                                        * dst_metadata.element_size
                                    )
                                    accepted_default_bytes_by_pair[pair] = (
                                        accepted_default_bytes_by_pair.get(pair, 0)
                                        + num_bytes
                                    )
                                for src_rank, _src_slice, dst_slice in sources:
                                    pair = (src_rank, dst_rank)
                                    num_bytes = (
                                        _slice_numel(
                                            tuple(dst_metadata.shape), dst_slice
                                        )
                                        * dst_metadata.element_size
                                    )
                                    accepted_candidate_bytes_by_pair[pair] = (
                                        accepted_candidate_bytes_by_pair.get(pair, 0)
                                        + num_bytes
                                    )
                            else:
                                source_route_rejections += 1
                                src_metadata = default_src_metadata
                                sources = default_sources
                    seconds_by_pair = _source_metadata_transfer_seconds(
                        resolved_name,
                        src_metadata,
                        dst_metadata,
                        dst_rank,
                        source_bandwidth_gbps,
                        source_latency_us
                        + (source_reroute_penalty_us if route_accepted else 0.0),
                    )
                    for pair, seconds in seconds_by_pair.items():
                        src_rank = pair[0]
                        link_load_s[pair] = link_load_s.get(pair, 0.0) + seconds
                        source_load_s[src_rank] = source_load_s.get(src_rank, 0.0) + seconds
                        destination_load_s[dst_rank] = (
                            destination_load_s.get(dst_rank, 0.0) + seconds
                        )
                    baseline_seconds_by_pair = _source_metadata_transfer_seconds(
                        resolved_name,
                        default_src_metadata,
                        dst_metadata,
                        dst_rank,
                        source_bandwidth_gbps,
                        source_latency_us,
                    )
                    for pair, seconds in baseline_seconds_by_pair.items():
                        src_rank = pair[0]
                        baseline_link_load_s[pair] = (
                            baseline_link_load_s.get(pair, 0.0) + seconds
                        )
                        baseline_source_load_s[src_rank] = (
                            baseline_source_load_s.get(src_rank, 0.0) + seconds
                        )
                        baseline_destination_load_s[dst_rank] = (
                            baseline_destination_load_s.get(dst_rank, 0.0) + seconds
                        )
                    for src_rank, src_slice, dst_slice in default_sources:
                        baseline_task_id = baseline_next_task_id
                        baseline_next_task_id += 1
                        baseline_plans_for_all_ranks[dst_rank].recv_ops.append(
                            TransferOp(
                                param_name=dst_metadata.name,
                                peer_rank=src_rank,
                                is_send=False,
                                my_slice=dst_slice,
                                peer_slice=src_slice,
                                task_id=baseline_task_id,
                            )
                        )
                        baseline_plans_for_all_ranks[src_rank].send_ops.append(
                            TransferOp(
                                param_name=default_src_metadata.name,
                                peer_rank=dst_rank,
                                is_send=True,
                                my_slice=src_slice,
                                peer_slice=dst_slice,
                                task_id=baseline_task_id,
                            )
                        )
                for src_rank, src_slice, dst_slice in sources:
                    task_id = next_task_id
                    next_task_id += 1

                    if route_accepted:
                        rerouted_task_ids_by_rank[src_rank].add(task_id)
                        rerouted_task_ids_by_rank[dst_rank].add(task_id)

                    plans_for_all_ranks[dst_rank].recv_ops.append(
                        TransferOp(
                            param_name=dst_metadata.name,
                            peer_rank=src_rank,
                            is_send=False,
                            my_slice=dst_slice,
                            peer_slice=src_slice,
                            task_id=task_id,
                        )
                    )
                    plans_for_all_ranks[src_rank].send_ops.append(
                        TransferOp(
                            param_name=src_metadata.name,
                            peer_rank=dst_rank,
                            is_send=True,
                            my_slice=src_slice,
                            peer_slice=dst_slice,
                            task_id=task_id,
                        )
                    )
        if record_search:
            search_stats["assignment_loop_s"] = time.perf_counter() - assignment_start
        if source_bandwidth_gbps is not None:
            projected_critical_s = max(
                [0.0]
                + list(link_load_s.values())
                + list(source_load_s.values())
                + list(destination_load_s.values())
            )
            baseline_projected_critical_s = max(
                [0.0]
                + list(baseline_link_load_s.values())
                + list(baseline_source_load_s.values())
                + list(baseline_destination_load_s.values())
            )
            projected_global_gain_pct = 100.0 * (
                baseline_projected_critical_s - projected_critical_s
            ) / max(baseline_projected_critical_s, 1.0e-12)
            tentative_changes = source_route_changes
            global_gate_accepted = (
                tentative_changes > 0
                and projected_global_gain_pct >= source_reroute_min_global_gain_pct
            )
            if record_search:
                search_stats.update(
                    predicted_default_H_s=baseline_projected_critical_s,
                    predicted_H_s=projected_critical_s,
                    cached_plan_predicted_H_s=(projected_critical_s if global_gate_accepted
                                              else baseline_projected_critical_s),
                    gate_outcome=("accepted" if global_gate_accepted else
                                  "rejected" if tentative_changes else "no_change"),
                )
            if not global_gate_accepted:
                plans_for_all_ranks = baseline_plans_for_all_ranks
                rerouted_task_ids_by_rank = {
                    rank_id: set() for rank_id in range(world_size)
                }
                source_route_rejections += tentative_changes
                source_route_global_rejections = tentative_changes
                source_route_changes = 0
                source_route_accepted_bytes = 0
                accepted_default_bytes_by_pair.clear()
                accepted_candidate_bytes_by_pair.clear()
            print(
                "Bandwidth-aware source assignment: "
                f"decisions={source_route_decisions} "
                f"multi_source={source_route_multi_source_decisions} "
                f"proposed={source_route_proposals} "
                f"changed_from_default={source_route_changes} "
                f"rerouted_bytes={source_route_accepted_bytes} "
                f"exclude_local_fallbacks={source_exclude_local_fallbacks} "
                f"rejected_total={source_route_rejections} "
                f"rejected_low_gain="
                f"{source_route_rejections - source_route_small_rejections - source_route_contention_rejections} "
                f"rejected_small={source_route_small_rejections} "
                f"rejected_no_dynamic_gain={source_route_contention_rejections} "
                f"rejected_global={source_route_global_rejections} "
                f"min_gain_pct={source_reroute_min_gain_pct:.2f} "
                f"min_contention_gain_pct="
                f"{source_reroute_min_contention_gain_pct:.2f} "
                f"min_global_gain_pct={source_reroute_min_global_gain_pct:.2f} "
                f"reroute_penalty_us={source_reroute_penalty_us:.2f} "
                f"min_bytes={source_reroute_min_bytes} "
                f"accepted_gain_mean_pct="
                f"{statistics.fmean(accepted_predicted_gains) if accepted_predicted_gains else 0.0:.2f} "
                f"accepted_contention_gain_mean_pct="
                f"{statistics.fmean(accepted_contention_gains) if accepted_contention_gains else 0.0:.2f} "
                f"baseline_projected_critical_s={baseline_projected_critical_s:.6f} "
                f"candidate_projected_critical_s={projected_critical_s:.6f} "
                f"projected_global_gain_pct={projected_global_gain_pct:.2f} "
                f"global_gate_accepted={global_gate_accepted}",
                flush=True,
            )
        route_stats = {
            "decisions": source_route_decisions,
            "multi_source": source_route_multi_source_decisions,
            "proposed": source_route_proposals,
            "accepted": source_route_changes,
            "rerouted_bytes": source_route_accepted_bytes,
            "accepted_default_bytes_by_link": [
                {"src": src, "dst": dst, "bytes": num_bytes}
                for (src, dst), num_bytes in sorted(
                    accepted_default_bytes_by_pair.items()
                )
            ],
            "accepted_candidate_bytes_by_link": [
                {"src": src, "dst": dst, "bytes": num_bytes}
                for (src, dst), num_bytes in sorted(
                    accepted_candidate_bytes_by_pair.items()
                )
            ],
            "exclude_local_fallbacks": source_exclude_local_fallbacks,
            "rejected": source_route_rejections,
            "rejected_small": source_route_small_rejections,
            "rejected_no_dynamic_gain": source_route_contention_rejections,
            "rejected_global": source_route_global_rejections,
            "projected_global_gain_pct": (
                projected_global_gain_pct
                if source_bandwidth_gbps is not None
                else 0.0
            ),
            "global_gate_accepted": (
                global_gate_accepted if source_bandwidth_gbps is not None else False
            ),
        }
        if record_search:
            route_stats["source_search"] = search_stats
        if source_bandwidth_gbps is not None and global_gate_accepted:
            for rank_id, rank_plan in plans_for_all_ranks.items():
                # Dirty refresh can deliberately use the untouched source layout
                # while the base migration evaluates bandwidth-aware rerouting.
                setattr(
                    rank_plan,
                    "baseline_plan",
                    baseline_plans_for_all_ranks[rank_id],
                )
        for rank_id, rank_plan in plans_for_all_ranks.items():
            rank_plan.rerouted_task_ids = frozenset(
                rerouted_task_ids_by_rank[rank_id]
            )
            setattr(rank_plan, "source_route_stats", route_stats)
        plans_list = [plans_for_all_ranks[r] for r in range(world_size)]

        # Free planning intermediates on rank 0 before the scatter.
        del plans_for_all_ranks, dst_param_metadata_by_rank, src_param_metadata
    else:
        plans_list = None

    # Scatter: each rank receives only its own plan (not all plans).
    my_plan_list = [None]
    torch.distributed.scatter_object_list(my_plan_list, plans_list, group_src=0, group=group)
    my_plan = my_plan_list[0]
    del plans_list  # Free the full list on rank 0.

    logger.info(
        f"Rank {my_global_rank}: Received plan - {len(my_plan.recv_ops)} recvs, "
        f"{len(my_plan.send_ops)} sends"
    )

    return my_plan
