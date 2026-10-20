#!/usr/bin/env python3
"""WeaveTP planner ablation on one node (TP2->TP4, EP=2), one torchrun per launch.

The four levels differ only in planning (see README.md). The live benchmark
module is imported unchanged; this process patches its module globals
(``_decode`` and, for L1 only, ``launch_reshard_plan``) around each migration.
Rank r drives the r-th GPU in PCI order (CUDA_DEVICE_ORDER=PCI_BUS_ID).
"""

from __future__ import annotations

import os
import sys


def node_cpus(node: int) -> list[int]:
    """CPU ids of one NUMA node, parsed exactly like cpu_flood.py."""
    out = []
    with open(f"/sys/devices/system/node/node{node}/cpulist") as handle:
        for part in handle.read().strip().split(","):
            a, _, b = part.partition("-")
            out += range(int(a), int(b or a) + 1)
    return out


def read(path: str) -> str:
    with open(path) as handle:
        return handle.read().strip()


def rank_layout(rank_cpus: int) -> tuple[list[int], dict[int, list[int]]]:
    """NUMA node of each rank's GPU, and each rank's slice of the first CPUs of that node."""
    root = "/sys/bus/pci/devices"
    nodes = [int(read(f"{root}/{address}/numa_node")) for address in sorted(os.listdir(root))
             if read(f"{root}/{address}/vendor") == "0x10de"
             and read(f"{root}/{address}/class").startswith("0x03")]
    cpus = {}
    for node in sorted(set(nodes)):
        available = node_cpus(node)
        for slot, rank in enumerate(r for r in range(len(nodes)) if nodes[r] == node):
            cpus[rank] = available[slot * rank_cpus:(slot + 1) * rank_cpus]
    return nodes, cpus


if "LOCAL_RANK" in os.environ:
    # Bind before torch starts any thread; later threads inherit the mask.
    NODES, RANK_CPUS = rank_layout(int(os.environ["ABLATION_RANK_CPUS"]))
    os.sched_setaffinity(0, RANK_CPUS[int(os.environ["LOCAL_RANK"])])

import json
import math
import random
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples" / "rl"))

import benchmark_live_moe_tp as B  # noqa: E402

from megatron.core.inference.contexts import StaticInferenceContext  # noqa: E402
from megatron.core.resharding import (  # noqa: E402
    BandwidthAwareRefitPolicy,
    ResidualBandwidthTracker,
    ResidualBandwidthWaveScheduler,
    build_centralized_reshard_plan,
    collect_transfer_tasks,
    filter_plan_by_task_ids,
    restrict_plan_sequence,
    select_adaptive_hybrid_policy,
    swap_model_weights,
)
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.rl.parallel_utils import build_inference_pg_collection  # noqa: E402
from megatron.training import get_args  # noqa: E402
from megatron.training import get_model as get_training_model  # noqa: E402
from megatron.training.arguments import (  # noqa: E402
    core_transformer_config_from_args,
    parse_and_validate_args,
)
from megatron.training.initialize import initialize_megatron  # noqa: E402
from tools.resharding.tp_build_src_dst_demo import profile_p2p_bandwidth_matrix  # noqa: E402

TPOT_STEPS = 10
SEED = 2026
REAL_DECODE = B._decode
REAL_LAUNCH = B.launch_reshard_plan


def parse_flood(spec: str) -> dict[int, int]:
    """'none' or 'node:count,node:count'."""
    if spec == "none":
        return {}
    return {int(node): int(count) for node, count in (item.split(":") for item in spec.split(","))}


def allowed_ranks(nodes: list[int], m: int) -> list[int]:
    """First m ranks taken alternately from each NUMA node, in rank order."""
    by_node = [[r for r in range(len(nodes)) if nodes[r] == node] for node in sorted(set(nodes))]
    order = [ranks[i] for i in range(max(map(len, by_node))) for ranks in by_node if i < len(ranks)]
    return sorted(order[:m])


def bandwidth_ratios(matrix: dict, nodes: list[int]) -> dict:
    """Mean intra-node bandwidth over the mean cross-node bandwidth, overall and per direction."""
    def mean(select):
        return statistics.fmean(gbps for (s, d), gbps in matrix.items() if s != d and select(s, d))

    intra = mean(lambda s, d: nodes[s] == nodes[d])
    out = dict(intra_gbps=intra, ratio_mean=intra / mean(lambda s, d: nodes[s] != nodes[d]))
    for a in sorted(set(nodes)):
        for b in sorted(set(nodes) - {a}):
            out[f"ratio_{a}to{b}"] = intra / mean(lambda s, d: nodes[s] == a and nodes[d] == b)
    return out


def pad_records(rows: list[tuple]) -> list[tuple[int, int, int, int]]:
    """L1 padding: per (src, dst, src param), shard bytes minus the bytes actually needed.

    rows: (src, dst, src_name, needed_bytes, task_id, shard_bytes) for every send op.
    Returns (anchor_task_id, src, dst, pad_bytes), anchored on the group's smallest task id.
    """
    groups: dict[tuple, list] = {}
    for src, dst, name, needed, task_id, shard in rows:
        group = groups.setdefault((src, dst, name), [0, task_id, shard])
        group[0] += needed
        group[1] = min(group[1], task_id)
    return sorted((anchor, src, dst, shard - needed)
                  for (src, dst, _name), (needed, anchor, shard) in groups.items() if shard > needed)


class Floods:
    """cpu_flood.py children of global rank 0; the tag argument lets bash `pkill -f` them."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.label = "none"
        self.procs: list[subprocess.Popen] = []

    def set(self, label: str) -> None:
        if label == self.label:
            return
        self.stop()
        script = str(HERE / "cpu_flood.py")
        self.procs = [subprocess.Popen([sys.executable, "-I", "-B", script, "local", str(node), str(index),
                                        self.tag], stdout=subprocess.PIPE, text=True)
                      for node, count in parse_flood(label).items() for index in range(count)]
        for proc in self.procs:
            if proc.stdout.readline().strip() != "ready":
                raise RuntimeError(f"cpu_flood.py pid {proc.pid} exited before ready")
        self.label = label

    def stop(self) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            proc.wait()
        self.procs = []
        self.label = "none"


def add_args(parser):
    B.add_live_args(parser)
    group = parser.add_argument_group(title="planner ablation")
    group.add_argument("--ablation-row", choices=("smoke", "a", "b", "c"), required=True)
    group.add_argument("--ablation-out", required=True)
    group.add_argument("--ablation-floods", default="none",
                       help="';'-separated flood configs: 'none' or 'node:count,...'")
    group.add_argument("--ablation-m", default="", help="comma list of allowed sender counts (row c)")
    group.add_argument("--ablation-repeats", type=int, default=3)
    group.add_argument("--ablation-tag", required=True)
    return parser


def setup(args) -> SimpleNamespace:
    world = dist.get_world_size()
    ep = args.expert_model_parallel_size
    src_model = get_training_model(
        lambda pre_process, post_process, **kwargs: B.model_provider(
            pre_process=pre_process, post_process=post_process, parallel_output=False),
        wrap_with_ddp=False,
    )[0].cuda().eval()
    dst_pg = build_inference_pg_collection(
        world, tp_size=args.live_dst_tp, pp_size=1, ep_size=ep, expt_tp_size=args.live_dst_tp,
        use_tp_pp_dp_mapping=args.use_tp_pp_dp_mapping)
    dst_config = core_transformer_config_from_args(args)
    dst_config.tensor_model_parallel_size = args.live_dst_tp
    dst_config.expert_tensor_parallel_size = args.live_dst_tp
    dst_config.expert_model_parallel_size = ep
    dst_model = get_training_model(
        lambda pre_process, post_process, **kwargs: B.model_provider(
            pre_process=pre_process, post_process=post_process, parallel_output=False,
            pg_collection=dst_pg, config=dst_config),
        wrap_with_ddp=False,
    )[0].cuda().eval()
    ranks = list(range(world))
    reshard_group = dist.new_group(ranks=ranks, backend="nccl")
    control_group = dist.new_group(ranks=ranks, backend="gloo")
    swap_model_weights(src_model, dst_model,
                       refit_method=NCCLCopyService(group=reshard_group, p2p_order=args.live_p2p_order),
                       group=reshard_group, release_cache=False)
    model_parallel_cuda_manual_seed(args.seed + 101)
    src_context, dst_context = (StaticInferenceContext(max_batch_size=args.micro_batch_size,
                                                       max_sequence_length=args.max_position_embeddings)
                                for _ in range(2))
    B._prefill(src_model, src_context, args)
    B._prefill(dst_model, dst_context, args)
    return SimpleNamespace(
        args=args, world=world, rank=dist.get_rank(), nodes=NODES, cpus=RANK_CPUS,
        src_model=src_model, dst_model=dst_model, src_context=src_context, dst_context=dst_context,
        src_bundle=B.LiveStateBundle(src_model, B.StaticKVCacheModule(
            src_context, src_model.pg_collection, request_id=None)),
        dst_bundle=B.LiveStateBundle(dst_model, B.StaticKVCacheModule(
            dst_context, dst_model.pg_collection, request_id=None)),
        reshard_group=reshard_group, control_group=control_group,
        service=NCCLCopyService(group=reshard_group, p2p_order=args.live_p2p_order,
                                ensure_all_ranks_participate=True),
        links=B._collect_active_collective_links(
            src_model, control_group, active_experts=args.live_active_experts_tuple,
            num_experts=args.num_experts),
        kv_base=args.live_prompt_tokens, cap=args.live_expansion_max_wave_tasks,
        plans={}, floods=Floods(args.ablation_tag), records=[])


def check_flood_cpus(ctx, labels: list[str]) -> None:
    """Flood processes pin cpus(node)[-1-idx]; they must not share a CPU with any rank."""
    ranks = {cpu for cpus in ctx.cpus.values() for cpu in cpus}
    for label in labels:
        for node, count in parse_flood(label).items():
            overlap = ranks & {node_cpus(node)[-1 - index] for index in range(count)}
            if overlap:
                raise ValueError(f"flood {label} overlaps rank CPUs {sorted(overlap)}")


def gather(ctx, value):
    out = [None] * ctx.world
    dist.all_gather_object(out, value, group=ctx.control_group)
    return out


def rewind(ctx) -> None:
    """Every migration moves the same KV prefix: both contexts restart after the prompt."""
    for context in (ctx.src_context, ctx.dst_context):
        context.sequence_len_offset = ctx.kv_base
        context.enable_decode_mode()


def measure_profile(ctx) -> dict:
    """Pairwise P2P matrix (64 MB, 2 iterations) averaged over three runs of the existing profiler."""
    probe = SimpleNamespace(profile_p2p_bytes=64 << 20, profile_p2p_iters=2, profile_p2p_passes=1,
                            profile_p2p_warmup_iters=1, profile_p2p_output=None,
                            scheduler_local_bandwidth_gbps=ctx.args.live_default_local_gbps)
    runs = [profile_p2p_bandwidth_matrix(probe) for _ in range(3)]
    matrix = {pair: statistics.fmean(run[pair] for run in runs) for pair in runs[0]}
    return dict(matrix=matrix, ratios=bandwidth_ratios(matrix, ctx.nodes))


def tpot_baseline(ctx) -> float:
    """Median of TPOT_STEPS decode steps without switching."""
    rewind(ctx)
    values = [B._all_max(REAL_DECODE(ctx.src_model, ctx.src_context, ctx.args, token_value=step)[1],
                         ctx.control_group) for step in range(TPOT_STEPS)]
    rewind(ctx)
    return statistics.median(values)


def level_kwargs(ctx, level: str, capacity: dict | None, allowed: list[int] | None) -> dict:
    """The only place where the four levels differ."""
    switches = {"L1": dict(prefer_local_source=False), "L2": {},
                "L3": dict(source_accumulate_load=False, source_global_gate=False), "L4": {}}[level]
    if level in ("L1", "L2"):
        return switches
    args = ctx.args
    kwargs = BandwidthAwareRefitPolicy(
        bandwidth_gbps=capacity, reference_bandwidth_gbps=capacity,
        reroute_penalty_us=args.live_reroute_penalty_us,
        reroute_min_gain_pct=args.live_reroute_min_gain_pct,
        reroute_min_contention_gain_pct=args.live_reroute_min_contention_gain_pct,
        reroute_min_global_gain_pct=args.live_reroute_min_global_gain_pct,
        reroute_min_bytes=args.live_reroute_min_bytes,
        prefer_local_source=not args.live_allow_nonlocal_reroute, p2p_order=args.live_p2p_order,
    ).planner_kwargs()
    # The greedy search config only makes the planner record predicted H.
    return dict(kwargs, **switches,
                source_search_config=dict(algorithm="greedy", budget_s=300.0, memory_gib=8.0, mip_rel_gap=0.0),
                source_allowed_ranks=frozenset(allowed) if allowed is not None else None)


def build_plan(ctx, level: str, capacity: dict | None, allowed: list[int] | None) -> SimpleNamespace:
    kwargs = level_kwargs(ctx, level, capacity, allowed)
    dist.barrier(group=ctx.control_group)
    start = time.perf_counter()
    plan = build_centralized_reshard_plan(ctx.src_bundle, ctx.dst_bundle, num_experts=ctx.args.num_experts,
                                          group=ctx.reshard_group, **kwargs)
    build_s = B._all_max(time.perf_counter() - start, ctx.control_group)
    base = restrict_plan_sequence(plan, start=0, end=ctx.kv_base, include_non_kv=True)
    params = dict(ctx.src_bundle.named_parameters())
    rows = []
    for op in base.send_ops:
        tensor = params[op.param_name]
        shard = tensor[:ctx.kv_base] if op.param_name.startswith("kv::") else tensor
        rows.append((ctx.rank, op.peer_rank, op.param_name, tensor[op.my_slice].numel() * tensor.element_size(),
                     op.task_id, shard.numel() * tensor.element_size()))
    rows = [row for part in gather(ctx, rows) for row in part]
    pads = pad_records(rows) if level == "L1" else []
    remote = sum(row[3] for row in rows if row[0] != row[1])
    sender = [0] * ctx.world  # network bytes per sender, padding included
    for src, dst, _name, size, _task, _shard in rows:
        sender[src] += size if src != dst else 0
    for _anchor, src, dst, size in pads:
        sender[src] += size if src != dst else 0
    pad_bytes = sum(size for *_, size in pads)
    entry = SimpleNamespace(
        level=level, allowed=allowed, plan=plan, base=base, build_s=build_s, tasks=len(rows), pads=pads,
        stats=plan.source_route_stats, task_bytes=sum(row[3] for row in rows), pad_bytes=pad_bytes,
        network_bytes=sum(sender), local_bytes=sum(row[3] for row in rows if row[0] == row[1])
        + sum(size for _anchor, src, dst, size in pads if src == dst),
        cross_socket_share=sum(row[3] for row in rows if ctx.nodes[row[0]] != ctx.nodes[row[1]]) / remote,
        sender_bytes=sender)
    if pads:
        # One read-only send buffer and one shared receive buffer; pad data is never read.
        device = torch.device("cuda", torch.cuda.current_device())
        entry.tx = torch.empty(max([size for _a, src, _d, size in pads if src == ctx.rank], default=0),
                               dtype=torch.uint8, device=device)
        entry.rx = torch.empty(max([size for _a, _s, dst, size in pads if dst == ctx.rank], default=0),
                               dtype=torch.uint8, device=device)
    B.print_rank_0(f"[plan] {level} allowed={allowed} build_s={build_s:.2f} tasks={entry.tasks} "
                   f"network={entry.network_bytes} local={entry.local_bytes} pad={pad_bytes}")
    return entry


def install_pads(ctx, entry, wave_pad_bytes: list[int]) -> None:
    """Send each L1 pad in the wave that carries its anchor task, inside the same NCCL batch."""
    mine: dict[int, list] = {}
    for index, (anchor, src, dst, size) in enumerate(entry.pads):
        if ctx.rank in (src, dst):
            mine.setdefault(anchor, []).append((-2 - index, src, dst, size))  # -1 is the marker

    def launch(plan, src_module, dst_module, service, **kwargs):
        sent = 0
        for anchor in sorted({op.task_id for op in plan.send_ops + plan.recv_ops} & mine.keys()):
            for pad_id, src, dst, size in mine[anchor]:
                if src == ctx.rank:
                    service.submit_send(entry.tx[:size], dst, task_id=pad_id)
                    sent += size
                if dst == ctx.rank:
                    service.submit_recv(entry.rx[:size], src, task_id=pad_id)
        wave_pad_bytes.append(sent)
        return REAL_LAUNCH(plan, src_module, dst_module, service, **kwargs)

    B.launch_reshard_plan = launch


def run_delta(ctx, entry) -> None:
    """KV appended during the overlap, chunked by the wave cap (B's RELEASE=0 delta path)."""
    delta_end = ctx.src_context.sequence_len_offset
    delta_plan = restrict_plan_sequence(entry.plan, start=ctx.kv_base, end=delta_end, include_non_kv=False)
    with torch.inference_mode():
        tasks = collect_transfer_tasks(delta_plan, ctx.dst_bundle, group=ctx.reshard_group)
        for first in range(0, len(tasks), ctx.cap):
            chunk = filter_plan_by_task_ids(delta_plan, [t.task_id for t in tasks[first:first + ctx.cap]])
            REAL_LAUNCH(chunk, ctx.src_bundle, ctx.dst_bundle, ctx.service, group=ctx.reshard_group,
                        synchronize_group=False, synchronize_device=False, release_cache=False).wait().commit()
        torch.cuda.current_stream().synchronize()
    ctx.dst_context.sequence_len_offset = delta_end
    ctx.dst_context.enable_decode_mode()


def migrate(ctx, entry, profile: dict, baseline_tpot_ms: float, validate: bool = False) -> dict:
    args = ctx.args
    rewind(ctx)
    tracker = ResidualBandwidthTracker(profile["matrix"], ewma_alpha=args.live_bandwidth_ewma,
                                       floor_fraction=args.live_bandwidth_floor_fraction)
    scheduler = ResidualBandwidthWaveScheduler(
        tracker, max_waves=args.live_max_waves, max_wave_bytes=args.live_max_wave_bytes,
        max_wave_tasks=ctx.cap, hotness_weight=args.live_hotness_weight)
    # Adaptive hybrid keeps only its wave rule; the plan is never replaced.
    mode = "residual" if math.ceil(entry.tasks / ctx.cap) <= args.live_adaptive_residual_max_waves else "fifo"
    stages, steps, wave_pad_bytes = [], [], []
    service = ctx.service

    def decode(model, context, decode_args, *, token_value):
        logits, ms = REAL_DECODE(model, context, decode_args, token_value=token_value)
        steps.append((time.perf_counter_ns(), ms))
        return logits, ms

    B._decode = decode
    if entry.pads:
        install_pads(ctx, entry, wave_pad_bytes)
    service.set_launch_callback(
        lambda: stages.append([pair for pairs in service._inflight._timing_stages.values() for pair in pairs]))
    try:
        torch.cuda.synchronize()
        dist.barrier(group=ctx.control_group)
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        t0 = time.perf_counter_ns()
        metrics, token = B._run_async_waves(
            plan=entry.base, src_bundle=ctx.src_bundle, dst_bundle=ctx.dst_bundle,
            active_model=ctx.src_model, active_context=ctx.src_context, service=service,
            tracker=tracker, scheduler=scheduler, reshard_group=ctx.reshard_group,
            control_group=ctx.control_group, args=args, token_counter=0,
            foreground_baseline_tpot_ms=baseline_tpot_ms, foreground_links=ctx.links,
            wave_scheduler_mode=mode, max_wave_tasks=ctx.cap)
        service.set_launch_callback(None)  # delta launches are not waves
        run_delta(ctx, entry)
        switch_wall_s = B._all_max((time.perf_counter_ns() - t0) / 1e9, ctx.control_group)
        decode(ctx.dst_model, ctx.dst_context, args, token_value=token)  # first token on the target
    finally:
        service.set_launch_callback(None)
        B._decode, B.launch_reshard_plan = REAL_DECODE, REAL_LAUNCH
    torch.cuda.synchronize()
    waves = [(min(origin.elapsed_time(s) for s, _ in pairs), max(origin.elapsed_time(e) for _, e in pairs))
             for pairs in stages]
    record = {}
    if validate:
        ctx.dst_context.sequence_len_offset -= 1  # recompute the target token at the source position
        record["validation_max_diff"] = B._validate_cutover(
            ctx.src_model, ctx.src_context, ctx.dst_model, ctx.dst_context, args, ctx.control_group, token)[0]
    parts = gather(ctx, dict(waves=waves, steps=steps, t0=t0, pad=wave_pad_bytes))
    wave_dev_ms = [max(p["waves"][k][1] - p["waves"][k][0] for p in parts) for k in range(len(waves))]
    # The last recorded step is the first token on the target, outside the window.
    tpot_ms = [max(p["steps"][i][1] for p in parts) for i in range(len(steps) - 1)]
    times = [max(p["t0"] for p in parts)] + [max(p["steps"][i][0] for p in parts) for i in range(len(steps))]
    pad_per_wave = [sum(per_rank) for per_rank in zip(*(p["pad"] for p in parts))] or [0] * len(waves)
    transfer_ms = sum(wave_dev_ms)
    stats = entry.stats
    projected = stats["projected_global_gain_pct"]
    decision = select_adaptive_hybrid_policy(
        direction="2->4", requested_scheduler_mode="residual",
        candidate_route_available=stats["accepted"] > 0, projected_transport_gain_pct=projected,
        baseline_task_count=ctx.plans[("L2",)].tasks, candidate_task_count=entry.tasks,
        max_wave_tasks=ctx.cap, min_end_to_end_gain_pct=args.live_adaptive_min_predicted_e2e_gain_pct,
        residual_max_waves=args.live_adaptive_residual_max_waves, foreground_pressure_fraction=0.0,
        max_foreground_pressure=args.live_hybrid_max_foreground_pressure)
    required = ctx.plans[("L2",)].task_bytes
    bandwidth_aware = entry.level in ("L3", "L4")
    record.update(
        level=entry.level, allowed_ranks=entry.allowed, execution_mode=mode, tasks=entry.tasks,
        waves=len(waves), plan_build_s=entry.build_s, transfer_ms=transfer_ms,
        transfer_span_ms=max(p["waves"][-1][1] for p in parts) - min(p["waves"][0][0] for p in parts),
        switch_wall_ms=switch_wall_s * 1e3, goodput_gbps=required * 8 / (transfer_ms * 1e6),
        wave_dev_ms=wave_dev_ms, wave_wall_ms=[w["elapsed_s"] * 1e3 for w in metrics["wave_records"]],
        wave_pad_bytes=pad_per_wave,
        wave_bytes=[w["bytes"] + pad for w, pad in zip(metrics["wave_records"], pad_per_wave)],
        tpot_ms=tpot_ms, token_interval_ms=[(b - a) / 1e6 for a, b in zip(times, times[1:])],
        required_bytes=required, network_bytes=entry.network_bytes, local_bytes=entry.local_bytes,
        pad_bytes=entry.pad_bytes, sender_bytes=entry.sender_bytes, sender_bytes_max=max(entry.sender_bytes),
        sender_bytes_mean=statistics.fmean(entry.sender_bytes),
        predicted_H_s=stats["source_search"]["predicted_H_s"] if bandwidth_aware else None,
        projected_global_gain_pct=projected, global_gate_accepted=stats["global_gate_accepted"],
        global_gate_would_reject=projected < args.live_reroute_min_global_gain_pct if bandwidth_aware else None,
        adaptive_would_fallback=decision.plan_variant == "baseline", adaptive_reason=decision.reason)
    return record


def plan_for(ctx, level: str, flood: str, profile: dict, allowed: list[int] | None):
    """Plan once per (level x point); L1/L2 do not depend on the profile or on m."""
    key = (level,) if level in ("L1", "L2") else (level, flood, tuple(allowed or ()))
    if key not in ctx.plans:
        ctx.plans[key] = build_plan(ctx, level, profile["matrix"], allowed)
    return ctx.plans[key]


def emit(ctx, record: dict) -> None:
    if ctx.rank == 0:
        ctx.records.append(record)
        with open(Path(ctx.args.ablation_out) / "result.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def set_flood(ctx, label: str) -> None:
    if ctx.rank == 0:
        ctx.floods.set(label)
    dist.barrier(group=ctx.control_group)


def run_row(ctx, levels: list[str]) -> dict:
    args = ctx.args
    floods = args.ablation_floods.split(";")
    if args.ablation_row == "c":
        points = [dict(flood=floods[0], m=int(m)) for m in args.ablation_m.split(",")]
    else:
        points = [dict(flood=flood, m=None) for flood in floods]
    profiles, baselines = {}, {}
    ctx.plans[("L2",)] = build_plan(ctx, "L2", None, None)
    for index, point in enumerate(points):
        started = time.perf_counter()
        flood, m = point["flood"], point["m"]
        set_flood(ctx, flood)
        if flood not in profiles:
            profiles[flood], baselines[flood] = measure_profile(ctx), tpot_baseline(ctx)
        allowed = allowed_ranks(ctx.nodes, m) if m is not None else None
        # Row c runs L1/L2 only at m = all ranks; their plans do not depend on m.
        point_levels = [level for level in levels if m in (None, ctx.world) or level in ("L3", "L4")]
        for round_index in range(1 + args.ablation_repeats):
            order = list(point_levels)
            random.Random(f"{SEED}:{args.ablation_row}:{index}:{round_index}").shuffle(order)
            for position, level in enumerate(order):
                entry = plan_for(ctx, level, flood, profiles[flood], allowed)
                record = migrate(ctx, entry, profiles[flood], baselines[flood])
                record.update(row=args.ablation_row, point=index, n_layers=args.num_layers, flood=flood, m=m,
                              round=round_index, warmup=round_index == 0, order=position,
                              bandwidth=profiles[flood]["ratios"], baseline_tpot_ms=baselines[flood],
                              unstable=False)
                emit(ctx, record)
        B.print_rank_0(f"[point] row={args.ablation_row} {index} flood={flood} m={m} n={args.num_layers} "
                       f"elapsed_s={time.perf_counter() - started:.1f}")
    last = points[-1]["flood"]
    recheck = measure_profile(ctx)["ratios"]
    unstable = abs(recheck["ratio_mean"] / profiles[last]["ratios"]["ratio_mean"] - 1.0) > 0.15
    if ctx.rank == 0 and unstable:
        for record in ctx.records:
            record["unstable"] = record["flood"] == last
        (Path(args.ablation_out) / "result.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in ctx.records), encoding="utf-8")
    return dict(
        profiles={label: dict(ratios=p["ratios"], matrix_gbps=[
            [p["matrix"][(s, d)] for d in range(ctx.world)] for s in range(ctx.world)])
            for label, p in profiles.items()},
        baseline_tpot_ms=baselines, recheck=dict(flood=last, ratios=recheck, unstable=unstable))


def memory_report(ctx, title: str) -> list:
    rows = gather(ctx, (torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30))
    B.print_rank_0(f"[memory] {title} (allocated/reserved GiB): " + " ".join(
        f"gpu{rank}={alloc:.2f}/{reserved:.2f}" for rank, (alloc, reserved) in enumerate(rows)))
    return rows


def run_smoke(ctx, levels: list[str]) -> dict:
    """27 layers: every level switches and validates once; with flood, L2 validates again and
    L3/L2 and L4/L3 (m=1) transfer ratios are printed."""
    args = ctx.args
    flood = args.ablation_floods
    report = dict(memory_after_build=memory_report(ctx, "after build"))
    idle, idle_tpot = measure_profile(ctx), tpot_baseline(ctx)
    ctx.plans[("L2",)] = build_plan(ctx, "L2", None, None)
    results = {}
    for level in levels:
        entry = plan_for(ctx, level, "none", idle, None)
        if level == "L1":
            torch.cuda.reset_peak_memory_stats()
        results[level] = record = migrate(ctx, entry, idle, idle_tpot, validate=True)
        if level == "L1":
            report["memory_l1_migration"] = memory_report(ctx, "during L1 migration")
        record.update(row="smoke", flood="none", m=None, bandwidth=idle["ratios"], baseline_tpot_ms=idle_tpot)
        emit(ctx, record)
        B.print_rank_0(f"[smoke] {level} validation_max_diff={record['validation_max_diff']:.4g} "
                       f"plan_build_s={entry.build_s:.2f} transfer_ms={record['transfer_ms']:.1f} "
                       f"switch_wall_ms={record['switch_wall_ms']:.1f} network={record['network_bytes']} "
                       f"local={record['local_bytes']} pad={record['pad_bytes']}")
    set_flood(ctx, flood)
    flooded, flooded_tpot = measure_profile(ctx), tpot_baseline(ctx)
    one = allowed_ranks(ctx.nodes, 1)
    on = {}
    for name, level, allowed in (("L2", "L2", None), ("L3", "L3", None), ("L3_m1", "L3", one),
                                 ("L4_m1", "L4", one)):
        on[name] = record = migrate(ctx, plan_for(ctx, level, flood, flooded, allowed), flooded,
                                    flooded_tpot, validate=name == "L2")
        record.update(row="smoke", flood=flood, m=1 if allowed else None, bandwidth=flooded["ratios"],
                      baseline_tpot_ms=flooded_tpot)
        emit(ctx, record)
    set_flood(ctx, "none")
    report["memory_end"] = memory_report(ctx, "end")
    B.print_rank_0(
        f"[smoke] validation passed for {list(results)} and for L2 under flood {flood}\n"
        f"[smoke] decode step ms: idle={idle_tpot:.2f} flood={flooded_tpot:.2f}; "
        f"bandwidth idle={idle['ratios']} flood={flooded['ratios']}\n"
        f"[smoke] L2 cross-socket share of network bytes={ctx.plans[('L2',)].cross_socket_share:.3f}\n"
        f"[smoke] flood {flood}: transfer L3/L2={on['L3']['transfer_ms'] / on['L2']['transfer_ms']:.3f}, "
        f"m=1 L4/L3={on['L4_m1']['transfer_ms'] / on['L3_m1']['transfer_ms']:.3f}")
    return report


def main() -> None:
    launch_start = time.perf_counter()
    levels = os.environ["WEAVETP_ABLATION_LEVEL"].split(",")  # unset: the ablation is off
    ctx = None
    try:
        parse_and_validate_args(
            extra_args_provider=add_args,
            args_defaults={"tokenizer_type": "NullTokenizer", "no_load_optim": True, "no_load_rng": True,
                           "no_save_optim": True, "no_save_rng": True, "transformer_impl": "local"},
            ignore_unknown_args=False)
        args = get_args()
        args.apply_rope_fusion = False
        args.add_bias_linear = False
        args.add_qkv_bias = False
        args.bias_gelu_fusion = False
        args.gradient_accumulation_fusion = False
        args.sequence_parallel = False
        initialize_megatron()
        args = get_args()
        args.live_active_experts_tuple = B._parse_experts(
            args.live_active_experts, args.num_experts, args.moe_router_topk)
        args.live_pressure_ranks_tuple = ()
        Path(args.ablation_out).mkdir(parents=True, exist_ok=True)
        ctx = setup(args)
        check_flood_cpus(ctx, args.ablation_floods.split(";"))
        B.print_rank_0(f"[launch] row={args.ablation_row} layers={args.num_layers} levels={levels} "
                       f"nodes={ctx.nodes} rank_cpus={ctx.cpus} setup_s={time.perf_counter() - launch_start:.1f}")
        summary = run_smoke(ctx, levels) if args.ablation_row == "smoke" else run_row(ctx, levels)
        elapsed = time.perf_counter() - launch_start
        if ctx.rank == 0:
            summary.update(row=args.ablation_row, num_layers=args.num_layers, levels=levels, nodes=ctx.nodes,
                           rank_cpus=ctx.cpus, launch_s=elapsed,
                           l2_cross_socket_share=ctx.plans[("L2",)].cross_socket_share)
            (Path(args.ablation_out) / "launch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        B.print_rank_0(f"[launch] row={args.ablation_row} layers={args.num_layers} elapsed_s={elapsed:.1f}")
    finally:
        if ctx is not None:
            ctx.floods.stop()
        if sys.exc_info()[0] is None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
