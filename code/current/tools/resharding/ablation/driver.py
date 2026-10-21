#!/usr/bin/env python3
"""WeaveTP planner ablation (TP2->TP4, EP=2), one torchrun per launch.

The four levels differ only in planning (level_kwargs). The live benchmark module
is imported unchanged; this process patches its module globals (``_decode`` and
``launch_reshard_plan``) around each migration. --decode-mode pause keeps the
forward pass out of the migration window; overlap decodes continuously.
Local rank r drives the r-th GPU in PCI order (CUDA_DEVICE_ORDER=PCI_BUS_ID).
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
    """NUMA node of each local GPU, and each local rank's slice of the first CPUs of that node."""
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
import random
import statistics
import subprocess
import time
from collections import Counter
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
from tools.resharding.ablation.planstats import (  # noqa: E402
    bandwidth_ratios,
    pad_records,
    plan_diff_bytes,
    plan_stats,
)
from tools.resharding.tp_build_src_dst_demo import profile_p2p_bandwidth_matrix  # noqa: E402

ROUNDS = 3
TPOT_STEPS = 10
SEED = 2026
REAL_DECODE = B._decode
REAL_LAUNCH = B.launch_reshard_plan


def parse_flood(spec: str) -> dict[int, int]:
    """'none' or 'node:count,node:count'."""
    if spec == "none":
        return {}
    return {int(node): int(count) for node, count in (item.split(":") for item in spec.split(","))}


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
    group.add_argument("--decode-mode", choices=("pause", "overlap"), required=True,
                       help="pause: no forward inside the migration window; overlap: continuous decode")
    group.add_argument("--ablation-row", choices=("smoke", "a", "b", "xnode"), required=True)
    group.add_argument("--ablation-out", required=True)
    group.add_argument("--ablation-floods", default="none",
                       help="';'-separated flood configs: 'none' or 'node:count,...'")
    group.add_argument("--ablation-tag", required=True)
    group.add_argument("--ablation-metadata", help="smoke only: planner metadata dump for offline replanning")
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
    # A domain is the machine for cross-machine runs and the GPU's NUMA node otherwise.
    places = [None] * world
    dist.all_gather_object(places, (int(os.environ["GROUP_RANK"]), NODES[int(os.environ["LOCAL_RANK"])]),
                           group=control_group)
    machines = len({machine for machine, _node in places})
    return SimpleNamespace(
        args=args, world=world, rank=dist.get_rank(), cpus=RANK_CPUS,
        domains=[machine if machines > 1 else node for machine, node in places],
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
    """Flood processes pin cpus(node)[-1-idx]; they must not share a CPU with any local rank."""
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
    """Pairwise P2P matrix (64 MB, 2 iterations) averaged over two runs of the existing profiler."""
    probe = SimpleNamespace(profile_p2p_bytes=64 << 20, profile_p2p_iters=2, profile_p2p_passes=1,
                            profile_p2p_warmup_iters=1, profile_p2p_output=None,
                            scheduler_local_bandwidth_gbps=ctx.args.live_default_local_gbps)
    runs = [profile_p2p_bandwidth_matrix(probe) for _ in range(2)]
    matrix = {pair: statistics.fmean(run[pair] for run in runs) for pair in runs[0]}
    return dict(matrix=matrix, ratios=bandwidth_ratios(matrix, ctx.domains))


def tpot_baseline(ctx) -> float:
    """Median of TPOT_STEPS decode steps without switching."""
    rewind(ctx)
    values = [B._all_max(REAL_DECODE(ctx.src_model, ctx.src_context, ctx.args, token_value=step)[1],
                         ctx.control_group) for step in range(TPOT_STEPS)]
    rewind(ctx)
    return statistics.median(values)


def level_kwargs(ctx, level: str, capacity: dict | None) -> dict:
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
    # The greedy search config only makes the planner record predicted H and the gate outcome.
    return dict(kwargs, **switches,
                source_search_config=dict(algorithm="greedy", budget_s=300.0, memory_gib=8.0, mip_rel_gap=0.0))


def build_plan(ctx, level: str, flood: str | None, capacity: dict | None) -> SimpleNamespace:
    dist.barrier(group=ctx.control_group)
    start = time.perf_counter()
    plan = build_centralized_reshard_plan(ctx.src_bundle, ctx.dst_bundle, num_experts=ctx.args.num_experts,
                                          group=ctx.reshard_group, **level_kwargs(ctx, level, capacity))
    build_s = B._all_max(time.perf_counter() - start, ctx.control_group)
    base = restrict_plan_sequence(plan, start=0, end=ctx.kv_base, include_non_kv=True)
    src_params = dict(ctx.src_bundle.named_parameters())
    dst_params = dict(ctx.dst_bundle.named_parameters())
    op_bytes, send_rows, rows = {}, [], []
    for op in base.send_ops:
        tensor = src_params[op.param_name]
        op_bytes[op.task_id] = tensor[op.my_slice].numel() * tensor.element_size()
        shard = tensor[:ctx.kv_base] if op.param_name.startswith("kv::") else tensor
        send_rows.append((ctx.rank, op.peer_rank, op.param_name, op_bytes[op.task_id], op.task_id,
                          shard.numel() * tensor.element_size()))
    for op in base.recv_ops:
        tensor = dst_params[op.param_name]
        op_bytes[op.task_id] = tensor[op.my_slice].numel() * tensor.element_size()
        key = (op.param_name, tuple((s.start, s.stop, s.step) for s in op.my_slice))
        rows.append((op.peer_rank, ctx.rank, key, op_bytes[op.task_id]))
    rows = [row for part in gather(ctx, rows) for row in part]
    pads = pad_records([row for part in gather(ctx, send_rows) for row in part]) if level == "L1" else []
    route = plan.source_route_stats
    search = route.get("source_search", {})
    entry = SimpleNamespace(
        level=level, plan=plan, base=base, build_s=build_s, rows=rows, pads=pads, op_bytes=op_bytes,
        tasks=len(rows), gate_outcome=search.get("gate_outcome"), predicted_H_s=search.get("predicted_H_s"),
        projected_global_gain_pct=route["projected_global_gain_pct"],
        stats=plan_stats(rows, pads, ctx.domains, ctx.plans[("L2",)].rows if level != "L2" else rows))
    if pads:
        # One read-only send buffer and one shared receive buffer; pad data is never read.
        device = torch.device("cuda", torch.cuda.current_device())
        entry.tx = torch.empty(max([size for _a, src, _d, size in pads if src == ctx.rank], default=0),
                               dtype=torch.uint8, device=device)
        entry.rx = torch.empty(max([size for _a, _s, dst, size in pads if dst == ctx.rank], default=0),
                               dtype=torch.uint8, device=device)
    if ctx.rank == 0:
        with open(Path(ctx.args.ablation_out) / "plans.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(
                row=ctx.args.ablation_row, n_layers=ctx.args.num_layers, flood=flood, level=level,
                build_s=build_s, tasks=entry.tasks, gate_outcome=entry.gate_outcome,
                predicted_H_s=entry.predicted_H_s, projected_global_gain_pct=entry.projected_global_gain_pct,
                **entry.stats)) + "\n")
    B.print_rank_0(f"[plan] {level} flood={flood} build_s={build_s:.2f} tasks={entry.tasks} "
                   f"network={entry.stats['network_bytes']} local={entry.stats['local_bytes']} "
                   f"pad={entry.stats['pad_bytes']} gate={entry.gate_outcome}")
    return entry


def plan_for(ctx, level: str, flood: str, profile: dict) -> SimpleNamespace:
    """Plan once per (level x point); L1/L2 do not depend on the profile."""
    if level in ("L1", "L2"):
        key, flood = (level,), None
    else:
        key = (level, flood)
    if key not in ctx.plans:
        ctx.plans[key] = build_plan(ctx, level, flood, profile["matrix"])
    return ctx.plans[key]


def dump_metadata(ctx, path: str) -> None:
    """Untimed planning call whose only purpose is the planner metadata dump (rank 0 writes it)."""
    uniform = {(s, d): 1.0 for s in range(ctx.world) for d in range(ctx.world)}
    build_centralized_reshard_plan(ctx.src_bundle, ctx.dst_bundle, num_experts=ctx.args.num_experts,
                                   group=ctx.reshard_group, source_search_audit_path=path,
                                   **level_kwargs(ctx, "L4", uniform))


def install_launch(ctx, entry, waves: list[dict]) -> None:
    """Count each wave's bytes per rank; for L1 also send each pad in the wave of its anchor task."""
    mine: dict[int, list] = {}
    for index, (anchor, src, dst, size) in enumerate(entry.pads):
        if ctx.rank in (src, dst):
            mine.setdefault(anchor, []).append((-2 - index, src, dst, size))  # -1 is the marker

    def launch(plan, src_module, dst_module, service, **kwargs):
        wave = dict(send_bytes=0, recv_bytes=0, local_bytes=0, pad_bytes=0)
        for op in plan.send_ops:
            wave["local_bytes" if op.peer_rank == ctx.rank else "send_bytes"] += entry.op_bytes[op.task_id]
        for op in plan.recv_ops:
            if op.peer_rank != ctx.rank:
                wave["recv_bytes"] += entry.op_bytes[op.task_id]
        task_ids = {op.task_id for op in plan.send_ops + plan.recv_ops}
        for anchor in sorted(task_ids & mine.keys()):
            for pad_id, src, dst, size in mine[anchor]:
                if src == ctx.rank:
                    service.submit_send(entry.tx[:size], dst, task_id=pad_id)
                    wave["pad_bytes"] += size
                    wave["local_bytes" if dst == ctx.rank else "send_bytes"] += size
                if dst == ctx.rank:
                    service.submit_recv(entry.rx[:size], src, task_id=pad_id)
                    if src != ctx.rank:
                        wave["recv_bytes"] += size
        waves.append(dict(wave, tasks=len(task_ids)))
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


def migrate(ctx, entry, profile: dict, baseline_tpot_ms: float, validate: bool) -> dict:
    args = ctx.args
    pause = args.decode_mode == "pause"
    rewind(ctx)
    tracker = ResidualBandwidthTracker(profile["matrix"], ewma_alpha=args.live_bandwidth_ewma,
                                       floor_fraction=args.live_bandwidth_floor_fraction)
    scheduler = ResidualBandwidthWaveScheduler(
        tracker, max_waves=args.live_max_waves, max_wave_bytes=args.live_max_wave_bytes,
        max_wave_tasks=ctx.cap, hotness_weight=args.live_hotness_weight)
    stages, waves, steps = [], [], []
    service = ctx.service

    def decode(model, context, decode_args, *, token_value):
        if pause:
            return None, 0.0
        logits, ms = REAL_DECODE(model, context, decode_args, token_value=token_value)
        steps.append((time.perf_counter_ns(), ms))
        return logits, ms

    B._decode = decode
    install_launch(ctx, entry, waves)
    service.set_launch_callback(lambda: stages.append(
        {name: list(pairs) for name, pairs in service._inflight._timing_stages.items()}))
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
            wave_scheduler_mode="fifo", max_wave_tasks=ctx.cap)
        service.set_launch_callback(None)  # delta launches are not waves
        run_delta(ctx, entry)
        switch_wall_ms = B._all_max((time.perf_counter_ns() - t0) / 1e6, ctx.control_group)
        if validate or not pause:
            # First token on the target, after the window.
            steps.append((time.perf_counter_ns(),
                          REAL_DECODE(ctx.dst_model, ctx.dst_context, args, token_value=token)[1]))
    finally:
        service.set_launch_callback(None)
        B._decode, B.launch_reshard_plan = REAL_DECODE, REAL_LAUNCH
    torch.cuda.synchronize()

    def interval(pairs):
        if not pairs:
            return None
        return [min(origin.elapsed_time(s) for s, _ in pairs), max(origin.elapsed_time(e) for _, e in pairs)]

    for wave, stage in zip(waves, stages):
        wave.update(nccl=interval(stage.get("nccl")), local=interval(stage.get("local")))
    record = dict(validated=validate)
    if validate:
        ctx.dst_context.sequence_len_offset -= 1  # recompute the target token at the source position
        record["validation_max_diff"] = B._validate_cutover(
            ctx.src_model, ctx.src_context, ctx.dst_model, ctx.dst_context, args, ctx.control_group, token)[0]
    parts = gather(ctx, dict(waves=waves, steps=steps, t0=t0))
    by_wave = [[part["waves"][k] for part in parts] for k in range(len(waves))]

    def extent(wave):
        intervals = [x for x in (wave["nccl"], wave["local"]) if x]
        return min(x[0] for x in intervals), max(x[1] for x in intervals)

    def length(x):
        return x[1] - x[0] if x else 0.0

    wave_dev = [max(length(extent(w)) for w in ranks) for ranks in by_wave]
    wave_net = [max(length(w["nccl"]) for w in ranks) for ranks in by_wave]
    wave_local = [max(length(w["local"]) for w in ranks) for ranks in by_wave]
    span_ms = max(extent(w)[1] for w in by_wave[-1]) - min(extent(w)[0] for w in by_wave[0])
    t_data = sum(wave_dev)
    commit_ms = sum(w["commit_s"] for w in metrics["wave_records"]) * 1e3
    l2 = ctx.plans[("L2",)].stats
    record.update(
        level=entry.level, tasks=entry.tasks, waves=len(waves), plan_build_s=entry.build_s,
        required_bytes=l2["network_bytes"] + l2["local_bytes"],
        **{key: value for key, value in entry.stats.items() if key != "link_bytes"},
        predicted_H_s=entry.predicted_H_s, gate_outcome=entry.gate_outcome,
        projected_global_gain_pct=entry.projected_global_gain_pct,
        T_data_ms=t_data, T_data_net_ms=sum(wave_net), T_data_local_ms=sum(wave_local),
        span_ms=span_ms, gap_ms=span_ms - t_data, switch_wall_ms=switch_wall_ms, commit_ms=commit_ms,
        other_ms=switch_wall_ms - span_ms - commit_ms,
        wave_dev_ms=wave_dev, wave_net_ms=wave_net, wave_local_ms=wave_local,
        wave_wall_ms=[w["elapsed_s"] * 1e3 for w in metrics["wave_records"]],
        wave_bottleneck_rank=[max(range(ctx.world), key=lambda r: extent(ranks[r])[1]) for ranks in by_wave],
        # Per wave x rank: nccl/local [start, end] ms from the origin, bytes and task count.
        wave_rank=by_wave)
    if not pause:
        times = [max(p["t0"] for p in parts)] + [max(p["steps"][i][0] for p in parts) for i in range(len(steps))]
        record.update(tpot_ms=[max(p["steps"][i][1] for p in parts) for i in range(len(steps) - 1)],
                      token_interval_ms=[(b - a) / 1e6 for a, b in zip(times, times[1:])],
                      baseline_tpot_ms=baseline_tpot_ms)
    return record


def check_point(ctx, entries: dict, flood: str) -> dict:
    """Bandwidth-aware levels must differ from L2 on flood or cross-machine points."""
    l3, l4 = entries["L3"], entries["L4"]
    reroute = l3.stats["reroute_sender_bytes"] + l3.stats["reroute_local_to_net_bytes"]
    check = dict(l3_reroute_bytes=reroute, l4_vs_l3_diff_bytes=plan_diff_bytes(l4.rows, l3.rows),
                 l4_gate_outcome=l4.gate_outcome)
    check["inactive"] = ((flood != "none" or ctx.args.ablation_row == "xnode")
                         and (reroute == 0 or l4.gate_outcome != "accepted"))
    B.print_rank_0(f"[check] flood={flood} {check}" + (" WARNING: inactive point" if check["inactive"] else ""))
    return check


def emit(ctx, record: dict) -> None:
    if ctx.rank == 0:
        ctx.records.append(record)
        with open(Path(ctx.args.ablation_out) / "result.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def set_flood(ctx, label: str) -> None:
    if ctx.rank == 0:
        ctx.floods.set(label)
    dist.barrier(group=ctx.control_group)


def memory_report(ctx, title: str) -> None:
    rows = gather(ctx, (torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30))
    B.print_rank_0(f"[memory] {title} (allocated/reserved GiB): " + " ".join(
        f"gpu{rank}={alloc:.2f}/{reserved:.2f}" for rank, (alloc, reserved) in enumerate(rows)))


def smoke_report(records: list[dict], levels: list[str]) -> None:
    medians = {}
    for level in levels:
        runs = [r for r in records if r["level"] == level and not r["warmup"]]
        t_data = [r["T_data_ms"] for r in runs]
        medians[level] = statistics.median(t_data)
        wave_cv = max(statistics.pstdev(values) / statistics.fmean(values)
                      for values in zip(*(r["wave_dev_ms"] for r in runs)))
        bottlenecks = Counter(rank for r in runs for rank in r["wave_bottleneck_rank"])
        print(f"[smoke] {level} T_data_median_ms={medians[level]:.1f} "
              f"T_data_cv={statistics.pstdev(t_data) / statistics.fmean(t_data):.3f} max_wave_cv={wave_cv:.3f} "
              f"reroute_bytes={runs[0]['reroute_sender_bytes'] + runs[0]['reroute_local_to_net_bytes']} "
              f"sender_max={runs[0]['sender_bytes_max']} sender_mean={runs[0]['sender_bytes_mean']:.0f} "
              f"bottlenecks={dict(sorted(bottlenecks.items()))}", flush=True)
    print(f"[smoke] T_data median ratio L3/L2={medians['L3'] / medians['L2']:.3f} "
          f"L4/L3={medians['L4'] / medians['L3']:.3f}", flush=True)


def run_launch(ctx) -> dict:
    args = ctx.args
    floods = args.ablation_floods.split(";")
    profiles, baselines, validated = {}, {}, set()
    ctx.plans[("L2",)] = build_plan(ctx, "L2", None, None)
    if args.ablation_metadata:
        dump_metadata(ctx, args.ablation_metadata)
    for index, flood in enumerate(floods):
        started = time.perf_counter()
        set_flood(ctx, flood)
        if flood not in profiles:
            profiles[flood] = measure_profile(ctx)
            baselines[flood] = tpot_baseline(ctx) if args.decode_mode == "overlap" else 0.0
        entries = {level: plan_for(ctx, level, flood, profiles[flood]) for level in ctx.levels}
        check = check_point(ctx, entries, flood) if {"L3", "L4"} <= entries.keys() else dict(inactive=False)
        # Warm up once per level, on the launch's first point only.
        for round_index, warmup in enumerate([True] * (index == 0) + [False] * ROUNDS):
            order = list(ctx.levels)
            random.Random(f"{SEED}:{args.ablation_row}:{index}:{round_index}").shuffle(order)
            for position, level in enumerate(order):
                validate = not warmup and level not in validated
                if validate:
                    validated.add(level)
                record = migrate(ctx, entries[level], profiles[flood], baselines[flood], validate)
                record.update(mode=args.decode_mode, row=args.ablation_row, point=index,
                              n_layers=args.num_layers, flood=flood, round=round_index, warmup=warmup,
                              order=position, bandwidth=profiles[flood]["ratios"], unstable=False, **check)
                emit(ctx, record)
        B.print_rank_0(f"[point] row={args.ablation_row} {index} flood={flood} n={args.num_layers} "
                       f"elapsed_s={time.perf_counter() - started:.1f}")
    recheck = measure_profile(ctx)["ratios"]
    unstable = abs(recheck["ratio_mean"] / profiles[floods[-1]]["ratios"]["ratio_mean"] - 1.0) > 0.15
    if ctx.rank == 0 and unstable:
        for record in ctx.records:
            record["unstable"] = record["flood"] == floods[-1]
        (Path(args.ablation_out) / "result.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in ctx.records), encoding="utf-8")
    return dict(
        profiles={label: dict(ratios=p["ratios"], matrix_gbps=[
            [p["matrix"][(s, d)] for d in range(ctx.world)] for s in range(ctx.world)])
            for label, p in profiles.items()},
        baseline_tpot_ms=baselines, recheck=dict(flood=floods[-1], ratios=recheck, unstable=unstable))


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
        if args.decode_mode == "pause":
            args.live_max_overlap_steps = 1  # one no-op decode per wave, then wait for the transfer
        Path(args.ablation_out).mkdir(parents=True, exist_ok=True)
        ctx = setup(args)
        ctx.levels = levels
        check_flood_cpus(ctx, args.ablation_floods.split(";"))
        B.print_rank_0(f"[launch] mode={args.decode_mode} row={args.ablation_row} layers={args.num_layers} "
                       f"levels={levels} domains={ctx.domains} local_rank_cpus={ctx.cpus} "
                       f"setup_s={time.perf_counter() - launch_start:.1f}")
        if args.ablation_row == "smoke":
            memory_report(ctx, "after build")
        summary = run_launch(ctx)
        if args.ablation_row == "smoke":
            memory_report(ctx, "end")
            if ctx.rank == 0:
                smoke_report(ctx.records, levels)
        elapsed = time.perf_counter() - launch_start
        if ctx.rank == 0:
            summary.update(mode=args.decode_mode, row=args.ablation_row, num_layers=args.num_layers,
                           levels=levels, domains=ctx.domains, launch_s=elapsed)
            (Path(args.ablation_out) / "launch.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        B.print_rank_0(f"[launch] mode={args.decode_mode} row={args.ablation_row} layers={args.num_layers} "
                       f"elapsed_s={elapsed:.1f}")
    finally:
        if ctx is not None:
            ctx.floods.stop()
        if sys.exc_info()[0] is None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
