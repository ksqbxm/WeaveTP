"""Tiny real eight-GPU TP2->TP4/EP2 source-search execution, not a benchmark.

Run with torchrun. Each case uses a fresh NCCL group and the production planner,
peer-size-desc transport and launch/wait/commit. No checkpoint or env.sh changes.
"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.planner import build_centralized_reshard_plan

from .harness import Bundle, param
from .reference import Layout

CASES = ("dfs", "dp", "dijkstra", "heap", "ls", "ilp_gap0", "ilp_gap1e-4")


def rank_groups(tp):
    return dict(
        tp=[list(range(start, start + tp)) for start in range(0, 8, tp)],
        ep=[[base + lane, base + lane + tp] for base in range(0, 8, 2 * tp)
            for lane in range(tp)],
        dp=[list(range(lane, 8, tp)) for lane in range(tp)],
    )


def fixture(rank, device, topology):
    bundles, expected = {}, {}
    for tp in (2, 4):
        group_ranks = next(ranks for ranks in rank_groups(tp)["tp"] if rank in ranks)
        expert = (rank // tp) % 2
        # Exactly representable FP32/BF16 values; distinct global experts.
        full = (torch.arange(64).reshape(4, 16) % 31 + 40 * expert).to(torch.bfloat16)
        weight = Layout(tuple(group_ranks), 1, 2).shard(full, rank)
        truth = {"experts.weight0": weight, "norm": torch.arange(4, dtype=torch.bfloat16)}
        entries = {}
        for name, correct in truth.items():
            # Non-contiguous views exercise real source preparation and writeback.
            shape = list(correct.shape)
            shape[-1] *= 2
            backing = torch.full(shape, float("nan"), dtype=correct.dtype, device=device)
            value = backing[..., ::2]
            if tp == 2:
                value.copy_(correct.to(device))
            parameter = param(value)
            parameter.tensor_model_parallel = name.startswith("experts.")
            parameter.partition_dim, parameter.partition_stride = 1, 2
            parameter.allreduce = not name.startswith("experts.")
            entries[name] = parameter
        bundles[tp] = Bundle(entries)
        bundles[tp].pg_collection = topology[tp]
        if tp == 4:
            expected = truth
    return bundles[2], bundles[4], expected


def planner_kwargs(case):
    # Slow baseline cross-TP links force a meaningful source change, with the
    # normal 5% Global Gate still enabled. This is NOT a measured profile.
    slow = {(1, 2), (1, 3), (2, 4), (2, 5)}
    return dict(
        source_bandwidth_gbps={(s, d): .001 if (s, d) in slow else 50.
                               for s in range(8) for d in range(8)},
        source_latency_us=0., source_reroute_penalty_us=0., source_reroute_min_bytes=16,
        source_reroute_min_global_gain_pct=5., prefer_local_source=False,
        source_search_config=dict(algorithm="ilp" if case.startswith("ilp_") else case,
                                  budget_s=10., memory_gib=2.,
                                  mip_rel_gap=1e-4 if case == "ilp_gap1e-4" else 0.),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/data")):
        parser.error("--output must be under /data")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    if dist.get_world_size() != 8:
        raise ValueError("This EP2 fixture requires eight ranks/devices")
    topology = {}
    for tp in (2, 4):
        chosen = {}
        for kind, groups in rank_groups(tp).items():
            for ranks in groups:
                group = dist.new_group(ranks, backend="gloo", timeout=timedelta(seconds=90))
                if rank in ranks:
                    chosen[kind] = group
        topology[tp] = SimpleNamespace(**chosen, expt_tp=chosen["tp"], pp=None)
    report = dict(rank=rank, torch=torch.__version__, nccl=torch.cuda.nccl.version(),
                  scope="tiny synthetic TP2->TP4/EP2; not full-model performance", cases=[])
    for case in args.cases:
        src, dst, expected = fixture(rank, "cuda", topology)
        data_group = dist.new_group(list(range(8)), backend="nccl", timeout=timedelta(seconds=60))
        plan = build_centralized_reshard_plan(src, dst, num_experts=2, **planner_kwargs(case))
        stats = plan.source_route_stats
        assert stats["global_gate_accepted"] and stats["accepted"] > 0, stats
        assert stats["source_search"]["status"] == "complete", stats
        # No post-plan warmup collective hides a first-batch participation bug.
        service = NCCLCopyService(group=data_group, p2p_order="peer-size-desc",
                                  ensure_all_ranks_participate=True)
        before = {name: tensor.detach().clone() for name, tensor in src.entries.items()}
        ready = torch.cuda.Event()
        ready.record()
        ready.synchronize()  # Fixture availability only; production prepares the send slices.
        txn = launch_reshard_plan(plan, src, dst, service)
        txn.wait().commit()
        for name, correct in expected.items():
            torch.testing.assert_close(dst.entries[name].detach().cpu(), correct, rtol=0, atol=0)
            torch.testing.assert_close(src.entries[name], before[name], rtol=0, atol=0)
        row = dict(case=case, result="PASS", routes=stats,
                   transport=service.last_launch_stats)
        report["cases"].append(row)
        (output / f"rank_{rank}.json").write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
        dist.destroy_process_group(data_group)
        if rank == 0:
            print(f"PASS {case}: forced plan executed; exact BF16 payload and source preserved", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
