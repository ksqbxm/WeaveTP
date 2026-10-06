"""Two-GPU ordering, chunk reuse and foreground-completion probes.

Ordering and overlap have separate workloads and checks. All numerical/timing
failures are agreed over Gloo before another NCCL batch may be submitted.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist

from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.utils import ReshardPlan, TransferOp
from tools.resharding.correctness.harness import Bundle, param
from tools.resharding.standby_weights import StandbyWeights, release_standby


def completions_during_transfer(transfer_ms, foreground_ms):
    """Count actual foreground completions strictly inside a transport interval.

    A foreground step may begin before transport. Merely intersecting it, or
    completing after it, does not establish forward progress while pending.
    """
    return sum(any(max(start, fg_start) < fg_end < end for start, end in transfer_ms)
               for fg_start, fg_end in foreground_ms)


def complete_case(record, records, output):
    """Called after all GPU work completes, including on a local check failure."""
    local = {"rank": dist.get_rank(), "failed_checks": [k for k, ok in record["checks"].items() if not ok]}
    outcomes = [None] * dist.get_world_size()
    dist.all_gather_object(outcomes, local)
    failures = [row for row in outcomes if row["failed_checks"]]
    record["failures"] = failures
    records.append(record)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"ordering_rank{dist.get_rank()}.json").write_text(json.dumps({
        "passed": not failures,
        "foreground": "matrix computation; real MoE decode is tested by the benchmark",
        "cases": records,
    }, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise AssertionError(f"coordinated ordering acceptance failure: {failures}")


def make_plan(rank, start, end):
    index = (slice(None), slice(start, end))
    sends, recvs = [], []
    for tid, name in enumerate(("local", "remote", "strided")):
        peer = rank if name == "local" else 1 - rank
        sends.append(TransferOp(name, peer, True, index, index, rank * 3 + tid))
        recvs.append(TransferOp(name, peer, False, index, index, peer * 3 + tid))
    return ReshardPlan(sends, recvs)


def run_probe(group, producer, packed, scope, records, output):
    rank = dist.get_rank()
    width = 16 if scope == "ordering" else 4096  # 64 MiB per overlap tensor.
    names = ("local", "remote", "strided")
    source, target = {}, {}
    columns = torch.arange(width, device="cuda", dtype=torch.float32) / width
    for index, name in enumerate(names):
        shape = (width, width * (2 if name == "strided" else 1))
        source[name], target[name] = (param(torch.empty(shape, device="cuda")[:, ::2]
                                         if name == "strided" else torch.empty(shape, device="cuda"))
                                     for _ in range(2))
        source[name].data.copy_(columns + rank * 4 + index + 1)
        target[name].data.zero_()
    source, target = Bundle(source), Bundle(target)
    weights = StandbyWeights(target, protected_tensors=tuple(source.parameters()))
    item_bytes = width * width * 4
    service = NCCLCopyService(group=group, p2p_order="task-round",
                              pack_target_bytes=2 * item_bytes if packed else 0,
                              pack_max_item_bytes=item_bytes if packed else 0,
                              persistent_pack_buffers=packed)
    service.producer_stream = producer
    full = make_plan(rank, 0, width)
    plans = [make_plan(rank, start, start + width // 2) for start in (0, width // 2)] \
        if scope == "ordering" else [full]
    foreground = torch.randn(1024, 1024, device="cuda")
    foreground_output = torch.empty_like(foreground)
    torch.mm(foreground, foreground, out=foreground_output)
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        launch_reshard_plan(full, source, target, service).wait().commit()
    torch.cuda.synchronize()  # Warmup only, outside the measured cycles.

    reuse_observed = False
    for cycle in range(3):
        release_standby(weights, [service])
        weights.allocate_and_poison(producer)
        producer.synchronize()  # Allocation/first poison are outside the proof window.
        origin = torch.cuda.Event(enable_timing=True)
        poisoned = torch.cuda.Event(enable_timing=True)
        foreground_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                             for _ in range(64)] if scope == "overlap" else []
        dist.barrier()  # Foreground is local matmul: no post-launch NCCL rendezvous.
        origin.record()
        producer.wait_event(origin)
        stage_events, recv_addresses = {}, []
        with torch.cuda.stream(producer):
            if scope == "ordering":
                # Re-poison after a delay to exercise the producer dependencies.
                # The overlap test never relies on this delay remaining pending.
                torch.cuda._sleep(100_000_000)
                for parameter in weights.parameters:
                    parameter.fill_(float("nan"))
            poisoned.record()
        for plan in plans:
            with torch.cuda.stream(producer):
                transaction = launch_reshard_plan(plan, source, target, service)
            # Save existing device events before wait() clears the handle. This
            # probe deliberately inspects internals; no production tracing API.
            for name, pairs in transaction._copy_handle._timing_stages.items():
                stage_events.setdefault(name, []).extend(pairs)
            recv_addresses.append({row[1].data_ptr() for row in transaction._recv_writebacks if row[0] == "default"})
            for start, end in foreground_events:
                start.record()
                torch.mm(foreground, foreground, out=foreground_output)
                end.record()
            with torch.cuda.stream(producer):
                transaction.wait().commit()
        producer.synchronize()
        torch.cuda.current_stream().synchronize()
        stages = {name: [(origin.elapsed_time(start), origin.elapsed_time(end)) for start, end in pairs]
                  for name, pairs in stage_events.items()}
        foreground_ms = [(origin.elapsed_time(start), origin.elapsed_time(end)) for start, end in foreground_events]
        poison_ms = origin.elapsed_time(poisoned)
        checks = {}
        for index, (name, parameter) in enumerate(target.entries.items()):
            peer = rank if name == "local" else 1 - rank
            checks[f"exact_{name}"] = bool(torch.eq(parameter, columns + peer * 4 + index + 1).all().item())
        checks["finite"] = bool(weights.finite_flag().item())
        checks["poison_before_copy"] = all(start >= poison_ms for name in ("local", "nccl")
                                            for start, _ in stages[name])
        completions = completions_during_transfer(stages["nccl"], foreground_ms)
        reused = len(recv_addresses) > 1 and bool(recv_addresses[0] & recv_addresses[1])
        reuse_observed |= reused
        if scope == "overlap":
            checks["foreground_completion_during_nccl"] = completions > 0
        elif cycle == 2:
            checks["receive_storage_reuse_exercised"] = reuse_observed
        complete_case({"scope": scope, "packed": packed, "cycle": cycle, "checks": checks,
                       "poison_ms": poison_ms, "stages_ms": stages, "foreground_ms": foreground_ms,
                       "foreground_completions_during_nccl": completions,
                       "receive_storage_reused": reused},
                      records, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    if dist.get_world_size() != 2:
        raise ValueError("run this probe with exactly two GPU workers")
    data_group = dist.new_group(backend="nccl")
    producer = torch.cuda.Stream()
    records = []
    try:
        with torch.inference_mode():
            for packed in (False, True):
                for scope in ("ordering", "overlap"):
                    run_probe(data_group, producer, packed, scope, records, args.output)
    finally:
        dist.destroy_process_group(data_group)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
