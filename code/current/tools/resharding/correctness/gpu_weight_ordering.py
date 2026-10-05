"""Two-GPU NCCL ordering probe with delayed poison and independent foreground work."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 2, "run this probe with exactly two GPU workers"
    data_group = dist.new_group(backend="nccl")
    producer = torch.cuda.Stream()
    records = []
    try:
        for packed in (False, True):
            source = Bundle({name: param(torch.full((16, 32), rank + index + 1., device="cuda")[:, ::2]
                                        if name == "strided" else torch.full((16, 16), rank + index + 1., device="cuda"))
                             for index, name in enumerate(("local", "remote", "strided"))})
            target = Bundle({name: param(torch.zeros(16, 32, device="cuda")[:, ::2]
                                        if name == "strided" else torch.zeros(16, 16, device="cuda"))
                             for name in source.entries})
            weights = StandbyWeights(target, protected_tensors=tuple(source.parameters()))
            service = NCCLCopyService(group=data_group, p2p_order="task-round", pack_target_bytes=4096 if packed else 0,
                                      pack_max_item_bytes=2048 if packed else 0, persistent_pack_buffers=packed)
            service.producer_stream = producer
            full = (slice(None), slice(None))
            sends, recvs = [], []
            for index, name in enumerate(source.entries):
                peer_out, peer_in = ((rank, rank) if index == 0 else ((rank + 1) % size, (rank - 1) % size))
                sends.append(TransferOp(name, peer_out, True, full, full, rank * 3 + index))
                recvs.append(TransferOp(name, peer_in, False, full, full, peer_in * 3 + index))
            plan = ReshardPlan(sends, recvs)
            foreground = torch.randn(256, 256, device="cuda")
            output = foreground @ foreground
            # Warm the actual P2P communicator and foreground kernels, outside
            # the proof. Otherwise lazy NCCL/cuBLAS setup can consume the delay.
            producer.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(producer):
                warmup = launch_reshard_plan(plan, source, target, service)
            warmup.wait().commit()
            torch.cuda.synchronize()
            for cycle in range(3):
                release_standby(weights, [service])
                with torch.cuda.stream(producer):
                    torch.cuda._sleep(500_000_000)
                weights.allocate_and_poison(producer)
                with torch.cuda.stream(producer):
                    transaction = launch_reshard_plan(plan, source, target, service)
                dist.barrier()
                progress = 0
                for _ in range(8):
                    output = foreground @ foreground
                    step = torch.cuda.Event()
                    step.record()
                    step.synchronize()  # Foreground stream only, not the transfer stream.
                    progress += int(not transaction.done())
                transaction.wait().commit()
                torch.cuda.current_stream().synchronize()
                for index, (name, parameter) in enumerate(target.entries.items()):
                    peer = rank if index == 0 else (rank - 1) % size
                    assert torch.equal(parameter, torch.full_like(parameter, peer + index + 1.))
                assert weights.finite_flag().item()
                assert progress > 0, "no foreground completion observed while migration was pending"
                records.append({"packed": packed, "cycle": cycle, "foreground_steps_while_pending": progress})
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"ordering_rank{rank}.json").open("x", encoding="utf-8") as stream:
            json.dump({"passed": True, "scope": "real NCCL, local/remote copies, delayed poison, strided writeback, packing",
                       "foreground": "matrix computation; real MoE decode is tested by the benchmark", "cases": records}, stream, indent=2)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
