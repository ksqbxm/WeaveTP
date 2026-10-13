"""Real CUDA/NCCL + Host acceptance test, also runnable with multi-node torchrun.

Single host: torchrun --standalone --nproc-per-node=3 <this file>
For cross-node coverage use torchrun rendezvous arguments and at least 3 ranks.
CPU tests do not replace this test; no throughput benchmark is performed here.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from megatron.core.resharding import anchortp
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.utils import ReshardPlan, TransferOp
from tools.resharding.standby_weights import StandbyWeights


@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="requires three real GPUs and NCCL")
def test_real_host_p2p_preallocation_and_released_storage():
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=3", str(Path(__file__).resolve())],
        check=True, timeout=180,
    )


def run():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    control = dist.new_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world >= 3  # rank 2 deliberately has no model transfer tasks.
    hosts, links = anchortp.topology(control)
    rates = {(i, j): (1000 if i == j else 10) for i in range(world) for j in range(world)}

    def model(value):
        module = torch.nn.Module()
        module.register_parameter("w", torch.nn.Parameter(
            torch.full((64, 8), value, dtype=torch.float32, device="cuda"), requires_grad=False,
        ))
        return module

    for release in (False, True):
        for host_reload in (False, True):
            source, target = model(10.0 + rank), model(float("nan"))
            storage = StandbyWeights(target, protected_tensors=tuple(source.parameters()))
            service = NCCLCopyService(p2p_order="task-round", ensure_all_ranks_participate=True)
            preparation = torch.cuda.Stream()
            service.producer_stream = preparation
            transfers = [(0, 0, 0, 2), (1, 0, 2, 8), (0, 1, 0, 2), (0, 1, 2, 6)]
            if not host_reload:
                transfers.append((0, 1, 6, 8))
            sends, recvs = [], []
            for tid, (src, dst, lo, hi) in enumerate(transfers):
                s = (slice(None), slice(lo, hi))
                if rank == src:
                    sends.append(TransferOp("w", dst, True, s, s, tid))
                if rank == dst:
                    recvs.append(TransferOp("w", src, False, s, s, tid))
            plan = ReshardPlan(sends, recvs)
            host_tasks = [anchortp.HostTask("w", (slice(None), slice(6, 8)), torch.full((64, 2), 99.0))] if rank == 1 and host_reload else []
            for _ in range(2):  # New receive resources after every storage rebuild.
                if release:
                    storage.release()
                start = time.perf_counter()
                if release:
                    storage.allocate_and_poison(preparation)
                else:
                    target.w.data.fill_(float("nan"))
                first_launch = False
                real_launch, real_empty = service.launch, torch.empty

                def launch():
                    nonlocal first_launch
                    first_launch = True
                    return real_launch()

                def allocate(*args, **kwargs):
                    assert not first_launch, "receive allocation after first transport batch"
                    return real_empty(*args, **kwargs)

                with patch.object(service, "launch", launch), patch("torch.empty", allocate):
                    result = anchortp.execute(
                        plan, source, target, service, bandwidth=rates, hosts=hosts, links=links,
                        max_tasks=1, control_group=control, host_tasks=host_tasks, prealloc_started=start,
                    )
                assert result["reload_bytes"] == (64 * 2 * 4 if host_reload else 0)
                assert result["paths"]["local_copy"]["tasks"] == 1
                assert result["overlap_steps"] == 0 and result["prealloc_s"] <= result["wall_s"]
                if rank == 0:
                    assert (target.w[:, :2] == 10).all() and (target.w[:, 2:] == 11).all()
                if rank == 1:
                    assert (target.w[:, :6] == 10).all()
                    assert (target.w[:, 6:] == (99 if host_reload else 10)).all()
                dist.barrier(group=control)
    # Host-only upload is independent of the ordinary fault-free P2P switch.
    target = model(float("nan"))
    host_tasks = [anchortp.HostTask("w", (slice(None), slice(None)), torch.full((64, 8), 123.0))] if rank == 1 else []
    result = anchortp.execute(
        ReshardPlan([], []), None, target,
        NCCLCopyService(p2p_order="task-round", ensure_all_ranks_participate=True),
        bandwidth=rates, hosts=hosts, control_group=control, host_tasks=host_tasks,
    )
    assert result["waves"] == 0 and result["reload_bytes"] == 64 * 8 * 4
    if rank == 1:
        assert (target.w == 123).all()
    dist.destroy_process_group()


if __name__ == "__main__":
    run()
