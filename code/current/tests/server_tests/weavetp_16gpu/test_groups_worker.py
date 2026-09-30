"""T06/T08 communication check: torchrun on 2 x 8 GPUs, no model/checkpoint.

Use only after the T08 approvals and both-host idle/network checks. This tests
the existing group builder; it does not certify the benchmark's future T06 hooks.
"""

import argparse
import os
import socket
from datetime import timedelta
from pathlib import Path
from uuid import UUID

from common import ROOT, caches, require, write


def verify_groups(rows):
    require(len(rows) == 16 and {r["rank"] for r in rows} == set(range(16)), "need all 16 real ranks")
    require(len({r["gpu_uuid"] for r in rows}) == 16, "workers share a GPU UUID")
    by_rank = {r["rank"]: r for r in rows}
    for row in rows:
        rank = row["rank"]
        require(row["hostname"].split(".")[0].lower() == ("sl3060" if rank < 8 else "sl3061"), "wrong node placement")
        require(row["local_rank"] == rank % 8, "wrong CUDA local rank")
        for tp in (2, 4):
            groups = row["groups"][str(tp)]
            for name, size in {"tp": tp, "expt_tp": tp, "ep": 2, "dp": 16 // tp,
                               "expt_dp": 8 // tp, "pp": 1, "cp": 1}.items():
                members = groups[name]
                require(len(members) == size and len(set(members)) == size and rank in members,
                        f"rank={rank} TP={tp}: actual {name} membership/size mismatch")
                for peer in members:
                    require(by_rank[peer]["groups"][str(tp)][name] == members, "inconsistent group membership")
                if name in ("tp", "expt_tp", "ep"):
                    require(len({by_rank[p]["hostname"] for p in members}) == 1, f"{name} crosses nodes")


def main(out):
    import sys

    require(out.resolve().is_relative_to("/data") and out.is_dir(), "precreate a fresh OUT_DIR under /data")
    os.environ.update(caches(out))
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.distributed as dist
    from megatron.rl.parallel_utils import build_inference_pg_collection

    local_rank = int(os.environ["LOCAL_RANK"])
    require(int(os.environ["WORLD_SIZE"]) == 16 and int(os.environ["LOCAL_WORLD_SIZE"]) == 8, "only 2 x 8 workers allowed")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        token = torch.ones(1, device=device)
        dist.all_reduce(token)
        rank = dist.get_rank()
        local = {"rank": rank, "local_rank": local_rank, "hostname": socket.gethostname(),
                 "gpu_uuid": "GPU-" + str(UUID(bytes=bytes(torch.cuda.get_device_properties(device).uuid.bytes))),
                 "groups": {}}
        for tp in (2, 4):
            collection = build_inference_pg_collection(16, tp_size=tp, pp_size=1, cp_size=1,
                                                       ep_size=2, expt_tp_size=tp)
            local["groups"][str(tp)] = {}
            for name in ("tp", "expt_tp", "ep", "dp", "expt_dp", "pp", "cp"):
                group = getattr(collection, name)
                members = dist.get_process_group_ranks(group)
                require(dist.get_world_size(group) == len(members), "actual ProcessGroup size differs")
                local["groups"][str(tp)][name] = members
                value = torch.tensor([rank], device=device, dtype=torch.int64)
                dist.all_reduce(value, group=group)
                require(value.item() == sum(members), f"{name} collective did not reach actual members")
        rows = [None] * 16
        dist.all_gather_object(rows, local)
        verify_groups(rows)
        if rank == 0:
            write(out / "groups.json", {"status": "passed", "ranks": rows})
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    main(parser.parse_args().out_dir)
