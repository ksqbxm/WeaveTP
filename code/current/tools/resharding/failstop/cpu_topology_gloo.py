"""Four real CPU processes: group order, collectives, pointers, and noncontiguous P2P."""

import json
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path


def worker(rank, directory):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    import torch.distributed as dist

    from megatron.core.failstop import EpochResources, build_replica_pg_collections
    from megatron.core.failstop_rebind import ModelBinding
    from megatron.core.resharding.copy_services.nccl_copy_service import (
        NCCLCopyService,
        RecvOp,
        SendOp,
    )

    torch.set_num_threads(1)
    store = dist.FileStore(str(directory / 'store'), 4)
    dist.init_process_group('gloo', store=store, rank=rank, world_size=4, timeout=timedelta(seconds=30))
    replicas = [((3, 1), 1, 1, 2), ((2, 0), 1, 1, 2)]
    logs = []
    old = build_replica_pg_collections(replicas, backend='gloo', records=logs)[rank % 2 == 0]
    for name, group in vars(old).items():
        if name == 'local_ranks':
            continue
        members = dist.get_process_group_ranks(group)
        value = torch.tensor(rank + 1)
        dist.all_reduce(value, group=group)
        assert value.item() == sum(r + 1 for r in members), name
    model = torch.nn.Linear(2, 2, bias=False)
    model.pg_collection, model.tp_group = old, old.tp
    resource = EpochResources()
    binding = ModelBinding(model, old, resource, {'tp': 1})
    pointer, value = model.weight.data_ptr(), model.weight.detach().clone()
    new = build_replica_pg_collections(replicas, backend='gloo')[rank % 2 == 0]
    binding.rebind(new)
    resource.activate(('new',))
    model(torch.ones(2))
    assert model.weight.data_ptr() == pointer and torch.equal(model.weight, value)
    group = new.tp_ep
    service = object.__new__(NCCLCopyService)
    service.group, service.group_relative_ranks = group, True
    service.rank, service.world_size = group.rank(), group.size()
    peer = 1 - service.rank
    send, recv = torch.tensor([rank]), torch.empty(1, dtype=torch.int64)
    ops = service._build_task_round_p2p_ops([SendOp(-1, send, peer)], [RecvOp(-1, recv, peer)])
    for work in dist.batch_isend_irecv(ops):
        work.wait()
    assert recv.item() == dist.get_global_rank(group, peer)
    # Only EP0 has a reversed requested TP order. EP must pair the actual
    # TP coordinates after new_group sorting, not the requested positions.
    ordered = build_replica_pg_collections([((1, 0, 2, 3), 2, 2, 2)], backend='gloo')[0]
    value = torch.tensor(ordered.local_ranks['tp'])
    dist.all_reduce(value, group=ordered.ep)
    assert value.item() == 2 * ordered.local_ranks['tp'], 'EP paired different actual TP coordinates'
    (directory / f'rank{rank}.json').write_text(json.dumps({'status': 'passed', 'groups': logs}))
    dist.destroy_process_group()


def main():
    if len(sys.argv) == 3:
        worker(int(sys.argv[1]), Path(sys.argv[2]))
        return
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        children, logs = [], []
        try:
            for rank in range(4):
                log = (root / f'child{rank}.log').open('w')
                logs.append(log)
                children.append(subprocess.Popen([sys.executable, '-B', __file__, str(rank), directory],
                    stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'OMP_NUM_THREADS': '1'}))
            for child in children:
                assert child.wait(timeout=90) == 0, 'CPU PG worker failed'
            for rank in range(4):
                assert json.loads((root / f'rank{rank}.json').read_text())['status'] == 'passed'
            print('CPU GLOO PASS: 4/4 processes; ordered groups, TP/EP coordinate pairing, collective sums, pointer rebind, subgroup P2P')
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
            for log in logs:
                log.close()
            for path in root.glob('child*.log'):
                print(path.read_text(errors='replace'))


if __name__ == '__main__':
    main()
