"""MegatronRuntime.load_local, the serving workers' checkpoint load, on Megatron's real dist_checkpointing: this rank's shards
of its own layout, read with no collective. The 16-rank serving world holds two independent model replicas whose
expert shards both carry replica_id 0 (experts take the DP part from their replica's expt_dp group), so
load_checkpoint's global sharding validation counted each of them twice on the servers; load_local validates nothing
globally and calls no collective (no tracker all-reduce, no barrier), which the counters below check."""

import contextlib
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from model_runtime import MegatronRuntime
from torch.distributed import distributed_c10d as c10d

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from megatron.core import dist_checkpointing, parallel_state  # noqa: E402
from megatron.core.dist_checkpointing import ShardedTensor  # noqa: E402
from megatron.training import checkpointing  # noqa: E402

FULL = torch.arange(24, dtype=torch.float32).reshape(4, 6)
COLLECTIVES = ('all_reduce', 'all_gather', 'all_gather_into_tensor', 'all_gather_object', 'all_to_all', 'all_to_all_single',
               'barrier', 'monitored_barrier', 'broadcast', 'broadcast_object_list', 'gather', 'gather_object', 'reduce',
               'reduce_scatter', 'reduce_scatter_tensor', 'scatter', 'scatter_object_list')


class Shard(torch.nn.Module):
    """A model holding rows [coord * 4 / tp, ...) of FULL, as a TP layout's coordinate would."""

    def __init__(self, tp, coord):
        super().__init__()
        self.tp, self.coord = tp, coord
        self.w = torch.nn.Parameter(torch.full((4 // tp, 6), float('nan')))

    def sharded_state_dict(self, metadata=None):
        assert metadata is not None and 'dp_cp_group' in metadata  # as Megatron's modules require
        return {'w': ShardedTensor.from_rank_offsets('w', self.w.data, (0, self.coord, self.tp))}


@contextlib.contextmanager
def counted(calls):
    """Every collective of torch.distributed (both namespaces) appends its name to `calls`."""
    with contextlib.ExitStack() as stack:
        for name in COLLECTIVES:
            original = getattr(c10d, name)

            def wrapped(*args, _name=name, _original=original, **kwargs):
                calls.append(_name)
                return _original(*args, **kwargs)
            for module in (dist, c10d):
                stack.enter_context(patch.object(module, name, wrapped))
        yield


class LoadLocalTests(unittest.TestCase):
    def test_load_local_reads_this_coordinate_without_a_collective(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'latest_checkpointed_iteration.txt').write_text('release')
            (root / 'release').mkdir()
            dist.init_process_group('gloo', store=dist.FileStore(str(root / 'store'), 1), rank=0, world_size=1,
                                    timeout=timedelta(seconds=30))
            try:
                full, saved = Shard(1, 0), []
                full.w.data.copy_(FULL)
                # the save is CUDA-bound (its writer syncs the copies, its finalize broadcasts a failure flag on the device)
                with counted(saved), patch.multiple(torch.cuda, synchronize=lambda *_: None, current_device=lambda: 'cpu'):
                    dist_checkpointing.save({'model': full.sharded_state_dict({'dp_cp_group': None})}, str(root / 'release'))
                self.assertIn('barrier', saved)  # the counters see Megatron's collectives (the save gathers, broadcasts, barriers)
                model, loaded = Shard(2, 1), []
                with counted(loaded), patch.object(parallel_state, 'get_data_parallel_group', lambda **_: None), \
                        patch.object(checkpointing, 'load_checkpoint', side_effect=AssertionError('load_checkpoint called')):
                    MegatronRuntime(SimpleNamespace(load=str(root)), None).load_local(model)
            finally:
                dist.destroy_process_group()
            self.assertEqual(loaded, [])
            self.assertTrue(torch.equal(model.w.data, FULL[2:]))


if __name__ == '__main__':
    unittest.main()
