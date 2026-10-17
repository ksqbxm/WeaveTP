"""Step 2 CPU contracts, actual tensor pointers, and unchanged homogeneous entry."""

import importlib.util
import sys
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import common
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from megatron.core.failstop import (
    EpochResources,
    FaultBackend,
    FaultSession,
    build_replica_pg_collections,
    choose_local_order,
    make_copy_service,
    overlap_bytes,
    replica_group_specs,
)
from megatron.core.failstop_rebind import ModelBinding
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService


def rows():
    return common.topology({node: [{'index': i, 'uuid': f'{node}-{i}'} for i in range(8)]
                            for node in common.NODES})


def session(mode='emulate', **kwargs):
    return FaultSession(rows(), 'cpu_step2', FaultBackend(mode, fail=Mock(), rejoin=Mock()), **kwargs)


class Group:
    def __init__(self, ranks, rank):
        self.ranks, self.global_rank = tuple(sorted(ranks)), rank

    def size(self):
        return len(self.ranks)

    def rank(self):
        return self.ranks.index(self.global_rank)


def group_patches(rank, size):
    return patch.multiple(dist,
        is_initialized=lambda: True, get_rank=lambda group=None: rank if group is None else group.rank(),
        get_world_size=lambda group=None: size if group is None else group.size(),
        get_process_group_ranks=lambda group: list(group.ranks), get_backend=lambda *_: 'fake',
        new_subgroups_by_enumeration=lambda groups, **_: (
            next(Group(g, rank) for g in groups if rank in g), [Group(g, rank) for g in groups]),
        new_group=Mock(side_effect=lambda ranks, **_: Group(ranks, rank)))


class MembershipTests(unittest.TestCase):
    def test_modes_share_device_membership_and_source_roles(self):
        for mode in ('emulate', 'real'):
            s = session(mode)
            original = s.rank_map()
            s.fail(s.failed)
            self.assertEqual(len(s.members), 12)
            self.assertEqual(s.state, 'failed/recovering')
            for slot in s.failed:
                self.assertEqual(s.modules(slot, object(), object()), (None, None))
                self.assertIsNone(s.rank_map()[slot]['new_rank'])
            self.assertEqual([r['new_rank'] for r in s.rank_map() if r['active']],
                             list(range(12)) if mode == 'real' else [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15])
            s.complete()
            s.rejoin(s.failed)
            for slot in s.failed:
                target = object()
                self.assertEqual(s.modules(slot, object(), target), (None, target))
            s.complete()
            self.assertEqual([r['new_rank'] for r in s.rank_map()], list(range(16)))
            self.assertEqual([r['source_etp'] for r in s.rank_map()], [r['etp'] for r in original])

    def test_old_events_wrong_rounds_and_rank_lists_are_rejected(self):
        s = session()
        event = s.event('fail')
        for invalid in (replace(event, epoch=0), replace(event, round=1), replace(event, launch='other'), replace(event, id='replay')):
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                s.fail(s.failed, invalid)
        with self.assertRaisesRegex(RuntimeError, 'coordinates'):
            s.fail([0, 1, 4, 5])
        s.fail(s.failed, event)
        with self.assertRaises(RuntimeError):
            s.fail(s.failed, event)
        with self.assertRaises(RuntimeError):
            s.rejoin(s.failed)
        s.complete()
        s.rejoin(s.failed)
        s.complete()
        s.next_round()
        with self.assertRaises(RuntimeError):
            s.fail(s.failed, event)

    def test_failure_callback_cannot_leave_apparently_normal_world(self):
        s = session()
        s.backend.fail.side_effect = RuntimeError('kill failed')
        with self.assertRaisesRegex(RuntimeError, 'kill failed'):
            s.fail(s.failed)
        self.assertFalse(s.resources.ready)
        self.assertEqual(s.state, 'failed/recovering')

    def test_epoch_caches_and_pending_work_are_disposed_before_new_world(self):
        s = session('real')
        old = s.cache_key()
        work = Mock()
        s.resources.plans[old] = 'plan'
        s.resources.groups[old] = 'group'
        s.resources.pending.append(work)
        aborted = Mock()
        s.resources.invalidate(aborted)
        aborted.assert_called_once()
        work.wait.assert_not_called()
        s.fail(s.failed)
        s.complete()
        self.assertNotEqual(old, s.cache_key())
        self.assertFalse(s.resources.plans or s.resources.groups or s.resources.pending)
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            with s.resources.forward(old):
                pass

    def test_drain_failure_stops_before_cache_reuse(self):
        r = EpochResources()
        r.pending.append(Mock(wait=Mock(return_value=False)))
        with self.assertRaisesRegex(RuntimeError, 'work failed'):
            r.invalidate()
        self.assertFalse(r.ready)
        self.assertEqual(len(r.pending), 1)

    def test_reuse_optimizer_counts_actual_byte_spans_and_ties(self):
        source = [('expert17.w', 0, 32), ('dense', 0, 8)]
        self.assertEqual(overlap_bytes(source, [('expert18.w', 0, 32), ('dense', 4, 16)]), 4)
        self.assertEqual(choose_local_order(['b', 'a'], {'a': [8, 32], 'b': [24, 0]}), ('b', 'a'))
        self.assertEqual(choose_local_order(['b', 'a'], {'a': [8, 8], 'b': [8, 8]}), ('a', 'b'))

    def test_default_mode_and_invalid_scenario(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(FaultBackend(fail=Mock(), rejoin=Mock()).mode, 'emulate')
        invalid = rows()
        invalid[0]['node'] = 'SL3061'
        with self.assertRaisesRegex(RuntimeError, 'scenario'):
            FaultSession(invalid, 'bad', FaultBackend(fail=Mock(), rejoin=Mock()))


class GroupTests(unittest.TestCase):
    def test_every_rank_creates_same_groups_including_nonmembers(self):
        replicas = [((3, 1), 1, 1, 2), ((0, 4), 1, 1, 2)]
        sequences = []
        for rank in range(5):
            records = []
            with group_patches(rank, 5):
                pgs = build_replica_pg_collections(replicas, records=records)
                sequences.append(dist.new_group.call_args_list)
                if rank == 2:
                    self.assertEqual(pgs, [None, None])
                if rank == 1:
                    self.assertEqual(pgs[0].local_ranks['ep'], 0)
                    self.assertTrue(any(r['requested'] == [3, 1] and r['actual'] == [1, 3] for r in records))
        self.assertTrue(all(sequence == sequences[0] for sequence in sequences))

    def test_invalid_replica_specs(self):
        for replicas in ([((0, 1, 2), 2, 2, 2)], [((0, 1), 1, 1, 2), ((1, 2), 1, 1, 2)]):
            with self.assertRaises(RuntimeError):
                replica_group_specs(replicas)

    def test_actual_sorting_updates_local_map_without_relabeling_sources(self):
        s = session(local_orders={0: (11, 10), 1: (15, 14)})
        s.fail(s.failed)
        before = s.cache_key()
        s.resources.plans[before] = 'old'
        records = [{'replica': replica, 'field': field, 'actual': sorted(ranks)}
                   for replica, field, ranks in replica_group_specs(s.replicas())]
        s.adopt_group_order(records)
        self.assertEqual(s.rank_map()[10]['etp'], 0)
        self.assertEqual(s.rank_map()[10]['source_etp'], 2)
        self.assertNotEqual(before, s.cache_key())
        self.assertFalse(s.resources.plans)

    def test_failed_gpu_cannot_join_marker_ring(self):
        s = session()
        s.fail(s.failed)
        with group_patches(0, 16):
            with self.assertRaisesRegex(RuntimeError, 'failed GPU'):
                make_copy_service(s, s.failed[0], Group(range(16), 0))
            with self.assertRaisesRegex(RuntimeError, 'absent GPUs'):
                make_copy_service(s, 0, Group(range(16), 0))

    def test_copy_peer_rank_mode_is_opt_in(self):
        service = object.__new__(NCCLCopyService)
        group, tensor = object(), object()
        with patch.object(dist, 'P2POp') as p2p:
            service.group_relative_ranks = False
            service._p2p_op(dist.isend, tensor, 2, group)
            p2p.assert_called_with(dist.isend, tensor, 2, group=group)
            service.group_relative_ranks = True
            service._p2p_op(dist.isend, tensor, 2, group)
            p2p.assert_called_with(dist.isend, tensor, group=group, group_peer=2)

    def test_homogeneous_entry_unchanged(self):
        # Load the actual module without optional RL/pydantic package startup.
        path = ROOT / 'megatron/rl/parallel_utils.py'
        spec = importlib.util.spec_from_file_location('homogeneous_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for rank in (0, 3, 8, 15):
            with group_patches(rank, 16):
                pg = module.build_inference_pg_collection(16, tp_size=4, pp_size=1, cp_size=1, ep_size=2, expt_tp_size=4)
                self.assertEqual(pg.tp.ranks, tuple(range(rank // 4 * 4, rank // 4 * 4 + 4)))
                self.assertEqual(pg.ep.ranks, (rank // 8 * 8 + rank % 4, rank // 8 * 8 + rank % 4 + 4))
                self.assertEqual(pg.dp.ranks, tuple(range(rank % 4, 16, 4)))


class BindingTests(unittest.TestCase):
    def fixture(self):
        model = torch.nn.Linear(3, 3, bias=False)
        model.register_buffer('kept', torch.arange(3))
        resources = EpochResources()
        resources.activate(('epoch', 0))
        with group_patches(0, 4):
            pg = build_replica_pg_collections([((0, 1, 2, 3), 2, 2, 2)])[0]
            model.pg_collection, model.tp_group = pg, pg.tp
            model.token_dispatcher = SimpleNamespace(tp_group=pg.expt_tp, ep_group=pg.ep,
                tp_ep_group=pg.tp_ep, tp_size=2, tp_rank=0, ep_size=2,
                cudagraph_attrs=['routing_map'], routing_map=torch.ones(3))
            binding = ModelBinding(model, pg, resources, {'tp': 2, 'etp': 0})
        return model, binding, resources

    def test_group_rebind_preserves_pointers_metadata_and_forward(self):
        model, binding, resources = self.fixture()
        ptr, value = model.weight.data_ptr(), model.weight.detach().clone()
        resources.invalidate()
        with group_patches(0, 4):
            new = build_replica_pg_collections([((0, 1, 2, 3), 2, 2, 2)])[0]
            binding.rebind(new)
        self.assertIs(model.pg_collection, new)
        self.assertIs(model.token_dispatcher.tp_group, new.expt_tp)
        self.assertIsNone(model.token_dispatcher.routing_map)
        self.assertEqual(model.weight.data_ptr(), ptr)
        self.assertTrue(torch.equal(value, model.weight))
        self.assertEqual(binding.source_metadata, {'tp': 2, 'etp': 0})
        resources.activate(('epoch', 1))
        model(torch.ones(3))

    def test_shape_change_is_source_only_not_fake_resharding(self):
        model, binding, resources = self.fixture()
        resources.invalidate()
        with group_patches(0, 4):
            new = build_replica_pg_collections([((0, 2), 1, 1, 2)])[0]
            with self.assertRaisesRegex(RuntimeError, 'migrated target'):
                binding.rebind(new)
            binding.rebind(new, source_only=True)
        resources.activate(('epoch', 1))
        with self.assertRaisesRegex(RuntimeError, 'source-only'):
            model(torch.ones(3))
        self.assertEqual(binding.source_metadata['tp'], 2)

    def test_forward_boundary_waits_for_active_call_and_blocks_next_call(self):
        model, binding, resources = self.fixture()
        entered, release, invalidated = threading.Event(), threading.Event(), threading.Event()
        def forward(x):
            entered.set()
            release.wait(2)
            return x
        model.forward = forward
        thread = threading.Thread(target=lambda: model(torch.ones(3)))
        thread.start()
        self.assertTrue(entered.wait(1))
        def invalidate():
            resources.invalidate()
            invalidated.set()
        other = threading.Thread(target=invalidate)
        other.start()
        self.assertFalse(invalidated.wait(0.05))
        release.set()
        thread.join(2)
        other.join(2)
        self.assertTrue(invalidated.is_set())
        with self.assertRaises(RuntimeError):
            model(torch.ones(3))


if __name__ == '__main__':
    unittest.main()
