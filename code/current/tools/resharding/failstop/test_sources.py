"""Independent slice oracle, real mmap backing and mixed-plan CPU contracts."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
from source_fixture import create_checkpoint, demand, read_oracle
from source_profile import fit

from megatron.core.resharding import planner
from megatron.core.resharding.host_weights import HostAttachment, HostOwner, budget
from megatron.core.resharding.mixed_sources import Shard, build_mixed_plan, increments, normalized

CAPACITY = {'ram_free': 1 << 30, 'shm_free': 1 << 30, 'memlock': 1 << 30}


def profile(devices, nodes=('SL3060', 'SL3061')):
    rate = {'bytes_per_s': 1000000.0, 'latency_s': 0.00001}
    return {'h2d': {d: rate for d in devices}, 'host_capacity': {n: rate for n in nodes},
            'p2p': {a: {b: rate for b in devices if a != b} for a in devices}}


def part(device='a', start=0, stop=8, expert=0, *, kind='gpu', node='SL3061', rank=0):
    return Shard(device, node, rank, kind, device, 'weight', expert, 'int64', 8,
                 (2, 8), (0, start), (2, stop), 'TP4')


class SourcesTests(unittest.TestCase):
    def plan(self, sources, targets, measured=None, epoch=1, ranks=None):
        ranks = ranks or {'a': 0, 'b': 1, 'c': 2}
        return build_mixed_plan(sources, targets, measured or profile(ranks), membership=ranks,
                                rank_map=ranks, epoch=epoch)

    def test_local_has_no_L_S_D_but_keeps_transfer(self):
        tasks, _, loads = self.plan([part()], [part()])
        self.assertEqual(loads, {})
        self.assertEqual(sum(t.payload_bytes for t in tasks), 128)
        self.assertEqual(tasks[0].source_kind, 'local')
        with patch.object(planner, '_determine_source_ranks_for_dst_param', return_value=[
                (0, (slice(None),), (slice(0, 4),)), (1, (slice(None),), (slice(4, 8),))]):
            meta = type('Metadata', (), {'shape': (8,), 'element_size': 8})()
            costs = planner._source_metadata_transfer_seconds('w', meta, meta, 0, {(1, 0): 8}, 2)
            self.assertEqual(set(costs), {(1, 0)})
            self.assertAlmostEqual(costs[(1, 0)], 32 / 1e9 + 2e-6)

    def test_host_D_M_no_S_and_partial_GPU_complement(self):
        host = part('SL3061/host', 0, 4, kind='host', rank=None)
        gpu = part('a', 4, 8)
        target = part('b', rank=1)
        inc = increments(host, target, 64, profile(('a', 'b')))
        self.assertEqual(set(inc), {('D', 'b'), ('M', 'SL3061')})
        tasks, _, _ = self.plan([None, host, gpu], [target])
        self.assertEqual([t.source.kind for t in tasks], ['host', 'gpu'])
        result = torch.full((2, 16), -1, dtype=torch.int64)[:, ::2]
        full = torch.arange(16).view(2, 8)
        sources = {'a': full[:, 4:], 'SL3061/host': full[:, :4]}
        coverage = torch.zeros_like(result)
        for t in tasks:
            result[t.target.index(t.start, t.stop)].copy_(sources[t.source.device][t.source.index(t.start, t.stop)])
            coverage[t.target.index(t.start, t.stop)] += 1
        self.assertTrue(torch.equal(result, full))
        self.assertTrue(bool((coverage == 1).all()))

    def test_missing_coverage_expert_foreign_host_failed_and_rank_rejected(self):
        target = part('b', rank=1)
        for source in (None, part(stop=4), part(expert=1), replace(part(), active=False),
                       part('host', kind='host', node='SL3060', rank=None)):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'uncovered'):
                self.plan([source], [target])
        with self.assertRaisesRegex(ValueError, 'membership'):
            self.plan([part('dead')], [target])
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            self.plan([part()], [target, target])
        with self.assertRaises(KeyError):
            self.plan([part()], [target], measured={'p2p': {}})

    def test_partial_TP4_and_TP2_keep_old_coordinates(self):
        sources = [part('a', 0, 4), replace(part('c', 4, 8, rank=2), layout='TP2')]
        tasks, _, _ = self.plan(sources, [part('b', rank=1)])
        self.assertEqual([(t.source.layout, t.start, t.stop) for t in tasks],
                         [('TP4', (0, 0), (2, 4)), ('TP2', (0, 4), (2, 8))])

    def test_measured_node_capacity_can_reject_fast_single_card_host(self):
        host = part('SL3061/host', kind='host', rank=None)
        measured = profile(('a', 'b', 'c'))
        measured['h2d']['b'] = {'bytes_per_s': 1e9, 'latency_s': 0}
        measured['host_capacity']['SL3061'] = {'bytes_per_s': 1e9, 'latency_s': 0}
        fast, _, loads = self.plan([host, part()], [part('b', rank=1)], measured)
        self.assertEqual(fast[0].source.kind, 'host')
        self.assertEqual({key[0] for key in loads}, {'D', 'M'})
        measured['host_capacity']['SL3061'] = {'bytes_per_s': 1, 'latency_s': 0}
        limited, _, _ = self.plan([host, part()], [part('b', rank=1)], measured)
        self.assertEqual(limited[0].source.kind, 'gpu')

    def test_load_aware_choices_and_cache_invalidates_coverage_epoch_profile_rank(self):
        tasks, key, _ = self.plan([part(), part('c', rank=2)], [part('b', rank=1)])
        self.assertEqual(tasks[0].source.device, 'a')
        self.assertNotEqual(key, self.plan([part(), part('c', rank=2)], [part('b', rank=1)], epoch=2)[1])
        changed = profile(('a', 'b', 'c'))
        changed['p2p']['a']['b'] = {'bytes_per_s': 1, 'latency_s': 0}
        alternate, other, _ = self.plan([part(), part('c', rank=2)], [part('b', rank=1)], measured=changed)
        self.assertEqual(alternate[0].source.device, 'c')
        self.assertNotEqual(key, other)
        self.assertNotEqual(key, self.plan([part(stop=4), part('c', rank=2)], [part('b', rank=1)])[1])
        renumbered = {'a': 2, 'b': 0, 'c': 1}
        replay, other, _ = self.plan([part(rank=2), part('c', rank=1)], [part('b', rank=0)], ranks=renumbered)
        self.assertEqual(normalized(tasks), normalized(replay))
        self.assertNotEqual(key, other)

    def test_budget_refuses_before_weight_open_and_modes_report_peaks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = create_checkpoint(root / 'checkpoint', 'SL3061', 64)
            manifest = json.loads(path.read_text())
            ids = demand(manifest['entries'])
            chosen, report = budget(manifest, 'on-demand', ids, **CAPACITY)
            self.assertLess(report['backing_bytes'], report['full_backing_bytes'])
            self.assertEqual(report['temporary_payload_bytes'], 0)
            self.assertEqual({e['id'] for e in chosen}, set(ids))
            with self.assertRaises(MemoryError):
                HostOwner(path, root / 'backing', 'on-demand', ids, capacity={**CAPACITY, 'shm_free': 1})
            self.assertFalse((root / 'backing').exists())
            with self.assertRaises(MemoryError):
                budget(manifest, 'full', ids, **{**CAPACITY, 'memlock': 1})
            with self.assertRaisesRegex(ValueError, 'coverage'):
                budget(manifest, 'on-demand', ['missing'], **CAPACITY)

    def test_one_backing_survives_attachment_close_and_rejoin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = create_checkpoint(root / 'checkpoint', 'SL3061', 64)
            manifest = json.loads(path.read_text())
            owner = HostOwner(path, root / 'backing', 'on-demand', demand(manifest['entries']), capacity=CAPACITY)
            try:
                for _ in range(2):
                    one = HostAttachment(owner.report, register=False)
                    two = HostAttachment(owner.report, register=False)
                    try:
                        self.assertEqual(one.report['backing_inode'], two.report['backing_inode'])
                        entry = owner.report['entries'][0]
                        self.assertTrue(torch.equal(one.tensors[entry['id']], read_oracle(
                            path, entry['name'], entry['expert'], entry['start'][1], entry['stop'][1])))
                    finally:
                        one.close()
                        two.close()
                    self.assertTrue((root / 'backing').exists())
                    owner.verify_unchanged()
                owner.mapping[0:1] = bytes([owner.mapping[0] ^ 1])
                with self.assertRaisesRegex(RuntimeError, 'read-only'):
                    owner.verify_unchanged()
            finally:
                owner.close()
            self.assertFalse((root / 'backing').exists())

    def test_demand_does_not_open_unselected_checkpoint_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = create_checkpoint(root / 'checkpoint', 'SL3061', 64)
            manifest = json.loads(path.read_text())
            ids = demand(manifest['entries'])
            for entry in manifest['entries']:
                if entry['id'] not in ids:
                    (path.parent / entry['file']).unlink()
            owner = HostOwner(path, root / 'backing', 'on-demand', ids, capacity=CAPACITY)
            owner.close()
            with self.assertRaises(FileNotFoundError):
                HostOwner(path, root / 'full', 'full', ids, capacity=CAPACITY)
            self.assertFalse((root / 'full').exists())

    def test_profile_fit_positive_and_no_silent_missing_sample(self):
        result = fit([{'bytes': size, 'elapsed_s': [0.01 + size / 100, 0.01 + size / 100]}
                      for size in (10, 20, 40)])
        self.assertAlmostEqual(result['bytes_per_s'], 100)
        self.assertAlmostEqual(result['latency_s'], 0.01)
        for bad in ([{'bytes': 1, 'elapsed_s': [1]}], [{'bytes': 1, 'elapsed_s': [0]}, {'bytes': 2, 'elapsed_s': [1]}]):
            with self.assertRaises(ValueError):
                fit(bad)


if __name__ == '__main__':
    unittest.main()
