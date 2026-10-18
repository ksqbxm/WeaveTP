"""AnchorTP fixed-source planning: static distance priority, no host sources, copied batching semantics."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import anchortp_plan
import anchortp_schedule
from test_sources import part, profile

from megatron.core.resharding.live import TransferTask

DEVICES = ('a', 'b', 'c')
NODES = {'a': 'SL3061', 'b': 'SL3061', 'c': 'SL3060'}
RANKS = {'a': 0, 'b': 1, 'c': 2}


def shard(device, start, stop, **kw):
    return part(device, start, stop, node=NODES[device], rank=RANKS[device], **kw)


def plan(sources, targets, measured=None, **kw):
    return anchortp_plan.build_fixed_plan(sources, targets, measured or profile(DEVICES), membership=list(DEVICES), epoch=1,
                                          rank_map=dict(RANKS), **kw)


class FixedPlanTests(unittest.TestCase):
    def test_nearest_holder_wins_and_atoms_follow_coverage(self):
        tasks, key, record = plan([shard('a', 0, 4), shard('b', 0, 8), shard('c', 0, 8)], [shard('a', 0, 8)])
        self.assertEqual([(t.source.device, t.start, t.stop) for t in tasks], [('a', (0, 0), (2, 4)), ('b', (0, 4), (2, 8))])
        self.assertEqual([t.source_kind for t in tasks], ['local', 'gpu'])
        self.assertEqual(record['planner'], 'anchortp')
        self.assertEqual(record['implementation_ref'], anchortp_schedule.IMPLEMENTATION_REF)
        self.assertEqual((record['paths']['local_copy']['tasks'], record['paths']['gpu_same_node']['tasks'], record['paths']['network']['tasks']),
                         (1, 1, 0))
        self.assertEqual(len(key), 64)

    def test_cross_node_only_when_nothing_closer_covers(self):
        tasks, _, record = plan([shard('b', 4, 8), shard('c', 0, 8)], [shard('a', 0, 8)])
        self.assertEqual([(t.source.device, t.start, t.stop) for t in tasks], [('c', (0, 0), (2, 4)), ('b', (0, 4), (2, 8))])
        self.assertEqual(record['paths']['network']['bytes'], 2 * 4 * 8)

    def test_stable_tie_between_equal_distance_holders(self):
        tasks, _, _ = plan([shard('c', 0, 8), shard('b', 0, 8)], [shard('a', 0, 8)])
        self.assertEqual([t.source.device for t in tasks], ['b'])
        tasks, _, _ = plan([shard('c', 0, 8, expert=0), shard('b', 0, 8, expert=0)], [shard('a', 0, 8, expert=0)])
        self.assertEqual([t.source.device for t in tasks], ['b'])

    def test_host_shards_are_never_sources_and_gaps_fail(self):
        host = part('SL3061/host', 0, 8, kind='host', node='SL3061', rank=None)
        with self.assertRaisesRegex(ValueError, 'uncovered'):
            plan([host, shard('b', 4, 8)], [shard('a', 0, 8)])
        with self.assertRaisesRegex(ValueError, 'membership'):
            plan([part('z', 0, 8, node='SL3060', rank=9)], [shard('a', 0, 8)])

    def test_batches_follow_static_bandwidth_then_task_id(self):
        measured = profile(DEVICES)
        measured['p2p']['c']['a'] = {'bytes_per_s': 5e6, 'latency_s': 1e-5}  # the cross-node link is faster here
        sources = [shard('b', 0, 2), shard('c', 2, 8)]
        tasks, _, record = plan(sources, [shard('a', 0, 8)], measured, max_tasks=1)
        self.assertEqual([(t.source.device, t.wave) for t in tasks], [('b', 1), ('c', 0)])
        self.assertEqual(record['batches'], 2)
        tasks, _, record = plan(sources, [shard('a', 0, 8)], measured)
        self.assertEqual({t.wave for t in tasks}, {0})

    def test_schedule_is_the_copied_reproduction_behaviour(self):
        def task(i, src=0, dst=1, size=4):
            return TransferTask(i, src, dst, size, 'w')

        tasks = [task(7, 2), task(3), task(1), task(9, size=30)]
        rates = {(0, 1): 10, (2, 1): 100}
        self.assertEqual(anchortp_schedule.schedule(tasks, rates, max_tasks=2, max_bytes=8), [[7, 1], [3], [9]])
        self.assertEqual([t.task_id for t in tasks], [7, 3, 1, 9])
        with self.assertRaisesRegex(ValueError, 'unique'):
            anchortp_schedule.schedule(tasks + [tasks[0]], rates)
        with self.assertRaisesRegex(ValueError, 'positive'):
            anchortp_schedule.schedule(tasks, {(0, 1): float('nan'), (2, 1): 100})
        self.assertEqual(anchortp_schedule.CONSTRAINTS['live_max_overlap_steps'], 0)


if __name__ == '__main__':
    unittest.main()
