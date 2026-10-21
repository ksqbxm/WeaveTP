"""Fail-slow serving with 16 concurrent CPU actors (the Step 6 B3 simulation of test_serving_protocol): the gate's
injection timeline, the wall-clock deadline and the five methods; no reference audit (reference_audit skipped). The
co-runners are the simulated supervisor's handshake only; nothing here runs on a GPU."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_serving_protocol as serving_test
import torch
from serving_config import failslow_settings

from tools.resharding.e2e import failslow_metrics
from tools.resharding.e2e.toy_inputs import write_toy_inputs

TIMELINE = {'warmup_s': 0.6, 'slow_s': 2.0, 'tail_s': 2.0, 'detect_s': 0.3}


def pool(directory, count, failstop):
    return write_toy_inputs(directory, count=4000, failstop=False)  # far more than the toy model finishes before the deadline


class FailslowProtocolTests(unittest.TestCase):
    def launch(self, method, run_mode='correctness'):
        settings = failslow_settings(method=method, level=None if method == 'nofault' else 'W', fault_gpus=[0, 1, 4, 5],
                                     mode='emulate', **TIMELINE)
        extra = {'scenario': 'failslow', 'method': 'ignore' if method == 'nofault' else method, 'failslow': settings, 'run_mode': run_mode}
        with tempfile.TemporaryDirectory() as directory, patch.object(serving_test, 'write_toy_inputs', pool):
            root = Path(directory)
            torch.use_deterministic_algorithms(run_mode == 'correctness')
            try:
                outcome = serving_test.Simulation('emulate', root, 1, extra).run()
            finally:
                torch.use_deterministic_algorithms(False)
            result = json.loads((root / 'failslow_result.json').read_text())
            plans = sorted(p.name for p in root.glob('source_plan_*.json'))
            audits = [e['weights'] for p in root.glob('serving_checks_*.json') for e in json.loads(p.read_text())['events'] if 'weights' in e]
        skipped = 'timing' if run_mode == 'timing' else 'reference_audit skipped (fail-slow / restart)'
        self.assertTrue(audits and all(a == {'skipped': skipped} for a in audits), audits)
        self.assertEqual(outcome['metrics']['failslow_check'], result['check'])
        self.assertTrue(result['check']['passed'] and result['check']['event_order'], result['check'])
        self.assertGreater(result['requests_unfinished'], 0)  # the deadline cut the pool, nothing drained it
        events, settings = result['events_s'], result['settings']
        self.assertGreaterEqual(events['detect'] - events['inject_start'], settings['detect_s'] - 1e-6)
        self.assertGreaterEqual(events['deadline'] - events['inject_start'], settings['slow_s'] + settings['tail_s'] - 1e-6)
        self.assertEqual((result['method'], result['gpu_sharing']), (method, 'shared'))  # the simulation runs shared_gpu_mode
        metrics, series = failslow_metrics.analyze(result)
        self.assertGreater(metrics['throughput_tok_s'], 0)
        self.assertEqual(sum(series['global']), sum(len(r['t_tokens']) for r in result['requests']))
        return result, plans, metrics, [s for s in result['switches'] if s['domain'] == 'EDP0']

    def test_weavetp_shrinks_at_detection_and_expands_after_the_injection(self):
        result, plans, _, edp0 = self.launch('weavetp')
        self.assertEqual([(s['direction'], s['overlap_steps']) for s in edp0], [('shrink', 0), ('expand', 1)])
        self.assertEqual(plans, ['source_plan_0_delta.json', 'source_plan_0_expand.json', 'source_plan_0_shrink.json'])
        events = result['events_s']
        self.assertLess(events['detect'], events['shrink_done'])
        self.assertLess(events['recovered'], events['expand_done'])
        self.assertEqual(len(result['injection']['started']['pids']), 4)
        self.assertEqual(result['injection']['stopped']['leftover'], [])

    def test_anchortp_pauses_both_migrations(self):
        result, _, _, edp0 = self.launch('anchortp')
        self.assertEqual([(s['direction'], s['overlap_steps'], s['bytes']['kv_delta']) for s in edp0],
                         [('shrink', 0, 0), ('expand', 0, 0)])

    def test_restart_reloads_from_disk_and_replays(self):
        result, plans, metrics, edp0 = self.launch('restart')
        self.assertEqual(plans, [])  # nothing migrates
        self.assertEqual([(s['direction'], s['bytes']['weights'], s['bytes']['kv_prefix']) for s in edp0],
                         [('shrink', 0, 0), ('expand', 0, 0)])
        loaded, events = result['restart_loaded_s'], result['events_s']
        self.assertTrue(events['detect'] < loaded['shrink'] <= events['shrink_done'])
        self.assertTrue(events['recovered'] < loaded['expand'] <= events['expand_done'])
        self.assertGreater(result['replayed_tokens'], 0)
        downtime = metrics['restart_downtime']
        self.assertTrue(downtime['total_s'] > 0 and downtime['shrink']['stop_s'] < downtime['shrink']['resume_s'])

    def test_ignore_keeps_the_layout(self):
        result, plans, _, edp0 = self.launch('ignore')
        self.assertEqual((edp0, plans), ([], []))
        self.assertIn('slow_stop', result['events_s'])

    def test_nofault_injects_nothing(self):
        result, _, _, edp0 = self.launch('nofault')
        self.assertEqual((edp0, result['level'], result['injection']['started']), ([], None, None))
        self.assertNotIn('slow_start', result['events_s'])

    def test_timing_run_switches_without_audits(self):
        result, _, _, edp0 = self.launch('weavetp', run_mode='timing')
        self.assertEqual([s['direction'] for s in edp0], ['shrink', 'expand'])


if __name__ == '__main__':
    unittest.main()
