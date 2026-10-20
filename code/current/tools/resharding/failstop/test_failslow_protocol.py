"""Fail-slow serving with 16 concurrent CPU actors (the Step 6 B3 simulation of test_serving_protocol): the gate's
injection timeline, the wall-clock deadline, the five methods and the TP4 self-hash check. The co-runners are the
simulated supervisor's handshake only; nothing here runs on a GPU."""

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
        self.assertEqual(outcome['metrics']['failslow_check'], result['check'])
        self.assertTrue(result['check']['passed'] and result['check']['event_order'], result['check'])
        self.assertGreater(result['requests_unfinished'], 0)  # the deadline cut the pool, nothing drained it
        events, settings = result['events_s'], result['settings']
        self.assertGreaterEqual(events['detect'] - events['inject_start'], settings['detect_s'] - 1e-6)
        self.assertGreaterEqual(events['deadline'] - events['inject_start'], settings['slow_s'] + settings['tail_s'] - 1e-6)
        self.assertEqual(result['method'], method)
        metrics, series = failslow_metrics.analyze(result)
        self.assertGreater(metrics['throughput_tok_s'], 0)
        self.assertEqual(sum(series['global']), sum(len(r['t_tokens']) for r in result['requests']))
        return result, plans, metrics, [s for s in result['switches'] if s['domain'] == 'EDP0']

    def check_hashes(self, result):
        self.assertEqual(sorted(map(int, result['tp4_hash'])), [8, 9, 10, 11, 12, 13, 14, 15])
        self.assertTrue(all(h['passed'] and h['parameters'] > 0 for h in result['tp4_hash'].values()))

    def test_weavetp_shrinks_at_detection_and_expands_after_the_injection(self):
        result, plans, _, edp0 = self.launch('weavetp')
        self.assertEqual([(s['direction'], s['overlap_steps']) for s in edp0], [('shrink', 0), ('expand', 1)])
        self.assertEqual(plans, ['source_plan_0_delta.json', 'source_plan_0_expand.json', 'source_plan_0_shrink.json'])
        events = result['events_s']
        self.assertLess(events['detect'], events['shrink_done'])
        self.assertLess(events['recovered'], events['expand_done'])
        self.assertEqual(len(result['injection']['started']['pids']), 4)
        self.assertEqual(result['injection']['stopped']['leftover'], [])
        self.check_hashes(result)

    def test_anchortp_pauses_both_migrations(self):
        result, _, _, edp0 = self.launch('anchortp')
        self.assertEqual([(s['direction'], s['overlap_steps'], s['bytes']['kv_delta']) for s in edp0],
                         [('shrink', 0, 0), ('expand', 0, 0)])
        self.check_hashes(result)

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
        self.check_hashes(result)

    def test_ignore_keeps_the_layout(self):
        result, plans, _, edp0 = self.launch('ignore')
        self.assertEqual((edp0, plans, result['tp4_hash'], result['check']['tp4_hash_passed']), ([], [], {}, None))
        self.assertIn('slow_stop', result['events_s'])

    def test_nofault_injects_nothing(self):
        result, _, _, edp0 = self.launch('nofault')
        self.assertEqual((edp0, result['level'], result['injection']['started']), ([], None, None))
        self.assertNotIn('slow_start', result['events_s'])

    def test_timing_run_has_no_hash_audit(self):
        result, _, _, edp0 = self.launch('weavetp', run_mode='timing')
        self.assertEqual(([s['direction'] for s in edp0], result['tp4_hash']), (['shrink', 'expand'], {}))

    def test_a_reload_that_differs_fails_the_hash_check(self):
        build = serving_test.FakeRuntime.build

        def corrupt(runtime, tp, groups, *, load):
            model = build(runtime, tp, groups, load=load)
            if load == 'local' and tp == 4:
                next(model.parameters()).data.add_(1)
            return model
        with patch.object(serving_test.FakeRuntime, 'build', corrupt), self.assertRaises(AssertionError) as caught:
            self.launch('restart')
        self.assertIn("'tp4_hash_passed': False", str(caught.exception))


if __name__ == '__main__':
    unittest.main()
