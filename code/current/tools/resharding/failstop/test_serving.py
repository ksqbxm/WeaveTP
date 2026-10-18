"""CPU contract of the Step 6 quick runner: both fake-backend runs and the e2e tests yield the summary the quick driver accepts."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_serving


class ServingQuickRunner(unittest.TestCase):
    def test_summary_flags_runs_and_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            code = run_serving.main(['--node', 'SL3061', '--suite-id', 's6q_cpu', '--output', temporary,
                                     '--quick', '--correctness-only', '--budget-seconds', '280'])
            summary = json.loads(Path(temporary, 'SL3061', 'summary.json').read_text(encoding='utf-8'))
            self.assertFalse(Path(temporary, 'SL3061', 'error.txt').exists())
        self.assertEqual(code, 0)
        self.assertTrue(summary['all_checks_passed'])
        flags = ('layer', 'backend', 'single_node', 'quick_mode', 'correctness_only', 'shared_gpu_mode', 'timing_valid', 'accepted', 'gpus_used')
        self.assertEqual({key: summary[key] for key in flags},
                         {'layer': 'a', 'backend': 'fake', 'single_node': True, 'quick_mode': True, 'correctness_only': True,
                          'shared_gpu_mode': False, 'timing_valid': False, 'accepted': False, 'gpus_used': 0})
        layer = summary['layers']['a']
        self.assertTrue(layer['passed'])
        self.assertEqual(layer['runs']['scaling']['switches'], ['2->4', '4->2'])
        self.assertEqual(layer['runs']['failstop']['switches'], ['shrink', 'expand'])
        self.assertGreater(layer['runs']['failstop']['replay_prefill_tokens'], 0)
        self.assertTrue(layer['tests']['passed'])
        self.assertRegex(layer['tests']['summary_line'], r'\d+ passed')

    def test_only_the_quick_correctness_form_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(RuntimeError):
            run_serving.main(['--node', 'SL3061', '--suite-id', 's6', '--output', temporary, '--quick'])

    def test_b3_needs_the_frozen_profile(self):
        # The compiled Megatron helpers are a server prerequisite; this checks the b3 input validation after it.
        with (tempfile.TemporaryDirectory() as temporary, self.assertRaises(RuntimeError) as failure,
              patch.object(run_serving, 'require_megatron_helpers', lambda: None)):
            run_serving.main(['--node', 'SL3060', '--suite-id', 's6', '--output', temporary, '--quick', '--correctness-only', '--layer', 'b3',
                              '--profile-file', f'{temporary}/absent.json', '--reference-root', temporary])
        self.assertIn('profile', str(failure.exception))


if __name__ == '__main__':
    unittest.main()
