"""CPU contract of the Step 6 quick runner: both fake-backend runs and the e2e tests yield the summary the quick driver accepts."""

import json
import tempfile
import unittest
from pathlib import Path

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
        flags = ('backend', 'single_node', 'quick_mode', 'correctness_only', 'shared_gpu_mode', 'timing_valid', 'accepted', 'gpus_used')
        self.assertEqual({key: summary[key] for key in flags},
                         {'backend': 'fake', 'single_node': True, 'quick_mode': True, 'correctness_only': True,
                          'shared_gpu_mode': False, 'timing_valid': False, 'accepted': False, 'gpus_used': 0})
        self.assertEqual(summary['runs']['scaling']['switches'], ['2->4', '4->2'])
        self.assertEqual(summary['runs']['failstop']['switches'], ['shrink', 'expand'])
        self.assertGreater(summary['runs']['failstop']['replay_prefill_tokens'], 0)
        self.assertTrue(summary['tests']['passed'])
        self.assertRegex(summary['tests']['summary_line'], r'\d+ passed')

    def test_only_the_quick_correctness_form_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(RuntimeError):
            run_serving.main(['--node', 'SL3061', '--suite-id', 's6', '--output', temporary, '--quick'])


if __name__ == '__main__':
    unittest.main()
