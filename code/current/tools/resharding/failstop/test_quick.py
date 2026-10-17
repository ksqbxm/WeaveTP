"""Quick delivery must restore, package, and match peer success before PASS."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import quick_driver
import run_topology


class QuickTests(unittest.TestCase):
    def run_driver(self, *, failure=None, peer='PASS', step=2, child_output=b'', min_free=None, node='SL3060'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / 'kickoff' / 'local'
            logs.mkdir(parents=True)
            run = root / 's2q_cpu'
            (run / node).mkdir(parents=True)
            flags = {'quick_mode': True, 'correctness_only': True, 'shared_gpu_mode': True, 'timing_valid': False, 'accepted': False}
            summary = ({**flags, 'all_checks_passed': True, 'layouts_requested': ['tp2'],
                        'deterministic_step_time_ratio': {} if failure == 'metric' else {'tp2': {'median_ratio': 1.25}}}
                       if step == 5 else
                       {**flags, 'all_cases_passed': True, 'emulate': 'passed', 'real': 'passed',
                        'edp1_rejoin_pause_s': None if failure == 'metric' else 0.5})
            (run / node / 'summary.json').write_text(json.dumps(summary))
            state, calls = {'head': 'a' * 40}, []
            def git(argv, **kwargs):
                args = argv[3:]
                calls.append(args)
                if args[0] == 'checkout':
                    if failure == 'restore' and args[-1] == quick_driver.BASE:
                        raise RuntimeError('restore failure')
                    state['head'] = args[-1]
                if args[0] == 'rev-parse':
                    return state['head']
                if args[0] == 'status' and failure == 'dirty':
                    return ' M tracked.py'
                return ''
            child = Mock()
            child.stdout = io.BytesIO(child_output)
            child.wait.return_value = 1 if failure == 'suite' else 0
            child.poll.return_value = 0
            if failure == 'timeout':
                child.wait.side_effect = [subprocess.TimeoutExpired('suite', 1), 1]
                child.poll.return_value = None
            conn, stream = Mock(), Mock()
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, fcntl=SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=Mock())))
                stack.enter_context(patch.object(sys, 'argv', ['quick_driver', '--node', node, '--sha', 'a' * 40,
                                                              '--repo', str(root), '--log-dir', str(logs), '--step', str(step)] +
                                                 ([] if min_free is None else ['--min-free-disk-gb', str(min_free)])))
                stack.enter_context(patch.object(subprocess, 'check_output', side_effect=git))
                popen = stack.enter_context(patch.object(subprocess, 'Popen', return_value=child))
                stack.enter_context(patch.object(quick_driver, 'rendezvous', return_value=(conn, stream, 's2q_cpu')))
                stack.enter_context(patch.object(quick_driver, 'send'))
                stack.enter_context(patch.object(quick_driver, 'receive', return_value={'result': peer}))
                if failure == 'package':
                    stack.enter_context(patch.object(quick_driver.tarfile, 'open', side_effect=OSError('disk full')))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                code = quick_driver.main()
            result = json.loads((logs / 'final.json').read_text())
            self.assertEqual(code, int(result['result'] != 'PASS'))
            self.assertIn(f"result={result['result']}", output.getvalue())
            self.assertEqual((logs / 'exit_code').read_text().strip(), str(code))
            if failure != 'dirty':
                self.assertIn(['checkout', '--detach', quick_driver.BASE], calls)
                argv = popen.call_args.args[0]
                self.assertEqual(float(argv[argv.index('--min-free-disk-gb') + 1]), 1.0 if min_free is None else min_free)
                self.assertEqual(('--mode' in argv, argv[2].endswith({5: 'run_model.py', 4: 'run_async.py', 3: 'run_sources.py',
                                                                      2: 'run_topology.py'}[step])), (step != 5, True))
                self.assertEqual((logs / 'runner.log').read_bytes(), child_output)
            else:
                self.assertFalse(any(c[0] == 'checkout' for c in calls))
            return result

    def test_pass_requires_restoration_packaging_and_both_modes(self):
        result = self.run_driver()
        self.assertEqual(result['result'], 'PASS')
        self.assertEqual(result['restored_commit'], quick_driver.BASE)
        self.assertTrue(result['quick_mode'] and result['shared_gpu_mode'])

    def test_failure_paths_never_print_pass(self):
        for failure in ('suite', 'restore', 'package', 'dirty', 'timeout', 'metric'):
            with self.subTest(failure=failure):
                self.assertEqual(self.run_driver(failure=failure)['result'], 'FAIL')
        self.assertEqual(self.run_driver(peer='FAIL')['result'], 'FAIL')

    def test_step3_reports_rejoin_pause(self):
        result = self.run_driver(step=3)
        self.assertEqual(result['result'], 'PASS')
        self.assertEqual(result['step'], 3)
        self.assertEqual(result['edp1_rejoin_pause_s'], 0.5)

    def test_peer_node_passes_without_the_controller_metric(self):
        # SL3061's suite summary has no metric; only SL3060 computes and validates it.
        for step in (2, 3, 4):
            result = self.run_driver(step=step, node='SL3061', failure='metric')
            self.assertEqual(result['result'], 'PASS')
            self.assertNotIn('edp1_rejoin_pause_s', result)
        self.assertEqual(self.run_driver(step=4, node='SL3060', failure='metric')['result'], 'FAIL')

    def test_rejoin_pause_summary_uses_only_the_controller_node_results(self):
        peer = [{'mode': 'emulate', 'status': 'passed'}, {'mode': 'real', 'status': 'passed'}]
        self.assertIsNone(run_topology.edp1_rejoin_pause('SL3061', peer))
        controller = [{'mode': 'emulate', 'status': 'passed', 'metrics': {'edp1_rejoin_pause_s': None}},
                      {'mode': 'real', 'status': 'passed', 'metrics': {'edp1_rejoin_pause_s': 6.09}}]
        self.assertEqual(run_topology.edp1_rejoin_pause('SL3060', controller), 6.09)
        with self.assertRaises(KeyError):
            run_topology.edp1_rejoin_pause('SL3060', peer)

    def test_step4_success_and_all_finalization_failures(self):
        result = self.run_driver(step=4)
        self.assertEqual((result['result'], result['step']), ('PASS', 4))
        for failure in ('suite', 'restore', 'package', 'dirty', 'timeout', 'metric'):
            self.assertEqual(self.run_driver(step=4, failure=failure)['result'], 'FAIL')
        self.assertEqual(self.run_driver(step=4, peer='FAIL')['result'], 'FAIL')

    def test_step5_requires_ratio_per_layout_and_prints_its_own_line(self):
        result = self.run_driver(step=5)
        self.assertEqual((result['result'], result['step'], result['layouts']), ('PASS', 5, ['tp2']))
        self.assertEqual(result['deterministic_step_time_ratio'], {'tp2': 1.25})
        self.assertNotIn('edp1_rejoin_pause_s', result)
        for failure in ('suite', 'restore', 'package', 'dirty', 'timeout', 'metric'):
            self.assertEqual(self.run_driver(step=5, failure=failure)['result'], 'FAIL')
        self.assertIn('deterministic-contract', self.run_driver(step=5, failure='metric')['error'])
        self.assertIn('error.txt', self.run_driver(step=5, failure='suite')['error'])
        self.assertEqual(self.run_driver(step=5, peer='FAIL')['result'], 'FAIL')

    def test_preflight_failure_reports_runner_error_and_disk_threshold(self):
        output = (b'{"disk_space": {"free_bytes": 4880000000}}\nTraceback (most recent call last):\n'
                  b'  File "run_topology.py", line 91, in main\n'
                  b"RuntimeError: insufficient /data space: {'free_bytes': 4880000000, 'min_free_gb': 5}\n")
        result = self.run_driver(step=4, failure='suite', child_output=output, min_free=1)
        self.assertEqual(result['result'], 'FAIL')
        self.assertIn('insufficient /data space', result['runner_error'])
        self.assertIn('insufficient /data space', result['error'])
        self.assertIn('no case started', result['error'])
        self.assertNotIn('failed_cases', result)
        self.assertEqual(self.run_driver(step=4, failure='suite')['runner_error'], 'runner produced no output')

    def test_default_and_quick_case_matrices(self):
        cases = run_topology.cases('both', 5, 3, 'test')
        self.assertEqual([(c['mode'], c['rounds']) for c in cases], [('emulate', 5)] + [('real', 1)] * 3)
        self.assertEqual(len(run_topology.cases('both', 1, 1, 'quick')), 2)
        with self.assertRaises(RuntimeError):
            run_topology.cases('emulate', 0, 1, 'bad')


if __name__ == '__main__':
    unittest.main()
