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
    def run_driver(self, *, failure=None, peer='PASS'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / 'kickoff' / 'local'
            logs.mkdir(parents=True)
            run = root / 's2q_cpu'
            (run / 'SL3060').mkdir(parents=True)
            (run / 'SL3060' / 'summary.json').write_text(json.dumps({
                'all_cases_passed': True, 'quick_mode': True, 'emulate': 'passed', 'real': 'passed'}))
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
            child.wait.return_value = 1 if failure == 'suite' else 0
            child.poll.return_value = 0
            if failure == 'timeout':
                child.wait.side_effect = [subprocess.TimeoutExpired('suite', 1), 1]
                child.poll.return_value = None
            conn, stream = Mock(), Mock()
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.dict(sys.modules, fcntl=SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=Mock())))
                stack.enter_context(patch.object(sys, 'argv', ['quick_driver', '--node', 'SL3060', '--sha', 'a' * 40,
                                                              '--repo', str(root), '--log-dir', str(logs)]))
                stack.enter_context(patch.object(subprocess, 'check_output', side_effect=git))
                stack.enter_context(patch.object(subprocess, 'Popen', return_value=child))
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
            else:
                self.assertFalse(any(c[0] == 'checkout' for c in calls))
            return result

    def test_pass_requires_restoration_packaging_and_both_modes(self):
        result = self.run_driver()
        self.assertEqual(result['result'], 'PASS')
        self.assertEqual(result['restored_commit'], quick_driver.BASE)
        self.assertTrue(result['quick_mode'] and result['shared_gpu_mode'])

    def test_failure_paths_never_print_pass(self):
        for failure in ('suite', 'restore', 'package', 'dirty', 'timeout'):
            with self.subTest(failure=failure):
                self.assertEqual(self.run_driver(failure=failure)['result'], 'FAIL')
        self.assertEqual(self.run_driver(peer='FAIL')['result'], 'FAIL')

    def test_default_and_quick_case_matrices(self):
        cases = run_topology.cases('both', 5, 3, 'test')
        self.assertEqual([(c['mode'], c['rounds']) for c in cases], [('emulate', 5)] + [('real', 1)] * 3)
        self.assertEqual(len(run_topology.cases('both', 1, 1, 'quick')), 2)
        with self.assertRaises(RuntimeError):
            run_topology.cases('emulate', 0, 1, 'bad')


if __name__ == '__main__':
    unittest.main()
