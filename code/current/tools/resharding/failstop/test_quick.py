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
    def run_driver(self, *, failure=None, peer='PASS', step=2, child_output=b'', min_free=None, node='SL3060',
                   single_node=False, layer='a', method='weavetp'):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / 'kickoff' / 'local'
            logs.mkdir(parents=True)
            layouts = ['tp2', 'tp4'] if single_node else ['tp2']
            flags = {'quick_mode': True, 'correctness_only': True, 'shared_gpu_mode': step != 6 or layer != 'a', 'timing_valid': False,
                     'accepted': False}
            layers = {'a': {'passed': failure != 'metric', 'runs': {'scaling': {'check_passed': True, 'switches': ['2->4', '4->2']},
                                                                   'failstop': {'check_passed': failure != 'metric', 'switches': ['shrink', 'expand']}},
                            'tests': {'passed': True, 'summary_line': '35 passed'}},
                      'b1': {'passed': failure != 'metric', 'nrmse_max': 0.012, 'cosine_min': 0.999, 'top1_agreement': 1.0,
                             'm1_ms': {'static_decode': 30.0, 'block_alone_decode': 31.0, 'block_batched_decode': 34.0},
                             'attention_impl': {'decode': 'torch_splitkv', 'reason': 'requested'}, 'memory': {'physical_vs_segment_accounting_error': 0.0}},
                      'b2': {'passed': failure != 'metric', 'in_process_bitwise': {'tp2_a': True, 'tp2_b': True, 'tp4': True},
                             'cross_process_tp2_bitwise': failure != 'metric', 'forwards': {'tp4': 140}, 'crossed_block_boundary': True},
                      'b3': {'passed': failure != 'metric', 'mode': 'emulate', 'method': method,
                             'metrics': {'requests_completed': 24, 'check_passed': True, 'method': method,
                                         'edp0_switches': [{'direction': 'shrink', 'overlap_steps': 0, 'kv_delta_bytes': 0},
                                                           {'direction': 'expand', 'overlap_steps': 0 if method == 'anchortp' else 2, 'kv_delta_bytes': 0}]},
                             'files': {'result.json': '/data/x/result.json'}}}
            layers['inflight'] = {**layers['b3'], 'inflight_at_expand': 3, 'analysis': {'exit_code': 0, 'rejected': []}}
            summary = ({**flags, 'all_checks_passed': True, 'layouts_requested': layouts,
                        'deterministic_step_time_ratio': {} if failure == 'metric' else {l: {'median_ratio': 1.25} for l in layouts}}
                       if step == 5 else
                       {**flags, 'all_checks_passed': failure != 'metric', 'layer': layer, 'backend': 'fake' if layer == 'a' else 'block',
                        'single_node': layer not in quick_driver.TWO_NODE_LAYERS, 'gpus_used': 0 if layer == 'a' else 8,
                        'layers': {layer: layers[layer]}}
                       if step == 6 else
                       {**flags, 'all_cases_passed': True, 'emulate': 'passed', 'real': 'passed',
                        'edp1_rejoin_pause_s': None if failure == 'metric' else 0.5})
            state, calls = {'head': 'a' * 40}, []

            def rendezvous(*_, **__):
                self.assertFalse(single_node, 'a single-node launch must not wait for a peer')
                return conn, stream, 's2q_cpu'

            def popen(argv, **kwargs):
                # Only the runner's argv names the suite of a single-node launch (no rendezvous assigns it).
                run = root / argv[argv.index('--suite-id') + 1]
                (run / node).mkdir(parents=True)
                (run / node / 'summary.json').write_text(json.dumps(summary))
                return child
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
                                                 (['--layer', layer] if step == 6 else []) +
                                                 (['--method', method] if step == 6 and layer in quick_driver.TWO_NODE_LAYERS else []) +
                                                 ([] if min_free is None else ['--min-free-disk-gb', str(min_free)]) +
                                                 (['--single-node'] if single_node else [])))
                stack.enter_context(patch.object(subprocess, 'check_output', side_effect=git))
                popen = stack.enter_context(patch.object(subprocess, 'Popen', side_effect=popen))
                stack.enter_context(patch.object(quick_driver, 'rendezvous', side_effect=rendezvous))
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
                self.assertEqual(('--mode' in argv, argv[2].endswith({6: 'run_serving.py', 5: 'run_model.py', 4: 'run_async.py',
                                                                      3: 'run_sources.py', 2: 'run_topology.py'}[step])),
                                 (step < 5 or (step == 6 and layer in quick_driver.TWO_NODE_LAYERS), True))
                self.assertEqual('--layer' in argv and argv[argv.index('--layer') + 1] == layer, step == 6)
                self.assertEqual('--method' in argv and argv[argv.index('--method') + 1] == method,
                                 step == 6 and layer in quick_driver.TWO_NODE_LAYERS)
                self.assertEqual('--layouts' in argv and argv[argv.index('--layouts') + 1] == 'tp2,tp4', single_node and step == 5)
                self.assertEqual((logs / 'runner.log').read_bytes(), child_output)
            else:
                self.assertFalse(any(c[0] == 'checkout' for c in calls))
            result['output'] = output.getvalue()
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
        self.assertIn('single_node=false', result['output'])

    def test_step5_single_node_runs_both_layouts_without_a_peer(self):
        result = self.run_driver(step=5, single_node=True)
        self.assertEqual((result['result'], result['single_node'], result['budget_s'], result['layouts']), ('PASS', True, 600, ['tp2', 'tp4']))
        self.assertTrue(result['suite'].startswith('s5q_') and result['suite'].endswith('_single'))
        self.assertNotIn('peer_wait_s', result)
        self.assertIn('single_node=true', result['output'])
        self.assertEqual(self.run_driver(step=5, single_node=True, failure='metric')['result'], 'FAIL')
        self.assertEqual(self.run_driver(step=5)['budget_s'], 300)
        with patch.object(sys, 'argv', ['quick_driver', '--node', 'SL3060', '--sha', 'a' * 40, '--repo', 'r', '--log-dir', 'l',
                                        '--step', '4', '--single-node']), \
                patch.dict(sys.modules, fcntl=SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=Mock())), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                quick_driver.main()
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


    def test_step6_is_the_cpu_fake_backend_check_on_one_node(self):
        result = self.run_driver(step=6, single_node=True)
        self.assertEqual((result['result'], result['step'], result['single_node'], result['budget_s'], result['shared_gpu_mode']),
                         ('PASS', 6, True, 300, False))
        self.assertTrue(result['suite'].startswith('s6q_') and result['suite'].endswith('_single'))
        self.assertEqual(result['runs'], {'scaling': ['2->4', '4->2'], 'failstop': ['shrink', 'expand']})
        self.assertIn('QUICK CHECK step=6 layer=a result=PASS', result['output'])
        self.assertIn('backend=fake single_node=true accepted=false', result['output'])
        failed = self.run_driver(step=6, single_node=True, failure='metric')
        self.assertEqual(failed['result'], 'FAIL')
        self.assertIn('layer a', failed['error'])
        self.assertIn('pytest.log', self.run_driver(step=6, single_node=True, failure='suite')['error'])
        with self.assertRaises(SystemExit):  # Step 6 has no two-node form
            self.run_driver(step=6)

    def test_step6_real_model_layers_single_node_and_two_node(self):
        for layer in ('b1', 'b2'):
            result = self.run_driver(step=6, single_node=True, layer=layer)
            self.assertEqual((result['result'], result['layer'], result['shared_gpu_mode'], result['budget_s']), ('PASS', layer, True, 300))
            self.assertIn(f'QUICK CHECK step=6 layer={layer} result=PASS', result['output'])
            self.assertIn('backend=block attention=torch_splitkv', result['output'])
            self.assertEqual(self.run_driver(step=6, single_node=True, layer=layer, failure='metric')['result'], 'FAIL')
        b3 = self.run_driver(step=6, layer='b3')  # two nodes: the rendezvous assigns the suite
        self.assertEqual((b3['result'], b3['single_node'], b3['layer']), ('PASS', False, 'b3'))
        self.assertIn('single_node=false', b3['output'])
        self.assertIn('QUICK CHECK step=6 layer=b3 method=weavetp result=PASS', b3['output'])
        anchortp = self.run_driver(step=6, layer='b3', method='anchortp')
        self.assertEqual(anchortp['result'], 'PASS')
        self.assertIn('QUICK CHECK step=6 layer=b3 method=anchortp result=PASS', anchortp['output'])
        self.assertIn('"method": "anchortp"', anchortp['output'])
        with self.assertRaises(SystemExit):
            self.run_driver(step=6, layer='b3', single_node=True)
        with self.assertRaises(SystemExit):
            self.run_driver(step=6, layer='b1')

    def test_step6_inflight_layer_runs_on_two_nodes_with_a_600_s_budget(self):
        result = self.run_driver(step=6, layer='inflight')
        self.assertEqual((result['result'], result['single_node'], result['layer'], result['budget_s']), ('PASS', False, 'inflight', 600))
        self.assertEqual((result['inflight']['inflight_at_expand'], result['inflight']['analysis']['rejected']), (3, []))
        self.assertIn('QUICK CHECK step=6 layer=inflight method=weavetp result=PASS', result['output'])
        self.assertIn('"inflight_at_expand": 3', result['output'])
        self.assertEqual(self.run_driver(step=6, layer='inflight', failure='metric')['result'], 'FAIL')
        self.assertEqual(self.run_driver(step=6, layer='b3')['budget_s'], 300)
        with self.assertRaises(SystemExit):
            self.run_driver(step=6, layer='inflight', single_node=True)


if __name__ == '__main__':
    unittest.main()
