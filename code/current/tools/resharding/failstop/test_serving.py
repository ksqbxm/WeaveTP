"""CPU contract of the Step 6 quick runner: both fake-backend runs and the e2e tests yield the summary the quick driver accepts."""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_serving
import serving_config
import serving_probe
from common import NODES

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.unit_tests.e2e import test_metrics as gate_files  # noqa: E402


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

    def two_node_layer(self, temporary, node, layer, gate=None):
        """run_serving.main for a two-node layer with run_case replaced: returns (run_case config, summary)."""
        root = Path(temporary)
        (root / 'profile.json').write_text('{}')
        for tp in (2, 4):
            (root / 'ref' / f'tp{tp}' / 'reference').mkdir(parents=True, exist_ok=True)
            for rank in range(2 * tp):
                (root / 'ref' / f'tp{tp}' / 'reference' / f'parameters_rank_{rank}.json').write_text('{}')
        seen = {}

        def run_case(config):
            seen['config'] = config
            output = Path(config['output'])
            if gate is not None:  # the gate's result.json / events_actual.json
                shutil.copytree(gate(root / 'gate' / node), output)
            output.mkdir(exist_ok=True)
            metrics = {'check_passed': True, 'kv_check_passed': True, 'edp0_switches': []}
            (output / 'controller_result.json').write_text(json.dumps({'status': 'passed', 'metrics': metrics}))
            return {'status': 'passed'}

        argv = ['--node', node, '--suite-id', 's6q_cpu', '--output', str(root / 'run'), '--quick', '--correctness-only', '--layer', layer,
                '--profile-file', str(root / 'profile.json'), '--reference-root', str(root / 'ref')]
        with patch.multiple(run_serving, run_case=run_case, require_megatron_helpers=lambda: None, repository_commit=lambda: 'c' * 40,
                            check_data_space=lambda *_: None), \
                patch.object(serving_config, 'pidfd_implementations', return_value={}), patch.dict(os.environ), \
                contextlib.redirect_stdout(io.StringIO()):
            try:
                run_serving.main(argv)
            except RuntimeError:
                pass
        return seen['config'], json.loads((root / 'run' / node / 'summary.json').read_text(encoding='utf-8'))

    def test_two_node_layers_give_both_nodes_the_same_serving_config(self):
        # The gate refuses a launch whose nodes disagree on a shared key; the toy input paths are part of 'serving'.
        for layer in run_serving.TWO_NODE_LAYERS:
            with self.subTest(layer=layer), tempfile.TemporaryDirectory() as temporary:
                configs = {node: self.two_node_layer(temporary, node, layer, gate=self.gate)[0] for node in NODES}
                serving_probe.require_same_config(configs['SL3060'], configs['SL3061'])
                requests = Path(configs['SL3060']['serving']['requests'])
                self.assertEqual(requests.parent.name, f'inputs_{layer}')
                lengths = [json.loads(line)['output_len'] for line in requests.read_text().splitlines()]
                self.assertEqual((len(lengths), min(lengths), max(lengths)), (200, 96, 288) if layer == 'inflight' else (24, 3, 9))

    @staticmethod
    def gate(path, cutover=None):
        """test_metrics' hand-built gate files (EDP0 expand cutover at 40 s, requests running across it); `cutover`
        moves the cutover, e.g. past every request (nothing in flight)."""
        directory = gate_files.launch(path)
        if cutover is not None:
            result = json.loads((directory / 'result.json').read_text())
            next(s for s in result['switches'] if s['direction'] == 'expand')['milestones']['cutover'] = cutover
            (directory / 'result.json').write_text(json.dumps(result))
        return directory

    def test_inflight_layer_needs_handed_over_requests_and_an_accepted_analysis(self):
        for cutover, passed in ((None, True), (1000.0, False)):
            with self.subTest(cutover=cutover), tempfile.TemporaryDirectory() as temporary:
                _, summary = self.two_node_layer(temporary, 'SL3060', 'inflight', gate=lambda path: self.gate(path, cutover))
                layer = summary['layers']['inflight']
                self.assertEqual((layer['passed'], summary['all_checks_passed'], layer['inflight_at_expand'] > 0), (passed, passed, passed))
                self.assertEqual((layer['analysis']['exit_code'], layer['analysis']['rejected']), (0, []))
                self.assertEqual('inflight_error' in layer, not passed)


if __name__ == '__main__':
    unittest.main()
