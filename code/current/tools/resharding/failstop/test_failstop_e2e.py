"""Formal entry: configuration per run mode, argument refusals, dry-run manifest, exclusive-GPU refusal."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import run_failstop_e2e
import serving_config
from common import DETERMINISTIC_ENVIRONMENT

from tools.resharding.e2e.toy_inputs import write_toy_inputs


class ServingConfigTests(unittest.TestCase):
    def build(self, root, run_mode, **overrides):
        inputs = write_toy_inputs(root / 'inputs', count=8, failstop=overrides.pop('failstop', True))
        profile = root / 'profile.json'
        profile.write_text('{}')
        kwargs = dict(node='SL3060', launch='s_x', mode='emulate', output=root / 's_x', master='10.60.14.1', port=29860, commit='c' * 40,
                      remaining=1000.0, min_free_disk_gb=5, quick=False, shared_gpu=False, run_mode=run_mode, method='anchortp',
                      scenario='failstop', checkpoint='/ckpt', seq_length=2048, profile_file=profile, reference_root=None,
                      requests=inputs['requests'], workload=inputs['workload'], plan=inputs['plan'], b_max=8, capacity_tokens=None)
        kwargs.update(overrides)
        with patch.object(serving_config, 'pidfd_implementations', return_value={}):  # a Linux-only probe
            return serving_config.serving_config(**kwargs)

    def test_timing_and_correctness_configurations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timing = self.build(root, 'timing')
            self.assertEqual((timing['deterministic_mode'], timing['timing_valid'], timing['shared_gpu_mode'], timing['quick_mode'],
                              timing['correctness_only'], timing['method'], timing['scenario'], timing['run_mode']),
                             (False, True, False, False, False, 'anchortp', 'failstop', 'timing'))
            self.assertNotIn('--deterministic-mode', timing['model_argv'])
            self.assertEqual(timing['startup_timeout_s'], 600)
            correctness = self.build(root, 'correctness', quick=True, shared_gpu=True)
            self.assertEqual((correctness['deterministic_mode'], correctness['timing_valid'], correctness['shared_gpu_mode'],
                              correctness['quick_mode'], correctness['correctness_only']), (True, False, True, True, True))
            self.assertIn('--deterministic-mode', correctness['model_argv'])
            self.assertEqual(correctness['startup_timeout_s'], 240)
            with self.assertRaisesRegex(RuntimeError, 'never quick'):
                self.build(root, 'timing', quick=True)
            with self.assertRaisesRegex(RuntimeError, 'never quick'):
                self.build(root, 'timing', shared_gpu=True)
            with self.assertRaisesRegex(RuntimeError, 'reference parameter hashes'):
                self.build(root, 'correctness', node='SL3061')

    def test_environment_follows_run_mode(self):
        with patch.dict('os.environ', {'PYTORCH_CUDA_ALLOC_CONF': 'x', **DETERMINISTIC_ENVIRONMENT}, clear=False):
            import os

            serving_config.apply_environment('timing')
            self.assertTrue(all(key not in os.environ for key in (*DETERMINISTIC_ENVIRONMENT, 'PYTORCH_CUDA_ALLOC_CONF')))
            serving_config.apply_environment('correctness')
            self.assertTrue(all(os.environ[key] == value for key, value in DETERMINISTIC_ENVIRONMENT.items()))

    def test_exclusive_precheck_refuses_busy_gpus(self):
        idle = {'gpus': [{'index': i, 'uuid': f'u{i}', 'used_mib': 0, 'util_percent': 0, 'processes': []} for i in range(8)]}
        busy = {'gpus': [{**g, 'processes': [{'pid': 7, 'name': 'python'}] if g['index'] == 3 else []} for g in idle['gpus']]}
        mps = {'clients': set(), 'servers': set()}
        with patch.object(serving_config, 'gpu_snapshot', return_value=idle), patch.object(serving_config, 'mps_snapshot', return_value=mps):
            self.assertEqual(serving_config.require_exclusive_gpus(Path('unused'))['gpus'][0]['uuid'], 'u0')
        with patch.object(serving_config, 'gpu_snapshot', return_value=busy), patch.object(serving_config, 'mps_snapshot', return_value=mps):
            with self.assertRaisesRegex(RuntimeError, 'unrelated GPU process'):
                serving_config.require_exclusive_gpus(Path('unused'))


class EntryTests(unittest.TestCase):
    def argv(self, root, inputs, plan, **extra):
        argv = ['--node', 'SL3060', '--suite-id', 's7', '--output', '/data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance',
                '--requests', str(inputs['requests']), '--workload', str(inputs['workload']), '--plan', str(plan),
                '--profile-file', str(root / 'profile.json'), '--dry-run']
        for key, value in extra.items():
            argv += [f'--{key.replace("_", "-")}', str(value)]
        return argv

    def test_dry_run_manifest_and_refusals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'profile.json').write_text('{}')
            failstop = write_toy_inputs(root / 'f', count=24, failstop=True)
            nofault = write_toy_inputs(root / 'n', count=24, failstop=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = run_failstop_e2e.main(self.argv(root, failstop, failstop['plan'], method='anchortp', launches=3))
            manifest = json.loads(out.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(manifest['launches'], [f's7_anchortp_failstop_emulate_{i:02d}' for i in range(3)])
            self.assertEqual((manifest['instances'], manifest['fail_count'], manifest['run_mode']), (24, 6, 'timing'))
            self.assertEqual(set(manifest['sha256']), {'requests', 'workload', 'plan', 'profile'})
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_failstop_e2e.main(self.argv(root, nofault, nofault['plan'], scenario='nofault'))
            self.assertEqual(code, 0)
            with self.assertRaisesRegex(RuntimeError, 'does not match scenario'):
                run_failstop_e2e.main(self.argv(root, nofault, nofault['plan']))
            with self.assertRaisesRegex(RuntimeError, 'correctness-run flag'):
                run_failstop_e2e.main(self.argv(root, failstop, failstop['plan']) + ['--correctness-only'])
            with self.assertRaisesRegex(RuntimeError, 'under /data'):
                run_failstop_e2e.main(self.argv(root, failstop, failstop['plan'], output=str(root)))


if __name__ == '__main__':
    unittest.main()
