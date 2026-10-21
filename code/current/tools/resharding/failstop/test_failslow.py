"""Fail-slow units: the gate timeline's event order, the combination/budget checks of the entry, the Step 9 configs of
both nodes, the control service's wall-clock close and the co-runner reaping by script name + tag (Linux only)."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import failslow_inject
import run_failslow_quick
import run_failstop_e2e
import serving_config as sc
import serving_probe
from common import NODES
from failslow_probe import Timeline

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.e2e import controller as ctl  # noqa: E402
from tools.resharding.e2e import remote_control  # noqa: E402
from tools.resharding.e2e.toy_inputs import write_toy_inputs  # noqa: E402

TIMELINE = {'warmup_s': 120.0, 'slow_s': 600.0, 'tail_s': 120.0, 'detect_s': 10.0}
SWITCHING = ['serving_start', 'slow_start', 'inject_start', 'detect', 'slow_stop', 'inject_end', 'recovered', 'shrink_done',
             'expand_done', 'deadline', 'serving_done']


def settings(method='weavetp', level='W', fault=(0, 1, 4, 5), **timeline):
    return sc.failslow_settings(method=method, level=level, fault_gpus=list(fault), mode='emulate', **{**TIMELINE, **timeline})


class TimelineTests(unittest.TestCase):
    def test_switching_order_lets_the_shrink_finish_after_the_injection(self):
        timeline = Timeline(TIMELINE, inject=True, switches=True)
        for t, event in enumerate(SWITCHING):  # a slow restart reload: shrink_done after the injection ended
            timeline.mark(event, float(t))
        self.assertEqual(timeline.complete(), [])

    def test_out_of_order_and_repeated_events_are_refused(self):
        for events, bad in ((['serving_start', 'slow_start'], 'detect'), (['serving_start', 'slow_start', 'inject_start', 'detect'], 'recovered'),
                            (['serving_start', 'slow_start', 'inject_start', 'detect', 'slow_stop', 'inject_end', 'recovered'], 'expand_done'),
                            (['serving_start'], 'serving_start')):
            timeline = Timeline(TIMELINE, inject=True, switches=True)
            for t, event in enumerate(events):
                timeline.mark(event, float(t))
            with self.assertRaises(RuntimeError):
                timeline.mark(bad, 99.0)

    def test_an_event_earlier_than_its_prerequisite_is_refused(self):
        timeline = Timeline(TIMELINE, inject=True, switches=True)
        timeline.mark('serving_start', 5.0)
        with self.assertRaises(RuntimeError):
            timeline.mark('slow_start', 4.0)

    def test_variants_leave_out_injection_and_switch_events(self):
        nofault, ignore = Timeline(TIMELINE, inject=False, switches=False), Timeline(TIMELINE, inject=True, switches=False)
        self.assertNotIn('slow_start', nofault.events)
        self.assertNotIn('shrink_done', ignore.events)
        self.assertIn('slow_stop', ignore.events)
        for t, event in enumerate(nofault.events):
            nofault.mark(event, float(t))
        self.assertEqual(nofault.complete(), [])
        with self.assertRaises(RuntimeError):
            nofault.mark('slow_start', 50.0)

    def test_detection_must_fall_inside_the_slow_period(self):
        for detect in (0.0, 600.0, 700.0):
            with self.assertRaises(RuntimeError):
                Timeline({**TIMELINE, 'detect_s': detect}, inject=True, switches=True)


class SettingsTests(unittest.TestCase):
    def test_switching_methods_release_exactly_the_fault_coordinates(self):
        self.assertEqual(settings()['fault_gpus'], [0, 1, 4, 5])
        for method in ('weavetp', 'anchortp', 'restart'):
            with self.assertRaises(RuntimeError):
                settings(method, fault=(2, 3, 6, 7))
        self.assertEqual(settings('ignore', fault=(2, 3))['fault_gpus'], [2, 3])

    def test_refused_combinations(self):
        for kwargs in ({'method': 'weavetp', 'level': None}, {'method': 'bogus'}, {'fault': (0, 0, 1)}, {'detect_s': 600.0},
                       {'warmup_s': -1.0}):
            with self.assertRaises(RuntimeError):
                settings(**kwargs)
        with self.assertRaises(RuntimeError):
            sc.failslow_settings(method='weavetp', level='W', fault_gpus=[0, 1, 4, 5], mode='real', **TIMELINE)

    def test_nofault_has_no_level(self):
        self.assertEqual((settings('nofault', level='S')['level'], settings('nofault')['label']), (None, 'nofault'))


class BudgetTests(unittest.TestCase):
    ARGS = ['--node', 'SL3060', '--suite-id', 'fs', '--output', '/data/x', '--requests', 'r', '--workload', 'w', '--plan', 'p',
            '--profile-file', 'f', '--scenario', 'failslow', '--slow-level', 'M']

    def config(self, remaining, **overrides):
        with tempfile.TemporaryDirectory() as directory, patch.object(sc, 'pidfd_implementations', return_value={}):  # Linux-only probe
            profile = Path(directory) / 'profile.json'
            profile.write_text('{}')
            kwargs = dict(node='SL3061', launch='l', mode='emulate', output='/data/x', master='m', port=1, commit='c', remaining=remaining,
                          min_free_disk_gb=1, quick=False, shared_gpu=False, run_mode='timing', method='weavetp', scenario='failslow',
                          checkpoint='ck', seq_length=2048, profile_file=profile, reference_root=None, requests='r', workload='w',
                          plan='p', b_max=8, capacity_tokens=None, failslow=settings())
            return sc.serving_config(**{**kwargs, **overrides})

    def test_formal_launch_budget_and_timeouts_cover_d_600(self):
        need = sc.failslow_launch_seconds(settings())
        self.assertEqual(need, 120 + 600 + 120 + sc.FAILSLOW_STARTUP_S + 2 * sc.FAILSLOW_RELOAD_S + sc.FAILSLOW_MARGIN_S)
        config = self.config(need)
        self.assertGreaterEqual(config['stage_timeout_s'], 600 + 120 + 2 * sc.FAILSLOW_RELOAD_S)
        self.assertGreaterEqual(config['launch_timeout_s'], 120 + 600 + 120 + 2 * sc.FAILSLOW_RELOAD_S)
        self.assertEqual(config['failslow']['slow_s'], 600.0)
        with self.assertRaises(RuntimeError):
            self.config(need - 1)

    def test_correctness_on_sl3061_needs_no_reference_hashes(self):
        self.assertEqual(self.config(4000, run_mode='correctness', shared_gpu=True)['reference_root'], '.')

    def test_fail_slow_methods_and_settings_stay_in_their_scenario(self):
        with self.assertRaises(RuntimeError):
            self.config(4000, scenario='failstop')
        for method, mode in (('ignore', 'emulate'), ('restart', 'real')):
            with self.assertRaisesRegex(RuntimeError, 'runs only with --scenario failslow'):
                self.config(4000, scenario='failstop', failslow=None, method=method, mode=mode)

    def test_restart_runs_the_emulate_failstop_cycle_with_a_reload_allowance(self):
        config = self.config(4000, scenario='failstop', failslow=None, method='restart', node='SL3060', run_mode='correctness',
                             quick=True, shared_gpu=True)
        self.assertEqual((config['method'], config['failslow'], config['stage_timeout_s']),
                         ('restart', None, sc.FAILSLOW_RELOAD_S + sc.FAILSLOW_MARGIN_S))  # the quick 240 s would cut a reload
        args = run_failstop_e2e.parse(self.ARGS[:-4] + ['--method', 'restart'])
        self.assertEqual((args.scenario, args.failslow, args.budget_seconds), ('failstop', None, 3600))
        with self.assertRaisesRegex(RuntimeError, 'runs only with --scenario failslow'):
            run_failstop_e2e.parse(self.ARGS[:-4] + ['--method', 'restart', '--mode', 'real'])

    def test_shared_gpu_fail_slow_launches_are_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = write_toy_inputs(Path(directory), count=8)
            argv = ['--node', 'SL3060', '--suite-id', 'fs', '--output', '/data/x', '--requests', str(inputs['requests']),
                    '--workload', str(inputs['workload']), '--plan', str(inputs['plan']), '--profile-file', str(inputs['plan']),
                    '--scenario', 'failslow', '--slow-level', 'W', '--dry-run']
            for extra, sharing in (([], 'exclusive'), (['--shared-gpu-ok'], 'shared')):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(run_failstop_e2e.main(argv + extra), 0)
                self.assertEqual(json.loads(out.getvalue())['gpu_sharing'], sharing)

    def test_entry_defaults_the_budget_and_refuses_a_short_one(self):
        args = run_failstop_e2e.parse(self.ARGS + ['--launches', '3', '--method', 'restart'])
        self.assertEqual(args.budget_seconds, 3 * sc.failslow_launch_seconds(args.failslow) + 60)
        self.assertEqual(args.failslow['level'], 'M')
        with self.assertRaises(RuntimeError):
            run_failstop_e2e.parse(self.ARGS + ['--launches', '3', '--budget-seconds', '3600'])
        self.assertEqual(run_failstop_e2e.parse(self.ARGS + ['--method', 'nofault']).failslow['label'], 'nofault')
        with self.assertRaises(RuntimeError):
            run_failstop_e2e.parse(self.ARGS[:-4] + ['--method', 'ignore'])  # ignore without --scenario failslow
        self.assertEqual(run_failstop_e2e.parse(self.ARGS[:-4]).budget_seconds, 3600)


class QuickStep9Tests(unittest.TestCase):
    def configs(self, root, node):
        """run_failslow_quick on one node with run_case replaced: the serving config of every launch."""
        captured = {}

        def run_case(config):
            captured[config['launch']] = config
            return {'status': 'passed'}
        argv = ['--node', node, '--suite-id', 's9q', '--output', str(root / 'run'), '--quick', '--correctness-only',
                '--profile-file', str(root / 'profile.json'), '--reference-root', str(root / 'ref')]
        with patch.multiple(run_failslow_quick, run_case=run_case, hog_pids=lambda: [], require_megatron_helpers=lambda: None,
                            repository_commit=lambda: 'c' * 40, check_data_space=lambda *_: None,
                            gate_checks=lambda path: {'passed': True}, failstop_gate_checks=lambda path: {'passed': True}), \
                patch.object(sc, 'pidfd_implementations', return_value={}), patch.dict(os.environ), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_failslow_quick.main(argv), 0)
        return captured

    def test_both_nodes_build_configs_the_gate_accepts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'profile.json').write_text('{}')
            for tp in (2, 4):
                (root / 'ref' / f'tp{tp}' / 'reference').mkdir(parents=True)
                for rank in range(2 * tp):
                    (root / 'ref' / f'tp{tp}' / 'reference' / f'parameters_rank_{rank}.json').write_text('{}')
            configs = {node: self.configs(root, node) for node in NODES}  # each node builds its own, as on the servers
            self.assertEqual(sorted(configs['SL3060']), sorted([f's9q_{m}_failslow' for m in run_failslow_quick.METHODS] + ['s9q_restart_failstop']))
            for launch, config in configs['SL3060'].items():
                serving_probe.require_same_config(config, configs['SL3061'][launch])
            failstop = configs['SL3060']['s9q_restart_failstop']
            self.assertEqual((Path(failstop['serving']['requests']).parent, failstop['method'], failstop['scenario']),
                             (root / 'run' / 'inputs_failstop', 'restart', 'failstop'))
            peer = configs['SL3061']['s9q_weavetp_failslow']
            per_node = {**peer, 'serving': {**peer['serving'], 'requests': str(root / 'run' / 'SL3061' / 'inputs' / 'requests.jsonl')}}
            with self.assertRaisesRegex(RuntimeError, 'node config mismatch: serving'):  # inputs under the node directory
                serving_probe.require_same_config(configs['SL3060']['s9q_weavetp_failslow'], per_node)


class CloseTests(unittest.TestCase):
    def service(self, count=6):
        ledger = ctl.Ledger(ctl.RankMap({0: {'tp': 4, 'ranks': (0, 1, 2, 3)}, 1: {'tp': 4, 'ranks': (4, 5, 6, 7)}}))
        queue = [ctl.Request(i, f'r{i}', 4, 8) for i in range(count)]
        controller = ctl.AdmissionController(ledger, queue, capacity=lambda r: 10 ** 6, b_max=1)
        channel = remote_control.MemoryChannel()
        return remote_control.ControlService(controller, channel, lambda: 0.0, {'D0': (0,), 'D1': (1,)}, lambda r: 10 ** 6), channel

    def test_close_ends_every_domain_without_admitting(self):
        service, channel = self.service()
        service.start()
        self.assertEqual(len(channel.get('decision/D0/0')['admits']), 1)
        service.close()
        decision = service.push('D0', 1)
        self.assertEqual((decision.phase, decision.admits, service.finished['D0']), ('done', [], True))
        self.assertEqual(service.controller.pointer, 2)  # nothing more left the queue

    def test_a_held_or_switching_domain_keeps_stepping_after_close(self):
        service, _ = self.service()
        service.start()
        service.hold.add('D0')
        service.trigger('D1', 'shrink')
        service.close()
        self.assertEqual(service.push('D0', 1).phase, 'normal')
        self.assertEqual(service.push('D1', 1).switch, 'shrink')
        self.assertFalse(service.finished['D0'] or service.finished['D1'])


class HogScanTests(unittest.TestCase):
    def test_matches_the_script_name_and_the_tag_only(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid, argv in ((11, ['python', '/x/failslow_hog.py', '--tag', 'a', '--ready', 'r']), (12, ['python', 'failslow_hog.py', '--tag', 'b']),
                              (13, ['python', 'serving_worker.py', '--tag', 'a']), (14, ['python', '-c', 'x', '--tag a failslow_hog.py'])):
                (proc / str(pid)).mkdir()
                (proc / str(pid) / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in argv) + b'\0')
            (proc / 'self').mkdir()
            self.assertEqual(sorted(failslow_inject.hog_pids('a', proc)), [11])
            self.assertEqual(sorted(failslow_inject.hog_pids(None, proc)), [11, 12])


@unittest.skipUnless(Path('/proc/self/cmdline').exists(), 'the co-runner scan reads /proc (Linux)')
class InjectorTests(unittest.TestCase):
    def test_stop_reaps_only_this_tag_by_script_name(self):
        def sleeper(tag):  # a stand-in whose command line names failslow_hog.py and the tag
            return subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', str(failslow_inject.HOG), '--tag', tag])
        mine, other = sleeper('launch_a'), sleeper('launch_b')
        try:
            time.sleep(0.3)
            self.assertEqual(failslow_inject.hog_pids('launch_a'), [mine.pid])
            injector = failslow_inject.Injector.__new__(failslow_inject.Injector)
            injector.tag, injector.children = 'launch_a', []
            self.assertEqual(injector.stop(), {'leftover': []})
            self.assertIsNotNone(mine.wait(5))
            self.assertIsNone(other.poll())
        finally:
            for child in (mine, other):
                if child.poll() is None:
                    child.kill()
                child.wait()
        self.assertEqual(failslow_inject.hog_pids('launch_b'), [])
        self.assertTrue(os.access(failslow_inject.HOG, os.R_OK))


if __name__ == '__main__':
    unittest.main()
