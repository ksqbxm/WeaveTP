"""CPU/mock smoke regressions. All fixtures are synthetic; no SSH/CUDA execution."""

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import smoke

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code/current/tests/unit_tests/resharding'))
import test_compare_weavetp_16gpu as control
import test_profile_weavetp_16gpu as profiles
import test_weavetp_observations as observations

compare = smoke.compare
BASELINE = '3bc12ce74a0ed048d62a9ffd5ca64d1940779705'
HELPER = REPO / 'code/current' / compare.HELPER


def dry_environment():
    work = '/data/ubuntu/lxh/weavetp'
    env = dict(os.environ, WORK=work, ROOT_OUT=work + '/acceptance/T085_DRY_ONLY/formal',
               PROFILE=work + '/profiles/T08_EXPLICIT/measurement/profile.json',
               CODE_DIR=work + '/WeaveTP/code/current', NODE1_CODE_DIR=work + '/WeaveTP/code/current',
               PYTHON=work + '/envs/megatron/bin/python', NODE1_PYTHON=work + '/envs/megatron/bin/python',
               NODE_RANK='0', MASTER_ADDR='10.60.14.1', MASTER_PORT='29500',
               NODE1_SSH='ubuntu@10.60.14.2', CHECKPOINT='/data/models/DeepSeek-V2-Lite-megatron-v2',
               PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    env.pop('MPS_OWNER', None)
    return env


def dry_run(baseline=False, smoke_mode=False):
    env = dry_environment()
    if smoke_mode:
        env['ROOT_OUT'] = env['WORK'] + '/smoke/smoke_DRY_ONLY'
    command = [sys.executable, '-B', '-X', 'utf8']
    program = None
    if baseline:
        source = subprocess.check_output(['git', 'show', BASELINE + ':code/current/' + compare.HELPER], cwd=REPO)
        program = ('import sys; sys.path.insert(0, ' + repr(str(HELPER.parent)) + '); '
                   'exec(compile(' + repr(source) + ', ' + repr(str(HELPER)) + ", 'exec'), "
                   "{'__name__': '__main__', '__file__': " + repr(str(HELPER)) + '})')
        command += ['-']
    else:
        command += [str(HELPER)]
    command += ['--dry-run'] + (['--smoke'] if smoke_mode else [])
    return subprocess.run(command, input=program.encode() if program is not None else None,
                          env=env, capture_output=True, check=True, timeout=30).stdout


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = control.CompareTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path = self.fixture.path
        self.settings = copy.deepcopy(self.fixture.settings)
        self.settings['root_out'] = '/data/test/smoke/smoke_fixture'
        self.request = compare.make_smoke_case(self.settings, '/data/test')
        self.request['out_dir'] = str(self.path / 'smoke' / 'one' / 'weavetp' / 'r1')
        self.request['env']['OUT_DIR'] = self.request['out_dir']
        fixture = control.fixtures.FormalSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_fixture(world_size=16)
        self.data = fixture.read_run(6)
        observed = observations.IntegrationTests().result_fixture()
        self.data.update(checkpoint_loaded=True, final_active_tp=2, parallel_groups=observed['parallel_groups'])
        self.data['switches'] = self.data['switches'][:2]
        for record, obs in zip(self.data['switches'], observed['switches']):
            record.update(validation_max_diff=.1, plan_observation=obs['plan_observation'])

    def write_result(self, data=None, path=None):
        path = path or self.path / 'result.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(('  ' + json.dumps(data or self.data) + '\n\n').encode())
        return path

    def test_formal_dry_run_is_byte_identical_to_baseline(self):
        before, after = dry_run(True), dry_run()
        expected = json.loads(before)
        for row in expected:
            if row['request']['case'] == 'weavetp':
                row['request']['env'].update(ALLOW_AWARE_SHRINK='1', REROUTE_MIN_GAIN_PCT='0.0')
        self.assertEqual(expected, json.loads(after))
        self.assertEqual(len(json.loads(after)), 9)

    def test_smoke_dry_run_has_one_case_and_only_switch_setting_differs(self):
        rows = json.loads(dry_run(smoke_mode=True))
        self.assertEqual(len(rows), 1)
        request = rows[0]['request']
        self.assertEqual((request['case'], request['repeat'], request['mode']), ('weavetp', 1, 'smoke'))
        formal = json.loads(dry_run())[2]['request']
        for key in formal['env'].keys() - {'OUT_DIR', 'RUN_ID'}:
            self.assertEqual(request['env'][key], '2' if key == 'SWITCHES' else formal['env'][key])
        self.assertTrue(request['out_dir'].startswith(request['work'] + '/smoke/'))
        self.assertTrue(request['env']['RUN_ID'].startswith('smoke'))
        self.assertEqual(rows[0]['node_commands'], [compare.rpc_command(request, n, 'run') for n in (0, 1)])

    def test_smoke_dry_run_never_calls_rpc_or_writes(self):
        with mock.patch.dict(os.environ, WORK='/data/test'), \
                mock.patch.object(compare, 'settings_from_env', return_value=self.settings), \
                mock.patch.object(compare, 'rpc', side_effect=AssertionError('RPC')), \
                mock.patch.object(compare, 'save', side_effect=AssertionError('write')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(compare.main(['--smoke', '--dry-run']), 0)
        self.assertEqual(len(json.loads(output.getvalue())), 1)

    def test_actual_wrapper_receives_two_switches_on_both_nodes(self):
        checkpoint = self.path / 'checkpoint'
        checkpoint.mkdir()
        profile = self.path / 'profile.json'
        profile.write_text('{}')
        capture = self.path / 'argv.bin'
        interpreter = self.path / 'mock_python.sh'
        interpreter.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE"\n'
                               'printf "%s\\0" "$NCCL_DEBUG" >> "$CAPTURE"\n', newline='\n')
        interpreter.chmod(0o755)
        for rank in (0, 1):
            env = dict(os.environ, **compare.node_environment(self.request, rank))
            env.update(PATH=os.environ['PATH'], PYTHON=interpreter.as_posix(),
                       CHECKPOINT=checkpoint.as_posix(), PROFILE=profile.as_posix(),
                       CAPTURE=capture.as_posix(), MSYS_NO_PATHCONV='1',
                       OUT_DIR=(self.path / f'argv_node{rank}').as_posix())
            result = subprocess.run([str(control.BASH), compare.WRAPPER], cwd=REPO / 'code/current',
                                    env=env, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            argv = capture.read_bytes().decode().split('\0')[:-1]
            for key, value in (('--live-switches', '2'), ('--global-batch-size', '8'),
                               ('--seq-length', '1024'), ('--live-expansion-max-wave-tasks', '8192'),
                               ('--live-shrink-max-wave-tasks', '4096'), ('--live-adaptive-residual-max-waves', '4')):
                self.assertEqual(argv[argv.index(key) + 1], value)
            self.assertEqual(argv[-1], 'WARN')

    def test_path_modes_reject_cross_use_traversal_and_prefix_tricks(self):
        compare.check_output_mode('/data/work/smoke/smoke_one', True, '/data/work')
        compare.check_output_mode('/data/work/smoke_other/formal')
        for root, work in [('/data/work/formal', '/data/work'), ('/data/work/smoke', '/data/work'),
                           ('/data/work/smoke/../formal', '/data/work'),
                           ('/data/work/smoke_other/run', '/data/work'),
                           ('/data/work2/smoke/run', '/data/work'),
                           ('/data/work/smoke/run', None), ('/tmp/work/smoke/run', '/tmp/work')]:
            with self.subTest(root=root, work=work), self.assertRaises(ValueError):
                compare.check_output_mode(root, True, work)
        for path in ('/data/work/smoke/run', '/data/work/smoke/../formal'):
            with self.assertRaises(ValueError):
                compare.check_output_mode(path)

    def test_live_smoke_rejects_symlink_escape(self):
        with mock.patch.object(compare, 'os', SimpleNamespace(name='posix')), \
                mock.patch.object(Path, 'resolve', return_value=Path('/data/elsewhere')):
            with self.assertRaisesRegex(ValueError, 'canonical'):
                compare.check_output_mode('/data/work/smoke/run', True, '/data/work')

    def test_smoke_validator_preserves_original_bytes(self):
        path = self.write_result()
        raw = path.read_bytes()
        self.assertEqual(compare.validate_smoke_result(path, 'weavetp'), compare.digest(raw))
        self.assertEqual(path.read_bytes(), raw)
        self.assertNotIn('mode', json.loads(raw))

    def test_invalid_smoke_results_fail(self):
        changes = [lambda d: d.update(checkpoint_loaded=False), lambda d: d.update(final_active_tp=4),
                   lambda d: d.update(logit_validation_mode='absolute'), lambda d: d.update(logit_max_nrmse=.9),
                   lambda d: d['switches'].pop(), lambda d: d['switches'].extend(copy.deepcopy(d['switches'])),
                   lambda d: d['switches'][0].update(direction='4->2'),
                   lambda d: d['switches'][1].update(index=0),
                   lambda d: d['switches'][0].update(validation_max_diff=float('nan')),
                   lambda d: d['parallel_groups'][0].update(nccl_debug='INFO')]
        for change in changes:
            data = copy.deepcopy(self.data)
            change(data)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                compare.validate_smoke_result(self.write_result(data), 'weavetp')

    def test_actual_parallel_groups_are_required_for_both_layouts(self):
        for tp, name in (('2', 'dp'), ('2', 'expt_dp'), ('4', 'dp'), ('4', 'expt_dp')):
            data = copy.deepcopy(self.data)
            data['parallel_groups'][0]['groups'][tp][name]['size'] = 1
            with self.subTest(tp=tp, name=name), self.assertRaises(ValueError):
                compare.validate_smoke_result(self.write_result(data), 'weavetp')

    def test_both_directions_require_default_and_adopted_plans(self):
        for index in (0, 1):
            for name in ('default', 'adopted'):
                for value in (None, {}):
                    data = copy.deepcopy(self.data)
                    data['switches'][index]['plan_observation'][name] = value
                    with self.subTest(index=index, name=name, value=value), self.assertRaises(ValueError):
                        compare.validate_smoke_result(self.write_result(data), 'weavetp')

    def four_switches(self):
        data = copy.deepcopy(self.data)
        data['switches'] = copy.deepcopy(data['switches']) + copy.deepcopy(data['switches'])
        for index, record in enumerate(data['switches']):
            record['index'] = index
        return data

    def test_formal_rejects_smoke_path_even_with_four_switches(self):
        path = self.write_result(self.four_switches(), self.path / 'smoke' / 'result.json')
        with self.assertRaisesRegex(ValueError, 'smoke'):
            compare.validate_result(path, 'weavetp')

    def test_formal_rejects_request_or_completion_marker_after_copying_four_switches(self):
        for name in ('request.json', 'complete.json'):
            directory = self.path / name.replace('.', '_')
            path = self.write_result(self.four_switches(), directory / 'result.json')
            # W2 also needs executor receipts; four records alone are insufficient.
            with self.assertRaises(FileNotFoundError):
                compare.validate_result(path, 'weavetp')
            compare.save(directory / name, {'mode': 'smoke'})
            with self.assertRaisesRegex(ValueError, 'smoke'):
                compare.validate_result(path, 'weavetp')

    def test_formal_also_rejects_explicit_result_marker(self):
        data = self.four_switches()
        data['mode'] = 'smoke'
        with self.assertRaisesRegex(ValueError, 'smoke'):
            compare.validate_result(self.write_result(data), 'weavetp')

    def test_success_uses_smoke_validator_and_never_rewrites_benchmark_result(self):
        captured = []
        class Nodes(control.Nodes):
            def __call__(inner, request, rank, action):
                result = super().__call__(request, rank, action)
                if rank == 0 and action == 'run':
                    captured.append(Path(request['out_dir'], 'result.json').read_bytes())
                return result
        nodes = Nodes(self.data)
        with mock.patch.object(compare, 'check_output_mode'), \
                mock.patch.object(compare, 'validate_result', side_effect=AssertionError('formal validator')):
            compare.run_case(self.request, nodes)
        out = Path(self.request['out_dir'])
        self.assertEqual((out / 'result.json').read_bytes(), captured[0])
        marker = json.loads((out / 'complete.json').read_bytes())
        self.assertEqual(marker['mode'], 'smoke')
        self.assertEqual(marker['result_sha256'], compare.digest(captured[0]))
        with mock.patch.object(compare, 'check_output_mode'), self.assertRaisesRegex(ValueError, 'no reuse'):
            compare.run_case(self.request, nodes)

    def test_failed_peer_stops_without_retry_and_retains_evidence(self):
        nodes = control.Nodes(self.data, 'node1_failure')
        with mock.patch.object(compare, 'check_output_mode'), self.assertRaisesRegex(ValueError, 'node1 failed'):
            compare.run_case(self.request, nodes)
        self.assertTrue(nodes.stopped.is_set())
        self.assertEqual(sum(action == 'run' for _, _, _, action in nodes.calls), 2)
        self.assertFalse(Path(self.request['out_dir'], 'complete.json').exists())
        self.assertTrue(list(Path(self.request['out_dir']).glob('failure.*.json')))

    def test_preflight_identity_is_pinned_before_launch(self):
        self.request['expected_identity'] = {'commit': 'a' * 40, 'profile_sha256': 'b' * 64}
        nodes = control.Nodes(self.data)
        with mock.patch.object(compare, 'check_output_mode'), self.assertRaisesRegex(ValueError, 'preflight'):
            compare.run_case(self.request, nodes)
        self.assertNotIn('run', [row[-1] for row in nodes.calls])

    def test_smoke_and_formal_share_idle_and_recovery_functions(self):
        # Preserve recovery/RPC implementations; exercise the shared idle gate below.
        import ast
        before = ast.parse(subprocess.check_output(
            ['git', 'show', BASELINE + ':code/current/' + compare.HELPER], cwd=REPO).decode())
        after = ast.parse(HELPER.read_text(encoding='utf-8'))
        for name in ('gpu_memory', 'tagged_processes', 'cleanup', 'rpc', 'rpc_command'):
            old = next(n for n in before.body if getattr(n, 'name', None) == name)
            new = next(n for n in after.body if getattr(n, 'name', None) == name)
            self.assertEqual(ast.dump(old), ast.dump(new), name)
        for mode in ('formal', 'smoke'):
            for memory in (200, 201):
                request = copy.deepcopy(self.request)
                request['out_dir'] = str(self.path / f'{mode}_{memory}')
                request['env']['OUT_DIR'] = request['out_dir']
                request['env']['CHECKPOINT'] = str(self.path)
                if mode == 'formal':
                    request.pop('mode')
                state = {'gpus': '\n'.join(f'{i}, GPU-{i}, {memory}, 0' for i in range(8)), 'processes': ''}
                with mock.patch.object(compare, 'check_output_mode'), \
                        mock.patch.object(compare, 'data_path', side_effect=str), \
                        mock.patch.object(compare.socket, 'gethostname', return_value='SL3061'), \
                        mock.patch.object(compare, 'identity', return_value={}), \
                        mock.patch.object(compare, 'gpu_state', return_value=state), \
                        mock.patch.object(compare.subprocess, 'Popen') as popen:
                    if memory == 200:
                        compare.node_action('prepare', request, 1)
                        saved = json.loads(Path(request['out_dir'], 'request.json').read_bytes())
                        self.assertEqual(saved.get('mode'), 'smoke' if mode == 'smoke' else None)
                        with mock.patch.object(compare, 'tagged_processes', return_value=[]):
                            popen.return_value.wait.return_value = 0
                            self.assertTrue(compare.node_action('run', request, 1)['quiescent'])
                    else:
                        with self.assertRaisesRegex(ValueError, r'memory.used=201.*200 MiB'):
                            compare.node_action('prepare', request, 1)
                        popen.assert_not_called()

    def test_smoke_timeout_and_interrupt_reuse_existing_cleanup(self):
        self.request['env']['CHECKPOINT'] = str(self.path)
        state = {'gpus': '\n'.join(f'{i}, GPU-{i}, 65, 0' for i in range(8)), 'processes': ''}
        for reason in (subprocess.TimeoutExpired('worker', 3600), KeyboardInterrupt('operator')):
            request = copy.deepcopy(self.request)
            request['out_dir'] = str(self.path / type(reason).__name__)
            request['env']['OUT_DIR'] = request['out_dir']
            with mock.patch.object(compare, 'check_output_mode'), \
                    mock.patch.object(compare, 'data_path', side_effect=str), \
                    mock.patch.object(compare.socket, 'gethostname', return_value='SL3061'), \
                    mock.patch.object(compare, 'identity', return_value={}), \
                    mock.patch.object(compare, 'gpu_state', return_value=state), \
                    mock.patch.object(compare, 'tagged_processes', return_value=[]), \
                    mock.patch.object(compare, 'cleanup', return_value={'ok': True}) as cleanup, \
                    mock.patch.object(compare.subprocess, 'Popen') as popen:
                compare.node_action('prepare', request, 1)
                popen.return_value.wait.side_effect = [reason, -15]
                receipt = compare.node_action('run', request, 1)
                cleanup.assert_called_once_with(request, 1)
                self.assertEqual(receipt['exit_code'], -15)
                self.assertEqual(popen.return_value.wait.call_args_list[0], mock.call(timeout=3600))

    def test_shell_keeps_operator_profile_even_if_env_file_overrides_it(self):
        probe = r'''
source() { export PROFILE=wrong PROFILE_SHA256=wrong PYTHON=unused; }
exec() { printf '%s\n' "$PROFILE" "$PROFILE_SHA256"; }
builtin source "$1"
'''
        env = dict(os.environ, PROFILE='/data/t08/measurement/profile.json', PROFILE_SHA256='b' * 64)
        result = subprocess.run([str(control.BASH), '-c', probe, 'probe',
                                 str(Path(__file__).with_name('run_smoke.sh'))], env=env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), [env['PROFILE'], env['PROFILE_SHA256']])

    def test_profile_inputs_are_mandatory(self):
        for env in ({}, {'PROFILE': '/data/p'}, {'PROFILE_SHA256': 'a' * 64},
                    {'PROFILE': '/data/p', 'PROFILE_SHA256': 'wrong'}):
            with self.assertRaises(ValueError):
                smoke.explicit_profile(env)

    def test_profile_hash_and_t08_record_are_both_required_without_log_checks(self):
        path = self.path / 't08_fixture' / 'measurement' / 'profile.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(profiles.fixture()), encoding='utf-8')
        sha = compare.digest(path.read_bytes())
        record = path.with_name('profile.json.sha256')
        with self.assertRaises(FileNotFoundError):
            smoke.check_profile(path, sha)
        record.write_text(f'{sha}  profile.json\n')
        with mock.patch.object(profiles.profile, 'network_evidence', side_effect=AssertionError('network check')):
            self.assertEqual(smoke.check_profile(path, sha), sha)
        with self.assertRaises(ValueError):
            smoke.check_profile(path, '0' * 64)
        record.write_text(f'{sha}  wrong.json\n')
        with self.assertRaises(ValueError):
            smoke.check_profile(path, sha)

    def test_preflight_gates_and_no_gpu_launch(self):
        code = self.path / 'repo' / 'code/current'
        request = copy.deepcopy(self.request)
        request.update(work=str(self.path), expected_identity={'commit': 'a' * 40, 'profile_sha256': 'b' * 64})
        request['nodes'][0].update(code_dir=str(code), python=sys.executable)
        env = dict(WORK=str(self.path), REPO=str(code.parents[1]), CODE_DIR=str(code), NODE_RANK='0',
                   NODE1_CODE_DIR=request['nodes'][1]['code_dir'], PYTHON=sys.executable,
                   NODE0_PYTHON=sys.executable, WEAVETP_COMMIT='a' * 40,
                   **{k: request['env'][k] for k in ('MASTER_ADDR', 'MASTER_PORT', 'CHECKPOINT', 'NCCL_DEBUG')})
        state = {'gpus': '\n'.join(f'{i}, GPU-{i}, 65, 0' for i in range(8)), 'processes': ''}
        for failure in (None, 'host', 'commit', 'disk', 'warn', 'profile', 'busy'):
            changed = dict(env)
            if failure == 'warn':
                changed['NCCL_DEBUG'] = 'INFO'
            with self.subTest(failure=failure), \
                    mock.patch.object(compare, 'check_output_mode'), \
                    mock.patch.object(smoke.socket, 'gethostname', return_value='other' if failure == 'host' else 'SL3060'), \
                    mock.patch.object(compare, 'git_identity', return_value='bad' if failure == 'commit' else 'a' * 40), \
                    mock.patch.object(smoke.shutil, 'disk_usage', return_value=SimpleNamespace(free=1 if failure == 'disk' else 20_000_000_000)), \
                    mock.patch.object(smoke, 'check_profile', side_effect=ValueError('hash') if failure == 'profile' else None, return_value='b' * 64), \
                    mock.patch.object(compare, 'gpu_state', return_value=state), \
                    mock.patch.object(compare, 'check_idle', wraps=compare.check_idle) as gate, \
                    mock.patch.object(compare.subprocess, 'Popen', side_effect=AssertionError('launch')):
                if failure == 'busy':
                    gate.side_effect = ValueError('occupied')
                if failure:
                    with self.assertRaises(ValueError):
                        smoke.node_preflight(request, 0, changed)
                else:
                    self.assertEqual(smoke.node_preflight(request, 0, changed)['free_bytes'], 20_000_000_000)
                    gate.assert_called_once_with(state, 0, None)

    def test_controller_success_failure_interrupt_and_warning_only(self):
        for scenario in ('success', 'preflight0', 'preflight1', 'run_failure', 'interrupt'):
            work = self.path / scenario
            env = dry_environment()
            env.update(WORK=str(work), CODE_DIR=str(REPO / 'code/current'),
                       WEAVETP_COMMIT='a' * 40, PROFILE_SHA256='b' * 64)
            events = []
            def preflight(request, rank, root):
                events.append(f'preflight{rank}')
                if scenario == f'preflight{rank}':
                    raise ValueError('preflight failed')
            def run(request):
                events.append('run')
                if scenario == 'interrupt':
                    raise KeyboardInterrupt('operator')
                if scenario == 'run_failure':
                    raise ValueError('peer failed; evidence saved by compare')
                out = Path(request['out_dir'])
                self.write_result(path=out / 'result.json')
                compare.save(out / 'complete.json', {'mode': 'smoke', 'exits': [{'exit_code': 0}] * 2})
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(smoke.socket, 'gethostname', return_value='SL3060'), \
                    mock.patch.object(compare, 'data_path', side_effect=str), \
                    mock.patch.object(compare, 'check_output_mode'), \
                    mock.patch.object(smoke, 'preflight', side_effect=preflight), \
                    mock.patch.object(compare, 'run_case', side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(smoke.controller(), int(scenario != 'success'))
            self.assertIn('| 检查项 |', output.getvalue())
            if scenario == 'success':
                self.assertIn('WARNING', output.getvalue())
                self.assertIn('T085_RESULT exit=0', output.getvalue())
                self.assertEqual(events, ['preflight0', 'preflight1', 'run'])
            else:
                self.assertIn('T085_RESULT exit=1', output.getvalue())
                if scenario.startswith('preflight'):
                    self.assertNotIn('run', events)

    def test_controller_never_writes_into_existing_root(self):
        work = self.path / 'collision'
        root = work / 'smoke' / 'smoke_FIXED_123'
        root.mkdir(parents=True)
        sentinel = root / 'existing.txt'
        sentinel.write_bytes(b'preserve me')
        env = dry_environment()
        env.update(WORK=str(work), CODE_DIR=str(REPO / 'code/current'),
                   WEAVETP_COMMIT='a' * 40, PROFILE_SHA256='b' * 64)
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(smoke.socket, 'gethostname', return_value='SL3060'), \
                mock.patch.object(compare, 'data_path', side_effect=str), \
                mock.patch.object(compare, 'check_output_mode'), \
                mock.patch.object(smoke.time, 'strftime', return_value='FIXED'), \
                mock.patch.object(smoke.os, 'getpid', return_value=123), \
                mock.patch.object(smoke, 'preflight', side_effect=AssertionError('preflight')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(smoke.controller(), 1)
        self.assertEqual(list(root.iterdir()), [sentinel])
        self.assertEqual(sentinel.read_bytes(), b'preserve me')


if __name__ == '__main__':
    unittest.main(verbosity=2)
