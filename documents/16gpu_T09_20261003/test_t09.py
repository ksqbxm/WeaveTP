"""CPU fixtures: W2 evidence, byte-stable baselines, resume and isolation."""

import contextlib
import copy
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import isolate_failed as isolate
import t09

sys.path.insert(0, str(t09.REPO / 'documents/16gpu_T085_20261001'))
import test_smoke as smoke_tests

compare = t09.compare
control = smoke_tests.control
observations = smoke_tests.observations


def w2_fixture(data):
    data = copy.deepcopy(data)
    data.update(bandwidth_aware_shrink=True, reroute_min_gain_pct=0,
                checkpoint_loaded=True, final_active_tp=2, parallel_groups=observations.group_fixture())
    for index, record in enumerate(data['switches']):
        decision = observations.switch_record(record['direction'])
        decision['index'] = index
        parts = []
        for rank in range(16):
            candidate = observations.plan(1, observations.gate_stats())
            parts.append(observations.obs.local_switch_observation(
                decision, default_plan=observations.plan(), cached_plan=candidate,
                adopted_plan=candidate, base_plan=observations.restrict_plan_sequence(candidate, start=0, end=8, include_non_kv=True),
                delta_plan=observations.restrict_plan_sequence(candidate, start=8, end=11, include_non_kv=False),
                bundle=observations.Bundle(), rank=rank, restrict_sequence=observations.restrict_plan_sequence,
                enabled=True, allow_aware_shrink=True, threshold=5.0))
        observed = observations.obs.merge_switch_observations(parts, {r: 'SL3060' if r < 8 else 'SL3061' for r in range(16)})
        record.update(validation_max_diff=.1, plan_variant='baseline',
                      plan_observation=observed)
    return data


class W2Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = control.CompareTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.settings = self.fixture.settings
        self.request = list(compare.make_cases(self.settings))[2]
        fixture = control.fixtures.FormalSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_fixture(16)
        self.data = w2_fixture(fixture.read_run(6))
        self.out = Path(self.request['out_dir'])
        self.out.mkdir(parents=True)
        self.write('request.json', self.request)
        for rank in (0, 1):
            self.write(f'exit.node{rank}.json', {'rank': rank, 'config_sha256': compare.config_hash(self.request),
                       'exit_code': 0, 'error': None, 'quiescent': True, 'env': self.request['env']})
        self.path = self.out / 'result.json'
        self.write('result.json', self.data)
        self.write('complete.json', {'config_sha256': compare.config_hash(self.request),
                   'result_sha256': compare.digest(self.path.read_bytes()),
                   'exits': [t09.read(self.out / f'exit.node{r}.json') for r in (0, 1)]})

    def write(self, name, value):
        (self.out / name).write_text(json.dumps(value), encoding='utf-8')
        if name == 'result.json' and (self.out / 'complete.json').is_file():
            complete = t09.read(self.out / 'complete.json')
            complete['result_sha256'] = compare.digest((self.out / name).read_bytes())
            (self.out / 'complete.json').write_text(json.dumps(complete), encoding='utf-8')

    def test_w2_geometry_accepts_baseline_fifo_labels_and_original_bytes(self):
        raw = self.path.read_bytes()
        self.assertEqual(compare.validate_result(self.path, 'weavetp'), compare.digest(raw))
        self.assertEqual(self.path.read_bytes(), raw)

    def test_w0_missing_configuration_and_mismatched_configuration_rejected(self):
        mutations = [lambda d: d.update(bandwidth_aware_shrink=False),
                     lambda d: d.pop('bandwidth_aware_shrink'), lambda d: d.update(reroute_min_gain_pct=10)]
        for change in mutations:
            data = copy.deepcopy(self.data)
            change(data)
            self.write('result.json', data)
            with self.assertRaises(ValueError):
                compare.validate_result(self.path, 'weavetp')

    def test_request_and_complete_configuration_are_mandatory(self):
        for key in ('ALLOW_AWARE_SHRINK', 'REROUTE_MIN_GAIN_PCT'):
            for value in (None, 'bad'):
                request = copy.deepcopy(self.request)
                request['env'][key] = value
                self.write('request.json', request)
                with self.assertRaises(ValueError):
                    compare.validate_result(self.path, 'weavetp')
        self.write('request.json', self.request)
        (self.out / 'request.json').unlink()
        with self.assertRaises(FileNotFoundError):
            compare.validate_result(self.path, 'weavetp')
        self.write('request.json', self.request)
        complete = {'config_sha256': compare.config_hash(self.request), 'result_sha256': compare.digest(self.path.read_bytes()),
                    'exits': [t09.read(self.out / f'exit.node{r}.json') for r in (0, 1)]}
        self.write('complete.json', complete)
        compare.validate_result(self.path, 'weavetp')
        complete['exits'][1]['env'].pop('REROUTE_MIN_GAIN_PCT')
        self.write('complete.json', complete)
        with self.assertRaises(ValueError):
            compare.validate_result(self.path, 'weavetp')

    def test_each_direction_requires_aware_geometry_and_accepted_gate(self):
        for index in range(4):
            for field in ('geometry', 'gate', 'traffic'):
                data = copy.deepcopy(self.data)
                obs = data['switches'][index]['plan_observation']
                if field == 'geometry':
                    obs['adopted'] = obs['default']
                elif field == 'gate':
                    obs['candidate_gate']['global_gate_accepted'] = False
                else:
                    obs['adopted']['traffic']['total']['cross_node_bytes'] += 1
                self.write('result.json', data)
                with self.assertRaises(ValueError):
                    compare.validate_result(self.path, 'weavetp')


class OperationsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = control.CompareTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.settings = self.fixture.settings

    def test_fixed_directional_requests_and_commands_byte_identical(self):
        old, new = map(json.loads, (smoke_tests.dry_run(True), smoke_tests.dry_run()))
        for a, b in zip(old, new):
            if a['request']['case'] == 'weavetp':
                self.assertEqual({k for k in a['request']['env'] if a['request']['env'][k] != b['request']['env'][k]},
                                 {'ALLOW_AWARE_SHRINK', 'REROUTE_MIN_GAIN_PCT'})
                a['request']['env'].update(ALLOW_AWARE_SHRINK='1', REROUTE_MIN_GAIN_PCT='0.0')
            self.assertEqual(json.dumps(a, indent=2).encode(), json.dumps(b, indent=2).encode())

    def test_only_then_resume_reuses_executor_and_preserves_requests_and_round_order(self):
        all_cases = t09.selected_cases(self.settings)
        only = t09.selected_cases(self.settings, 'weavetp_r1')
        nodes, launched = {}, []
        fixture = control.fixtures.FormalSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.build_fixture(16)
        expected_results = {}
        def execute(request):
            key = request['case'], request['repeat']
            if key not in nodes:
                number = compare.CASES.index(request['case']) * 3 + request['repeat'] - 1
                data = fixture.read_run(number)
                data.update(checkpoint_loaded=True, final_active_tp=2)
                for record in data['switches']:
                    record['validation_max_diff'] = .1
                if request['case'] == 'weavetp':
                    data = w2_fixture(data)
                expected_results[key] = data
                nodes[key] = control.Nodes(data)
                launched.append(key)
            def call(r, rank, action):
                receipt = nodes[key](r, rank, action)
                if action == 'prepare' and rank == 0:
                    compare.save(Path(r['out_dir']) / 'request.json', r)
                return receipt
            compare.run_case(request, call)
        with mock.patch.object(t09, 'launch_summary'), mock.patch.object(t09, 'round_summary') as summary, \
                contextlib.redirect_stdout(io.StringIO()):
            for cases in (only, all_cases):
                t09.run_queue(self.settings, cases, {}, '/data/work', {}, check=lambda *a: None, execute=execute)
        self.assertEqual(len(launched), 9)
        self.assertEqual(launched, [('weavetp', 1)] + [(r['case'], r['repeat']) for r in all_cases if r not in only])
        self.assertEqual([c.args[1] for c in summary.call_args_list], [1, 1, 2, 3])
        self.assertEqual(t09.selected_cases(self.settings, rounds='r1'), all_cases[:3])
        self.assertEqual(only[0], all_cases[2])
        self.assertEqual(sum(c[-1] == 'run' for n in nodes.values() for c in n.calls), 18)
        self.assertTrue(all((Path(r['out_dir']) / 'complete.json').is_file() for r in all_cases))
        self.assertFalse(any((Path(r['out_dir']) / 'exit.node1.json').exists() for r in all_cases))
        for request in all_cases:
            self.assertEqual(t09.read(Path(request['out_dir']) / 'result.json'),
                             expected_results[request['case'], request['repeat']])
        resumed_bytes = {(r['case'], r['repeat']): (Path(r['out_dir']) / 'result.json').read_bytes() for r in all_cases}
        once_settings = {**self.settings, 'root_out': self.settings['root_out'] + '_once'}
        once = list(compare.make_cases(once_settings))
        nodes, launched = {}, []
        with mock.patch.object(t09, 'launch_summary'), mock.patch.object(t09, 'round_summary'), \
                contextlib.redirect_stdout(io.StringIO()):
            t09.run_queue(once_settings, once, {}, '', {}, check=lambda *a: None, execute=execute)
        self.assertEqual(launched, [(r['case'], r['repeat']) for r in all_cases])
        for r in once:
            self.assertEqual((Path(r['out_dir']) / 'result.json').read_bytes(), resumed_bytes[r['case'], r['repeat']])

    def test_failed_directory_is_ignored_by_executor_and_manifest_paths(self):
        failed = Path(self.settings['root_out']) / '_failed' / 'weavetp_r1_old'
        failed.mkdir(parents=True)
        (failed / 'result.json').write_text('invalid abandoned evidence')
        request = list(compare.make_cases(self.settings))[0]
        nodes = control.Nodes(self.fixture.result)
        compare.run_case(request, nodes)
        compare.run_case(request, nodes)
        fixture = control.fixtures.FormalSummaryTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        archive = fixture.batch / '_failed' / 'weavetp_r1_old'
        archive.mkdir(parents=True)
        (archive / 'result.json').write_text('invalid')
        self.assertEqual(len(control.fixtures.formal.load_launches(fixture.manifest_path)[1]), 9)

    def test_paths_and_selection_fail_closed(self):
        t09.formal_path('/data/work/formal/run', '/data/work')
        for root in ('/tmp/run', '/data/work/formal', '/data/work/formal/../bad', '/data/work/smoke/run',
                     '/data/work/formal/_failed/run', '/data/work2/formal/run'):
            with self.assertRaises(ValueError):
                t09.formal_path(root, '/data/work')
        for only, rounds in (('bad', ''), ('weavetp_r1', 'r1'), ('', 'r2')):
            with self.assertRaises(ValueError):
                t09.selected_cases(self.settings, only, rounds)

    def test_dry_run_nine_only_and_no_io(self):
        env = smoke_tests.dry_environment()
        env.update(ROOT_OUT=env['WORK'] + '/formal/T09_DRY_ONLY', PROFILE_SHA256=t09.read(t09.HERE / 'cpu_reference.json')['profile_sha256'], ONLY='', ROUNDS='')
        for only, count in (('', 9), ('weavetp_r1', 1)):
            env['ONLY'] = only
            output = io.StringIO()
            with mock.patch.dict(os.environ, env), mock.patch.object(t09, 'preflight', side_effect=AssertionError('no SSH')), \
                    mock.patch.object(compare, 'run_case', side_effect=AssertionError('no launch')), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(t09.main(['--dry-run']), 0)
            self.assertEqual(len(json.loads(output.getvalue())), count)

    def test_disk_constant_matches_budget(self):
        self.assertEqual(t09.MIN_FREE_BYTES, max(5_000_000_000, 9 * 500_000_000 * 2))

    def test_round_summary_rejects_stale_or_changed_completion(self):
        request = self.fixture.requests[0]
        compare.run_case(request, control.Nodes(self.fixture.result))
        out = Path(request['out_dir'])
        original = t09.read(out / 'complete.json')
        for field in ('config_sha256', 'result_sha256'):
            with self.subTest(field=field):
                marker = {**original, field: '0' * 64}
                (out / 'complete.json').write_text(json.dumps(marker), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'completion'), contextlib.redirect_stdout(io.StringIO()):
                    t09.round_summary(self.settings, 1)

    def test_preflight_checks_both_nodes_and_rejects_each_gate(self):
        work = self.fixture.path
        code = work / 'repo' / 'code/current'
        profile = work / 'measurement/profile.json'
        profile.parent.mkdir()
        profile.write_text('{}')
        sha, commit = 'a' * 64, 'b' * 40
        profile.with_name('profile.json.sha256').write_text(sha + '  profile.json\n')
        settings = copy.deepcopy(self.settings)
        settings['profile'] = str(profile)
        for node in settings['nodes']:
            node.update(code_dir=str(code), python=sys.executable)
        request = list(compare.make_cases(settings))[0]
        payload = {'request': request, 'work': str(work), 'root_out': str(work / 'formal/batch'),
                   'identity': {'commit': commit, 'profile_sha256': sha}}
        for rank in (0, 1):
            env = {**request['env'], 'NODE_RANK': str(rank), 'WORK': str(work),
                   'REPO': str(work / 'repo'), 'CODE_DIR': str(code), 'NODE1_CODE_DIR': str(code),
                   'PYTHON': sys.executable, 'NODE0_PYTHON': sys.executable, 'NODE1_PYTHON': sys.executable,
                   'WEAVETP_COMMIT': commit}
            with mock.patch.object(t09, 'formal_path'), mock.patch.object(compare, 'data_path'), \
                    mock.patch.object(t09.socket, 'gethostname', return_value=f'SL306{rank}'), \
                    mock.patch.object(compare, 'git_identity', return_value=commit), \
                    mock.patch.object(compare, 'load_profile'), mock.patch.object(compare, 'gpu_state', return_value={}), \
                    mock.patch.object(compare, 'check_idle') as idle, \
                    mock.patch.object(t09.shutil, 'disk_usage') as disk:
                disk.return_value.free = t09.MIN_FREE_BYTES
                self.assertEqual(t09.node_preflight(payload, rank, env)['rank'], rank)
                disk.return_value.free -= 1
                with self.assertRaisesRegex(ValueError, '/data needs'):
                    t09.node_preflight(payload, rank, env)
                disk.return_value.free += 1
                root = Path(payload['root_out'])
                root.mkdir(parents=True, exist_ok=True)
                if rank == 1:
                    compare.save(root / 'compare.lock', {'transaction': 'unfinished isolation'})
                    with self.assertRaisesRegex(ValueError, 'lock'):
                        t09.node_preflight(payload, rank, env)
                    (root / 'compare.lock').unlink()
                for key, value in (('NCCL_DEBUG', 'INFO'), ('WEAVETP_COMMIT', 'c' * 40), ('NODE_RANK', 'bad')):
                    with self.assertRaises(ValueError):
                        t09.node_preflight(payload, rank, {**env, key: value})
                with mock.patch.object(compare, 'git_identity', side_effect=ValueError('dirty')):
                    with self.assertRaisesRegex(ValueError, 'dirty'):
                        t09.node_preflight(payload, rank, env)
                with mock.patch.object(compare, 'load_profile', side_effect=ValueError('hash')):
                    with self.assertRaisesRegex(ValueError, 'hash'):
                        t09.node_preflight(payload, rank, env)
                idle.side_effect = ValueError('occupied GPU')
                with self.assertRaisesRegex(ValueError, 'occupied GPU'):
                    t09.node_preflight(payload, rank, env)

    def test_failure_stops_queue_and_end_is_printed(self):
        output = io.StringIO()
        calls = []
        def fail(request):
            calls.append(request)
            raise RuntimeError('fixture failure')
        with contextlib.redirect_stdout(output), self.assertRaises(RuntimeError):
            t09.run_queue(self.settings, list(compare.make_cases(self.settings)), {}, '', {},
                          check=lambda *a: None, execute=fail)
        self.assertEqual(len(calls), 1)
        self.assertIn('LAUNCH_END', output.getvalue())


class IsolationTests(unittest.TestCase):
    def node_fixtures(self):
        fixture = control.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        payloads = []
        for rank in (0, 1):
            root = fixture.path / f'node{rank}'
            request = copy.deepcopy(fixture.requests[0])
            source = root / 'fixed/r1'
            source.mkdir(parents=True)
            request['out_dir'] = str(source)
            compare.save(source / 'request.json', request)
            payloads.append({'request': request, 'root_out': str(root), 'work': str(root.parent),
                             'source': str(source), 'target': str(root / '_failed/fixed_r1_fixture'),
                             'transaction': 'fixed_r1_fixture', 'reason': 'reason'})
        return payloads

    def test_prepare_rejection_releases_owned_lock_and_preserves_foreign_lock(self):
        payloads = self.node_fixtures()
        def call(payload, rank, action):
            with mock.patch.object(isolate.socket, 'gethostname', return_value=f'SL306{rank}'):
                return isolate.node_action(payloads[rank], rank, action)

        peer = Path(payloads[1]['root_out'])
        for rejection in ('completed', 'foreign_lock'):
            with self.subTest(rejection=rejection):
                rejected = peer / ('fixed/r1/complete.json' if rejection == 'completed' else 'compare.lock')
                compare.save(rejected, {'untouched': True})
                with mock.patch.object(t09, 'formal_path'), mock.patch.object(isolate, 'case_processes', return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, 'both original locations restored'):
                        isolate.coordinate({}, call)
                self.assertFalse((Path(payloads[0]['root_out']) / 'compare.lock').exists())
                self.assertEqual(t09.read(rejected), {'untouched': True})
                if rejection == 'completed':
                    self.assertFalse((peer / 'compare.lock').exists())
                for payload in payloads:
                    self.assertTrue(Path(payload['source']).is_dir())
                    self.assertFalse(Path(payload['target']).exists())
                rejected.unlink()

    def test_lost_reply_after_prepare_or_move_restores_real_directories(self):
        for failed_action in ('prepare', 'move'):
            with self.subTest(failed_action=failed_action):
                payloads = self.node_fixtures()
                originals = [Path(p['source'], 'request.json').read_bytes() for p in payloads]
                def call(payload, rank, action):
                    with mock.patch.object(isolate.socket, 'gethostname', return_value=f'SL306{rank}'):
                        receipt = isolate.node_action(payloads[rank], rank, action)
                    if rank == 1 and action == failed_action:
                        raise RuntimeError('reply lost after successful filesystem operation')
                    return receipt
                with mock.patch.object(t09, 'formal_path'), mock.patch.object(isolate, 'case_processes', return_value=[]):
                    with self.assertRaisesRegex(RuntimeError, 'both original locations restored'):
                        isolate.coordinate({}, call)
                for payload, raw in zip(payloads, originals):
                    self.assertEqual(Path(payload['source'], 'request.json').read_bytes(), raw)
                    self.assertFalse(Path(payload['target']).exists())
                    self.assertFalse(Path(payload['root_out'], 'compare.lock').exists())

    def test_node_rename_records_reason_and_rollback_preserves_bytes(self):
        fixture = control.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        request = fixture.requests[0]
        root = Path(fixture.settings['root_out'])
        source = root / 'fixed/r1'
        source.mkdir(parents=True)
        request['out_dir'] = str(source)
        compare.save(source / 'request.json', request)
        (source / 'node0.log').write_bytes(b'original evidence\n')
        payload = {'request': request, 'root_out': str(root), 'work': str(root.parent),
                   'source': str(source), 'target': str(root / '_failed/fixed_r1_fixture'),
                   'transaction': 'fixed_r1_fixture', 'reason': 'operator explanation'}
        with mock.patch.object(t09, 'formal_path'), mock.patch.object(isolate, 'case_processes', return_value=[]), \
                mock.patch.object(isolate.socket, 'gethostname', return_value='SL3060'):
            for action in ('prepare', 'move', 'verify'):
                isolate.node_action(payload, 0, action)
            target = Path(payload['target'])
            self.assertFalse(source.exists())
            self.assertEqual((target / 'node0.log').read_bytes(), b'original evidence\n')
            self.assertEqual(t09.read(target / 'isolation.fixed_r1_fixture.json')['reason'], 'operator explanation')
            for action in ('rollback', 'unlock'):
                isolate.node_action(payload, 0, action)
            self.assertTrue(source.is_dir())
            self.assertFalse((root / 'compare.lock').exists())

    def test_node_refuses_residual_process_and_completed_evidence(self):
        fixture = control.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        request = fixture.requests[0]
        root = Path(fixture.settings['root_out'])
        source = root / 'fixed/r1'
        source.mkdir(parents=True)
        payload = {'request': request, 'root_out': str(root), 'work': str(root.parent),
                   'source': str(source), 'target': str(root / '_failed/fixed_r1_fixture'),
                   'transaction': 'fixed_r1_fixture', 'reason': 'reason'}
        with mock.patch.object(t09, 'formal_path'), mock.patch.object(isolate, 'case_processes', return_value=[123]) as procs, \
                mock.patch.object(isolate.socket, 'gethostname', return_value='SL3060'):
            with self.assertRaisesRegex(ValueError, 'tagged processes'):
                isolate.node_action(payload, 0, 'prepare')
            procs.return_value = []
            (source / 'complete.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'completed case'):
                isolate.node_action(payload, 0, 'prepare')
            self.assertFalse((root / 'compare.lock').exists())

    def test_process_match_is_exact_including_supervisor(self):
        fixture = control.CompareTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        proc = fixture.path / 'proc'
        out = '/data/work/formal/batch/weavetp/r1'
        for pid, argument in ((991001, out), (991002, out + '0'), (991003, out + '/result.json')):
            entry = proc / str(pid)
            entry.mkdir(parents=True)
            (entry / 'cmdline').write_bytes(('python\0compare_weavetp_16gpu.py\0' + argument + '\0').encode())
        self.assertEqual(sorted(isolate.case_processes(out, proc)), [991001, 991003])

    def test_both_prepare_before_move_and_verify_before_unlock(self):
        calls = []
        isolate.coordinate({'target': 'fixture', 'reason': 'CPU fixture'}, lambda p, r, a: calls.append((r, a)))
        self.assertEqual(calls, [(0, 'prepare'), (1, 'prepare'), (0, 'move'), (1, 'move'),
                                 (0, 'verify'), (1, 'verify'), (1, 'unlock'), (0, 'unlock')])

    def test_second_move_failure_restores_both_before_unlock(self):
        calls, moved = [], set()
        def call(p, rank, action):
            calls.append((rank, action))
            if action == 'move':
                if rank == 1:
                    raise RuntimeError('fixture second-node failure')
                moved.add(rank)
            if action == 'rollback':
                moved.discard(rank)
                return {'locked': True}
        with self.assertRaisesRegex(RuntimeError, 'both original locations restored'):
            isolate.coordinate({}, call)
        self.assertEqual(moved, set())
        self.assertEqual(calls[-4:], [(0, 'rollback'), (1, 'rollback'), (1, 'unlock'), (0, 'unlock')])

    def test_lost_move_reply_and_unreachable_rollback_preserve_locks(self):
        calls = []
        def call(p, rank, action):
            calls.append((rank, action))
            if rank == 1 and action in ('move', 'rollback'):
                raise RuntimeError('unreachable')
            if action == 'rollback':
                return {'locked': True}
        with self.assertRaisesRegex(RuntimeError, 'UNCONFIRMED'):
            isolate.coordinate({}, call)
        self.assertNotIn('unlock', [a for r, a in calls])

    def test_peer_unlock_failure_keeps_controller_locked(self):
        calls = []
        def call(payload, rank, action):
            calls.append((rank, action))
            if (rank, action) == (1, 'unlock'):
                raise RuntimeError('peer unlock unconfirmed')
        with self.assertRaisesRegex(RuntimeError, 'peer unlock unconfirmed'):
            isolate.coordinate({}, call)
        self.assertNotIn((0, 'unlock'), calls)


if __name__ == '__main__':
    unittest.main(verbosity=2)
