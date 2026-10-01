"""CPU regressions for the actual T08 functions; no SSH, torch or GPU execution."""
import contextlib
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import t08

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASH = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else '/bin/bash'
COMMIT = 'a' * 40


class NodeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=HERE)
        self.addCleanup(temp.cleanup)
        self.run = Path(temp.name).resolve()
        self.env = {key: 'fixture' for key in t08.SHARED_KEYS}
        self.env.update(WORK=str(self.run), CODE_DIR=str(ROOT / 'code/current'),
                        MASTER_PORT='29500', WEAVETP_COMMIT=COMMIT)
        self.compare = mock.Mock()
        self.compare.gpu_state.return_value = {'gpu0': 65}
        self.compare.gpu_memory.side_effect = lambda value: value
        self.compare.tagged_processes.return_value = []
        self.compare.git_identity.return_value = COMMIT
        mutex = threading.Lock()
        @contextlib.contextmanager
        def lock(run):
            with mutex:
                yield
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(t08, 'launch_lock', lock).start()
        mock.patch.object(t08.time, 'sleep').start()
        mock.patch.object(t08.signal, 'SIGKILL', getattr(t08.signal, 'SIGKILL', 9), create=True).start()

    def make_out(self):
        (self.run / 'measurement').mkdir()
        t08.save(self.run / 'launch_gpu.node0.json', {'gpu0': 65})

    def test_cancel_before_start_prevents_popen(self):
        self.assertTrue(t08.stop_profile(self.run, 0, self.compare)['ok'])
        with mock.patch.object(t08.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'cancelled'):
                t08.run_profile(self.run, 0, self.env, self.compare)
        popen.assert_not_called()
        self.assertFalse((self.run / 'measurement').exists())

    def test_cleanup_confirmation_uses_live_state_not_exit_receipt(self):
        self.make_out()
        self.assertTrue(t08.stop_profile(self.run, 0, self.compare)['ok'])
        self.assertFalse((self.run / 'exit.node0.json').exists())
        self.assertFalse((self.run / 'measurement/prepared.node0.json').exists())

    def test_cleanup_rejects_remaining_pid_or_unrestored_memory(self):
        self.make_out()
        for pids, memory in (([123], 65), ([], 200)):
            with self.subTest(pids=pids, memory=memory), mock.patch.object(t08.os, 'kill'):
                self.compare.tagged_processes.return_value = pids
                self.compare.gpu_state.return_value = {'gpu0': memory}
                self.assertFalse(t08.stop_profile(self.run, 0, self.compare)['ok'])

    def test_cleanup_observation_failure_is_unconfirmed(self):
        self.make_out()
        self.compare.gpu_state.side_effect = RuntimeError('nvidia-smi unavailable')
        result = t08.stop_profile(self.run, 0, self.compare)
        self.assertFalse(result['ok'])
        self.assertIn('unavailable', result['error'])

    def test_interruption_and_timeout_reap_a_real_cpu_child(self):
        real_popen = subprocess.Popen
        for reason in (KeyboardInterrupt('operator'), subprocess.TimeoutExpired('fixture', 3600)):
            with self.subTest(reason=type(reason).__name__):
                run = self.run / type(reason).__name__
                run.mkdir()
                children = []
                # The wrapper interrupts only the first wait; subsequent waits
                # and process liveness are those of the actual CPU subprocess.
                def spawn(command, **kwargs):
                    child = real_popen(command, **kwargs)
                    children.append(child)
                    waited = False
                    def wait(timeout):
                        nonlocal waited
                        if not waited:
                            waited = True
                            raise reason
                        return child.wait(timeout=timeout)
                    return SimpleNamespace(wait=wait)
                def kill(pid, sig):
                    self.assertEqual(pid, children[0].pid)
                    children[0].kill()
                    children[0].wait(timeout=5)
                command = [sys.executable, '-B', '-c', 'import time; time.sleep(30)']
                self.compare.tagged_processes.side_effect = lambda out: [c.pid for c in children if c.poll() is None]
                with mock.patch.object(t08, 'profile_command', return_value=command), \
                        mock.patch.object(t08.subprocess, 'Popen', side_effect=spawn), \
                        mock.patch.object(t08.os, 'kill', side_effect=kill):
                    with self.assertRaises(type(reason)):
                        t08.run_profile(run, 0, self.env, self.compare)
                self.assertIsNotNone(children[0].poll())
                receipt = t08.read(run / 'exit.node0.json')
                self.assertTrue(receipt['cleanup']['ok'])
                self.assertIsNotNone(receipt['error'])

    def test_cancellation_waits_for_popen_publication(self):
        entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
        stopped = threading.Event()
        failures = []
        class Child:
            def wait(self, timeout):
                if not stopped.wait(5):
                    raise RuntimeError('cancel did not stop child')
                return 1
        def spawn(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError('test release missing')
            return Child()
        def runner():
            try:
                t08.run_profile(self.run, 0, self.env, self.compare)
            except ValueError:
                pass  # Expected terminated child; cleanup is still executed.
            except BaseException as exc:
                failures.append(str(exc))
        def cancel():
            try:
                self.assertTrue(t08.stop_profile(self.run, 0, self.compare)['ok'])
                cancelled.set()
            except BaseException as exc:
                failures.append(str(exc))
        self.compare.tagged_processes.side_effect = lambda out: [] if stopped.is_set() else [123]
        with mock.patch.object(t08, 'profile_command', return_value=['CPU fake']), \
                mock.patch.object(t08, 'defer_launch_interrupts', contextlib.nullcontext), \
                mock.patch.object(t08.subprocess, 'Popen', side_effect=spawn), \
                mock.patch.object(t08.os, 'kill', side_effect=lambda pid, sig: stopped.set()):
            a = threading.Thread(target=runner)
            a.start()
            self.assertTrue(entered.wait(5))
            b = threading.Thread(target=cancel)
            b.start()
            try:
                self.assertFalse(cancelled.wait(0.05))
            finally:
                release.set()
                a.join(5)
                b.join(5)
        self.assertFalse(a.is_alive() or b.is_alive())
        self.assertTrue(cancelled.is_set())
        self.assertEqual(failures, [])

    def test_signal_during_popen_does_not_lose_the_child_handle(self):
        child = mock.Mock(wait=mock.Mock(return_value=1))
        def spawn(*args, **kwargs):
            handler = t08.signal.getsignal(t08.signal.SIGTERM)
            handler(t08.signal.SIGTERM, None)
            return child
        with mock.patch.object(t08, 'profile_command', return_value=['CPU fake']), \
                mock.patch.object(t08.subprocess, 'Popen', side_effect=spawn):
            with self.assertRaisesRegex(KeyboardInterrupt, 'during launch'):
                t08.run_profile(self.run, 0, self.env, self.compare)
        child.wait.assert_called_once_with(timeout=15)
        self.assertEqual(t08.read(self.run / 'exit.node0.json')['exit_code'], 1)

    def test_run_uses_frozen_environment(self):
        t08.save(self.run / 'deployment.json', {'node_rank': 0, 'env': self.env})
        with mock.patch.dict(os.environ, MASTER_PORT='29999', CODE_DIR='changed'), \
                mock.patch.object(t08.socket, 'gethostname', return_value='SL3060'), \
                mock.patch.object(Path, 'is_relative_to', return_value=True), \
                mock.patch.object(t08.shutil, 'disk_usage', return_value=SimpleNamespace(free=20_000_000_000)), \
                mock.patch.dict(sys.modules, compare_weavetp_16gpu=self.compare, profile_weavetp_16gpu=mock.Mock()), \
                mock.patch.object(t08, 'run_profile') as launch:
            t08.node('run', 0, self.run)
        self.assertEqual(launch.call_args.args[2]['MASTER_PORT'], '29500')
        self.assertEqual(launch.call_args.args[2]['CODE_DIR'], self.env['CODE_DIR'])

    def test_low_disk_and_dirty_commit_prevent_launch(self):
        for free, commit, pin in ((19_999_999_999, COMMIT, COMMIT),
                                  (20_000_000_000, 'b' * 40, COMMIT),
                                  (20_000_000_000, 'short', 'short')):
            self.env['WEAVETP_COMMIT'] = pin
            (self.run / 'deployment.json').write_text(json.dumps({'node_rank': 0, 'env': self.env}))
            with self.subTest(free=free, commit=commit), \
                    mock.patch.object(t08.socket, 'gethostname', return_value='SL3060'), \
                    mock.patch.object(Path, 'is_relative_to', return_value=True), \
                    mock.patch.object(t08.shutil, 'disk_usage', return_value=SimpleNamespace(free=free)), \
                    mock.patch.dict(sys.modules, compare_weavetp_16gpu=self.compare, profile_weavetp_16gpu=mock.Mock()), \
                    mock.patch.object(t08, 'run_profile') as launch:
                self.compare.git_identity.return_value = commit
                with self.assertRaises(ValueError):
                    t08.node('run', 0, self.run)
                launch.assert_not_called()

    def test_busy_gpu_prevents_launch(self):
        self.compare.check_idle.side_effect = ValueError('GPU occupied')
        with mock.patch.object(t08.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(ValueError, 'occupied'):
                t08.run_profile(self.run, 0, self.env, self.compare)
        popen.assert_not_called()

    def test_actual_launcher_dry_run_preserves_info_rdma_and_task_tag(self):
        real_output = subprocess.check_output
        env = dict(os.environ, CODE_DIR=str(ROOT / 'code/current'), NODE_RANK='0',
                   MASTER_ADDR='192.0.2.10', MASTER_PORT='29500', PYTHON='/data/env/bin/python',
                   NCCL_DEBUG='WARN', NCCL_NET='Socket', NCCL_IB_GID_INDEX='3', MSYS_NO_PATHCONV='1')
        def dry_run(args, **kwargs):
            return real_output([BASH, *args[1:]], **kwargs)
        with mock.patch.object(t08.subprocess, 'check_output', side_effect=dry_run), \
                mock.patch.object(Path, 'mkdir'):
            command = t08.profile_command(env, PurePosixPath('/data/t08_test/measurement'))
        self.assertEqual(command[:2], ['env', '-i'])
        for token in ('NCCL_DEBUG=INFO', 'NCCL_IB_HCA=mlx5_0', 'NCCL_IB_DISABLE=0',
                      '--nnodes=2', '--nproc_per_node=8', '--max_restarts=0', '--out-dir',
                      '/data/t08_test/measurement'):
            self.assertIn(token, command)
        self.assertFalse(any(token.startswith(('NCCL_NET=', 'NCCL_IB_GID_INDEX=')) for token in command))

    def test_node_cli_accepts_the_streamed_source_arguments(self):
        code = ('import runpy,socket,sys; socket.gethostname=lambda: "cpu-fixture"; '
                'sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")')
        result = subprocess.run([sys.executable, '-B', '-c', code, str(HERE / 't08.py'),
                                 'node', 'cleanup', '0', '/data/t08_fixture', ''],
                                capture_output=True, text=True, timeout=15)
        # Explicit fake hostname fails before any server writes, on any host.
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('invalid T08 command', result.stderr)
        self.assertIn('wrong host', result.stderr)

    def test_profile_verification_checks_both_hash_and_actual_transport(self):
        sys.path.insert(0, str(ROOT / 'code/current/tests/unit_tests/resharding'))
        from test_profile_weavetp_16gpu import fixture, profile
        for case in ('valid', 'socket', 'hash-mismatch'):
            with self.subTest(case=case):
                run = self.run / case
                out = run / 'measurement'
                out.mkdir(parents=True)
                t08.save(run / 'deployment.json', {'node_rank': 0, 'env': self.env})
                t08.save(run / 'exit.node0.json', {'exit_code': 0, 'error': None})
                t08.save(run / 'launch_gpu.node0.json', {'gpu0': 65})
                t08.save(out / 'profile.json', fixture())
                digest = hashlib.sha256((out / 'profile.json').read_bytes()).hexdigest()
                for rank in range(8):
                    network = 'Socket' if case == 'socket' and rank == 7 else 'IB'
                    (out / f'nccl.rank{rank}.log').write_text(f'NCCL INFO Channel 00 : 0 -> 8 via NET/{network}/0\n')
                output = io.StringIO()
                with mock.patch.object(t08.socket, 'gethostname', return_value='SL3060'), \
                        mock.patch.object(Path, 'is_relative_to', return_value=True), \
                        mock.patch.dict(sys.modules, compare_weavetp_16gpu=self.compare, profile_weavetp_16gpu=profile), \
                        contextlib.redirect_stdout(output):
                    if case == 'valid':
                        t08.node('verify', 0, run, digest)
                        self.assertIn('NET/IB matrix=16x16 pairs=240', output.getvalue())
                        self.assertEqual((out / 'profile.json.sha256').read_text(), f'{digest}  profile.json\n')
                    else:
                        with self.assertRaises(ValueError):
                            t08.node('verify', 0, run, '0' * 64 if case == 'hash-mismatch' else digest)
                        self.assertNotIn('T08_PROFILE_OK', output.getvalue())


class ControllerTests(unittest.TestCase):
    def test_wait_pair_notices_failure_without_waiting_for_peer(self):
        running = mock.Mock(poll=mock.Mock(return_value=None))
        failed = mock.Mock(poll=mock.Mock(return_value=7))
        with self.assertRaisesRegex(ValueError, 'node1 RPC exit=7'):
            t08.wait_pair({0: running, 1: failed})
        running.wait.assert_not_called()

    def test_wait_pair_timeout(self):
        with mock.patch.object(t08.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaisesRegex(ValueError, 'timed out'):
                t08.wait_pair({0: mock.Mock(poll=mock.Mock(return_value=None))}, timeout=1)

    def test_controller_scenarios(self):
        for scenario in ('success', 'preflight:0', 'preflight:1', 'run:0', 'run:1',
                         'verify:1', 'checkpoint-mismatch', 'settings-mismatch', 'cleanup-unreachable'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory(dir=HERE) as temporary:
                root = Path(temporary).resolve()
                events, children = [], []
                env = {key: 'fixture' for key in t08.SHARED_KEYS}
                env.update(WORK=str(root), NODE1_SSH='fixture-peer')
                def remote(path):
                    path = Path(path)
                    return root / 'peer' / path.relative_to(root)
                class RPC:
                    def __init__(self, args, stdin):
                        if args[0] == 'ssh':
                            shell = shlex.split(args[-1])[-1]
                            args = shlex.split(shell.split('exec ', 1)[1])
                        pos = args.index('node')
                        self.phase, self.rank, self.run = args[pos+1], int(args[pos+2]), Path(args[pos+3])
                        events.append((self.phase, self.rank))
                        children.append(self)
                        self.status = 0
                        if scenario == f'{self.phase}:{self.rank}':
                            self.status = 7
                        if scenario == 'cleanup-unreachable':
                            if (self.phase, self.rank) == ('run', 0):
                                self.status = 7
                            elif (self.phase, self.rank) == ('run', 1):
                                self.status = None
                            elif (self.phase, self.rank) == ('cleanup', 1):
                                self.status = 8
                        out = self.run if self.rank == 0 else remote(self.run)
                        if self.phase == 'prepare':
                            (out / 'checkpoint').mkdir(parents=True)
                            snapshot = dict(env)
                            if scenario == 'settings-mismatch' and self.rank == 1:
                                snapshot['MASTER_PORT'] = 'other'
                            t08.save(out / 'deployment.json', {'node_rank': self.rank, 'env': snapshot})
                            files = {'weights': 'changed' if scenario == 'checkpoint-mismatch' and self.rank else 'same'}
                            t08.save(out / 'checkpoint/acceptance.json', {'status': 'passed', 'details': {
                                'hostname': ('sl3060', 'sl3061')[self.rank], 'files': files}})
                        if self.phase == 'run':
                            (out / 'measurement').mkdir()
                            if self.rank == 0:
                                (out / 'measurement/profile.json').write_text('{}')
                    def poll(self):
                        return self.status
                    def wait(self, timeout):
                        if self.status is None:
                            raise subprocess.TimeoutExpired('fixture RPC', timeout)
                        return self.status
                    def kill(self):
                        self.status = -9
                def scp(args, **kwargs):
                    src, dst = args[-2:]
                    src = remote(src.split(':', 1)[1]) if src.startswith('fixture-peer:') else Path(src)
                    dst = remote(dst.split(':', 1)[1]) if dst.startswith('fixture-peer:') else Path(dst)
                    dst.write_bytes(src.read_bytes())
                output = io.StringIO()
                with mock.patch.dict(os.environ, env), \
                        mock.patch.object(t08.socket, 'gethostname', return_value='SL3060'), \
                        mock.patch.object(Path, 'is_relative_to', return_value=True), \
                        mock.patch.object(t08.subprocess, 'Popen', side_effect=RPC), \
                        mock.patch.object(t08.subprocess, 'run', side_effect=scp), \
                        contextlib.redirect_stdout(output):
                    if scenario == 'success':
                        t08.controller()
                        self.assertIn('T08_RESULT exit=0', output.getvalue())
                        self.assertEqual(events[-2:], [('complete', 0), ('complete', 1)])
                        self.assertLess(events.index(('preflight', 1)), events.index(('run', 0)))
                    else:
                        with self.assertRaises(ValueError):
                            t08.controller()
                        self.assertNotIn('T08_RESULT exit=0', output.getvalue())
                        self.assertIn(('cleanup', 0), events)
                        self.assertIn(('cleanup', 1), events)
                        self.assertFalse(any(phase == 'complete' for phase, _ in events))
                        if scenario.startswith('preflight') or scenario.endswith('mismatch'):
                            self.assertFalse(any(phase == 'run' for phase, _ in events))
                        if scenario == 'cleanup-unreachable':
                            self.assertIn('cleanup 未确认', output.getvalue())
                self.assertTrue(all(child.poll() is not None for child in children))


if __name__ == '__main__':
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__]))
    print(output.getvalue())
    records = [{'name': 'T08 regressions', 'tests': result.testsRun, 'exit_code': int(not result.wasSuccessful()),
                'output': output.getvalue()}]
    for name, command in (
        ('bash-n', [BASH, '-n', str(HERE / 'run_t08.sh')]),
        ('profile tests', [sys.executable, '-B', '-X', 'utf8', str(ROOT / 'code/current/tests/unit_tests/resharding/test_profile_weavetp_16gpu.py')]),
    ):
        completed = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=60,
                                   env=dict(os.environ, PYTHONUTF8='1'))
        records.append({'name': name, 'command': command, 'exit_code': completed.returncode,
                        'stdout': completed.stdout, 'stderr': completed.stderr})
    source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ('t08.py', 'run_t08.sh', 'check_commands.py')}
    (HERE / 'local_validation.json').write_text(json.dumps({'gpu_started': False, 'ssh_executed': False,
        'source_hashes': source_hashes, 'checks': records}, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    sys.exit(0 if all(row['exit_code'] == 0 for row in records) else 1)
