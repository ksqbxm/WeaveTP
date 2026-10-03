"""T08 controller and node operations; reuse the verified T07 profiler unchanged."""
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ENV_FILE = '/data/ubuntu/lxh/weavetp/env.sh'
MIN_FREE_BYTES = 5_000_000_000
SHARED_KEYS = ('WORK', 'REPO', 'CODE_DIR', 'NODE1_CODE_DIR', 'MASTER_ADDR', 'MASTER_PORT',
               'NODE0_PYTHON', 'NODE1_PYTHON', 'CHECKPOINT', 'WEAVETP_COMMIT')
CACHE_KEYS = ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME', 'CUDA_CACHE_PATH', 'TORCH_HOME',
              'TORCH_EXTENSIONS_DIR', 'TRITON_CACHE_DIR', 'HF_HOME', 'HUGGINGFACE_HUB_CACHE',
              'TRANSFORMERS_CACHE', 'PIP_CACHE_DIR')
SSH_OPTIONS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=3']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


@contextlib.contextmanager
def launch_lock(run):
    # Linux advisory lock shared by launch and cancellation.
    import fcntl
    with (run / 'launch.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def defer_launch_interrupts():
    # Do not lose a just-forked child to SIGINT/SIGTERM before Popen returns.
    pending = []
    previous = {sig: signal.signal(sig, lambda signum, frame: pending.append(signum))
                for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if pending:
        raise KeyboardInterrupt(f'signal {pending[0]} during launch')


def report(env, rank, run, message):
    path = Path(env['WORK']) / 'reports/trial_report.md'
    path.parent.mkdir(exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(f'\nT08 node={rank} {message}; evidence={run}\n')


def stop_profile(run, rank, compare):
    """Cancel future launch first, then stop only exact-tagged task processes."""
    with launch_lock(run):
        (run / 'stop_requested').touch()
        out = run / 'measurement'
        result = {'ok': True, 'status': 'no launch started', 'remaining_pids': []}
        try:
            if out.exists():
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    for pid in compare.tagged_processes(str(out)):
                        try:
                            os.kill(pid, sig)
                        except ProcessLookupError:
                            pass
                    time.sleep(2)
                remaining = compare.tagged_processes(str(out))
                after = compare.gpu_state()
                baseline = compare.gpu_memory(read(run / f'launch_gpu.node{rank}.json'))
                current = compare.gpu_memory(after)
                restored = current.keys() == baseline.keys() and all(
                    current[key] <= baseline[key] + 16 for key in baseline)
                result = {'ok': not remaining and restored, 'remaining_pids': remaining,
                          'memory_restored': restored, 'gpu_after': after}
                result['status'] = '已确认' if result['ok'] else '未确认'
        except Exception as exc:
            result = {'ok': False, 'status': '未确认', 'error': str(exc)}
        save(run / f'cleanup.node{rank}.{time.time_ns()}.json', result)
        return result


def profile_command(env, out):
    """Use the production environment/argv without a shell/timeout parent."""
    launcher = Path(env['CODE_DIR']) / 'tools/resharding/run_weavetp_16gpu_profile.sh'
    raw = subprocess.check_output(['bash', str(launcher), '--dry-run'],
                                  env=dict(env, OUT_DIR=str(out)), text=True, timeout=30)
    command = shlex.split(raw)
    for argument in command:
        key, _, value = argument.partition('=')
        if key in CACHE_KEYS:
            Path(value).mkdir(parents=True, exist_ok=True)
    return command


def run_profile(run, rank, env, compare):
    out = run / 'measurement'
    state = compare.gpu_state()
    compare.check_idle(state, rank, env.get('MPS_OWNER'))
    receipt = {'exit_code': None, 'error': None}
    process = None
    try:
        with defer_launch_interrupts(), launch_lock(run):
            require(not (run / 'stop_requested').exists(), 'cancelled before launch')
            save(run / f'launch_gpu.node{rank}.json', state)
            out.mkdir()
            command = profile_command(env, out)
            save(run / f'command.node{rank}.json', command)
            with (run / f'launcher.node{rank}.log').open('x') as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        # At lock release Popen has completed exec. The live launcher already
        # has profile_weavetp_16gpu.py and the exact OUT_DIR in its argv.
        receipt['exit_code'] = process.wait(timeout=3600)
        require(receipt['exit_code'] == 0, f'profile exit={receipt["exit_code"]}')
    except BaseException as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
        receipt['cleanup'] = stop_profile(run, rank, compare)
        if process is not None:
            try:
                receipt['exit_code'] = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                receipt['error'] += '; launcher exit unconfirmed'
        raise
    finally:
        save(run / f'exit.node{rank}.json', receipt)
    print(f'T08_NODE_EXIT node={rank} exit=0', flush=True)


def node(phase, rank, run, expected_hash=''):
    require(rank in (0, 1), 'invalid node rank')
    require(socket.gethostname().split('.')[0].lower() == ('sl3060', 'sl3061')[rank], 'wrong host')
    require(run == run.resolve() and run.is_relative_to('/data'), 'noncanonical /data path')
    if phase == 'cleanup' and not run.exists():
        print(f'T08_CLEANUP node={rank} no task directory; no launch')
        return
    if phase == 'prepare':
        env = {key: os.environ[key] for key in (*SHARED_KEYS, 'NODE_RANK', 'PYTHON', 'HOME')
               if key != 'CHECKPOINT'}
        env['CHECKPOINT'] = '/data/models/DeepSeek-V2-Lite-megatron-v2'
        env.update({key: os.environ[key] for key in ('USER', 'LOGNAME', 'LD_LIBRARY_PATH', 'MPS_OWNER')
                    if key in os.environ})
        require(env['NODE_RANK'] == str(rank), 'wrong NODE_RANK in env.sh')
        require(env['PYTHON'] == env[('NODE0_PYTHON', 'NODE1_PYTHON')[rank]] == sys.executable,
                'wrong existing interpreter')
        work, code = Path(env['WORK']), Path(env['REPO']) / 'code/current'
        require(work == work.resolve() and work.is_relative_to('/data'), 'WORK must be under /data')
        require(code.resolve().is_relative_to(work) and str(code) == env['CODE_DIR'] == env['NODE1_CODE_DIR'],
                'code must be the verified WORK/REPO/code/current')
        require(run.parent == work / 'profiles' and run.name.startswith('t08_'), 'unexpected task path')
        env.update(PATH=f"{Path(sys.executable).parent}:/usr/bin:/bin",
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
        env.update({key: str(run / 'cache' / key.lower()) for key in CACHE_KEYS})
    else:
        # All later operations use this sole configuration, never a new env.sh.
        env = read(run / 'deployment.json')['env']
    sys.path.insert(0, str(Path(env['CODE_DIR']) / 'tools/resharding'))
    import compare_weavetp_16gpu as compare
    import profile_weavetp_16gpu as profile

    if phase == 'cleanup':
        result = stop_profile(run, rank, compare)
        report(env, rank, run, f'failed/interrupted; cleanup={result["status"]}')
        print(json.dumps(result, ensure_ascii=False), flush=True)
        require(result['ok'], 'cleanup 未确认')
        return
    if phase in ('prepare', 'preflight', 'run'):
        commit = env['WEAVETP_COMMIT']
        require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
                'WEAVETP_COMMIT must be a full lowercase Git SHA')
        require(compare.git_identity(env['CODE_DIR']) == commit,
                'wrong commit or dirty Git checkout')
        require(shutil.disk_usage('/data').free >= MIN_FREE_BYTES, '/data needs >=5 GB; STOP')
    tests = Path(env['CODE_DIR']) / 'tests/server_tests/weavetp_16gpu'
    if phase == 'prepare':
        run.mkdir(parents=True)
        save(run / 'deployment.json', {'node_rank': rank, 'env': env})
        for key in CACHE_KEYS:
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-B', str(tests / 'test_checkpoint.py'),
                        '--checkpoint', env['CHECKPOINT'], '--output-dir', str(run / 'checkpoint')],
                       env=env, check=True, timeout=1500)
        print(f'T08_PREPARED node={rank} commit={commit} GPU_STARTED=0', flush=True)
    elif phase == 'preflight':
        addresses = json.loads(subprocess.check_output(
            ['ip', '-j', '-4', 'addr', 'show', 'dev', 'eno1np0'], text=True, timeout=15))
        ips = [a['local'] for interface in addresses for a in interface['addr_info']
               if a['family'] == 'inet' and a['scope'] == 'global']
        require(len(ips) == 1, 'expected one IPv4 on eno1np0')
        if rank == 0:
            require(ips[0] == env['MASTER_ADDR'], 'master must use eno1np0 IPv4')
            with socket.socket() as probe:
                probe.bind((env['MASTER_ADDR'], int(env['MASTER_PORT'])))
        subprocess.run([sys.executable, '-B', str(tests / 'test_node.py'), '--node-rank', str(rank),
                        '--expected-gid', ips[0], '--min-free-gib', str(MIN_FREE_BYTES / (1 << 30)),
                        '--mps-owner', env.get('MPS_OWNER', ''), '--checkpoint', env['CHECKPOINT'],
                        '--require-idle', '--output-dir', str(run / 'preflight')],
                       env=env, check=True, timeout=150)
        report(env, rank, run, f'deployment passed; commit={commit}; >=5 GB; GPUs idle')
        print(f'T08_PREFLIGHT_OK node={rank} >=5GB GPUs=8 idle', flush=True)
    elif phase == 'run':
        run_profile(run, rank, env, compare)
    elif phase == 'verify':
        receipt = read(run / f'exit.node{rank}.json')
        require(receipt['exit_code'] == 0 and receipt['error'] is None, 'node launch did not succeed')
        out = run / 'measurement'
        _, digest = profile.load_profile(out / 'profile.json', expected_hash)
        logs = sorted(out.glob('nccl.*.log'))
        require(len(logs) == 8, f'expected 8 original NCCL logs, got {len(logs)}')
        for log in logs:
            profile.network_evidence(log)
        print(next(line for line in logs[0].read_text(errors='replace').splitlines()
                   if 'via NET/IB' in line), flush=True)
        after = compare.gpu_state()
        baseline = compare.gpu_memory(read(run / f'launch_gpu.node{rank}.json'))
        current = compare.gpu_memory(after)
        remaining = compare.tagged_processes(str(out))
        save(run / f'after.node{rank}.json', {'gpu_state': after, 'remaining_pids': remaining})
        require(not remaining and current.keys() == baseline.keys() and all(
            current[key] <= baseline[key] + 16 for key in baseline), 'task processes/memory not restored')
        with (out / 'nccl.sha256').open('x') as stream:
            for log in logs:
                stream.write(f'{hashlib.sha256(log.read_bytes()).hexdigest()}  {log.name}\n')
        with (out / 'profile.json.sha256').open('x') as stream:
            stream.write(f'{digest}  profile.json\n')
        print(f'T08_PROFILE_OK node={rank} NET/IB matrix=16x16 pairs=240 sha256={digest}', flush=True)
    elif phase == 'complete':
        report(env, rank, run, f'profile passed; NET/IB; matrix=16x16; sha256={expected_hash}; '
               'processes exited; GPU memory restored; formal NCCL_DEBUG=WARN')
    else:
        raise ValueError(f'unknown phase: {phase}')


def match_deployment(run):
    a, b = read(run / 'checkpoint/acceptance.json'), read(run / 'checkpoint.node1.json')
    require(a['status'] == b['status'] == 'passed', 'checkpoint check failed')
    require(a['details']['hostname'] == 'sl3060' and b['details']['hostname'] == 'sl3061', 'wrong peer')
    require(a['details']['files'] == b['details']['files'], 'checkpoint contents differ')
    local, peer = read(run / 'deployment.json'), read(run / 'deployment.node1.json')
    require(local['node_rank'] == 0 and peer['node_rank'] == 1, 'wrong node rank')
    require({k: local['env'][k] for k in SHARED_KEYS} == {k: peer['env'][k] for k in SHARED_KEYS},
            'two-node configuration differs')


def wait_pair(processes, timeout=3700):
    deadline = time.monotonic() + timeout
    pending = dict(processes)
    while pending:
        for rank, process in list(pending.items()):
            status = process.poll()
            if status is not None:
                require(status == 0, f'node{rank} RPC exit={status}')
                del pending[rank]
        require(time.monotonic() < deadline, 'two-node launch timed out')
        if pending:
            time.sleep(0.2)


def controller():
    require(socket.gethostname().split('.')[0].lower() == 'sl3060', 'controller requires SL3060')
    env = os.environ.copy()
    work = Path(env['WORK'])
    require(work == work.resolve() and work.is_relative_to('/data'), 'WORK must be under /data')
    run = work / 'profiles' / f't08_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{os.getpid()}'
    source = Path(__file__).resolve()
    peer = env['NODE1_SSH']
    processes = {}

    def start(phase, rank, digest=''):
        args = [env[('NODE0_PYTHON', 'NODE1_PYTHON')[rank]], '-B', '-X', 'utf8', '-',
                'node', phase, str(rank), str(run), digest]
        if rank == 1:
            shell = f'source {shlex.quote(ENV_FILE)}; ' if phase == 'prepare' else ''
            shell += 'exec ' + shlex.join(args)
            args = ['ssh', *SSH_OPTIONS, peer, shlex.join(['bash', '-euc', shell])]
        with source.open('rb') as script:
            return subprocess.Popen(args, stdin=script)

    def call(phase, rank, digest=''):
        process = start(phase, rank, digest)
        try:
            status = process.wait(timeout=1600 if phase == 'prepare' else 180)
            require(status == 0, f'node{rank} {phase} exit={status}')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    def copy(src, dst):
        subprocess.run(['scp', *SSH_OPTIONS, src, dst], check=True, timeout=90)

    print(f'T08_RUN={run}', flush=True)
    try:
        print('[1/3] Checkpoint hashes and deployment (5-20 min estimate)', flush=True)
        for rank in (0, 1):
            call('prepare', rank)
        copy(f'{peer}:{run}/checkpoint/acceptance.json', str(run / 'checkpoint.node1.json'))
        copy(f'{peer}:{run}/deployment.json', str(run / 'deployment.node1.json'))
        match_deployment(run)
        print('T08_DEPLOY_PAIR_OK GPU_STARTED=0', flush=True)
        print('[2/3] Both-host idle/space checks, then one profile (3-10 min estimate)', flush=True)
        for rank in (0, 1):
            call('preflight', rank)
        for rank in (0, 1):
            processes[rank] = start('run', rank)
        wait_pair(processes)
        print('[3/3] Profile copy, NET/IB, hash and resource verification', flush=True)
        path = run / 'measurement/profile.json'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        copy(str(path), f'{peer}:{path}')
        for rank in (0, 1):
            call('verify', rank, digest)
        for rank in (0, 1):
            call('complete', rank, digest)
        print(f'PROFILE={path}\nPROFILE_SHA256={digest}\nT08_RESULT exit=0 evidence={run}', flush=True)
    except BaseException as exc:
        print(f'STOP: {exc}; no automatic retry', flush=True)
        cleanup = []
        for rank in (0, 1):
            try:
                call('cleanup', rank)
                cleanup.append({'node_rank': rank, 'confirmed': True})
            except Exception as error:
                cleanup.append({'node_rank': rank, 'confirmed': False, 'error': str(error)})
                print(f'node{rank} cleanup 未确认: {error}', flush=True)
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if run.is_dir():
            save(run / 'failure.json', {'error': str(exc), 'cleanup': cleanup})
        print(f'T08_RESULT exit=1 evidence={run}', flush=True)
        raise


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if sys.argv[1:] == ['controller']:
            controller()
        else:
            require(len(sys.argv) == 6 and sys.argv[1] == 'node', 'invalid T08 command')
            node(sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), sys.argv[5])
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
