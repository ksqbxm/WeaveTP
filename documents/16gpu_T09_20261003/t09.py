"""T09 operations: reuse the formal executor, never retry or wait for resources."""

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'code/current/tools/resharding'))
import compare_weavetp_16gpu as compare

# Decimal GB: ceil(max(5, 9 * 0.500 * 2)); see trial_report.md item 5/6.
MIN_FREE_BYTES = 9_000_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def formal_path(root, work):
    path, base = PurePosixPath(root), PurePosixPath(work)
    require(base.is_absolute() and base.is_relative_to('/data') and '..' not in base.parts
            and '..' not in path.parts and path.is_relative_to(base / 'formal')
            and path != base / 'formal' and '_failed' not in path.parts,
            'ROOT_OUT must be a batch below $WORK/formal/')
    compare.check_output_mode(root)
    if os.name != 'nt':
        require(Path(root).resolve() == Path(root) and Path(work).resolve() == Path(work),
                'WORK/ROOT_OUT must be canonical paths without symlinks')


def selected_cases(settings, only='', rounds=''):
    cases = list(compare.make_cases(settings))
    require(rounds in ('', 'r1'), 'ROUNDS supports only r1 (or unset for all rounds)')
    names = {f"{r['case']}_r{r['repeat']}" for r in cases}
    require(not only or only in names, 'ONLY must be <case>_r<1|2|3>')
    require(not (only and rounds), 'use ONLY or ROUNDS, not both')
    return [r for r in cases if (not only or f"{r['case']}_r{r['repeat']}" == only)
            and (not rounds or r['repeat'] == 1)]


def reference(expected_sha):
    data = read(HERE / 'cpu_reference.json')
    require(data['profile_sha256'] == expected_sha,
            'CPU W2 reference belongs to a different profile; recompute before using it')
    return data


def node_preflight(payload, rank, env):
    request, work = payload['request'], payload['work']
    node = request['nodes'][rank]
    require(socket.gethostname().split('.')[0].lower() == node['hostname'].lower(), 'wrong host')
    require(env['NODE_RANK'] == str(rank) and env['WORK'] == work, 'env.sh node/work mismatch')
    formal_path(payload['root_out'], work)
    code = Path(env['REPO']) / 'code/current'
    require(str(code) == env['CODE_DIR'] == node['code_dir'] and code.resolve() == code
            and code.is_relative_to(work), 'CODE_DIR must be canonical WORK/REPO/code/current')
    require(env['NODE1_CODE_DIR'] == request['nodes'][1]['code_dir'], 'NODE1_CODE_DIR mismatch')
    require(env['PYTHON'] == node['python'] == sys.executable
            and env[('NODE0_PYTHON', 'NODE1_PYTHON')[rank]] == node['python'], 'interpreter mismatch')
    commit = env['WEAVETP_COMMIT']
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), 'need full commit SHA')
    require(compare.git_identity(code) == commit == payload['identity']['commit'], 'HEAD/env.sh mismatch')
    for key in ('MASTER_ADDR', 'MASTER_PORT', 'CHECKPOINT', 'NCCL_DEBUG'):
        require(env.get(key) == request['env'][key], f'env.sh {key} mismatch')
    require(env.get('MPS_OWNER') == request['env'].get('MPS_OWNER'), 'MPS_OWNER mismatch')
    profile = Path(request['env']['PROFILE'])
    compare.data_path(str(profile))
    require(profile.name == 'profile.json' and profile.parent.name == 'measurement',
            'PROFILE must be T08 measurement/profile.json')
    sha = payload['identity']['profile_sha256']
    compare.load_profile(profile, sha)
    require(profile.with_name('profile.json.sha256').read_text().split() == [sha, 'profile.json'],
            'PROFILE hash differs from T08 evidence')
    free = shutil.disk_usage('/data').free
    require(free >= MIN_FREE_BYTES, f'/data needs >= {MIN_FREE_BYTES} bytes; free={free}')
    require(not (Path(payload['root_out']) / 'isolate.lock').exists(), 'unresolved isolation lock')
    state = compare.gpu_state()
    compare.check_idle(state, rank, env.get('MPS_OWNER'))
    return {'rank': rank, 'identity': payload['identity'], 'free_bytes': free,
            'nccl_debug': env['NCCL_DEBUG'], 'gpu_before': state}


def node_command(payload, rank, arguments):
    node = payload['request']['nodes'][rank]
    script = Path(node['code_dir']).parents[1] / 'documents/16gpu_T09_20261003' / arguments[0]
    command = [node['python'], '-B', '-X', 'utf8', str(script), *arguments[1:]]
    shell = 'source ' + shlex.quote(payload['work'] + '/env.sh') + ' >&2; exec ' + shlex.join(command)
    command = ['bash', '-euc', shell]
    if rank == 1:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node['ssh'], shlex.join(command)]
    return command


def preflight(payload, rank):
    command = node_command(payload, rank, ['t09.py', '--node-preflight', str(rank)])
    result = subprocess.run(command, input=json.dumps(payload), capture_output=True,
                            text=True, encoding='utf-8', timeout=90)
    folder = Path(payload['root_out']) / '_control'
    folder.mkdir(exist_ok=True)
    compare.save(folder / f'preflight.{time.time_ns()}.node{rank}.json',
                 {'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    require(result.returncode == 0, f'node{rank} preflight failed: {result.stderr}')
    row = json.loads(result.stdout)
    require(row['rank'] == rank and row['identity'] == payload['identity'], 'preflight identity mismatch')
    print(f"PREFLIGHT node={rank} free_bytes={row['free_bytes']} NCCL_DEBUG=WARN idle=8", flush=True)


def launch_summary(request, refs):
    data = read(Path(request['out_dir']) / 'result.json')
    for index, row in enumerate(data['switches']):
        observed = row['plan_observation']
        default, adopted = [observed[k]['traffic']['total']['cross_node_bytes'] for k in ('default', 'adopted')]
        suffix = ''
        if request['case'] == 'weavetp':
            cpu = refs['switches'][index]['adopted_cross_node_bytes']
            warning = adopted > 2 * cpu or adopted * 2 < cpu
            suffix = f" cpu_reference_bytes={cpu}" + (' WARNING factor>2' if warning else '')
        print(f"SWITCH {request['env']['RUN_ID']} index={index} direction={row['direction']} "
              f"switch_wall_s={row['switch_wall_s']:.6f} default_bytes={default} adopted_bytes={adopted}{suffix}", flush=True)


def round_summary(settings, repeat):
    rows = []
    for request in compare.make_cases(settings):
        if request['repeat'] != repeat:
            continue
        out = Path(request['out_dir'])
        if not (out / 'complete.json').is_file():
            rows.append(f"{request['case']}=pending")
            continue
        compare.validate_result(out / 'result.json', request['case'])
        switches = read(out / 'result.json')['switches']
        expansion = sum(switches[i]['switch_wall_s'] for i in (0, 2)) / 2
        shrink = sum(switches[i]['switch_wall_s'] for i in (1, 3)) / 2
        rows.append(f"{request['case']} expansion={expansion:.6f}s shrink={shrink:.6f}s")
    print(f"ROUND r{repeat}: " + '; '.join(rows), flush=True)


def run_queue(settings, cases, identity, work, refs, check=preflight, execute=compare.run_case):
    for index, request in enumerate(cases):
        payload = {'request': request, 'root_out': settings['root_out'], 'identity': identity, 'work': work}
        for rank in (0, 1):
            check(payload, rank)
        started, status = time.monotonic(), 'failed'
        skipped = (Path(request['out_dir']) / 'complete.json').is_file()
        print(f"LAUNCH_START RUN_ID={request['env']['RUN_ID']} elapsed_s=0 status=starting", flush=True)
        try:
            execute(request)
            status = 'skipped_verified' if skipped else 'exit_0'
        finally:
            print(f"LAUNCH_END RUN_ID={request['env']['RUN_ID']} elapsed_s={time.monotonic()-started:.3f} status={status}", flush=True)
            if status == 'failed':
                print('SWITCH summary unavailable: launch failed/incomplete; retain raw evidence', flush=True)
        launch_summary(request, refs)
        if index + 1 == len(cases) or cases[index + 1]['repeat'] != request['repeat']:
            round_summary(settings, request['repeat'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--node-preflight', type=int, choices=(0, 1))
    args = parser.parse_args(argv)
    if args.node_preflight is not None:
        print(json.dumps(node_preflight(json.load(sys.stdin), args.node_preflight, os.environ)))
        return 0
    env = os.environ
    require(all(env.get(k) for k in ('PROFILE', 'PROFILE_SHA256', 'ROOT_OUT')), 'PROFILE/PROFILE_SHA256/ROOT_OUT required')
    sha = env['PROFILE_SHA256']
    require(len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'need full lowercase PROFILE_SHA256')
    formal_path(env['ROOT_OUT'], env['WORK'])
    settings = compare.settings_from_env()
    cases = selected_cases(settings, env.get('ONLY', ''), env.get('ROUNDS', ''))
    refs = reference(sha)
    if args.dry_run:
        print(json.dumps([{'request': r, 'node_commands': [compare.rpc_command(r, n, 'run') for n in (0, 1)]}
                          for r in cases], indent=2))
        return 0
    require(socket.gethostname().split('.')[0].lower() == 'sl3060', 'controller requires SL3060')
    require(REPO / 'code/current' == Path(env['CODE_DIR']), 'entrypoint must belong to CODE_DIR checkout')
    root = Path(settings['root_out'])
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / 'isolate.lock').exists(), 'unresolved isolation; do not resume')
    lock = root / 'compare.lock'
    compare.save(lock, {'pid': os.getpid(), 'hostname': socket.gethostname()})
    try:
        identity = {'commit': env['WEAVETP_COMMIT'], 'profile_sha256': sha}

        def pinned_rpc(request, rank, action):
            result = compare.rpc(request, rank, action)
            if action in ('inspect', 'prepare'):
                require(result['identity'] == identity, 'code/profile changed since T09 preflight')
            return result

        run_queue(settings, cases, identity, env['WORK'], refs,
                  execute=lambda r: compare.run_case(r, call=pinned_rpc))
    finally:
        lock.unlink()
    return 0


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
