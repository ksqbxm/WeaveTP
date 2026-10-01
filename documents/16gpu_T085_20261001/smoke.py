"""T08.5: preflight two hosts, then reuse the compare executor for one smoke case."""

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
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code/current/tools/resharding'))
import compare_weavetp_16gpu as compare


def require(condition, message):
    if not condition:
        raise ValueError(message)


def explicit_profile(env):
    profile, sha = env.get('PROFILE'), env.get('PROFILE_SHA256', '')
    require(profile and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
            'PROFILE and a full lowercase PROFILE_SHA256 must be copied from successful T08 output')
    compare.data_path(profile)
    return profile, sha


def check_profile(path, expected):
    profile = Path(path)
    require(profile.name == 'profile.json' and profile.parent.name == 'measurement',
            'PROFILE must be the explicit T08 measurement/profile.json')
    _, actual = compare.load_profile(profile, expected)
    record = profile.with_name('profile.json.sha256')
    require(record.read_text(encoding='utf-8').split() == [expected, 'profile.json'],
            'PROFILE_SHA256 differs from T08 profile.json.sha256')
    return actual


def node_preflight(request, rank, env):
    """Read-only gates; GPU queries do not launch CUDA work."""
    node = request['nodes'][rank]
    require(socket.gethostname().split('.')[0].lower() == node['hostname'].lower(), 'wrong host')
    require(env['NODE_RANK'] == str(rank), 'env.sh NODE_RANK mismatch')
    require(env['WORK'] == request['work'], 'env.sh WORK mismatch')
    compare.check_output_mode(request['out_dir'], True, env['WORK'])
    code = Path(env['REPO']) / 'code/current'
    require(str(code) == env['CODE_DIR'] == node['code_dir'] and code.resolve() == code
            and code.is_relative_to(env['WORK']), 'CODE_DIR must be canonical WORK/REPO/code/current')
    require(env['NODE1_CODE_DIR'] == request['nodes'][1]['code_dir'], 'NODE1_CODE_DIR mismatch')
    require(env['PYTHON'] == node['python'] == sys.executable, 'wrong existing interpreter')
    require(env[('NODE0_PYTHON', 'NODE1_PYTHON')[rank]] == node['python'], 'node interpreter mismatch')
    commit = env['WEAVETP_COMMIT']
    require(len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
            'WEAVETP_COMMIT must be a full lowercase SHA')
    require(compare.git_identity(code) == commit == request['expected_identity']['commit'],
            'WEAVETP_COMMIT/HEAD mismatch or dirty checkout')
    for key in ('MASTER_ADDR', 'MASTER_PORT', 'CHECKPOINT', 'NCCL_DEBUG'):
        require(env.get(key) == request['env'][key], f'env.sh {key} mismatch')
    require(env.get('MPS_OWNER') == request['env'].get('MPS_OWNER'), 'MPS_OWNER mismatch')
    free = shutil.disk_usage('/data').free
    require(free >= 20_000_000_000, '/data needs >=20 GB; no automatic retry')
    sha = check_profile(request['env']['PROFILE'], request['expected_identity']['profile_sha256'])
    root = Path(request['out_dir']).parents[1]
    require(not Path(request['out_dir']).exists(), 'smoke OUT_DIR already exists')
    if rank == 1:
        require(not root.exists(), 'peer smoke ROOT_OUT already exists')
    state = compare.gpu_state()
    compare.check_idle(state, rank, env.get('MPS_OWNER'))
    return {'rank': rank, 'commit': commit, 'profile_sha256': sha, 'free_bytes': free,
            'nccl_debug': env['NCCL_DEBUG'], 'gpu_before': state}


def preflight(request, rank, root):
    node = request['nodes'][rank]
    script = Path(node['code_dir']).parents[1] / 'documents/16gpu_T085_20261001/smoke.py'
    command = [node['python'], '-B', '-X', 'utf8', str(script), '--node-preflight', str(rank)]
    if rank == 1:
        shell = ('source ' + shlex.quote(request['work'] + '/env.sh') + ' >&2; exec '
                 + shlex.join(command))
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node['ssh'],
                   shlex.join(['bash', '-euc', shell])]
    result = subprocess.run(command, input=json.dumps(request), capture_output=True,
                            text=True, encoding='utf-8', timeout=90)
    compare.save(root / f'preflight.node{rank}.json', {
        'command': command, 'exit_code': result.returncode,
        'stdout': result.stdout, 'stderr': result.stderr,
    })
    require(result.returncode == 0, f'node{rank} preflight exit={result.returncode}: {result.stderr}')
    record = json.loads(result.stdout)
    require(record['rank'] == rank and record['commit'] == request['expected_identity']['commit']
            and record['profile_sha256'] == request['expected_identity']['profile_sha256'],
            f'node{rank} preflight identity mismatch')
    return record


def summary_rows(request):
    out = Path(request['out_dir'])
    complete = json.loads((out / 'complete.json').read_bytes())
    data = json.loads((out / 'result.json').read_bytes())
    require(complete['mode'] == 'smoke', 'missing smoke completion marker')
    rows = [('两机退出码', '通过', '/'.join(str(r['exit_code']) for r in complete['exits'])),
            ('checkpoint_loaded', '通过', str(data['checkpoint_loaded'])),
            ('切换方向', '通过', '扩容 2→4 → 缩容 4→2'),
            ('BF16-relative', '通过', '原 benchmark 阈值检查完成；两端成功退出'),
            ('组断言 TP2', '通过', '实测 dense DP=8，EDP=4（16 ranks）'),
            ('组断言 TP4', '通过', '实测 dense DP=4，EDP=2（16 ranks）'),
            ('双向计划', '通过', 'default/adopted 均存在'),
            ('进程与显存恢复', '通过', '两端 quiescent=true；沿用正式基线 +16 MiB')]
    for record, reference in zip(data['switches'], (14.4, 28.8)):
        label = '扩容' if record['direction'] == '2->4' else '缩容'
        observed = record['plan_observation']
        default = observed['default']['traffic']['total']['cross_node_bytes']
        adopted = observed['adopted']['traffic']['total']['cross_node_bytes']
        status = 'WARNING' if abs(default / 1e9 - reference) > reference * .2 else '仅记录'
        rows.append((f'{label}跨机字节', status,
                     f'default={default} ({default / 1e9:.3f} GB)，'
                     f'adopted={adopted} ({adopted / 1e9:.3f} GB)，default 参考 {reference} GB'))
        metrics = compare.switch_metrics(record)
        rows.append((f'{label}时序', '仅记录', f"waves={record['base']['waves']}，"
                     f"Transport={metrics['transport_s']:.6f}s，"
                     f"迁移墙钟={metrics['migration_wall_s']:.6f}s，"
                     f"切换墙钟={metrics['switch_wall_s']:.6f}s"))
    return rows


def controller():
    root, request = None, None
    created = False
    stage = '入口参数'
    rows = []
    error = None
    try:
        env = os.environ
        require(socket.gethostname().split('.')[0].lower() == 'sl3060', 'controller requires SL3060')
        profile, sha = explicit_profile(env)
        work = compare.data_path(env['WORK'])
        root = Path(work) / 'smoke' / f'smoke_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{os.getpid()}'
        compare.check_output_mode(str(root), True, work)
        require(REPO / 'code/current' == Path(env['CODE_DIR']), 'entrypoint must belong to CODE_DIR checkout')
        os.environ['ROOT_OUT'] = str(root)
        settings = compare.settings_from_env()
        settings.update(root_out=str(root), profile=profile)
        request = compare.make_smoke_case(settings, work)
        request['expected_identity'] = {'commit': env['WEAVETP_COMMIT'], 'profile_sha256': sha}
        root.mkdir(parents=True, exist_ok=False)
        created = True
        compare.save(root / 'request.json', request)
        print(f'T085_ROOT_OUT={root}', flush=True)
        for rank in (0, 1):
            stage = f'node{rank} 启动前检查'
            preflight(request, rank, root)
            rows.append((stage, '通过', '代码/画像/WARN/空间/正式 GPU 门禁'))
        stage = '双机 launch、结果与退出验收'
        compare.run_case(request)
        rows.extend(summary_rows(request))
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        rows.append((stage, '失败', error.replace('\n', ' ')))
        if created:
            compare.save(root / f'failure.{time.time_ns()}.json', {'stage': stage, 'error': error})
    table = '\n'.join(['| 检查项 | 结果 | 实测值 / 说明 |', '|---|---|---|']
                      + [f'| {name} | {status} | {detail} |' for name, status, detail in rows])
    print(table, flush=True)
    if created:
        with (root / 'trial_report.md').open('x', encoding='utf-8') as stream:
            stream.write(f'# T08.5 冒烟（不计入正式数据）\n\n{table}\n\n证据：{root}\n')
    print(f'T085_RESULT exit={int(error is not None)} stage={stage} evidence={root}', flush=True)
    return int(error is not None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-preflight', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.node_preflight is not None:
        print(json.dumps(node_preflight(json.load(sys.stdin), args.node_preflight, os.environ)))
        return 0
    return controller()


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'STOP: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        raise SystemExit(1)
