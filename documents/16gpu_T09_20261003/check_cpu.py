"""Run all T02-T08.5/T09 CPU suites and save reviewable dry-run evidence."""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import t09

LEGACY = t09.REPO / 'documents/16gpu_T085_20261001/check_cpu.py'
spec = importlib.util.spec_from_file_location('prior_cpu', LEGACY)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
BASELINE = '3bc12ce74a0ed048d62a9ffd5ca64d1940779705'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def validation(out):
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               TMPDIR=str(out), TMP=str(out), TEMP=str(out))
    env.pop('PYTHONOPTIMIZE', None)
    results = []
    suites = prior.SUITES + [t09.HERE / 'test_t09.py']
    for index, path in enumerate(suites):
        folder = out / f'{index:02d}_{path.stem}'
        folder.mkdir()
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--worker', str(path),
                   '--output-dir', str(folder)]
        run = subprocess.run(command, cwd=folder, env=env, capture_output=True, timeout=180)
        (folder / 'stdout.log').write_bytes(run.stdout)
        (folder / 'stderr.log').write_bytes(run.stderr)
        stats = prior.json.loads((folder / 'stats.json').read_bytes()) if (folder / 'stats.json').is_file() else {}
        results.append({'suite': str(path.relative_to(t09.REPO)), 'exit_code': run.returncode, **stats})
        print(f"{path.name}: exit={run.returncode} passed={stats.get('passed')}/{stats.get('tests')}", flush=True)
    sys.path.insert(0, str(LEGACY.parent))
    import test_smoke

    dry_env = test_smoke.dry_environment()
    dry_env.update(PROFILE_SHA256=t09.read(t09.HERE / 'cpu_reference.json')['profile_sha256'],
                   ROOT_OUT=dry_env['WORK'] + '/formal/T09_DRY_ONLY', ONLY='', ROUNDS='')
    dry_runs = {}
    for label, only, rounds in (('all', '', ''), ('only_weavetp_r1', 'weavetp_r1', ''), ('r1', '', 'r1')):
        run = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(t09.HERE / 't09.py'), '--dry-run'],
                             env={**dry_env, 'ONLY': only, 'ROUNDS': rounds}, capture_output=True, check=True)
        raw = run.stdout.replace(b'\r\n', b'\n')
        (out / f'dry_run_{label}.json').write_bytes(raw)
        dry_runs[label] = {'sha256': sha(raw), 'cases': len(json.loads(raw))}
    before, after = test_smoke.dry_run(True), test_smoke.dry_run()
    changes = []
    for old, new in zip(json.loads(before), json.loads(after), strict=True):
        label = f"{new['request']['case']}_r{new['request']['repeat']}"
        original_sha = sha(json.dumps(old, indent=2).encode())
        if old['request']['case'] == 'weavetp':
            old['request']['env'].update(ALLOW_AWARE_SHRINK='1', REROUTE_MIN_GAIN_PCT='0.0')
        assert old == new
        changes.append({'case': label, 'baseline_sha256': original_sha,
                        'current_sha256': sha(json.dumps(new, indent=2).encode()),
                        'request_sha256': sha(json.dumps(new['request'], indent=2).encode()),
                        'node_commands_sha256': sha(json.dumps(new['node_commands'], indent=2).encode()),
                        'approved_difference_only': True})
    prior.write_json(out / 'baseline_comparison.json', changes)
    summary_path = 'code/current/tools/resharding/summarize_weavetp_formal.py'
    old_tree = ast.parse(subprocess.check_output(['git', 'show', BASELINE + ':' + summary_path], cwd=t09.REPO))
    new_tree = ast.parse((t09.REPO / summary_path).read_bytes())
    # Only configuration/input exclusion/reporting changed; timing/statistics remain identical ASTs.
    allowed = {'validate_config', 'load_launches', 'render_markdown'}
    new_functions = {n.name: ast.dump(n) for n in new_tree.body if isinstance(n, ast.FunctionDef)}
    unchanged = []
    for node in old_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name not in allowed:
            assert ast.dump(node) == new_functions[node.name], node.name
            unchanged.append(node.name)
    for path in suites + list(t09.HERE.glob('*.py')):
        compile(path.read_bytes(), str(path), 'exec')
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else '/bin/bash'
    shells = list((t09.REPO / 'code/current/tools/resharding').glob('run*weavetp*sh'))
    shells += [t09.REPO / 'code/current/tools/resharding' / name for name in (
        'run_deepseek_v2_lite_directional_wave_compare.sh', 'run_deepseek_v2_lite_live_benchmark.sh',
        'run_live_moe_tp_benchmark.sh')]
    shells += [t09.HERE / 'run_t09.sh', LEGACY.parent / 'run_smoke.sh',
               t09.REPO / 'documents/16gpu_T08_20261001/run_t08.sh']
    for path in shells:
        subprocess.run([bash, '-n', str(path)], capture_output=True, check=True)
    protected = ['code/current/megatron/', 'code/current/examples/rl/benchmark_live_moe_tp.py',
                 'documents/16gpu_T08_20261001/']
    assert not subprocess.check_output(['git', 'diff', BASELINE, '--', *protected], cwd=t09.REPO)
    subprocess.run(['git', 'diff', '--check'], cwd=t09.REPO, capture_output=True, check=True)
    total, passed = sum(r.get('tests', 0) for r in results), sum(r.get('passed', 0) for r in results)
    receipt = {'tests': total, 'passed': passed, 'suites': results, 'dry_runs': dry_runs,
               'statistics_functions_unchanged': unchanged, 'protected_diff_empty': True,
               'gpu_started': False, 'server_connected': False,
               'ok': total == passed and all(r['exit_code'] == 0 for r in results)}
    prior.write_json(out / 'validation.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return int(not receipt['ok'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.worker:
        raise SystemExit(prior.worker(args.worker, out))
    if out.is_relative_to(t09.REPO) or not out.is_relative_to(Path('/data').resolve()):
        raise SystemExit('CPU output must be under /data, outside the checkout')
    out.mkdir(parents=True, exist_ok=False)
    raise SystemExit(validation(out))
