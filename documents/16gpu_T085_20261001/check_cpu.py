"""Run T02-T08 and T08.5 on CPU; keep every artifact outside the checkout."""

import argparse
import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
UNIT = REPO / 'code/current/tests/unit_tests/resharding'
SUITES = [UNIT / name for name in (
    'test_summarize_weavetp_formal.py', 'test_compare_weavetp_16gpu.py',
    'test_profile_weavetp_16gpu.py', 'test_server_acceptance.py',
    'test_weavetp_review_findings.py', 'test_weavetp_observations.py')]
SUITES += [REPO / 'documents/16gpu_T06_T07_review_20261001/reproduce_findings.py',
           REPO / 'documents/16gpu_T08_20261001/check_commands.py', HERE / 'test_smoke.py']


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def worker(path, out):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('t085_cpu_suite', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = tempfile.TemporaryDirectory

    def temporary(*args, **kwargs):
        # T08's original tests request dir=HERE. Never write inside its locked directory.
        kwargs['dir'] = out
        return original(*args, **kwargs)

    with mock.patch.object(tempfile, 'TemporaryDirectory', side_effect=temporary):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    stats = {'tests': result.testsRun, 'failures': len(result.failures),
             'errors': len(result.errors), 'skipped': len(result.skipped)}
    stats['passed'] = stats['tests'] - sum(stats[k] for k in ('failures', 'errors', 'skipped'))
    write_json(out / 'stats.json', stats)
    return int(not result.wasSuccessful())


def validate(out):
    protected = list((REPO / 'documents/16gpu_T08_20261001').rglob('*'))
    protected += [REPO / 'code/current/tools/resharding/summarize_weavetp_formal.py']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='',
               TMPDIR=str(out), TMP=str(out), TEMP=str(out))
    env.pop('PYTHONOPTIMIZE', None)
    records = []
    for index, path in enumerate(SUITES):
        folder = out / f'{index:02d}_{path.stem}'
        folder.mkdir()
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                   '--worker', str(path), '--output-dir', str(folder)]
        result = subprocess.run(command, cwd=folder, env=env, capture_output=True, timeout=180)
        (folder / 'stdout.log').write_bytes(result.stdout)
        (folder / 'stderr.log').write_bytes(result.stderr)
        stats_path = folder / 'stats.json'
        stats = json.loads(stats_path.read_bytes()) if stats_path.is_file() else {}
        records.append({'suite': str(path.relative_to(REPO)), 'exit_code': result.returncode, **stats})
        print(f"{path.name}: exit={result.returncode} passed={stats.get('passed')}/{stats.get('tests')}", flush=True)
    from test_smoke import dry_run

    old, new, smoke = dry_run(True), dry_run(), dry_run(smoke_mode=True)
    for name, raw in (('formal_before.json', old), ('formal_after.json', new), ('smoke_dry_run.json', smoke)):
        (out / name).write_bytes(raw)
    checks = {'formal_byte_identical': old == new,
              'formal_before_sha256': hashlib.sha256(old).hexdigest(),
              'formal_after_sha256': hashlib.sha256(new).hexdigest()}
    compiled = list(dict.fromkeys(SUITES + [HERE / 'smoke.py', Path(__file__).resolve(),
                    REPO / 'documents/16gpu_T08_20261001/t08.py',
                    REPO / 'code/current/tools/resharding/compare_weavetp_16gpu.py',
                    REPO / 'code/current/tools/resharding/profile_weavetp_16gpu.py',
                    REPO / 'code/current/tools/resharding/summarize_weavetp_formal.py',
                    REPO / 'code/current/tools/resharding/weavetp_observations.py']))
    (out / 'py_compile').mkdir()
    for index, path in enumerate(compiled):
        py_compile.compile(str(path), cfile=str(out / 'py_compile' / f'{index}.pyc'), doraise=True)
    checks['py_compile_files'] = len(compiled)
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else '/bin/bash'
    shells = [REPO / 'code/current/tools/resharding' / name for name in (
        'run_deepseek_v2_lite_directional_wave_compare.sh', 'run_deepseek_v2_lite_live_benchmark.sh',
        'run_live_moe_tp_benchmark.sh', 'run_weavetp_16gpu_profile.sh')]
    shells += [REPO / 'documents/16gpu_T07_20261001/run_t07.sh',
               REPO / 'documents/16gpu_T08_20261001/run_t08.sh', HERE / 'run_smoke.sh']
    for path in shells:
        subprocess.run([bash, '-n', str(path)], check=True, capture_output=True)
    checks['bash_n_files'] = len(shells)
    subprocess.run(['git', 'diff', '--check'], cwd=REPO, check=True, capture_output=True)
    checks['diff_check'] = True
    checks['protected_bytes_unchanged'] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
                                            for p, sha in before.items())
    untouched = ['documents/16gpu_T08_20261001/', 'code/current/megatron/',
                 'code/current/tools/resharding/summarize_weavetp_formal.py']
    diff = subprocess.check_output(['git', 'diff', '634ac91bb66c050d75da7925de5e1c8a1713bf15', '--', *untouched], cwd=REPO)
    checks['protected_diff_empty'] = not diff
    total = sum(r.get('tests', 0) for r in records)
    passed = sum(r.get('passed', 0) for r in records)
    ok = (all(r['exit_code'] == 0 for r in records) and passed == total and old == new
          and checks['protected_bytes_unchanged'] and checks['protected_diff_empty'])
    write_json(out / 'validation.json', {'gpu_started': False, 'server_connected': False,
               'tests': total, 'passed': passed, 'suites': records, 'checks': checks, 'ok': ok})
    print(json.dumps({'tests': total, 'passed': passed, 'checks': checks, 'ok': ok}, ensure_ascii=False), flush=True)
    return int(not ok)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.is_relative_to(REPO) or not out.is_relative_to(Path('/data').resolve()):
        raise SystemExit('CPU output must be under /data, outside the checkout')
    if args.worker:
        raise SystemExit(worker(args.worker, out))
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(out)
    raise SystemExit(validate(out))
