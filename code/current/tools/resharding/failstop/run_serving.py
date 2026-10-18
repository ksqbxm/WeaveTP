"""Step 6 A-part quick runner: the fake-backend serving loop (scaling 2->4 and 4->2, fail-stop
fail -> replay -> expand) plus the e2e CPU tests, on one node without GPUs.

summary.json carries the quick-driver flags; `accepted` stays false until the real-model B part
runs on the servers. Writes `<output>/<node>/{scaling,failstop}/out/*`, `pytest.log`, `summary.json`
and `error.txt` on failure.
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import NODES, SERVING_PROBE, require, write_json

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'examples' / 'rl'))

import run_e2e_scaling  # noqa: E402

from tools.resharding.e2e.toy_inputs import write_toy_inputs  # noqa: E402

RUNS = {
    'scaling': dict(count=24, failstop=False, policy='teacher_forcing', world_size=8, expected=['2->4', '4->2'],
                    switches=[{'direction': '2->4', 'anchor': {'kind': 'before_admit', 'seq': 12}},
                              {'direction': '4->2', 'anchor': {'kind': 'after_complete', 'seq': 14}}]),
    'failstop': dict(count=20, failstop=True, policy='greedy', world_size=16, expected=['shrink', 'expand'], switches=None),
}
TESTS = ('tests/unit_tests/e2e',)


def fake_run(name, root):
    """One full driver loop on the fake backend; passes when the checker, the switch sequence and every validation pass."""
    spec = RUNS[name]
    inputs = write_toy_inputs(root / name / 'inputs', count=spec['count'], switches=spec['switches'], failstop=spec['failstop'])
    out = root / name / 'out'
    argv = ['--e2e-requests', str(inputs['requests']), '--e2e-workload', str(inputs['workload']), '--e2e-plan', str(inputs['plan']),
            '--e2e-out-dir', str(out), '--e2e-backend', 'fake', '--e2e-run-mode', 'correctness', '--e2e-experiment', name,
            '--e2e-decode-policy', spec['policy'], '--e2e-world-size', str(spec['world_size']), '--e2e-b-max', '3',
            '--e2e-capacity-tokens', '400']
    (root / name / 'command.txt').write_text(json.dumps(argv) + '\n', encoding='utf-8')
    code = run_e2e_scaling.main(argv)
    result = json.loads((out / 'result.json').read_text(encoding='utf-8'))
    report = json.loads((out / 'check.json').read_text(encoding='utf-8'))
    directions = [s['direction'] for s in result['switches']]
    validations = [s['validation']['passed'] for s in result['switches']]
    record = {'check_passed': code == 0 and report['passed'] and directions == spec['expected'] and all(validations),
              'exit_code': code, 'switches': directions, 'validation': validations, 'totals': result['totals'], 'output': str(out)}
    if name == 'failstop':
        record['replay_prefill_tokens'] = result['failstop']['replay_prefill_tokens']
        record['events_s'] = result['failstop']['events_s']
    return record


def run_tests(root, deadline):
    """The e2e CPU tests with the runner's interpreter; the last pytest line is the summary."""
    require(importlib.util.find_spec('pytest') is not None, 'pytest is not installed in this Python environment')
    argv = [sys.executable, '-B', '-m', 'pytest', '-q', '-o', 'addopts=', '--noconftest', '-p', 'no:cacheprovider', *TESTS]
    log = root / 'pytest.log'
    with log.open('wb') as output:
        code = subprocess.run(argv, cwd=CODE_ROOT, stdout=output, stderr=subprocess.STDOUT,
                              timeout=max(1.0, deadline - time.monotonic())).returncode
    lines = log.read_text(encoding='utf-8', errors='replace').strip().splitlines()
    summary = lines[-1] if lines else 'pytest produced no output'
    passed = code == 0 and re.search(r'\d+ passed', summary) is not None
    return {'exit_code': code, 'passed': passed, 'summary_line': summary, 'log': str(log)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=NODES, required=True)
    parser.add_argument('--suite-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--budget-seconds', type=float, default=300)
    parser.add_argument('--min-free-disk-gb', type=float, default=1, help='accepted for the shared quick-driver argv; this runner needs no /data space check')
    args = parser.parse_args(argv)
    require(args.quick and args.correctness_only, 'the Step 6 A part runs only as the quick correctness check')
    started = time.monotonic()
    deadline = started + args.budget_seconds
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=False)
    summary = {'step': 6, 'probe': SERVING_PROBE, 'node': args.node, 'suite_id': args.suite_id, 'backend': 'fake', 'gpus_used': 0,
               'single_node': True, 'quick_mode': True, 'correctness_only': True, 'shared_gpu_mode': False, 'timing_valid': False,
               'accepted': False, 'runs': {}, 'tests': None, 'all_checks_passed': False}
    try:
        for name in RUNS:
            summary['runs'][name] = fake_run(name, root)
        summary['tests'] = run_tests(root, deadline - 15)
        summary['all_checks_passed'] = all(r['check_passed'] for r in summary['runs'].values()) and summary['tests']['passed']
        require(summary['all_checks_passed'], 'fake-backend run or e2e tests failed; see summary.json and pytest.log')
    except BaseException:
        summary['error'] = traceback.format_exc()
        (root / 'error.txt').write_text(summary['error'], encoding='utf-8')
        raise
    finally:
        summary['elapsed_s'] = time.monotonic() - started
        write_json(root / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('runs', 'tests', 'all_checks_passed', 'elapsed_s')}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
