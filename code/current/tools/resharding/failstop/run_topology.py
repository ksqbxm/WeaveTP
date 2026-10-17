"""Explicit Step 2 suite. Does not run unless invoked; default mode is emulate."""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from common import DEFAULTS, NODES, require, write_json
from run_probe import repository_commit, run_case
from safety import check_data_space, pidfd_implementations


def cases(mode, rounds, real_launches, suite):
    require(rounds > 0 and real_launches > 0, 'round/launch counts must be positive')
    result = []
    if mode in ('emulate', 'both'):
        result.append({'mode': 'emulate', 'rounds': rounds, 'launch': suite + '_emulate'})
    if mode in ('real', 'both'):
        result.extend({'mode': 'real', 'rounds': 1, 'launch': f'{suite}_real_{i:02d}'}
                      for i in range(real_launches))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=NODES, required=True)
    parser.add_argument('--suite-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('emulate', 'real', 'both'), default=os.environ.get('FAILSTOP_MODE', 'emulate'))
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--real-launches', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--allow-foreign-gpu-processes', action='store_true')
    parser.add_argument('--master', default='10.60.14.1')
    parser.add_argument('--port-base', type=int, default=29840)
    parser.add_argument('--min-free-disk-gb', type=float, default=5)
    parser.add_argument('--budget-seconds', type=float, default=1200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    import re

    require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,65}', args.suite_id)), 'invalid suite id')
    selected = cases(args.mode, args.rounds, args.real_launches, args.suite_id)
    require(args.budget_seconds > 0 and 1024 <= args.port_base < 65535 - len(selected), 'invalid budget/port')
    require(str(args.output).replace('\\', '/').startswith('/data/') and '..' not in args.output.parts,
            'output must be under /data')
    if args.dry_run:
        print(json.dumps({'cases': selected, 'quick_mode': args.quick, 'shared_gpu_mode': args.allow_foreign_gpu_processes,
                          'output': str(args.output), 'node': args.node}, indent=2))
        return 0
    require(sys.platform == 'linux' and Path('/data').is_mount(), 'requires Linux and mounted /data')
    require(args.output.resolve().is_relative_to(Path('/data').resolve()), 'output escapes /data')
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=False)
    commit = repository_commit()
    check_data_space(args.min_free_disk_gb, root / 'disk_space.jsonl')
    for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        os.environ.pop(key, None)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['TMPDIR'] = str(root / 'tmp')
    Path(os.environ['TMPDIR']).mkdir()
    os.environ['CUDA_CACHE_PATH'] = str(root / 'cuda_cache')
    write_json(root / 'invocation.json', {'commit': commit, 'argv': sys.argv, 'cases': selected})
    deadline = time.monotonic() + args.budget_seconds
    results = []
    for index, case in enumerate(selected):
        if results and results[-1]['status'] != 'passed':
            results.append({**case, 'status': 'not_run', 'reason': 'prior case failed'})
            continue
        remaining = deadline - time.monotonic()
        require(remaining > 30, 'suite budget exhausted before next launch')
        config = {**DEFAULTS, **case, 'probe': 'step02_topology_rebind', 'kind': case['mode'],
                  'node': args.node, 'output': str(root / case['launch']), 'master': args.master,
                  'port': args.port_base + index, 'commit': commit, 'mps_pipe': None,
                  'repeats': 1 if args.quick else 10, 'quick_mode': args.quick,
                  'shared_gpu_mode': args.allow_foreign_gpu_processes,
                  'min_free_disk_gb': args.min_free_disk_gb, **pidfd_implementations(),
                  'startup_timeout_s': min(90, remaining - 20),
                  'stage_timeout_s': min(60, remaining - 20),
                  'reclaim_timeout_s': min(60, remaining - 20), 'launch_timeout_s': remaining}
        print(f"Starting {case['launch']} mode={case['mode']} rounds={case['rounds']}", flush=True)
        result = run_case(config)
        results.append({**case, **result})
        write_json(root / 'progress.json', results)
    passed = len(results) == len(selected) and all(r['status'] == 'passed' for r in results)
    summary = {'status': 'passed' if passed else 'failed', 'all_cases_passed': passed,
               'accepted': passed and not args.quick, 'quick_mode': args.quick,
               'shared_gpu_mode': args.allow_foreign_gpu_processes, 'commit': commit, 'results': results,
               'emulate': 'passed' if any(r['mode'] == 'emulate' and r['status'] == 'passed' for r in results) else 'pending',
               'real': 'passed' if any(r['mode'] == 'real' and r['status'] == 'passed' for r in results) else 'pending'}
    write_json(root / 'summary.json', summary)
    print(f"STEP 2 result={'PASS' if passed else 'FAIL'} logs={root}", flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
