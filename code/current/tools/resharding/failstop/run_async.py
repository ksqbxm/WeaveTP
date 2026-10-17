"""Explicit Step 4 suite. Does not run unless invoked; default mode is emulate."""

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

from common import DEFAULTS, NODES, require, write_json
from run_probe import repository_commit, run_case
from run_topology import cases, summarize
from safety import check_data_space, pidfd_implementations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=NODES, required=True)
    parser.add_argument('--suite-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('emulate', 'real', 'both'), default=os.environ.get('FAILSTOP_MODE', 'emulate'))
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--real-launches', type=int, default=1)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--correctness-only', action='store_true',
                        help='allow shared GPUs for correctness tests only; never formal acceptance or timing data')
    parser.add_argument('--master', default='10.60.14.1')
    parser.add_argument('--port-base', type=int, default=29880)
    parser.add_argument('--min-free-disk-gb', type=float, default=5)
    parser.add_argument('--budget-seconds', type=float, default=1200)
    parser.add_argument('--weight-elements', type=int, default=8388608,
                        help='float32 elements per full expert weight; large enough that the TP2->TP4 transfer spans several decode steps')
    parser.add_argument('--memory-budget-mib', type=int, default=2048)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    require(args.weight_elements >= 64 and args.weight_elements % 4 == 0, 'invalid weight size')
    require(args.memory_budget_mib > 0, 'invalid memory budget')
    correctness_only = args.correctness_only or args.quick
    mode_flags = {'quick_mode': args.quick, 'correctness_only': correctness_only,
                  'shared_gpu_mode': correctness_only, 'timing_valid': not correctness_only}
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,65}', args.suite_id)), 'invalid suite id')
    selected = cases(args.mode, args.rounds, args.real_launches, args.suite_id)
    require(math.isfinite(args.budget_seconds) and args.budget_seconds > 0 and
            1024 <= args.port_base <= 65536 - len(selected), 'invalid budget/port')
    require(math.isfinite(args.min_free_disk_gb) and args.min_free_disk_gb >= 0, 'invalid free disk threshold')
    require(str(args.output).replace('\\', '/').startswith('/data/') and '..' not in args.output.parts,
            'output must be under /data')
    if args.dry_run:
        print(json.dumps({'cases': selected, **mode_flags,
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
    write_json(root / 'invocation.json', {'commit': commit, 'argv': sys.argv, 'cases': selected, **mode_flags})
    deadline = time.monotonic() + args.budget_seconds
    results = []
    for index, case in enumerate(selected):
        if results and results[-1]['status'] != 'passed':
            results.append({**case, 'status': 'not_run', 'reason': 'prior case failed'})
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 30:
            results.append({**case, 'status': 'failed', 'reason': 'suite budget exhausted before launch'})
            write_json(root / 'progress.json', results)
            continue
        config = {**{k: v for k, v in DEFAULTS.items() if k != 'repeats'}, **case,
                  'probe': 'step04_async_driver_and_memory', 'kind': case['mode'],
                  'weight_elements': args.weight_elements, 'memory_budget_bytes': args.memory_budget_mib << 20,
                  'node': args.node, 'output': str(root / case['launch']), 'master': args.master,
                  'port': args.port_base + index, 'commit': commit, 'mps_pipe': None,
                  **mode_flags,
                  'min_free_disk_gb': args.min_free_disk_gb, **pidfd_implementations(),
                  'startup_timeout_s': min(180 if correctness_only else 90, remaining - 20),
                  'stage_timeout_s': min(120 if correctness_only else 60, remaining - 20),
                  'reclaim_timeout_s': min(120 if correctness_only else 60, remaining - 20),
                  'launch_timeout_s': remaining}
        print(f"Starting {case['launch']} mode={case['mode']} rounds={case['rounds']}", flush=True)
        result = run_case(config)
        results.append({**case, **result})
        write_json(root / 'progress.json', results)
    summary = {**summarize(selected, results, quick=args.quick, correctness_only=correctness_only), 'commit': commit}
    summary['mechanism_matrix_passed'] = summary['accepted']
    summary.update(accepted=False, prerequisites_verified=False, step3_integration='separate acceptance required')
    passed = summary['all_cases_passed']
    write_json(root / 'summary.json', summary)
    print(f"STEP 4 result={'PASS' if passed else 'FAIL'} correctness_only={correctness_only} logs={root}", flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    sys.exit(main())
