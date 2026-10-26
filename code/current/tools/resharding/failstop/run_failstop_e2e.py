"""Formal fail-stop serving runs on DeepSeek-V2-Lite, two nodes (EDP0 = SL3061, EDP1 = SL3060).

One invocation runs `--launches N` independent launches of one cell of the matrix:
  --method weavetp|anchortp   the recovery method (serving_worker.SWITCHERS / model_worker.PLANNERS); restart (--mode
                              emulate): EDP0 reloads TP2, then TP4, from disk in the same processes
  --scenario failstop|nofault the fault cycle of the plan, or the fixed TP4 x 2 baseline (calibration, no-fault reference)
  --mode emulate|real         fault mode (real: kill -9 and world rebuild)
  --run-mode timing|correctness
  --scenario failslow --method weavetp|anchortp|restart|ignore|nofault --slow-level W|M|S [--fault-gpus 0,1,4,5]
      [--warmup-s 120 --slow-seconds 600 --tail-s 120 --detect-seconds 10]: co-runners on the SL3061 fault GPUs,
      wall-clock deadline (failslow_probe); --mode emulate; nofault = the same timeline without injection; the budget
      defaults to launches x serving_config.failslow_launch_seconds and a shorter one is refused
Both nodes run the same command with the same --suite-id; SL3060 is the control gate. Each launch gets its own
TCPStore port (--port-base + index) and output directory `<output>/<node>/<suite>_<method>_<scenario>_<mode>_<k>`;
`matrix_manifest.json` records the request/workload/plan/profile hashes, the commit and the launch order, and
`progress.json` the per-launch outcome (a failed launch stops the sequence: later launches are `not_run`).
Timing launches refuse to start unless every GPU of the node is idle (serving_config.require_exclusive_gpus);
with --shared-gpu-ok they only check free memory (serving_config.require_gpu_memory), run with shared_gpu_mode=true
(timing_valid stays true, result.json gpu_sharing='shared') and record the foreign GPU processes before and after
each launch in `<launch>_foreign_gpu_processes.json`.
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

from common import NODES, require, require_megatron_helpers, write_json
from run_probe import repository_commit, run_case
from safety import check_data_space, foreign_gpu_processes
from serving_config import (
    RUN_MODES,
    SCENARIOS,
    SLOW_LEVELS,
    apply_environment,
    failslow_launch_seconds,
    failslow_settings,
    require_exclusive_gpus,
    require_gpu_memory,
    require_method,
    serving_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.e2e import requests_io  # noqa: E402
from tools.resharding.e2e.methods import METHODS  # noqa: E402


def stop(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--node', choices=NODES, required=True)
    parser.add_argument('--suite-id', required=True, help='typed identically on both nodes')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=sorted({*METHODS, 'nofault'}), default='weavetp',
                        help='ignore/nofault: --scenario failslow only (nofault = no injection, the baseline); restart: failslow '
                             'or emulate failstop')
    parser.add_argument('--scenario', choices=SCENARIOS, default='failstop')
    parser.add_argument('--mode', choices=('emulate', 'real', 'kill'), default='emulate')
    parser.add_argument('--run-mode', choices=RUN_MODES, default='timing')
    parser.add_argument('--launches', type=int, default=1, help='independent launches of this cell, in order')
    parser.add_argument('--correctness-only', action='store_true',
                        help='correctness runs may share GPUs with other tasks; never valid with --run-mode timing')
    parser.add_argument('--shared-gpu-ok', action='store_true',
                        help='timing runs on GPUs other tasks also use: no exclusive precheck, only a free-memory check; '
                             'timing_valid stays true, result.json says gpu_sharing=shared, foreign GPU processes are '
                             'recorded before and after each launch')
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True, help='plan_failstop.json or plan_nofault.json of the request set')
    parser.add_argument('--profile-file', type=Path, required=True, help='frozen Step 3 profile.json, same path on both nodes')
    parser.add_argument('--reference-root', type=Path, help='run_model.py reference hashes (correctness runs, read on SL3061)')
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', '/data/models/DeepSeek-V2-Lite-megatron-v2'))
    parser.add_argument('--seq-length', type=int, default=2048, help='max_position_embeddings: longest prompt + 256 outputs')
    parser.add_argument('--b-max', type=int, default=8, help='decode slots per dense replica')
    parser.add_argument('--capacity-tokens', type=int, default=None, help='KV reservation capacity per replica (default: layout formula)')
    parser.add_argument('--master', default='10.60.14.1')
    parser.add_argument('--port-base', type=int, default=29860)
    parser.add_argument('--budget-seconds', type=float, default=None,
                        help='for the whole invocation (default 3600; failslow: launches x the per-launch budget)')
    parser.add_argument('--slow-level', choices=sorted(SLOW_LEVELS), help='failslow: 1/2/3 co-runners per slow GPU (W/M/S)')
    parser.add_argument('--fault-gpus', default='0,1,4,5', help='failslow: SL3061 GPU indices that run the co-runners')
    parser.add_argument('--warmup-s', type=float, default=120.0, help='failslow: serving before the injection')
    parser.add_argument('--slow-seconds', type=float, default=600.0, help='failslow: injection length D')
    parser.add_argument('--tail-s', type=float, default=120.0, help='failslow: serving after the injection, to the deadline')
    parser.add_argument('--detect-seconds', type=float, default=10.0, help='failslow: detection delay after the injection starts')
    parser.add_argument('--min-free-disk-gb', type=float, default=5)
    parser.add_argument('--dry-run', action='store_true', help='print the launch configurations and exit')
    args = parser.parse_args(argv)
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,48}', args.suite_id)), 'invalid suite id')
    require(args.launches > 0, 'launch count must be positive')
    args.failslow = None
    if args.scenario == 'failslow':
        args.failslow = failslow_settings(method=args.method, level=args.slow_level, fault_gpus=[int(g) for g in args.fault_gpus.split(',')],
                                          warmup_s=args.warmup_s, slow_s=args.slow_seconds, tail_s=args.tail_s,
                                          detect_s=args.detect_seconds, mode=args.mode)
        need = args.launches * failslow_launch_seconds(args.failslow) + 60
        args.budget_seconds = need if args.budget_seconds is None else args.budget_seconds
        require(args.budget_seconds >= need, f'--budget-seconds {args.budget_seconds:.0f} is short of {need:.0f} s '
                                             f'for {args.launches} fail-slow launches')
    else:
        require_method(args.method, args.scenario, args.mode)
        args.budget_seconds = 3600 if args.budget_seconds is None else args.budget_seconds
    require(math.isfinite(args.budget_seconds) and args.budget_seconds > 0 and
            1024 <= args.port_base <= 65536 - args.launches, 'invalid budget/port')
    require(not (args.correctness_only and args.run_mode == 'timing'), '--correctness-only is a correctness-run flag')
    require(not (args.shared_gpu_ok and args.run_mode != 'timing'), '--shared-gpu-ok is a timing-run flag (correctness: --correctness-only)')
    require(args.scenario == 'nofault' or args.mode in ('emulate', 'real', 'kill'), 'invalid mode')
    require(str(args.output).replace('\\', '/').startswith('/data/') and '..' not in args.output.parts, 'output must be under /data')
    return args


def launch_id(args, index):
    return f'{args.suite_id}_{args.method}_{args.scenario}_{args.mode}_{index:02d}'


def check_inputs(args):
    """The plan must belong to the request file and match the scenario; the hashes go into the manifest."""
    requests = requests_io.load_requests(args.requests)
    workload = requests_io.load_workload(args.workload, requests)
    plan = requests_io.load_plan(args.plan, requests_io.sha256_file(args.requests))
    is_failstop = plan['version'] == 'v2-failstop'
    require(is_failstop == (args.scenario == 'failstop'), f"plan {args.plan} does not match scenario {args.scenario}")
    instances = requests_io.expand_instances(workload, requests)
    return {'requests': len(requests), 'instances': len(instances), 'workload_id': workload['workload_id'],
            'sha256': {name: requests_io.sha256_file(path) for name, path in
                       (('requests', args.requests), ('workload', args.workload), ('plan', args.plan), ('profile', args.profile_file))},
            'fail_count': requests_io.failstop_fail_count(len(instances)) if is_failstop else None,
            'rejoin_seconds': next(e['anchor']['seconds'] for e in plan['events'] if e['type'] == 'rejoin') if is_failstop else None}


def gpu_preflight(args, root, launch):
    """Before a launch (a refusal leaves this record and no launch directory): a timing run needs every GPU idle; with
    --shared-gpu-ok only enough free memory, and the foreign GPU processes are recorded."""
    if args.shared_gpu_ok:
        write_json(root / f'{launch}_preflight_memory.json', require_gpu_memory(args.node, args.seq_length, args.b_max))
        write_json(root / f'{launch}_foreign_gpu_processes.json',
                   {'before': foreign_gpu_processes(root / f'{launch}_foreign_gpu_processes.jsonl')})
    elif args.run_mode == 'timing':
        write_json(root / f'{launch}_preflight_exclusive.json',
                   require_exclusive_gpus(root / f'{launch}_preflight_exclusive.jsonl', os.environ.get('MPS_OWNER')))


def foreign_after(root, launch, output):
    """After a shared launch, once run_case confirmed (or failed to confirm) its own processes' exit: the foreign GPU
    processes again, this launch's registered PIDs left out."""
    path = root / f'{launch}_foreign_gpu_processes.json'
    registry = output / 'registered_processes.jsonl'
    own = {json.loads(line)['pid'] for line in registry.read_text().splitlines()} if registry.exists() else set()
    write_json(path, {**json.loads(path.read_text()), 'after': foreign_gpu_processes(root / f'{launch}_foreign_gpu_processes.jsonl', own)})
    return path


def main(argv=None):
    signal.signal(signal.SIGTERM, stop)
    args = parse(argv)
    manifest = {'suite': args.suite_id, 'node': args.node, 'method': args.method, 'scenario': args.scenario, 'mode': args.mode,
                'run_mode': args.run_mode, 'gpu_sharing': 'shared' if args.shared_gpu_ok or args.correctness_only else 'exclusive',
                'launches': [launch_id(args, i) for i in range(args.launches)],
                'argv': sys.argv, 'failslow': args.failslow, 'budget_seconds': args.budget_seconds, **check_inputs(args)}
    if args.dry_run:
        print(json.dumps(manifest, indent=2, default=str))
        return 0
    require(sys.platform == 'linux' and Path('/data').is_mount(), 'requires Linux and mounted /data')
    require_megatron_helpers()
    require(args.output.resolve().is_relative_to(Path('/data').resolve()), 'output escapes /data')
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=True)
    manifest['commit'] = commit = repository_commit()
    check_data_space(args.min_free_disk_gb, root / 'disk_space.jsonl')
    apply_environment(args.run_mode)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['TMPDIR'] = str(root / 'tmp')
    Path(os.environ['TMPDIR']).mkdir(exist_ok=True)
    os.environ['CUDA_CACHE_PATH'] = str(root / 'cuda_cache')
    write_json(root / f'matrix_manifest_{args.method}_{args.scenario}_{args.mode}.json', manifest)
    deadline = time.monotonic() + args.budget_seconds
    results = []
    progress = root / f'progress_{args.method}_{args.scenario}_{args.mode}.json'
    for index in range(args.launches):
        launch = launch_id(args, index)
        if results and results[-1]['status'] != 'passed':
            results.append({'launch': launch, 'status': 'not_run', 'reason': 'prior launch failed'})
            write_json(progress, results)
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 120:
            results.append({'launch': launch, 'status': 'failed', 'reason': 'budget exhausted before launch'})
            write_json(progress, results)
            continue
        output = root / launch
        require(not output.exists(), f'{output} exists: reruns take a new suite id')
        gpu_preflight(args, root, launch)
        config = serving_config(node=args.node, launch=launch, mode=args.mode, output=output, master=args.master,
                                port=args.port_base + index, commit=commit, remaining=remaining, min_free_disk_gb=args.min_free_disk_gb,
                                quick=False, shared_gpu=args.correctness_only or args.shared_gpu_ok, run_mode=args.run_mode,
                                method='ignore' if args.method == 'nofault' else args.method, failslow=args.failslow,
                                scenario=args.scenario, checkpoint=args.checkpoint, seq_length=args.seq_length,
                                profile_file=args.profile_file, reference_root=args.reference_root, requests=args.requests,
                                workload=args.workload, plan=args.plan, b_max=args.b_max, capacity_tokens=args.capacity_tokens)
        print(f'Starting {launch} method={args.method} scenario={args.scenario} mode={args.mode} run_mode={args.run_mode}', flush=True)
        outcome = run_case(config)
        record = {'launch': launch, 'output': str(output), **{k: outcome[k] for k in outcome if k != 'epochs'},
                  'gpu_sharing': 'shared' if config['shared_gpu_mode'] else 'exclusive'}
        if args.shared_gpu_ok:
            record['foreign_gpu_processes'] = str(foreign_after(root, launch, output))
        if args.node == NODES[0] and outcome['status'] == 'passed':
            record['metrics'] = outcome['metrics']
        results.append(record)
        write_json(progress, results)
    passed = len(results) == args.launches and all(r['status'] == 'passed' for r in results)
    print(f"FAILSTOP E2E {args.method} {args.scenario} {args.mode} {args.run_mode} result={'PASS' if passed else 'FAIL'} "
          f"launches={sum(r['status'] == 'passed' for r in results)}/{args.launches} logs={root}", flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
