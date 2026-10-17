"""Step 5 suite entry: per-GPU memory preflight, real-model deterministic reference/candidate/timing
runs per layout, and the cross-mode migration-plan comparison. Nothing runs unless invoked.

The fail-stop fail->shrink->rejoin->expand integration of the real model follows Step 3/4 GPU
acceptance; until then the candidate is an independent fresh checkpoint load, accepted=false.
"""

import argparse
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from common import (
    DETERMINISTIC_ENVIRONMENT,
    MODEL_PROBE,
    NODES,
    SHARED_GPU_NOTE,
    require,
    write_json,
)
from determinism import compare_launch_plans, step_time_ratio
from model_memory import LAYOUTS, format_table, preflight
from run_probe import repository_commit
from safety import check_data_space

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.correctness.router_determinism_driver import model_arguments

ROLES = ('reference', 'candidate', 'timing')
# Quick: three checkpoint loads per node within the 300 s budget, both layouts across the pair.
QUICK_LAYOUTS = {'SL3060': ['tp4'], 'SL3061': ['tp2']}
FORMAL_LAYOUTS = {'SL3060': ['tp4'], 'SL3061': ['tp2', 'tp4']}


def command(python, layout, role, checkpoint, output, prompt_tokens, decode_steps, seq_length, request_id, reference=None):
    argv = [python, '-m', 'torch.distributed.run', '--standalone', f"--nproc_per_node={LAYOUTS[layout]['gpus']}",
            'tools/resharding/failstop/model_reference.py', *model_arguments(layout, checkpoint, prompt_tokens, seq_length),
            '--mr-output', str(output), '--mr-role', role, '--mr-decode-steps', str(decode_steps),
            '--mr-request-id', request_id]
    if role != 'timing':
        argv.append('--deterministic-mode')
    if role == 'candidate':
        argv += ['--mr-reference', str(reference)]
    return argv


def environment(role, gpus, mps_pipe, base=None):
    """Live-launcher runtime plus the pinned deterministic environment, or the clean default for timing."""
    env = {k: v for k, v in (os.environ if base is None else base).items() if k not in DETERMINISTIC_ENVIRONMENT}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1',
               CUDA_DEVICE_MAX_CONNECTIONS='8', CUDA_MPS_PIPE_DIRECTORY=mps_pipe, PYTHONDONTWRITEBYTECODE='1')
    if role != 'timing':
        env.update(DETERMINISTIC_ENVIRONMENT)
    return env


def evaluate(results):
    """Per-layout verdict from the three roles' result.json files."""
    reference, candidate, timing = (results[role] for role in ROLES)
    checks = {'reference_deterministic': reference['manifest']['mode'] == 'deterministic' and reference['manifests_identical'],
              'candidate_deterministic': candidate['manifest']['mode'] == 'deterministic' and candidate['manifests_identical'],
              'timing_default': timing['manifest']['mode'] == 'default',
              'parameter_checksums_match': reference['parameter_checksums'] == candidate['parameter_checksums'],
              'history_match': reference['history_sha256'] == candidate['history_sha256'],
              'candidate_bitwise': candidate.get('bitwise') is True}
    ratio = step_time_ratio(reference['timing_rank0'], timing['timing_rank0'])
    return {'passed': all(checks.values()), 'checks': checks, 'first_mismatch': candidate.get('first_mismatch'),
            'candidate_kind': candidate.get('candidate_kind'), 'step_time_ratio': ratio,
            'manifests': {role: results[role]['manifest'] for role in ROLES}}


def summarize(layouts, evaluations, memory, *, quick, correctness_only):
    passed = all(layout in evaluations and evaluations[layout]['passed'] for layout in layouts)
    ratios = {layout: e['step_time_ratio'] for layout, e in evaluations.items()}
    return {'step': 5, 'probe': MODEL_PROBE, 'status': 'passed' if passed else 'failed', 'all_checks_passed': passed,
            'layouts_requested': list(layouts), 'layouts': evaluations, 'deterministic_step_time_ratio': ratios,
            'memory_preflight': {key: memory.get(key) for key in ('scenario_sufficient_on_this_node', 'scenario_shortages',
                                                                     'run_shortages')},
            'quick_mode': quick, 'correctness_only': correctness_only, 'shared_gpu_mode': correctness_only,
            'timing_valid': False, 'accepted': False,
            'timing_note': SHARED_GPU_NOTE if correctness_only else
            'exclusive GPU; the step-time ratio is an overhead estimate, not a performance result',
            'candidate_kind': 'fresh_checkpoint_load',
            'plan_consistency': {'status': 'pending', 'tool': 'run_model.py --compare-plans <default launch> <deterministic launch>',
                                 'requires': 'two Step 3 launches with the same --profile-file, one per mode'},
            'fail_stop_integration': 'pending: Step 3 and Step 4 GPU acceptance first'}


def stop(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')


def main(argv=None):
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare-plans', nargs=2, metavar=('REFERENCE_LAUNCH', 'CANDIDATE_LAUNCH'),
                        help='compare the normalized migration plans of two launches and exit')
    parser.add_argument('--node', choices=NODES)
    parser.add_argument('--suite-id')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--correctness-only', action='store_true',
                        help='allow shared GPUs for correctness tests only; never formal acceptance or timing data')
    parser.add_argument('--layouts', help='comma-separated subset of tp2,tp4; default depends on node and --quick')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', '/data/models/DeepSeek-V2-Lite-megatron-v2'))
    parser.add_argument('--prompt-tokens', type=int, default=32)
    parser.add_argument('--decode-steps', type=int, default=4)
    parser.add_argument('--seq-length', type=int, default=128)
    parser.add_argument('--min-free-disk-gb', type=float, default=5)
    parser.add_argument('--budget-seconds', type=float, default=1200)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.compare_plans:
        result = compare_launch_plans(*args.compare_plans)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result['equal'] else 1
    require(args.node and args.suite_id and args.output is not None, '--node, --suite-id and --output are required')
    correctness_only = args.correctness_only or args.quick
    layouts = (args.layouts.split(',') if args.layouts else (QUICK_LAYOUTS if args.quick else FORMAL_LAYOUTS)[args.node])
    gpus = [int(x) for x in args.gpus.split(',')]
    require(layouts and set(layouts) <= set(LAYOUTS) and len(gpus) == 8 and len(set(gpus)) == 8, 'invalid --layouts/--gpus')
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,65}', args.suite_id)), 'invalid suite id')
    require(math.isfinite(args.budget_seconds) and args.budget_seconds > 0, 'invalid budget')
    require(math.isfinite(args.min_free_disk_gb) and args.min_free_disk_gb >= 0, 'invalid free disk threshold')
    require(args.prompt_tokens + args.decode_steps < args.seq_length, 'prompt+decode exceeds --seq-length')
    require(str(args.output).replace('\\', '/').startswith('/data/') and '..' not in args.output.parts,
            'output must be under /data')
    mode_flags = {'quick_mode': args.quick, 'correctness_only': correctness_only,
                  'shared_gpu_mode': correctness_only, 'timing_valid': False}
    if args.dry_run:
        print(json.dumps({'step': 5, 'probe': MODEL_PROBE, 'node': args.node, 'layouts': layouts, 'roles': list(ROLES),
                          'output': str(args.output), **mode_flags}, indent=2))
        return 0
    require(sys.platform == 'linux' and Path('/data').is_mount(), 'requires Linux and mounted /data')
    require(args.output.resolve().is_relative_to(Path('/data').resolve()), 'output escapes /data')
    require(Path(args.checkpoint, 'latest_checkpointed_iteration.txt').is_file(), f'converted checkpoint not found: {args.checkpoint}')
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=False)
    commit = repository_commit()
    check_data_space(args.min_free_disk_gb, root / 'disk_space.jsonl')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    write_json(root / 'invocation.json', {'commit': commit, 'argv': sys.argv, 'layouts': layouts, **mode_flags})
    memory = preflight(args.node, args.seq_length, gpus, layouts)
    write_json(root / 'memory_preflight.json', memory)
    print(format_table(memory), flush=True)
    require(not memory['run_shortages'], 'insufficient GPU memory: ' + '; '.join(memory['run_shortages']))
    deadline = time.monotonic() + args.budget_seconds
    code_root = Path(__file__).resolve().parents[3]  # torchrun resolves tools/... relative to code/current
    mps_pipe = str(root / 'absent_mps_pipe')
    evaluations, child = {}, None
    try:
        for layout in layouts:
            used = gpus[:LAYOUTS[layout]['gpus']]
            results = {}
            for role in ROLES:
                output = root / layout / role
                output.mkdir(parents=True)
                argv = command(sys.executable, layout, role, args.checkpoint, output, args.prompt_tokens, args.decode_steps,
                               args.seq_length, f'{args.suite_id}-{layout}', reference=root / layout / 'reference')
                (output / 'command.txt').write_text(shlex.join(argv) + '\n')
                print(f'Starting {layout} {role} on GPUs {used}', flush=True)
                with (output / 'torchrun.log').open('wb') as log:
                    child = subprocess.Popen(argv, cwd=code_root, env=environment(role, used, mps_pipe), stdout=log,
                                             stderr=subprocess.STDOUT)
                    code = child.wait(timeout=max(1, deadline - time.monotonic()))
                child = None
                result = output / 'result.json'
                require(code == 0 and result.exists(), f'{layout} {role} torchrun exited {code}; see {output / "torchrun.log"}')
                results[role] = json.loads(result.read_text())
            evaluations[layout] = evaluate(results)
            write_json(root / f'model_checks_{layout}.json', evaluations[layout])
            require(evaluations[layout]['passed'], f'{layout} checks failed: {evaluations[layout]["checks"]}')
    except BaseException:
        (root / 'error.txt').write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
    summary = {**summarize(layouts, evaluations, memory, quick=args.quick, correctness_only=correctness_only), 'commit': commit}
    if time.monotonic() > deadline:
        summary.update(status='failed', all_checks_passed=False, budget_error=f'exceeded {args.budget_seconds} s budget')
    write_json(root / 'summary.json', summary)
    ratios = {layout: round(r['median_ratio'], 3) for layout, r in summary['deterministic_step_time_ratio'].items()}
    print(f"STEP 5 result={'PASS' if summary['all_checks_passed'] else 'FAIL'} layouts={layouts} "
          f"deterministic_step_time_ratio={ratios} correctness_only={correctness_only} logs={root}", flush=True)
    return 0 if summary['all_checks_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
