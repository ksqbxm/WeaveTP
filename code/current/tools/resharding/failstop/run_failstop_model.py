"""Step 5 real-model fail-stop entry: launch, replay specs, reference replay, A6 comparison.

  launch   two-node fail-stop launch (emulate / real / both) of DeepSeek-V2-Lite on the
           Step 4 protocol with model_worker.py; needs the frozen Step 3 profile and the
           reference parameter hashes written by run_model.py's reference role.
  specs    collect every checkpoint of a launch (both nodes' roots) into replay specs.
  replay   one torchrun per layout on this node: fresh checkpoint load, replay every spec.
  compare  A6 verdict: every launch record equals its replayed reference bitwise.

Nothing runs unless invoked; accepted stays false until A6 and A7 are both delivered.
"""

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from common import (
    DEFAULTS,
    DETERMINISTIC_ENVIRONMENT,
    MODEL_PROBE,
    NODES,
    PYTHON_FIRST_PATH,
    SHARED_GPU_NOTE,
    require,
    require_megatron_helpers,
    write_json,
)
from model_checks import compare, write_specs
from model_memory import LAYOUTS
from run_probe import repository_commit, run_case
from run_topology import cases, summarize
from safety import check_data_space, pidfd_implementations

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.correctness.router_determinism_driver import model_arguments

GIB = 1 << 30


def worker_arguments(checkpoint, prompt_tokens, seq_length):
    """Megatron arguments every fail-stop worker parses: the TP4 build over the 16-rank world, deterministic mode."""
    argv = model_arguments('tp4', checkpoint, prompt_tokens, seq_length)
    argv[argv.index('--global-batch-size') + 1] = '4'  # 16 ranks: TP4 x EP2 x DP2, micro batch 1 per DP replica pair
    return argv + ['--deterministic-mode']


def launch(args):
    require(sys.platform == 'linux' and Path('/data').is_mount(), 'requires Linux and mounted /data')
    require(str(args.output).replace('\\', '/').startswith('/data/') and '..' not in args.output.parts, 'output must be under /data')
    require(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,65}', args.suite_id)), 'invalid suite id')
    require(math.isfinite(args.budget_seconds) and args.budget_seconds > 0, 'invalid budget')
    profile = Path(args.profile_file)
    reference = Path(args.reference_root)
    require(profile.is_file(), f'frozen profile not found: {profile}')
    for tp in (2, 4):
        require(all((reference / f'tp{tp}' / 'reference' / f'parameters_rank_{r}.json').is_file() for r in range(2 * tp)),
                f'reference parameter hashes for tp{tp} are missing under {reference}')
    selected = cases(args.mode, args.rounds, args.real_launches, args.suite_id)
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=False)
    commit = repository_commit()
    check_data_space(args.min_free_disk_gb, root / 'disk_space.jsonl')
    for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        os.environ.pop(key, None)
    os.environ.update(DETERMINISTIC_ENVIRONMENT)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    mode_flags = {'quick_mode': False, 'correctness_only': args.correctness_only, 'shared_gpu_mode': args.correctness_only,
                  'timing_valid': False}
    write_json(root / 'invocation.json', {'commit': commit, 'argv': sys.argv, 'cases': selected, **mode_flags})
    deadline = time.monotonic() + args.budget_seconds
    results = []
    for index, case in enumerate(selected):
        if results and results[-1]['status'] != 'passed':
            results.append({**case, 'status': 'not_run', 'reason': 'prior case failed'})
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 60:
            results.append({**case, 'status': 'failed', 'reason': 'suite budget exhausted before launch'})
            continue
        config = {**{k: v for k, v in DEFAULTS.items() if k != 'repeats'}, **case,
                  'probe': MODEL_PROBE, 'kind': case['mode'], 'profile_only': False, 'node': args.node,
                  'output': str(root / case['launch']), 'master': args.master, 'port': args.port_base + index,
                  'commit': commit, 'mps_pipe': None, **mode_flags, 'min_free_disk_gb': args.min_free_disk_gb,
                  **pidfd_implementations(),
                  'startup_timeout_s': min(600, remaining - 60), 'stage_timeout_s': min(600, remaining - 60),
                  'reclaim_timeout_s': min(180, remaining - 60), 'launch_timeout_s': remaining,
                  'profile_file': str(profile), 'reference_root': str(reference), 'deterministic_mode': True,
                  'kv_heads': 16, 'memory_margin_bytes': 2 * GIB, 'memory_limit_bytes': None,
                  'expand_delta_steps': args.expand_delta_steps, 'phase0_steps': args.phase0_steps,
                  'model_argv': worker_arguments(args.checkpoint, args.prompt_tokens, args.seq_length)}
        print(f"Starting {case['launch']} mode={case['mode']} rounds={case['rounds']}", flush=True)
        results.append({**case, **run_case(config)})
        write_json(root / 'progress.json', results)
    summary = {**summarize(selected, results, quick=False, correctness_only=args.correctness_only), 'commit': commit,
               'accepted': False, 'a6': 'run `specs`, `replay` and `compare` on the checkpoints of both nodes',
               'a7': 'not implemented in this batch', 'transport': 'synchronous waves; Step 4 overlap not measured',
               'timing_note': SHARED_GPU_NOTE if args.correctness_only else 'correctness launch; no performance data'}
    write_json(root / 'summary.json', summary)
    print(f"STEP 5 FAILSTOP result={'PASS' if summary['all_cases_passed'] else 'FAIL'} logs={root}", flush=True)
    return 0 if summary['all_cases_passed'] else 1


def replay(args):
    specs = json.loads(Path(args.specs).read_text(encoding='utf-8'))
    code_root = Path(__file__).resolve().parents[3]
    args.output.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF')}
    env.update(DETERMINISTIC_ENVIRONMENT, NCCL_DEBUG='WARN', TORCH_NCCL_ASYNC_ERROR_HANDLING='1', CUDA_DEVICE_MAX_CONNECTIONS='8',
               CUDA_MPS_PIPE_DIRECTORY=str(args.output / 'absent_mps_pipe'), PYTHONDONTWRITEBYTECODE='1', PATH=PYTHON_FIRST_PATH)
    require_megatron_helpers()
    for layout in sorted(specs):
        gpus = list(range(LAYOUTS[layout]['gpus']))
        argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(gpus)}',
                'tools/resharding/failstop/model_reference.py', *model_arguments(layout, args.checkpoint, 32, args.seq_length),
                '--mr-output', str(args.output), '--mr-role', 'replay', '--mr-spec', str(args.specs), '--deterministic-mode']
        (args.output / f'command_{layout}.txt').write_text(shlex.join(argv) + '\n')
        print(f'Replaying {len(specs[layout])} {layout} checkpoints on GPUs {gpus}', flush=True)
        with (args.output / f'torchrun_{layout}.log').open('wb') as log:
            code = subprocess.run(argv, cwd=code_root, env={**env, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus))},
                                  stdout=log, stderr=subprocess.STDOUT).returncode
        require(code == 0, f'{layout} replay torchrun exited {code}; see {args.output / f"torchrun_{layout}.log"}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('launch')
    run.add_argument('--node', choices=NODES, required=True)
    run.add_argument('--suite-id', required=True)
    run.add_argument('--output', type=Path, required=True)
    run.add_argument('--mode', choices=('emulate', 'real', 'both'), default='emulate')
    run.add_argument('--rounds', type=int, default=5)
    run.add_argument('--real-launches', type=int, default=1)
    run.add_argument('--profile-file', required=True, help='frozen Step 3 profile.json of these sixteen GPUs')
    run.add_argument('--reference-root', required=True, help='run_model.py output root holding tp2/ and tp4/ reference runs')
    run.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', '/data/models/DeepSeek-V2-Lite-megatron-v2'))
    run.add_argument('--prompt-tokens', type=int, default=32)
    run.add_argument('--seq-length', type=int, default=128)
    run.add_argument('--phase0-steps', type=int, default=3)
    run.add_argument('--expand-delta-steps', type=int, default=3)
    run.add_argument('--correctness-only', action='store_true')
    run.add_argument('--master', default='10.60.14.1')
    run.add_argument('--port-base', type=int, default=29840)
    run.add_argument('--min-free-disk-gb', type=float, default=5)
    run.add_argument('--budget-seconds', type=float, default=3600)
    spec = commands.add_parser('specs')
    spec.add_argument('--launch-roots', nargs='+', required=True, help='<output>/<node>/<launch> of both nodes')
    spec.add_argument('--output', type=Path, required=True)
    rep = commands.add_parser('replay')
    rep.add_argument('--specs', type=Path, required=True)
    rep.add_argument('--output', type=Path, required=True)
    rep.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', '/data/models/DeepSeek-V2-Lite-megatron-v2'))
    rep.add_argument('--seq-length', type=int, default=128)
    cmp = commands.add_parser('compare')
    cmp.add_argument('--launch-roots', nargs='+', required=True)
    cmp.add_argument('--replay-root', type=Path, required=True)
    cmp.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'launch':
        return launch(args)
    if args.command == 'specs':
        print(json.dumps(write_specs(args.launch_roots, args.output), indent=2))
        return 0
    if args.command == 'replay':
        return replay(args)
    result = compare(args.launch_roots, args.replay_root)
    if args.output:
        write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'checkpoints'}, indent=2))
    print(f"A6 result={result['result']} checkpoints={len(result['checkpoints'])}", flush=True)
    return 0 if result['result'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
