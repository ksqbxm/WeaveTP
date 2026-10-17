"""Single-node driver for the DeepSeek-V2-Lite native-router bitwise determinism check.

CPU only: pins the delivered commit, lists the per-GPU memory each layout needs,
launches one torchrun per layout (TP2 on four GPUs, TP4 on eight), restores the
formal commit and packages the evidence. GPU work is in router_determinism.py.
"""

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

BASE = 'f7d555998b45756998465924d733fd70e553d6b4'
GIB = 1 << 30

# DeepSeek-V2-Lite as launched by run_live_moe_tp_benchmark.sh (local backend, bf16).
MODEL = {'layers': 27, 'moe_layers': 26, 'hidden': 2048, 'ffn': 10944, 'heads': 16,
         'qk_head_dim': 128, 'qk_pos_emb_head_dim': 64, 'v_head_dim': 128, 'kv_lora_rank': 512,
         'experts': 64, 'moe_ffn': 1408, 'shared_ffn': 2816, 'vocab': 102400, 'bytes_per_param': 2}
LAYOUTS = {'tp2': {'tp': 2, 'ep': 2, 'gpus': 4}, 'tp4': {'tp': 4, 'ep': 2, 'gpus': 8}}
# Forward-path operators whose output order is not fixed unless deterministic mode is on.
SUSPECT_OPERATORS = [
    {'operator': 'moe_utils.unpermute: output_tokens.scatter_add_ (bf16 atomics over top-6 expert outputs)',
     'controlled_by': 'torch.use_deterministic_algorithms(True) switches to index_add_'},
    {'operator': 'NCCL TP all-reduce / reduce-scatter algorithm selection',
     'controlled_by': 'NCCL_ALGO pinned (Megatron --deterministic-mode requires it)'},
    {'operator': 'cuBLAS GEMM algorithm/workspace selection (dense, MLA, expert and router GEMMs)',
     'controlled_by': 'CUBLAS_WORKSPACE_CONFIG=:4096:8 with deterministic algorithms'},
]
# Allocations the preflight does not size analytically; they are measured after the run.
UNESTIMATED_ALLOCATIONS = ['cuBLAS workspaces', 'NCCL communicator buffers', 'MoE all-to-all scratch',
                           'attention score matrices for the prompt', 'caching-allocator reserve']


def estimate_rank_parameter_bytes(tp, etp, ep, vocab=None):
    """Exact per-rank parameter bytes for the local-backend DeepSeek-V2-Lite build.

    Routers are stored in params_dtype; q_lora_rank is None so there is no q_layernorm.
    """
    m = MODEL
    vocab = m['vocab'] if vocab is None else vocab
    h, q_head_dim = m['hidden'], m['qk_head_dim'] + m['qk_pos_emb_head_dim']
    attention = (h + m['heads'] * q_head_dim * h // tp + (m['kv_lora_rank'] + m['qk_pos_emb_head_dim']) * h // tp +
                 m['kv_lora_rank'] + m['heads'] * (m['qk_head_dim'] + m['v_head_dim']) * m['kv_lora_rank'] // tp +
                 h * m['heads'] * m['v_head_dim'] // tp + h)
    dense_mlp = 2 * m['ffn'] * h // tp + h * m['ffn'] // tp
    expert = 2 * m['moe_ffn'] * h // etp + h * m['moe_ffn'] // etp
    moe = m['experts'] * h + m['experts'] // ep * expert + 2 * m['shared_ffn'] * h // tp + h * m['shared_ffn'] // tp
    params = (vocab * h // tp + m['layers'] * attention + (m['layers'] - m['moe_layers']) * dense_mlp +
              m['moe_layers'] * moe + h + vocab * h // tp)
    return params * m['bytes_per_param']


def estimate_kv_cache_bytes(tp, max_sequence_length, batch_size=1):
    """StaticInferenceContext MLA cache: per layer key [S,B,groups/tp,q_head_dim] + value [...,v_head_dim]."""
    m = MODEL
    per_position = m['heads'] // tp * (m['qk_head_dim'] + m['qk_pos_emb_head_dim'] + m['v_head_dim'])
    return m['layers'] * max_sequence_length * batch_size * per_position * m['bytes_per_param']


def plan_memory(layout, max_sequence_length, margin_bytes=2 * GIB):
    spec = LAYOUTS[layout]
    weights = estimate_rank_parameter_bytes(spec['tp'], spec['tp'], spec['ep'])
    kv = estimate_kv_cache_bytes(spec['tp'], max_sequence_length)
    return {'layout': layout, 'tp': spec['tp'], 'etp': spec['tp'], 'ep': spec['ep'], 'gpus': spec['gpus'],
            'weight_bytes_per_gpu': weights, 'kv_cache_bytes_per_gpu': kv, 'margin_bytes': margin_bytes,
            'required_free_bytes_per_gpu': weights + kv + margin_bytes,
            'not_estimated_measured_after_run': UNESTIMATED_ALLOCATIONS}


def query_gpus(indices):
    text = subprocess.check_output(
        ['nvidia-smi', f"--id={','.join(map(str, indices))}", '--query-gpu=index,uuid,memory.total,memory.free',
         '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in text.strip().splitlines():
        index, uuid, total, free = [x.strip() for x in line.split(',')]
        rows.append({'index': int(index), 'uuid': uuid, 'total_bytes': int(total) << 20, 'free_bytes': int(free) << 20})
    if [r['index'] for r in rows] != list(indices):
        raise RuntimeError(f'nvidia-smi returned {rows} for GPUs {indices}')
    return rows


def preflight(gpus, layouts, max_sequence_length, checkpoint):
    """Per-GPU requirement table; refuses to launch when any used GPU lacks the estimated memory."""
    if not Path(checkpoint, 'latest_checkpointed_iteration.txt').is_file():
        raise RuntimeError(f'converted checkpoint not found: {checkpoint}')
    inventory = query_gpus(gpus)
    plans, shortages = {}, []
    for layout in layouts:
        plan = plan_memory(layout, max_sequence_length)
        used = inventory[:plan['gpus']]
        if len(used) < plan['gpus']:
            raise RuntimeError(f'{layout} needs {plan["gpus"]} GPUs, {len(used)} listed')
        plan['gpus_used'] = [{**g, 'sufficient': g['free_bytes'] >= plan['required_free_bytes_per_gpu']} for g in used]
        shortages.extend(f"{layout}: GPU {g['index']} free {g['free_bytes'] / GIB:.2f} GiB < required "
                         f"{plan['required_free_bytes_per_gpu'] / GIB:.2f} GiB" for g in plan['gpus_used'] if not g['sufficient'])
        plans[layout] = plan
    return {'checkpoint': checkpoint, 'inventory': inventory, 'layouts': plans, 'shortages': shortages}


def torchrun_command(python, layout, gpus, checkpoint, output, prompt_tokens, decode_steps, repeats, modes, seq_length):
    spec = LAYOUTS[layout]
    return [python, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={spec["gpus"]}',
            'tools/resharding/correctness/router_determinism.py',
            '--tensor-model-parallel-size', str(spec['tp']), '--expert-tensor-parallel-size', str(spec['tp']),
            '--expert-model-parallel-size', str(spec['ep']), '--pipeline-model-parallel-size', '1',
            '--micro-batch-size', '1', '--global-batch-size', str(spec['gpus'] // spec['tp']),
            '--live-model-preset', 'deepseek-v2-lite', '--live-router-mode', 'native',
            '--live-prompt-tokens', str(prompt_tokens),
            '--rd-output', str(output), '--rd-decode-steps', str(decode_steps), '--rd-repeats', str(repeats),
            '--rd-modes', ','.join(modes),
            '--num-layers', '27', '--hidden-size', '2048', '--ffn-hidden-size', '10944',
            '--num-attention-heads', '16', '--num-query-groups', '16', '--kv-channels', '16',
            '--multi-latent-attention', '--kv-lora-rank', '512', '--v-head-dim', '128', '--qk-head-dim', '128',
            '--qk-layernorm', '--qk-pos-emb-head-dim', '64', '--num-experts', '64',
            '--moe-layer-freq', '([0]+[1]*26)', '--moe-ffn-hidden-size', '1408', '--moe-grouped-gemm',
            '--moe-router-score-function', 'softmax', '--moe-router-topk', '6',
            '--moe-router-topk-scaling-factor', '1.0', '--moe-router-pre-softmax',
            '--moe-shared-expert-intermediate-size', '2816', '--moe-token-dispatcher-type', 'alltoall',
            '--seq-length', str(seq_length), '--max-position-embeddings', str(seq_length),
            '--vocab-size', '102400', '--make-vocab-size-divisible-by', '3200',
            '--normalization', 'RMSNorm', '--norm-epsilon', '1e-6', '--no-persist-layer-norm', '--swiglu', '--bf16',
            '--untie-embeddings-and-output-weights', '--position-embedding-type', 'rope', '--rotary-percent', '1.0',
            '--rotary-base', '10000', '--rotary-scaling-factor', '40', '--mscale', '0.707', '--mscale-all-dim', '0.707',
            '--attention-softmax-in-fp32', '--no-masked-softmax-fusion',
            '--load', checkpoint, '--ckpt-format', 'torch_dist',
            '--disable-bias-linear', '--no-gradient-accumulation-fusion', '--no-rope-fusion',
            '--hidden-dropout', '0.0', '--attention-dropout', '0.0', '--transformer-impl', 'local', '--seed', '1234']


def torchrun_env(gpus, mps_pipe):
    # Same runtime environment as the live launcher, plus the deterministic-mode prerequisites
    # (pinned NCCL algorithm, cuBLAS workspace) and the project's MPS isolation pipe.
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)), 'NCCL_DEBUG': 'WARN',
            'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'CUDA_DEVICE_MAX_CONNECTIONS': '8',
            'NCCL_ALGO': 'Ring', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_MPS_PIPE_DIRECTORY': mps_pipe,
            'PYTHONDONTWRITEBYTECODE': '1'}


def layout_verdict(result):
    """One line per mode from a layout JSON written by router_determinism.py."""
    parts = []
    for mode, row in result['modes'].items():
        if row.get('error'):
            parts.append(f"{mode}=error({row['error'].splitlines()[-1][:80]})")
        elif row['bitwise']:
            parts.append(f'{mode}=bitwise')
        else:
            mismatch = row['mismatch']
            detail = '; '.join(f"{tensor} step={m['step']}" + (f" layer={m['layer']}" if 'layer' in m else '')
                               if (m := mismatch[tensor]) else f'{tensor} same'
                               for tensor in ('logits', 'routing_map', 'probs'))
            parts.append(f"{mode}=differs({detail}; rank={mismatch['rank']}; identical-input steps={mismatch['steps_with_identical_input']})")
    return ', '.join(parts)


def summarize(layouts, results, modes):
    """PASS only when every requested layout completed, parameters matched the preflight formula and the
    deterministic mode (when requested) reproduced logits and routing bitwise on every rank."""
    rows, passed = {}, True
    for layout in layouts:
        result = results.get(layout)
        if result is None:
            rows[layout] = {'status': 'not_run'}
            passed = False
            continue
        deterministic_ok = 'deterministic' not in modes or result['modes'].get('deterministic', {}).get('bitwise') is True
        ok = result['parameter_bytes_match'] and all(not r.get('error') for r in result['modes'].values()) and deterministic_ok
        rows[layout] = {'status': 'passed' if ok else 'failed', 'verdict': layout_verdict(result),
                        'parameter_bytes_match': result['parameter_bytes_match'],
                        'default_mode_bitwise': result['modes'].get('default', {}).get('bitwise'),
                        'default_routing_map_differs': result['modes'].get('default', {}).get('routing_map_differs')}
        passed = passed and ok
    return {'result': 'PASS' if passed else 'FAIL', 'layouts': rows, 'modes': list(modes),
            'suspect_operators_when_default_differs': SUSPECT_OPERATORS,
            'scope': 'native-router same-input repeatability per layout; TP2 vs TP4 are not compared'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node', choices=('SL3060', 'SL3061'), required=True)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--layouts', default='tp2,tp4')
    parser.add_argument('--modes', default='default,deterministic')
    parser.add_argument('--prompt-tokens', type=int, default=32)
    parser.add_argument('--decode-steps', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seq-length', type=int, default=128)
    parser.add_argument('--budget-seconds', type=float, default=900)
    args = parser.parse_args()
    gpus = [int(x) for x in args.gpus.split(',')]
    layouts = [x for x in args.layouts.split(',') if x]
    modes = [x for x in args.modes.split(',') if x]
    if not (layouts and set(layouts) <= set(LAYOUTS) and modes and set(modes) <= {'default', 'deterministic'}):
        raise SystemExit('invalid --layouts/--modes')
    if len(gpus) < max(LAYOUTS[l]['gpus'] for l in layouts) or args.prompt_tokens + args.decode_steps >= args.seq_length:
        raise SystemExit('not enough GPUs for the requested layouts, or prompt+decode exceeds --seq-length')
    started, deadline = time.monotonic(), time.monotonic() + args.budget_seconds
    run = args.log_dir
    status = {'check': 'router_determinism', 'node': args.node, 'sha': args.sha, 'result': 'FAIL',
              'quick_mode': True, 'timing_valid': False, 'accepted': False}
    restore, child, results = False, None, {}

    def git(*argv):
        return subprocess.check_output(['git', '-C', str(args.repo), *argv], text=True, stderr=subprocess.STDOUT).strip()

    try:
        if not re.fullmatch('[0-9a-f]{40}', args.sha):
            raise RuntimeError('delivery must pin a full SHA')
        if git('status', '--porcelain', '--untracked-files=no'):
            raise RuntimeError('tracked repository files are dirty')
        git('cat-file', '-e', BASE + '^{commit}')
        restore = True
        git('checkout', '--detach', args.sha)
        if git('rev-parse', 'HEAD') != args.sha:
            raise RuntimeError('checkout identity mismatch')
        status['preflight'] = preflight(gpus, layouts, args.seq_length, args.checkpoint)
        (run / 'preflight.json').write_text(json.dumps(status['preflight'], indent=2) + '\n')
        print(json.dumps({'preflight': status['preflight']}, indent=2), flush=True)
        if status['preflight']['shortages']:
            raise RuntimeError('insufficient GPU memory: ' + '; '.join(status['preflight']['shortages']))
        mps_pipe = str(run / 'absent_mps_pipe')
        for layout in layouts:
            output = run / layout
            output.mkdir()
            used = gpus[:LAYOUTS[layout]['gpus']]
            command = torchrun_command(args.python, layout, used, args.checkpoint, output, args.prompt_tokens,
                                       args.decode_steps, args.repeats, modes, args.seq_length)
            (output / 'command.txt').write_text(shlex.join(command) + '\n')
            print(f'Starting {layout} on GPUs {used}', flush=True)
            with (output / 'torchrun.log').open('wb') as log:
                child = subprocess.Popen(command, cwd=args.repo / 'code/current', env=torchrun_env(used, mps_pipe),
                                         stdout=log, stderr=subprocess.STDOUT)
                code = child.wait(timeout=max(1, deadline - time.monotonic()))
            child = None
            result_path = output / 'result.json'
            if code or not result_path.exists():
                raise RuntimeError(f'{layout} torchrun exited {code}; see {output / "torchrun.log"}')
            results[layout] = json.loads(result_path.read_text())
        status.update(summarize(layouts, results, modes))
    except BaseException:
        status.update(result='FAIL', error=traceback.format_exc())
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if restore:
            try:
                git('checkout', '--detach', BASE)
                if git('rev-parse', 'HEAD') != BASE:
                    raise RuntimeError('restore identity mismatch')
                status['restored_commit'] = BASE
            except BaseException:
                status.update(result='FAIL', restore_error=traceback.format_exc())
        status['elapsed_s'] = time.monotonic() - started
        if status['elapsed_s'] > args.budget_seconds:
            status.update(result='FAIL', budget_error=f'exceeded {args.budget_seconds} s budget')
        try:
            (run / 'summary.json').write_text(json.dumps(status, indent=2) + '\n')
            archive = run.parent / f'{run.name}.tar.gz'
            with tarfile.open(archive, 'x:gz') as output:
                output.add(run, arcname=run.name)
            status['archive'] = str(archive)
        except BaseException:
            status.update(result='FAIL', package_error=traceback.format_exc())
        (run / 'final.json').write_text(json.dumps(status, indent=2) + '\n')
        code = 0 if status['result'] == 'PASS' else 1
        (run / 'exit_code').write_text(f'{code}\n')
        verdicts = '; '.join(f"{l}: {r.get('verdict', r['status'])}" for l, r in status.get('layouts', {}).items())
        if not verdicts and status.get('error'):
            verdicts = status['error'].strip().splitlines()[-1]
        print(json.dumps(status, indent=2), flush=True)
        print(f"ROUTER DETERMINISM result={status['result']} ({verdicts}) elapsed={status['elapsed_s']:.0f}s "
              f"logs={run} archive={status.get('archive')}", flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
