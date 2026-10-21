"""Step 6 quick runner, layered (all layers: `accepted` stays false, correctness only).

  --layer a   fake backend full loop (scaling + fail-stop) and the e2e CPU tests; no GPU (A part).
  --layer b1  real model, one node, TP4 (8 GPUs): block-KV serving against the static-KV path
              (logits nrmse/cosine, M1 step times, pool accounting), torch split-KV decode.
  --layer b2  real model, one node, deterministic mode: batched bitwise check - two independent TP2
              processes (GPUs 0-3 and 4-7) and one TP4 process, each running the schedule twice.
  --layer b3  two nodes: the serving driver inside the Step 5 fail-stop worker, fault -> shrink ->
              replay -> rejoin -> expand on DeepSeek-V2-Lite (emulate by default, real with --mode real).
  --layer inflight  b3 with 200 long toy requests (96-288 output tokens), so EDP0 still serves requests when the
              expand comes: the joining ranks decode handed-over requests, the expand KV check covers them, and
              analyze_failstop_e2e must accept the gate's result files (fails when nothing was in flight).
Writes `<output>/<node>/summary.json` (quick-driver flags), the layer's own files and `error.txt` on failure; the
two-node layers' toy inputs sit in `<output>/inputs_<layer>` so both nodes' serving configs name the same paths.
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import analyze_failstop_e2e
import model_memory
import torch
from common import NODES, SERVING_PROBE, require, require_megatron_helpers, write_json
from determinism import compare_records
from run_model import environment, format_table
from run_probe import repository_commit, run_case
from run_topology import cases
from safety import check_data_space
from serving_config import apply_environment, serving_config

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT / 'examples' / 'rl'))

import run_e2e_scaling  # noqa: E402

from tools.resharding.correctness.router_determinism_driver import model_arguments  # noqa: E402
from tools.resharding.e2e.toy_inputs import write_toy_inputs  # noqa: E402

RUNS = {
    'scaling': dict(count=24, failstop=False, policy='teacher_forcing', world_size=8, expected=['2->4', '4->2'],
                    switches=[{'direction': '2->4', 'anchor': {'kind': 'before_admit', 'seq': 12}},
                              {'direction': '4->2', 'anchor': {'kind': 'after_complete', 'seq': 14}}]),
    'failstop': dict(count=20, failstop=True, policy='greedy', world_size=16, expected=['shrink', 'expand'], switches=None),
}
TESTS = ('tests/unit_tests/e2e',)
GIB = 1 << 30
TWO_NODE_LAYERS = ('b3', 'inflight')
INFLIGHT = {'count': 200, 'output_scale': 32}  # ~2000 decode steps: EDP0 still serves when the expand comes


# ---- layer a -------------------------------------------------------------------------------------------------
def fake_run(name, root):
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
    require(importlib.util.find_spec('pytest') is not None, 'pytest is not installed in this Python environment')
    argv = [sys.executable, '-B', '-m', 'pytest', '-q', '-o', 'addopts=', '--noconftest', '-p', 'no:cacheprovider', *TESTS]
    log = root / 'pytest.log'
    with log.open('wb') as output:
        code = subprocess.run(argv, cwd=CODE_ROOT, stdout=output, stderr=subprocess.STDOUT,
                              timeout=max(1.0, deadline - time.monotonic())).returncode
    lines = log.read_text(encoding='utf-8', errors='replace').strip().splitlines()
    summary = lines[-1] if lines else 'pytest produced no output'
    return {'exit_code': code, 'passed': code == 0 and re.search(r'\d+ passed', summary) is not None, 'summary_line': summary, 'log': str(log)}


def layer_a(args, root, deadline):
    runs = {name: fake_run(name, root) for name in RUNS}
    tests = run_tests(root, deadline - 15)
    return {'runs': runs, 'tests': tests, 'passed': all(r['check_passed'] for r in runs.values()) and tests['passed']}


# ---- layers b1 / b2 ------------------------------------------------------------------------------------------
def torchrun(nproc, rendezvous=None):
    command = [sys.executable, '-m', 'torch.distributed.run', f'--nproc_per_node={nproc}', '--nnodes=1']
    command += ['--standalone'] if rendezvous is None else ['--rdzv-backend=c10d', f'--rdzv-endpoint=127.0.0.1:{rendezvous}', f'--rdzv-id=step6_{rendezvous}']
    return command + ['tools/resharding/failstop/serving_layers.py']


def launch_layers(name, layout, gpus, out, args, *, deterministic, extra=(), rendezvous=None, mps_pipe):
    """One serving_layers torchrun; returns the Popen and the log path (the caller waits)."""
    argv = torchrun(len(gpus), rendezvous) + model_arguments(layout, args.checkpoint, 8, args.seq_length) + \
        ['--sl-layer', name, '--sl-output', str(out), '--sl-attention', 'torch', *extra] + (['--deterministic-mode'] if deterministic else [])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'command.txt').write_text(shlex.join(argv) + '\n', encoding='utf-8')
    log = (out / 'torchrun.log').open('wb')
    env = environment('reference' if deterministic else 'timing', gpus, mps_pipe)
    child = subprocess.Popen(argv, cwd=CODE_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return child, out / 'torchrun.log'


def wait_children(children, deadline):
    for child, log in children:
        remaining = max(1.0, deadline - time.monotonic())
        try:
            code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=30)
            raise RuntimeError(f'torchrun exceeded the budget; see {log}')
        require(code == 0, f'torchrun exited {code}; see {log}')


def preflight(args, root, layouts, gpus=range(8)):
    memory = model_memory.preflight(args.node, args.seq_length, gpus, layouts)
    write_json(root / 'memory_preflight.json', memory)
    print(format_table(memory), flush=True)
    require(not memory['run_shortages'], 'insufficient GPU memory: ' + '; '.join(memory['run_shortages']))
    return memory


def layer_b1(args, root, deadline):
    require_megatron_helpers()
    preflight(args, root, ['tp4'])
    child, log = launch_layers('b1', 'tp4', list(range(8)), root / 'b1', args, deterministic=False, mps_pipe=str(root / 'absent_mps_pipe'))
    wait_children([(child, log)], deadline)
    result = json.loads((root / 'b1' / 'result.json').read_text(encoding='utf-8'))
    served = next((r for r in result['ranks'] if 'batched' in r), None)
    require(served is not None, 'no rank served the probe request')
    rank0 = result['ranks'][0]
    return {'passed': result['passed'], 'attention_impl': rank0['attention_impl'], 'nrmse_max': max(served['alone']['nrmse_max'], served['batched']['nrmse_max']),
            'cosine_min': min(served['alone']['cosine_min'], served['batched']['cosine_min']), 'top1_agreement': served['batched']['top1_agreement'],
            'm1_ms': rank0['m1_ms'], 'memory': rank0['memory'], 'thresholds': result['thresholds'], 'result': str(root / 'b1' / 'result.json')}


def cross_process_bitwise(a, b, ranks):
    mismatches = []
    for rank in ranks:
        left = torch.load(a / f'records_rank_{rank}.pt', weights_only=False)['records']
        right = torch.load(b / f'records_rank_{rank}.pt', weights_only=False)['records']
        if len(left) != len(right):
            mismatches.append({'rank': rank, 'reason': f'{len(left)} vs {len(right)} forwards'})
            continue
        first = next((c for c in (compare_records(x, y) for x, y in zip(left, right)) if not c['equal']), None)
        if first is not None:
            mismatches.append({'rank': rank, **first})
    return {'passed': not mismatches, 'mismatches': mismatches[:3], 'ranks': len(ranks)}


def layer_b2(args, root, deadline):
    require_megatron_helpers()
    memory = preflight(args, root, ['tp2', 'tp4'])
    need = memory['reference_runs']['tp2']['required_free_bytes_per_gpu']
    short = [g['index'] for g in memory['inventory'][4:8] if g['free_bytes'] < need]
    require(not short, f'the second TP2 process needs {need / GIB:.2f} GiB free on GPUs 4-7; short on {short}')
    mps = str(root / 'absent_mps_pipe')
    pair = [launch_layers('b2', 'tp2', list(range(0, 4)), root / 'b2' / 'tp2_a', args, deterministic=True, rendezvous=args.port_base, mps_pipe=mps),
            launch_layers('b2', 'tp2', list(range(4, 8)), root / 'b2' / 'tp2_b', args, deterministic=True, rendezvous=args.port_base + 1, mps_pipe=mps)]
    wait_children(pair, deadline)
    wait_children([launch_layers('b2', 'tp4', list(range(8)), root / 'b2' / 'tp4', args, deterministic=True, mps_pipe=mps)], deadline)
    results = {name: json.loads((root / 'b2' / name / 'result.json').read_text(encoding='utf-8')) for name in ('tp2_a', 'tp2_b', 'tp4')}
    cross = cross_process_bitwise(root / 'b2' / 'tp2_a', root / 'b2' / 'tp2_b', range(4))
    write_json(root / 'b2' / 'cross_process.json', cross)
    rank0 = results['tp4']['ranks'][0]
    return {'passed': all(r['passed'] for r in results.values()) and cross['passed'] and all(r['manifests_identical'] for r in results.values()),
            'in_process_bitwise': {name: r['passed'] for name, r in results.items()}, 'cross_process_tp2_bitwise': cross['passed'],
            'forwards': {name: r['ranks'][0]['forwards'] for name, r in results.items()},
            'crossed_block_boundary': rank0['crossed_block_boundary'], 'dummy_forwards_tp4_rank0': rank0['dummy_forwards'],
            'attention_impl': rank0['attention_impl'], 'results': {name: str(root / 'b2' / name / 'result.json') for name in results}}


# ---- layer b3 ------------------------------------------------------------------------------------------------
def b3_passed(outcome, metrics, method):
    """Gate metrics (SL3060 only): every request completed, the result self-audit and the expand KV check passed;
    AnchorTP additionally paused EDP0 (no foreground step) and moved no delta on both switches."""
    if outcome['status'] != 'passed':
        return False
    if metrics is None:
        return True
    passed = metrics['check_passed'] and metrics['kv_check_passed']
    if method == 'anchortp':
        passed = passed and all(s['overlap_steps'] == 0 and s['kv_delta_bytes'] == 0 for s in metrics['edp0_switches'])
    return bool(passed)


def layer_b3(args, root, deadline, count=None, output_scale=1):
    require_megatron_helpers()
    require(Path(args.profile_file or '').is_file(), f'--profile-file (frozen Step 3 profile) is required for b3; not found: {args.profile_file}')
    # The gate compares the nodes' serving configs (paths included): the inputs live outside the node directories.
    count = args.b3_requests if count is None else count
    inputs = write_toy_inputs(args.output / f'inputs_{args.layer}', count=count, failstop=True, rejoin_seconds=args.b3_rejoin_seconds,
                              output_scale=output_scale)
    case = cases(args.mode, 1, 1, args.suite_id)[0]
    apply_environment('correctness')
    config = serving_config(node=args.node, launch=case['launch'], mode=case['mode'], output=root / case['launch'], master=args.master,
                            port=args.port_base, commit=repository_commit(), remaining=deadline - time.monotonic(),
                            min_free_disk_gb=args.min_free_disk_gb, quick=True, shared_gpu=True, run_mode='correctness',
                            method=args.method, scenario='failstop', checkpoint=args.checkpoint, seq_length=args.seq_length,
                            profile_file=args.profile_file or '', reference_root=args.reference_root, requests=inputs['requests'],
                            workload=inputs['workload'], plan=inputs['plan'], b_max=4, capacity_tokens=None)
    print(f"Starting {case['launch']} mode={case['mode']} method={args.method} (serving worker, {count} toy requests)", flush=True)
    outcome = run_case(config)
    # The gate (SL3060) owns the metrics and the result files; SL3061's controller result carries the status only.
    result = json.loads((Path(config['output']) / 'controller_result.json').read_text(encoding='utf-8'))
    metrics = result['metrics'] if args.node == NODES[0] else None
    return {'launch': case['launch'], 'mode': case['mode'], 'method': args.method, 'status': outcome['status'], 'output': config['output'],
            'metrics': metrics,
            'files': {name: str(Path(config['output']) / name) for name in ('result.json', 'check.json', 'events_actual.json')} if metrics else None,
            'passed': b3_passed(outcome, metrics, args.method)}


def handed_over(result):
    """Requests running on EDP0 at its expand cutover (last admit before it on EDP0, completed after it): the rows the
    joining ranks decode without having seen their admits."""
    cutover = next(s['milestones']['cutover'] for s in result['switches'] if s['domain'] == 'EDP0' and s['direction'] == 'expand')
    seqs = []
    for row in result['requests']:
        before = [a for a in row['admits'] if a['t'] < cutover]
        if before and before[-1]['replica'] in (0, 1) and row['t_done'] > cutover:
            seqs.append(row['seq'])
    return seqs


def layer_inflight(args, root, deadline):
    record = layer_b3(args, root, deadline, **INFLIGHT)
    if record['metrics'] is None or record['status'] != 'passed':  # SL3061 (the gate owns the result files), or a failed launch
        return record
    output = Path(record['output'])
    seqs = handed_over(json.loads((output / 'result.json').read_text(encoding='utf-8')))
    analysis = root / 'analysis'
    code = analyze_failstop_e2e.main([str(output), '--out', str(analysis)])
    rejected = json.loads((analysis / 'rejected.json').read_text(encoding='utf-8'))
    record.update(inflight_at_expand=len(seqs), inflight_seqs=seqs[:16], analysis={'exit_code': code, 'rejected': rejected, 'out': str(analysis)},
                  passed=record['passed'] and bool(seqs) and code == 0 and not rejected)
    if not seqs:
        record['inflight_error'] = 'no request was running on EDP0 at the expand cutover: the workload drained first'
    return record


LAYERS = {'a': layer_a, 'b1': layer_b1, 'b2': layer_b2, 'b3': layer_b3, 'inflight': layer_inflight}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--node', choices=NODES, required=True)
    parser.add_argument('--suite-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--budget-seconds', type=float, default=300)
    parser.add_argument('--min-free-disk-gb', type=float, default=1)
    parser.add_argument('--layer', choices=tuple(LAYERS), default='a')
    parser.add_argument('--mode', choices=('emulate', 'real'), default='emulate', help='b3/inflight only')
    parser.add_argument('--method', choices=('weavetp', 'anchortp'), default='weavetp', help='b3/inflight only: recovery method under test')
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', '/data/models/DeepSeek-V2-Lite-megatron-v2'))
    parser.add_argument('--seq-length', type=int, default=512, help='b1/b2/b3: max_position_embeddings (rotary table and static baseline)')
    parser.add_argument('--profile-file', help='b3: frozen Step 3 profile.json')
    parser.add_argument('--reference-root', help='b3: run_model.py output root with tp2/ and tp4/ reference parameter hashes')
    parser.add_argument('--master', default='10.60.14.1')
    parser.add_argument('--port-base', type=int, default=29840)
    parser.add_argument('--b3-requests', type=int, default=24)
    parser.add_argument('--b3-rejoin-seconds', type=float, default=5.0)
    args = parser.parse_args(argv)
    require(args.quick and args.correctness_only, 'the Step 6 layers run only as the quick correctness check')
    started = time.monotonic()
    deadline = started + args.budget_seconds
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=False)
    if args.layer != 'a':
        check_data_space(args.min_free_disk_gb, root / 'disk_space.jsonl')
    summary = {'step': 6, 'probe': SERVING_PROBE, 'layer': args.layer, 'node': args.node, 'suite_id': args.suite_id,
               'backend': 'fake' if args.layer == 'a' else 'block', 'attention': 'none' if args.layer == 'a' else 'torch_splitkv',
               'gpus_used': 0 if args.layer == 'a' else 8, 'mode': args.mode if args.layer in TWO_NODE_LAYERS else None,
               'method': args.method if args.layer in TWO_NODE_LAYERS else None,
               'single_node': args.layer not in TWO_NODE_LAYERS, 'quick_mode': True, 'correctness_only': True, 'shared_gpu_mode': args.layer != 'a',
               'timing_valid': False, 'accepted': False, 'all_checks_passed': False, 'seq_length': args.seq_length}
    try:
        summary['layers'] = {args.layer: LAYERS[args.layer](args, root, deadline)}
        summary['all_checks_passed'] = summary['layers'][args.layer]['passed']
        require(summary['all_checks_passed'], f'layer {args.layer} failed; see summary.json')
    except BaseException:
        summary['error'] = traceback.format_exc()
        (root / 'error.txt').write_text(summary['error'], encoding='utf-8')
        raise
    finally:
        summary['elapsed_s'] = time.monotonic() - started
        write_json(root / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('layer', 'layers', 'all_checks_passed', 'elapsed_s')}, indent=2, default=str), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
