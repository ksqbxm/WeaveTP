"""Step 9 quick check (quick_check.sh --step 9): three short fail-slow launches and one fail-stop restart launch,
shared GPUs, correctness mode.

weavetp, anchortp and restart in turn, level W on SL3061 GPUs 0,1,4,5, warmup 20 s, D 30 s, tail 20 s, detection
after 10 s, a toy request pool the deadline cuts; then restart in the emulate fail-stop cycle (toy requests that all
complete; EDP0 reloads TP2 and then TP4 from disk, reference_audit='skipped'). Per launch: the launch passed and no
failslow_hog.py process is left on this node; on SL3060 also, for fail-slow, the event order and requests left
unfinished at the deadline (the wall-clock cut). The toy inputs sit in `<output>/inputs_<scenario>` (the gate compares
both nodes' serving configs, paths included). Writes `<output>/<node>/summary.json`.

--method weavetp|restart|ignore|anchortp runs only that method's fail-slow launch (W). On SL3060 each fail-slow launch
also gets a summary from failslow_result.json's existing fields ('缺失' where a launch has none): the restart TP2/TP4
reload times, both switches' durations, the EDP0/EDP1 decode steps inside them, replayed_tokens.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from common import NODES, require, require_megatron_helpers, write_json
from failslow_inject import hog_pids
from run_probe import repository_commit, run_case
from safety import check_data_space
from serving_config import (
    FAILSLOW_MARGIN_S,
    FAILSLOW_RELOAD_S,
    FAILSLOW_STARTUP_S,
    apply_environment,
    failslow_launch_seconds,
    failslow_settings,
    serving_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.e2e.requests_io import sha256_file  # noqa: E402
from tools.resharding.e2e.toy_inputs import write_toy_inputs  # noqa: E402

QUICK = {'warmup_s': 20.0, 'slow_s': 30.0, 'tail_s': 20.0, 'detect_s': 10.0}
METHODS, LEVEL, FAULT_GPUS, REQUESTS = ('weavetp', 'anchortp', 'restart'), 'W', (0, 1, 4, 5), 2000
FAILSTOP = {'count': 64, 'output_scale': 8, 'rejoin_seconds': 5.0}  # the fail-stop restart launch: outputs 24-72 tokens
LAUNCHES = METHODS + ('failstop_restart',)
MISSING = '缺失'


def settings(method):
    return failslow_settings(method=method, level=LEVEL, fault_gpus=list(FAULT_GPUS), mode='emulate', **QUICK)


def launch_names(method=None):
    """The launches of one quick check: all four, or only the given method's fail-slow launch."""
    return LAUNCHES if method is None else (method,)


def budget_seconds(method=None):
    """The fail-slow launches, then the fail-stop restart launch (startup, two reloads, margin), plus 120 s; with a
    --method, that one fail-slow launch plus 120 s."""
    if method is not None:
        return failslow_launch_seconds(settings(method)) + 120
    failstop = FAILSLOW_STARTUP_S + 2 * FAILSLOW_RELOAD_S + FAILSLOW_MARGIN_S
    return sum(failslow_launch_seconds(settings(method)) for method in METHODS) + failstop + 120


def launch_summary(result):
    """The quick-check digest of one fail-slow launch from failslow_result.json's existing fields."""
    switches = {(s['domain'], s['direction']): s for s in result['switches']}
    loaded, worker = result.get('restart_loaded_s') or {}, result.get('worker_events_s') or {}

    def since(end, start):
        return end - start if isinstance(end, (int, float)) and isinstance(start, (int, float)) else MISSING

    def switch_seconds(direction):
        milestones = switches.get(('EDP0', direction), {}).get('milestones', {})
        return since(milestones.get('source_release_done'), milestones.get('switch_trigger'))
    return {'restart_reload_s': {'tp2': since(loaded.get('shrink'), worker.get('fail')),
                                 'tp4': since(loaded.get('expand'), worker.get('rejoin_start'))},
            'switch_s': {'shrink': switch_seconds('shrink'), 'expand': switch_seconds('expand')},
            'decode_steps_in_switch': {f'{domain}/{direction}': switches.get((domain, direction), {}).get('overlap_steps', MISSING)
                                       for domain in ('EDP0', 'EDP1') for direction in ('shrink', 'expand')},
            'replayed_tokens': result.get('replayed_tokens', MISSING)}


def toy_inputs(directory, count=REQUESTS):
    """Deterministic requests long enough to keep every replica busy across the timeline (prompts 16-48, outputs 32-128)."""
    directory.mkdir(parents=True, exist_ok=True)
    rows = [{'id': f'q{i}', 'source': 'toy', 'src_index': i, 'prompt_token_ids': [100 + (i * 7 + k) % 1000 for k in range(16 + i % 33)],
             'output_token_ids': [0] * (32 + (i * 13) % 97), 'tokenizer': 'toy', 'sha256_text': 'x'} for i in range(count)]
    for row in rows:
        row.update(input_len=len(row['prompt_token_ids']), output_len=len(row['output_token_ids']))
    requests = directory / 'requests.jsonl'
    requests.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    workload = {'workload_id': 'failslow_quick', 'arrival': 'continuous', 'seed': 1,
                'segments': [{'index': 0, 'kind': 'short', 'request_ids': [r['id'] for r in rows]}]}
    (directory / 'workload.json').write_text(json.dumps(workload), encoding='utf-8')
    (directory / 'plan.json').write_text(json.dumps({'version': 'v2', 'request_file_sha256': sha256_file(requests), 'switches': []}))
    return {'requests': requests, 'workload': directory / 'workload.json', 'plan': directory / 'plan.json'}


def gate_checks(path):
    """SL3060: the gate's failslow_result.json of one launch."""
    result = json.loads(path.read_text(encoding='utf-8'))
    checks = {'event_order': result['check']['event_order'], 'cut_by_deadline': result['requests_unfinished'] > 0,
              'requests_completed': result['requests_completed'], 'deadline_overrun_s': result['deadline_overrun_s'],
              'summary': launch_summary(result)}
    checks['passed'] = checks['event_order'] and checks['cut_by_deadline']
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--node', choices=NODES, required=True)
    parser.add_argument('--suite-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--correctness-only', action='store_true')
    parser.add_argument('--method', choices=METHODS + ('ignore',), help='only this fail-slow launch (W)')
    parser.add_argument('--budget-seconds', type=float, default=None)
    parser.add_argument('--min-free-disk-gb', type=float, default=1)
    parser.add_argument('--profile-file', required=True, help='frozen Step 3 profile.json, same path on both nodes')
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT', '/data/models/DeepSeek-V2-Lite-megatron-v2'))
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--master', default='10.60.14.1')
    parser.add_argument('--port-base', type=int, default=29890)
    args = parser.parse_args(argv)
    require(args.quick and args.correctness_only, 'Step 9 runs only as the quick correctness check')
    args.budget_seconds = budget_seconds(args.method) if args.budget_seconds is None else args.budget_seconds
    started = time.monotonic()
    deadline = started + args.budget_seconds
    root = args.output / args.node
    root.mkdir(parents=True, exist_ok=False)
    summary = {'step': 9, 'node': args.node, 'suite_id': args.suite_id, 'methods': list(METHODS) if args.method is None else [args.method],
               'level': LEVEL, 'settings': QUICK,
               'quick_mode': True, 'correctness_only': True, 'shared_gpu_mode': True, 'timing_valid': False, 'accepted': False,
               'all_checks_passed': False, 'budget_s': args.budget_seconds, 'launches': {}}
    try:
        require_megatron_helpers()
        check_data_space(args.min_free_disk_gb, root / 'disk_space.jsonl')
        inputs = {'failslow': toy_inputs(args.output / 'inputs_failslow'),
                  'failstop': write_toy_inputs(args.output / 'inputs_failstop', failstop=True, **FAILSTOP)}
        apply_environment('correctness')
        for index, name in enumerate(launch_names(args.method)):
            method, scenario = ('restart', 'failstop') if name == 'failstop_restart' else (name, 'failslow')
            launch = f'{args.suite_id}_{method}_{scenario}'
            failslow = scenario == 'failslow'
            config = serving_config(node=args.node, launch=launch, mode='emulate', output=root / launch, master=args.master,
                                    port=args.port_base + index, commit=repository_commit(), remaining=deadline - time.monotonic(),
                                    min_free_disk_gb=args.min_free_disk_gb, quick=True, shared_gpu=True, run_mode='correctness',
                                    method=method, scenario=scenario, checkpoint=args.checkpoint, seq_length=args.seq_length,
                                    profile_file=args.profile_file, reference_root=None,
                                    requests=inputs[scenario]['requests'], workload=inputs[scenario]['workload'],
                                    plan=inputs[scenario]['plan'], b_max=8, capacity_tokens=None, failslow=settings(method) if failslow else None)
            print(f'Starting {launch} ' + (f'(fail-slow {LEVEL}, {REQUESTS} toy requests, deadline-cut)' if failslow else
                                           f"(emulate fail-stop, {FAILSTOP['count']} toy requests)"), flush=True)
            outcome = run_case(config)
            record = {'status': outcome['status'], 'leftover_hogs': hog_pids(), 'output': str(root / launch)}
            if args.node == NODES[0] and outcome['status'] == 'passed' and failslow:
                record['gate'] = gate_checks(root / launch / 'failslow_result.json')
            record['passed'] = outcome['status'] == 'passed' and not record['leftover_hogs'] and record.get('gate', {}).get('passed', True)
            summary['launches'][name] = record
            require(record['passed'], f'{name} launch failed its checks: {record}')
        summary['all_checks_passed'] = True
    except BaseException:
        summary['error'] = traceback.format_exc()
        (root / 'error.txt').write_text(summary['error'], encoding='utf-8')
        raise
    finally:
        summary['elapsed_s'] = time.monotonic() - started
        write_json(root / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('launches', 'all_checks_passed', 'elapsed_s')}, indent=2, default=str), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
