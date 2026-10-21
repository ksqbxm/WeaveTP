"""One configuration builder for a serving launch: the Step 6 B3 quick check and the formal run_failstop_e2e
entry build the same dict here, so the quick path and the formal path differ only in the flags below.

run_mode:
  correctness  deterministic mode (--deterministic-mode + pinned environment), bitwise validation, weight hash
               and KV snapshot audits; the quick check and the server correctness runs.
  timing       default mode (no deterministic argument, environment stripped), no audits, no snapshots, exclusive
               GPUs (shared_gpu_mode=false, every GPU of the node idle before the launch), timing_valid=true.
               With shared_gpu (run_failstop_e2e --shared-gpu-ok) the exclusive precheck is replaced by a free-memory
               check, shared_gpu_mode=true, timing_valid stays true and result.json says gpu_sharing='shared'.
scenario:
  failstop     fail at completed_count >= ceil(0.25 N), rejoin after the plan's seconds (both methods).
  nofault      the same request set on the fixed TP4 x 2 layout: the calibration run and the no-fault baseline.
"""

import os
import sys
from pathlib import Path

import model_memory
from common import DEFAULTS, DETERMINISTIC_ENVIRONMENT, SERVING_PROBE, require
from safety import check_gpu_processes, gpu_snapshot, mps_snapshot, pidfd_implementations

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))

from tools.resharding.correctness.router_determinism_driver import model_arguments  # noqa: E402
from tools.resharding.e2e.methods import METHODS  # noqa: E402

GIB = 1 << 30
GEOMETRY = {'layers': 27, 'heads': 16, 'group_heads': 4, 'k_dim': 192, 'v_dim': 128}
RUN_MODES = ('correctness', 'timing')
SCENARIOS = ('failstop', 'nofault')


def worker_arguments(checkpoint, prompt_tokens, seq_length, run_mode):
    argv = model_arguments('tp4', checkpoint, prompt_tokens, seq_length)
    argv[argv.index('--global-batch-size') + 1] = '4'
    return argv + (['--deterministic-mode'] if run_mode == 'correctness' else [])


def apply_environment(run_mode):
    """The launcher's own environment, inherited by the supervisors and workers it spawns."""
    for key in ('PYTORCH_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF'):
        os.environ.pop(key, None)
    if run_mode == 'correctness':
        os.environ.update(DETERMINISTIC_ENVIRONMENT)
    else:
        for key in DETERMINISTIC_ENVIRONMENT:
            os.environ.pop(key, None)


def require_exclusive_gpus(log, mps_owner=None):
    """A timing launch refuses to start unless every GPU of this node is idle and free of foreign processes;
    the resident MPS server owned by `mps_owner` (environment MPS_OWNER) is the one tolerated exception."""
    snapshot = gpu_snapshot(log)
    mps = mps_snapshot(log, snapshot, shared_gpu_mode=False, mps_owner=mps_owner)
    foreign = check_gpu_processes(snapshot, mps, shared_gpu_mode=False, mps_owner=mps_owner)
    exempt = {p['pid'] for processes in foreign.values() for p in processes}
    busy = [g['index'] for g in snapshot['gpus'] if any(p['pid'] not in exempt for p in g['processes'])]
    require(not busy, f'GPUs {busy} hold processes; timing runs need all 16 GPUs idle')
    require(not mps['clients'], f"MPS clients present: {mps['clients']}")
    return {**snapshot, 'mps': mps, 'mps_owner_exception': {uuid: processes for uuid, processes in foreign.items() if processes}}


def require_gpu_memory(node, seq_length, b_max):
    """A shared timing launch (--shared-gpu-ok) only checks that every GPU of this node has the free memory of its
    scenario role (model_memory's coexistence windows, KV for b_max sequences of seq_length); other tasks may stay."""
    report = model_memory.preflight(node, seq_length, range(8), [], batch=b_max)
    require(not report['scenario_shortages'], 'insufficient GPU memory for a shared timing launch: ' + '; '.join(report['scenario_shortages']))
    return report


def serving_config(*, node, launch, mode, output, master, port, commit, remaining, min_free_disk_gb, quick, shared_gpu,
                   run_mode, method, scenario, checkpoint, seq_length, profile_file, reference_root, requests, workload, plan,
                   b_max, capacity_tokens, rejoin_note=None):
    require(run_mode in RUN_MODES, f'run_mode must be one of {RUN_MODES}')
    require(method in METHODS, f'method must be one of {sorted(METHODS)}')
    require(scenario in SCENARIOS, f'scenario must be one of {SCENARIOS}')
    correctness = run_mode == 'correctness'
    require(correctness or not quick, 'timing runs are never quick')
    profile = Path(profile_file)
    require(profile.is_file(), f'frozen Step 3 profile is required; not found: {profile}')
    reference = Path(reference_root or '')
    if correctness and node == 'SL3061':  # only the EDP0 node audits migrated targets against the reference hashes
        for tp in (2, 4):
            require(all((reference / f'tp{tp}' / 'reference' / f'parameters_rank_{r}.json').is_file() for r in range(2 * tp)),
                    f'reference parameter hashes for tp{tp} are missing under {reference} (run_model.py reference role)')
    require(remaining > 90, 'budget exhausted before the launch')
    long = min(600, remaining - 30) if not quick else min(240, remaining - 30)
    return {**{k: v for k, v in DEFAULTS.items() if k != 'repeats'},
            'mode': mode, 'rounds': 1, 'launch': launch, 'probe': SERVING_PROBE, 'kind': mode, 'profile_only': False,
            'node': node, 'output': str(output), 'master': master, 'port': port, 'commit': commit, 'mps_pipe': None,
            'method': method, 'scenario': scenario, 'run_mode': run_mode,
            'quick_mode': quick, 'correctness_only': correctness, 'shared_gpu_mode': shared_gpu,
            'mps_owner': None if shared_gpu else os.environ.get('MPS_OWNER'),
            'timing_valid': not correctness, 'deterministic_mode': correctness,
            'min_free_disk_gb': min_free_disk_gb, **pidfd_implementations(),
            'startup_timeout_s': long, 'stage_timeout_s': long, 'reclaim_timeout_s': min(120, remaining - 30),
            'launch_timeout_s': remaining, 'profile_file': str(profile), 'reference_root': str(reference),
            'kv_heads': GEOMETRY['heads'], 'memory_margin_bytes': 2 * GIB, 'memory_limit_bytes': None,
            'model_argv': worker_arguments(checkpoint, 8, seq_length, run_mode),
            'serving': {'requests': str(requests), 'workload': str(workload), 'plan': str(plan), 'b_max': b_max,
                        'decode_policy': 'greedy', 'block_tokens': 64, 'segment_tokens': 4096, 'max_delta_steps': 32,
                        'attention': 'torch', 'growth': {'shrink': 0, 'expand': 2}, 'geometry': GEOMETRY,
                        'capacity_tokens': capacity_tokens},
            'rejoin_note': rejoin_note}
