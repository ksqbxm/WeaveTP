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
  failstop     fail at completed_count >= ceil(0.25 N), rejoin after the plan's seconds (weavetp, anchortp; restart in
               emulate: EDP0 reloads TP2 / TP4 from disk in the same processes).
  nofault      the same request set on the fixed TP4 x 2 layout: the calibration run and the no-fault baseline.
  failslow     co-runners on the fault GPUs for D seconds after a warmup, detection after detect_s, a wall-clock
               deadline D + tail after the injection starts (failslow_probe); emulate mechanics, nothing is killed.
"""

import os
import sys
from pathlib import Path

import model_memory
from common import DEFAULTS, DETERMINISTIC_ENVIRONMENT, FAIL_COORDS, SERVING_PROBE, audits_reference, require
from safety import check_gpu_processes, gpu_snapshot, mps_snapshot, pidfd_implementations

CODE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE_ROOT))

from tools.resharding.correctness.router_determinism_driver import model_arguments  # noqa: E402
from tools.resharding.e2e.methods import METHODS  # noqa: E402

GIB = 1 << 30
GEOMETRY = {'layers': 27, 'heads': 16, 'group_heads': 4, 'k_dim': 192, 'v_dim': 128}
RUN_MODES = ('correctness', 'timing')
SCENARIOS = ('failstop', 'nofault', 'failslow')
FAILSLOW_METHODS = ('weavetp', 'anchortp', 'restart', 'ignore', 'nofault')  # nofault: no injection, the worker runs ignore
SLOW_LEVELS = {'W': 1, 'M': 2, 'S': 3}  # co-runners per slow GPU (Malleus level-1/2/3)
# Fail-slow budget on top of warmup + D + tail (seconds): process start and the initial TP4 load, each of restart's two
# disk reloads, and the teardown margin. The formal D = 600 s launch gets 120 + 600 + 120 + 600 + 2 x 300 + 300 s.
FAILSLOW_STARTUP_S, FAILSLOW_RELOAD_S, FAILSLOW_MARGIN_S = 600, 300, 300


def failslow_settings(*, method, level, fault_gpus, warmup_s, slow_s, tail_s, detect_s, mode):
    """config['failslow']; refuses what the fixed fail-stop topology cannot run: the switching methods release exactly the
    GPUs of FAIL_COORDS (EDP0 position 4 * ep + etp), any other fault set runs only with ignore/nofault."""
    require(method in FAILSLOW_METHODS, f'fail-slow methods are {FAILSLOW_METHODS}')
    require(mode == 'emulate', 'fail-slow keeps every process alive: --mode emulate')
    require(method == 'nofault' or level in SLOW_LEVELS, f'--slow-level must be one of {sorted(SLOW_LEVELS)}')
    require(min(warmup_s, tail_s) >= 0 and 0 < detect_s < slow_s, 'need warmup, tail >= 0 and 0 < detect < slow seconds')
    fault = sorted(set(fault_gpus))
    require(fault and len(fault) == len(fault_gpus) and min(fault) >= 0, f'invalid fault GPU set {fault_gpus}')
    released = sorted(4 * ep + etp for ep, etp in FAIL_COORDS)
    require(method in ('ignore', 'nofault') or fault == released,
            f'{method} releases the fail-stop fault coordinates, i.e. GPUs {released}; other sets only with ignore/nofault')
    return {'label': method, 'level': None if method == 'nofault' else level, 'fault_gpus': fault,
            'warmup_s': warmup_s, 'slow_s': slow_s, 'tail_s': tail_s, 'detect_s': detect_s}


def failslow_launch_seconds(settings):
    """Budget of one fail-slow launch: the timeline, startup, restart's two disk reloads and the margin."""
    return (settings['warmup_s'] + settings['slow_s'] + settings['tail_s'] + FAILSLOW_STARTUP_S + 2 * FAILSLOW_RELOAD_S +
            FAILSLOW_MARGIN_S)


def failslow_stage_seconds(settings):
    """Longest single wait inside a fail-slow launch: an emulate victim waits about D for the spawn, then the reload."""
    return settings['slow_s'] + settings['tail_s'] + 2 * FAILSLOW_RELOAD_S + FAILSLOW_MARGIN_S


def require_method(method, scenario, mode):
    """weavetp/anchortp run every scenario, restart also the emulate fail-stop cycle, ignore/nofault only fail-slow;
    --mode kill only weavetp fail-stop."""
    require(scenario == 'failslow' or method in ('weavetp', 'anchortp') or (method, scenario, mode) == ('restart', 'failstop', 'emulate'),
            f'{method} runs only with --scenario failslow (restart also with --scenario failstop --mode emulate)')
    require(mode != 'kill' or (method, scenario) == ('weavetp', 'failstop'), '--mode kill runs only --method weavetp --scenario failstop')


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
                   b_max, capacity_tokens, rejoin_note=None, failslow=None):
    require(run_mode in RUN_MODES, f'run_mode must be one of {RUN_MODES}')
    require(method in METHODS, f'method must be one of {sorted(METHODS)}')
    require(scenario in SCENARIOS, f'scenario must be one of {SCENARIOS}')
    require((scenario == 'failslow') == (failslow is not None), 'fail-slow settings go with the failslow scenario only')
    require_method(method, scenario, mode)
    correctness = run_mode == 'correctness'
    require(correctness or not quick, 'timing runs are never quick')
    profile = Path(profile_file)
    require(profile.is_file(), f'frozen Step 3 profile is required; not found: {profile}')
    reference = Path(reference_root or '')
    if correctness and node == 'SL3061' and audits_reference(scenario, method):  # EDP0 audits migrated targets
        for tp in (2, 4):
            require(all((reference / f'tp{tp}' / 'reference' / f'parameters_rank_{r}.json').is_file() for r in range(2 * tp)),
                    f'reference parameter hashes for tp{tp} are missing under {reference} (run_model.py reference role)')
    require(remaining > 90, 'budget exhausted before the launch')
    long = min(600, remaining - 30) if not quick else min(240, remaining - 30)
    stage = long
    if failslow is not None:  # never cut a formal D = 600 s launch by a timeout
        require(remaining >= failslow_launch_seconds(failslow), f'a fail-slow launch needs {failslow_launch_seconds(failslow):.0f} s, '
                                                                 f'{remaining:.0f} s of budget left')
        stage = min(max(long, failslow_stage_seconds(failslow)), remaining - 30)
    elif method == 'restart':  # fail-stop restart: the peers wait through a disk reload (and the rejoin delay)
        stage = min(max(long, FAILSLOW_RELOAD_S + FAILSLOW_MARGIN_S), remaining - 30)
    return {**{k: v for k, v in DEFAULTS.items() if k != 'repeats'},
            'mode': mode, 'rounds': 1, 'launch': launch, 'probe': SERVING_PROBE, 'kind': mode, 'profile_only': False,
            'node': node, 'output': str(output), 'master': master, 'port': port, 'commit': commit, 'mps_pipe': None,
            'method': method, 'scenario': scenario, 'run_mode': run_mode,
            'quick_mode': quick, 'correctness_only': correctness, 'shared_gpu_mode': shared_gpu,
            'mps_owner': None if shared_gpu else os.environ.get('MPS_OWNER'),
            'timing_valid': not correctness, 'deterministic_mode': correctness,
            'min_free_disk_gb': min_free_disk_gb, **pidfd_implementations(),
            'startup_timeout_s': long, 'stage_timeout_s': stage, 'reclaim_timeout_s': min(120, remaining - 30),
            'launch_timeout_s': remaining, 'profile_file': str(profile), 'reference_root': str(reference),
            'kv_heads': GEOMETRY['heads'], 'memory_margin_bytes': 2 * GIB, 'memory_limit_bytes': None,
            'model_argv': worker_arguments(checkpoint, 8, seq_length, run_mode),
            'serving': {'requests': str(requests), 'workload': str(workload), 'plan': str(plan), 'b_max': b_max,
                        'decode_policy': 'greedy', 'block_tokens': 64, 'segment_tokens': 4096, 'max_delta_steps': 32, 'max_overlap_steps': 1,
                        'attention': 'torch', 'growth': {'shrink': 0, 'expand': 2}, 'geometry': GEOMETRY,
                        'capacity_tokens': capacity_tokens},
            'rejoin_note': rejoin_note, 'failslow': failslow}
