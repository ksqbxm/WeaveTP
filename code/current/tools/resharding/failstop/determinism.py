"""Step 5 §2.1 deterministic-mode contract.

Every bitwise logits/weight check compares two records that each prove they were produced
under --deterministic-mode with the pinned NCCL/cuBLAS environment and share layout, request
history and loaded parameters. Migration plans written by default-mode and deterministic-mode
launches must be identical item by item (stable identities, slices, bytes, waves). The quick
check records the deterministic/default per-step time ratio as an overhead estimate only.
"""

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

from common import DETERMINISTIC_ENVIRONMENT, require

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.correctness.router_determinism import first_mismatch


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def mode_manifest(deterministic_mode_arg, *, algorithms_enabled, environ):
    """What a process can prove about its own mode; stored next to every bitwise record."""
    environment = {key: environ.get(key) for key in DETERMINISTIC_ENVIRONMENT}
    deterministic = bool(deterministic_mode_arg) and bool(algorithms_enabled) and environment == DETERMINISTIC_ENVIRONMENT
    return {'mode': 'deterministic' if deterministic else 'default',
            'deterministic_mode_arg': bool(deterministic_mode_arg),
            'deterministic_algorithms': bool(algorithms_enabled), 'environment': environment}


def require_deterministic(manifest, side):
    problems = []
    if not manifest['deterministic_mode_arg']:
        problems.append('--deterministic-mode not set')
    if not manifest['deterministic_algorithms']:
        problems.append('torch deterministic algorithms disabled')
    problems.extend(f'{key}={manifest["environment"].get(key)!r} (required {value!r})'
                    for key, value in DETERMINISTIC_ENVIRONMENT.items() if manifest['environment'].get(key) != value)
    require(not problems, f'{side} is not in deterministic mode: ' + '; '.join(problems))


def require_default(manifest, side):
    """The timing baseline is the formal performance configuration: no deterministic argument or algorithms."""
    require(not manifest['deterministic_mode_arg'] and not manifest['deterministic_algorithms'],
            f'{side} must run in default mode')


def history(request_id, tokens):
    tokens = [int(t) for t in tokens]
    return {'request_id': request_id, 'tokens': tokens, 'length': len(tokens),
            'sha256': canonical_digest([request_id, tokens])}


def bitwise_record(tensor, *, step, manifest, layout, history, parameter_checksum):
    """Original-dtype CPU bits plus every identity a comparison must match before looking at them."""
    bits = tensor.detach().cpu().contiguous()
    return {'step': step, 'dtype': str(bits.dtype), 'shape': list(bits.shape), 'bits': bits,
            'manifest': manifest, 'layout': layout, 'history': history, 'parameter_checksum': parameter_checksum}


def compare_records(reference, candidate):
    """Exact comparison; refuses unless both sides prove deterministic mode and describe the same state."""
    require_deterministic(reference['manifest'], 'reference')
    require_deterministic(candidate['manifest'], 'candidate')
    for key in ('layout', 'step', 'parameter_checksum'):
        require(reference[key] == candidate[key],
                f'{key} differs ({reference[key]!r} vs {candidate[key]!r}); records are not comparable')
    require(reference['history']['sha256'] == candidate['history']['sha256'],
            'request history differs; records are not comparable')
    mismatch = first_mismatch(reference['bits'], candidate['bits'])
    return {'step': reference['step'], 'equal': mismatch is None, 'first_mismatch': mismatch,
            'dtype': reference['dtype'], 'shape': reference['shape'], 'layout': reference['layout'],
            'history_sha256': reference['history']['sha256'], 'history_length': reference['history']['length'],
            'parameter_checksum': reference['parameter_checksum']}


def normalize_tasks(rows):
    """Plan rows without temporary ranks: stable device identities, slices, bytes and waves remain."""
    result = []
    for row in rows:
        row = json.loads(json.dumps(row))
        for endpoint in ('source', 'target'):
            row[endpoint].pop('rank', None)
        result.append(row)
    return result


def plan_consistency(reference_rows, candidate_rows):
    """Item-by-item equality of two normalized plans, naming the first differing task and fields."""
    reference, candidate = normalize_tasks(reference_rows), normalize_tasks(candidate_rows)
    difference = None
    for index, (a, b) in enumerate(zip(reference, candidate)):
        if a != b:
            fields = sorted(key for key in set(a) | set(b) if a.get(key) != b.get(key))
            difference = {'index': index, 'fields': fields, 'reference': {k: a.get(k) for k in fields},
                          'candidate': {k: b.get(k) for k in fields}}
            break
    if difference is None and len(reference) != len(candidate):
        difference = {'index': min(len(reference), len(candidate)), 'fields': ['task_count'],
                      'reference': {'task_count': len(reference)}, 'candidate': {'task_count': len(candidate)}}
    return {'equal': difference is None, 'tasks': [len(reference), len(candidate)], 'first_difference': difference}


def compare_launch_plans(reference_root, candidate_root):
    """Compare the source_plan_<round>_<phase>.json files two launches wrote on the same node.

    Plans are comparable only with the same frozen profile; input equality (membership, epoch,
    coverage) separates a planner difference from a snapshot difference.
    """
    reference_root, candidate_root = Path(reference_root), Path(candidate_root)
    names = sorted(p.name for p in reference_root.glob('source_plan_*.json'))
    require(names, f'no source_plan_*.json under {reference_root}')
    require(names == sorted(p.name for p in candidate_root.glob('source_plan_*.json')),
            'launches planned different rounds/phases')
    plans = {}
    for name in names:
        reference = json.loads((reference_root / name).read_text(encoding='utf-8'))
        candidate = json.loads((candidate_root / name).read_text(encoding='utf-8'))
        row = plan_consistency(reference['tasks'], candidate['tasks'])
        row['profile_sha256'] = [reference['profile_sha256'], candidate['profile_sha256']]
        row['profile_equal'] = reference['profile_sha256'] == candidate['profile_sha256']
        row['inputs_equal'] = {key: reference[key] == candidate[key] for key in ('membership', 'epoch', 'sources', 'targets')}
        row['environments'] = [reference['environment'], candidate['environment']]
        row['equal'] = row['equal'] and row['profile_equal']
        plans[name] = row
    return {'equal': all(p['equal'] for p in plans.values()), 'plans': plans,
            'scope': 'normalized migration plans of two launches; identical only with one frozen profile'}


def step_time_ratio(deterministic_samples, default_samples):
    """deterministic / default GPU time per step on identical input and layout; an overhead estimate only."""
    require(len(deterministic_samples) == len(default_samples) > 0, 'timing samples do not pair up')
    rows = []
    for det, base in zip(deterministic_samples, default_samples):
        for key in ('layout', 'step', 'context'):
            require(det[key] == base[key], f'timing samples differ in {key}; not the same input/layout')
        require(all(math.isfinite(s['gpu_ms']) and s['gpu_ms'] > 0 for s in (det, base)), 'invalid step time')
        rows.append({'step': det['step'], 'context': det['context'], 'deterministic_ms': det['gpu_ms'],
                     'default_ms': base['gpu_ms'], 'ratio': det['gpu_ms'] / base['gpu_ms']})
    decode = [r['ratio'] for r in rows if r['step'] > 0]
    return {'layout': deterministic_samples[0]['layout'], 'per_step': rows,
            'prefill_ratio': next((r['ratio'] for r in rows if r['step'] == 0), None),
            'decode_median_ratio': statistics.median(decode) if decode else None,
            'median_ratio': statistics.median(r['ratio'] for r in rows),
            'scope': 'overhead estimate only; not a performance conclusion'}
