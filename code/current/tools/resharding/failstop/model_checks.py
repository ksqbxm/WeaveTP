"""A6 bitwise comparison of a fail-stop launch against independent reference replays.

A launch leaves one checkpoint per serving run and member:
checkpoints/r<round>/<label>/slot_<slot>.pt with the run's spec (layout, prefilled tokens,
decoded tokens, optional KV snapshot directory) and its logits records. The reference
replays each spec in a fresh process of the same layout (model_reference.py --mr-role
replay) and writes replay/<id>/rank_<coord + tp*ep>.pt. Comparison goes through the Step 5
contract: both manifests deterministic, same layout/history/parameter identity, exact bits.
"""

import json
from pathlib import Path

import torch
from common import require
from determinism import compare_records


def spec_id(round_label, label, slot):
    return f'{round_label}_{label}_slot{slot}'


def checkpoints(roots):
    """Every (id, path) under the given launch roots (one per node), sorted for stable reports."""
    found = []
    for root in roots:
        for path in sorted(Path(root).glob('checkpoints/r*/*/slot_*.pt')):
            found.append((spec_id(path.parents[1].name, path.parent.name, int(path.stem.split('_')[1])), path))
    require(found, 'no checkpoints under the launch roots')
    require(len({i for i, _ in found}) == len(found), 'duplicate checkpoint ids across the launch roots')
    return found


def collect_specs(roots):
    """Replay specs grouped by layout; KV snapshot directories are rewritten relative to their launch root."""
    specs = {}
    for identifier, path in checkpoints(roots):
        saved = torch.load(path, weights_only=False)
        spec = dict(saved['spec'])
        spec.update(id=identifier, request_id=saved['request_id'], slot=saved['slot'], replica=saved['replica'])
        specs.setdefault(f"tp{spec['layout']['tp']}", []).append(spec)
    return specs


def compare(roots, replay_root):
    """PASS only when every record of every checkpoint equals its replayed reference bitwise."""
    rows, passed = [], True
    for identifier, path in checkpoints(roots):
        saved = torch.load(path, weights_only=False)
        layout = saved['spec']['layout']
        rank = layout['coord'] + layout['tp'] * layout['ep']
        reference_path = Path(replay_root) / identifier / f'rank_{rank}.pt'
        row = {'id': identifier, 'layout': layout, 'records': len(saved['records']), 'reference': str(reference_path)}
        if not reference_path.exists():
            row.update(status='missing_reference')
            passed = False
        else:
            reference = torch.load(reference_path, weights_only=False)['records']
            if len(reference) != len(saved['records']):
                row.update(status='record_count_differs', reference_records=len(reference))
                passed = False
            else:
                try:
                    results = [compare_records(a, b) for a, b in zip(reference, saved['records'])]
                    first = next((r for r in results if not r['equal']), None)
                    row.update(status='bitwise' if first is None else 'differs', first_mismatch=first)
                    passed = passed and first is None
                except RuntimeError as error:  # a side outside deterministic mode or a different state
                    row.update(status='not_comparable', reason=str(error))
                    passed = False
        rows.append(row)
    return {'result': 'PASS' if passed else 'FAIL', 'checkpoints': rows,
            'scope': 'A6 logits per serving run against independent same-layout replays; weights and KV were '
                     'validated inside the launch before each cutover'}


def write_specs(roots, output):
    specs = collect_specs(roots)
    Path(output).write_text(json.dumps(specs, indent=2) + '\n', encoding='utf-8')
    return {layout: len(rows) for layout, rows in specs.items()}
