"""Per-GPU memory of the Step 5 real-model scenario: DeepSeek-V2-Lite TP4<->TP2 with both layouts resident.

Sizes are the router determinism check's exact per-rank parameter and static-KV formulas; each
coexistence window is a Step 4 PeakBudget, so this preflight and the runner's switch admission
use one accounting. Allocations not sized analytically (cuBLAS workspaces, NCCL buffers, MoE
scratch, attention scores, allocator reserve) are listed, covered by the margin and measured after a run.
"""

import sys
from pathlib import Path

from async_state import PeakBudget
from common import FAIL_COORDS, NODES, node_rows, require

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.correctness.router_determinism_driver import (
    GIB,
    LAYOUTS,
    UNESTIMATED_ALLOCATIONS,
    estimate_kv_cache_bytes,
    estimate_rank_parameter_bytes,
    query_gpus,
)

SURVIVOR, REJOIN, HEALTHY = 'edp0_survivor_dual_layout', 'edp0_failed_rejoin_target', 'edp1_tp4_steady_source'
MARGIN = 2 * GIB


def weight_bytes(size):
    return estimate_rank_parameter_bytes(size, size, 2)


def role(row):
    if row['replica'] == 'EDP1':
        return HEALTHY
    return REJOIN if (row['ep'], row['etp']) in FAIL_COORDS else SURVIVOR


def windows(role_name, seq_length, batch=1, margin=MARGIN):
    """Every window in which two layouts share one GPU, as the Step 4 admission budget fields."""
    w2, w4 = weight_bytes(2), weight_bytes(4)
    k2, k4 = estimate_kv_cache_bytes(2, seq_length, batch), estimate_kv_cache_bytes(4, seq_length, batch)
    if role_name == SURVIVOR:
        # Shrink: the old TP4 shard stays the source until the TP2 target is verified and the
        # invalidated in-flight TP4 KV is released only at the switch; TP2 KV is re-prefilled.
        # Expand: TP2 keeps serving while the TP4 target and its KV prefix/delta fill.
        # Remote receives are staged per wave, bounded by the target bytes.
        return {'shrink': PeakBudget(w4, w2, k4, k2, w2, 0, margin),
                'expand': PeakBudget(w2, w4, k2, k4, w4, 0, margin)}
    if role_name == REJOIN:
        return {'expand': PeakBudget(0, w4, 0, k4, w4, 0, margin)}
    # EDP1 only sends: packed copies of its TP4 slices are bounded by its own shard.
    return {'steady': PeakBudget(w4, 0, k4, 0, 0, w4, margin)}


def requirement(role_name, seq_length, batch=1, margin=MARGIN):
    rows = {name: dict(vars(b), total=b.total) for name, b in windows(role_name, seq_length, batch, margin).items()}
    binding = max(rows, key=lambda name: rows[name]['total'])
    return {'role': role_name, 'windows': rows, 'binding_window': binding, 'required_free_bytes': rows[binding]['total']}


def reference_requirements(seq_length, batch=1, margin=MARGIN):
    """Independent reference runs are sequential single-layout processes; they never add to the scenario peak."""
    result = {}
    for layout, spec in LAYOUTS.items():
        weights, kv = weight_bytes(spec['tp']), estimate_kv_cache_bytes(spec['tp'], seq_length, batch)
        result[layout] = {'gpus': spec['gpus'], 'weight_bytes_per_gpu': weights, 'kv_cache_bytes_per_gpu': kv,
                          'margin_bytes': margin, 'required_free_bytes_per_gpu': weights + kv + margin}
    return result


def scenario_table(inventory, seq_length):
    """All sixteen scenario GPUs with role and requirement; free memory is known only where queried."""
    rows = []
    for name in NODES:
        gpus = inventory.get(name) or [{'index': i, 'uuid': f'unqueried-{name}-{i}', 'free_bytes': None, 'total_bytes': None}
                                       for i in range(8)]
        for row in node_rows(name, gpus, len(rows)):
            gpu = next(g for g in gpus if g['index'] == row['gpu_index'])
            need = requirement(role(row), seq_length)
            rows.append({**row, 'role': need['role'], 'binding_window': need['binding_window'],
                         'required_free_bytes': need['required_free_bytes'], 'free_bytes': gpu['free_bytes'],
                         'total_bytes': gpu['total_bytes'],
                         'sufficient': None if gpu['free_bytes'] is None else gpu['free_bytes'] >= need['required_free_bytes']})
    return rows


def preflight(node, seq_length, gpus, run_layouts, query=query_gpus):
    """Per-GPU table for the whole scenario plus the requirement of this launch's reference runs."""
    inventory = query(list(gpus))
    require(len(inventory) == 8, 'the scenario table needs all eight GPUs of this node')
    table = scenario_table({node: inventory}, seq_length)
    scenario_shortages = [f"{r['node']} GPU {r['gpu_index']} ({r['role']}, {r['binding_window']}): free "
                          f"{r['free_bytes'] / GIB:.2f} GiB < required {r['required_free_bytes'] / GIB:.2f} GiB"
                          for r in table if r['sufficient'] is False]
    reference = reference_requirements(seq_length)
    # The layouts run one after another, so each GPU needs the largest single-layout requirement.
    run_required = {g['index']: max((reference[l]['required_free_bytes_per_gpu'] for l in run_layouts
                                     if i < reference[l]['gpus']), default=0) for i, g in enumerate(inventory)}
    run_shortages = []
    for layout in run_layouts:
        need = reference[layout]['required_free_bytes_per_gpu']
        run_shortages.extend(f"{layout} run: GPU {g['index']} free {g['free_bytes'] / GIB:.2f} GiB < required "
                             f"{need / GIB:.2f} GiB" for g in inventory[:reference[layout]['gpus']] if g['free_bytes'] < need)
    return {'node': node, 'seq_length': seq_length, 'inventory': inventory, 'scenario': table,
            'scenario_shortages': scenario_shortages, 'scenario_sufficient_on_this_node': not scenario_shortages,
            'reference_runs': reference, 'run_layouts': list(run_layouts), 'run_shortages': run_shortages,
            'run_required_free_bytes_per_gpu': run_required,
            'roles': {name: requirement(name, seq_length) for name in (SURVIVOR, REJOIN, HEALTHY)},
            'not_estimated_measured_after_run': UNESTIMATED_ALLOCATIONS,
            'formula': 'router_determinism_driver.estimate_rank_parameter_bytes/estimate_kv_cache_bytes; '
                       'coexistence windows as async_state.PeakBudget'}


def format_table(report):
    lines = [f"Step 5 memory preflight on {report['node']} (seq_length={report['seq_length']}):",
             'slot node   gpu replica ep etp role                        window  required  free      ok']
    for r in report['scenario']:
        free = '?' if r['free_bytes'] is None else f"{r['free_bytes'] / GIB:.2f}"
        lines.append(f"{r['slot']:>4} {r['node']} {r['gpu_index']:>3} {r['replica']:7} {r['ep']:>2} {r['etp']:>3} "
                     f"{r['role']:27} {r['binding_window']:7} {r['required_free_bytes'] / GIB:7.2f}  {free:8} "
                     f"{'-' if r['sufficient'] is None else 'yes' if r['sufficient'] else 'NO'}")
    for layout, row in report['reference_runs'].items():
        lines.append(f"reference run {layout}: {row['gpus']} GPUs x {row['required_free_bytes_per_gpu'] / GIB:.2f} GiB "
                     f"(sequential; not added to the scenario peak)")
    lines.append(f"this launch ({', '.join(report['run_layouts'])}) requires per GPU: " + ', '.join(
        f"GPU {index} {need / GIB:.2f} GiB" for index, need in report['run_required_free_bytes_per_gpu'].items() if need))
    lines.append('not estimated, measured after the run: ' + ', '.join(report['not_estimated_measured_after_run']))
    return '\n'.join(lines)
