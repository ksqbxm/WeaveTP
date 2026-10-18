"""The recovery methods the fail-stop run can use and what their results say about themselves (torch-free).

The serving loop, the control gate, the events, the timeline and the result files are shared; a method
differs only in the worker's switcher and the planner it hands the transport. These records go into
result.json so a run names the semantics it was produced under.
"""

METHODS = {
    'weavetp': {
        'name': 'weavetp',
        'source_policy': 'bandwidth-aware mixed plan over residual TP4 + healthy replica (Step 3 greedy)',
        'kv_semantics': 'shrink: EDP0 in-flight requests replay from prompt + committed history; '
                        'expand: block-KV prefix (A zone) during the base wave, pre-cutover delta (B zone)',
        'kv_semantics_status': '00_总计划 §6.2',
        'implementation_ref': None,
        'advantage_sources': ['weights via P2P from residual TP4 + EDP1, no checkpoint reload',
                              'EDP1 keeps serving between the waves',
                              'emulate: no process rebuild; real: four processes rebuilt',
                              'replay only EDP0 in-flight requests'],
    },
    'anchortp': {
        'name': 'anchortp',
        'source_policy': 'fixed-path incl. healthy replica, no host daemon',
        'kv_semantics': 'shrink: EDP0 in-flight requests replay (as weavetp); expand: A and B zones moved together at the '
                        'frozen lengths, no delta, no foreground decode on EDP0 during the migration',
        'kv_semantics_status': '未与论文核对',
        'implementation_ref': 'eba3bc8',
        'advantage_sources': ['fixed intersection sources ordered by static bandwidth (anchortp_plan)',
                              'EDP0 paused during both migrations (overlap_steps=0)',
                              'EDP1 keeps serving between the waves',
                              'replay only EDP0 in-flight requests'],
    },
}


def describe(name):
    if name not in METHODS:
        raise ValueError(f'unknown recovery method {name!r}; expected one of {sorted(METHODS)}')
    return METHODS[name]
