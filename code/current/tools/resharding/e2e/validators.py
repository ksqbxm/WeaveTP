"""The only place that knows the fixed cluster: 16 GPUs, SL3060/SL3061, replicas within a node, TP2 nested in TP4."""

NODES = ('SL3060', 'SL3061')
WORLD = 16


def validate_world(rank_map, node_of_rank):
    ranks = sorted(r for spec in rank_map.replicas.values() for r in spec['ranks'])
    if ranks != list(range(len(ranks))):
        raise ValueError('ranks must be exactly 0..n-1 across replicas')
    if len(ranks) not in (WORLD, WORLD - 4):
        raise ValueError(f'expected {WORLD} (or {WORLD - 4} degraded) ranks, got {len(ranks)}')
    for replica, spec in rank_map.replicas.items():
        nodes = {node_of_rank[r] for r in spec['ranks']}
        if len(nodes) != 1 or next(iter(nodes)) not in NODES:
            raise ValueError(f'replica {replica} spans nodes {nodes}')
        if spec['tp'] != len(spec['ranks']) or spec['tp'] not in (2, 4):
            raise ValueError(f'replica {replica}: tp {spec["tp"]} with ranks {spec["ranks"]}')
    return {'ranks': len(ranks), 'replicas': len(rank_map.replicas), 'nodes': sorted({node_of_rank[r] for r in ranks})}


def validate_nesting(tp2_map, tp4_map):
    """Every TP2 replica lies inside one TP4 replica (D16 2i, 2i+1 -> i)."""
    for replica, spec in tp2_map.replicas.items():
        parents = {tp4_map.replica_of(r)[0] for r in spec['ranks']}
        if len(parents) != 1:
            raise ValueError(f'TP2 replica {replica} is not nested in one TP4 replica')
    return True
