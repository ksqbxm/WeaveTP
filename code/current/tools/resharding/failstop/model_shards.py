"""Real-model shards for the Step 3 mixed planner, in global tensor coordinates.

Megatron stamps every parameter with tensor_model_parallel / partition_dim /
partition_stride; expert parameters are named local_experts.<i> and owned by the EP
coordinate. From those and a replica's (tp, ep, etp) coordinates each parameter and
each static-KV tensor becomes one or more Shards whose coordinates never depend on
world rank, so TP4 survivors, EDP1 TP4 and future host shards can all cover a TP2 or
TP4 target. The same functions produce the independent KV head mapping the A6
reference replay uses, so no planner output is needed to construct an expectation.
"""

import re
import sys
from pathlib import Path

import torch
from common import require

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from megatron.core.failstop import choose_local_order
from megatron.core.resharding.mixed_sources import Shard

EXPERTS, EP_SIZE = 64, 2


def identity(row):
    return f"{row['node']}/{row['gpu_uuid']}"


def model_local_orders(rows):
    """Survivor TP4 coordinate c lies inside TP2 coordinate c//2 for every partitioned
    parameter, so the survivor that takes that coordinate reuses its whole shard and the
    other reuses nothing. Ties break on stable device identity (choose_local_order)."""
    orders = {}
    for ep in range(2):
        survivors = {identity(r): r for r in rows if r['replica'] == 'EDP0' and r['ep'] == ep and r['etp'] >= 2}
        reusable = {d: [1 if rank == r['etp'] // 2 else 0 for rank in range(2)] for d, r in survivors.items()}
        orders[ep] = tuple(survivors[d]['slot'] for d in choose_local_order(survivors, reusable))
    return orders
EXPERT_NAME = re.compile(r'(?P<prefix>.*\.local_experts\.)(?P<index>\d+)(?P<suffix>\..*)')


def layout_name(tp):
    return f'TP{tp}'


def canonical(name, ep):
    """Resolved name and global expert id: local_experts.<i> on EP coordinate ep is expert ep*32+i."""
    match = EXPERT_NAME.fullmatch(name)
    if match is None:
        return name, None
    expert = ep * (EXPERTS // EP_SIZE) + int(match['index'])
    return f"{match['prefix']}{expert}{match['suffix']}", expert


def parameter_shards(parameters, *, device, node, rank, tp, coord, ep):
    """(Shard, view) pairs for one rank's parameters in a TP{tp} replica.

    parameters: iterable of (name, tensor) with Megatron's parallel attributes. A
    partition_stride of s yields s shards: local block k covers global
    [(coord + k*tp)*L/s, +L/s) along partition_dim. Replicated tensors cover the whole
    global shape, so any rank holding them, including the target itself, is a source.
    """
    require(0 <= coord < tp, 'TP coordinate outside the layout')
    result = []
    for name, tensor in parameters:
        resolved, expert = canonical(name, ep)
        parallel = bool(getattr(tensor, 'tensor_model_parallel', False))
        dim, stride = int(getattr(tensor, 'partition_dim', -1)), int(getattr(tensor, 'partition_stride', 1))
        shape = list(tensor.shape)
        if not parallel:
            result.append((Shard(device, node, rank, 'gpu', f'{layout_name(tp)}:{resolved}#0', resolved, expert,
                                 str(tensor.dtype), tensor.element_size(), tuple(shape), (0,) * len(shape),
                                 tuple(shape), layout_name(tp)), tensor))
            continue
        require(0 <= dim < len(shape) and stride >= 1 and shape[dim] % stride == 0, f'invalid partition of {name}')
        block = shape[dim] // stride
        global_shape = tuple(n * tp if d == dim else n for d, n in enumerate(shape))
        for k in range(stride):
            start = [0] * len(shape)
            stop = list(global_shape)
            start[dim], stop[dim] = (coord + k * tp) * block, (coord + k * tp + 1) * block
            view = tensor.narrow(dim, k * block, block)
            result.append((Shard(device, node, rank, 'gpu', f'{layout_name(tp)}:{resolved}#{k}', resolved, expert,
                                 str(tensor.dtype), tensor.element_size(), global_shape, tuple(start), tuple(stop),
                                 layout_name(tp)), view))
    return result


def kv_shards(kv, *, device, node, rank, tp, coord, heads, start, stop):
    """(Shard, view) per layer and kind for tokens [start, stop) of a static KV cache.

    kv: {layer: (key, value)} with tensors [S, B, heads/tp, dim]; this coordinate holds
    heads [coord*heads/tp, (coord+1)*heads/tp). The view is the token range only, so
    Shard.index() offsets match the view's own coordinates.
    """
    require(heads % tp == 0 and 0 <= start < stop, 'invalid KV layout or token range')
    local = heads // tp
    result = []
    for layer, pair in sorted(kv.items()):
        for kind, tensor in zip(('key', 'value'), pair):
            require(tensor.shape[2] == local and stop <= tensor.shape[0], f'KV tensor shape mismatch at layer {layer}')
            shape = (tensor.shape[0], tensor.shape[1], heads, tensor.shape[3])
            result.append((Shard(device, node, rank, 'gpu', f'{layout_name(tp)}:kv.{layer}.{kind}', f'kv.{layer}.{kind}',
                                 None, str(tensor.dtype), tensor.element_size(), shape,
                                 (start, 0, coord * local, 0), (stop, tensor.shape[1], (coord + 1) * local, tensor.shape[3]),
                                 layout_name(tp)), tensor[start:stop]))
    return result


def map_kv_heads(snapshot, *, source_tp, target_tp, target_coord, heads):
    """Independent reference mapping: the target coordinate's KV from a complete source snapshot.

    snapshot: {source_coord: {layer: (key, value)}} covering the same token range on
    every source coordinate. Returns {layer: (key, value)} restricted to the target
    coordinate's heads, built by head arithmetic alone.
    """
    require(heads % source_tp == 0 and heads % target_tp == 0, 'heads must divide both layouts')
    source_local, target_local = heads // source_tp, heads // target_tp
    first, last = target_coord * target_local, (target_coord + 1) * target_local
    result = {}
    layers = sorted(next(iter(snapshot.values())))
    for layer in layers:
        pieces = ([], [])
        for head in range(first, last):
            source_coord, offset = divmod(head, source_local)
            for kind, piece in enumerate(snapshot[source_coord][layer]):
                pieces[kind].append(piece[:, :, offset:offset + 1])
        result[layer] = tuple(torch.cat(p, dim=2) for p in pieces)
    return result


def coverage(shards):
    """Bytes claimed per identity; the test oracle for exact tiling of a layout."""
    totals = {}
    for shard, _ in shards:
        size = shard.element_size
        for a, b in zip(shard.start, shard.stop):
            size *= b - a
        totals[shard.identity] = totals.get(shard.identity, 0) + size
    return totals
