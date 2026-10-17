"""Real-model shard derivation and KV head mapping, checked against global-coordinate oracles.

A toy parameter set with Megatron's parallel attributes (replicated, column, column
with stride 2, row, experts under EP2) stands in for DeepSeek-V2-Lite; values encode
global coordinates so every byte that lands in a target can be verified independently.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import model_shards
from test_sources import profile

from megatron.core.resharding.mixed_sources import build_mixed_plan

H, W = 8, 16  # global rows / columns of every toy tensor
SPECS = {  # name -> (partition_dim or None, stride, expert)
    'decoder.layers.0.input_layernorm.weight': (None, 1, False),
    'decoder.layers.0.self_attention.linear_q_proj.weight': (0, 1, False),
    'decoder.layers.0.mlp.linear_fc1.weight': (0, 2, False),
    'decoder.layers.0.self_attention.linear_proj.weight': (1, 1, False),
    'decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight': (0, 2, True),
    'decoder.layers.1.mlp.experts.local_experts.1.linear_fc2.weight': (1, 1, True),
}


def global_tensor(name, expert):
    """Row-major coordinates plus a per-name and per-expert offset; distinct everywhere."""
    base = (sum(map(ord, name)) % 97) * 10000 + (0 if expert is None else expert * 1000000)
    return (torch.arange(H * W, dtype=torch.int64) + base).reshape(H, W)


def shard_of(name, ep, tp, coord):
    """Oracle: what TP{tp} coordinate `coord` holds, by Megatron's layout rules."""
    dim, stride, _ = SPECS[name]
    full = global_tensor(*model_shards.canonical(name, ep))
    if dim is None:
        return full.clone()
    block = full.shape[dim] // (tp * stride)
    pieces = [full.narrow(dim, (coord + k * tp) * block, block) for k in range(stride)]
    return torch.cat(pieces, dim=dim).clone()


def toy_parameters(tp, coord, ep):
    """Named tensors with the attributes Megatron stamps; experts are the EP coordinate's own."""
    result = []
    for name, (dim, stride, _) in SPECS.items():
        tensor = shard_of(name, ep, tp, coord)
        tensor.tensor_model_parallel = dim is not None
        tensor.partition_dim = -1 if dim is None else dim
        tensor.partition_stride = stride
        result.append((name, tensor))
    return result


def device(replica, ep, coord):
    return f'{replica}/{ep}/{coord}'


def apply(tasks, views):
    """Tensor keys are unique per device, as the executor's per-device dictionaries use them."""
    for task in tasks:
        source, target = views[task.source.device, task.source.tensor], views[task.target.device, task.target.tensor]
        target[task.target.index(task.start, task.stop)].copy_(source[task.source.index(task.start, task.stop)])


class ParameterShardTests(unittest.TestCase):
    def test_canonical_names_and_global_expert_ids(self):
        self.assertEqual(model_shards.canonical('a.local_experts.5.linear_fc1.weight', 1),
                         ('a.local_experts.37.linear_fc1.weight', 37))
        self.assertEqual(model_shards.canonical('a.mlp.router.weight', 1), ('a.mlp.router.weight', None))

    def test_tp4_coordinates_tile_every_global_tensor_exactly(self):
        for ep in range(2):
            shards = []
            for coord in range(4):
                shards += model_shards.parameter_shards(toy_parameters(4, coord, ep), device=device('EDP0', ep, coord),
                                                        node='SL3061', rank=coord, tp=4, coord=coord, ep=ep)
            totals = model_shards.coverage(shards)
            for shard, view in shards:
                self.assertEqual(totals[shard.identity], (4 if shard.tensor.endswith('layernorm.weight#0') else 1) * H * W * 8)
                expected = global_tensor(shard.name, shard.expert)[tuple(slice(a, b) for a, b in zip(shard.start, shard.stop))]
                self.assertTrue(torch.equal(view, expected), shard.tensor)
            experts = {s.expert for s, _ in shards if s.expert is not None}
            self.assertEqual(experts, {ep * 32, ep * 32 + 1})
        fc1 = [s for s, _ in model_shards.parameter_shards(toy_parameters(4, 1, 0), device='d', node='n', rank=1, tp=4, coord=1, ep=0)
               if s.name.endswith('mlp.linear_fc1.weight')]
        # Stride 2: gate block then up block, each L/2 rows at global offsets (coord + k*tp)*L/2.
        self.assertEqual([(s.start[0], s.stop[0]) for s in fc1], [(1, 2), (5, 6)])

    def test_shrink_plan_fills_tp2_targets_bitwise_from_survivors_and_edp1(self):
        prof_devices = [device('EDP0', ep, c) for ep in range(2) for c in range(4)] + \
                       [device('EDP1', ep, c) for ep in range(2) for c in range(4)]
        ranks = {d: i for i, d in enumerate(prof_devices)}
        for ep in range(2):
            views, sources = {}, []
            for coord in (2, 3):  # EDP0 survivors keep their TP4 shards as sources
                for shard, view in model_shards.parameter_shards(toy_parameters(4, coord, ep), device=device('EDP0', ep, coord),
                                                                 node='SL3061', rank=ranks[device('EDP0', ep, coord)], tp=4, coord=coord, ep=ep):
                    views[shard.device, shard.tensor] = view
                    sources.append(shard)
            for coord in range(4):  # EDP1 healthy TP4 shards of the same EP
                for shard, view in model_shards.parameter_shards(toy_parameters(4, coord, ep), device=device('EDP1', ep, coord),
                                                                 node='SL3060', rank=ranks[device('EDP1', ep, coord)], tp=4, coord=coord, ep=ep):
                    views[shard.device, shard.tensor] = view
                    sources.append(shard)
            targets = []
            for new_coord, survivor in enumerate((2, 3)):
                target_device = device('EDP0', ep, survivor)
                empty = [(name, torch.full_like(t, -1)) for name, t in toy_parameters(2, new_coord, ep)]
                for (name, tensor), (_, source) in zip(empty, toy_parameters(2, new_coord, ep)):
                    for attribute in ('tensor_model_parallel', 'partition_dim', 'partition_stride'):
                        setattr(tensor, attribute, getattr(source, attribute))
                for shard, view in model_shards.parameter_shards(empty, device=target_device, node='SL3061',
                                                                 rank=ranks[target_device], tp=2, coord=new_coord, ep=ep):
                    views[shard.device, shard.tensor] = view
                    targets.append((shard, view, new_coord))
            tasks, _, _ = build_mixed_plan(sources, [t for t, _, _ in targets], profile(prof_devices),
                                           membership=prof_devices, epoch=1, rank_map=ranks)
            apply(tasks, views)
            for shard, view, new_coord in targets:
                expected = global_tensor(shard.name, shard.expert)[tuple(slice(a, b) for a, b in zip(shard.start, shard.stop))]
                self.assertTrue(torch.equal(view, expected), f'{shard.tensor} differs after shrink')
            # Replicated tensors are copied locally; EP-foreign experts are never used as sources.
            self.assertTrue(all(t.source_kind == 'local' for t in tasks if t.target.name.endswith('layernorm.weight')))
            self.assertTrue(all(t.source.expert == t.target.expert for t in tasks))


class KVShardTests(unittest.TestCase):
    def kv(self, tp, coord, tokens=6, batch=1, heads=16, dims=(192, 128), layers=(1, 2)):
        local = heads // tp
        result = {}
        for layer in layers:
            pair = []
            for kind, dim in enumerate(dims):
                full = (torch.arange(tokens * batch * heads * dim, dtype=torch.int64).reshape(tokens, batch, heads, dim)
                        + layer * 10 ** 9 + kind * 10 ** 8)
                pair.append(full[:, :, coord * local:(coord + 1) * local].clone())
            result[layer] = tuple(pair)
        return result

    def test_kv_shards_cover_token_range_and_own_heads(self):
        shards = model_shards.kv_shards(self.kv(2, 1), device='d', node='n', rank=3, tp=2, coord=1, heads=16, start=2, stop=5)
        self.assertEqual(len(shards), 4)
        key, view = shards[0]
        self.assertEqual((key.shape, key.start, key.stop), ((6, 1, 16, 192), (2, 0, 8, 0), (5, 1, 16, 192)))
        self.assertTrue(torch.equal(view, self.kv(2, 1)[1][0][2:5]))
        self.assertTrue(torch.equal(view[key.index((3, 0, 8, 0), (4, 1, 16, 192))], self.kv(2, 1)[1][0][3:4]))
        with self.assertRaisesRegex(RuntimeError, 'token range'):
            model_shards.kv_shards(self.kv(2, 1), device='d', node='n', rank=3, tp=2, coord=1, heads=16, start=5, stop=5)

    def test_head_mapping_matches_direct_global_slicing(self):
        snapshot = {coord: self.kv(2, coord) for coord in range(2)}
        for target_coord in range(4):
            mapped = model_shards.map_kv_heads(snapshot, source_tp=2, target_tp=4, target_coord=target_coord, heads=16)
            expected = self.kv(4, target_coord)
            for layer in expected:
                for kind in range(2):
                    self.assertTrue(torch.equal(mapped[layer][kind], expected[layer][kind]))


if __name__ == '__main__':
    unittest.main()
