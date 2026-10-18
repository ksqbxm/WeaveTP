"""Block pool, compaction, KV plan builder (both mappings, both renderers) and paged attention on CPU tensors."""

import pytest
import torch

from tools.resharding.e2e import attention_paged, block_kv
from tools.resharding.e2e.controller import RankMap

SMALL = dict(layers=2, group_heads=2, k_dim=6, v_dim=4, block_tokens=4, segment_tokens=16, dtype=torch.float32)


def fill(pool, block, value):
    key, val = pool.block_views(block)
    key.fill_(value)
    val.fill_(-value)


def test_pool_allocates_segments_on_demand_releases_them_and_exposes_parameters():
    pool = block_kv.BlockPool(groups=2, **SMALL)
    first = pool.allocate(5)  # four blocks per segment -> two segments
    assert first == [0, 1, 2, 3, 4] and len(pool.segments) == 2
    assert pool.physical_bytes() == 2 * 2 * 16 * 2 * 2 * (6 + 4) * 4
    assert pool.used_bytes() == 5 * 4 * 2 * 2 * 2 * (6 + 4) * 4
    pool.release([4])
    assert pool.release_empty_segments() == 1 and pool.segments[1] is None and 4 not in pool.free
    assert [name for name, _ in pool.named_parameters()] == ['kv::seg_0000.key', 'kv::seg_0000.value']
    assert pool.runs([0, 1, 3]) == [(0, 0, 8), (0, 12, 16)]
    with pytest.raises(RuntimeError, match='not in use'):
        pool.release([4])


def test_context_prepare_step_positions_tables_and_mask():
    pool = block_kv.BlockPool(groups=1, **SMALL)
    context = block_kv.BlockKVContext(pool, max_delta_steps=2)
    context.admit(7, 5)
    context.admit(8, 3)
    step = context.prepare_step(prefill=[(7, 5), (8, 3)], decode=[])
    assert step['positions'].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
    assert step['slots'][4] == (context.tables[7].blocks[1], 0)
    mask = step['prefill_mask']
    assert mask.shape == (8, 8) and not mask[4, 0] and mask[0, 4] and mask[5, 0] and not mask[7, 5]
    step = context.prepare_step(prefill=[], decode=[7, 8])
    assert step['positions'].tolist() == [5, 3] and step['lengths'].tolist() == [6, 4]
    assert step['block_tables'].shape == (2, 2) and step['block_tables'][1, 1] == -1
    context.release(8)
    assert 8 not in context.tables and len(pool.in_use) == 2


def test_compaction_preserves_content_and_makes_zones_contiguous():
    pool = block_kv.BlockPool(groups=1, **SMALL)
    context = block_kv.BlockKVContext(pool, max_delta_steps=2)
    lengths = {1: 9, 2: 5, 3: 13}
    for seq, length in lengths.items():
        context.admit(seq, length - 1)
        context.tables[seq].length = length
    # Interleave allocations so the blocks of one request are not consecutive.
    expected = {}
    for seq in lengths:
        for position, block in enumerate(context.tables[seq].blocks):
            fill(pool, block, seq * 100 + position)
            expected[seq, position] = seq * 100 + position
    assignment = {1: 0, 2: 1, 3: 0}  # seq -> target replica
    report = block_kv.compact(context, assignment)
    for seq, table in context.tables.items():
        for position, block in enumerate(table.blocks):
            key, _ = pool.block_views(block)
            if (seq, position) in expected:
                assert float(key[0, 0, 0, 0, 0]) == expected[seq, position], (seq, position)
    zones = report['zones']
    # Target 0's A zone holds the full blocks of seq 1 (2) and seq 3 (3); its B zone the open blocks
    # of both plus a reserved next block where slack < max_delta_steps.
    assert zones[0, 'A'] == (0, 5) and zones[1, 'A'] == (5, 6)
    assert all(len(runs) <= 2 for runs in report['runs'].values())
    assert report['tokens'][0, 'A'] == 20 and report['moved_blocks'] > 0
    assert len(pool.in_use) == sum(len(t.blocks) for t in context.tables.values())


def assignment_tokens(builder, tables, assignment):
    return builder.zone_tokens_from_tables(tables, assignment, builder.block_tokens, max_delta_steps=2)


def test_kv_plan_builder_d16_both_directions_and_failstop_share_one_implementation():
    builder = block_kv.KVPlanBuilder(heads=16, group_heads=4, block_tokens=4, segment_tokens=16, layers=2, k_dim=6, v_dim=4)
    tp2, tp4 = RankMap.dense(16, 2), RankMap.dense(16, 4)
    tables = {seq: block_kv.BlockTable(seq, list(range(3)), 9) for seq in range(6)}
    # 2->4: TP2 replicas 2i, 2i+1 merge into TP4 replica i.
    assignment = {0: (0, 0), 1: (0, 0), 2: (1, 0), 3: (2, 1), 4: (3, 1), 5: (3, 1)}
    moves = builder.kv_moves(tp2, tp4, assignment, assignment_tokens(builder, tables, assignment))
    for move in moves:
        k = move.head_group
        i, c = move.target_replica, move.source_replica
        assert move.target_rank == 4 * i + k and move.source_rank == 4 * i + 2 * (c - 2 * i) + k // 2
    assert {m.source_rank == m.target_rank for m in moves} == {True, False}  # one of the two sources is local
    assert len(moves) <= 2 * 2 * 16 * 2
    # 4->2: TP4 replica i splits to children 2i, 2i+1 by assignment; child j's rank 4i+2(c-2i)+j takes groups 2j, 2j+1.
    back = {0: (0, 0), 1: (0, 1), 2: (0, 0), 3: (1, 2), 4: (1, 3), 5: (1, 3)}
    moves4 = builder.kv_moves(tp4, tp2, back, assignment_tokens(builder, tables, back))
    for move in moves4:
        i, child, k = move.source_replica, move.target_replica, move.head_group
        assert move.source_rank == 4 * i + k and move.target_rank == 4 * i + 2 * (child - 2 * i) + k // 2
    # Fail-stop EDP0 expand: same replica ids, TP2 survivors (coords 0/1 on ranks 10,11 / 14,15) -> TP4 on all eight.
    src = RankMap({0: {'tp': 2, 'ranks': (10, 11)}, 1: {'tp': 2, 'ranks': (14, 15)}})
    dst = RankMap({0: {'tp': 4, 'ranks': (8, 9, 10, 11)}, 1: {'tp': 4, 'ranks': (12, 13, 14, 15)}})
    same = {s: (s % 2, s % 2) for s in range(6)}
    fs = builder.kv_moves(src, dst, same, assignment_tokens(builder, tables, same))
    for move in fs:
        c, k = move.source_replica, move.head_group
        assert move.source_rank == src.replicas[c]['ranks'][k // 2] and move.target_rank == dst.replicas[c]['ranks'][k]
        assert move.source_tokens[1:] == move.target_tokens[1:] and move.source_tokens[0] == move.target_tokens[0]
    # Task counts do not grow with the number of requests (zones, not requests, make tasks).
    many = {seq: block_kv.BlockTable(seq, list(range(3)), 9) for seq in range(60)}
    many_assignment = {s: (s % 2, s % 2) for s in range(60)}
    fs_many = builder.kv_moves(src, dst, many_assignment, assignment_tokens(builder, many, many_assignment))
    assert len(fs_many) <= 2 * 27 * 2 * 16 and len(fs_many) <= 8 * len(fs)  # only segment splitting adds moves


def test_both_renderers_emit_equal_bytes_and_bounded_task_counts():
    builder = block_kv.KVPlanBuilder(heads=16, group_heads=4, block_tokens=4, segment_tokens=16, layers=2, k_dim=6, v_dim=4)
    tp2, tp4 = RankMap.dense(16, 2), RankMap.dense(16, 4)
    tables = {seq: block_kv.BlockTable(seq, list(range(5)), 17) for seq in range(8)}
    assignment = {s: (s % 8, (s % 8) // 2) for s in range(8)}
    moves = builder.kv_moves(tp2, tp4, assignment, assignment_tokens(builder, tables, assignment))
    identities = {r: (f'node{r // 8}/GPU-{r}', f'node{r // 8}') for r in range(16)}
    sources, targets = builder.as_shards(moves, tp2, tp4, identities, dtype_name='torch.bfloat16', element_size=2)
    shard_bytes = sum(2 * (t.stop[0] - t.start[0]) * 2 * 4 * d for t, d in zip(targets, [6, 4] * (len(targets) // 2)))
    ops = [op for rank in range(16) for op in builder.as_transfer_ops(moves, rank, tp2, tp4)]
    recv = [op for op in ops if not op.is_send]
    op_bytes = sum((op.my_slice[1].stop - op.my_slice[1].start) * 2 * 4 * (6 if op.param_name.endswith('key') else 4) * 2 for op in recv)
    assert shard_bytes == op_bytes > 0
    assert len(recv) == len(targets) and len(ops) == 2 * len(targets)
    assert len(targets) <= 2 * 27 * 2 * 16 and len(recv) <= 2 * 27 * 2 * 16
    # Every target shard is covered by exactly its source shard of the same identity and coordinates.
    for source, target in zip(sources, targets):
        assert source.identity == target.identity and (source.start, source.stop) == (target.start, target.stop)
    assert all(op.param_name.startswith('kv::seg_') for op in ops)


def test_paged_reference_splitkv_and_selection():
    torch.manual_seed(0)
    pool = block_kv.BlockPool(groups=2, **SMALL)
    context = block_kv.BlockKVContext(pool, max_delta_steps=2)
    lengths = {0: 7, 1: 18, 2: 3}
    for seq, length in lengths.items():
        context.admit(seq, length - 1)
        context.tables[seq].length = length
    for segment in pool.segments:
        segment[0].normal_()
        segment[1].normal_()
    step = context.prepare_step(prefill=[], decode=list(lengths))
    heads = pool.groups * pool.group_heads
    q = torch.randn(3, heads, pool.k_dim)
    scale = 1 / pool.k_dim ** 0.5
    reference = attention_paged.paged_decode_reference(q, pool, 1, step['block_tables'], step['lengths'], scale)
    for row, seq in enumerate(lengths):
        k, v = attention_paged.gather_request(pool, 1, context.tables[seq].blocks, context.tables[seq].length)
        assert torch.equal(reference[row], attention_paged.dense_attention(q[row], k, v, scale))
    fallback = attention_paged.paged_decode_splitkv(q, pool, 1, step['block_tables'], step['lengths'], scale)
    assert torch.allclose(fallback, reference, atol=1e-5, rtol=1e-5)
    assert attention_paged.select_implementation('torch') == {'decode': 'torch_splitkv', 'reason': 'requested'}
    auto = attention_paged.select_implementation('auto', triton_kernel=None)
    assert auto['decode'] == 'torch_splitkv' and auto['reason'].startswith('triton_')
    if attention_paged.HAVE_TRITON:
        probe = lambda: (q, pool, 1, step['block_tables'], step['lengths'], scale)
        bad = attention_paged.select_implementation('auto', triton_kernel=lambda *a: reference + 1.0, probe=probe)
        assert bad['decode'] == 'torch_splitkv' and bad['reason'].startswith('triton_self_check_failed')
        good = attention_paged.select_implementation('auto', triton_kernel=lambda *a: reference, probe=probe)
        assert good['decode'] == 'triton'
        with pytest.raises(RuntimeError):
            attention_paged.select_implementation('triton', triton_kernel=lambda *a: 1 / 0, probe=probe)
    else:
        with pytest.raises(RuntimeError, match='not importable'):
            attention_paged.select_implementation('triton')
