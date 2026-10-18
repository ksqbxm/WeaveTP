"""The three Megatron hooks as BlockKVContext methods: block-diagonal prefill, paged decode and dummy forwards
reproduce dense attention over each request's own history, on CPU with a toy attention layer."""

import torch

from tools.resharding.e2e import attention_paged, block_kv, paged_decode_triton

torch.manual_seed(0)
NP, DK, DV, VOCAB = 4, 8, 6, 32
POOL = dict(groups=2, group_heads=2, k_dim=DK, v_dim=DV, block_tokens=4, segment_tokens=16, layers=2, dtype=torch.float32)
EMBED = {kind: torch.randn(VOCAB, NP, dim) for kind, dim in (('q', DK), ('k', DK), ('v', DV))}
ROTARY = torch.randn(64, 1, 1, DK)
SCALE = 0.3


def projections(tokens, positions):
    """Toy MLA stand-in: per-token q/k/v [S, 1, np, dim]; k carries the rotary row of its position (post-RoPE cache)."""
    q = EMBED['q'][tokens][:, None]
    k = (EMBED['k'][tokens] + ROTARY[positions][:, 0]).unsqueeze(1)
    v = EMBED['v'][tokens][:, None]
    return q, k, v


def dense_module(query, key, value, mask):
    """DotProductAttention semantics: [sq, b, np, d] in, [sq, b, np*dv] out, bool mask True = masked."""
    scores = torch.einsum('sbhd,tbhd->bhst', query, key) * SCALE
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum('bhst,tbhd->sbhd', probs, value)
    return out.reshape(out.shape[0], out.shape[1], -1)


def reference(history):
    """Dense causal attention of the last token over the whole history of one request."""
    tokens = torch.tensor(history)
    positions = torch.arange(len(history))
    q, k, v = projections(tokens, positions)
    return dense_module(q[-1:], k, v, None)[0, 0]


def test_prefill_decode_and_dummy_match_dense_attention():
    context = block_kv.BlockKVContext(block_kv.BlockPool(**POOL), attention=attention_paged.paged_decode_splitkv)
    histories = {0: [3, 1, 4, 1, 5], 1: [9, 2, 6, 5, 3, 5, 8]}
    for seq, prompt in histories.items():
        context.admit(seq, len(prompt))
    context.prepare_step(prefill=[(seq, len(p)) for seq, p in histories.items()], decode=[])
    context.begin_forward('prefill', torch.device('cpu'))
    tokens = torch.tensor(histories[0] + histories[1])
    assert context.rotary_positions.tolist() == list(range(5)) + list(range(7))
    q, k, v = projections(tokens, context.rotary_positions)
    for layer in (1, 2):
        q, k, v = context.adjust_key_value(layer, q, k, v)
        out = context.core_attention(layer, q, k, v, None, dense_attention=dense_module, softmax_scale=SCALE)
        assert out.shape == (12, 1, NP * DV)
        # Every request's last prefill position equals the dense attention over its own prompt only.
        assert torch.allclose(out[4, 0], reference(histories[0]), atol=1e-5)
        assert torch.allclose(out[11, 0], reference(histories[1]), atol=1e-5)
    context.end_forward()
    for _ in range(6):  # crosses the 4-token block boundary for both requests
        new = {seq: (len(h) * 7) % VOCAB for seq, h in histories.items()}
        step = context.prepare_step(prefill=[], decode=[0, 1])
        context.begin_forward('decode', torch.device('cpu'))
        assert context.rotary_positions.tolist() == [len(histories[0]), len(histories[1])]
        q, k, v = projections(torch.tensor([new[0], new[1]]), context.rotary_positions)
        for seq in histories:
            histories[seq].append(new[seq])
        for layer in (1, 2):
            q, k, v = context.adjust_key_value(layer, q, k, v)
            out = context.core_attention(layer, q, k, v, None, dense_attention=dense_module, softmax_scale=SCALE)
            assert out.shape == (2, 1, NP * DV)
            for row, seq in enumerate(histories):
                assert torch.allclose(out[row, 0], reference(histories[seq]), atol=1e-5), (row, layer)
        context.end_forward()
        assert step['lengths'].tolist() == [len(histories[0]), len(histories[1])]
    before = [s[0].clone() for s in context.pool.segments if s is not None]
    context.begin_forward('dummy', torch.device('cpu'))
    q, k, v = projections(torch.tensor([7]), context.rotary_positions)
    q, k, v = context.adjust_key_value(1, q, k, v)
    out = context.core_attention(1, q, k, v, None, dense_attention=dense_module, softmax_scale=SCALE)
    context.end_forward()
    assert out.shape == (1, 1, NP * DV) and context.is_decode_only() is False
    assert all(torch.equal(a, s[0]) for a, s in zip(before, [s for s in context.pool.segments if s is not None]))


def test_triton_kernel_is_absent_here_and_the_selection_falls_back():
    assert not paged_decode_triton.HAVE_TRITON and paged_decode_triton.KERNEL is None
    pool = block_kv.BlockPool(**POOL)
    record = attention_paged.select_implementation('auto', triton_kernel=paged_decode_triton.KERNEL,
                                                   probe=lambda: (torch.zeros(1, NP, DK), pool, 0, torch.zeros(1, 1, dtype=torch.int64),
                                                                  torch.ones(1, dtype=torch.int64), SCALE))
    assert record == {'decode': 'torch_splitkv', 'reason': 'triton_not_importable'}
    assert attention_paged.select_implementation('torch') == {'decode': 'torch_splitkv', 'reason': 'requested'}
