"""BlockBackend on a CPU toy model that calls the block-KV hooks the way Megatron does: batched prefill and paged
decode commit the same tokens as a per-request dense model; dummies write nothing; replays prefill the committed history."""

import torch

from tests.unit_tests.e2e import test_block_context_forward as toy
from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e.block_backend import BlockBackend
from tools.resharding.e2e.decode_policy import Greedy

GEOMETRY = dict(layers=2, heads=4, group_heads=2, k_dim=toy.DK, v_dim=toy.DV)
HEAD = torch.randn(toy.NP * toy.DV, toy.VOCAB)


def toy_forward(ids, positions, context):
    """GPTModel stand-in: two attention layers through the hooks, then a linear head -> logits [1, S, vocab]."""
    tokens = ids[0]
    q, k, v = toy.projections(tokens, context.rotary_positions)
    out = None
    for layer in (1, 2):
        q, k, v = context.adjust_key_value(layer, q, k, v)
        out = context.core_attention(layer, q, k, v, None, dense_attention=toy.dense_module, softmax_scale=toy.SCALE)
    return (out[:, 0] @ HEAD)[None]


def dense_reference_tokens(history, steps):
    """Greedy continuation of one request with dense attention over its full history."""
    tokens = list(history)
    for _ in range(steps):
        q, k, v = toy.projections(torch.tensor(tokens), torch.arange(len(tokens)))
        out = toy.dense_module(q[-1:], k, v, None)
        tokens.append(int((out[0, 0] @ HEAD).argmax()))
    return tokens


def backend(prompts, **kwargs):
    # tp=1: the toy layer produces all four heads (two groups of two) on this rank
    return BlockBackend(toy_forward, 0, 1, prompts, geometry=GEOMETRY, block_tokens=4, segment_tokens=16, dtype=torch.float32,
                        device='cpu', **kwargs)


def test_batched_block_kv_serving_matches_per_request_dense_decoding():
    prompts = {0: [3, 1, 4, 1, 5], 1: [9, 2, 6, 5, 3, 5, 8], 2: [1, 1, 2]}
    ledger = ctl.Ledger(ctl.RankMap.dense(2, 2))
    rows = {seq: ctl.Running(seq, 0, 0, 7, len(p), 8) for seq, p in prompts.items()}
    for row in rows.values():
        ledger.replica(0).add(row)
    kv = backend(prompts)
    assert kv.attention_impl['decode'] == 'torch_splitkv'
    seconds, count = kv.prefill(0, [0, 1], ledger, Greedy())
    assert count == 2 and seconds > 0
    for _ in range(5):  # crosses block boundaries for both requests
        kv.decode(0, [rows[0], rows[1]], ledger, Greedy())
    kv.prefill(0, [2], ledger, Greedy())  # a request admitted while others run: block-diagonal prefill alone, then joins decode
    kv.decode(0, [rows[0], rows[1], rows[2]], ledger, Greedy())
    for seq, prompt in prompts.items():
        generated = len(kv.histories[seq]) - len(prompt)
        assert kv.histories[seq] == dense_reference_tokens(prompt, generated), seq
    checksum = kv.checksum(0)
    assert kv.dummy(0, 'decode') > 0 and kv.checksum(0) == checksum and kv.step_times['dummy']
    assert kv.kv_bytes(0) > 0 and kv.memory()['pool_physical_bytes'] >= kv.memory()['pool_used_bytes']
    for seq in prompts:
        kv.release(0, seq)
    assert kv.kv_bytes(0) == 0 and all(s is None for s in kv.pool.segments)


def test_replay_prefills_prompt_plus_committed_history_from_the_source():
    prompts = {5: [2, 7, 1, 8]}
    committed = {5: [11, 13, 17]}
    kv = backend(prompts, history_source=lambda seq: committed[seq])
    ledger = ctl.Ledger(ctl.RankMap.dense(2, 2))
    row = ctl.Running(5, 0, 4, 9, len(prompts[5]) + 3, 6, generated_at_admit=3)
    ledger.replica(0).add(row)
    kv.prefill(0, [5], ledger, Greedy())
    assert kv.histories[5][:7] == prompts[5] + committed[5] and len(kv.histories[5]) == 8
    assert kv.histories[5] == dense_reference_tokens(prompts[5] + committed[5], 1)
    kv.decode(0, [row], ledger, Greedy())
    assert row.generated_so_far == 4  # three committed before the replay plus the prefill's token
