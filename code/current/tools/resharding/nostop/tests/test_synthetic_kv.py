"""CPU proof of TP-invariant contents, bounded prefill and cycle restoration."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.resharding.nostop import synthetic_kv
from tools.resharding.nostop.tests.test_diagnostics import fake_diagnostic


def context(heads):
    return NS(sequence_len_offset=8, enable_decode_mode=lambda: None,
              key_value_memory_dict={1: (torch.empty(5248, 4, heads, 3, dtype=torch.bfloat16),
                                        torch.empty(5248, 4, heads, 2, dtype=torch.bfloat16))})


def prefix():
    return {1: (torch.randn(8, 4, 4, 3).bfloat16(), torch.randn(8, 4, 4, 2).bfloat16())}


def test_identical_tp2_tp4_global_heads_and_no_global_rng_changes():
    real = prefix()
    before = torch.get_rng_state().clone()
    layouts = {}
    for tp in (2, 4):
        shards = [context(4 // tp) for _ in range(tp)]
        for rank, shard in enumerate(shards):
            synthetic_kv.restore(shard, real, rank, 0, torch)
            assert shard.sequence_len_offset == 2048
        layouts[tp] = [torch.cat([c.key_value_memory_dict[1][kind] for c in shards], dim=2) for kind in (0, 1)]
    for k in (0, 1):
        assert torch.equal(layouts[2][k], layouts[4][k])
        assert torch.equal(layouts[2][k][:8], real[1][k])
        assert torch.count_nonzero(layouts[2][k][2048:]) == 0
        assert abs(layouts[2][k][8:2048].float().std().item() - .02) < .001
        assert not torch.equal(layouts[2][k][8:2048, 0], layouts[2][k][8:2048, 1])
    assert torch.equal(before, torch.get_rng_state())
    other = context(2)
    synthetic_kv.restore(other, real, 0, 1, torch)
    assert not torch.equal(other.key_value_memory_dict[1][0][8:2048], layouts[2][0][8:2048, :, :2])


def test_canonical_prefix_uses_one_tp2_replica_per_domain():
    rows = [dict(rank=r, request_domain=0, tp_ranks=[r//2*2, r//2*2+1],
                 prefix={1: (torch.full((8,4,2,3),r), torch.full((8,4,2,2),r))}) for r in range(4)]
    result = synthetic_kv.canonical_prefix(list(reversed(rows)), 0, torch)
    assert torch.equal(result[1][0], torch.cat([rows[0]["prefix"][1][0], rows[1]["prefix"][1][0]],dim=2))


def test_initial_validation_sees_identical_2048_prefix_in_both_layouts(monkeypatch):
    d, _, _, _ = fake_diagnostic(monkeypatch)
    d.torch, d.origin, d.index = torch, None, -1
    d.args.live_request_domain_id = 0
    src, dst = context(2), context(1)
    real = prefix()
    rows = [dict(rank=r, request_domain=0, tp_ranks=[r//2*2, r//2*2+1],
                 prefix={1: tuple(t[:, :, (r%2)*2:(r%2+1)*2] for t in real[1])}) for r in range(4)]
    def gather(output, local, *, group):
        assert group == "gloo" and all(t.device.type == "cpu" and t.shape[0] == 8 for t in local["prefix"][1])
        output[:] = rows
    d.dist = NS(get_rank=lambda group: 0, get_world_size=lambda group: 4,
                get_process_group_ranks=lambda group: [0,1], all_gather_object=gather)
    model = NS(pg_collection=NS(tp="tp"))
    def validate(*args):
        assert src.sequence_len_offset == dst.sequence_len_offset == 2048
        for kind in (0,1):
            source, target = src.key_value_memory_dict[1][kind], dst.key_value_memory_dict[1][kind]
            assert torch.equal(source[:8], real[1][kind][:,:,:2])
            assert torch.equal(source[:2048,:,:1], target[:2048])
        return "validated"
    assert d.validate(validate, model, src, model, dst, d.args, "gloo", 39) == "validated"
    d.memory.observe_context.assert_called_once_with("after_prefill_validation")
    d.memory.begin.assert_called_once_with("plan_setup")


def test_cycle_discards_normal_tail_and_restores_same_2048_contents(monkeypatch):
    d, _, _, _ = fake_diagnostic(monkeypatch)
    d.torch = NS(inference_mode=torch.inference_mode, Generator=torch.Generator, randn=torch.randn,
                 bfloat16=torch.bfloat16, cuda=NS(current_stream=lambda: NS(synchronize=lambda: None)))
    d.prefix = prefix()
    d.args.live_request_domain_id = 0
    d.dist.get_rank = lambda group: 0
    source, target = context(2), context(1)
    model = NS(pg_collection=NS(tp="tp"))
    weights = {tp: NS(_check_storage=Mock(),num_bytes=900) for tp in (2,4)}
    offsets = []
    def decode(_model, ctx, args, *, token_value):
        offsets.append(ctx.sequence_len_offset)
        for t in ctx.key_value_memory_dict[1]:
            t[ctx.sequence_len_offset].fill_(token_value)
        ctx.sequence_len_offset += 1
    d.b._decode = decode
    def run(i):
        return d.before_switch(i,2,model,source,model,target,weights,{},None,None,None,None,999)
    token = run(0)
    reference = [t[:2048].clone() for t in source.key_value_memory_dict[1]]
    for ctx in (source,target):
        for t in ctx.key_value_memory_dict[1]:
            t.fill_(-7)
        ctx.sequence_len_offset = 4000
    assert run(2) == token
    assert offsets == list(range(2048,2060))*2
    assert source.sequence_len_offset == target.sequence_len_offset == 2048
    assert all(torch.equal(t[:2048],r) for t,r in zip(source.key_value_memory_dict[1],reference))
    assert torch.equal(target.key_value_memory_dict[1][0][:2048], reference[0][:,:,:1])
    assert d.states[-1]["real_prefill_tokens"] == 8 and d.states[-1]["measured"]


def test_prefill_wrapper_passes_only_eight_tokens(monkeypatch):
    d, ctx, model, _ = fake_diagnostic(monkeypatch)
    original = Mock(return_value=torch.ones(4,1,16))
    d.prefill(original,model,ctx,d.args)
    d.prefill(original,model,ctx,d.args)
    assert all(call.args[2].live_prompt_tokens == 8 for call in original.call_args_list)
    d.memory.begin.assert_called_once_with("prefill")
