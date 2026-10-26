"""BlockBackend on a CPU toy model that calls the block-KV hooks the way Megatron does: batched prefill and paged
decode commit the same tokens as a per-request dense model; dummies write nothing; replays prefill the committed history."""

from types import SimpleNamespace

import torch

from tests.unit_tests.e2e import test_block_context_forward as toy
from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e import live_runtime, plan_executor
from tools.resharding.e2e.block_backend import BlockBackend
from tools.resharding.e2e.decode_policy import Greedy
from tools.resharding.e2e.timeline import SwitchEventWriter, TimelineWriter

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


class KeepKVSwitcher:
    """A switch that leaves the domain's KV where it is (EDP1 in the fail-stop serving: a sender, its context persists)."""

    def compact(self, direction):
        return {'moved_blocks': 0}

    def base_wave(self, direction, decode_fn):
        return {'overlap_steps': 0, 'bytes': {}}

    def audit(self, direction):
        return {}

    def delta(self, direction):
        return {'bytes': {}}

    def cutover(self, direction):
        pass

    def release_source(self, direction):
        pass

    def regroup(self, direction):
        pass


def test_a_request_evicted_at_a_switch_peak_prefills_again_on_the_replica_it_left(tmp_path):
    """The peak check (perform_switch, peak mode evict) puts the newest admissions back at the queue head; admitted again
    on the replica they left, their prefill must find none of their first admission's KV (BlockKVContext.admit refused
    'seq N already resident')."""
    prompts = {0: [3, 1, 4, 1, 5], 1: [9, 2, 6, 5, 3]}
    clock = live_runtime.FakeClock()
    controller = ctl.AdmissionController(ctl.Ledger(ctl.RankMap.dense(2, 2)), [ctl.Request(s, f'r{s}', len(p), 6) for s, p in prompts.items()],
                                         capacity=lambda r: 400, b_max=2)
    control = live_runtime.ControlNode(controller, clock)
    committed = {}  # the history each requeue publishes (RemoteControl.publish_histories) minus the prompt
    kv = backend(prompts, history_source=lambda seq: committed[seq])
    kv.switcher_factory = lambda run, direction: KeepKVSwitcher()
    run = live_runtime.DomainRun('EDP1', (0,), live_runtime.LocalComm('EDP1', (0,)), kv, Greedy(), None, TimelineWriter(tmp_path / 't.jsonl'),
                                 SwitchEventWriter(tmp_path / 'e.jsonl'), control, plan_executor.BitwiseValidation(False), {'shrink': 0},
                                 {'mode': 'evict', 'budget': 1}, clock, layout=2)
    driver = live_runtime.Driver(run)
    control.push('EDP1', 0, run.replicas)
    driver.step()  # both admitted and prefilled on replica 0
    assert set(kv.context.tables) == {0, 1}
    committed.update({seq: kv.histories[seq][len(prompts[seq]):] for seq in prompts})
    outcome = driver.switch(SimpleNamespace(kind='control', step=run.step, overshoot=0, direction='shrink'))
    assert sorted(outcome.evictions) == [0, 1] and not kv.context.tables
    control.set_phase('EDP1', False, 'normal')
    control.push('EDP1', run.step, run.replicas)
    driver.step()  # admitted again on replica 0: a fresh prefill of prompt + committed tokens
    assert set(kv.context.tables) == {0, 1} and all(row.replica == 0 for row in run.ledger.replica(0).running.values())
