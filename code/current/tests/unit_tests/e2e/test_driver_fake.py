"""The whole driver loop on the fake backend: scaling with planned switches, fail-stop with fail/replay/expand."""

import json
import sys
from pathlib import Path

import pytest

from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e.decode_policy import Greedy
from tools.resharding.e2e.timeline import read_jsonl
from tools.resharding.e2e.toy_inputs import write_toy_inputs as write_inputs

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples' / 'rl'))
import run_e2e_scaling as entry  # noqa: E402


def run(tmp_path, *extra):
    argv = ['--e2e-requests', str(tmp_path / 'requests.jsonl'), '--e2e-workload', str(tmp_path / 'workload.json'),
            '--e2e-plan', str(tmp_path / 'plan.json'), '--e2e-out-dir', str(tmp_path / 'out'), '--e2e-backend', 'fake',
            '--e2e-run-mode', 'correctness', '--e2e-b-max', '3', '--e2e-capacity-tokens', '400', *extra]
    assert entry.main(argv) == 0
    return json.loads((tmp_path / 'out' / 'result.json').read_text()), json.loads((tmp_path / 'out' / 'check.json').read_text())


def test_scaling_run_with_both_switch_directions(tmp_path):
    write_inputs(tmp_path, switches=[{'direction': '2->4', 'anchor': {'kind': 'before_admit', 'seq': 12}},
                                     {'direction': '4->2', 'anchor': {'kind': 'after_complete', 'seq': 14}}])
    result, report = run(tmp_path, '--e2e-experiment', 'scaling', '--e2e-decode-policy', 'teacher_forcing', '--e2e-world-size', '8')
    assert report['passed'], report
    assert [s['direction'] for s in result['switches']] == ['2->4', '4->2']
    assert all(s['validation']['passed'] for s in result['switches'])
    assert result['switches'][0]['bytes']['kv_prefix'] > 0 and result['switches'][0]['overlap_steps'] == ctl.GROWTH_STEPS['2->4']
    assert all(r['t_admit'] <= r['t_first'] <= r['t_done'] for r in result['requests'])
    rows = read_jsonl(tmp_path / 'out' / 'timeline.jsonl')
    steps = [r for r in rows if r['kind'] == 'step']
    assert {r['layout'] for r in steps} == {2, 4}
    assert all(abs(sum(r[k] for k in ('t_prefill', 't_decode', 't_sync', 't_ctrl_fetch', 't_ctrl_apply', 't_wait')) - r['step_time']) < 1e-9 for r in steps)
    assert any(r['kind'] == 'stall' for r in rows) or all(s['overlap_steps'] for s in result['switches'])
    assert result['totals']['generated_tokens'] == sum(r['output_len'] for r in result['requests'])


def test_dummy_forwards_write_no_kv_count_no_token_and_land_in_t_sync(tmp_path):
    write_inputs(tmp_path, count=6)
    from tools.resharding.e2e import fake_backend
    rank_map = ctl.RankMap.dense(4, 2)
    backend = fake_backend.FakeBackend(rank_map, {0: [1, 2, 3]})
    ledger = ctl.Ledger(rank_map)
    ledger.replica(0).add(ctl.Running(0, 0, 0, 5, 3, 3))
    backend.prefill(0, [0], ledger, Greedy())
    before = backend.checksum(0), backend.checksum(1)
    assert backend.dummy(1, 'decode') > 0 and (backend.checksum(0), backend.checksum(1)) == before
    result, _ = run(tmp_path, '--e2e-experiment', 'scaling', '--e2e-world-size', '8', '--e2e-decode-policy', 'greedy')
    rows = read_jsonl(tmp_path / 'out' / 'timeline.jsonl')
    assert sum(r['t_sync'] for r in rows if r['kind'] == 'step') > 0
    assert result['totals']['generated_tokens'] == sum(r['generated_tokens'] for r in result['requests'])


def test_failstop_run_replays_edp0_and_expands(tmp_path):
    write_inputs(tmp_path, count=20, failstop=True)
    result, report = run(tmp_path, '--e2e-experiment', 'failstop', '--e2e-decode-policy', 'greedy')
    assert report['passed'], report
    assert [s['direction'] for s in result['switches']] == ['shrink', 'expand']
    fs = result['failstop']
    assert fs['replay_prefill_tokens'] > 0 and 'fail' in fs['events_s'] and fs['events_s']['shrink_done'] >= fs['events_s']['fail']
    assert fs['events_s']['rejoin_start'] >= fs['events_s']['fail'] + 1.0
    assert result['switches'][0]['bytes'] == {'weights': 1, 'kv_prefix': 0, 'kv_delta': 0}  # shrink replays, moves no KV
    assert result['switches'][1]['bytes']['kv_prefix'] > 0 and result['switches'][1]['validation']['passed']
    edp1 = read_jsonl(tmp_path / 'out' / 'timeline_EDP1.jsonl')
    edp0 = read_jsonl(tmp_path / 'out' / 'timeline_EDP0.jsonl')
    assert edp1 and edp0 and {r['layout'] for r in edp0 if r['kind'] == 'step'} == {4, 2}
    assert all(r['replica'] is not None for r in result['requests'])
    assert result['config']['validate_cutover'] is True
    # Per-token stamps: one per committed token, monotone, first == t_first, last == t_done; a replayed
    # request lists both admits but keeps its original admit/first-token times.
    for r in result['requests']:
        assert len(r['t_tokens']) == r['output_len'] and r['t_tokens'] == sorted(r['t_tokens'])
        assert r['t_first'] == r['t_tokens'][0] and r['t_done'] == r['t_tokens'][-1]
        assert r['t_admit'] == r['admits'][0]['t'] and r['admits'][0]['generated_at_admit'] == 0
    replayed = [r for r in result['requests'] if len(r['admits']) > 1]
    assert replayed and sum(a['generated_at_admit'] for r in replayed for a in r['admits'][1:]) +         sum(r['input_len'] for r in replayed) <= fs['replay_prefill_tokens']
    assert all(r['admits'][1]['t'] > r['t_first'] and r['replica'] == r['admits'][-1]['replica'] for r in replayed)
    with pytest.raises(ValueError, match='contradicts'):
        entry.main(['--e2e-requests', str(tmp_path / 'requests.jsonl'), '--e2e-workload', str(tmp_path / 'workload.json'),
                    '--e2e-out-dir', str(tmp_path / 'x'), '--e2e-backend', 'fake', '--e2e-run-mode', 'timing', '--e2e-validate-cutover', 'on'])
