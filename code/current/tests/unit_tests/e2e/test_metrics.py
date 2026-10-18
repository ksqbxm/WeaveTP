"""Recovery metrics on hand-built launches: thresholds, censoring, replay-aware latency, token conservation."""

import json
import sys
from pathlib import Path

import pytest

from tools.resharding.e2e import metrics as m

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools' / 'resharding' / 'failstop'))
import analyze_failstop_e2e as cli  # noqa: E402


def request(seq, replica, arrive, first, step, n, *, replay=None):
    """n tokens at `step` spacing from `first`; `replay` = (t_requeue_admit, replica, generated_at_admit, pause) splits the stream."""
    tokens = [first + i * step for i in range(n)]
    admits = [{'t': arrive, 'step': 0, 'replica': replica, 'layout': 4, 'generated_at_admit': 0}]
    if replay is not None:
        t_admit, new_replica, done, pause = replay
        tokens = tokens[:done] + [tokens[done - 1] + pause + (i + 1) * step for i in range(n - done)]
        admits.append({'t': t_admit, 'step': 50, 'replica': new_replica, 'layout': 2, 'generated_at_admit': done})
    return {'seq': seq, 'id': f'r{seq}', 'segment': 0, 'kind': 'short', 'replica': admits[-1]['replica'], 'layout_at_admit': 4,
            'input_len': 8, 'output_len': n, 'generated_tokens': n, 't_arrive': arrive, 't_admit': arrive, 't_first': tokens[0],
            't_done': tokens[-1], 'admits': admits, 't_tokens': tokens}


def launch(tmp_path, *, fail=20.0, rejoin=35.0, tp4_ready=40.0, horizon=90.0, scenario='failstop'):
    """Steady 4 tok/s per replica; EDP0 (replicas 0/1) silent in [fail, fail+3) and [38, tp4_ready+1)."""
    rows = []
    seq = 0
    for replica in (0, 1, 2, 3):
        t = 0.25 * replica
        while t < horizon - 1:
            n = 4
            first = t
            edp0 = replica in (0, 1)
            if edp0 and fail <= first < fail + 3:
                first = fail + 3
            if edp0 and 38 <= first < tp4_ready + 1:
                first = tp4_ready + 1
            rows.append(request(seq, replica, t, first, 0.25, n))
            seq += 1
            t = first + n * 0.25
    rows.append(request(seq, 0, fail - 5.0, fail - 4.5, 0.5, 8, replay=(fail + 2.0, 2, 6, 4.5)))
    switches = [{'domain': 'EDP0', 'direction': 'shrink', 'overlap_steps': 0, 'bytes': {'kv_delta': 0}, 'milestones': {'cutover': fail + 2.5}},
                {'domain': 'EDP0', 'direction': 'expand', 'overlap_steps': 2, 'bytes': {'kv_delta': 100}, 'milestones': {'cutover': tp4_ready}}]
    failstop = None if scenario == 'nofault' else {'replay_prefill_tokens': 14, 'stats_valid': True}
    result = {'method': 'weavetp', 'config': {'experiment': scenario, 'mode': 'emulate', 'run_mode': 'timing', 'timing_valid': True},
              'totals': {'generated_tokens': sum(r['output_len'] for r in rows)}, 'requests': rows, 'switches': switches,
              'failstop': failstop, 'serving_wall_s': horizon}
    sha = {'requests': 'a', 'plan': 'b', 'workload': 'c', 'profile': 'd'}
    events = {'launch': f'l_{scenario}', 'sha256': sha, 'recovery_s': {'fail': fail, 'rejoin_request': rejoin, 'spawn': rejoin + 0.5,
                                                                        'switch_ack_expand': 38.0, 'serving_start': 0.0},
              'delays': {'edp1_pause_s': None, 'rejoin_delay_s': 0.5, 'rejoin_join_s': 3.0}}
    directory = tmp_path / f'launch_{scenario}'
    directory.mkdir(parents=True)
    (directory / 'result.json').write_text(json.dumps(result))
    (directory / 'events_actual.json').write_text(json.dumps(events))
    return directory


def test_failstop_metrics_recover_against_prefail_reference(tmp_path):
    directory = launch(tmp_path)
    metrics, series = m.analyze(m.load_launch(directory))
    assert metrics['prefail_reference_valid'] and 15.0 <= metrics['prefail_reference_tok_s'] <= 17.0
    assert abs(metrics['shrink_tfs_s'] - 3.0) < 1e-6 and abs(metrics['edp0_downtime_shrink_s'] - 3.0) < 1e-6
    assert metrics['expand_time_s'] == pytest.approx(40.0 - 35.5)
    assert 2.5 <= metrics['edp0_downtime_expand_s'] <= 4.0  # last EDP0 token before the expand ack -> first after TP4 ready
    assert metrics['expand_ttp_censored'] is None and metrics['expand_ttp_s'] >= 3.0
    assert metrics['full_recovery_s'] == pytest.approx(metrics['expand_ttp_s'] + 40.0 - 20.0)
    assert metrics['shrink_ttp_censored'] is None and metrics['shrink_reference_source'].startswith('self')
    assert metrics['replay_requests'] == 1 and metrics['replay_prefill_tokens'] == 14
    assert sum(series['global']) == metrics['generated_tokens']
    assert sum(series['EDP0'][21:23]) == 0  # the fault silence is visible per second (a token at t=fail lands in bin 20)
    # Replay-aware latency: the replayed request keeps its original first token and its TPOT includes the pause.
    replayed = next(r for r in m.load_launch(directory)['result']['requests'] if len(r['admits']) > 1)
    assert replayed['t_first'] == 15.5 and (replayed['t_tokens'][-1] - replayed['t_tokens'][0]) / 7 > 0.5  # pause inside TPOT
    window = metrics['latency_fail_to_recovery']
    assert window['tpot_s']['n'] >= 1 and metrics['latency_all']['ttft_s']['n'] == metrics['requests']


def test_censoring_and_invalid_references(tmp_path):
    early = m.load_launch(launch(tmp_path, fail=5.0, rejoin=20.0, tp4_ready=25.0, horizon=60.0))
    metrics, _ = m.analyze(early)
    assert not metrics['prefail_reference_valid'] and metrics['expand_ttp_s'] is None
    assert metrics['expand_ttp_censored'] == 'no reference' and metrics['full_recovery_s'] is None
    series = [0] * 30
    assert m.recovery_time(series, 0, 10.0) == (None, 'censored: criterion not met before 30.0 s')
    series = [10] * 4 + [20] * 26
    t, reason = m.recovery_time(series, 0, 20.0)
    assert reason is None and t == 11.0  # 5 s trailing mean reaches 19 >= 0.95 x 20 at second 8, held through seconds 8, 9, 10
    assert m.recovery_time(series, 0, None) == (None, 'no reference')


def test_validation_rejects_inconsistent_rows(tmp_path):
    directory = launch(tmp_path)
    bad = m.load_launch(directory)
    bad['result']['requests'][0]['t_tokens'].pop()
    with pytest.raises(ValueError, match='token stamps'):
        m.validate(bad['result'])
    bad = m.load_launch(directory)
    bad['result']['totals']['generated_tokens'] += 1
    with pytest.raises(ValueError, match='generated token total'):
        m.validate(bad['result'])
    events = json.loads((directory / 'events_actual.json').read_text())
    events['sha256']['requests'] = 'other'
    result = json.loads((directory / 'result.json').read_text())
    result['config']['sha256'] = {'requests': 'a'}
    (directory / 'result.json').write_text(json.dumps(result))
    (directory / 'events_actual.json').write_text(json.dumps(events))
    with pytest.raises(ValueError, match='requests hash'):
        m.load_launch(directory)


def test_latency_percentiles_and_single_token_requests():
    rows = [request(0, 0, 0.0, 1.0, 0.5, 1), request(1, 0, 0.0, 2.0, 0.5, 3)]
    result = {'requests': rows}
    out = m.latency(result)
    assert out['tpot_s'] == {'p50': 0.5, 'p99': 0.5, 'n': 1} and out['ttft_s'] == {'p50': 1.0, 'p99': 2.0, 'n': 2}
    assert m.latency(result, (1.5, 1.8))['ttft_s']['n'] == 0 and m.latency(result, (2.2, 2.3))['tpot_s']['n'] == 1
    assert m.percentiles([]) == {'p50': None, 'p99': None, 'n': 0}


def test_summary_and_cli(tmp_path):
    a = launch(tmp_path / 'a')
    b = launch(tmp_path / 'b', fail=22.0, rejoin=37.0, tp4_ready=42.0)
    nofault = launch(tmp_path / 'c', scenario='nofault')
    out = tmp_path / 'out'
    assert cli.main([str(tmp_path), '--out', str(out)]) == 0
    rows = list(__import__('csv').DictReader((out / 'summary.csv').open(encoding='utf-8')))
    tfs = next(r for r in rows if r['metric'] == 'shrink_tfs_s' and r['scenario'] == 'failstop')
    assert (tfs['n_valid'], tfs['n_launches']) == ('2', '2') and float(tfs['mean']) == pytest.approx(3.0) and tfs['sample_std'] != ''
    single = [r for r in rows if r['scenario'] == 'nofault']
    assert single and all(r['sample_std'] == '' for r in single) and {r['metric'] for r in single} >= {'throughput_tok_s', 'ttft_p99_s'}
    assert (out / 'per_second_l_failstop.csv').exists() and (out / 'launch_metrics.csv').exists()
    assert json.loads((out / 'rejected.json').read_text()) == []
    (a / 'result.json').write_text('{}')
    assert cli.main([str(a), str(b), '--out', str(tmp_path / 'out2')]) == 1
    assert len(json.loads((tmp_path / 'out2' / 'rejected.json').read_text())) == 1
    assert nofault.exists()
