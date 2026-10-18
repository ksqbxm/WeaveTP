"""requests_io loading rules, decode policies and pluggable anchors."""

import json
import math

import pytest
import torch

from tools.resharding.e2e import anchors, decode_policy, requests_io
from tools.resharding.e2e.controller import Request


def write_requests(path, count=4):
    rows = [{'id': f'r{i}', 'source': 'toy', 'src_index': i, 'prompt_token_ids': list(range(3 + i)),
             'output_token_ids': list(range(2 + i)), 'input_len': 3 + i, 'output_len': 2 + i,
             'tokenizer': 'toy', 'sha256_text': 'x'} for i in range(count)]
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    return rows


def test_requests_workload_and_instances(tmp_path):
    requests_path = tmp_path / 'requests.jsonl'
    write_requests(requests_path)
    requests = requests_io.load_requests(requests_path)
    workload = {'workload_id': 'w', 'arrival': 'phased', 'seed': 1,
                'segments': [{'index': 0, 'kind': 'short', 'request_ids': ['r0', 'r1', 'r0']},
                             {'index': 1, 'kind': 'long', 'arrival': 'open_loop', 'request_ids': ['r2', 'r3'], 'arrive_s': [1.5, 2.5]}]}
    (tmp_path / 'w.json').write_text(json.dumps(workload))
    loaded = requests_io.load_workload(tmp_path / 'w.json', requests)
    instances = requests_io.expand_instances(loaded, requests)
    assert [(i.seq, i.id, i.segment, i.arrive_s) for i in instances] == [(0, 'r0', 0, 0.0), (1, 'r1', 0, 0.0), (2, 'r0', 0, 0.0),
                                                                           (3, 'r2', 1, 1.5), (4, 'r3', 1, 2.5)]
    assert [i.seq for i in requests_io.expand_instances(loaded, requests, segments=(1, 1))] == [0, 1]
    bad = dict(workload, segments=[{'index': 0, 'kind': 'short', 'request_ids': ['nope']}])
    (tmp_path / 'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='unknown request ids'):
        requests_io.load_workload(tmp_path / 'bad.json', requests)
    broken = requests_path.read_text().replace('"input_len": 3', '"input_len": 9', 1)
    (tmp_path / 'broken.jsonl').write_text(broken)
    with pytest.raises(ValueError, match='input_len'):
        requests_io.load_requests(tmp_path / 'broken.jsonl')


def test_plans_require_matching_request_hash_and_known_anchors(tmp_path):
    requests_path = tmp_path / 'requests.jsonl'
    write_requests(requests_path)
    digest = requests_io.sha256_file(requests_path)
    scale = {'version': 'v2', 'request_file_sha256': digest,
             'switches': [{'anchor': {'kind': 'before_admit', 'seq': 3}, 'direction': '2->4'}]}
    (tmp_path / 'scale.json').write_text(json.dumps(scale))
    assert requests_io.load_plan(tmp_path / 'scale.json', digest)['switches'][0]['direction'] == '2->4'
    failstop = {'version': 'v2-failstop', 'request_file_sha256': digest,
                'events': [{'id': 'fail', 'type': 'fail', 'anchor': {'kind': 'completed_count', 'value': 1}},
                           {'id': 'rejoin', 'type': 'rejoin', 'anchor': {'kind': 'after_event_seconds', 'event': 'fail', 'seconds': 45}, 'wait_for': 'shrink_done'}]}
    (tmp_path / 'fs.json').write_text(json.dumps(failstop))
    assert requests_io.load_plan(tmp_path / 'fs.json', digest)['version'] == 'v2-failstop'
    with pytest.raises(ValueError, match='different request file'):
        requests_io.load_plan(tmp_path / 'fs.json', 'f' * 64)
    (tmp_path / 'weird.json').write_text(json.dumps({**scale, 'switches': [{'anchor': {'kind': 'at_time', 'seq': 1}}]}))
    with pytest.raises(ValueError, match='unknown scale anchor'):
        requests_io.load_plan(tmp_path / 'weird.json', digest)
    assert requests_io.failstop_fail_count(1000) == 250 and requests_io.failstop_fail_count(7) == math.ceil(1.75)


def test_decode_policies():
    rows = [Request(0, 'a', 2, 3), Request(1, 'b', 2, 2)]
    logits = torch.tensor([[0.1, 0.9, 0.0], [0.7, 0.2, 0.1]])
    greedy = decode_policy.Greedy()
    assert greedy.next_tokens(logits, rows) == [1, 0]
    assert not greedy.is_done(rows[0], 2) and greedy.is_done(rows[0], 3)  # EOS never ends a request early
    forcing = decode_policy.TeacherForcing({0: [11, 12, 13], 1: [21, 22]})
    for row, generated in zip(rows, (1, 0)):
        row.generated_so_far = generated
    assert forcing.next_tokens(logits, rows) == [12, 21]
    assert decode_policy.POLICIES['greedy'] is decode_policy.Greedy


def test_anchors_fire_once_with_overshoot():
    state = {'step': 5, 'now_s': 50.0, 'head_seq': 7, 'done': {1, 2, 3}, 'completed_count': 3, 'events': {'fail': 2.0, 'shrink_done': 30.0}}
    before = anchors.build({'kind': 'before_admit', 'seq': 6})
    trigger = before.check(state)
    assert trigger.kind == 'before_admit' and trigger.overshoot == {'instances_past_anchor': 1} and before.freeze_seq() == 6
    assert before.check(state) is None
    assert anchors.build({'kind': 'before_admit', 'seq': 9}).check(state) is None
    after = anchors.build({'kind': 'after_complete', 'seq': 2})
    assert after.check(state).overshoot == {'completions_past_anchor': 1}
    assert anchors.build({'kind': 'after_complete', 'seq': 9}).check(state) is None
    count = anchors.build({'kind': 'completed_count', 'fraction': 0.25, 'total': 10})
    assert count.check(state).overshoot == {'completed_count': 3, 'threshold': 3, 'overshoot': 0}
    assert anchors.build({'kind': 'completed_count', 'value': 4}).check(state) is None
    rejoin = anchors.build({'kind': 'after_event_seconds', 'event': 'fail', 'seconds': 45, 'wait_for': 'shrink_done'})
    assert rejoin.check({**state, 'now_s': 46.0}) is None  # due at 47
    assert rejoin.check({**state, 'events': {'fail': 2.0}}) is None  # shrink not done
    fired = rejoin.check(state)
    assert fired.overshoot['due_s'] == 47.0 and fired.overshoot['late_s'] == 3.0 and rejoin.check(state) is None
    assert set(anchors.REGISTRY) == {'before_admit', 'after_complete', 'completed_count', 'after_event_seconds'}
