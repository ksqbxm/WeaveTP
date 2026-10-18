"""ShareGPT subset selection with a stand-in tokenizer: filter, fixed output lengths, prefix property, plans."""

import json

import pytest

from tools.resharding.e2e import make_failstop_requests as mk
from tools.resharding.e2e import requests_io


def encode(texts):
    return [[ord(c) % 97 + 3 for c in text] for text in texts]


def items(n):
    for i in range(n):
        prompt = 'p' * (3 + i % 40)          # 4..43 input tokens with the BOS
        reference = 'r' * (2 + (i * 37) % 400)  # 2..401 reference tokens
        yield i, f'id{i}', prompt, reference


def test_filter_lengths_and_prefix_property():
    selected, selection = mk.select(items(300), encode, 1, count=50, seed=20261009, max_input=1024, max_output=256)
    assert len(selected) == 50 and selection['scanned'] == 300 and selection['dropped']['output'] > 0
    for row in selected:
        assert row['prompt_token_ids'][0] == 1 and row['input_len'] == len(row['prompt_token_ids'])
        assert 4 <= row['output_len'] <= 256 and row['output_len'] == len(row['output_token_ids'])
        assert row['output_len'] == min(row['reference_len'], 256)
    larger, _ = mk.select(items(300), encode, 1, count=80, seed=20261009, max_input=1024, max_output=256)
    assert [r['id'] for r in larger[:50]] == [r['id'] for r in selected]
    with pytest.raises(ValueError, match='pass the filter'):
        mk.select(items(10), encode, 1, count=50, seed=1, max_input=1024, max_output=256)
    tight, _ = mk.select(items(300), encode, 1, count=5, seed=1, max_input=8, max_output=256)
    assert all(r['input_len'] <= 8 for r in tight)


def test_written_set_loads_through_requests_io(tmp_path):
    selected, selection = mk.select(items(120), encode, 1, count=40, seed=7, max_input=1024, max_output=256)
    manifest = mk.write_set(tmp_path, selected, seed=7, fail_fraction=0.25, rejoin_seconds=45.0, max_input=1024, max_output=256,
                            sources={'x.json': 'deadbeef'}, selection=selection, tokenizer='stand-in')
    requests = requests_io.load_requests(tmp_path / 'requests.jsonl')
    workload = requests_io.load_workload(tmp_path / 'workload.json', requests)
    instances = requests_io.expand_instances(workload, requests)
    assert len(instances) == 40 and workload['workload_id'] == 'failstop_sharegpt_N40_v1'
    digest = requests_io.sha256_file(tmp_path / 'requests.jsonl')
    failstop = requests_io.load_plan(tmp_path / 'plan_failstop.json', digest)
    assert failstop['events'][0]['anchor'] == {'kind': 'completed_count', 'value': 10}
    assert failstop['events'][0]['coords'] == mk.COORDS and failstop['events'][1]['anchor']['seconds'] == 45.0
    nofault = requests_io.load_plan(tmp_path / 'plan_nofault.json', digest)
    assert nofault['switches'] == []
    assert manifest['fail']['completed_count'] == 10 and manifest['sha256']['requests.jsonl'] == digest
    assert manifest['lengths']['output_max'] <= 256 and manifest['count'] == 40
    with pytest.raises(ValueError, match='different request file'):
        requests_io.load_plan(tmp_path / 'plan_failstop.json', 'other')
    assert json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))['decode'].startswith('greedy')
