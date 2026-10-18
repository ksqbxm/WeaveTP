"""Deterministic toy requests/workload/plan for CPU runs of the fake backend (tests and the Step 6 quick check)."""

import json
from pathlib import Path

from .requests_io import sha256_file

VOCAB = 16


def write_toy_inputs(directory, count=24, long_every=3, switches=None, failstop=False, rejoin_seconds=1.0):
    """requests.jsonl, workload.json and plan.json under `directory`; returns their paths.

    Prompts are 5-7 tokens, outputs 3-9 tokens; the workload is two segments (short, long);
    the plan is a v2 scale plan with `switches` or the v2-failstop fail/rejoin pair.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        prompt = [(i * 7 + k) % VOCAB for k in range(5 + (i % 3))]
        out = 3 + (i % 4) * (2 if i % long_every == 0 else 1)
        rows.append({'id': f'r{i}', 'source': 'toy', 'src_index': i, 'prompt_token_ids': prompt,
                     'output_token_ids': [(i + k) % VOCAB for k in range(out)], 'input_len': len(prompt), 'output_len': out,
                     'tokenizer': 'toy', 'sha256_text': 'x'})
    requests = directory / 'requests.jsonl'
    requests.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
    workload = {'workload_id': 'toy', 'arrival': 'continuous', 'seed': 1,
                'segments': [{'index': 0, 'kind': 'short', 'request_ids': [f'r{i}' for i in range(count // 2)]},
                             {'index': 1, 'kind': 'long', 'request_ids': [f'r{i}' for i in range(count // 2, count)]}]}
    (directory / 'workload.json').write_text(json.dumps(workload), encoding='utf-8')
    digest = sha256_file(requests)
    if failstop:
        plan = {'version': 'v2-failstop', 'request_file_sha256': digest, 'events': [
            {'id': 'fail', 'type': 'fail', 'anchor': {'kind': 'completed_count', 'fraction': 0.25}},
            {'id': 'rejoin', 'type': 'rejoin', 'anchor': {'kind': 'after_event_seconds', 'event': 'fail', 'seconds': rejoin_seconds},
             'wait_for': 'shrink_done'}]}
    else:
        plan = {'version': 'v2', 'request_file_sha256': digest, 'switches': switches or []}
    (directory / 'plan.json').write_text(json.dumps(plan), encoding='utf-8')
    return {'requests': requests, 'workload': directory / 'workload.json', 'plan': directory / 'plan.json'}
