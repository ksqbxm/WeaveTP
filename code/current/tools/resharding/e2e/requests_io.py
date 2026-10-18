"""Request sets, workloads and switch plans: loading with sha256 checks, instance expansion (W1)."""

import hashlib
import json
import math
from pathlib import Path

from .controller import Request

ARRIVALS = ('continuous', 'phased', 'open_loop')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_requests(path):
    """requests.jsonl rows keyed by id; token lists and lengths must agree."""
    rows = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        require(row['input_len'] == len(row['prompt_token_ids']), f"{row['id']}: input_len differs from prompt tokens")
        require(row['output_len'] == len(row['output_token_ids']) and row['output_len'] >= 1, f"{row['id']}: invalid output tokens")
        require(row['id'] not in rows, f"duplicate request id {row['id']}")
        rows[row['id']] = row
    require(rows, 'empty request file')
    return rows


def load_workload(path, requests):
    workload = json.loads(Path(path).read_text(encoding='utf-8'))
    require(workload['arrival'] in ARRIVALS, 'unknown arrival mode')
    for index, segment in enumerate(workload['segments']):
        require(segment['index'] == index and segment['kind'] in ('short', 'long'), f'segment {index} malformed')
        require(segment.get('arrival', workload['arrival']) in ARRIVALS, f'segment {index}: unknown arrival')
        missing = [r for r in segment['request_ids'] if r not in requests]
        require(not missing, f'segment {index}: unknown request ids {missing[:3]}')
        if segment.get('arrival', workload['arrival']) == 'open_loop':
            require(len(segment.get('arrive_s', [])) == len(segment['request_ids']), f'segment {index}: arrive_s length')
    return workload


def expand_instances(workload, requests, segments=None):
    """Request instances in FCFS order; `seq` is the position in the concatenated segments."""
    chosen = range(len(workload['segments'])) if segments is None else range(segments[0], segments[1] + 1)
    result, seq = [], 0
    for index in chosen:
        segment = workload['segments'][index]
        arrivals = segment.get('arrive_s') if segment.get('arrival', workload['arrival']) == 'open_loop' else None
        for position, request_id in enumerate(segment['request_ids']):
            row = requests[request_id]
            result.append(Request(seq, request_id, row['input_len'], row['output_len'], segment['kind'], index,
                                  arrive_s=arrivals[position] if arrivals else 0.0))
            seq += 1
    return result


def load_plan(path, request_sha256):
    """v2 scale plan or v2-failstop plan; the request file hash must match; no switches = fixed layout."""
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    require(plan.get('request_file_sha256', plan.get('requests_sha256')) == request_sha256,
            'plan was generated for a different request file')
    if plan.get('version') == 'v2-failstop':
        kinds = {e['type'] for e in plan['events']}
        require(kinds == {'fail', 'rejoin'}, 'v2-failstop plan needs exactly fail and rejoin events')
        return plan
    switches = plan.get('switches', [])
    for switch in switches:
        require(switch['anchor']['kind'] in ('before_admit', 'after_complete'), 'unknown scale anchor kind')
        require('seq' in switch['anchor'], 'scale anchors address instances by seq')
    return plan


def failstop_fail_count(total_requests, fraction=0.25):
    return math.ceil(fraction * total_requests)
