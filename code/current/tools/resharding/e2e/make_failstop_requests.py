"""The fail-stop request set: a ShareGPT subset with fixed output lengths <= 256, plus its workload and plans.

Tokenization follows documents/e2e_prep_20261008/tools/make_requests.py (DeepSeek-V2-Lite-hf tokenizer, no chat
template, add_special_tokens=False, one BOS prepended to the prompt). Selection: conversations with >= 2 turns,
turn 1 = prompt, turn 2 = reference; keep 4 <= input_len <= max_input and reference length >= 4; the output
length is min(len(reference), max_output) and the reference tokens are cut to it (greedy decoding ignores EOS
and runs to exactly output_len tokens). The kept pool is shuffled once with random.Random(f"{seed}/sharegpt")
and the first `count` are taken, so a larger count extends a smaller one (the calibration iterates on count).

Outputs under --out: requests.jsonl, workload.json (one continuous segment), plan_failstop.json (v2-failstop:
fail at completed_count ceil(fraction*N) on EDP0, rejoin after_event_seconds, wait_for shrink_done),
plan_nofault.json (v2, no switches) and manifest.json (sources, rule, SHA256 of every file, length stats).
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

from .requests_io import sha256_file

DEFAULT_SHAREGPT = '/data/ubuntu/lxh/weavetp/e2e_prep/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json'
DEFAULT_HF = '/data/models/DeepSeek-V2-Lite-hf'
COORDS = [{'etp': 0, 'ep': 0}, {'etp': 1, 'ep': 0}, {'etp': 0, 'ep': 1}, {'etp': 1, 'ep': 1}]


def conversations(path):
    """(index, id, prompt text, reference text) of every ShareGPT conversation with at least two turns."""
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = rows if isinstance(rows, list) else rows['data']
    for index, row in enumerate(rows):
        turns = row.get('conversations') or []
        if len(turns) >= 2:
            yield index, str(row.get('id', '')), turns[0]['value'], turns[1]['value']


def select(items, encode, bos, *, count, seed, max_input, max_output):
    """items: iterable of (index, id, prompt, reference); encode: texts -> token id lists (no special tokens)."""
    items = list(items)
    prompts = encode([prompt for _, _, prompt, _ in items])
    references = encode([reference for _, _, _, reference in items])
    pool, dropped = [], {'input': 0, 'output': 0}
    for (index, src_id, prompt, reference), p, r in zip(items, prompts, references):
        input_len = len(p) + 1
        if not 4 <= input_len <= max_input:
            dropped['input'] += 1
            continue
        if len(r) < 4:
            dropped['output'] += 1
            continue
        output_len = min(len(r), max_output)
        pool.append({'id': f'sharegpt/{index:06d}', 'source': 'sharegpt', 'src_index': index, 'src_id': src_id,
                     'prompt_token_ids': [bos] + list(p), 'output_token_ids': list(r[:output_len]),
                     'input_len': input_len, 'output_len': output_len, 'reference_len': len(r), 'tokenizer': 'DeepSeek-V2-Lite-hf',
                     'sha256_text': hashlib.sha256((prompt + '\x00' + reference).encode()).hexdigest()})
    random.Random(f'{seed}/sharegpt').shuffle(pool)
    if len(pool) < count:
        raise ValueError(f'{len(pool)} requests pass the filter; {count} requested')
    return pool[:count], {'pool': len(pool), 'dropped': dropped, 'scanned': len(items)}


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q / 100 * len(values)))] if values else 0


def write_set(out, selected, *, seed, fail_fraction, rejoin_seconds, max_input, max_output, sources, selection, tokenizer):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    requests = out / 'requests.jsonl'
    requests.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected), encoding='utf-8')
    count = len(selected)
    workload = {'workload_id': f'failstop_sharegpt_N{count}_v1', 'arrival': 'continuous', 'seed': seed,
                'segments': [{'index': 0, 'kind': 'short', 'request_ids': [row['id'] for row in selected]}]}
    (out / 'workload.json').write_text(json.dumps(workload, indent=1), encoding='utf-8')
    digest = sha256_file(requests)
    failstop = {'version': 'v2-failstop', 'request_file_sha256': digest, 'events': [
        {'id': 'fail', 'type': 'fail', 'anchor': {'kind': 'completed_count', 'value': math.ceil(fail_fraction * count)},
         'replica': 'EDP0', 'coords': COORDS},
        {'id': 'rejoin', 'type': 'rejoin', 'anchor': {'kind': 'after_event_seconds', 'event': 'fail', 'seconds': rejoin_seconds},
         'wait_for': 'shrink_done'}]}
    (out / 'plan_failstop.json').write_text(json.dumps(failstop, indent=1), encoding='utf-8')
    (out / 'plan_nofault.json').write_text(json.dumps({'version': 'v2', 'request_file_sha256': digest, 'switches': []}, indent=1),
                                           encoding='utf-8')
    inputs, outputs = [r['input_len'] for r in selected], [r['output_len'] for r in selected]
    manifest = {'created': time.strftime('%Y-%m-%dT%H:%M:%S'), 'count': count, 'seed': seed, 'tokenizer': tokenizer,
                'tokenizer_rule': 'no chat template, add_special_tokens=False, prepend 1 BOS to prompt',
                'filter': f'4 <= input_len <= {max_input}, reference >= 4 tokens, output_len = min(reference, {max_output})',
                'decode': 'greedy, fixed output_len, EOS ignored, native router',
                'fail': {'fraction': fail_fraction, 'completed_count': math.ceil(fail_fraction * count), 'replica': 'EDP0', 'coords': COORDS},
                'rejoin_seconds': rejoin_seconds, 'sources': sources, 'selection': selection,
                'lengths': {'input_p50': percentile(inputs, 50), 'input_p90': percentile(inputs, 90), 'input_max': max(inputs),
                            'output_mean': round(sum(outputs) / count, 1), 'output_p50': percentile(outputs, 50), 'output_max': max(outputs),
                            'input_tokens': sum(inputs), 'output_tokens': sum(outputs),
                            'at_max_output': sum(1 for r in selected if r['reference_len'] >= max_output)},
                'sha256': {name: sha256_file(out / name) for name in ('requests.jsonl', 'workload.json', 'plan_failstop.json', 'plan_nofault.json')}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--count', type=int, required=True, help='requests in the set (the calibration picks this)')
    parser.add_argument('--sharegpt', default=DEFAULT_SHAREGPT)
    parser.add_argument('--hf', default=DEFAULT_HF, help='HF tokenizer directory')
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--max-input', type=int, default=1024)
    parser.add_argument('--max-output', type=int, default=256)
    parser.add_argument('--fail-fraction', type=float, default=0.25)
    parser.add_argument('--rejoin-seconds', type=float, default=45.0)
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf, trust_remote_code=True)
    if tokenizer.bos_token_id is None:
        raise ValueError('tokenizer has no BOS token')

    def encode(texts, batch=64):
        ids = []
        for start in range(0, len(texts), batch):
            ids += tokenizer(texts[start:start + batch], add_special_tokens=False)['input_ids']
        return ids

    selected, selection = select(conversations(args.sharegpt), encode, tokenizer.bos_token_id, count=args.count, seed=args.seed,
                                 max_input=args.max_input, max_output=args.max_output)
    manifest = write_set(args.out, selected, seed=args.seed, fail_fraction=args.fail_fraction, rejoin_seconds=args.rejoin_seconds,
                         max_input=args.max_input, max_output=args.max_output, sources={args.sharegpt: sha256_file(args.sharegpt)},
                         selection=selection, tokenizer=args.hf)
    print(json.dumps({k: manifest[k] for k in ('count', 'lengths', 'fail', 'sha256')}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
