"""result.json per the W4 schema, fail-stop fields alongside, and the --check self-audit."""

import json
from pathlib import Path

SCHEMA_VERSION = 'e2e-serving-v1'


def totals(request_rows, duration_s, logical_steps):
    inputs = sum(r['input_len'] for r in request_rows)
    outputs = sum(r['output_len'] for r in request_rows)
    generated = sum(r['generated_tokens'] for r in request_rows)
    return {'duration_s': duration_s, 'input_tokens': inputs, 'output_tokens': outputs, 'generated_tokens': generated,
            'throughput_total_tok_s': (inputs + generated) / duration_s if duration_s else None,
            'throughput_gen_tok_s': generated / duration_s if duration_s else None, 'logical_steps': logical_steps}


def build_result(*, method, workload_id, segments_run, request_rows, switches, phase_durations_s, attention_impl, config,
                 duration_s, logical_steps, failstop=None):
    result = {'schema_version': SCHEMA_VERSION, 'method': method, 'workload_id': workload_id, 'segments_run': list(segments_run),
              'totals': totals(request_rows, duration_s, logical_steps), 'switches': switches, 'requests': request_rows,
              'phase_durations_s': phase_durations_s, 'attention_impl': attention_impl, 'config': config, 'failstop': failstop}
    return result


def write_result(path, result):
    Path(path).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def recompute_from_timeline(rows):
    """Independent recount from timeline rows: generated tokens from step rows, phase time from steps + stalls."""
    steps = [r for r in rows if r['kind'] == 'step']
    stalls = [r for r in rows if r['kind'] == 'stall']
    return {'generated_tokens': sum(r['gen_tokens'] for r in steps),
            'step_seconds': sum(r['step_time'] for r in steps), 'stall_seconds': sum(r['seconds'] for r in stalls),
            'completed': sum(r['completed'] for r in steps)}


def check(result, timeline_rows, tolerance=1e-3):
    """Σ gen_tokens equals the generated total; phase durations and stalls add up to the run duration (±0.1%)."""
    recount = recompute_from_timeline(timeline_rows)
    phase_total = sum(result['phase_durations_s'].values())
    duration = result['totals']['duration_s']
    report = {'generated_tokens_match': recount['generated_tokens'] == result['totals']['generated_tokens'],
              'completed_match': recount['completed'] == len(result['requests']),
              'phase_sum_s': phase_total, 'duration_s': duration,
              'phase_sum_matches_duration': abs(phase_total - duration) <= tolerance * max(duration, 1e-9),
              'timeline_seconds': recount['step_seconds'] + recount['stall_seconds']}
    report['timeline_matches_duration'] = abs(report['timeline_seconds'] - duration) <= tolerance * max(duration, 1e-9)
    report['passed'] = all(report[k] for k in ('generated_tokens_match', 'completed_match', 'phase_sum_matches_duration',
                                                'timeline_matches_duration'))
    return report
