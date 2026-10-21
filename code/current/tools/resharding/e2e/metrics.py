"""Fail-stop recovery metrics from a launch's result.json and events_actual.json (00_总计划 §8, 07_Step7 §3-4).

Every input time is in seconds since serving_start on the gate clock (serving_probe.write_outputs shifts the
worker records). A token belongs to the replica of the request's latest admit at or before its commit time, so
per-second throughput is counted per dense replica (0/1 = EDP0, 2/3 = EDP1) and globally; re-prefilled tokens
are never in t_tokens (only committed tokens are stamped), so they never count as new tokens.

Recovery criterion: the 5 s trailing mean of the global per-second series >= `ratio` x reference for 3
consecutive whole seconds; the recovery time is the end of the first such second. When the criterion is never
met before the observation ends the metric is censored (value None, reason recorded). The expand/full-recovery
reference is the global mean over the 10 s before the fault (invalid when the fault came earlier than 10 s);
the shrink reference is the degraded reference given by the caller (tokens/s), or, when none is given, the
global mean over the second half of the degraded window (TP2 first token -> rejoin request), marked as such.
"""

import csv
import json
import math
import statistics
from pathlib import Path

EDP0 = (0, 1)
EDP1 = (2, 3)
WINDOW_S, HOLD_S, RATIO, PREFAIL_S = 5, 3, 0.95, 10.0


def load_launch(directory):
    directory = Path(directory)
    result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
    events = json.loads((directory / 'events_actual.json').read_text(encoding='utf-8'))
    for name in ('requests', 'plan', 'workload', 'profile'):
        if result['config'].get('sha256', {}).get(name, events['sha256'][name]) != events['sha256'][name]:
            raise ValueError(f'{directory}: result/events disagree on the {name} hash')
    return {'launch': events['launch'], 'directory': str(directory), 'result': result, 'events': events}


def validate(result):
    """Reject a result whose request rows cannot be reconciled: every request completed with exactly output_len
    stamped tokens in time order, first/done stamps consistent, no admit older than its predecessor."""
    problems = []
    for row in result['requests']:
        tokens = row['t_tokens']
        if len(tokens) != row['output_len'] or tokens != sorted(tokens):
            problems.append(f"seq {row['seq']}: {len(tokens)} token stamps for output_len {row['output_len']}")
        elif not (row['t_first'] == tokens[0] and row['t_done'] == tokens[-1]):
            problems.append(f"seq {row['seq']}: t_first/t_done differ from the token stamps")
        if not row['admits'] or [a['step'] for a in row['admits']] != sorted(a['step'] for a in row['admits']):
            problems.append(f"seq {row['seq']}: admits missing or out of order")
    if sum(r['output_len'] for r in result['requests']) != result['totals']['generated_tokens']:
        problems.append('generated token total differs from the request rows')
    if problems:
        raise ValueError('; '.join(problems[:5]))


def token_events(result):
    """(t, replica, seq) of every committed token."""
    events = []
    for row in result['requests']:
        admits = row['admits']
        for t in row['t_tokens']:
            replica = next(a['replica'] for a in reversed(admits) if a['t'] <= t + 1e-9)
            events.append((t, replica, row['seq']))
    events.sort()
    return events


def per_second(events, horizon):
    """Per-second new tokens per replica and globally over [0, ceil(horizon)) seconds."""
    seconds = max(1, math.ceil(horizon))
    series = {r: [0] * seconds for r in EDP0 + EDP1}
    for t, replica, _ in events:
        index = min(seconds - 1, max(0, int(math.floor(t))))
        series[replica][index] += 1
    series['global'] = [sum(series[r][i] for r in EDP0 + EDP1) for i in range(seconds)]
    series['EDP0'] = [sum(series[r][i] for r in EDP0) for i in range(seconds)]
    series['EDP1'] = [sum(series[r][i] for r in EDP1) for i in range(seconds)]
    return series


def mean_rate(events, start, end, replicas=None):
    """Tokens per second over [start, end) (None when the window is empty)."""
    if end <= start:
        return None
    count = sum(1 for t, r, _ in events if start <= t < end and (replicas is None or r in replicas))
    return count / (end - start)


def recovery_time(series, start, reference, *, ratio=RATIO, window=WINDOW_S, hold=HOLD_S, observed_until=None):
    """End of the first second from `start` after which the trailing-`window` mean of `series` stayed >= ratio x
    reference for `hold` consecutive whole seconds; None + reason when not reached before the series ends."""
    if reference is None:
        return None, 'no reference'
    threshold = ratio * reference
    first = int(math.ceil(start))
    streak = 0
    for second in range(first, len(series)):
        trailing = series[max(0, second - window + 1):second + 1]
        if len(trailing) < window:
            streak = 0
            continue
        streak = streak + 1 if statistics.fmean(trailing) >= threshold else 0
        if streak >= hold:
            return float(second + 1), None
    end = len(series) if observed_until is None else observed_until
    return None, f'censored: criterion not met before {end:.1f} s'


def percentiles(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {'p50': None, 'p99': None, 'n': 0}

    def nearest_rank(q):
        return values[min(len(values) - 1, max(0, math.ceil(q / 100 * len(values)) - 1))]
    return {'p50': nearest_rank(50), 'p99': nearest_rank(99), 'n': len(values)}


def latency(result, window=None):
    """TTFT from the original submission; TPOT = (last - first token) / (n - 1), None with one token.
    Window TTFT: requests whose first token falls in the window; window TPOT: requests with a token interval that
    intersects the window (intervals kept whole; a pause across the window edge is not cut)."""
    ttft, tpot = [], []
    for row in result['requests']:
        tokens = row['t_tokens']
        first = row['t_first'] - row['t_arrive']
        per_token = None if len(tokens) < 2 else (tokens[-1] - tokens[0]) / (len(tokens) - 1)
        if window is None:
            ttft.append(first)
            tpot.append(per_token)
            continue
        lo, hi = window
        if lo <= row['t_first'] < hi:
            ttft.append(first)
        if per_token is not None and any(lo < b and a < hi for a, b in zip(tokens, tokens[1:])):
            tpot.append(per_token)
    return {'ttft_s': percentiles(ttft), 'tpot_s': percentiles(tpot)}


def failstop_metrics(launch, degraded_reference=None):
    result, events = launch['result'], launch['events']
    validate(result)
    tokens = token_events(result)
    horizon = max(result['serving_wall_s'], max(t for t, _, _ in tokens) + 1.0)
    series = per_second(tokens, horizon)
    fs = result['failstop']
    gate = events['recovery_s']
    t_fail, t_rejoin_request, t_spawn = gate['fail'], gate['rejoin_request'], gate['spawn']
    edp0_switches = {s['direction']: s for s in result['switches'] if s['domain'] == 'EDP0'}
    expand_ack = gate['switch_ack_expand']
    tp4_ready = edp0_switches['expand']['milestones']['cutover']
    edp0_tokens = [t for t, r, _ in tokens if r in EDP0]
    tp2_first = next((t for t in edp0_tokens if t > t_fail), None)
    last_before_expand = max((t for t in edp0_tokens if t <= expand_ack), default=None)
    tp4_first = next((t for t in edp0_tokens if t > tp4_ready), None)
    prefail = mean_rate(tokens, t_fail - PREFAIL_S, t_fail) if t_fail >= PREFAIL_S else None
    if degraded_reference is None and tp2_first is not None:
        midpoint = (tp2_first + t_rejoin_request) / 2
        shrink_reference, shrink_reference_source = mean_rate(tokens, midpoint, t_rejoin_request), 'self: second half of the degraded window'
    else:
        shrink_reference, shrink_reference_source = degraded_reference, 'degraded reference (caller)'
    shrink_ttp, shrink_censor = (None, 'no TP2 token') if tp2_first is None else \
        recovery_time(series['global'], tp2_first, shrink_reference, observed_until=t_rejoin_request)
    if shrink_ttp is not None and shrink_ttp > t_rejoin_request:
        shrink_ttp, shrink_censor = None, 'censored: expand started before the shrink recovery'
    expand_ttp, expand_censor = recovery_time(series['global'], tp4_ready, prefail, observed_until=horizon)
    full, full_censor = (expand_ttp, expand_censor)
    metrics = {
        'launch': launch['launch'], 'method': result['method'], 'scenario': result['config']['experiment'], 'mode': result['config']['mode'],
        'run_mode': result['config']['run_mode'], 'timing_valid': result['config']['timing_valid'], 'gpu_sharing': result['gpu_sharing'],
        'stats_valid': fs['stats_valid'],
        'serving_wall_s': result['serving_wall_s'], 'requests': len(result['requests']),
        'generated_tokens': result['totals']['generated_tokens'],
        'throughput_tok_s': result['totals']['generated_tokens'] / result['serving_wall_s'],
        't_fail_s': t_fail, 't_rejoin_request_s': t_rejoin_request, 't_rejoin_start_s': t_spawn, 'tp4_ready_s': tp4_ready,
        'shrink_tfs_s': None if tp2_first is None else tp2_first - t_fail,
        'shrink_ttp_s': None if shrink_ttp is None or tp2_first is None else shrink_ttp - tp2_first,
        'shrink_ttp_censored': shrink_censor, 'shrink_reference_tok_s': shrink_reference, 'shrink_reference_source': shrink_reference_source,
        'expand_time_s': tp4_ready - t_spawn,
        'expand_ttp_s': None if expand_ttp is None else expand_ttp - tp4_ready, 'expand_ttp_censored': expand_censor,
        'full_recovery_s': None if full is None else full - t_fail, 'full_recovery_censored': full_censor,
        'prefail_reference_tok_s': prefail, 'prefail_reference_valid': prefail is not None,
        'edp0_downtime_shrink_s': None if tp2_first is None else tp2_first - t_fail,
        'edp0_downtime_expand_s': None if last_before_expand is None or tp4_first is None else tp4_first - last_before_expand,
        'edp1_pause_s': events['delays']['edp1_pause_s'], 'rejoin_delay_s': events['delays']['rejoin_delay_s'],
        'rejoin_join_s': events['delays']['rejoin_join_s'],
        'replay_prefill_tokens': fs['replay_prefill_tokens'],
        'replay_requests': sum(1 for r in result['requests'] if len(r['admits']) > 1),
        'overlap_steps': {d: s['overlap_steps'] for d, s in edp0_switches.items()},
        'kv_delta_bytes': {d: s['bytes']['kv_delta'] for d, s in edp0_switches.items()},
        'latency_all': latency(result),
        'latency_fail_to_recovery': latency(result, (t_fail, full if full is not None else horizon)),
    }
    if result['method'] == 'restart':
        downtime = restart_downtime(events['worker_events_s'], fs['restart_loaded_s'], tokens)
        metrics.update(restart_downtime=downtime, restart_downtime_s=downtime['total_s'])
    return metrics, series


def restart_downtime(worker, loaded, events):
    """restart, per switch: the stop (worker fail / rejoin_start), the reload done (`loaded`), the first EDP0 token after
    the switch (resume) and the downtime stop -> resume; total_s = both switches (fail-stop and fail-slow alike)."""
    edp0 = [t for t, replica, _ in events if replica in EDP0]
    out = {}
    for direction, stop, done in (('shrink', 'fail', 'shrink_done'), ('expand', 'rejoin_start', 'expand_done')):
        resume = next((t for t in edp0 if t >= worker[done]), None)
        out[direction] = {'stop_s': worker[stop], 'loaded_s': loaded[direction], 'resume_s': resume,
                          'downtime_s': None if resume is None else resume - worker[stop]}
    parts = [out[d]['downtime_s'] for d in out]
    out['total_s'] = None if None in parts else sum(parts)
    return out


def nofault_metrics(launch):
    result = launch['result']
    validate(result)
    tokens = token_events(result)
    horizon = max(result['serving_wall_s'], max(t for t, _, _ in tokens) + 1.0)
    series = per_second(tokens, horizon)
    metrics = {'launch': launch['launch'], 'method': result['method'], 'scenario': result['config']['experiment'],
               'mode': result['config']['mode'], 'run_mode': result['config']['run_mode'], 'timing_valid': result['config']['timing_valid'],
               'gpu_sharing': result['gpu_sharing'], 'stats_valid': True, 'serving_wall_s': result['serving_wall_s'], 'requests': len(result['requests']),
               'generated_tokens': result['totals']['generated_tokens'],
               'throughput_tok_s': result['totals']['generated_tokens'] / result['serving_wall_s'],
               'latency_all': latency(result)}
    return metrics, series


def analyze(launch, degraded_reference=None):
    if launch['result']['failstop'] is None:
        return nofault_metrics(launch)
    return failstop_metrics(launch, degraded_reference)


SUMMARY_FIELDS = ('serving_wall_s', 'throughput_tok_s', 'shrink_tfs_s', 'shrink_ttp_s', 'expand_time_s', 'expand_ttp_s', 'full_recovery_s',
                  'edp0_downtime_shrink_s', 'edp0_downtime_expand_s', 'edp1_pause_s', 'rejoin_delay_s', 'replay_prefill_tokens',
                  'replay_requests', 'ttft_p50_s', 'ttft_p99_s', 'tpot_p50_s', 'tpot_p99_s', 'window_ttft_p99_s', 'window_tpot_p99_s',
                  'restart_downtime_s')


def flatten(metrics):
    flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    flat.update(ttft_p50_s=metrics['latency_all']['ttft_s']['p50'], ttft_p99_s=metrics['latency_all']['ttft_s']['p99'],
                tpot_p50_s=metrics['latency_all']['tpot_s']['p50'], tpot_p99_s=metrics['latency_all']['tpot_s']['p99'])
    window = metrics.get('latency_fail_to_recovery')
    flat.update(window_ttft_p99_s=None if window is None else window['ttft_s']['p99'],
                window_tpot_p99_s=None if window is None else window['tpot_s']['p99'])
    return flat


def summarize(all_metrics):
    """Per (method, scenario, mode, gpu_sharing) group and metric: mean, sample std (ddof=1, None below two samples), n_valid,
    n_censored; launches on shared GPUs (--shared-gpu-ok) never mix with exclusive ones."""
    groups = {}
    for metrics in all_metrics:
        groups.setdefault((metrics['method'], metrics['scenario'], metrics['mode'], metrics['gpu_sharing']), []).append(flatten(metrics))
    rows = []
    for (method, scenario, mode, gpu_sharing), members in sorted(groups.items()):
        for field in SUMMARY_FIELDS:
            values = [m[field] for m in members if field in m and m[field] is not None]
            censored = sum(1 for m in members if field in m and m[field] is None)
            if not values and not censored:
                continue
            rows.append({'method': method, 'scenario': scenario, 'mode': mode, 'gpu_sharing': gpu_sharing, 'metric': field,
                         'mean': statistics.fmean(values) if values else None,
                         'sample_std': statistics.stdev(values) if len(values) >= 2 else None,
                         'n_valid': len(values), 'n_censored_or_missing': censored, 'n_launches': len(members)})
    return rows


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if row.get(k) is None else row.get(k)) for k in fields})


def write_series(path, series):
    fields = ['second', 'global', 'EDP0', 'EDP1', *[f'replica_{r}' for r in EDP0 + EDP1]]
    rows = [{'second': i, 'global': series['global'][i], 'EDP0': series['EDP0'][i], 'EDP1': series['EDP1'][i],
             **{f'replica_{r}': series[r][i] for r in EDP0 + EDP1}} for i in range(len(series['global']))]
    write_csv(path, rows, fields)
