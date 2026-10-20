"""Fail-slow figure metrics from failslow_result.json (failstop/failslow_probe.py); torch-free.

Window = [inject_start, inject_start + D + tail) on the gate clock (nofault: the nominal injection start), the same
length for every method. Throughput = committed output tokens in the window / its length; the figure normalises
it to the mean of the nofault launches of the same timeline. TTFT runs from the first admission (every request is
submitted at t = 0, so the arrival-based TTFT is the queue position); TPOT is metrics.latency's definition.
slow_rate_measured = median EDP0 decode time in [inject_start, detect) / in the 30 s before the injection.
restart downtime per switch = first EDP0 token after the switch - the stop (worker fail / rejoin_start).
"""

import json
import math
import statistics
from pathlib import Path

from . import metrics as m

BASELINE_S = 30.0
SEVERITIES, METHODS = ('W', 'M', 'S', '-'), ('ignore', 'restart', 'anchortp', 'weavetp', 'nofault')
FIG1 = ('severity', 'method', 'n', 'mean', 'std', 'p99_tpot', 'p99_ttft', 'slow_rate_measured_mean', 'restart_downtime_s')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def window(result):
    start = result['events_s']['inject_start']
    return start, start + result['settings']['slow_s'] + result['settings']['tail_s']


def window_throughput(events, start, end):
    """Committed tokens per second over [start, end)."""
    return m.mean_rate(events, start, end)


def window_latency(rows, start, end):
    """P50/P99 of TTFT (first admit -> first token; first token in the window) and TPOT (a token interval in it)."""
    ttft, tpot = [], []
    for row in rows:
        tokens = row['t_tokens']
        if row['t_first'] is not None and start <= row['t_first'] < end:
            ttft.append(row['t_first'] - row['t_admit'])
        if len(tokens) > 1 and any(start < b and a < end for a, b in zip(tokens, tokens[1:])):
            tpot.append((tokens[-1] - tokens[0]) / (len(tokens) - 1))
    return {'ttft_s': m.percentiles(ttft), 'tpot_s': m.percentiles(tpot)}


def slow_rate(steps, inject_start, detect):
    during = [d for t, d in steps if inject_start <= t < detect]
    before = [d for t, d in steps if inject_start - BASELINE_S <= t < inject_start]
    return statistics.median(during) / statistics.median(before) if during and before else None


def restart_downtime(result, events):
    """Per switch: stop, load done, first EDP0 token after the switch (resume), downtime; None for other methods."""
    if result['method'] != 'restart':
        return None
    worker, loaded = result['worker_events_s'], result['restart_loaded_s'] or {}
    edp0 = [t for t, replica, _ in events if replica in m.EDP0]
    out = {}
    for direction, stop, done in (('shrink', 'fail', 'shrink_done'), ('expand', 'rejoin_start', 'expand_done')):
        resume = next((t for t in edp0 if t >= worker[done]), None)
        out[direction] = {'stop_s': worker[stop], 'loaded_s': loaded.get(direction), 'resume_s': resume,
                          'downtime_s': None if resume is None else resume - worker[stop]}
    parts = [out[d]['downtime_s'] for d in out]
    out['total_s'] = None if None in parts else sum(parts)
    return out


def analyze(result):
    """One launch: its metrics and its per-second series (timeline.csv)."""
    start, end = window(result)
    events = m.token_events(result)
    series = m.per_second(events, max([end] + [t for t, _, _ in events]))
    downtime = restart_downtime(result, events)
    return {'launch': result['launch'], 'method': result['method'], 'severity': result['level'] or '-',
            'settings': result['settings'], 'window_s': [start, end], 'throughput_tok_s': window_throughput(events, start, end),
            **window_latency(result['requests'], start, end),
            'slow_rate_measured': slow_rate(result['edp0_decode_steps'], start, result['events_s']['detect']),
            'restart_downtime': downtime, 'restart_downtime_s': downtime['total_s'] if downtime else None,
            'requests_unfinished': result['requests_unfinished'], 'deadline_overrun_s': result['deadline_overrun_s'],
            'check_passed': result['check']['passed']}, series


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def fig1(launches):
    """One row per (severity, method): throughput normalised to the mean nofault throughput of the same timeline."""
    timeline = lambda x: tuple(sorted(x['settings'].items()))  # noqa: E731
    reference = {}
    for x in launches:
        if x['method'] == 'nofault' and x['throughput_tok_s'] is not None:
            reference.setdefault(timeline(x), []).append(x['throughput_tok_s'])
    groups = {}
    for x in launches:
        base = mean(reference.get(timeline(x), []))
        x['normalized'] = None if not base or x['throughput_tok_s'] is None else x['throughput_tok_s'] / base
        groups.setdefault((x['severity'], x['method']), []).append(x)
    rows = []
    for key in sorted(groups, key=lambda k: (SEVERITIES.index(k[0]) if k[0] in SEVERITIES else 9,
                                             METHODS.index(k[1]) if k[1] in METHODS else 9)):
        group, values = groups[key], [x['normalized'] for x in groups[key] if x['normalized'] is not None]
        rows.append({'severity': key[0], 'method': key[1], 'n': len(values), 'mean': mean(values),
                     'std': statistics.stdev(values) if len(values) > 1 else None,
                     'p99_tpot': mean(x['tpot_s']['p99'] for x in group), 'p99_ttft': mean(x['ttft_s']['p99'] for x in group),
                     'slow_rate_measured_mean': mean(x['slow_rate_measured'] for x in group),
                     'restart_downtime_s': mean(x['restart_downtime_s'] for x in group) if key[1] == 'restart' else None})
    return rows


def format_value(value):
    return '' if value is None else (f'{value:.6g}' if isinstance(value, float) and math.isfinite(value) else value)


def write_fig1(out, launches):
    rows = fig1(launches)
    m.write_csv(Path(out) / 'fig1.csv', [{k: format_value(v) for k, v in row.items()} for row in rows], FIG1)
    return rows
