"""Fail-slow control gate on SL3060 (run_failstop_e2e --scenario failslow): the injection timeline on the gate clock.

serving_start -(warmup)-> slow_start -> co-runners ready = inject_start -(detect)-> detect [-> shrink ... shrink_done]
and, independently, inject_start + D -> slow_stop -> slow_stopped = inject_end -> recovered [-> expand ... expand_done]
-> deadline = inject_start + D + tail: the control service closes (requests still running stay unfinished).
nofault injects nothing (inject_start/inject_end/detect are the nominal times), ignore never switches, and
weavetp/anchortp/restart run the fail-stop shrink/expand cycle in emulate mode: detection is the fault, the
recovery is the rejoin. Results go to failslow_result.json (run_probe's own outcome is launch_result.json).
"""

import threading
import time
from pathlib import Path

from common import releases_fault_set, require, write_json

from tools.resharding.e2e import requests_io

# event -> events that must precede it; injection-only and switch-only events are left out of the other variants.
PREREQUISITES = {'serving_start': (), 'slow_start': ('serving_start',), 'inject_start': ('serving_start', 'slow_start'),
                 'detect': ('inject_start',), 'shrink_done': ('detect',), 'slow_stop': ('detect',), 'inject_end': ('slow_stop', 'detect'),
                 'recovered': ('inject_end',), 'expand_done': ('recovered', 'shrink_done'), 'deadline': ('recovered',),
                 'serving_done': ('deadline', 'expand_done')}
INJECTION, SWITCHING = ('slow_start', 'slow_stop'), ('shrink_done', 'expand_done')


class Timeline:
    """The run's events in a legal order (detection and the end of the injection race only with the shrink)."""

    def __init__(self, settings, inject, switches):
        require(0 < settings['detect_s'] < settings['slow_s'], 'detection must fall inside the slow period')
        self.events = [e for e in PREREQUISITES if (inject or e not in INJECTION) and (switches or e not in SWITCHING)]
        self.times, self.lock = {}, threading.Lock()

    def mark(self, event, t):
        with self.lock:
            require(event in self.events and event not in self.times, f'unexpected or repeated event {event}')
            before = [p for p in PREREQUISITES[event] if p in self.events]
            require(all(p in self.times and self.times[p] <= t for p in before), f'{event} at {t} before {before}: {self.times}')
            self.times[event] = t

    def complete(self):
        return [e for e in self.events if e not in self.times]


def run(config, bus, journal, service, times, publish, wait, wait_until, fault_cycle):
    """The timeline; returns it with the injection records. fault_cycle(fire, rejoin) runs the shrink/expand cycle."""
    settings = config['failslow']
    inject, switches = bool(settings['level']), releases_fault_set(config)
    timeline = Timeline(settings, inject, switches)
    record = {'timeline': timeline, 'started': None, 'stopped': None, 'errors': []}
    timeline.mark('serving_start', times['serving_start'])

    def at(t, what):
        wait_until(lambda: time.monotonic() >= t, max(1.0, t - time.monotonic()) + 60, what)

    at(times['serving_start'] + settings['warmup_s'], 'the end of the warmup')
    if inject:
        publish('slow_start', {'level': settings['level'], 'fault_gpus': settings['fault_gpus']})
        timeline.mark('slow_start', times['slow_start'])
        record['started'] = wait(['slow_started'], config['startup_timeout_s'])[0]
    start = time.monotonic() if inject else times['serving_start'] + settings['warmup_s']
    timeline.mark('inject_start', start)
    times['inject_start'] = start
    detected = threading.Event()

    def end_injection():
        try:
            detected.wait()
            at(start + settings['slow_s'], 'the end of the slow period')
            if inject:
                publish('slow_stop')
                timeline.mark('slow_stop', times['slow_stop'])
                record['stopped'] = wait(['slow_stopped'], config['stage_timeout_s'])[0]
                require(not record['stopped']['leftover'], f"co-runners left: {record['stopped']['leftover']}")
            times['inject_end'] = time.monotonic() if inject else start + settings['slow_s']
            timeline.mark('inject_end', times['inject_end'])
            publish('recovered')
            timeline.mark('recovered', times['recovered'])
        except BaseException as error:  # surfaced by the main flow below
            record['errors'].append(repr(error))

    stopper = threading.Thread(target=end_injection, daemon=True)
    stopper.start()
    at(start + settings['detect_s'], 'detection')
    publish('detect', {'method': config['method'], 'detect_s': settings['detect_s']})
    timeline.mark('detect', times['detect'])
    detected.set()

    def recovered():
        timeline.mark('shrink_done', times['shrink_done'])
        stopper.join(config['launch_timeout_s'])
        require(not record['errors'] and 'recovered' in times, f"injection end failed: {record['errors']}")
        return times['recovered']

    if switches:
        fire = {'kind': 'failslow_detect', 'event_id': f"{config['launch']}/detect", 'detect_s': settings['detect_s']}
        record['cycle'] = fault_cycle(lambda: fire, recovered)
        timeline.mark('expand_done', times['expand_done'])
    else:
        stopper.join(config['launch_timeout_s'])
        require(not record['errors'] and 'recovered' in times, f"injection end failed: {record['errors']}")
    at(start + settings['slow_s'] + settings['tail_s'], 'the deadline')
    service.close()
    times['deadline'] = time.monotonic()
    timeline.mark('deadline', times['deadline'])
    return record


def write_outputs(root, config, merged, summaries, times, record, edp0_slots, loaded, wait):
    """failslow_result.json from the fail-stop merge (serving_probe.merge_summaries: admitted request rows, leaders,
    switches, gate times) plus the timeline, the injection records, the TP4 hash checks, the EDP0 decode steps and
    restart's reload times (serving_probe.restart_loaded; None for the other methods)."""
    settings, origin = config['failslow'], times['serving_start']
    timeline = record['timeline']
    timeline.mark('serving_done', times['serving_done'])
    correctness, switches = config['run_mode'] == 'correctness', releases_fault_set(config)
    hashes = dict(zip(edp0_slots, wait([f'tp4_hash/{s}' for s in edp0_slots]))) if correctness and switches else {}
    rows =[dict(row, generated_tokens=len(row['t_tokens'])) for row in merged['rows']]  # a request may be cut at the deadline
    steps = sorted([row['t'] + merged['clocks'][f'EDP0/{replica}']['shift_to_gate_s'], row['t_decode']]
                   for (domain, replica), summary in summaries.items() if domain == 'EDP0'
                   for row in summary['timeline'] if row['kind'] == 'step' and row['t_decode'] > 0)
    leader = merged['leaders']['EDP0']
    unfinished = sum(1 for r in rows if r['t_done'] is None)
    check = {'event_order': not timeline.complete(), 'missing_events': timeline.complete(), 'cut_by_deadline': unfinished > 0,
             'tp4_hash_passed': all(h['passed'] for h in hashes.values()) if hashes else None}
    check['passed'] = check['event_order'] and check['tp4_hash_passed'] is not False
    result = {
        'schema_version': 'failslow-v1', 'launch': config['launch'], 'commit': config['commit'], 'method': settings['label'],
        'worker_method': config['method'], 'level': settings['level'], 'fault_gpus': settings['fault_gpus'],
        'settings': {k: settings[k] for k in ('warmup_s', 'slow_s', 'tail_s', 'detect_s')},
        'config': {k: config[k] for k in ('run_mode', 'mode', 'shared_gpu_mode', 'timing_valid', 'launch_timeout_s', 'stage_timeout_s')},
        'gpu_sharing': 'shared' if config['shared_gpu_mode'] else 'exclusive',
        'b_max': config['serving']['b_max'], 'time_base': 'seconds since serving_start on the SL3060 gate clock',
        'sha256': {'requests': requests_io.sha256_file(config['serving']['requests'])},
        'events_s': {k: v - origin for k, v in timeline.times.items()},
        'gate_s': merged['gate_s'],
        'deadline_overrun_s': times['deadline'] - (times['inject_start'] + settings['slow_s'] + settings['tail_s']),
        'worker_events_s': leader['failstop_events'], 'restart_loaded_s': loaded,
        'injection': {'started': record['started'], 'stopped': record['stopped']},
        'switches': [{k: s[k] for k in ('domain', 'direction', 'overlap_steps', 'bytes', 'milestones', 'audit')} for s in merged['switches']],
        'tp4_hash': {str(s): h for s, h in hashes.items()}, 'replayed_tokens': leader['replayed_tokens'],
        'requests_admitted': len(rows), 'requests_completed': len(rows) - unfinished, 'requests_unfinished': unfinished,
        'requests': rows, 'edp0_decode_steps': steps, 'clocks': merged['clocks'], 'check': check}
    write_json(Path(root) / 'failslow_result.json', result)
    return {'check': check, 'result': result}
