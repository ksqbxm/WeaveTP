"""Step 6 B3 control gate on SL3060: the Step 5 fault/rejoin gates driven by the serving plan's anchors,
with the admission control node (remote_control.ControlService) serving both EDP domains over the bus.

Flow: calibrate/preflight/topology (Steps 1-5) -> serving starts (phase 0 reports) -> completed_count
anchor fires -> both domains ordered to switch (shrink) -> kill/release -> transition 1 -> build ->
phase 1 reports (shrink_done) -> rejoin due = t_fail + seconds, waits for shrink_done -> reclaim/spawn ->
switch (expand) -> transition 2 -> build -> phase 2 reports (expand_done) -> every request completes ->
summaries -> finish. Writes result.json, check.json and events_actual.json next to the launch.
"""

import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from async_state import audit_sequence
from common import NODES, failed_slots, require, topology, write_json
from model_shards import model_local_orders

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from async_topology import AsyncSession as FaultSession

from megatron.core.failstop import FaultBackend, replica_group_specs
from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e import remote_control, requests_io, results
from tools.resharding.e2e.failstop_adapter import rank_map_from_rows

PROBE_KEYS = ('profile_file', 'reference_root', 'deterministic_mode', 'kv_heads', 'memory_margin_bytes', 'memory_limit_bytes', 'serving')
DOMAINS = {'EDP0': (0, 1), 'EDP1': (2, 3)}


def anchors(plan, total):
    fail = next(e for e in plan['events'] if e['type'] == 'fail')
    rejoin = next(e for e in plan['events'] if e['type'] == 'rejoin')
    anchor = fail['anchor']
    require(anchor['kind'] == 'completed_count' and rejoin['anchor']['kind'] == 'after_event_seconds', 'B3 plan needs completed_count + after_event_seconds anchors')
    count = anchor['value'] if 'value' in anchor else requests_io.failstop_fail_count(total, anchor['fraction'])
    return count, float(rejoin['anchor']['seconds']), fail, rejoin


def control(config, bus, journal, check):
    root, serving, mode = Path(config['output']), config['serving'], config['mode']
    requests = requests_io.load_requests(serving['requests'])
    workload = requests_io.load_workload(serving['workload'], requests)
    instances = requests_io.expand_instances(workload, requests)
    plan = requests_io.load_plan(serving['plan'], requests_io.sha256_file(serving['requests']))
    fail_count, rejoin_seconds, fail_spec, rejoin_spec = anchors(plan, len(instances))
    times = {}

    def wait(keys, timeout=None):
        return bus.wait(keys, timeout, check)

    def publish(key, value=True):
        journal.event(key, payload=value)
        bus.put(key, value)
        times[key] = time.monotonic()

    for node in NODES:
        samples = []
        for index in range(5):
            sent = time.monotonic()
            bus.put(f'ping/{node}/{index}', sent)
            remote = wait([f'pong/{node}/{index}'], config['startup_timeout_s'])[0]
            end = time.monotonic()
            samples.append({'offset': (sent + end) / 2 - remote, 'error': (end - sent) / 2})
        publish(f'clock/{node}', min(samples, key=lambda x: x['error']))
    preflight = wait([f'preflight/{n}' for n in NODES])
    for node, report in zip(NODES, preflight):
        require(report['host'].split('.')[0].upper() == node, 'wrong physical host')
        for key in ('commit', 'launch', 'mode', 'rounds', 'port', 'shared_gpu_mode', 'quick_mode', 'correctness_only', 'probe') + PROBE_KEYS:
            require(report['config'][key] == config[key], f'node config mismatch: {key}')
    rows = topology(dict(zip(NODES, [r['gpus'] for r in preflight])))
    victims = failed_slots(rows)
    publish('topology', rows)
    publish('start')
    wait([f'verified/0/{n}' for n in NODES])
    initial = dict(zip(range(16), wait([f'cuda/0/{s}' for s in range(16)])))
    publish('cuda_verified/0')
    session = FaultSession(rows, config['launch'], FaultBackend(mode, fail=lambda *_: None, rejoin=lambda *_: None),
                           local_orders=model_local_orders(rows))

    # ---- the control node --------------------------------------------------------------------------------------
    controller = ctl.AdmissionController(ctl.Ledger(rank_map_from_rows(session.rank_map())), instances,
                                         capacity=lambda replica: 0, b_max=serving['b_max'], barrier=workload['arrival'] == 'phased')
    fixed = serving.get('capacity_tokens')

    def base_capacity(replica):
        return fixed if fixed else ctl.effective_capacity(controller.ledger.replicas[replica].tp)
    service = remote_control.ControlService(controller, bus, time.monotonic, DOMAINS, base_capacity).start()
    thread = service.thread()

    def guard():
        check()
        require(service.error is None, f'control service failed: {service.error!r}')

    def wait_until(predicate, timeout, what):
        deadline = time.monotonic() + timeout
        while not predicate():
            guard()
            if bus.has('failure'):
                raise RuntimeError(f"peer failure: {bus.get('failure')}")
            require(time.monotonic() < deadline, f'timeout waiting for {what}')
            time.sleep(0.01)

    reports, maps, pauses = {}, {}, []

    def validate(phase):
        members = session.current_world()['membership']
        session.adopt_group_order([{'replica': replica, 'field': field, 'actual': sorted(ranks)}
                                   for replica, field, ranks in replica_group_specs(session.replicas())])
        values = wait([f'phase/0/{phase}/{s}' for s in members])
        audit_sequence(values)
        for s, report in zip(members, values):
            require(report['slot'] == s and report['pid'] == initial[s]['pid'], 'actual membership/PID mismatch')
            expected = session.rank_map()
            differing = [(a, b) for a, b in zip(report['rank_map'], expected) if a != b]
            require(not differing, f'actual group mapping mismatch for slot {s} (worker vs gate): {differing[:2]}')
            require(report['checks_passed'], 'worker correctness check failed')
            if mode == 'real' and phase == 2 and s in victims:
                require(report['empty_receiver'], 'replacement was not an empty receiver')
        if phase == 2:
            healthy = [r['rejoin_pause_s'] for r in values if rows[r['slot']]['replica'] == 'EDP1']
            require(all((p is None) == (mode == 'emulate') for p in healthy), 'EDP1 pause record mismatch')
            if mode == 'real':
                pauses.append({'round': 0, 'edp1_rejoin_pause_s': max(healthy),
                               'pause_semantics': 'abort of the old world to the rebuilt groups adopted (serving resumes after the switch)'})
        reports[f'0/{phase}'] = values
        maps[f'0/{phase}'] = session.rank_map()
        write_json(root / 'rank_map.json', maps)
        write_json(root / 'serving_phase_checks.json', reports)
        return time.monotonic()

    times['serving_start'] = validate(0)
    # ---- fault: the completed_count anchor --------------------------------------------------------------------
    wait_until(lambda: service.completed() >= fail_count, config['launch_timeout_s'], f'{fail_count} completions')
    times['fail'] = time.monotonic()
    trigger = {'completed_count': service.completed(), 'threshold': fail_count, 'overshoot': service.completed() - fail_count,
               'event_id': f"{config['launch']}/fail", 'kind': 'completed_count'}
    journal.event('anchor_fail', **trigger)
    service.hold.update(DOMAINS)  # the cycle must complete even if the workload drains first (then stats are invalid)
    for domain in DOMAINS:
        service.trigger(domain, 'shrink')
    wait([f'switch_ack/shrink/{s}' for s in range(16)])
    times['switch_ack_shrink'] = time.monotonic()
    publish('kill')
    if mode == 'real':
        wait(['kill_sent'])
        killed = wait(['kill_exited'])[0]
        require({r['slot'] for r in killed} == set(victims) and all(
            r['pid'] == initial[r['slot']]['pid'] and r['exit']['returncode'] == -signal.SIGKILL for r in killed), 'fault exit identity mismatch')
        times['kill_exited'] = time.monotonic()
    else:
        wait(['emulate_released'])
        released = wait([f'released/0/{s}' for s in victims])
        require(all(r['src_none'] and r['dst_none'] and r['data_bytes'] == r['marker_bytes'] == 0 for r in released), 'failed GPU data path not isolated')
        times['emulate_released'] = time.monotonic()
    event = session.event('fail')
    session.fail(victims, event)
    publish('transition/0/1', asdict(event))
    members = session.current_world()['membership']
    wait([f'aborted/0/1/{s}' for s in members])
    times['aborted_1'] = time.monotonic()
    publish('build/0/1')
    times['shrink_done'] = validate(1)
    session.complete()
    # ---- rejoin: due time, reclaim, spawn, expand -------------------------------------------------------------
    due = times['fail'] + rejoin_seconds
    wait_until(lambda: time.monotonic() >= due, max(1.0, rejoin_seconds + 5), 'rejoin due')
    times['rejoin_due'], times['rejoin_request'] = due, max(due, times['shrink_done'])
    reclaim = None
    if mode == 'real':
        publish('rejoin_request')
        reclaim = wait(['reclaimed'], config['reclaim_timeout_s'])[0]
        times['reclaim_done'] = time.monotonic()
    publish('spawn')
    wait([f'verified/2/{n}' for n in NODES], config['startup_timeout_s'])
    returned = dict(zip(victims, wait([f'cuda/2/{s}' for s in victims])))
    require(all(returned[s]['empty_receiver'] for s in victims), 'rejoin target prefilled')
    require(all((returned[s]['pid'] != initial[s]['pid']) == (mode == 'real') for s in victims), 'incorrect rejoin PID semantics')
    initial.update(returned)
    publish('cuda_verified/2')
    for domain in DOMAINS:
        service.trigger(domain, 'expand')
    wait([f'switch_ack/expand/{s}' for s in members])
    times['switch_ack_expand'] = time.monotonic()
    event = session.event('rejoin')
    session.rejoin(victims, event)
    publish('transition/0/2', asdict(event))
    wait([f'aborted/0/2/{s}' for s in members])
    times['aborted_2'] = time.monotonic()
    publish('build/0/2')
    times['world16_ready'] = times['build/0/2']
    times['expand_done'] = validate(2)
    session.complete()
    drained_before_rejoin = service.completed() == len(instances)
    service.hold.clear()
    # ---- drain ----------------------------------------------------------------------------------------------------
    wait_until(lambda: all(service.finished.values()), config['launch_timeout_s'], 'every request to complete')
    keys = [(domain, replica) for domain, replicas in DOMAINS.items() for replica in replicas]
    summaries = dict(zip(keys, wait([f'summary/{d}/{r}' for d, r in keys])))
    times['serving_done'] = time.monotonic()
    service.stop()
    thread.join(timeout=5)
    require(service.error is None, f'control service failed: {service.error!r}')
    require(controller.idle() and len(controller.done) == len(instances), 'requests left unfinished')
    publish('finish')
    wait([f'done/{s}' for s in range(16)])
    wait([f'clean/{n}' for n in NODES], config['reclaim_timeout_s'] + config['stage_timeout_s'])
    outputs = write_outputs(root, config, instances, workload, summaries, times, trigger, fail_spec, rejoin_spec, reports, maps, initial,
                            pauses, reclaim, controller, drained_before_rejoin)
    metrics = {'edp1_rejoin_pause_s': max((p['edp1_rejoin_pause_s'] for p in pauses), default=None),
               'requests_completed': len(controller.done), 'check_passed': outputs['check']['passed'],
               'replay_prefill_tokens': outputs['result']['failstop']['replay_prefill_tokens'],
               'kv_check_passed': outputs['result']['failstop']['expand_kv_check_passed'], 'stats_valid': not drained_before_rejoin,
               'shrink_wall_s': times['shrink_done'] - times['fail'], 'expand_wall_s': times['expand_done'] - times['rejoin_request']}
    return {'status': 'passed', 'mode': mode, 'rounds': 1, 'epochs': reports, 'rank_maps': maps, 'shared_gpu_mode': config['shared_gpu_mode'],
            'quick_mode': config['quick_mode'], 'commit': config['commit'], 'metrics': metrics, 'rejoin_pauses': pauses,
            'events_actual': str(root / 'events_actual.json')}


def write_outputs(root, config, instances, workload, summaries, times, trigger, fail_spec, rejoin_spec, reports, maps, initial, pauses,
                  reclaim, controller, drained_before_rejoin):
    """result.json / check.json (W4) and events_actual.json (06_Step6 §4) from the domain summaries and the gate's times."""
    logs = {}
    for summary in summaries.values():
        logs.update({int(k): v for k, v in summary['request_log'].items()})
    rows = []
    for instance in instances:
        log = logs[instance.seq]
        rows.append({'seq': instance.seq, 'id': instance.id, 'segment': instance.segment, 'kind': instance.kind, 'replica': log['replica'],
                     'layout_at_admit': log['layout_at_admit'], 'input_len': instance.input_len, 'output_len': instance.output_len,
                     'generated_tokens': instance.output_len, 't_arrive': instance.arrive_s, 't_admit': log['t_admit'],
                     't_first': log['t_first'], 't_done': log['t_done']})
    timeline_rows = [r for s in summaries.values() for r in s['timeline']]  # one timeline per dense replica
    phases = {}
    for row in timeline_rows:
        key = row['phase'] if row['kind'] == 'step' else 'switching'
        phases[key] = phases.get(key, 0.0) + (row['step_time'] if row['kind'] == 'step' else row['seconds'])
    leaders = {domain: summaries[domain, min(replicas)] for domain, replicas in DOMAINS.items()}
    switches = [dict(s, domain=domain) for domain, summary in leaders.items() for s in summary['switches']]
    edp0 = leaders['EDP0']
    duration = sum(phases.values())
    events_s = {**edp0['failstop_events'], **{k: v - times['serving_start'] for k, v in times.items() if isinstance(v, float)}}
    kv_checks = [s['validation'] for s in edp0['switches'] if s['direction'] == 'expand']
    failstop = {'events_s': events_s, 'replay_prefill_tokens': edp0['replayed_tokens'],
                'peak_eviction': [e for s in switches for e in s['evictions']],
                'expand_kv_check_passed': all(v.get('passed') for v in kv_checks) and bool(kv_checks),
                'tfs_breakdown': {'fault_to_transition_s': times['transition/0/1'] - times['fail'],
                                  'world_rebuild_s': times['build/0/1'] - times['transition/0/1'],
                                  'weights_and_cutover_s': times['shrink_done'] - times['build/0/1'],
                                  'shrink_wall_s': times['shrink_done'] - times['fail'],
                                  'expand_wall_s': times['expand_done'] - times['rejoin_request']},
                'advantage_sources': ['weights via P2P from residual TP4 + EDP1, no checkpoint reload', 'EDP1 keeps serving between the waves',
                                      'emulate: no process rebuild; real: four processes rebuilt', 'replay only EDP0 in-flight requests'],
                'serving_wall_s': times['serving_done'] - times['serving_start'],
                'duration_note': 'totals.duration_s accumulates the step time of every dense replica timeline; serving_wall_s is wall time',
                'stats_valid': not drained_before_rejoin,
                'stats_invalid_reason': 'every request completed before the rejoin; recovery throughput is not measurable' if drained_before_rejoin else None}
    result = results.build_result(method='weavetp', workload_id=workload['workload_id'], segments_run=sorted({i.segment for i in instances}),
                                  request_rows=rows, switches=switches, phase_durations_s=phases,
                                  attention_impl=edp0['attention_impl'],
                                  config={'run_mode': 'correctness' if config['deterministic_mode'] else 'timing', 'experiment': 'failstop',
                                          'backend': 'block', 'control': 'remote', 'mode': config['mode'], 'validate_cutover': True,
                                          'timing_valid': False, 'decode_policy': config['serving']['decode_policy'],
                                          'b_max': config['serving']['b_max'], 'commit': config['commit']},
                                  duration_s=duration, logical_steps=sum(s['steps'] for s in summaries.values()), failstop=failstop)
    results.write_result(root / 'result.json', result)
    report = results.check(result, timeline_rows)
    write_json(root / 'check.json', report)
    due = times['rejoin_due']
    cards = reclaim['cards'] if isinstance(reclaim, dict) else None  # the supervisor's gpu_reclaim record (real mode)
    events = {
        'schema_version': 'events_actual_v1', 'mode': config['mode'], 'launch': config['launch'], 'round': 0, 'commit': config['commit'],
        'sha256': {'requests': requests_io.sha256_file(config['serving']['requests']), 'plan': requests_io.sha256_file(config['serving']['plan']),
                   'workload': requests_io.sha256_file(config['serving']['workload']), 'profile': requests_io.sha256_file(config['profile_file'])},
        'plan': {'fail': fail_spec, 'rejoin': rejoin_spec, 'replica': 'EDP0', 'coords': sorted(((r['ep'], r['etp']) for r in maps['0/0'] if r['slot'] in failed_slots(maps['0/0']))),
                 'rejoin_due_s': due - times['serving_start'], 'dependency': 'shrink_done'},
        'actual_trigger': {**trigger, 't_controller_s': times['fail'] - times['serving_start']},
        'memory_snapshots': {phase: [{'slot': r['slot'], 'layout': r['layout'], 'memory': r['memory'], 'sampled_at_step': r['step'],
                                      'source': 'phase report of the latest committed step'} for r in values]
                             for phase, values in reports.items()},
        'recovery_s': {k: v - times['serving_start'] for k, v in times.items() if isinstance(v, float)},
        'state_mapping': {'rank_maps': maps, 'initial_pids': {str(s): r['pid'] for s, r in initial.items()}},
        'delays': {'rejoin_delay_s': times['spawn'] - due, 'rejoin_join_s': times['world16_ready'] - times['rejoin_request'],
                   'reclaim_wait_s': None if reclaim is None else max(0.0, times['reclaim_done'] - times['rejoin_request']),
                   'gpu_reclaim_s': None if cards is None else {uuid: card['gpu_reclaim_s'] for uuid, card in cards.items()},
                   'abort_s': {phase: times[f'aborted_{phase}'] - times[f'transition/0/{phase}'] for phase in (1, 2)},
                   'rebuild_s': {phase: times[f'build/0/{phase}'] - times[f'aborted_{phase}'] for phase in (1, 2)},
                   'edp1_pause_s': max((p['edp1_rejoin_pause_s'] for p in pauses), default=None),
                   'null_reason': None if reclaim is not None else 'emulate mode has no GPU reclaim or EDP1 abort pause'},
        'controller': {'queue': len(controller.queue), 'done': len(controller.done)}}
    write_json(root / 'events_actual.json', events)
    return {'result': result, 'check': report, 'events': events}
