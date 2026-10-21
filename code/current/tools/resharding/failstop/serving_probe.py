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

import failslow_probe
from async_state import audit_sequence
from common import NODES, failed_slots, require, topology, write_json
from model_shards import model_local_orders

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from async_topology import AsyncSession as FaultSession

from megatron.core.failstop import FaultBackend, replica_group_specs
from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e import remote_control, requests_io, results
from tools.resharding.e2e.failstop_adapter import rank_map_from_rows
from tools.resharding.e2e.methods import describe

PROBE_KEYS = ('method', 'profile_file', 'reference_root', 'deterministic_mode', 'kv_heads', 'memory_margin_bytes', 'memory_limit_bytes',
              'serving', 'failslow')
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
    failstop, failslow = plan['version'] == 'v2-failstop', config['scenario'] == 'failslow'
    require(failstop == (config['scenario'] == 'failstop'), 'the plan does not match the configured scenario')
    if failstop:
        fail_count, rejoin_seconds, fail_spec, rejoin_spec = anchors(plan, len(instances))
    else:
        require(not plan['switches'], 'a serving run takes no scale switches')
        fail_spec = rejoin_spec = None
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
        require_same_config(config, report['config'])
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
    fixed = serving['capacity_tokens']

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
    trigger, reclaim, drained_before_rejoin = None, None, False
    if failstop:
        trigger, reclaim, drained_before_rejoin = fault_cycle(config, bus, journal, service, session, times, validate, wait, wait_until,
                                                              publish, fail_count, rejoin_seconds, victims, members_of(session),
                                                              initial, rows, pauses, len(instances))
    elif failslow:  # detection is the fault, the end of the injection the rejoin; the deadline closes the service
        record = failslow_probe.run(config, bus, journal, service, times, publish, wait, wait_until, lambda fire, rejoin: fault_cycle(
            config, bus, journal, service, session, times, validate, wait, wait_until, publish, None, 0.0, victims, members_of(session),
            initial, rows, pauses, len(instances), fire=fire, rejoin=rejoin))
    # ---- drain ----------------------------------------------------------------------------------------------------
    wait_until(lambda: all(service.finished.values()), config['launch_timeout_s'], 'every request to complete')
    keys = [(domain, replica) for domain, replicas in DOMAINS.items() for replica in replicas]
    summaries = dict(zip(keys, wait([f'summary/{d}/{r}' for d, r in keys])))
    times['serving_done'] = time.monotonic()
    service.stop()
    thread.join(timeout=5)
    require(service.error is None, f'control service failed: {service.error!r}')
    require(failslow or (controller.idle() and len(controller.done) == len(instances)), 'requests left unfinished')
    edp0_slots = [s for s in range(16) if rows[s]['replica'] == 'EDP0']
    loaded = restart_loaded(wait, edp0_slots, victims, times['serving_start']) if config['method'] == 'restart' else None
    if failslow:  # before finish: the workers are still up and their hash keys are all posted
        outputs = failslow_probe.write_outputs(root, config, merge_summaries(summaries, instances, times, admitted_only=True), summaries,
                                               times, record, edp0_slots, loaded, wait)
    publish('finish')
    wait([f'done/{s}' for s in range(16)])
    wait([f'clean/{n}' for n in NODES], config['reclaim_timeout_s'] + config['stage_timeout_s'])
    if not failslow:
        outputs = write_outputs(root, config, instances, workload, summaries, times, trigger, fail_spec, rejoin_spec, reports, maps,
                                initial, pauses, reclaim, controller, drained_before_rejoin, loaded)
    metrics = {'requests_completed': len(controller.done), 'check_passed': outputs['check']['passed'], 'method': config['method'],
               'scenario': config['scenario'], 'run_mode': config['run_mode'], 'serving_wall_s': times['serving_done'] - times['serving_start'],
               'stats_valid': not drained_before_rejoin}
    if failslow:
        metrics.update({k: outputs['result'][k] for k in ('requests_admitted', 'requests_unfinished', 'deadline_overrun_s', 'events_s')},
                       failslow_check=outputs['check'], failslow_result=str(root / 'failslow_result.json'))
    if failstop:
        metrics.update({'edp1_rejoin_pause_s': max((p['edp1_rejoin_pause_s'] for p in pauses), default=None),
                        'replay_prefill_tokens': outputs['result']['failstop']['replay_prefill_tokens'],
                        'kv_check_passed': outputs['result']['failstop']['expand_kv_check_passed'],
                        'shrink_wall_s': times['shrink_done'] - times['fail'], 'expand_wall_s': times['expand_done'] - times['rejoin_request'],
                        'edp0_switches': outputs['result']['failstop']['edp0_switches']})
    return {'status': 'passed', 'mode': mode, 'rounds': 1, 'epochs': reports, 'rank_maps': maps, 'shared_gpu_mode': config['shared_gpu_mode'],
            'quick_mode': config['quick_mode'], 'commit': config['commit'], 'metrics': metrics, 'rejoin_pauses': pauses,
            'events_actual': str(root / 'events_actual.json')}


def require_same_config(config, peer):
    """The gate refuses a node whose launch configuration differs from its own in a key both nodes must share."""
    for key in ('commit', 'launch', 'mode', 'rounds', 'port', 'shared_gpu_mode', 'quick_mode', 'correctness_only', 'probe') + PROBE_KEYS:
        require(peer[key] == config[key], f'node config mismatch: {key}')


def restart_loaded(wait, edp0_slots, victims, origin):
    """restart: when each reload finished (the slowest rank, seconds since serving_start): shrink = TP2 on the survivors,
    expand = TP4 on all eight EDP0 ranks."""
    survivors = [s for s in edp0_slots if s not in victims]
    return {direction: max(wait([f'restart_loaded/{direction}/{s}' for s in slots])) - origin
            for direction, slots in (('shrink', survivors), ('expand', edp0_slots))}


def members_of(session):
    return lambda: session.current_world()['membership']


def fault_cycle(config, bus, journal, service, session, times, validate, wait, wait_until, publish, fail_count, rejoin_seconds, victims,
                members_of, initial, rows, pauses, total, fire=None, rejoin=None):
    """fail anchor -> shrink -> kill/release -> 12-world -> rejoin due -> reclaim/spawn -> expand -> 16-world (every method).
    fire() / rejoin() replace the completed_count anchor and the rejoin delay (fail-slow: detection, end of the injection)."""
    mode = config['mode']
    # ---- fault: the completed_count anchor --------------------------------------------------------------------
    if fire is None:
        wait_until(lambda: service.completed() >= fail_count, config['launch_timeout_s'], f'{fail_count} completions')
    times['fail'] = time.monotonic()
    trigger = fire() if fire is not None else {
        'completed_count': service.completed(), 'threshold': fail_count, 'overshoot': service.completed() - fail_count,
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
    members = members_of()
    wait([f'aborted/0/1/{s}' for s in members])
    times['aborted_1'] = time.monotonic()
    publish('build/0/1')
    times['shrink_done'] = validate(1)
    session.complete()
    # ---- rejoin: due time, reclaim, spawn, expand -------------------------------------------------------------
    due = times['fail'] + rejoin_seconds if rejoin is None else rejoin()
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
    drained_before_rejoin = service.completed() == total
    service.hold.clear()
    return trigger, reclaim, drained_before_rejoin


def write_outputs(root, config, instances, workload, summaries, times, trigger, fail_spec, rejoin_spec, reports, maps, initial, pauses,
                  reclaim, controller, drained_before_rejoin, loaded):
    """result.json / check.json (W4) and events_actual.json (06_Step6 §4) from the domain summaries and the gate's times;
    `loaded` = restart's reload times (restart_loaded), None for the migrating methods.

    Every time in result.json is in seconds since serving_start on the gate's clock (shift_to_gate)."""
    merged = merge_summaries(summaries, instances, times)
    clocks, rows, leaders, switches, gate_s = (merged[k] for k in ('clocks', 'rows', 'leaders', 'switches', 'gate_s'))
    timeline_rows = [r for s in summaries.values() for r in s['timeline']]  # one timeline per dense replica
    phases = {}
    for row in timeline_rows:
        key = row['phase'] if row['kind'] == 'step' else 'switching'
        phases[key] = phases.get(key, 0.0) + (row['step_time'] if row['kind'] == 'step' else row['seconds'])
    edp0 = leaders['EDP0']
    method = describe(config['method'])
    duration = sum(phases.values())
    events_s = {**edp0['failstop_events'], **gate_s}
    kv_checks = [c for s in summaries.values() for c in s['kv_checks']]  # every EDP0 rank that received KV
    timing = config['run_mode'] == 'timing'
    failstop = None if fail_spec is None else {
                'events_s': events_s, 'replay_prefill_tokens': edp0['replayed_tokens'],
                'peak_eviction': [e for s in switches for e in s['evictions']],
                'expand_kv_check_passed': None if timing or loaded else bool(kv_checks) and all(c['passed'] for c in kv_checks),
                'kv_check_note': 'skipped: timing run' if timing else 'restart: no KV moves, EDP0 requests replay' if loaded else
                                 "head mapping of the survivors' saved TP2 snapshots",
                'restart_loaded_s': loaded,
                'tfs_breakdown': {'fault_to_transition_s': times['transition/0/1'] - times['fail'],
                                  'world_rebuild_s': times['build/0/1'] - times['transition/0/1'],
                                  'weights_and_cutover_s': times['shrink_done'] - times['build/0/1'],
                                  'shrink_wall_s': times['shrink_done'] - times['fail'],
                                  'expand_wall_s': times['expand_done'] - times['rejoin_request']},
                'advantage_sources': method['advantage_sources'],
                'method': {k: method[k] for k in ('name', 'source_policy', 'kv_semantics', 'kv_semantics_status', 'implementation_ref')},
                'edp0_switches': [{'direction': s['direction'], 'overlap_steps': s['overlap_steps'], 'kv_delta_bytes': s['bytes']['kv_delta'],
                                   'kv_prefix_bytes': s['bytes']['kv_prefix'], 'weights_bytes': s['bytes']['weights']}
                                  for s in switches if s['domain'] == 'EDP0'],
                'serving_wall_s': times['serving_done'] - times['serving_start'],
                'duration_note': 'totals.duration_s accumulates the step time of every dense replica timeline; serving_wall_s is wall time',
                'stats_valid': not drained_before_rejoin,
                'stats_invalid_reason': 'every request completed before the rejoin; recovery throughput is not measurable' if drained_before_rejoin else None}
    result = results.build_result(method=config['method'], workload_id=workload['workload_id'], segments_run=sorted({i.segment for i in instances}),
                                  request_rows=rows, switches=switches, phase_durations_s=phases,
                                  attention_impl=edp0['attention_impl'],
                                  config={'run_mode': config['run_mode'], 'experiment': config['scenario'], 'method': config['method'],
                                          'backend': 'block', 'control': 'remote', 'mode': config['mode'], 'validate_cutover': not timing,
                                          'timing_valid': config['timing_valid'], 'shared_gpu_mode': config['shared_gpu_mode'],
                                          'decode_policy': config['serving']['decode_policy'],
                                          'b_max': config['serving']['b_max'], 'commit': config['commit'], 'launch': config['launch']},
                                  duration_s=duration, logical_steps=sum(s['steps'] for s in summaries.values()), failstop=failstop)
    result['clocks'] = clocks
    result['gpu_sharing'] = 'shared' if config['shared_gpu_mode'] else 'exclusive'
    result['serving_wall_s'] = times['serving_done'] - times['serving_start']
    result['time_base'] = 'seconds since serving_start on the SL3060 gate clock (worker records shifted by clocks[*].shift_to_gate_s)'
    results.write_result(root / 'result.json', result)
    report = results.check(result, timeline_rows)
    write_json(root / 'check.json', report)
    events = {
        'schema_version': 'events_actual_v1', 'mode': config['mode'], 'scenario': config['scenario'], 'method': config['method'],
        'run_mode': config['run_mode'], 'launch': config['launch'], 'round': 0, 'commit': config['commit'],
        'sha256': {'requests': requests_io.sha256_file(config['serving']['requests']), 'plan': requests_io.sha256_file(config['serving']['plan']),
                   'workload': requests_io.sha256_file(config['serving']['workload']), 'profile': requests_io.sha256_file(config['profile_file'])},
        'memory_snapshots': {phase: [{'slot': r['slot'], 'layout': r['layout'], 'memory': r['memory'], 'sampled_at_step': r['step'],
                                      'source': 'phase report of the latest committed step'} for r in values]
                             for phase, values in reports.items()},
        'recovery_s': gate_s, 'clocks': clocks,
        'state_mapping': {'rank_maps': maps, 'initial_pids': {str(s): r['pid'] for s, r in initial.items()}},
        'controller': {'queue': len(controller.queue), 'done': len(controller.done)}}
    if fail_spec is not None:
        due = times['rejoin_due']
        cards = reclaim['cards'] if isinstance(reclaim, dict) else None  # the supervisor's gpu_reclaim record (real mode)
        events.update({
            'plan': {'fail': fail_spec, 'rejoin': rejoin_spec, 'replica': 'EDP0',
                     'coords': sorted(((r['ep'], r['etp']) for r in maps['0/0'] if r['slot'] in failed_slots(maps['0/0']))),
                     'rejoin_due_s': due - times['serving_start'], 'dependency': 'shrink_done'},
            'actual_trigger': {**trigger, 't_controller_s': times['fail'] - times['serving_start']},
            'worker_events_s': edp0['failstop_events'],
            'delays': {'rejoin_delay_s': times['spawn'] - due, 'rejoin_join_s': times['world16_ready'] - times['rejoin_request'],
                       'reclaim_wait_s': None if reclaim is None else max(0.0, times['reclaim_done'] - times['rejoin_request']),
                       'gpu_reclaim_s': None if cards is None else {uuid: card['gpu_reclaim_s'] for uuid, card in cards.items()},
                       'abort_s': {phase: times[f'aborted_{phase}'] - times[f'transition/0/{phase}'] for phase in (1, 2)},
                       'rebuild_s': {phase: times[f'build/0/{phase}'] - times[f'aborted_{phase}'] for phase in (1, 2)},
                       'edp1_pause_s': max((p['edp1_rejoin_pause_s'] for p in pauses), default=None),
                       'null_reason': None if reclaim is not None else 'emulate mode has no GPU reclaim or EDP1 abort pause'}})
    write_json(root / 'events_actual.json', events)
    return {'result': result, 'check': report, 'events': events}


def merge_summaries(summaries, instances, times, admitted_only=False):
    """The domain summaries on the gate clock, merged: request rows (only admitted requests for a deadline-cut run),
    each domain's leader summary, the leaders' switches and the gate times since serving_start."""
    clocks = shift_to_gate(summaries, times)
    logs = results.merge_request_logs(summary['request_log'] for summary in summaries.values())
    leaders = {domain: summaries[domain, min(replicas)] for domain, replicas in DOMAINS.items()}
    return {'clocks': clocks, 'leaders': leaders,
            'rows': [results.request_row(i, logs[i.seq]) for i in instances if not admitted_only or i.seq in logs],
            'switches': [dict(s, domain=domain) for domain, summary in leaders.items() for s in summary['switches']],
            'gate_s': {k: v - times['serving_start'] for k, v in times.items() if isinstance(v, float)}}


def shift_to_gate(summaries, times):
    """Worker records (run clock, per-node origin) shifted in place by their summary's clock calibration."""
    clocks = {}
    for (domain, replica), summary in summaries.items():
        clock = summary['clock']
        shift = clock['origin_local'] + clock['offset_to_gate'] - times['serving_start']
        clocks[f'{domain}/{replica}'] = {**clock, 'shift_to_gate_s': shift, 'gate_origin_monotonic': times['serving_start']}
        summary['request_log'] = {seq: shifted_log(entry, shift) for seq, entry in summary['request_log'].items()}
        summary['failstop_events'] = {k: v + shift for k, v in summary['failstop_events'].items()}
        for switch in summary['switches']:
            switch['milestones'] = {k: v + shift for k, v in switch['milestones'].items()}
    return clocks


def shifted_log(entry, shift):
    """A worker's request_log entry on the gate time base."""
    moved = dict(entry)
    moved['admits'] = [{**a, 't': a['t'] + shift} for a in entry['admits']]
    moved['t_tokens'] = [t + shift for t in entry['t_tokens']]
    for key in ('t_first', 't_done'):
        if key in entry:
            moved[key] = entry[key] + shift
    return moved
