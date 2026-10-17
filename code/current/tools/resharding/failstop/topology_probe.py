"""Step 2 control gates; process ownership and reclaim reuse the Step 1 supervisor."""

import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from common import NODES, failed_slots, require, topology, write_json
from supervisor import Supervisor

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from topology_fixture import toy_local_orders

from megatron.core.failstop import FaultBackend, FaultSession, replica_group_specs


class EmulateSupervisor(Supervisor):
    def rejoin_preflight(self, baselines):
        # Released tensors do not destroy a live worker's CUDA context or its
        # prebuilt groups. All eight registered contexts must still be present.
        self.inspect('rejoin_pre_spawn', {p.pid: r['gpu_uuid'] for p, r in self.workers.values()})

    def kill_and_reclaim(self, baselines, allowed):
        self.transition('running', 'reclaimed')
        self.bus.put('emulate_release/0', True)
        self.wait([f'released/0/{s}' for s in failed_slots(self.config['topology'])])
        self.bus.put('emulate_released', True)

    def start_workers(self, slots, epoch):
        if epoch == 0:
            return super().start_workers(slots, epoch)
        require(self.state == 'reclaimed', 'emulate rejoin before release')
        # Original processes construct empty targets after observing spawn.
        require(all(self.workers[s][0].poll() is None for s in slots), 'emulate PID died')


def control(config, bus, journal, check):
    root = Path(config['output'])

    def wait(keys, timeout=None):
        return bus.wait(keys, timeout, check)

    def publish(key, value=True):
        journal.event(key, payload=value)
        bus.put(key, value)

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
        for key in ('commit', 'launch', 'mode', 'rounds', 'port', 'shared_gpu_mode', 'quick_mode', 'correctness_only'):
            require(report['config'][key] == config[key], f'node config mismatch: {key}')
    rows = topology(dict(zip(NODES, [r['gpus'] for r in preflight])))
    victims = failed_slots(rows)
    publish('topology', rows)
    publish('start')
    wait([f'verified/0/{n}' for n in NODES])
    initial = dict(zip(range(16), wait([f'cuda/0/{s}' for s in range(16)])))
    publish('cuda_verified/0')
    backend = FaultBackend(config['mode'], fail=lambda *_: None, rejoin=lambda *_: None)
    session = FaultSession(rows, config['launch'], backend, local_orders=toy_local_orders(rows))
    reports, maps = {}, {}

    def validate(round_id, phase):
        members = session.current_world()['membership']
        session.adopt_group_order([{'replica': replica, 'field': field, 'actual': sorted(ranks)}
                                   for replica, field, ranks in replica_group_specs(session.replicas())])
        values = wait([f'phase/{round_id}/{phase}/{s}' for s in members])
        mapping = sorted((r for r in session.rank_map() if r['active']), key=lambda r: r['new_rank'])
        expected_ids = [[r['slot'], initial[r['slot']]['pid']] for r in mapping]
        for s, report in zip(members, values):
            require(report['slot'] == s and report['identities'] == expected_ids, 'actual membership/PID mismatch')
            require(report['rank_map'] == session.rank_map(), 'actual group mapping mismatch')
            require(report['checks_passed'], 'worker correctness check failed')
            if config['mode'] == 'real' and phase == 2 and s in victims:
                require(report['empty_receiver'] and not report['used_as_source'], 'replacement was not an empty receiver')
        reports[f'{round_id}/{phase}'] = values
        maps[f'{round_id}/{phase}'] = session.rank_map()
        write_json(root / 'rank_map.json', maps)
        write_json(root / 'topology_checks.json', reports)

    for round_id in range(config['rounds']):
        validate(round_id, 0)
        if round_id == 0:
            publish('kill')
            if config['mode'] == 'real':
                wait(['kill_sent'])
                killed = wait(['kill_exited'])[0]
                require({r['slot'] for r in killed} == set(victims) and all(
                    r['pid'] == initial[r['slot']]['pid'] and r['exit']['returncode'] == -signal.SIGKILL
                    for r in killed), 'fault exit identity mismatch')
            else:
                wait(['emulate_released'])
        else:
            publish(f'emulate_release/{round_id}')
        if config['mode'] == 'emulate':
            released = wait([f'released/{round_id}/{s}' for s in victims])
            require(all(r['src_none'] and r['dst_none'] and r['data_bytes'] == r['marker_bytes'] == 0
                        for r in released), 'failed GPU data path not isolated')
        wait([f'healthy_progress/{round_id}/{r["slot"]}' for r in rows if r['replica'] == 'EDP1'])
        event = session.event('fail')
        session.fail(victims, event)
        publish(f'transition/{round_id}/1', asdict(event))
        members = session.current_world()['membership']
        wait([f'aborted/{round_id}/1/{s}' for s in members])
        publish(f'build/{round_id}/1')
        validate(round_id, 1)
        session.complete()
        if round_id == 0:
            if config['mode'] == 'real':
                publish('rejoin_request')
                wait(['reclaimed'], config['reclaim_timeout_s'])
            publish('spawn')
            wait([f'verified/2/{n}' for n in NODES], config['startup_timeout_s'])
            returned = dict(zip(victims, wait([f'cuda/2/{s}' for s in victims])))
            require(all(returned[s]['empty_receiver'] for s in victims), 'rejoin target prefilled')
            require(all((returned[s]['pid'] != initial[s]['pid']) == (config['mode'] == 'real')
                        for s in victims), 'incorrect rejoin PID semantics')
            initial.update(returned)
            publish('cuda_verified/2')
        else:
            publish(f'emulate_rejoin/{round_id}')
            wait([f'empty/{round_id}/{s}' for s in victims])
        # Each replica progresses on its own TP/EP groups, without a world barrier.
        wait([f'serving/{round_id}/{s}' for s in members])
        event = session.event('rejoin')
        session.rejoin(victims, event)
        publish(f'transition/{round_id}/2', asdict(event))
        wait([f'aborted/{round_id}/2/{s}' for s in members])
        publish(f'build/{round_id}/2')
        validate(round_id, 2)
        session.complete()
        if round_id + 1 < config['rounds']:
            session.next_round()
            publish(f'next/{round_id}')
    publish('finish')
    wait([f'done/{s}' for s in range(16)])
    wait([f'clean/{n}' for n in NODES], config['reclaim_timeout_s'] + config['stage_timeout_s'])
    return {'status': 'passed', 'mode': config['mode'], 'rounds': config['rounds'],
            'epochs': reports, 'rank_maps': maps, 'shared_gpu_mode': config['shared_gpu_mode'],
            'quick_mode': config['quick_mode'], 'commit': config['commit']}
