"""Fail-stop WeaveTP --mode kill with 16 concurrent CPU actors (the Step 6 B3 simulation of test_serving_protocol). The
simulated supervisor SIGKILLs the victims: a killed actor stops at its next bus wait, so a collective that still needs one
times out after 8 s. The degraded world is the prebuilt one: between the start of serving and the rejoin no process
group is created, no world is initialized or aborted (only the groups that hold a dead rank are), nothing is planned
or built; EDP1 decodes inside the shrink waves. The rejoin aborts and rebuilds the world as real does."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_failstop_e2e
import model_worker
import serving_worker
import test_serving_protocol as serving_test
import torch

from tools.resharding.e2e import remote_control

VICTIMS = (8, 9, 12, 13)
LIVE = tuple(s for s in range(16) if s not in VICTIMS)  # kill keeps the initial world's ranks: rank = slot


class Recording(serving_test.Simulation):
    """The serving simulation with every world, group, abort and collective call and every switch ack recorded as
    (event, (slot, epoch), stage, detail); stage: 'start' until every rank serves, 'serving' until the rejoin's
    transition, 'rejoin', 'finish'."""

    log = None

    def stage(self):
        if self.has('finish'):
            return 'finish'
        if self.has('transition/0/2'):
            return 'rejoin'
        return 'serving' if all(self.has(f'phase/0/0/{s}') for s in range(16)) else 'start'

    def note(self, event, detail):
        self.log.append((event, self.local.identity, self.stage(), detail))

    def init(self, backend, **kwargs):
        self.note('init', backend)
        super().init(backend, **kwargs)

    def group(self, ranks, **kwargs):
        self.note('new_group', tuple(sorted(ranks)))
        return super().group(ranks, **kwargs)

    def abort(self, group=None):
        self.note('abort', None if group is None else group.members)
        super().abort(group)

    def collective(self, group, value, kind):
        self.note('collective', (kind, group.members))
        return super().collective(group, value, kind)

    def put(self, key, value):
        if key.startswith(('switch_ack/', 'phase/0/1/')):  # only the workers ack and report
            self.note('ack' if key.startswith('switch_ack/') else 'degraded_report', key.split('/')[1])
        super().put(key, value)


def recorded(sim):
    """Planner calls, model builds, model-parallel state drops, requeues, and the groups holding a dead rank left after
    abort_dead_groups."""
    plan, build, destroyed = model_worker.PLANNERS['weavetp'], serving_test.FakeRuntime.build, serving_test.FakeRuntime.world_destroyed
    requeue, abort_dead = remote_control.RemoteControl.requeue, serving_worker.ServingWorker.abort_dead_groups

    def note(event, detail, function):
        def wrapped(*args, **kwargs):
            sim.note(event, detail(*args, **kwargs))
            return function(*args, **kwargs)
        return wrapped

    def after_abort(worker):
        abort_dead(worker)
        sim.note('dead_left', [g.members for g in sim.local.registry if g is not sim.local.default and set(g.members) & set(VICTIMS)])
    return [patch.dict(model_worker.PLANNERS, {'weavetp': note('plan', lambda *a, **k: None, plan)}),
            patch.object(serving_test.FakeRuntime, 'build', note('build', lambda self, tp, groups, load: (tp, load), build)),
            patch.object(serving_test.FakeRuntime, 'world_destroyed', note('world_destroyed', lambda self: None, destroyed)),
            patch.object(remote_control.RemoteControl, 'requeue', note('requeue', lambda self, rows, step: len(rows), requeue)),
            patch.object(serving_worker.ServingWorker, 'abort_dead_groups', after_abort),
            # the shrink's growth takes the whole open-block zone: an online-style budget would leave EDP1 no step in the
            # waves, kill's EDP1 still decodes in every one (no KV moves)
            patch.object(serving_worker.ServingSwitcher, 'growth', staticmethod(lambda serving: {'shrink': serving['max_delta_steps'], 'expand': 1}))]


class KillProtocolTests(unittest.TestCase):
    def launch(self, run_mode):
        log = []
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            simulation = Recording('kill', root, 1, {'method': 'weavetp', 'run_mode': run_mode})
            simulation.log = log
            for patcher in recorded(simulation):
                stack.enter_context(patcher)
            torch.use_deterministic_algorithms(run_mode == 'correctness')
            try:
                result = simulation.run()
            finally:
                torch.use_deterministic_algorithms(False)
            result_json, events = (json.loads((root / name).read_text()) for name in ('result.json', 'events_actual.json'))
            plans = {p.stem.removeprefix('source_plan_0_'): json.loads(p.read_text()) for p in root.glob('source_plan_*.json')}
            self.assertEqual(analyze_failstop_e2e.main([str(root), '--out', str(root / 'analysis')]), 0)
            self.assertEqual(json.loads((root / 'analysis' / 'rejected.json').read_text()), [])
        self.check(log, result, result_json, events, plans, run_mode)

    def check(self, log, result, result_json, events, plans, run_mode):
        serving, rejoin = ([e for e in log if e[2] == stage] for stage in ('serving', 'rejoin'))
        # a) from the start of serving to the rejoin: no group, no world, no plan, no build; the only aborts are of the
        # groups that hold a dead rank, on every live rank, until none is left (never the default world)
        self.assertEqual([e for e in serving if e[0] in ('new_group', 'init', 'plan', 'build')], [])
        aborts = [e for e in serving if e[0] == 'abort']
        self.assertTrue(aborts and all(e[3] is not None and set(e[3]) & set(VICTIMS) for e in aborts), aborts)
        self.assertEqual({e[1] for e in aborts}, {(s, 0) for s in LIVE})
        for identity in [(s, 0) for s in LIVE]:  # once its degraded world served (after its phase-1 report: shrink_done)
            mine = [e[0] for e in serving if e[1] == identity]
            self.assertLess(mine.index('degraded_report'), mine.index('abort'))
        self.assertEqual(sorted((e[1], e[3]) for e in log if e[0] == 'dead_left'), [((s, 0), []) for s in LIVE])
        # with them the initial world's model-parallel state goes, once per live rank (real drops it at the fault)
        self.assertEqual(sorted(e[1] for e in serving if e[0] == 'world_destroyed'), [(s, 0) for s in LIVE])
        # the rejoin: each live rank aborts the world and initializes the new one once; each replacement only initializes
        for identity in [(s, 0) for s in LIVE] + [(s, 2) for s in VICTIMS]:
            self.assertEqual([(e[0], e[3]) for e in rejoin if e[1] == identity and e[0] in ('init', 'abort')],
                             ([('abort', None)] if identity[1] == 0 else []) + [('init', 'nccl')])
        # the gate kills the victims once every rank acked the shrink: no victim takes part in a collective after its ack
        # (the peak eviction's broadcast over the EDP0 domain comes before it), nor in any group call after the start
        for slot in VICTIMS:
            mine = [e for e in log if e[1] == (slot, 0)]
            ack = mine.index(next(e for e in mine if e[0] == 'ack'))
            self.assertEqual([e for e in mine[ack:] if e[0] == 'collective'], [])
            self.assertEqual([e for e in mine if e[2] != 'start' and e[0] in ('new_group', 'init', 'abort', 'plan', 'build')], [])
        # the shrink's TP2 targets were built before serving (released, reallocated at the shrink): 16 TP4 loads and
        # 4 TP2 builds at the start (the TP4 loads collective-free: load_local), the 8 TP4 expand targets at the rejoin
        self.assertEqual(sorted(e[3] for e in log if e[0] == 'build' and e[2] == 'start'), [(2, False)] * 4 + [(4, 'local')] * 16)
        self.assertEqual([e[3] for e in rejoin if e[0] == 'build'], [(4, False)] * 8)
        # b) the shrink waves order on the barrier of the twelve live ranks only (a dead member would hang them)
        barriers = [e for e in serving if e[0] == 'collective' and e[3][0] == 'barrier']
        self.assertEqual(({e[3][1] for e in barriers}, {e[1] for e in barriers}), ({LIVE}, {(s, 0) for s in LIVE}))
        # c) EDP1 decodes in every shrink wave (max_overlap_steps = 1; no KV moves, so no open-block budget caps it);
        # EDP0's requests in flight are requeued and replay from prompt + history
        waves = 1 + max(t['wave'] for t in plans['shrink']['tasks'])
        edp1 = {s['direction']: s for s in result_json['switches'] if s['domain'] == 'EDP1'}
        self.assertEqual(edp1['shrink']['overlap_steps'], waves)
        self.assertGreater(sum(e[3] for e in serving if e[0] == 'requeue'), 0)
        metrics = result['metrics']
        self.assertEqual((metrics['requests_completed'], metrics['check_passed']), (12, True))
        self.assertGreater(metrics['replay_prefill_tokens'], 0)
        shrink = {s['direction']: s for s in result_json['failstop']['edp0_switches']}['shrink']
        self.assertEqual((shrink['kv_prefix_bytes'], shrink['kv_delta_bytes']), (0, 0))
        # d) only the shrink was planned before serving, in the degraded world on the frozen profile: no source is a
        # victim's; expand and delta are planned at the switch (12 planner calls at the start, 2 x 16 at the rejoin)
        self.assertEqual(sorted(plans), ['delta', 'expand', 'shrink'])
        self.assertEqual((plans['shrink']['planned'], plans['shrink']['plan_profile_sha256']), ('before serving', plans['shrink']['profile_sha256']))
        sources = {t['source']['rank'] for t in plans['shrink']['tasks']}
        self.assertTrue(sources and not sources & set(VICTIMS), sources)
        self.assertTrue(all('planned' not in plans[label] for label in ('expand', 'delta')))
        self.assertEqual([len([e for e in log if e[0] == 'plan' and e[2] == stage]) for stage in ('start', 'rejoin')], [12, 32])
        # e) the rejoin rebuilt the world: EDP1's pause is recorded, the expand KV equals the mapped snapshots (correctness),
        # the gate saw new PIDs and empty receivers (else it fails); the dead groups' abort time is in events_actual
        self.assertIsNotNone(metrics['edp1_rejoin_pause_s'])
        delays = events['delays']
        self.assertTrue(delays['edp1_pause_s'] is not None and delays['dead_group_abort_s'] is not None and delays['null_reason'] is None, delays)
        self.assertEqual(metrics['kv_check_passed'], True if run_mode == 'correctness' else None)
        self.assertEqual(result_json['failstop']['reference_audit'], 'passed' if run_mode == 'correctness' else 'skipped')
        self.assertEqual(result['mode'], 'kill')

    def test_kill_cycle_correctness(self):
        self.launch('correctness')

    def test_kill_cycle_timing(self):
        self.launch('timing')


if __name__ == '__main__':
    unittest.main()
