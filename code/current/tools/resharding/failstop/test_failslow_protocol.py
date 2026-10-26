"""Fail-slow serving with 16 concurrent CPU actors (the Step 6 B3 simulation of test_serving_protocol): the gate's
injection timeline, the wall-clock deadline and the five methods; no reference audit (reference_audit skipped). WeaveTP
switches online (common.online_switch): nothing is built, planned or grouped after startup, the KV migrates. The
co-runners are the simulated supervisor's handshake only; nothing here runs on a GPU."""

import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import model_worker
import serving_worker
import test_serving_protocol as serving_test
import torch
from serving_config import failslow_settings

from tools.resharding.e2e import failslow_metrics, remote_control
from tools.resharding.e2e.toy_inputs import write_toy_inputs

TIMELINE = {'warmup_s': 0.6, 'slow_s': 2.0, 'tail_s': 2.0, 'detect_s': 0.3}
SHRINK_KV, EXPAND_KV = "head mapping of the TP4 sources' saved block snapshots", "head mapping of the survivors' saved TP2 block snapshots"


def pool(directory, count, failstop):
    return write_toy_inputs(directory, count=4000, failstop=False)  # far more than the toy model finishes before the deadline


class Recording(serving_test.Simulation):
    """The serving simulation with what the online switch must not do after startup recorded per actor: process groups,
    model builds, planner calls, requeues, and the order of world transitions and transfers; plus the standby log."""

    log = None

    def group(self, ranks, **kwargs):
        if all(self.has(f'phase/0/0/{s}') for s in range(16)):  # serving has started on every rank
            self.log.append(('new_group', None, tuple(sorted(ranks))))
        return super().group(ranks, **kwargs)


def recorded(log, sim):
    """Patches that append (event, slot, detail) to `log` from inside the actors."""
    slot = lambda: sim.local.identity[0]  # noqa: E731
    build, plan = serving_test.FakeRuntime.build, model_worker.PLANNERS['weavetp']
    release, reallocate = serving_test.FakeRuntime.release, serving_test.FakeRuntime.reallocate
    transition, transfer, requeue = serving_worker.ServingWorker.world_transition, model_worker.Worker.transfer, remote_control.RemoteControl.requeue

    def note(event, detail, function):
        def wrapped(*args, **kwargs):
            log.append((event, slot(), detail(*args, **kwargs)))
            return function(*args, **kwargs)
        return wrapped
    return [patch.object(serving_test.FakeRuntime, 'build', note('build', lambda self, tp, groups, load: (tp, load), build)),
            patch.dict(model_worker.PLANNERS, {'weavetp': note('plan', lambda *a, **k: None, plan)}),
            patch.object(serving_test.FakeRuntime, 'release', staticmethod(note('release', lambda model, *a: model.tp, release))),
            patch.object(serving_test.FakeRuntime, 'reallocate', staticmethod(note('reallocate', lambda model: model.tp, reallocate))),
            patch.object(serving_worker.ServingWorker, 'world_transition', note('transition', lambda self, phase, round_id: phase, transition)),
            patch.object(model_worker.Worker, 'transfer', note('transfer', lambda self, round_id, phase, label, *a, **k: label, transfer)),
            patch.object(remote_control.RemoteControl, 'requeue', note('requeue', lambda self, rows, step: len(rows), requeue))]


class FailslowProtocolTests(unittest.TestCase):
    def launch(self, method, run_mode='correctness', log=None):
        settings = failslow_settings(method=method, level=None if method == 'nofault' else 'W', fault_gpus=[0, 1, 4, 5],
                                     mode='emulate', **TIMELINE)
        extra = {'scenario': 'failslow', 'method': 'ignore' if method == 'nofault' else method, 'failslow': settings, 'run_mode': run_mode}
        with tempfile.TemporaryDirectory() as directory, patch.object(serving_test, 'write_toy_inputs', pool), contextlib.ExitStack() as stack:
            root = Path(directory)
            simulation = Recording('emulate', root, 1, extra)
            simulation.log = [] if log is None else log
            if log is not None:
                for patcher in recorded(log, simulation):
                    stack.enter_context(patcher)
            torch.use_deterministic_algorithms(run_mode == 'correctness')
            try:
                outcome = simulation.run()
            finally:
                torch.use_deterministic_algorithms(False)
            result = json.loads((root / 'failslow_result.json').read_text())
            plans = sorted(p.name for p in root.glob('source_plan_*.json'))
            self.plan_files = {p.name: json.loads(p.read_text()) for p in root.glob('source_plan_*.json')}
            self.checks = {json.loads(p.read_text())['slot']: json.loads(p.read_text()) for p in root.glob('serving_checks_*.json')}
            audits = [e['weights'] for p in root.glob('serving_checks_*.json') for e in json.loads(p.read_text())['events'] if 'weights' in e]
        skipped = 'timing' if run_mode == 'timing' else 'reference_audit skipped (fail-slow / restart)'
        self.assertTrue(audits and all(a == {'skipped': skipped} for a in audits), audits)
        self.assertEqual(outcome['metrics']['failslow_check'], result['check'])
        self.assertTrue(result['check']['passed'] and result['check']['event_order'], result['check'])
        self.assertGreater(result['requests_unfinished'], 0)  # the deadline cut the pool, nothing drained it
        events, settings = result['events_s'], result['settings']
        self.assertGreaterEqual(events['detect'] - events['inject_start'], settings['detect_s'] - 1e-6)
        self.assertGreaterEqual(events['deadline'] - events['inject_start'], settings['slow_s'] + settings['tail_s'] - 1e-6)
        self.assertEqual((result['method'], result['gpu_sharing']), (method, 'shared'))  # the simulation runs shared_gpu_mode
        metrics, series = failslow_metrics.analyze(result)
        self.assertGreater(metrics['throughput_tok_s'], 0)
        self.assertEqual(sum(series['global']), sum(len(r['t_tokens']) for r in result['requests']))
        return result, plans, metrics, [s for s in result['switches'] if s['domain'] == 'EDP0']

    def test_weavetp_switches_online(self):
        log = []
        result, plans, _, edp0 = self.launch('weavetp', log=log)
        victims, survivors = (8, 9, 12, 13), (10, 11, 14, 15)
        events = result['events_s']
        self.assertLess(events['detect'], events['shrink_done'])
        self.assertLess(events['recovered'], events['expand_done'])
        self.assertEqual((len(result['injection']['started']['pids']), result['injection']['stopped']['leftover']), (4, []))
        self.assertEqual(plans, ['source_plan_0_delta.json', 'source_plan_0_expand.json', 'source_plan_0_shrink.json',
                                 'source_plan_0_shrink_delta.json'])
        # a) the shrink migrates the KV (prefix and delta) while both EDPs decode inside the waves; nothing replays;
        # every victim sends in every shrink wave before it is released; the TP2 KV equals the mapped TP4 snapshots.
        shrink, expand = ({s['direction']: s for s in edp0}[d] for d in ('shrink', 'expand'))
        self.assertTrue(shrink['bytes']['kv_prefix'] > 0 and shrink['bytes']['kv_delta'] > 0, shrink['bytes'])
        self.assertTrue(shrink['overlap_steps'] > 0 and {s['direction']: s for s in result['switches'] if s['domain'] == 'EDP1'}['shrink']['overlap_steps'] > 0)
        self.assertEqual((result['replayed_tokens'], [e for e in log if e[0] == 'requeue']), (0, []))
        for slot in victims:
            mine = [e for e in log if e[1] == slot and e[0] in ('transfer', 'transition', 'release')]
            self.assertEqual(mine[:4], [('transfer', slot, 'shrink'), ('transfer', slot, 'shrink_delta'), ('release', slot, 4), ('transition', slot, 1)])
        for slot in survivors:  # no world transition before the shrink transfer
            mine = [e for e in log if e[1] == slot and e[0] in ('transfer', 'transition')]
            self.assertEqual(mine[:3], [('transfer', slot, 'shrink'), ('transfer', slot, 'shrink_delta'), ('transition', slot, 1)])
            self.assertEqual([(c['reference'], c['passed']) for c in self.checks[slot]['kv_checks']],
                             [(SHRINK_KV, True), (EXPAND_KV, True)])
        for slot in victims:
            self.assertEqual([(c['reference'], c['passed']) for c in self.checks[slot]['kv_checks']], [(EXPAND_KV, True)])
        # f) the shrink was planned on the profile with the fault GPUs down-weighted, the expand on the frozen profile
        shrink_plan, expand_plan = (self.plan_files[f'source_plan_0_{label}.json'] for label in ('shrink', 'expand'))
        self.assertNotEqual(shrink_plan['plan_profile_sha256'], shrink_plan['profile_sha256'])
        self.assertEqual(expand_plan['plan_profile_sha256'], expand_plan['profile_sha256'])
        # b) after startup: no model build, no planner call (16 TP4 loads + 4 TP2 builds, 2 plans per rank, all before serving)
        self.assertEqual(sorted(e[2] for e in log if e[0] == 'build'), [(2, False)] * 4 + [(4, True)] * 16)
        self.assertEqual(len([e for e in log if e[0] == 'plan']), 32)
        # c) no process group is created once serving started
        self.assertEqual([e for e in log if e[0] == 'new_group'], [])
        # e) one resident layout per rank in each steady state: TP2 storage released until the shrink, TP4 released from
        # the shrink until the expand, TP2 released again after it (the standby storages are reallocated, never rebuilt)
        for slot in survivors:
            self.assertEqual([(e[0], e[2]) for e in log if e[1] == slot and e[0] in ('release', 'reallocate')],
                             [('release', 2), ('reallocate', 2), ('release', 4), ('reallocate', 4), ('release', 2)])
        for slot in victims:
            self.assertEqual([(e[0], e[2]) for e in log if e[1] == slot and e[0] in ('release', 'reallocate')], [('release', 4), ('reallocate', 4)])
        self.assertGreater(expand['overlap_steps'], 1)  # one in-wave step (budget max_delta_steps - growth) + the growth step

    def test_anchortp_pauses_both_migrations(self):
        result, _, _, edp0 = self.launch('anchortp')
        self.assertEqual([(s['direction'], s['overlap_steps'], s['bytes']['kv_delta']) for s in edp0],
                         [('shrink', 0, 0), ('expand', 0, 0)])

    def test_restart_reloads_from_disk_and_replays(self):
        result, plans, metrics, edp0 = self.launch('restart')
        self.assertEqual(plans, [])  # nothing migrates
        self.assertEqual([(s['direction'], s['bytes']['weights'], s['bytes']['kv_prefix']) for s in edp0],
                         [('shrink', 0, 0), ('expand', 0, 0)])
        loaded, events = result['restart_loaded_s'], result['events_s']
        self.assertTrue(events['detect'] < loaded['shrink'] <= events['shrink_done'])
        self.assertTrue(events['recovered'] < loaded['expand'] <= events['expand_done'])
        self.assertGreater(result['replayed_tokens'], 0)
        downtime = metrics['restart_downtime']
        self.assertTrue(downtime['total_s'] > 0 and downtime['shrink']['stop_s'] < downtime['shrink']['resume_s'])

    def test_ignore_keeps_the_layout(self):
        result, plans, _, edp0 = self.launch('ignore')
        self.assertEqual((edp0, plans), ([], []))
        self.assertIn('slow_stop', result['events_s'])

    def test_nofault_injects_nothing(self):
        result, _, _, edp0 = self.launch('nofault')
        self.assertEqual((edp0, result['level'], result['injection']['started']), ([], None, None))
        self.assertNotIn('slow_start', result['events_s'])

    def test_timing_run_switches_without_audits(self):
        result, _, _, edp0 = self.launch('weavetp', run_mode='timing')
        self.assertEqual([s['direction'] for s in edp0], ['shrink', 'expand'])


if __name__ == '__main__':
    unittest.main()
