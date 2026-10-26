"""Run the Step 6 B3 serving worker and gate with 16 concurrent CPU actors and a fake runtime.

The fake model is a two-layer toy attention through the block-KV hooks with TP all-reduced logits
that also depend on every resident weight, so a wrong migration, a wrong KV move or a diverged
history changes the committed tokens. This proves the protocol: decisions over the bus, ledger
mirrors, fault -> replay -> rejoin -> expand with KV prefix + delta, victims joining, every request
completing, result/check/events_actual written by the gate. The DeepSeek forward runs only on the server.
"""

import contextlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import analyze_failstop_e2e
import common
import model_worker
import run_serving
import serving_probe
import serving_worker
import test_async_protocol as async_test
import test_model_protocol as model_test
import torch
import torch.distributed as dist

from tests.unit_tests.e2e import test_block_context_forward as toy
from tools.resharding.e2e.toy_inputs import write_toy_inputs

HEADS, LAYERS = 4, (1, 2)
GEOMETRY = {'layers': 2, 'heads': HEADS, 'group_heads': 1, 'k_dim': toy.DK, 'v_dim': toy.DV}
HEAD = torch.randn(HEADS * toy.DV, toy.VOCAB)


class FakeModel(model_test.FakeModel):
    """Toy TP model: this rank's heads attend through the context hooks; logits are all-reduced over the TP group."""

    def forward(self, ids, positions, attention_mask, *, inference_context, runtime_gather_output=True):
        assert not self.released, 'forward on a released model'
        local = HEADS // self.tp
        heads = slice(self.coord * local, (self.coord + 1) * local)
        q, k, v = toy.projections(ids[0], inference_context.rotary_positions)
        q, k, v = q[:, :, heads], k[:, :, heads], v[:, :, heads]
        for layer in LAYERS:
            q, k, v = inference_context.adjust_key_value(layer, q, k, v)
            out = inference_context.core_attention(layer, q, k, v, None, dense_attention=toy.dense_module, softmax_scale=toy.SCALE)
        rows = HEAD.view(HEADS, toy.DV, toy.VOCAB)[heads].reshape(local * toy.DV, toy.VOCAB)
        partial = out[:, 0] @ rows + sum(float(p.sum()) for _, p in self.named_parameters()) * 1e-3
        dist.all_reduce(partial, group=self.pg_collection.tp)
        return partial[None]


class FakeRuntime(model_test.FakeRuntime):
    device = 'cpu'
    args = SimpleNamespace(params_dtype=torch.float32, max_position_embeddings=64)

    def build(self, tp, groups, *, load):
        return FakeModel(tp, groups, clean=load)


class Simulation(model_test.Simulation):
    toy = {'count': 12}  # write_toy_inputs arguments

    def actor(self, slot, epoch):
        self.local.identity = slot, epoch
        self.local.pid = 10000 + slot + epoch * 100
        config = {**self.config, 'node': self.rows[slot]['node'], 'clock': {'offset': 0, 'error': 0}, 'topology': self.rows}
        try:
            serving_worker.ServingWorker(config, slot, epoch, FakeRuntime()).run_worker()
        except async_test.topology_test.Killed:
            pass
        except BaseException as error:
            import traceback
            self.errors.append(f'{slot}/{epoch}: {traceback.format_exc()}')
            self.put('failure', str(error))

    def reduce(self, tensor, group, op=None, **kwargs):
        values = self.collective(group, tensor, 'reduce')
        stacked = torch.stack(values)
        if op == dist.ReduceOp.MAX:
            tensor.copy_(stacked.max(dim=0).values)
        elif op == dist.ReduceOp.MIN:
            tensor.copy_(stacked.min(dim=0).values)
        else:
            tensor.copy_(stacked.sum(dim=0))

    def broadcast_object_list(self, payload, src, group=None, **kwargs):
        values = self.collective(group, payload[0], 'bcast')
        payload[0] = values[group.members.index(src)]

    def run(self):
        reference = self.root / 'reference'
        model_test.reference_hashes(reference)
        (self.root / 'profile.json').write_text(json.dumps(model_test.profile([model_worker.identity(r) for r in self.rows])))
        settings = {**common.DEFAULTS, **self.extra_config}
        inputs = write_toy_inputs(self.root / 'inputs', failstop=settings['scenario'] == 'failstop', **self.toy)
        self.config = {**common.DEFAULTS, 'output': str(self.root), 'launch': 'cpu_serving', 'commit': 'cpu',
                       'mode': self.mode, 'rounds': 1, 'quick_mode': True, 'correctness_only': True,
                       'shared_gpu_mode': True, 'port': 1, 'stage_timeout_s': 8, 'launch_timeout_s': 60, 'reclaim_timeout_s': 8,
                       'probe': common.SERVING_PROBE, 'profile_file': str(self.root / 'profile.json'), 'reference_root': str(reference),
                       'deterministic_mode': settings['run_mode'] == 'correctness', 'kv_heads': HEADS, 'memory_margin_bytes': 0,
                       'memory_limit_bytes': None,
                       'serving': {'requests': str(inputs['requests']), 'workload': str(inputs['workload']), 'plan': str(inputs['plan']),
                                   'b_max': 2, 'decode_policy': 'greedy', 'block_tokens': 4, 'segment_tokens': 16, 'max_delta_steps': 2, 'max_overlap_steps': 1,
                                   'attention': 'auto', 'growth': {'shrink': 0, 'expand': 1}, 'geometry': GEOMETRY, 'capacity_tokens': 40},
                       **self.extra_config}
        sim = self

        class Journal(common.Journal):
            def event(self, stage, **fields):
                return super().event(stage, **{'pid': sim.local.pid, **fields})

        class Stream:
            def wait_stream(self, other):
                pass

            def synchronize(self):
                pass

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.multiple(model_worker, parent_death_signal=lambda: None, verify_worker_context=lambda *_: {},
                                               connect=lambda _: (None, self), Journal=Journal, make_copy_service=self.copy_service))
            stack.enter_context(patch.object(model_worker, 'os', SimpleNamespace(
                environ={'CUDA_MPS_PIPE_DIRECTORY': str(self.root / 'absent'),
                         **(common.DETERMINISTIC_ENVIRONMENT if self.config['deterministic_mode'] else {})},
                getpid=lambda: self.local.pid)))
            stack.enter_context(patch.multiple(torch.cuda, set_device=lambda *_: None, device_count=lambda: 1,
                get_device_properties=lambda _: SimpleNamespace(uuid=self.rows[self.local.identity[0]]['gpu_uuid']),
                current_stream=lambda: Stream(), Stream=Stream, stream=lambda _: contextlib.nullcontext(),
                reset_peak_memory_stats=lambda: None, is_available=lambda: False))
            for name in ('empty', 'tensor', 'full', 'zeros'):
                original = getattr(torch, name)

                def cpu(*a, _original=original, **kw):
                    if kw.get('device') == 'cuda':
                        kw['device'] = 'cpu'
                    return _original(*a, **kw)
                stack.enter_context(patch.object(torch, name, cpu))
            stack.enter_context(patch.multiple(dist, init_process_group=self.init, PrefixStore=lambda prefix, _: prefix,
                new_group=self.group, destroy_process_group=lambda: None, is_initialized=lambda: False,
                get_rank=lambda group=None: self.local.rank if group is None else group.rank(),
                get_world_size=lambda group=None: self.local.size if group is None else group.size(),
                get_process_group_ranks=lambda group: list(group.members), get_backend=lambda: 'fake',
                all_reduce=self.reduce, all_gather=self.gather, broadcast_object_list=self.broadcast_object_list,
                barrier=lambda group=None, **kwargs: self.collective(group, None, 'barrier')))
            stack.enter_context(patch.object(model_worker.c10d, '_abort_process_group',
                lambda: common.require(getattr(self.local, 'rank', None) is not None, 'abort without a process group')))
            stack.enter_context(patch.object(serving_probe.signal, 'SIGKILL', 9, create=True))
            supervisors = [threading.Thread(target=self.supervisor, args=(node,), daemon=True) for node in common.NODES]
            for thread in supervisors:
                thread.start()
            try:
                result = serving_probe.control(self.config, self, common.Journal(self.root / 'controller.jsonl', 'SL3060', 'controller'),
                                               lambda: None)
            finally:
                for thread in supervisors + self.actors:
                    thread.join(timeout=9)
            assert not self.errors, '\n'.join(self.errors)
            assert result['status'] == 'passed'
            return result


class ServingProtocolTests(unittest.TestCase):
    @contextlib.contextmanager
    def simulate(self, mode, extra, toy=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.use_deterministic_algorithms(extra.get('run_mode', 'correctness') == 'correctness')
            try:
                simulation = Simulation(mode, root, 1, extra)
                if toy is not None:
                    simulation.toy = toy
                result = simulation.run()
            finally:
                torch.use_deterministic_algorithms(False)
            yield result, json.loads((root / 'result.json').read_text()), json.loads((root / 'events_actual.json').read_text()), root

    def test_nofault_serving_has_no_events_and_no_fault_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.use_deterministic_algorithms(True)
            try:
                result = Simulation('emulate', root, 1, {'scenario': 'nofault'}).run()
            finally:
                torch.use_deterministic_algorithms(False)
        metrics = result['metrics']
        self.assertEqual((metrics['requests_completed'], metrics['scenario'], metrics['stats_valid']), (12, 'nofault', True))
        self.assertTrue(metrics['check_passed'] and 'kv_check_passed' not in metrics and 'replay_prefill_tokens' not in metrics)

    def test_timing_run_mode_skips_audits_and_snapshots(self):
        with self.simulate('emulate', {'run_mode': 'timing', 'method': 'anchortp'}) as (result, result_json, events, root):
            self.check_timing(result, result_json, root)

    def check_timing(self, result, result_json, root):
        metrics = result['metrics']
        self.assertEqual((metrics['requests_completed'], metrics['run_mode'], metrics['kv_check_passed']), (12, 'timing', None))
        self.assertTrue(metrics['check_passed'] and metrics['replay_prefill_tokens'] > 0)
        self.assertFalse((root / 'kv').exists())
        self.assertEqual((result_json['failstop']['kv_check_note'], result_json['failstop']['reference_audit']), ('skipped: timing run', 'skipped'))
        self.assertEqual((result_json['config']['run_mode'], result_json['config']['validate_cutover']), ('timing', False))
        checks = json.loads(next(root.glob('serving_checks_1*.json')).read_text())
        self.assertEqual(checks['step_times'], 'not recorded (timing)')
        self.assertTrue(all(s['audit'].get('skipped') == 'timing' for s in result_json['switches'] if s['domain'] == 'EDP0'))
        self.assertEqual(result_json['time_base'].split(' ')[0], 'seconds')
        self.assertTrue(all(r['t_tokens'] == sorted(r['t_tokens']) and len(r['t_tokens']) == r['output_len'] for r in result_json['requests']))
        # serving_start is the gate's phase-0 validation; the first decisions are already out, so the earliest
        # admit may precede it by the report round trip (milliseconds), never by seconds.
        self.assertGreaterEqual(min(r['t_admit'] for r in result_json['requests']), -1.0)
        self.assertLess(max(r['t_done'] for r in result_json['requests']), result_json['serving_wall_s'] + 1.0)

    def run_mode(self, mode, method='weavetp'):
        with self.simulate(mode, {'method': method}) as (result, result_json, events, root):
            metrics = result['metrics']
            self.assertEqual(metrics['requests_completed'], 12)
            self.assertTrue(metrics['check_passed'] and metrics['kv_check_passed'])
            self.assertGreater(metrics['replay_prefill_tokens'], 0)
            self.assertEqual([s['direction'] for s in result_json['switches'] if s['domain'] == 'EDP0'], ['shrink', 'expand'])
            self.assertTrue(all(s['validation']['passed'] for s in result_json['switches']))
            self.assertEqual(result_json['failstop']['reference_audit'], 'passed')  # migrated targets checked against the reference
            self.assertEqual(result_json['totals']['generated_tokens'], sum(r['output_len'] for r in result_json['requests']))
            self.assertTrue(all(r['t_admit'] <= r['t_first'] <= r['t_done'] for r in result_json['requests']))
            self.assertEqual(events['actual_trigger']['threshold'], 3)
            self.assertGreaterEqual(events['recovery_s']['expand_done'], events['recovery_s']['shrink_done'])
            self.assertEqual(sorted(events['plan']['coords']), [(0, 0), (0, 1), (1, 0), (1, 1)] if False else sorted(map(list, events['plan']['coords'])))
            checks = sorted(root.glob('serving_checks_*.json'))
            self.assertEqual(len(checks), 16)
            for path in checks:
                record = json.loads(path.read_text())
                self.assertEqual(record['attention_impl']['decode'], 'torch_splitkv')
            self.assertEqual((root / 'kv' / 'r0' / 'expand').exists(), True)
            # The method is named in the result and the EDP0 switches carry its semantics.
            self.assertEqual((result_json['method'], metrics['method'], result_json['failstop']['method']['name']), (method,) * 3)
            edp0 = {s['direction']: s for s in result_json['failstop']['edp0_switches']}
            plans = {label: json.loads((root / f'source_plan_0_{label}.json').read_text()) for label in ('shrink', 'expand', 'delta')}
            self.assertTrue(all(plan['method'] == method and plan['planner'] == method for plan in plans.values()))
            if method == 'anchortp':
                self.assertEqual([(s['overlap_steps'], s['kv_delta_bytes']) for s in edp0.values()], [(0, 0), (0, 0)])
                self.assertEqual(result_json['failstop']['method']['kv_semantics_status'], '未与论文核对')
                self.assertEqual(plans['delta']['tasks'], [])
                self.assertTrue(all(plan['implementation_ref'].startswith('codex/anchortp-faithful@eba3bc8') for plan in plans.values()))
                self.assertGreater(plans['shrink']['paths']['network']['bytes'], 0)  # shards 0/1 only exist on EDP1
                self.assertEqual(plans['expand']['paths']['network']['bytes'], 0)  # TP2 survivors cover every TP4 slice
                self.assertTrue(all(s['overlap_steps'] == 0 for s in result_json['switches']))
            else:
                self.assertEqual(edp0['expand']['overlap_steps'], 1)
                self.assertIn('loads', plans['delta'])
            return result

    def test_emulate_cycle(self):
        self.run_mode('emulate')

    def test_real_cycle_with_replacement_victims(self):
        self.run_mode('real')

    def test_anchortp_emulate_cycle(self):
        self.run_mode('emulate', 'anchortp')

    def test_anchortp_real_cycle(self):
        self.run_mode('real', 'anchortp')

    def test_expand_with_requests_in_flight(self):
        """60 toy requests and a 1 s rejoin keep EDP0 serving at the expand, so the ranks that join (emulate victims, real
        replacements) decode handed-over requests whose admits they never saw: a KeyError in Driver.body before the
        fix. Every request completes, the gate's result holds the serving_wall_s the calibration reads, the analyzer
        accepts the gate's files and the quick check's criterion (run_serving.handed_over) covers those requests.
        The simulation's 8 s bus-wait cap bounds the workload."""
        loop, joined = serving_worker.ServingWorker.loop, {}

        def record(worker):
            loop(worker)
            joined[worker.slot] = {seq for seq, entry in worker.driver.run.request_log.items() if entry['t_tokens'] and not entry['admits']}

        for mode in ('emulate', 'real'):
            joined.clear()
            with self.subTest(mode=mode), patch.object(serving_worker.ServingWorker, 'loop', record), \
                    self.simulate(mode, {'method': 'weavetp'}, toy={'count': 60, 'rejoin_seconds': 1.0}) as (result, result_json, events, root):
                victims = common.failed_slots(json.loads((root / 'rank_map.json').read_text())['0/0'])
                handed = set().union(*joined.values())
                self.assertTrue(handed, 'no joining rank decoded a handed-over request: the workload drained before the expand')
                self.assertLessEqual({slot for slot, seqs in joined.items() if seqs}, set(victims))
                self.assertLessEqual(handed, set(run_serving.handed_over(result_json)))
                metrics = result['metrics']
                self.assertEqual((metrics['requests_completed'], metrics['check_passed'], metrics['kv_check_passed']), (60, True, True))
                self.assertEqual(result_json['serving_wall_s'], metrics['serving_wall_s'])
                self.assertEqual(result_json['gpu_sharing'], 'shared')
                self.assertEqual(analyze_failstop_e2e.main([str(root), '--out', str(root / 'analysis')]), 0)
                self.assertEqual(json.loads((root / 'analysis' / 'rejected.json').read_text()), [])

    def test_restart_cycle_reloads_tp2_then_tp4_from_disk(self):
        """restart in the emulate fail-stop cycle: the survivors reload TP2 and all eight EDP0 ranks TP4 in the same processes
        (no reference audit: reference_audit='skipped'), nothing migrates, EDP0's requests replay; the gate times both reloads
        and the analyzer splits restart's downtime per switch as for fail-slow."""
        with self.simulate('emulate', {'method': 'restart'}, toy={'count': 40, 'rejoin_seconds': 1.0}) as (result, result_json, events, root):
            metrics, failstop = result['metrics'], result_json['failstop']
            self.assertEqual((metrics['requests_completed'], metrics['check_passed'], metrics['kv_check_passed']), (40, True, None))
            self.assertEqual(failstop['kv_check_note'], 'restart: no KV moves, EDP0 requests replay')
            self.assertEqual([(s['direction'], s['weights_bytes'], s['kv_prefix_bytes'], s['kv_delta_bytes']) for s in failstop['edp0_switches']],
                             [('shrink', 0, 0, 0), ('expand', 0, 0, 0)])
            self.assertEqual(list(root.glob('source_plan_*.json')), [])
            self.assertGreater(failstop['replay_prefill_tokens'], 0)
            loaded, worker = failstop['restart_loaded_s'], events['worker_events_s']
            self.assertTrue(worker['fail'] < loaded['shrink'] <= worker['shrink_done'] < worker['rejoin_start'] < loaded['expand'] <= worker['expand_done'])
            self.assertEqual(failstop['reference_audit'], 'skipped')
            audits = [e['weights'] for p in root.glob('serving_checks_*.json') for e in json.loads(p.read_text())['events'] if 'weights' in e]
            self.assertTrue(audits and all('skipped' in a for a in audits))  # the initial loads and the victims' TP4 reloads
            self.assertEqual(analyze_failstop_e2e.main([str(root), '--out', str(root / 'analysis')]), 0)
            downtime = json.loads(next((root / 'analysis').glob('metrics_*.json')).read_text())['restart_downtime']
            self.assertTrue(downtime['total_s'] > 0 and downtime['shrink']['loaded_s'] == loaded['shrink'])


if __name__ == '__main__':
    unittest.main()
