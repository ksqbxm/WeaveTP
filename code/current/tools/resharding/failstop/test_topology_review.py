"""Regressions found while reviewing Step 2, including the real supervisor path."""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import common
import run_topology
import safety
import supervisor
import test_topology
import torch
from test_topology import group_patches, rows, session
from topology_probe import EmulateSupervisor

from megatron.core.failstop import build_replica_pg_collections, replica_group_specs


class ReviewTests(unittest.TestCase):
    def test_next_round_rekeys_caches_and_keeps_current_forward_available(self):
        s = session()
        s.fail(s.failed)
        s.complete()
        s.rejoin(s.failed)
        s.complete()
        old = s.cache_key()
        s.resources.plans[old] = 'previous round'
        s.next_round()
        self.assertFalse(s.resources.plans)
        with s.resources.forward(s.cache_key()):
            pass
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            with s.resources.forward(old):
                pass
        s.resources.invalidate()
        with self.assertRaisesRegex(RuntimeError, 'full recovery'):
            s.next_round()

    def test_ep_and_dp_pair_actual_tp_coordinates_after_api_sorting(self):
        specs = replica_group_specs([((11, 10, 14, 15), 2, 2, 2)])
        for field in ('ep', 'dp', 'dp_cp'):
            pairs = [tuple(sorted(ranks)) for _, name, ranks in specs if name == field]
            self.assertEqual(pairs, [(10, 14), (11, 15)])

    def test_invalid_local_orders_fail_before_membership_mutation(self):
        for orders in ({}, {0: (10, 11)}, {0: (10, 11, 11), 1: (14, 15)}):
            with self.subTest(orders=orders), self.assertRaisesRegex(RuntimeError, 'local order'):
                session(local_orders=orders)

    def test_old_binding_stays_blocked_after_new_epoch_activation(self):
        model, binding, resources = test_topology.BindingTests().fixture()
        resources.invalidate()
        resources.activate(('epoch', 1))
        with self.assertRaisesRegex(RuntimeError, 'stale|unavailable'):
            model(torch.ones(3))
        binding.close()

    def test_rejected_nested_forward_does_not_release_outer_boundary(self):
        model, binding, resources = test_topology.BindingTests().fixture()
        unlocked = threading.Event()
        contenders = []

        def forward(x):
            binding.forward_ready = False
            try:
                with self.assertRaisesRegex(RuntimeError, 'source-only'):
                    model(x)
                def contend():
                    with resources.lock:
                        unlocked.set()
                contender = threading.Thread(target=contend)
                contenders.append(contender)
                contender.start()
                self.assertFalse(unlocked.wait(0.05), 'nested rejection released the outer forward lock')
            finally:
                binding.forward_ready = True
            return x

        model.forward = forward
        try:
            model(torch.ones(3))
        finally:
            for contender in contenders:
                contender.join(1)
            binding.close()

    def test_unsupported_native_handles_rejected_before_any_group_mutation(self):
        for field, value in (('_comm_manager', object()),
                             ('config', SimpleNamespace(cuda_graph_impl='local'))):
            model, binding, resources = test_topology.BindingTests().fixture()
            old = binding.groups
            setattr(model.token_dispatcher, field, value)
            resources.invalidate()
            with group_patches(0, 4):
                new = build_replica_pg_collections([((0, 1, 2, 3), 2, 2, 2)])[0]
                with self.assertRaisesRegex(RuntimeError, 'dispatcher|CUDA graph'):
                    binding.rebind(new)
            self.assertIs(model.pg_collection, old)
            self.assertIs(model.tp_group, old.tp)
            binding.close()

    def test_emulate_supervisor_rejoin_keeps_all_registered_cuda_contexts(self):
        # Exercise Supervisor.run itself: the threaded protocol simulation does
        # not execute its pre-spawn checks. Emulate never kills these contexts.
        for shared in (False, True):
            with self.subTest(shared=shared), tempfile.TemporaryDirectory() as directory:
                topology = rows()
                local = [r for r in topology if r['node'] == 'SL3061']
                config = {**common.DEFAULTS, 'output': directory, 'node': 'SL3061',
                          'launch': 'cpu', 'mode': 'emulate', 'topology': topology,
                          'stable_samples': 1, 'sample_s': 0, 'shared_gpu_mode': shared}
                bus = Mock()
                bus.wait.return_value = [{'offset': 0, 'error': 0}]
                s = EmulateSupervisor(config, bus, Mock())
                s.workers = {r['slot']: (Mock(pid=100 + r['slot'], poll=Mock(return_value=None),
                                             wait=Mock(return_value=0)),
                                        {'pid': 100 + r['slot'], 'gpu_uuid': r['gpu_uuid']}) for r in local}
                s.children.children = list(s.workers.values())
                s.children.exited = Mock()
                idle = {'end_local': 1, 'gpus': [{'uuid': r['gpu_uuid'], 'used_mib': 0, 'util_percent': 0,
                                  'processes': []} for r in local]}
                live = {'gpus': [{**g, 'used_mib': 1024, 'processes': [
                    {'pid': 100 + r['slot'], 'name': 'python', 'used_mib': 1024}]}
                    for g, r in zip(idle['gpus'], local)]}
                mps = {'servers': [], 'clients': []}
                def inspect(name, expected=None):
                    if name == 'rejoin_pre_spawn':
                        safety.check_gpu_processes(live, mps, expected, shared, s.task_pids)
                        return live, mps
                    return idle, mps
                s.inspect = Mock(side_effect=inspect)
                s.wait = Mock(side_effect=lambda keys, **kw: [topology] if keys == ['topology'] else [True])
                s.verify_contexts = Mock()
                start = s.start_workers
                s.start_workers = lambda slots, epoch: start(slots, epoch) if epoch else None
                with patch.object(supervisor, 'gpu_snapshot', return_value=idle):
                    s.run()
                self.assertEqual(s.state, 'rejoined')
                self.assertEqual(s.verify_contexts.call_args.args[1], 2)

    def test_real_preflight_still_requires_dead_victims_and_reclaimed_memory(self):
        topology = rows()
        victims = common.failed_slots(topology)
        with tempfile.TemporaryDirectory() as directory:
            config = {**common.DEFAULTS, 'output': directory, 'node': 'SL3061', 'topology': topology}
            s = supervisor.Supervisor(config, Mock(), Mock())
            local = [r for r in topology if r['node'] == 'SL3061']
            s.workers = {r['slot']: (Mock(pid=100 + r['slot']), {'gpu_uuid': r['gpu_uuid']}) for r in local}
            gpu = {'uuid': topology[victims[0]]['gpu_uuid'], 'used_mib': 100}
            s.inspect = Mock(return_value=({'gpus': [gpu]}, {}))
            with self.assertRaisesRegex(RuntimeError, 'memory rose'):
                s.rejoin_preflight({gpu['uuid']: {'threshold_mib': 8}})
            self.assertEqual(s.inspect.call_args.args[1],
                             {100 + r['slot']: r['gpu_uuid'] for r in local if r['slot'] not in victims})


class ModeTests(unittest.TestCase):
    def dry_run(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run_topology.main(['--node', 'SL3060', '--suite-id', 'cpu',
                '--output', '/data/cpu', '--dry-run', *extra]), 0)
        return json.loads(output.getvalue())

    def test_correctness_switch_and_quick_enable_shared_test_only_mode(self):
        self.assertFalse(self.dry_run()['shared_gpu_mode'])
        for extra in (('--correctness-only',), ('--quick',)):
            result = self.dry_run(*extra)
            self.assertTrue(result['correctness_only'] and result['shared_gpu_mode'])
            self.assertFalse(result['timing_valid'])

    def test_mode_summary_requires_every_selected_launch(self):
        selected = run_topology.cases('both', 5, 2, 'cpu')
        results = [{**case, 'status': 'passed'} for case in selected]
        results[-1]['status'] = 'failed'
        summary = run_topology.summarize(selected, results, quick=False, correctness_only=False)
        self.assertFalse(summary['accepted'])
        self.assertEqual(summary['real'], 'failed')
        results[-1]['status'] = 'passed'
        summary = run_topology.summarize(selected, results, quick=False, correctness_only=True)
        self.assertTrue(summary['all_cases_passed'])
        self.assertFalse(summary['accepted'] or summary['timing_valid'])
        self.assertEqual(summary['real'], 'passed')
        self.assertTrue(run_topology.summarize(selected, results, quick=False, correctness_only=False)['accepted'])
        self.assertFalse(run_topology.summarize(selected, results[:-1], quick=False, correctness_only=False)['all_cases_passed'])

    def test_removed_step2_flag_is_not_a_compatibility_alias(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.dry_run('--allow-foreign-gpu-processes')


if __name__ == '__main__':
    unittest.main()
