"""CPU contracts of Step 5: deterministic-mode manifests and comparisons, mode-independent plans,
the step-time ratio, the real-model per-GPU memory preflight and the suite driver. The DeepSeek
forward itself runs only on the server."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import determinism
import model_memory
import model_reference
import run_model
import torch
from common import DETERMINISTIC_ENVIRONMENT, FAIL_COORDS, topology
from test_sources import part, profile

from megatron.core.resharding.mixed_sources import build_mixed_plan, normalized
from tools.resharding.correctness.router_determinism_driver import GIB

DETERMINISTIC = determinism.mode_manifest(True, algorithms_enabled=True, environ=DETERMINISTIC_ENVIRONMENT)
DEFAULT = determinism.mode_manifest(False, algorithms_enabled=False, environ={})
W2, W4 = model_memory.weight_bytes(2), model_memory.weight_bytes(4)


class ManifestTests(unittest.TestCase):
    def test_deterministic_requires_argument_algorithms_and_environment(self):
        self.assertEqual(DETERMINISTIC['mode'], 'deterministic')
        determinism.require_deterministic(DETERMINISTIC, 'reference')
        cases = {'--deterministic-mode not set': determinism.mode_manifest(False, algorithms_enabled=True, environ=DETERMINISTIC_ENVIRONMENT),
                 'algorithms disabled': determinism.mode_manifest(True, algorithms_enabled=False, environ=DETERMINISTIC_ENVIRONMENT),
                 "NCCL_ALGO='Tree'": determinism.mode_manifest(True, algorithms_enabled=True,
                                                               environ={**DETERMINISTIC_ENVIRONMENT, 'NCCL_ALGO': 'Tree'}),
                 'CUBLAS_WORKSPACE_CONFIG=None': determinism.mode_manifest(True, algorithms_enabled=True, environ={'NCCL_ALGO': 'Ring'})}
        for reason, manifest in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(manifest['mode'], 'default')
                with self.assertRaisesRegex(RuntimeError, reason):
                    determinism.require_deterministic(manifest, 'candidate')
        determinism.require_default(DEFAULT, 'timing')
        with self.assertRaisesRegex(RuntimeError, 'default mode'):
            determinism.require_default(DETERMINISTIC, 'timing')


class RecordTests(unittest.TestCase):
    def record(self, manifest=DETERMINISTIC, bump=None, shape=(1, 1, 8), tokens=(1, 2, 3), checksum='p0', layout=None):
        logits = torch.full(shape, 1.0, dtype=torch.bfloat16)
        if bump is not None:
            logits.view(torch.int16).view(-1)[bump] += 1  # exactly one ULP
        return determinism.bitwise_record(logits, step=0, manifest=manifest, layout=layout or {'tp': 2},
                                          history=determinism.history('r0', tokens), parameter_checksum=checksum)

    def test_comparison_refuses_unless_both_deterministic_and_same_state(self):
        reference = self.record()
        self.assertTrue(determinism.compare_records(reference, self.record())['equal'])
        with self.assertRaisesRegex(RuntimeError, 'candidate is not in deterministic mode'):
            determinism.compare_records(reference, self.record(manifest=DEFAULT))
        with self.assertRaisesRegex(RuntimeError, 'reference is not in deterministic mode'):
            determinism.compare_records(self.record(manifest=DEFAULT), reference)
        for kwargs, reason in (({'tokens': (1, 2, 4)}, 'history'), ({'checksum': 'p1'}, 'parameter_checksum'),
                               ({'layout': {'tp': 4}}, 'layout')):
            with self.subTest(reason=reason), self.assertRaisesRegex(RuntimeError, reason):
                determinism.compare_records(reference, self.record(**kwargs))

    def test_comparison_reports_one_ulp_shape_and_hashes(self):
        reference, candidate = self.record(), self.record(bump=5)
        result = determinism.compare_records(reference, candidate)
        self.assertFalse(result['equal'])
        mismatch = result['first_mismatch']
        self.assertEqual((mismatch['index'], mismatch['count'], mismatch['reference']['bits'], mismatch['candidate']['bits']),
                         ([0, 0, 5], 1, '0x3f80', '0x3f81'))
        self.assertEqual((result['dtype'], result['shape'], result['history_length']), ('torch.bfloat16', [1, 1, 8], 3))
        self.assertEqual(determinism.compare_records(reference, self.record(shape=(1, 1, 9)))['first_mismatch']['kind'], 'shape/dtype')
        self.assertNotEqual(determinism.history('r0', [1, 2])['sha256'], determinism.history('r1', [1, 2])['sha256'])


class PlanTests(unittest.TestCase):
    def plan(self, ranks):
        sources = [part('a', 0, 4, rank=ranks['a']), part('b', 4, 8, rank=ranks['b'])]
        targets = [part('c', rank=ranks['c'])]
        tasks, _, _ = build_mixed_plan(sources, targets, profile(ranks), membership=ranks, rank_map=ranks, epoch=1)
        return tasks

    def test_normalization_matches_step3_and_ignores_temporary_ranks(self):
        tasks = self.plan({'a': 0, 'b': 1, 'c': 2})
        rows = [asdict(t) for t in tasks]
        self.assertEqual(determinism.normalize_tasks(rows), json.loads(json.dumps(normalized(tasks))))
        renumbered = [asdict(t) for t in self.plan({'a': 5, 'b': 3, 'c': 0})]
        self.assertNotEqual(rows, renumbered)
        self.assertEqual(determinism.plan_consistency(rows, renumbered), {'equal': True, 'tasks': [2, 2], 'first_difference': None})

    def test_first_difference_names_task_and_fields(self):
        rows = [asdict(t) for t in self.plan({'a': 0, 'b': 1, 'c': 2})]
        changed = json.loads(json.dumps(rows))
        changed[1]['payload_bytes'] += 8
        changed[1]['wave'] += 1
        result = determinism.plan_consistency(rows, changed)
        self.assertFalse(result['equal'])
        self.assertEqual((result['first_difference']['index'], result['first_difference']['fields']), (1, ['payload_bytes', 'wave']))
        self.assertEqual(determinism.plan_consistency(rows, rows[:1])['first_difference']['fields'], ['task_count'])

    def test_launch_comparison_requires_one_profile_and_reports_inputs(self):
        rows = [asdict(t) for t in self.plan({'a': 0, 'b': 1, 'c': 2})]
        plan = {'tasks': rows, 'profile_sha256': 'f' * 64, 'membership': ['a', 'b', 'c'], 'epoch': 1,
                'sources': [asdict(part('a', 0, 4)), asdict(part('b', 4, 8, rank=1))], 'targets': [asdict(part('c', rank=2))],
                'environment': dict(DETERMINISTIC_ENVIRONMENT)}
        with tempfile.TemporaryDirectory() as temporary:
            reference, candidate = Path(temporary, 'default'), Path(temporary, 'deterministic')
            for root in (reference, candidate):
                root.mkdir()
                (root / 'source_plan_0_1.json').write_text(json.dumps(plan))
            result = determinism.compare_launch_plans(reference, candidate)
            self.assertTrue(result['equal'])
            self.assertEqual(result['plans']['source_plan_0_1.json']['inputs_equal'],
                             {'membership': True, 'epoch': True, 'sources': True, 'targets': True})
            (candidate / 'source_plan_0_1.json').write_text(json.dumps({**plan, 'profile_sha256': 'e' * 64, 'environment': {}}))
            result = determinism.compare_launch_plans(reference, candidate)
            self.assertFalse(result['equal'])
            self.assertFalse(result['plans']['source_plan_0_1.json']['profile_equal'])
            self.assertEqual(result['plans']['source_plan_0_1.json']['environments'][1], {})
            (candidate / 'source_plan_0_2.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, 'different rounds/phases'):
                determinism.compare_launch_plans(reference, candidate)
            with self.assertRaisesRegex(RuntimeError, 'no source_plan'):
                determinism.compare_launch_plans(Path(temporary), candidate)


class RatioTests(unittest.TestCase):
    def samples(self, *ms, layout='tp2'):
        return [{'layout': layout, 'step': i, 'context': 32 + i, 'gpu_ms': m} for i, m in enumerate(ms)]

    def test_ratio_per_step_and_medians(self):
        result = determinism.step_time_ratio(self.samples(30.0, 12.0, 15.0), self.samples(20.0, 10.0, 10.0))
        self.assertEqual([r['ratio'] for r in result['per_step']], [1.5, 1.2, 1.5])
        self.assertEqual((result['prefill_ratio'], result['decode_median_ratio'], result['median_ratio']), (1.5, 1.35, 1.5))
        self.assertIn('not a performance conclusion', result['scope'])

    def test_ratio_refuses_mismatched_or_invalid_samples(self):
        base = self.samples(20.0, 10.0)
        with self.assertRaisesRegex(RuntimeError, 'context'):
            determinism.step_time_ratio([{**s, 'context': 99} for s in base], base)
        with self.assertRaisesRegex(RuntimeError, 'layout'):
            determinism.step_time_ratio(self.samples(20.0, 10.0, layout='tp4'), base)
        with self.assertRaisesRegex(RuntimeError, 'invalid step time'):
            determinism.step_time_ratio(self.samples(20.0, 0.0), base)
        with self.assertRaisesRegex(RuntimeError, 'pair up'):
            determinism.step_time_ratio(self.samples(20.0), base)


def inventory(free=None):
    return [{'index': i, 'uuid': f'GPU-{i}', 'total_bytes': 32 * GIB, 'free_bytes': (free or {}).get(i, 30 * GIB)} for i in range(8)]


class MemoryTests(unittest.TestCase):
    def test_roles_follow_the_fault_coordinates(self):
        rows = topology({node: [{'index': i, 'uuid': f'{node}-{i}'} for i in range(8)] for node in ('SL3060', 'SL3061')})
        roles = [model_memory.role(r) for r in rows]
        self.assertEqual(roles.count(model_memory.HEALTHY), 8)
        self.assertEqual([(r['ep'], r['etp']) for r in rows if model_memory.role(r) == model_memory.REJOIN], sorted(FAIL_COORDS))
        self.assertEqual({r['etp'] for r in rows if model_memory.role(r) == model_memory.SURVIVOR}, {2, 3})

    def test_windows_use_the_formula_and_the_step4_budget_fields(self):
        self.assertEqual((W2, W4), (8512592896, 4259830784))
        k2, k4 = model_memory.estimate_kv_cache_bytes(2, 128), model_memory.estimate_kv_cache_bytes(4, 128)
        survivor = model_memory.requirement(model_memory.SURVIVOR, 128)
        self.assertEqual(survivor['windows']['shrink']['total'], W4 + W2 + k4 + k2 + W2 + 2 * GIB)
        self.assertEqual(survivor['windows']['expand']['total'], W2 + W4 + k2 + k4 + W4 + 2 * GIB)
        self.assertEqual((survivor['binding_window'], survivor['required_free_bytes']), ('shrink', survivor['windows']['shrink']['total']))
        self.assertEqual(set(survivor['windows']['shrink']) - {'total'},
                         {'source_weights', 'target_weights', 'source_kv', 'target_kv', 'receive_buffers', 'temporary', 'margin'})
        self.assertEqual(model_memory.requirement(model_memory.REJOIN, 128)['required_free_bytes'], W4 + k4 + W4 + 2 * GIB)
        self.assertEqual(model_memory.requirement(model_memory.HEALTHY, 128)['required_free_bytes'], W4 + k4 + W4 + 2 * GIB)
        reference = model_memory.reference_requirements(128)
        self.assertEqual((reference['tp2']['gpus'], reference['tp2']['required_free_bytes_per_gpu']), (4, W2 + k2 + 2 * GIB))
        self.assertEqual((reference['tp4']['gpus'], reference['tp4']['required_free_bytes_per_gpu']), (8, W4 + k4 + 2 * GIB))

    def test_preflight_lists_all_sixteen_and_names_shortages(self):
        report = model_memory.preflight('SL3061', 128, range(8), ['tp2'], query=lambda _: inventory({2: 9 * GIB, 5: 9 * GIB}))
        self.assertEqual(len(report['scenario']), 16)
        local = [r for r in report['scenario'] if r['node'] == 'SL3061']
        self.assertEqual([r['role'] for r in local][:4], [model_memory.REJOIN] * 2 + [model_memory.SURVIVOR] * 2)
        self.assertTrue(all(r['sufficient'] is None and r['free_bytes'] is None for r in report['scenario'] if r['node'] == 'SL3060'))
        self.assertEqual([r['gpu_index'] for r in local if r['sufficient'] is False], [2, 5])
        self.assertTrue(all('GPU 2' in s or 'GPU 5' in s for s in report['scenario_shortages']))
        self.assertIn(model_memory.SURVIVOR, report['scenario_shortages'][0])
        self.assertFalse(report['scenario_sufficient_on_this_node'])
        self.assertEqual(len(report['run_shortages']), 1)
        self.assertIn('tp2 run: GPU 2', report['run_shortages'][0])
        text = model_memory.format_table(report)
        self.assertIn('NO', text)
        self.assertIn('reference run tp4: 8 GPUs', text)
        clean = model_memory.preflight('SL3060', 128, range(8), ['tp4'], query=lambda _: inventory())
        self.assertTrue(clean['scenario_sufficient_on_this_node'] and not clean['run_shortages'])
        with self.assertRaisesRegex(RuntimeError, 'eight GPUs'):
            model_memory.preflight('SL3060', 128, range(4), ['tp2'], query=lambda _: inventory()[:4])


def result(role, *, mode='deterministic', bitwise=True, ms=(30.0, 12.0)):
    manifest = DETERMINISTIC if mode == 'deterministic' else DEFAULT
    row = {'role': role, 'manifest': manifest, 'manifests_identical': True, 'history_sha256': 'h',
           'parameter_checksums': {'0': 'a', '1': 'b'},
           'timing_rank0': [{'layout': 'tp2', 'step': i, 'context': 32 + i, 'gpu_ms': m} for i, m in enumerate(ms)]}
    if role == 'candidate':
        row.update(bitwise=bitwise, first_mismatch=None if bitwise else {'rank': 1, 'step': 1}, candidate_kind='fresh_checkpoint_load')
    return row


class DriverTests(unittest.TestCase):
    def test_command_and_environment_per_role(self):
        reference = run_model.command('py', 'tp2', 'reference', '/ckpt', Path('/out/reference'), 32, 4, 128, 'suite-tp2')
        candidate = run_model.command('py', 'tp2', 'candidate', '/ckpt', Path('/out/candidate'), 32, 4, 128, 'suite-tp2',
                                      reference=Path('/out/reference'))
        timing = run_model.command('py', 'tp4', 'timing', '/ckpt', Path('/out/timing'), 32, 4, 128, 'suite-tp4')
        self.assertIn('--deterministic-mode', reference)
        self.assertIn('--deterministic-mode', candidate)
        self.assertNotIn('--deterministic-mode', timing)
        self.assertEqual(candidate[candidate.index('--mr-reference') + 1], str(Path('/out/reference')))
        self.assertNotIn('--mr-reference', reference)
        self.assertIn('--nproc_per_node=8', timing)
        self.assertIn('--live-router-mode', timing)
        inherited = {'NCCL_ALGO': 'Tree', 'CUBLAS_WORKSPACE_CONFIG': ':16:8', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
        with patch.dict(os.environ, inherited):
            deterministic = run_model.environment('reference', [0, 1, 2, 3], '/absent')
            default = run_model.environment('timing', [4, 5, 6, 7, 0, 1, 2, 3], '/absent')
        self.assertEqual((deterministic['NCCL_ALGO'], deterministic['CUBLAS_WORKSPACE_CONFIG'], deterministic['CUDA_VISIBLE_DEVICES']),
                         ('Ring', ':4096:8', '0,1,2,3'))
        self.assertTrue(all(key not in default for key in DETERMINISTIC_ENVIRONMENT))
        self.assertTrue(all('PYTORCH_CUDA_ALLOC_CONF' not in env for env in (deterministic, default)))
        self.assertEqual(default['CUDA_VISIBLE_DEVICES'], '4,5,6,7,0,1,2,3')

    def test_real_megatron_parse_args_accepts_the_entry_arguments(self):
        from megatron.training.arguments import parse_args

        command = run_model.command('py', 'tp2', 'candidate', '/ckpt', Path('/out/candidate'), 32, 4, 128, 'suite-tp2',
                                    reference=Path('/out/reference'))
        argv = command[command.index('tools/resharding/failstop/model_reference.py'):]
        with patch.object(sys, 'argv', argv):
            args = parse_args(model_reference.add_args)
        self.assertEqual((args.live_model_preset, args.live_router_mode, args.deterministic_mode), ('deepseek-v2-lite', 'native', True))
        self.assertEqual((args.tensor_model_parallel_size, args.expert_tensor_parallel_size, args.expert_model_parallel_size), (2, 2, 2))
        self.assertEqual((args.mr_role, args.mr_reference, args.mr_decode_steps, args.mr_request_id),
                         ('candidate', Path('/out/reference'), 4, 'suite-tp2'))
        self.assertEqual((args.load, args.ckpt_format, args.live_prompt_tokens), ('/ckpt', 'torch_dist', 32))

    def test_evaluate_and_summarize(self):
        results = {'reference': result('reference'), 'candidate': result('candidate'), 'timing': result('timing', mode='default', ms=(20.0, 10.0))}
        memory = {'scenario_sufficient_on_this_node': False, 'scenario_shortages': ['x'], 'run_shortages': []}
        evaluation = run_model.evaluate(results)
        self.assertTrue(evaluation['passed'])
        self.assertEqual(evaluation['step_time_ratio']['median_ratio'], 1.35)
        self.assertEqual(evaluation['manifests']['timing']['mode'], 'default')
        failed = run_model.evaluate({**results, 'candidate': result('candidate', bitwise=False)})
        self.assertEqual((failed['passed'], failed['first_mismatch']), (False, {'rank': 1, 'step': 1}))
        with self.assertRaisesRegex(RuntimeError, 'pair up'):
            run_model.evaluate({**results, 'timing': result('timing', mode='default', ms=(20.0,))})
        summary = run_model.summarize(['tp2'], {'tp2': evaluation}, memory, quick=True, correctness_only=True)
        self.assertTrue(summary['all_checks_passed'] and summary['quick_mode'] and summary['shared_gpu_mode'])
        self.assertFalse(summary['accepted'] or summary['timing_valid'])
        self.assertEqual(summary['deterministic_step_time_ratio']['tp2']['median_ratio'], 1.35)
        self.assertEqual((summary['plan_consistency']['status'], summary['memory_preflight']['scenario_shortages']), ('pending', ['x']))
        self.assertFalse(run_model.summarize(['tp2', 'tp4'], {'tp2': evaluation}, memory, quick=True,
                                             correctness_only=True)['all_checks_passed'])

    def test_dry_run_layouts_and_plan_comparison_cli(self):
        def dry(*argv):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = run_model.main(['--suite-id', 's5q_cpu', '--output', '/data/x', '--dry-run', *argv])
            return code, json.loads(output.getvalue())
        self.assertEqual(dry('--node', 'SL3061', '--quick')[1]['layouts'], ['tp2'])
        self.assertEqual(dry('--node', 'SL3060', '--quick')[1]['layouts'], ['tp4'])
        self.assertEqual(dry('--node', 'SL3061')[1]['layouts'], ['tp2', 'tp4'])
        self.assertEqual(dry('--node', 'SL3060', '--layouts', 'tp2')[1]['layouts'], ['tp2'])
        self.assertTrue(dry('--node', 'SL3061', '--quick')[1]['shared_gpu_mode'])
        with self.assertRaisesRegex(RuntimeError, 'layouts'):
            run_model.main(['--node', 'SL3061', '--suite-id', 's', '--output', '/data/x', '--dry-run', '--layouts', 'tp8'])
        with self.assertRaisesRegex(RuntimeError, 'gpus'):
            run_model.main(['--node', 'SL3061', '--suite-id', 's', '--output', '/data/x', '--dry-run', '--gpus', '0,1,2,3'])
        with self.assertRaisesRegex(RuntimeError, 'seq-length'):
            run_model.main(['--node', 'SL3061', '--suite-id', 's', '--output', '/data/x', '--dry-run', '--prompt-tokens', '128'])
        plan = {'tasks': [], 'profile_sha256': 'f' * 64, 'membership': [], 'epoch': 1, 'sources': [], 'targets': [],
                'environment': dict(DETERMINISTIC_ENVIRONMENT)}
        with tempfile.TemporaryDirectory() as temporary:
            for name in ('a', 'b'):
                Path(temporary, name).mkdir()
                Path(temporary, name, 'source_plan_0_1.json').write_text(json.dumps(plan))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_model.main(['--compare-plans', f'{temporary}/a', f'{temporary}/b']), 0)
                Path(temporary, 'b', 'source_plan_0_1.json').write_text(json.dumps({**plan, 'profile_sha256': 'e' * 64}))
                self.assertEqual(run_model.main(['--compare-plans', f'{temporary}/a', f'{temporary}/b']), 1)

    def test_shell_entry_maps_step5(self):
        script = Path(__file__).with_name('quick_check.sh').read_text(encoding='utf-8')
        self.assertIn('5) NAME=step05_model_correctness', script)


if __name__ == '__main__':
    unittest.main()
