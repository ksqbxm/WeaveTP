"""CPU contracts of the router determinism check; the GPU forward itself is only run on the server."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from tools.resharding.correctness import router_determinism as gpu
from tools.resharding.correctness import router_determinism_driver as driver


class EstimateTests(unittest.TestCase):
    def test_unsharded_total_matches_deepseek_v2_lite_parameter_count(self):
        # Official DeepSeek-V2-Lite: 15.7B parameters; this formula is checked exactly on GPU.
        total = driver.estimate_rank_parameter_bytes(1, 1, 1) // 2
        self.assertEqual(total, 15_706_484_224)

    def test_sharding_divides_dense_and_expert_parts_separately(self):
        full = driver.estimate_rank_parameter_bytes(1, 1, 1)
        tp2 = driver.estimate_rank_parameter_bytes(2, 2, 2)
        tp4 = driver.estimate_rank_parameter_bytes(4, 4, 2)
        self.assertLess(tp4, tp2)
        self.assertLess(tp2, full)
        # Eight TP4/EP2 ranks hold every expert once, dense shards twice and replicated tensors eight times.
        replicated = (26 * 64 * 2048 + 27 * (2048 + 512 + 2048) + 2048) * 2
        experts = 26 * 64 * 3 * 2048 * 1408 * 2
        self.assertEqual(tp4 * 8, experts + 2 * (full - experts - replicated) + 8 * replicated)
        self.assertEqual(driver.estimate_kv_cache_bytes(2, 128), 27 * 128 * 8 * 320 * 2)

    def test_plan_lists_requirement_per_gpu_and_unestimated_items(self):
        plan = driver.plan_memory('tp4', 128)
        self.assertEqual(plan['gpus'], 8)
        self.assertEqual(plan['required_free_bytes_per_gpu'],
                         plan['weight_bytes_per_gpu'] + plan['kv_cache_bytes_per_gpu'] + (2 << 30))
        self.assertIn('cuBLAS workspaces', plan['not_estimated_measured_after_run'])


class PreflightTests(unittest.TestCase):
    def test_shortage_names_gpu_and_requirement(self):
        rows = '\n'.join(f'{i}, GPU-{i}, 32607, {free}' for i, free in enumerate([30000, 30000, 3000, 30000] + [30000] * 4))
        with tempfile.TemporaryDirectory() as temporary, patch.object(subprocess, 'check_output', return_value=rows):
            Path(temporary, 'latest_checkpointed_iteration.txt').write_text('1\n')
            report = driver.preflight(list(range(8)), ['tp2', 'tp4'], 128, temporary)
        self.assertEqual(len(report['shortages']), 2)
        self.assertTrue(all('GPU 2' in s for s in report['shortages']))
        self.assertFalse(report['layouts']['tp2']['gpus_used'][2]['sufficient'])
        with self.assertRaisesRegex(RuntimeError, 'checkpoint not found'):
            driver.preflight([0], ['tp2'], 128, '/nonexistent')


class CompareTests(unittest.TestCase):
    def test_first_mismatch_reports_bits_of_one_ulp(self):
        a = torch.full((2, 3), 1.0, dtype=torch.bfloat16)
        b = a.clone()
        b.view(torch.int16).view(-1)[5] += 1  # exactly one ULP in bf16
        self.assertIsNone(gpu.first_mismatch(a, a.clone()))
        mismatch = gpu.first_mismatch(a, b)
        self.assertEqual((mismatch['index'], mismatch['count'], mismatch['reference']['bits']), ([1, 2], 1, '0x3f80'))
        self.assertEqual(mismatch['candidate']['bits'], '0x3f81')
        self.assertEqual(gpu.first_mismatch(a, a.float())['kind'], 'shape/dtype')

    def test_compare_runs_orders_logits_before_routing_and_names_layer(self):
        def run(flip_layer=None, flip_logits=False):
            records = []
            for step in range(2):
                routing_map = torch.zeros(4, 8, dtype=torch.bool)
                routing_map[:, :2] = True
                if flip_layer is not None and step == 1:
                    routing_map[0, 7] = True
                logits = torch.zeros(1, 1, 16, dtype=torch.bfloat16)
                if flip_logits and step == 1:
                    logits[0, 0, 3] = 1
                records.append({'logits': logits, 'routing': {5: {'routing_map': routing_map, 'probs': torch.zeros(4, 8, dtype=torch.bfloat16)}}})
            return records
        self.assertIsNone(gpu.compare_runs(run(), run()))
        self.assertEqual(gpu.compare_runs(run(), run(flip_layer=5))['tensor'], 'layer5.routing_map')
        self.assertEqual(gpu.compare_runs(run(), run(flip_logits=True, flip_layer=5))['tensor'], 'logits')
        self.assertEqual(gpu.coverage(run(flip_layer=5)), {5: [0, 1, 7]})


class SummaryTests(unittest.TestCase):
    def result(self, default_bitwise, deterministic_bitwise, match=True, error=None):
        return {'parameter_bytes_match': match, 'modes': {
            'default': {'bitwise': default_bitwise, 'first_mismatch': None if default_bitwise else
                        {'tensor': 'logits', 'step': 1, 'rank': 2}},
            'deterministic': {'bitwise': deterministic_bitwise, 'error': error, 'first_mismatch': None if deterministic_bitwise or error else
                              {'tensor': 'layer3.probs', 'step': 0, 'rank': 0}}}}

    def test_pass_requires_deterministic_mode_and_reports_default_mode(self):
        modes = ['default', 'deterministic']
        summary = driver.summarize(['tp2', 'tp4'], {'tp2': self.result(False, True), 'tp4': self.result(True, True)}, modes)
        self.assertEqual(summary['result'], 'PASS')
        self.assertIn('default=differs(logits step=1 rank=2)', summary['layouts']['tp2']['verdict'])
        self.assertFalse(summary['layouts']['tp2']['default_mode_bitwise'])
        self.assertEqual(driver.summarize(['tp2', 'tp4'], {'tp2': self.result(True, True)}, modes)['result'], 'FAIL')
        self.assertEqual(driver.summarize(['tp2'], {'tp2': self.result(True, False)}, modes)['result'], 'FAIL')
        self.assertEqual(driver.summarize(['tp2'], {'tp2': self.result(True, True, match=False)}, modes)['result'], 'FAIL')
        failed = driver.summarize(['tp2'], {'tp2': self.result(True, False, error='RuntimeError: scatter_add_cuda_kernel does not have a deterministic implementation')}, modes)
        self.assertEqual(failed['result'], 'FAIL')
        self.assertIn('scatter_add_cuda_kernel', failed['layouts']['tp2']['verdict'])
        self.assertEqual(driver.summarize(['tp2'], {'tp2': {'parameter_bytes_match': True, 'modes': {'default': {'bitwise': True, 'first_mismatch': None}}}}, ['default'])['result'], 'PASS')


class CommandTests(unittest.TestCase):
    def test_torchrun_command_pins_layout_native_router_and_checkpoint(self):
        command = driver.torchrun_command('py', 'tp4', [0, 1, 2, 3, 4, 5, 6, 7], '/ckpt', Path('/out'), 32, 4, 3, ['default'], 128)
        text = ' '.join(command)
        self.assertIn('--nproc_per_node=8', text)
        self.assertIn('--tensor-model-parallel-size 4 --expert-tensor-parallel-size 4 --expert-model-parallel-size 2', text)
        self.assertIn('--global-batch-size 2', text)
        self.assertIn('--live-router-mode native', text)
        self.assertIn('--load /ckpt --ckpt-format torch_dist', text)
        self.assertNotIn('--live-active-experts', text)
        env = driver.torchrun_env([0, 1, 2, 3], '/absent')
        self.assertEqual((env['CUDA_VISIBLE_DEVICES'], env['NCCL_ALGO'], env['CUBLAS_WORKSPACE_CONFIG']), ('0,1,2,3', 'Ring', ':4096:8'))

    def test_shell_entry_reads_pinned_sha_from_releases(self):
        script = Path(__file__).with_name('router_determinism_check.sh').read_text(encoding='utf-8')
        self.assertIn('["router_determinism"]', script)
        self.assertIn('router_determinism_driver.py', script)
        self.assertIn('nohup', script)


if __name__ == '__main__':
    unittest.main()
