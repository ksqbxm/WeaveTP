"""Like the servers: the supervisor's own worker command lines, parsed by the worker scripts' real argparse and the real
Megatron parse in MegatronRuntime.from_config (no model is built), for every method x scenario x mode x run mode the
entry points accept; and every failstop module imported in a fresh interpreter whose sys.path holds only the failstop
directory (no PYTHONPATH), as the controller, supervisor and worker processes start."""

import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import common
import model_worker
import run_failstop_model
import serving_config
import serving_worker
import supervisor
import torch
from test_probe import inventory

from megatron.training.global_vars import unset_global_variables
from tools.resharding.e2e.toy_inputs import write_toy_inputs

FAILSTOP = Path(__file__).resolve().parent
GPU = SimpleNamespace(major=12, minor=0, name='NVIDIA GeForce RTX 5090', total_memory=32 << 30)  # what each worker sees
SCRIPTS = {'serving_worker.py': (serving_worker, 'ServingWorker'), 'model_worker.py': (model_worker, 'Worker')}


class WorkerCommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.rows = common.topology(inventory())
        self.victims = common.failed_slots(self.rows)
        (self.root / 'profile.json').write_text('{}')
        for tp in (2, 4):
            (self.root / 'ref' / f'tp{tp}' / 'reference').mkdir(parents=True)
            for rank in range(2 * tp):
                (self.root / 'ref' / f'tp{tp}' / 'reference' / f'parameters_rank_{rank}.json').write_text('{}')
        self.inputs = {failstop: write_toy_inputs(self.root / f'inputs_{failstop}', count=8, failstop=failstop) for failstop in (True, False)}

    def tearDown(self):
        unset_global_variables()
        self.directory.cleanup()

    def serving_configs(self):
        """Every combination serving_config (and the fail-slow settings) accepts, built per node as the launchers do."""
        configs = {}
        for label, scenario, mode, run_mode in itertools.product(('weavetp', 'anchortp', 'restart', 'ignore', 'nofault'),
                                                                 serving_config.SCENARIOS, ('emulate', 'real', 'kill'), serving_config.RUN_MODES):
            try:
                failslow = serving_config.failslow_settings(method=label, level=None if label == 'nofault' else 'W', fault_gpus=[0, 1, 4, 5],
                                                            warmup_s=1.0, slow_s=2.0, tail_s=1.0, detect_s=1.0, mode=mode) \
                    if scenario == 'failslow' else None
                plan = self.inputs[scenario == 'failstop']
                with patch.object(serving_config, 'pidfd_implementations', return_value={}):
                    configs[label, scenario, mode, run_mode] = {node: serving_config.serving_config(
                        node=node, launch=f'{label}_{scenario}_{mode}_{run_mode}', mode=mode, output=self.root / label / scenario / mode / run_mode / node,
                        master='10.60.14.1', port=29860, commit='c' * 40, remaining=8000, min_free_disk_gb=1, quick=False,
                        shared_gpu=True, run_mode=run_mode, method='ignore' if label == 'nofault' else label, scenario=scenario,
                        checkpoint='/data/models/DeepSeek-V2-Lite-megatron-v2', seq_length=2048, profile_file=self.root / 'profile.json',
                        reference_root=self.root / 'ref', requests=plan['requests'], workload=plan['workload'], plan=plan['plan'],
                        b_max=8, capacity_tokens=None, failslow=failslow) for node in common.NODES}
            except RuntimeError:  # a combination the entry points refuse
                continue
        return configs

    def commands(self, config):
        """The supervisor's worker command lines on config['node']: the initial workers and, for real and kill fail-stop,
        the replacements (start_workers, with spawning recorded instead of started)."""
        config = {**config, 'topology': self.rows}
        Path(config['output']).mkdir(parents=True)
        common.write_json(Path(config['output']) / 'worker_config.json', config)  # as Supervisor.run writes it
        sup = supervisor.Supervisor(config, Mock(), Mock())
        commands = []

        def spawn(argv, role, gpu_uuid, env, replacement_for=None):
            commands.append(argv)
            child = Mock(pid=1000 + len(commands))
            child.poll.return_value = -9
            return child, {'pid': child.pid}
        with patch.object(sup.children, 'spawn', side_effect=spawn), patch.object(supervisor.signal, 'SIGKILL', 9, create=True):
            sup.start_workers([r['slot'] for r in self.rows if r['node'] == config['node']], 0)
            if config['node'] == 'SL3061' and config['mode'] != 'emulate' and common.releases_fault_set(config):
                sup.state, sup.killed = 'reclaimed', {slot: True for slot in self.victims}
                sup.start_workers(self.victims, 2)
        return config, commands

    def run_worker(self, argv, config):
        """The worker script's main() on this command line, in the environment its launcher sets for the run mode
        (serving_config.apply_environment): its argparse, the config it reads and the real Megatron parse of
        config['model_argv']. The worker object is recorded instead of run."""
        module, name = SCRIPTS[Path(argv[1]).name]
        built = []

        def record(config, slot, epoch, runtime):
            built.append((slot, epoch, runtime))
            return Mock()
        unset_global_variables()
        with patch.dict(os.environ), patch.object(sys, 'argv', argv[1:]), patch.object(torch.cuda, 'get_device_properties', return_value=GPU), \
                patch.object(module, name, side_effect=record), \
                patch.object(module, 'connect', side_effect=RuntimeError('a failing worker reports over the bus')):
            serving_config.apply_environment('correctness' if config['deterministic_mode'] else 'timing')
            module.main()
        return built[0]

    def check(self, config, commands, script):
        self.assertTrue(commands)
        groups = {}
        for argv in commands:  # one command line per worker: the script, the config and the slot/epoch pair
            self.assertEqual((Path(argv[1]).name, argv[2:4], argv[4], argv[6]),
                             (script, ['--config', str(Path(config['output']) / 'worker_config.json')], '--slot', '--epoch'))
            groups.setdefault(argv[7], []).append(argv)
        for epoch, members in groups.items():  # within an epoch the command lines differ only in --slot
            self.assertEqual({tuple(a[:5] + a[6:]) for a in members}, {tuple(members[0][:5] + members[0][6:])})
            slot, parsed_epoch, runtime = self.run_worker(members[0], config)
            self.assertEqual((slot, parsed_epoch), (int(members[0][5]), int(epoch)))
            args = runtime.args
            self.assertEqual((args.tensor_model_parallel_size, args.expert_model_parallel_size, args.world_size, args.deterministic_mode),
                             (4, 2, len(self.rows), config['deterministic_mode']))
        return groups

    def test_every_serving_combination_parses_on_every_worker(self):
        configs = self.serving_configs()
        accepted = {(label, scenario) for label, scenario, _, _ in configs}
        self.assertEqual(len(configs), 30)  # failstop 12 (restart: emulate only; kill: weavetp only), nofault 8, failslow 10 (emulate only)
        self.assertEqual(accepted, {(m, 'failstop') for m in ('weavetp', 'anchortp', 'restart')} | {(m, 'nofault') for m in ('weavetp', 'anchortp')} |
                         {(m, 'failslow') for m in ('weavetp', 'anchortp', 'restart', 'ignore', 'nofault')})
        replacements = 0
        for key, per_node in configs.items():
            for node, built in per_node.items():
                with self.subTest(combination=key, node=node):
                    config, commands = self.commands(built)
                    groups = self.check(config, commands, 'serving_worker.py')
                    replacements += len(groups.get('2', []))
        # the real fail-stop cells (weavetp/anchortp x correctness/timing) and the kill ones (weavetp) replace the victims
        self.assertEqual(replacements, 6 * 4)

    def test_step5_model_workers_parse(self):
        for mode in ('emulate', 'real'):
            with self.subTest(mode=mode):
                config = {**{k: v for k, v in common.DEFAULTS.items() if k != 'repeats'}, 'probe': common.MODEL_PROBE, 'mode': mode,
                          'node': 'SL3061', 'launch': f's5_{mode}', 'output': str(self.root / 's5' / mode), 'mps_pipe': None,
                          'deterministic_mode': True,
                          'model_argv': run_failstop_model.worker_arguments('/data/models/DeepSeek-V2-Lite-megatron-v2', 8, 2048)}
                config, commands = self.commands(config)
                groups = self.check(config, commands, 'model_worker.py')
                self.assertEqual(sorted(groups), ['0', '2'] if mode == 'real' else ['0'])

    def test_the_reference_tool_arguments_are_not_required(self):
        # The workers once parsed model_argv with model_reference's provider, whose --mr-output/--mr-role are required:
        # every real-model worker exited with "the following arguments are required: --mr-output, --mr-role".
        config = self.serving_configs()['weavetp', 'nofault', 'emulate', 'timing']['SL3060']
        self.assertFalse(any(a.startswith('--mr-') for a in config['model_argv']))
        config, commands = self.commands(config)
        self.assertEqual(self.run_worker(commands[0], config)[2].args.world_size, 16)


class IsolatedImportTests(unittest.TestCase):
    def test_every_module_imports_with_only_the_failstop_directory_on_sys_path(self):
        names = sorted(p.stem for p in FAILSTOP.glob('*.py') if not p.stem.startswith('test_'))
        probe = 'import importlib, sys; sys.path.insert(0, sys.argv[1]); importlib.import_module(sys.argv[2])'
        with tempfile.TemporaryDirectory() as cwd:  # -I: no PYTHONPATH, no user site, no script/cwd entry on sys.path

            def run(name):
                done = subprocess.run([sys.executable, '-I', '-W', 'ignore', '-c', probe, str(FAILSTOP), name], cwd=cwd,
                                      capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
                return name, done.returncode, done.stderr.strip().splitlines()[-1:] if done.returncode else []
            with ThreadPoolExecutor(max_workers=8) as pool:
                failed = {name: error for name, code, error in pool.map(run, names) if code}
        self.assertIn('failslow_probe', names)
        self.assertEqual(failed, {})


if __name__ == '__main__':
    unittest.main()
