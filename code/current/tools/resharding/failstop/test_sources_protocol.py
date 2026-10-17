"""Actual mixed planner/transport/controller on CPU with concurrent CUDA doubles.

The profile is synthetic and registration/NCCL are substituted. This validates
protocol and bytes, never GPU feasibility, bandwidth or timing performance.
"""

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import source_runtime
import torch
from common import DETERMINISTIC_ENVIRONMENT, TOPOLOGY_PROBES
from determinism import compare_launch_plans
from source_fixture import create_checkpoint, demand
from test_sources import CAPACITY, profile
from test_topology_protocol import Simulation

from megatron.core.resharding.host_weights import HostAttachment, HostOwner
from megatron.core.resharding.mixed_sources import digest


class Stream:
    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


class SourceProtocolTests(unittest.TestCase):
    def run_case(self, mode, rounds, library_mode, frozen=False):
        """Returns slot 0's plan files so launches can be compared after the directory is gone."""
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            sim = Simulation(mode, root, rounds, {'probe': TOPOLOGY_PROBES[3], 'source_width': 64,
                                                  'host_library_mode': library_mode,
                                                  'profile_file': None, 'profile_sha256': None})
            sim.cuda_stream = Stream()
            owners = []
            for node in ('SL3060', 'SL3061'):
                node_root = root / node
                node_root.mkdir()
                manifest = create_checkpoint(node_root / 'checkpoint', node, 64)
                owner = HostOwner(manifest, node_root / 'backing', library_mode,
                                  demand(json.loads(manifest.read_text())['entries']), capacity=CAPACITY)
                stack.callback(owner.close)
                sim.put(f'host_ready/{node}', owner.report)
                owners.append(owner)
            measured = profile([source_runtime.identity(row) for row in sim.rows])
            if frozen:
                # A frozen profile replaces measurement entirely; the collector must not run.
                (root / 'profile.json').write_text(json.dumps(measured))
                sim.extra_config.update(profile_file=str(root / 'profile.json'), profile_sha256=digest(measured))
            attachments = []

            def attach(report, **kwargs):
                result = HostAttachment(report, register=False, **kwargs)
                attachments.append(result)
                return result

            stack.enter_context(patch.object(source_runtime, 'collect',
                                             side_effect=RuntimeError('collect called with a frozen profile') if frozen else None,
                                             return_value=measured))
            stack.enter_context(patch.object(source_runtime, 'probe_registration', return_value={'cpu_double': True}))
            stack.enter_context(patch.object(source_runtime, 'HostAttachment', side_effect=attach))
            stack.enter_context(patch.object(source_runtime, 'make_copy_service', sim.copy_service))
            stack.enter_context(patch.object(torch.cuda, 'Stream', Stream))
            stack.enter_context(patch.object(torch.cuda, 'stream', lambda _: contextlib.nullcontext()))
            try:
                sim.run()
            finally:
                # A real killed process releases its mmap in the kernel; the CPU
                # actor replacement must release that attachment explicitly.
                for attachment in attachments:
                    if not attachment.mapping.closed:
                        attachment.close()
            reports = json.loads((root / 'topology_checks.json').read_text())
            for phase, values in reports.items():
                if phase.endswith('/0'):
                    continue
                for report in values:
                    self.assertTrue(report['sources']['bitwise'])
                    self.assertTrue(report['sources']['normalized_replay_equal'])
            for owner in owners:
                self.assertTrue(owner.path.exists())
            links = [json.loads(line) for file in root.glob('*/links_*.jsonl') for line in file.read_text().splitlines()]
            payloads = [r for r in links if r['source_kind'] != 'control']
            self.assertEqual({r['source_kind'] for r in payloads}, {'host', 'gpu', 'local'})
            self.assertTrue(all(r['completed_local'] >= r['submitted_local'] and
                                r['completion_granularity'] == 'batch' for r in payloads))
            self.assertEqual(sum(r['payload_bytes'] for r in payloads), rounds * 2 * 3 * 2 * 2 * 64 * 8)
            metrics = sim.get('topology_metrics')
            self.assertGreaterEqual(metrics['edp1_rejoin_pause_s'], 0)
            plans = {p.name: json.loads(p.read_text()) for p in (root / 'SL3060').glob('source_plan_*.json')}
            self.assertEqual(set(plans), {f'source_plan_{r}_{p}' + '.json' for r in range(rounds) for p in (1, 2)})
            for plan in plans.values():
                self.assertEqual(plan['profile_source'], 'frozen_file' if frozen else 'measured')
                self.assertEqual(plan['profile_sha256'], digest(measured))
                self.assertTrue(plan['sources'] and plan['targets'] and plan['membership'])
                self.assertEqual(set(plan['environment']), set(DETERMINISTIC_ENVIRONMENT))
            return plans

    def test_emulate_two_rounds_demand(self):
        self.run_case('emulate', 2, 'on-demand')

    def test_real_replacements_full_library(self):
        self.run_case('real', 1, 'full')

    def test_frozen_profile_launches_plan_identically_across_modes(self):
        """Step 5 §2.1: the same snapshot over one frozen profile plans the same under either mode environment."""
        with patch.dict(os.environ, DETERMINISTIC_ENVIRONMENT):
            deterministic = self.run_case('emulate', 1, 'on-demand', frozen=True)
        with patch.dict(os.environ, {}, clear=False):
            for key in DETERMINISTIC_ENVIRONMENT:
                os.environ.pop(key, None)
            default = self.run_case('emulate', 1, 'on-demand', frozen=True)
        self.assertEqual(deterministic['source_plan_0_1.json']['environment'], DETERMINISTIC_ENVIRONMENT)
        self.assertEqual(default['source_plan_0_1.json']['environment'], {key: None for key in DETERMINISTIC_ENVIRONMENT})
        with tempfile.TemporaryDirectory() as directory:
            roots = {}
            for name, plans in (('default', default), ('deterministic', deterministic)):
                roots[name] = Path(directory, name)
                roots[name].mkdir()
                for file, plan in plans.items():
                    (roots[name] / file).write_text(json.dumps(plan))
            result = compare_launch_plans(roots['default'], roots['deterministic'])
            self.assertTrue(result['equal'], result)
            self.assertTrue(all(all(p['inputs_equal'].values()) for p in result['plans'].values()))
            broken = json.loads((roots['deterministic'] / 'source_plan_0_2.json').read_text())
            broken['tasks'][0]['payload_bytes'] += 8
            (roots['deterministic'] / 'source_plan_0_2.json').write_text(json.dumps(broken))
            result = compare_launch_plans(roots['default'], roots['deterministic'])
            self.assertFalse(result['equal'])
            self.assertEqual(result['plans']['source_plan_0_2.json']['first_difference']['fields'], ['payload_bytes'])
            self.assertTrue(result['plans']['source_plan_0_1.json']['equal'])


if __name__ == '__main__':
    unittest.main()
