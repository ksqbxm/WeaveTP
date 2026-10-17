"""Actual mixed planner/transport/controller on CPU with concurrent CUDA doubles.

The profile is synthetic and registration/NCCL are substituted. This validates
protocol and bytes, never GPU feasibility, bandwidth or timing performance.
"""

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import source_runtime
import torch
from source_fixture import create_checkpoint, demand
from test_sources import CAPACITY, profile
from test_topology_protocol import Simulation

from megatron.core.resharding.host_weights import HostAttachment, HostOwner


class Stream:
    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


class SourceProtocolTests(unittest.TestCase):
    def run_case(self, mode, rounds, library_mode):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            sim = Simulation(mode, root, rounds, {'probe': 'step03_sources_and_transport', 'source_width': 64})
            sim.cuda_stream = Stream()
            owners = []
            for node in ('SL3060', 'SL3061'):
                node_root = root / node
                node_root.mkdir()
                manifest = create_checkpoint(node_root / 'checkpoint', node, 64)
                owner = HostOwner(manifest, node_root / 'backing', library_mode,
                                  demand(json.loads(manifest.read_text())['entries']), capacity=CAPACITY)
                stack.callback(owner.close)
                owner.report['manifest_path'] = str(manifest)
                sim.put(f'host_ready/{node}', owner.report)
                owners.append(owner)
            measured = profile([source_runtime.identity(row) for row in sim.rows])
            attachments = []

            def attach(report, **kwargs):
                result = HostAttachment(report, register=False, **kwargs)
                attachments.append(result)
                return result

            stack.enter_context(patch.object(source_runtime, 'collect', return_value=measured))
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

    def test_emulate_two_rounds_demand(self):
        self.run_case('emulate', 2, 'on-demand')

    def test_real_replacements_full_library(self):
        self.run_case('real', 1, 'full')


if __name__ == '__main__':
    unittest.main()
