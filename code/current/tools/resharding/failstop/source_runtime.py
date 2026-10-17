"""Step 3 checks attached to the existing Step 2 fault/rebuild protocol."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from common import append_json, require, write_json
from source_fixture import NAMES, read_oracle
from source_profile import collect

from megatron.core.failstop import make_copy_service
from megatron.core.resharding.host_weights import HostAttachment, probe_registration
from megatron.core.resharding.mixed_sources import Shard, build_mixed_plan, digest, normalized
from megatron.core.resharding.mixed_transport import execute


def identity(row):
    return f"{row['node']}/{row['gpu_uuid']}"


def shard(row, name, expert, start, stop, width, layout, tensor, *, host=False):
    return Shard(f"{row['node']}/host" if host else identity(row), row['node'],
                 None if host else row['new_rank'], 'host' if host else 'gpu', tensor,
                 name, expert, 'int64', 8, (2, width), (0, start), (2, stop), layout)


def snapshot(mapping, phase, width, host_reports):
    sources, targets = [], []
    for row in mapping:
        if not row['active']:
            continue
        for name in NAMES:
            # Expansion uses actual TP2 sources; returned empty TP4 ranks are excluded.
            old_size = 2 if phase == 2 and row['replica'] == 'EDP0' else 4
            if phase == 2 and row['replica'] == 'EDP0' and row['source_etp'] < 2:
                has_source = False
            else:
                has_source = True
            if has_source and name != 'host_only':
                # source_tp2 is recorded at the completed shrink, independent of new rank.
                tp = row.get('source_tp2', row['source_etp']) if old_size == 2 else row['source_etp']
                for part in range(tp * (8 // old_size), (tp + 1) * (8 // old_size)):
                    if name == 'mixed' and part % 2 == 0:
                        continue
                    sources.append(shard(row, name, row['ep'], part * width // 8,
                                         (part + 1) * width // 8, width, f'TP{old_size}',
                                         f'{name}_{row["ep"]}_{part}'))
            if row['replica'] == 'EDP0':
                size = 2 if phase == 1 else 4
                targets.append(shard(row, name, row['ep'], row['etp'] * width // size,
                                     (row['etp'] + 1) * width // size, width, f'TP{size}', f'{name}_target'))
    for node, report in host_reports.items():
        for entry in report['entries']:
            # Eligibility forces a complementary host/GPU test with measured costs
            # inside each eligible set, even when full backing mode is selected.
            if entry['name'] == 'gpu_only' or (entry['name'] == 'mixed' and entry['part'] % 2):
                continue
            sources.append(shard({'node': node}, entry['name'], entry['expert'], entry['start'][1],
                                 entry['stop'][1], width, 'host', entry['id'], host=True))
    return sources, targets


class SourceRuntime:
    def __init__(self, config, slot, epoch, bus, journal):
        self.config, self.slot, self.bus, self.journal = config, slot, bus, journal
        self.row = config['topology'][slot]
        self.device = identity(self.row)
        self.root = Path(config['output'])
        self.width = config['source_width']
        bus.put(f'host_registration/{epoch}/{slot}', journal.event('host_registration_preflight', **probe_registration()))
        self.reports = dict(zip(('SL3060', 'SL3061'), bus.wait(['host_ready/SL3060', 'host_ready/SL3061'])))
        report = self.reports[config['node']]
        self.structure = {e['name']: e for e in report['structure']}
        require(set(self.structure) == set(NAMES) and all(e['shape'] == [2, self.width]
                for e in self.structure.values()), 'host structure/config mismatch')
        self.host = HostAttachment(report, record=lambda **r: append_json(
            self.root / f'host_worker_{slot}_{epoch}.jsonl', r))
        self.sources, self.targets, self.tp2 = {}, {}, {}
        self.initial_h2d_bytes = 0
        if epoch == 0:
            for entry in json.loads(Path(report['manifest_path']).read_text())['entries']:
                if entry['expert'] == self.row['ep'] and entry['part'] // 2 == self.row['etp']:
                    if entry['name'] == 'host_only' or (entry['name'] == 'mixed' and entry['part'] % 2 == 0):
                        continue
                    cpu = read_oracle(report['manifest_path'], entry['name'], entry['expert'],
                                      entry['start'][1], entry['stop'][1])
                    storage = torch.empty((2, cpu.shape[1] * 2), dtype=torch.int64, device='cuda')
                    self.sources[entry['id']] = storage[:, ::2]
                    self.sources[entry['id']].copy_(cpu)
                    self.initial_h2d_bytes += cpu.numel() * cpu.element_size()
        # Replacement initialization reads structure only; its target stays poison.
        self.allocate_targets(4)
        journal.event('source_initialization', initial_h2d_bytes=self.initial_h2d_bytes,
                      empty_receiver=epoch == 2, host_backing=report['backing'], owner_pid=report['owner_pid'])
        self.profile = None

    def allocate_targets(self, size):
        if self.row['replica'] == 'EDP0':
            self.targets = {f'{name}_target': torch.full((entry['shape'][0], 2 * entry['shape'][1] // size), -1,
                           dtype=getattr(torch, entry['dtype']), device='cuda')[:, ::2]
                            for name, entry in self.structure.items()}

    def release(self):
        self.sources.clear()
        self.targets.clear()
        torch.cuda.synchronize()

    def validate(self, session, group, phase, round_id):
        if phase == 0:
            if self.profile is None:
                measured = collect(self.config, self.config['topology'], self.slot, group, self.host, self.journal)
                # Canonical profile includes controller-node clock only. All plans
                # consume exactly one published payload, not per-node metadata.
                if self.slot == 0:
                    self.bus.put('source_profile', measured)
                self.profile = self.bus.wait(['source_profile'])[0]
                if self.slot in (0, 8):
                    (self.root / 'profile.json').write_text(json.dumps(self.profile, sort_keys=True,
                        separators=(',', ':')), encoding='utf-8')
                    (self.root / 'profile.sha256').write_text(digest(self.profile) + '  profile.json\n')
            return {'profile_sha256': digest(self.profile), 'initial_h2d_bytes': self.initial_h2d_bytes,
                    'owner_pid': self.host.report['owner_pid'], 'backing': self.host.report['backing']}
        if self.profile is None:
            self.profile = self.bus.wait(['source_profile'])[0]
        mapping = session.rank_map()
        if phase == 2:
            self.tp2 = self.bus.wait([f'source_tp2/{round_id}'])[0]
            mapping = [{**r, 'source_tp2': self.tp2.get(str(r['slot']))} for r in mapping]
        sources, targets = snapshot(mapping, phase, self.width, self.reports)
        membership = [identity(r) for r in mapping if r['active']]
        rank_map = {identity(r): r['new_rank'] for r in mapping if r['active']}
        tasks, cache_key, loads = build_mixed_plan(sources, targets, self.profile,
                                                  membership=membership, epoch=session.epoch, rank_map=rank_map)
        session.resources.plans[cache_key] = tasks
        # Fixed snapshot replay under compressed and original ranks checks every
        # source/target/slice/wave, independently of run-time profile differences.
        alternate = {identity(r): r['slot'] if session.backend.mode == 'real' else i
                     for i, r in enumerate(r for r in mapping if r['active'])}
        replay, _, _ = build_mixed_plan(
            [replace(s, rank=alternate[s.device]) if s.kind == 'gpu' else s for s in sources],
            [replace(t, rank=alternate[t.device]) for t in targets], self.profile,
            membership=membership, epoch=session.epoch, rank_map=alternate)
        require(normalized(tasks) == normalized(replay), 'normalized mode plans differ')
        if self.slot == 0:
            write_json(self.root / f'source_plan_{round_id}_{phase}.json', {
                'tasks': [asdict(t) for t in tasks], 'cache_key': cache_key,
                'profile_sha256': digest(self.profile), 'normalized_replay_equal': True,
                'loads': [{'resource': k, 'seconds': v} for k, v in loads.items()]})
        self.allocate_targets(2 if phase == 1 else 4)
        actual = dist.get_process_group_ranks(group)

        def record_link(record):
            append_json(self.root / f'links_{self.slot}.jsonl', {
                **record, 'timing_endpoint': 'destination', 'node': self.config['node'],
                'clock_offset_s': self.config['clock']['offset'], 'clock_error_s': self.config['clock']['error']})

        stats = execute(tasks, device=self.device, sources=self.sources, targets=self.targets,
                        host=self.host, service_factory=lambda: make_copy_service(session, self.slot, group, p2p_order='task-round'),
                        rank_to_group={r: i for i, r in enumerate(actual)}, event='shrink' if phase == 1 else 'expand',
                        round_id=round_id, epoch=session.epoch, resources=session.resources,
                        record=record_link)
        row = mapping[self.slot]
        size = 2 if phase == 1 else 4
        if self.row['replica'] == 'EDP0':
            start, stop = row['etp'] * self.width // size, (row['etp'] + 1) * self.width // size
            for name in NAMES:
                expected = read_oracle(self.host.report['manifest_path'], name, row['ep'], start, stop)
                require(torch.equal(self.targets[f'{name}_target'].cpu(), expected), f'{name} checkpoint bitwise mismatch')
            expected_bytes = len(NAMES) * 2 * (stop - start) * 8
            require(stats['payload_bytes'] + stats['reuse_bytes'] == expected_bytes, 'link payload coverage mismatch')
            # Verified output becomes the source for expansion/next round.
            self.sources = {f'{name}_{row["ep"]}_{part}': self.targets[f'{name}_target'][:,
                            part * self.width // 8 - start:(part + 1) * self.width // 8 - start]
                            for name in ('gpu_only', 'mixed')
                            for part in range(row['etp'] * (8 // size), (row['etp'] + 1) * (8 // size))
                            if name != 'mixed' or part % 2}
        if phase == 1 and self.slot == 0:
            self.bus.put(f'source_tp2/{round_id}', {str(r['slot']): r['etp'] for r in mapping
                                                 if r['active'] and r['replica'] == 'EDP0'})
        return {**stats, 'bitwise': True, 'normalized_replay_equal': True,
                'profile_sha256': digest(self.profile), 'owner_pid': self.host.report['owner_pid'],
                'backing': self.host.report['backing'], 'backing_inode': self.host.report['backing_inode'],
                'initial_h2d_bytes': self.initial_h2d_bytes}

    def close(self):
        self.sources.clear()
        self.targets.clear()
        self.host.close()
