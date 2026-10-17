"""Step 3 checks attached to the existing Step 2 fault/rebuild protocol."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from common import DETERMINISTIC_ENVIRONMENT, NODES, append_json, host_backing, require, write_json
from source_fixture import NAMES, host_resident, identity, read_oracle
from source_profile import collect

from megatron.core.failstop import make_copy_service
from megatron.core.resharding.host_weights import HostAttachment, probe_registration
from megatron.core.resharding.mixed_sources import Shard, build_mixed_plan, digest, normalized
from megatron.core.resharding.mixed_transport import execute


def shard(row, name, expert, start, stop, width, layout, tensor, *, host=False):
    return Shard(f"{row['node']}/host" if host else identity(row), row['node'],
                 None if host else row['new_rank'], 'host' if host else 'gpu', tensor,
                 name, expert, 'int64', 8, (2, width), (0, start), (2, stop), layout)


def snapshot(mapping, phase, width, host_reports):
    """Sources are the shards each eligible GPU holds now; targets are EDP0's next layout."""
    sources, targets = [], []
    for row in mapping:
        if not row['active']:
            continue
        if row['source_eligible']:
            # EDP0 survivors expand from the TP2 shards assigned at the shrink, whatever
            # their current rank; a rejoined empty target is not source-eligible.
            old_size = 2 if phase == 2 and row['replica'] == 'EDP0' else 4
            tp = row['shrink_etp'] if old_size == 2 else row['source_etp']
            for name in NAMES:
                for part in range(tp * (8 // old_size), (tp + 1) * (8 // old_size)):
                    if not host_resident(name, part):
                        sources.append(shard(row, name, row['ep'], part * width // 8, (part + 1) * width // 8,
                                             width, f'TP{old_size}', f'{name}_{row["ep"]}_{part}'))
        if row['replica'] == 'EDP0':
            size = 2 if phase == 1 else 4
            for name in NAMES:
                targets.append(shard(row, name, row['ep'], row['etp'] * width // size,
                                     (row['etp'] + 1) * width // size, width, f'TP{size}', f'{name}_target'))
    for node, report in host_reports.items():
        for entry in report['entries']:
            # Only host-resident fixture parts are host candidates, so GPU and host
            # sources must complement each other even with the full library loaded.
            if host_resident(entry['name'], entry['part']):
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
        probe = probe_registration(host_backing(config).with_name(f"weavetp_{config['launch']}_probe_{slot}_{epoch}"))
        bus.put(f'host_registration/{epoch}/{slot}', journal.event('host_registration_preflight', **probe))
        self.reports = dict(zip(NODES, bus.wait([f'host_ready/{n}' for n in NODES])))
        report = self.reports[config['node']]
        self.structure = {e['name']: e for e in report['structure']}
        require(set(self.structure) == set(NAMES) and all(e['shape'] == [2, self.width]
                for e in self.structure.values()), 'host structure/config mismatch')
        self.host = HostAttachment(report, record=lambda **r: append_json(
            self.root / f'host_worker_{slot}_{epoch}.jsonl', r))
        self.sources, self.targets, self.profile = {}, {}, None
        self.initial_h2d_bytes = 0
        # A replacement (epoch 2) reads structure only; its weights arrive through the plan.
        if epoch == 0:
            for entry in report['structure']:
                if (entry['expert'] == self.row['ep'] and entry['part'] // 2 == self.row['etp'] and
                        not host_resident(entry['name'], entry['part'])):
                    cpu = read_oracle(report['manifest_path'], entry['name'], entry['expert'],
                                      entry['start'][1], entry['stop'][1])
                    # Strided GPU storage keeps every source a non-contiguous view.
                    self.sources[entry['id']] = torch.empty((2, 2 * cpu.shape[1]), dtype=torch.int64,
                                                            device='cuda')[:, ::2]
                    self.sources[entry['id']].copy_(cpu)
                    self.initial_h2d_bytes += cpu.numel() * cpu.element_size()
        journal.event('source_initialization', initial_h2d_bytes=self.initial_h2d_bytes, empty_receiver=epoch == 2,
                      host_backing=report['backing'], owner_pid=report['owner_pid'])

    def evidence(self):
        return {'profile_sha256': digest(self.profile), 'initial_h2d_bytes': self.initial_h2d_bytes,
                'owner_pid': self.host.report['owner_pid'], 'backing': self.host.report['backing'],
                'backing_inode': self.host.report['backing_inode']}

    def release(self):
        self.sources.clear()
        self.targets.clear()
        torch.cuda.synchronize()

    def close(self):
        self.release()
        self.host.close()

    def validate(self, session, group, phase, round_id):
        if phase == 0:
            if self.profile is None:
                frozen = self.config['profile_file']
                if frozen is None:
                    measured = collect(self.config, self.config['topology'], self.slot, group, self.host, self.journal)
                elif self.slot == 0:
                    # Step 5 §2.1: default- and deterministic-mode launches plan over one frozen
                    # profile, so their plans are comparable; a re-measured profile is not. The
                    # controller requires both nodes to pass the same file, since collection is collective.
                    measured = json.loads(Path(frozen).read_text(encoding='utf-8'))
                    require(set(measured['p2p']) == {identity(r) for r in self.config['topology']},
                            'frozen profile was measured on different devices')
                    self.journal.event('profile_frozen', path=frozen, profile_sha256=digest(measured))
                # Node clocks differ in the metadata; every plan hashes one published payload.
                if self.slot == 0:
                    self.bus.put('source_profile', measured)
                self.profile = self.bus.wait(['source_profile'])[0]
                if self.slot in (0, 8):
                    (self.root / 'profile.json').write_text(json.dumps(self.profile, sort_keys=True,
                        separators=(',', ':')), encoding='utf-8')
                    (self.root / 'profile.sha256').write_text(digest(self.profile) + '  profile.json\n')
            return self.evidence()
        if self.profile is None:  # a replacement joins at phase 2 without collecting
            self.profile = self.bus.wait(['source_profile'])[0]
        mapping = session.rank_map()
        if phase == 1 and self.slot == 0:
            self.bus.put(f'shrink_etp/{round_id}', {str(r['slot']): r['etp'] for r in mapping})
        if phase == 2:
            shrink = self.bus.wait([f'shrink_etp/{round_id}'])[0]
            mapping = [{**r, 'shrink_etp': shrink[str(r['slot'])]} for r in mapping]
        sources, targets = snapshot(mapping, phase, self.width, self.reports)
        membership = [identity(r) for r in mapping if r['active']]
        rank_map = {identity(r): r['new_rank'] for r in mapping if r['active']}
        tasks, cache_key, loads = build_mixed_plan(sources, targets, self.profile, membership=membership,
                                                  epoch=session.epoch, rank_map=rank_map)
        session.resources.plans[cache_key] = tasks
        # The same snapshot replayed under the other mode's rank numbering must plan identically.
        alternate = {identity(r): r['slot'] if session.backend.mode == 'real' else i
                     for i, r in enumerate(r for r in mapping if r['active'])}
        replay, _, _ = build_mixed_plan(
            [replace(s, rank=alternate[s.device]) if s.kind == 'gpu' else s for s in sources],
            [replace(t, rank=alternate[t.device]) for t in targets], self.profile,
            membership=membership, epoch=session.epoch, rank_map=alternate)
        require(normalized(tasks) == normalized(replay), 'normalized mode plans differ')
        if self.slot == 0:
            # The full snapshot and this process's mode environment make two launches' plans comparable offline.
            write_json(self.root / f'source_plan_{round_id}_{phase}.json', {
                'tasks': [asdict(t) for t in tasks], 'cache_key': cache_key,
                'profile_sha256': digest(self.profile),
                'profile_source': 'measured' if self.config['profile_file'] is None else 'frozen_file',
                'normalized_replay_equal': True, 'membership': membership, 'epoch': session.epoch, 'rank_map': rank_map,
                'sources': [asdict(s) for s in sources], 'targets': [asdict(t) for t in targets],
                'environment': {key: os.environ.get(key) for key in DETERMINISTIC_ENVIRONMENT},
                'loads': [{'resource': k, 'seconds': v} for k, v in loads.items()]})
        size = 2 if phase == 1 else 4
        if self.row['replica'] == 'EDP0':
            # Poisoned strided targets: the plan must write every element exactly once.
            self.targets = {f'{name}_target': torch.full((2, 2 * self.width // size), -1, dtype=torch.int64,
                                                         device='cuda')[:, ::2] for name in NAMES}
        actual = dist.get_process_group_ranks(group)
        clock = self.config['clock']
        stats = execute(tasks, device=self.device, sources=self.sources, targets=self.targets, host=self.host,
                        service_factory=lambda: make_copy_service(session, self.slot, group, p2p_order='task-round'),
                        rank_to_group={r: i for i, r in enumerate(actual)}, resources=session.resources,
                        labels={'event': 'shrink' if phase == 1 else 'expand', 'round': round_id,
                                'epoch': session.epoch},
                        record=lambda row: append_json(self.root / f'links_{self.slot}.jsonl', {
                            **row, 'timing_endpoint': 'destination', 'node': self.config['node'],
                            'clock_offset_s': clock['offset'], 'clock_error_s': clock['error']}))
        if self.row['replica'] == 'EDP0':
            row = mapping[self.slot]
            start, stop = row['etp'] * self.width // size, (row['etp'] + 1) * self.width // size
            for name in NAMES:
                expected = read_oracle(self.host.report['manifest_path'], name, row['ep'], start, stop)
                require(torch.equal(self.targets[f'{name}_target'].cpu(), expected),
                        f'{name} checkpoint bitwise mismatch')
            require(stats['payload_bytes'] + stats['reuse_bytes'] == len(NAMES) * 2 * (stop - start) * 8,
                    'link payload coverage mismatch')
            # The verified target becomes the source for the expansion or the next round.
            self.sources = {f'{name}_{row["ep"]}_{part}': self.targets[f'{name}_target'][
                                :, part * self.width // 8 - start:(part + 1) * self.width // 8 - start]
                            for name in NAMES
                            for part in range(row['etp'] * (8 // size), (row['etp'] + 1) * (8 // size))
                            if not host_resident(name, part)}
        return {**stats, 'bitwise': True, 'normalized_replay_equal': True, **self.evidence()}
