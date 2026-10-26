"""Step 5 real-model fail-stop worker: DeepSeek-V2-Lite TP4 -> TP2 -> TP4 on the Step 4 protocol.

Weights and KV move through the Step 3 mixed planner over model shards in global
coordinates: residual TP4 shards of the survivors and EDP1's healthy TP4 shards fill the
TP2 targets; TP2 shards and the TP2 KV (prefix, then a pre-switch delta) fill the TP4
targets. Every event is validated before anything is released: target parameter hashes
against the standalone reference load of the same layout and coordinate, migrated KV
against the head mapping of the survivors' saved snapshots. Each serving run leaves a
checkpoint (prefilled tokens, decoded tokens, optional KV snapshot, bitwise logits
records) that an independent reference replay checks afterwards.

Model-specific operations live behind a runtime object (MegatronRuntime on the server,
a CPU double in the tests); this module holds the protocol only. Transfers run as
synchronous waves in which every active member participates; Step 4's service/transfer
overlap is not measured here.
"""

import argparse
import json
import math
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from async_state import PeakBudget, ReplicaDriver, Request, require
from common import DETERMINISTIC_ENVIRONMENT, Journal, connect, online_switch, write_json
from determinism import bitwise_record, history, mode_manifest
from model_shards import identity, kv_shards, map_kv_heads, model_local_orders, parameter_shards
from safety import parent_death_signal, verify_worker_context

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
from async_topology import AsyncSession as FaultSession
from torch.distributed import distributed_c10d as c10d

from megatron.core.failstop import (
    ControlEvent,
    FaultBackend,
    build_replica_pg_collections,
    make_copy_service,
)
from megatron.core.failstop_rebind import ModelBinding
from megatron.core.resharding.mixed_sources import Shard, Task, build_mixed_plan, digest
from megatron.core.resharding.mixed_transport import execute


def weavetp_plan(sources, targets, profile, *, membership, epoch, rank_map):
    tasks, key, loads = build_mixed_plan(sources, targets, profile, membership=membership, epoch=epoch, rank_map=rank_map)
    return tasks, key, {'planner': 'weavetp', 'loads': [{'resource': k, 'seconds': v} for k, v in loads.items()]}


def anchortp_plan(sources, targets, profile, *, membership, epoch, rank_map):
    from anchortp_plan import build_fixed_plan

    return build_fixed_plan(sources, targets, profile, membership=membership, epoch=epoch, rank_map=rank_map)


# plan(sources, targets, profile, *, membership, epoch, rank_map) -> (tasks, cache_key, record). Sources and
# targets are flat shard lists in global coordinates: how many replicas sit on either side is not part of the
# interface (a split or a merge of replicas supplies the same lists).
PLANNERS = {'weavetp': weavetp_plan, 'anchortp': anchortp_plan,
            'ignore': weavetp_plan, 'restart': weavetp_plan}  # fail-slow: ignore never switches, restart migrates nothing


def downweighted(profile, devices, factor):
    """The profile with every p2p entry into or out of `devices` slowed to `factor` of its bandwidth (the fail-slow shrink
    plan: co-runners on those GPUs); every other entry is unchanged."""
    p2p = {src: {dst: dict(fit, bytes_per_s=fit['bytes_per_s'] * factor) if devices & {src, dst} else fit for dst, fit in row.items()}
           for src, row in profile['p2p'].items()}
    return {**profile, 'p2p': p2p}


def fixed_tasks(sources, targets, first_id, wave):
    """KV tasks of the online switch without the planner: every KV atom has exactly one holder, so each target shard
    pairs with the source shard of the same stream coverage."""
    by_cover = {(s.name, s.start, s.stop): s for s in sources}
    return [Task(first_id + i, wave, by_cover[t.name, t.start, t.stop], t, t.start, t.stop,
                 math.prod(b - a for a, b in zip(t.start, t.stop)) * t.element_size)
            for i, t in enumerate(sorted(targets, key=lambda t: (t.name, t.start, t.device)))]


class Fence:
    """Transfers complete synchronously before the cutover; the driver's handle contract is already satisfied."""

    def wait(self):
        return None

    def done(self):
        return True


class Run:
    """One serving model instance: what it prefilled (or the KV it inherited), what it decoded, its records."""

    def __init__(self, layout, prefill_tokens, kv_snapshot=None):
        self.layout, self.prefill_tokens, self.kv_snapshot = layout, list(prefill_tokens), kv_snapshot
        self.decode_tokens, self.records = [], []

    def spec(self):
        return {'layout': self.layout, 'prefill_tokens': self.prefill_tokens, 'decode_tokens': self.decode_tokens,
                'kv_snapshot': self.kv_snapshot}


class Worker:
    def __init__(self, config, slot, initial_epoch, runtime):
        self.config, self.slot, self.initial_epoch, self.runtime = config, slot, initial_epoch, runtime
        self.root, self.row = Path(config['output']), config['topology'][slot]
        self.store, self.bus = connect(config)
        self.journal = Journal(self.root / f'model_{slot}_{initial_epoch}.jsonl', config['node'], 'worker', **config['clock'])
        # real, kill: the victims are SIGKILLed and replaced; kill keeps the prebuilt degraded world until the rejoin
        self.real, self.kill, self.healthy = config['mode'] in ('real', 'kill'), config['mode'] == 'kill', self.row['replica'] == 'EDP1'
        self.session = FaultSession(config['topology'], config['launch'],
                                    FaultBackend(config['mode'], fail=lambda *_: None, rejoin=lambda *_: None),
                                    local_orders=model_local_orders(config['topology']))
        self.victim = slot in self.session.failed
        self.driver = ReplicaDriver(self.row['replica'])
        self.device = identity(self.row)
        self.profile = json.loads(Path(config['profile_file']).read_text(encoding='utf-8'))
        require(set(self.profile['p2p']) == {identity(r) for r in config['topology']}, 'profile devices differ from this launch')
        self.manifest = mode_manifest(config['deterministic_mode'], algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
                                      environ=os.environ)
        self.reference = Path(config['reference_root'])
        self.heads = config['kv_heads']
        self.model = self.context = self.binding = self.run = self.pack = None
        self.own_sources, self.own_targets = {}, {}
        self.tp, self.next_token, self.tp2_coord = 4, None, None
        self.request = Request(f"{config['launch']}-r0", 0, 0.0, ())
        self.runs, self.transfers, self.memory, self.pauses, self.events = [], [], [], [], []
        self.last_end = None
        self.prebuilt = {}
        self.planner = PLANNERS[config['method']]
        self.online = online_switch(config)
        self.degraded_rows = self.barrier = self.degraded = None
        self.plans = {}  # online switch, kill: the weight plans computed before serving, by label

    # ---- world and groups -------------------------------------------------------------------------------------
    def initialize(self):
        parent_death_signal()
        require(not Path(os.environ['CUDA_MPS_PIPE_DIRECTORY']).exists(), 'MPS isolation path exists')
        torch.cuda.set_device(0)
        require(torch.cuda.device_count() == 1, 'worker must see only its assigned GPU')
        require(str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') ==
                self.row['gpu_uuid'].removeprefix('GPU-'), 'GPU UUID mismatch')
        context_tensor = torch.empty(1, device='cuda')
        context = verify_worker_context(self.row['gpu_uuid'], self.root / f'context_{self.slot}_{self.initial_epoch}.jsonl')
        self.bus.put(f'cuda/{self.initial_epoch}/{self.slot}', self.journal.event('cuda_ready',
            slot=self.slot, gpu_uuid=self.row['gpu_uuid'], epoch=self.initial_epoch,
            empty_receiver=self.initial_epoch == 2, context_verification=context, manifest=self.manifest))
        self.bus.wait([f'cuda_verified/{self.initial_epoch}'])
        del context_tensor
        if self.initial_epoch == 0 and not self.session.backend.rebuilds(0):  # emulate, and kill until its rejoin
            dist.init_process_group('gloo', store=dist.PrefixStore(f"step05/{self.config['launch']}/control/", self.store),
                                    rank=self.slot, world_size=16, timeout=timedelta(seconds=self.config['stage_timeout_s']))
            stage = FaultSession(self.config['topology'], self.config['launch'], self.session.backend,
                                 local_orders=model_local_orders(self.config['topology']))
            self.prebuilt[0] = self.build(stage, 0)
            stage.fail(stage.failed)
            self.prebuilt[1] = self.build(stage, 1)
            if self.online or self.kill:  # the layout the shrink migrates into, with the TP2 coordinates its groups actually give
                stage.adopt_group_order(self.prebuilt[1]['records'])  # (as adopt() reconciles them; cleared at the rejoin)
                self.degraded_rows = stage.rank_map()
            stage.complete()
            if self.kill:  # the rejoin rebuilds the world (as real); the shrink is planned in this degraded world before serving
                self.degraded = stage
            else:
                stage.rejoin(stage.failed)
                self.prebuilt[2] = self.build(stage, 2)
                # The world never changes in emulate mode: serving models keep their communicators.
                self.prebuilt[2]['service2'] = self.prebuilt[1]['target']
            for phase in list(self.prebuilt)[1:]:
                if self.healthy:
                    self.prebuilt[phase]['target'] = self.prebuilt[0]['target']
            if self.online:  # EDP0's phase-2 groups are its phase-0 ones (same ranks): the initial TP4 model serves again
                self.prebuilt[2]['target'], self.prebuilt[2]['data'] = self.prebuilt[0]['target'], self.prebuilt[0]['data']
            if self.online or self.kill:
                # execute()'s overlap hook orders every wave's P2P before any model collective on this group: online, the
                # phase-0 world (the victims send); kill, the degraded world's twelve live ranks (the victims are dead by then)
                self.barrier = dist.new_group(sorted(stage.current_world()['ranks']), backend='gloo',
                                              timeout=timedelta(seconds=self.config['stage_timeout_s']))
                self.warm_groups()
        torch.cuda.reset_peak_memory_stats()

    def warm_groups(self):
        """Online switch: connect every prebuilt NCCL communicator this rank is in (one-element all_reduce) in one global
        order (members, phase, position), so no collective during the run pays communicator setup."""
        order = {}
        for phase, pack in sorted(self.prebuilt.items()):
            if phase == 1 and self.victim:  # outside the degraded world: no phase-1 group
                continue
            collections = [c for c in (pack['target'], pack['service2']) if c is not None]  # holds only member groups
            for position, group in enumerate([pack['data'], *(getattr(c, name) for c in collections for name in c.local_ranks)]):
                order.setdefault(id(group), (tuple(dist.get_process_group_ranks(group)), phase, position, group))
        for *_, group in sorted(order.values(), key=lambda item: item[:3]):
            dist.all_reduce(torch.zeros(1, device='cuda'), group=group)

    def build(self, session, phase):
        if session.backend.rebuilds(phase):
            members = session.current_world()['membership']
            dist.init_process_group('nccl', store=dist.PrefixStore(
                f"step05/{self.config['launch']}/epoch/{session.epoch}/", self.store),
                rank=members.index(self.slot), world_size=len(members),
                timeout=timedelta(seconds=self.config['stage_timeout_s']), device_id=torch.device('cuda', 0))
        records = []
        groups = build_replica_pg_collections(session.replicas(), backend='nccl',
            timeout=timedelta(seconds=self.config['stage_timeout_s']), records=records)
        data = dist.new_group(list(sorted(session.current_world()['ranks'])), backend='nccl',
                              timeout=timedelta(seconds=self.config['stage_timeout_s']))
        service2 = None
        if phase == 2:
            ranks = tuple(r['new_rank'] for r in session.rank_map()
                          if r['replica'] == 'EDP0' and r['slot'] not in session.failed)
            service2 = build_replica_pg_collections([(ranks, 2, 2, 2)], backend='nccl',
                timeout=timedelta(seconds=self.config['stage_timeout_s']))[0]
        return {'target': groups[1 if self.healthy else 0], 'data': data, 'records': records, 'service2': service2}

    def adopt(self, phase):
        pack = self.build(self.session, phase) if self.session.backend.rebuilds(phase) else self.prebuilt[phase]
        self.session.adopt_group_order(pack['records'])
        self.pack = pack
        if self.binding is not None and (self.healthy or phase == 2):
            # Only a model that keeps serving is rebound, and only when the world rebuilt its communicators.
            # Rebind at every epoch, also onto the same (emulate) groups: the binding's generation must follow the
            # session's, or the first forward after the transition is refused as stale.
            groups = pack['target'] if self.healthy else pack['service2']
            self.session.resources.ready = False
            if groups is self.binding.groups:
                self.binding.refresh()
            else:
                self.binding.rebind(groups)
            self.session.resources.activate(self.session.cache_key())
            require(groups.local_ranks['expt_tp'] == (self.coord if self.healthy else self.tp2_coord),
                    'rebind changed the serving shard coordinate')
        return pack

    @property
    def coord(self):
        return self.session.rank_map()[self.slot]['etp']

    @property
    def rank(self):
        return self.session.rank_map()[self.slot]['new_rank']

    # ---- serving ----------------------------------------------------------------------------------------------
    def reference_hashes(self, tp, coord):
        path = self.reference / f'tp{tp}' / 'reference' / f"parameters_rank_{coord + tp * self.row['ep']}.json"
        return json.loads(path.read_text(encoding='utf-8'))

    def start_run(self, model, groups, tp, hashes, *, prefill_tokens=None, kv_context=None, kv_snapshot=None):
        """Bind a serving model; either prefill a history or continue on the KV it received."""
        if self.binding is not None:
            self.binding.close()
        self.model, self.tp = model, tp
        self.binding = ModelBinding(model, groups, self.session.resources, {'tp': tp, 'coord': self.coord, 'ep': self.row['ep']})
        layout = {'tp': tp, 'coord': self.coord, 'ep': self.row['ep'], 'parameter_hashes_sha256': digest(hashes)}
        if kv_context is None:
            self.context = self.runtime.context()
            self.run = Run(layout, prefill_tokens)
            self.runs.append(self.run)
            logits = self.runtime.prefill(model, self.context, prefill_tokens)
            self.record(logits, 0)
            return int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        self.context = kv_context
        self.run = Run(layout, list(self.request.tokens), kv_snapshot)
        self.runs.append(self.run)
        return self.request.generated[-1] if self.request.generated else self.request.prompt[-1]

    def record(self, logits, step_index):
        self.run.records.append(bitwise_record(logits, step=step_index, manifest=self.manifest,
            layout={k: self.run.layout[k] for k in ('tp', 'coord', 'ep')},
            history=history(self.request.id, self.run.prefill_tokens + self.run.decode_tokens[:step_index]),
            parameter_checksum=self.run.layout['parameter_hashes_sha256']))

    def step(self, phase, next_token):
        """One committed decode step of the serving model; returns the token it produced."""
        ticket = self.driver.begin_step(self.request)
        submitted = time.monotonic()
        logits = self.runtime.decode(self.model, self.context, next_token)
        completed = time.monotonic()
        token = int(logits.reshape(-1, logits.shape[-1])[-1].argmax())
        self.run.decode_tokens.append(next_token)
        self.record(logits, len(self.run.decode_tokens))
        self.driver.commit(ticket, self.request, next_token)
        if self.pauses and self.pauses[-1]['end_local'] is None:
            self.pauses[-1]['end_local'] = submitted
        self.last_end = completed
        self.journal.event('service_commit', phase=phase, slot=self.slot, replica=self.row['replica'], step=self.driver.step,
                           sequence=self.driver.collective_sequence, token=next_token, seconds=completed - submitted)
        return token

    def collective_flag(self, flag):
        value = torch.tensor(int(flag), dtype=torch.int64, device='cuda')
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.binding.groups.tp_ep)
        return bool(value.item())

    def serve_until(self, key, phase, progress_key=None):
        """Keep committing until the key appears; a full static KV turns the loop into polling."""
        deadline = time.monotonic() + self.config['stage_timeout_s']
        while True:
            if self.runtime.kv_room(self.context) > 0:
                self.next_token = self.step(phase, self.next_token)
            if progress_key is not None:
                self.bus.put(progress_key, True)
            if self.collective_flag(self.bus.has(key)):
                return
            require(time.monotonic() < deadline, f'service timeout waiting for {key}')
            require(not self.bus.has('failure'), 'peer failure during service')
            time.sleep(.005)

    def save_run(self, label):
        directory = self.root / 'checkpoints' / f'r{self.session.round}' / label
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({'spec': self.run.spec(), 'records': self.run.records, 'manifest': self.manifest,
                    'request_id': self.request.id, 'slot': self.slot, 'replica': self.row['replica']},
                   directory / f'slot_{self.slot}.pt')

    def save_kv(self, label, length):
        """This TP2 coordinate's KV for tokens [0, length), for the receiving side's independent mapping."""
        directory = self.root / 'kv' / f'r{self.session.round}' / label
        directory.mkdir(parents=True, exist_ok=True)
        kv = {layer: tuple(t[:length].detach().cpu().clone() for t in pair) for layer, pair in self.runtime.kv(self.context).items()}
        torch.save({'tp': self.tp, 'coord': self.tp2_coord, 'length': length, 'kv': kv}, directory / f'coord_{self.tp2_coord}.pt')
        return str(directory)

    def load_kv(self, directory):
        snapshot = {}
        for path in sorted(Path(directory).glob('coord_*.pt')):
            saved = torch.load(path, weights_only=False)
            snapshot[saved['coord']] = saved['kv']
        require(set(snapshot) == {0, 1}, 'KV snapshots of both TP2 coordinates are required')
        return snapshot

    def phase_report(self, round_id, phase, *, empty_receiver=False):
        memory = {**self.runtime.memory(), 'weight_storage_bytes': self.runtime.weight_storage(self.model),
                  'kv_bytes': self.kv_bytes(), 'phase': f'{round_id}/{phase}', 'layout': self.tp}
        self.memory.append(memory)
        report = self.journal.event('phase_ready', slot=self.slot, replica=self.row['replica'], phase=phase, round=round_id,
            rank_map=self.session.rank_map(), checks_passed=True, step=self.driver.step, sequence=self.driver.collective_sequence,
            history=list(self.request.tokens), empty_receiver=empty_receiver, memory=memory, layout=self.tp,
            rejoin_pause_s=self.pauses[-1]['end_local'] - self.pauses[-1]['start_local'] if self.pauses else None)
        self.bus.put(f'phase/{round_id}/{phase}/{self.slot}', report)

    # ---- transitions ------------------------------------------------------------------------------------------
    def transition(self, phase, round_id):
        message = self.bus.wait([f'transition/{round_id}/{phase}'])[0]
        if self.real:
            c10d._abort_process_group()
            require(not dist.is_initialized(), 'old world still initialized after abort')
            self.runtime.world_destroyed()
            if self.healthy:
                self.pauses.append({'phase': phase, 'start_local': self.last_end, 'reason': 'abort/world rebuild', 'end_local': None})
        if phase == 1:
            self.session.fail(self.session.failed, ControlEvent(**message))
            if not self.healthy:
                self.driver.fail(self.session.epoch)
        else:
            self.session.rejoin(self.session.failed, ControlEvent(**message))
            self.driver.epoch = self.session.epoch
        self.bus.put(f'aborted/{round_id}/{phase}/{self.slot}', True)
        key = f'build/{round_id}/{phase}'
        if not self.real and self.healthy:
            self.serve_until(key, 'shrink' if phase == 1 else 'expand')
        else:
            self.bus.wait([key])

    # ---- migration --------------------------------------------------------------------------------------------
    def kv_bytes(self):
        if self.context is None:
            return 0
        return sum(t.numel() * t.element_size() for pair in self.runtime.kv(self.context).values() for t in pair)

    def admission(self, target_tp, kv_length):
        memory = self.runtime.memory()
        target_bytes = self.runtime.target_bytes(target_tp)
        plan = PeakBudget(0 if self.model is None else self.runtime.weight_storage(self.model), target_bytes,
                          self.kv_bytes(), self.runtime.kv_bytes(target_tp, kv_length), target_bytes, 0,
                          self.config['memory_margin_bytes'])
        plan.check(self.config['memory_limit_bytes'] or memory['total'], memory['free'], memory['allocated'])
        return dict(vars(plan), total=plan.total, target_layout=target_tp)

    def publish(self, round_id, label, sources, targets):
        """This member's planner inputs; tensors stay local, keyed by the shards' tensor names."""
        self.own_sources = {shard.tensor: view for shard, view in sources}
        self.own_targets = {shard.tensor: view for shard, view in targets}
        self.bus.put(f'snapshot_shards/{round_id}/{label}/{self.slot}',
                     {'sources': [asdict(s) for s, _ in sources], 'targets': [asdict(t) for t, _ in targets]})

    def weight_sources(self, model, tp, coord):
        return parameter_shards(model.named_parameters(), device=self.device, node=self.row['node'], rank=self.rank,
                                tp=tp, coord=coord, ep=self.row['ep'])

    def kv_views(self, context, tp, coord, start, stop):
        return kv_shards(self.runtime.kv(context), device=self.device, node=self.row['node'], rank=self.rank, tp=tp,
                         coord=coord, heads=self.heads, start=start, stop=stop)

    def transfer(self, round_id, phase, label, writer, overlap=None):
        """Every active member plans the published global snapshot identically and runs the same waves.

        Online switch (and kill's shrink): no planner runs. The weights follow the plan made before serving for `label` (a
        KV-only delta has none), which must cover exactly the published weight shards; KV atoms have one holder each and
        pair directly (fixed_tasks). overlap: execute()'s in-flight hook (ordering barrier + foreground decode inside every
        wave)."""
        mapping = self.session.rank_map()
        members = [r['slot'] for r in mapping if r['active']]
        rank_map = {identity(r): r['new_rank'] for r in mapping if r['active']}
        published = self.bus.wait([f'snapshot_shards/{round_id}/{label}/{s}' for s in members])

        def shards(key):
            return [Shard(**{**row, 'shape': tuple(row['shape']), 'start': tuple(row['start']), 'stop': tuple(row['stop'])})
                    for entry in published for row in entry[key]]
        sources, targets = shards('sources'), shards('targets')
        if self.online or label in self.plans:
            plan = self.plans.get(label, {'tasks': [], 'key': f'kv/{label}', 'record': {'planner': None}, 'shards': (set(), set())})
            weights = {s for s in sources if not s.name.startswith('kv.')}, {t for t in targets if not t.name.startswith('kv.')}
            require(weights == plan['shards'], f'{label}: the weight shards differ from the plan made before serving')
            kv = [s for s in sources if s.name.startswith('kv.')], [t for t in targets if t.name.startswith('kv.')]
            tasks = plan['tasks'] + fixed_tasks(*kv, len(plan['tasks']), 1 + max((t.wave for t in plan['tasks']), default=-1))
            cache_key, record = plan['key'], plan['record']
        else:
            tasks, cache_key, record = self.planner(sources, targets, self.profile, membership=list(rank_map),
                                                    epoch=self.session.epoch, rank_map=rank_map)
        self.session.resources.plans[cache_key] = tasks
        if writer:
            write_json(self.root / f'source_plan_{round_id}_{label}.json', {
                'method': self.config['method'], **record,
                'tasks': [asdict(t) for t in tasks], 'cache_key': cache_key, 'profile_sha256': digest(self.profile),
                'profile_source': 'frozen_file', 'membership': list(rank_map), 'epoch': self.session.epoch, 'rank_map': rank_map,
                'sources': [asdict(s) for s in sources], 'targets': [asdict(t) for t in targets],
                'environment': {k: os.environ.get(k) for k in DETERMINISTIC_ENVIRONMENT},
                'bytes_by_source_kind': {kind: sum(t.payload_bytes for t in tasks if t.source_kind == kind)
                                         for kind in sorted({t.source_kind for t in tasks})}})
        actual = dist.get_process_group_ranks(self.pack['data'])
        stats = execute(tasks, device=self.device, sources=self.own_sources, targets=self.own_targets, host=None,
                        service_factory=lambda: make_copy_service(self.session, self.slot, self.pack['data'], p2p_order='task-round'),
                        rank_to_group={r: i for i, r in enumerate(actual)}, resources=self.session.resources,
                        labels={'event': label, 'round': round_id, 'epoch': self.session.epoch},
                        record=lambda row: self.journal.event('link', **row), overlap=overlap)
        self.transfers.append({'phase': f'{round_id}/{phase}', 'label': label, **stats, 'tasks': len(tasks)})
        return stats

    def warm_pairs(self, pairs, session=None, group=None):
        """Online switch: one one-element exchange over every (source rank, target rank) pair a planned or KV wave will
        use, through the same copy service and order as execute(), so no wave pays a first P2P connection. session/group:
        the world the waves run in (default the current one; kill: the degraded world and its data group)."""
        session = self.session if session is None else session
        group = self.pack['data'] if group is None else group
        service = make_copy_service(session, self.slot, group, p2p_order='task-round')
        index = {r: i for i, r in enumerate(dist.get_process_group_ranks(group))}
        rank = session.rank_map()[self.slot]['new_rank']
        buffers = []
        for task, (source, target) in enumerate(sorted(pairs)):
            for peer, send in ((target, True), (source, False)):
                if (source if send else target) == rank:
                    buffers.append(torch.zeros(1, device='cuda'))
                    (service.submit_send if send else service.submit_recv)(buffers[-1], index[peer], task)
        service.producer_stream = torch.cuda.current_stream()
        service.launch().wait()
        return len(pairs)

    def plan_before_serving(self, label, sources, targets, profile, session=None):
        """Online switch: publish this rank's weight shards for `label`, gather everyone's and plan them now (every rank
        computes the same plan); transfer() runs it at the switch. session: the world the switch runs in (default the
        current one; kill: the degraded world, whose members exclude the victims)."""
        session = self.session if session is None else session
        self.bus.put(f'plan_shards/{label}/{self.slot}', {'sources': [asdict(s) for s in sources], 'targets': [asdict(t) for t in targets]})
        mapping = session.rank_map()
        rank_map = {identity(r): r['new_rank'] for r in mapping if r['active']}
        published = self.bus.wait([f"plan_shards/{label}/{r['slot']}" for r in mapping if r['active']])

        def shards(key):
            return [Shard(**{**row, 'shape': tuple(row['shape']), 'start': tuple(row['start']), 'stop': tuple(row['stop'])})
                    for entry in published for row in entry[key]]
        tasks, key, record = self.planner(shards('sources'), shards('targets'), profile, membership=list(rank_map),
                                          epoch=session.epoch, rank_map=rank_map)
        self.plans[label] = {'tasks': tasks, 'key': key, 'record': {**record, 'planned': 'before serving', 'plan_profile_sha256': digest(profile)},
                             'shards': (set(shards('sources')), set(shards('targets')))}
        return self.plans[label]

    def validate_weights(self, target, tp):
        expected = self.reference_hashes(tp, self.coord)
        actual = self.runtime.parameter_hashes(target)
        require(set(actual) == set(expected), 'parameter set differs from the reference load')
        differing = sorted(name for name in actual if actual[name] != expected[name])
        require(not differing, f'{len(differing)} parameters differ bitwise from the reference load, first {differing[:3]}')
        return {'parameters': len(actual), 'reference': 'standalone checkpoint load of the same layout and coordinate'}

    # ---- events -----------------------------------------------------------------------------------------------
    def shrink(self, round_id):
        """Survivors build empty TP2 targets filled from residual TP4 + EDP1 TP4; EDP1 joins the waves as a sender."""
        survivors = [s for s in range(16) if self.config['topology'][s]['replica'] == 'EDP0' and s not in self.session.failed]
        sources = self.weight_sources(self.model, 4, self.row['etp'])
        if self.healthy:
            self.publish(round_id, 'shrink', sources, [])
            self.serve_until(f'plan/{round_id}/shrink', 'shrink')
            self.transfer(round_id, 1, 'shrink', writer=False)
            self.serve_until(f'destination_ready/{round_id}/1', 'shrink', progress_key=f'serving/{round_id}/{self.slot}')
        else:
            self.tp2_coord = self.coord
            budget = self.admission(2, len(self.request.tokens))
            target = self.runtime.build(2, self.pack['target'], load=False)
            self.publish(round_id, 'shrink', sources, self.weight_sources(target, 2, self.coord))
            if self.slot == min(survivors):
                self.bus.put(f'plan/{round_id}/shrink', True)
            stats = self.transfer(round_id, 1, 'shrink', writer=self.slot == min(survivors))
            audit = self.validate_weights(target, 2)
            self.events.append({'round': round_id, 'event': 'shrink', 'budget': budget, 'transfer': stats, 'weights': audit,
                                'kv': 'none migrated: full prefill from prompt + committed history'})
            self.driver.start_transfer(Fence(), target_layout=2)
            source = self.model

            def release():
                self.save_run('phase0')
                self.runtime.release(source, list(target.parameters()), [])

            self.driver.cutover(self.request, validate=lambda: None, release=release)
            hashes = self.reference_hashes(2, self.coord)
            self.next_token = self.start_run(target, self.pack['target'], 2, hashes, prefill_tokens=list(self.request.tokens))
            self.bus.put(f'destination/{round_id}/1/{self.slot}', True)
            self.next_token = self.step('degraded', self.next_token)
        self.phase_report(round_id, 1)
        self.session.complete()
        if not self.healthy:
            self.serve_until(f'transition/{round_id}/2', 'degraded', progress_key=f'serving/{round_id}/{self.slot}')

    def expand(self, round_id):
        """TP4 targets on all EDP0 GPUs receive TP2 weights + KV prefix, then the pre-switch KV delta."""
        edp0 = [s for s in range(16) if self.config['topology'][s]['replica'] == 'EDP0']
        survivors = [s for s in edp0 if s not in self.session.failed]
        leader = min(survivors)
        if self.healthy:
            self.publish(round_id, 'expand', self.weight_sources(self.model, 4, self.row['etp']), [])
            self.serve_until(f'plan/{round_id}/expand', 'expand')
            self.transfer(round_id, 2, 'expand', writer=False)
            self.publish(round_id, 'delta', [], [])
            self.serve_until(f'plan/{round_id}/delta', 'expand')
            self.transfer(round_id, 2, 'delta', writer=False)
            self.bus.wait([f'validated/{round_id}/2/{s}' for s in edp0])
            self.phase_report(round_id, 2)
            self.session.complete()
            return
        if self.victim:
            record = self.bus.wait([f'snapshot/{round_id}'])[0]
            self.request = Request.from_dict(record['request'])
            self.driver.join(self.session.epoch)
            self.driver.follow(record['step'], record['sequence'])
        elif self.slot == leader:
            self.bus.put(f'snapshot/{round_id}', {'request': asdict(self.request), 'step': self.driver.step,
                                                'sequence': self.driver.collective_sequence})
        record = self.bus.wait([f'snapshot/{round_id}'])[0]
        require(self.request == Request.from_dict(record['request']), 'replica snapshot disagreement')
        snapshot = self.request.snapshot()
        prefix = len(self.request.tokens)
        budget = self.admission(4, prefix + self.config['expand_delta_steps'] + 1)
        target = self.runtime.build(4, self.pack['target'], load=False)
        target_context = self.runtime.context()
        self.runtime.allocate_kv(target_context, 4)
        sources = []
        if not self.victim:
            self.save_kv('prefix', prefix)
            sources = (self.weight_sources(self.model, 2, self.tp2_coord) +
                       self.kv_views(self.context, 2, self.tp2_coord, 0, prefix))
        self.publish(round_id, 'expand', sources,
                     self.weight_sources(target, 4, self.coord) + self.kv_views(target_context, 4, self.coord, 0, prefix))
        if self.slot == leader:
            self.bus.put(f'plan/{round_id}/expand', True)
        stats = self.transfer(round_id, 2, 'expand', writer=self.slot == leader)
        audit = self.validate_weights(target, 4)
        if not self.victim:
            # Survivors keep committing on TP2 so the pre-switch delta is non-empty.
            for _ in range(self.config['expand_delta_steps']):
                self.next_token = self.step('expand', self.next_token)
            if self.slot == leader:
                self.bus.put(f'delta/{round_id}', {'request': asdict(self.request), 'step': self.driver.step,
                                                 'sequence': self.driver.collective_sequence})
        current = self.bus.wait([f'delta/{round_id}'])[0]
        if self.victim:
            self.request = Request.from_dict(current['request'])
            self.driver.follow(current['step'], current['sequence'])
        require(self.request == Request.from_dict(current['request']), 'final history disagreement')
        length = len(self.request.tokens)
        kv_dir = None
        if not self.victim:
            kv_dir = self.save_kv('delta', length)
        self.publish(round_id, 'delta', [] if self.victim else self.kv_views(self.context, 2, self.tp2_coord, prefix, length),
                     self.kv_views(target_context, 4, self.coord, prefix, length))
        if self.slot == leader:
            self.bus.put(f'plan/{round_id}/delta', True)
            self.bus.put(f'kv_dir/{round_id}', kv_dir)
        delta = self.transfer(round_id, 2, 'delta', writer=self.slot == leader)
        kv_dir = self.bus.wait([f'kv_dir/{round_id}'])[0]
        mapped = map_kv_heads(self.load_kv(kv_dir), source_tp=2, target_tp=4, target_coord=self.coord, heads=self.heads)
        for layer, pair in self.runtime.kv(target_context).items():
            for kind, tensor in enumerate(pair):
                require(torch.equal(tensor[:length].cpu(), mapped[layer][kind][:length]),
                        f'migrated KV differs from the mapped source snapshot at layer {layer}')
        self.runtime.set_kv_length(target_context, length)
        self.events.append({'round': round_id, 'event': 'expand', 'budget': budget, 'transfer': stats, 'delta': delta,
                            'weights': audit, 'kv': {'prefix': prefix, 'length': length, 'reference': 'head mapping of the survivors\' saved TP2 snapshots'}})
        self.bus.put(f'validated/{round_id}/2/{self.slot}', True)
        self.bus.wait([f'validated/{round_id}/2/{s}' for s in edp0])
        self.driver.start_transfer(Fence(), target_layout=4, snapshot=snapshot)
        source = self.model

        def release():
            if source is not None:
                self.save_run('shrink')
                self.runtime.release(source, list(target.parameters()), [])

        self.driver.cutover(self.request, validate=lambda: None, release=release, delta_handle=Fence())
        self.next_token = self.start_run(target, self.pack['target'], 4, self.reference_hashes(4, self.coord),
                                         kv_context=target_context, kv_snapshot=kv_dir)
        self.next_token = self.step('recovered', self.next_token)
        self.phase_report(round_id, 2, empty_receiver=self.victim)
        self.session.complete()

    # ---- main flow --------------------------------------------------------------------------------------------
    def run_worker(self):
        self.initialize()
        if self.initial_epoch == 2:
            self.session.fail(self.session.failed)
            self.session.complete()
            self.session.rejoin(self.session.failed, ControlEvent(**self.bus.wait(['transition/0/2'])[0]))
            self.bus.wait(['build/0/2'])
            self.adopt(2)
            self.runtime.start()
            self.expand(0)
        else:
            for round_id in range(self.config['rounds']):
                self.adopt(0)
                if self.model is None:
                    self.runtime.start()
                    model = self.runtime.build(4, self.pack['target'], load=True)
                    self.request = Request(f"{self.config['launch']}-r{round_id}", 0, 0.0, tuple(self.runtime.prompt()))
                    hashes = self.reference_hashes(4, self.row['etp'])
                    self.events.append({'round': round_id, 'event': 'load', 'weights': self.validate_weights(model, 4)})
                    self.next_token = self.start_run(model, self.pack['target'], 4, hashes, prefill_tokens=list(self.request.prompt))
                for _ in range(self.config['phase0_steps']):
                    self.next_token = self.step('before', self.next_token)
                self.phase_report(round_id, 0)
                if self.healthy:
                    self.serve_until(f'transition/{round_id}/1', 'before', f'healthy_progress/{round_id}/{self.slot}')
                elif self.victim:
                    if self.real:
                        self.bus.wait(['finish'])
                        raise RuntimeError('real victim survived SIGKILL')
                    self.bus.wait([f'emulate_release/{round_id}'])
                    self.save_run('phase0')
                    self.binding.close()
                    self.runtime.release(self.model, [], [])
                    self.model = self.context = self.binding = self.run = None
                    self.bus.put(f'released/{round_id}/{self.slot}', {'src_none': True, 'dst_none': True, 'data_bytes': 0, 'marker_bytes': 0})
                self.transition(1, round_id)
                if not self.victim:
                    self.adopt(1)
                    self.shrink(round_id)
                else:
                    self.session.complete()
                    self.bus.wait(['spawn' if round_id == 0 else f'emulate_rejoin/{round_id}'])
                    if round_id == 0:
                        self.bus.put(f'cuda/2/{self.slot}', self.journal.event('empty_target_ready', gpu_uuid=self.row['gpu_uuid'], empty_receiver=True))
                        self.bus.wait(['cuda_verified/2'])
                    else:
                        self.bus.put(f'empty/{round_id}/{self.slot}', True)
                self.transition(2, round_id)
                self.adopt(2)
                self.expand(round_id)
                if round_id + 1 < self.config['rounds']:
                    self.bus.wait([f'next/{round_id}'])
                    self.session.next_round()
        self.save_run('final')
        write_json(self.root / f'model_checks_{self.slot}.json', {
            'slot': self.slot, 'replica': self.row['replica'], 'events': self.events, 'transfers': self.transfers,
            'memory': self.memory, 'pauses': self.pauses, 'runs': [r.spec() for r in self.runs], 'manifest': self.manifest,
            'transport': 'synchronous mixed-plan waves with every active member; Step 4 overlap is not measured here'})
        self.bus.wait(['finish'])
        if self.real:
            c10d._abort_process_group()
        else:
            dist.destroy_process_group()
        self.bus.put(f'done/{self.slot}', True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--epoch', type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        from model_runtime import MegatronRuntime

        Worker(config, args.slot, args.epoch, MegatronRuntime.from_config(config)).run_worker()
    except BaseException:
        error = {'slot': args.slot, 'pid': os.getpid(), 'traceback': traceback.format_exc()}
        write_json(Path(config['output']) / f'worker_error_{args.slot}_{args.epoch}.json', error)
        try:
            _, bus = connect(config)
            bus.put('failure', error)
        except Exception:
            pass
        raise


if __name__ == '__main__':
    main()
