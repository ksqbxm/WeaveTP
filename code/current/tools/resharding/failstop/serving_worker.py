"""Step 6 B3 worker: the serving driver inside the Step 5 real-model fail-stop worker.

Every rank of an EDP serves its dense replica with block-level KV through the e2e Driver;
decisions come from the control node on SL3060 over the bus (remote_control) and both
domains mirror the ledger. The fault and the rejoin are switches the control node orders:
EDP0 survivors run the Step 5 shrink/expand mechanics (mixed-plan weight transfer, reference
hash audit, expand KV prefix + delta rendered by KVPlanBuilder.as_shards), EDP1 joins the
waves as a sender, emulate victims release and wait, real victims are replaced by epoch-2
processes that join at the expand cutover with the ledger their replica's survivor hands over.
Transfers are synchronous waves (Step 4 overlap is not measured here).
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from common import Journal, audits_reference, connect, require, write_json
from determinism import require_default
from failslow_inject import LEVELS
from model_shards import identity
from model_worker import Worker as ModelWorker
from model_worker import downweighted
from torch.distributed import distributed_c10d as c10d

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from megatron.core.failstop import ControlEvent
from megatron.core.failstop_rebind import ModelBinding
from tools.resharding.e2e import attention_paged, block_kv
from tools.resharding.e2e import controller as ctl
from tools.resharding.e2e import failstop_adapter, remote_control, requests_io
from tools.resharding.e2e.block_backend import BlockBackend
from tools.resharding.e2e.decode_policy import POLICIES
from tools.resharding.e2e.failstop_adapter import REPLICA_IDS
from tools.resharding.e2e.live_runtime import DomainRun, Driver, GlooComm, WallClock
from tools.resharding.e2e.plan_executor import BitwiseValidation
from tools.resharding.e2e.timeline import SwitchEventWriter, TimelineWriter, read_jsonl

ROUND = 0  # the serving run is one fault/rejoin cycle


@dataclass(frozen=True)
class SwitchSpec:
    """What a switcher is asked to do.

    `assignment` maps seq -> (source replica, target replica) for the KV moves; None means every request
    stays on its replica, which is all the fail-stop shrink/expand needs. A layout change that splits one
    replica into two or merges two into one supplies an explicit assignment here (the planner's shard lists
    are already layout-free); nothing else in the switch protocol assumes one source replica per target.
    """

    direction: str
    assignment: dict | None = None


def kv_shard_views(pool, shards, rank):
    """Views of this rank's pool for KVPlanBuilder shards: kv::seg_NNNN.key|value#g<g>:<t0>:<t1> -> seg[g, t0:t1]."""
    views = []
    for shard in shards:
        if shard.rank != rank:
            continue
        name, spec = shard.tensor.split('#')
        segment, kind = int(name[len('kv::seg_'):len('kv::seg_') + 4]), name.rsplit('.', 1)[1]
        group, start, stop = spec[1:].split(':')
        tensor = pool.segments[segment][0 if kind == 'key' else 1]
        views.append((shard, tensor[int(group), int(start):int(stop)]))
    return views


class ServingWorker(ModelWorker):
    def __init__(self, config, slot, initial_epoch, runtime):
        super().__init__(config, slot, initial_epoch, runtime)
        serving = config['serving']
        self.serving = serving
        self.requests = requests_io.load_requests(serving['requests'])
        self.instances = requests_io.expand_instances(requests_io.load_workload(serving['workload'], self.requests), self.requests)
        self.prompts = {i.seq: self.requests[i.id]['prompt_token_ids'] for i in self.instances}
        self.domain = self.row['replica']
        self.replica_id = REPLICA_IDS[self.domain, self.row['ep']]
        self.replicas = tuple(sorted(i for (name, _), i in REPLICA_IDS.items() if name == self.domain))
        self.builder = block_kv.KVPlanBuilder(heads=serving['geometry']['heads'], group_heads=serving['geometry']['group_heads'],
                                              block_tokens=serving['block_tokens'], segment_tokens=serving['segment_tokens'],
                                              layers=serving['geometry']['layers'], k_dim=serving['geometry']['k_dim'],
                                              v_dim=serving['geometry']['v_dim'])
        self.backend = self.driver = None
        self.kv_checks = []
        self.switcher_class = SWITCHERS[config['method']]  # the only place a method differs: switcher + its planner
        # correctness: reference hash audits, KV snapshots and bitwise checks; timing: none of them (default mode).
        self.audits = config['run_mode'] == 'correctness'
        if not self.audits:
            require_default(self.manifest, 'serving worker')

    def validate_weights(self, target, tp):
        if not audits_reference(self.config['scenario'], self.config['method']):
            return {'skipped': 'reference_audit skipped (fail-slow / restart)'}
        return super().validate_weights(target, tp)

    # ---- world, groups, maps ---------------------------------------------------------------------------------
    def build(self, session, phase):
        pack = super().build(session, phase)
        # Every world rank creates every control group (collective): both domains, plus EDP0's serving members
        # alone (the survivors) for the steps a switch runs before the rejoining victims take over the handoff.
        timeout = timedelta(seconds=self.config['stage_timeout_s'])
        pack['control'] = {key: dist.new_group(sorted(self.domain_ranks(session, domain, serving_only)), backend='gloo', timeout=timeout)
                           for key, domain, serving_only in (('EDP0', 'EDP0', False), ('EDP1', 'EDP1', False), ('EDP0_serving', 'EDP0', True))}
        return pack

    def domain_ranks(self, session, domain, serving_only=False):
        """Current world ranks of a domain; survivors come first so the EDP0 leader never changes.
        serving_only drops members that are rejoining (not source eligible) until the cutover."""
        rows = [r for r in session.rank_map() if r['replica'] == domain and r['active'] and (r['source_eligible'] or not serving_only)]
        return [r['new_rank'] for r in sorted(rows, key=lambda r: (r['slot'] in session.failed, r['ep'], r['etp']))]

    def domain_comm(self, serving_only=False):
        key = f'{self.domain}_serving' if serving_only and self.domain == 'EDP0' else self.domain
        return GlooComm(self.domain, self.domain_ranks(self.session, self.domain, serving_only), self.pack['control'][key])

    def current_map(self):
        return failstop_adapter.rank_map_from_rows(self.session.rank_map())

    def tp2_map(self):
        """EDP0 as the TP2 layout the survivors actually served (coordinates recorded at the shrink, since the process
        groups sort their ranks), addressed with the current world's ranks: the expand KV source map."""
        rows = {r['slot']: r for r in self.session.rank_map()}
        coords = {int(slot): coord for slot, coord in self.bus.wait([f'tp2_coords/{ROUND}'])[0].items()}
        spec = dict(self.current_map().replicas)
        for ep in range(2):
            slots = sorted((s for s in coords if self.config['topology'][s]['ep'] == ep), key=lambda s: coords[s])
            spec[REPLICA_IDS['EDP0', ep]] = {'tp': 2, 'ranks': tuple(rows[slot]['new_rank'] for slot in slots)}
        return ctl.RankMap(spec)

    def identities(self):
        return {r['new_rank']: (identity(r), r['node']) for r in self.session.rank_map() if r['active']}

    @property
    def survivors(self):
        return [s for s in range(16) if self.config['topology'][s]['replica'] == 'EDP0' and s not in self.session.failed]

    def replica_leader(self):
        """Lowest surviving slot of this rank's dense replica: posts tables, handoff and the replica's summary."""
        return min(s for s in range(16) if self.config['topology'][s]['replica'] == self.domain and
                   self.config['topology'][s]['ep'] == self.row['ep'] and s not in self.session.failed)

    # ---- serving ----------------------------------------------------------------------------------------------
    def bind_model(self, model, tp):
        if self.binding is not None:
            self.binding.close()
        self.model, self.tp = model, tp
        self.binding = ModelBinding(model, self.pack['target'], self.session.resources,
                                    {'tp': tp, 'coord': self.coord, 'ep': self.row['ep']})

    def forward(self, ids, positions, context):
        return self.model(ids, positions, None, inference_context=context, runtime_gather_output=True)

    def history_source(self, seq):
        """Committed tokens of a request replayed from another replica, as published at its requeue."""
        row = self.driver.run.ledger.row(seq)
        return self.bus.wait([f'history/{seq}/{row.input_len}'])[0]

    def make_backend(self, tp, context=None):
        args = self.runtime.args
        if self.backend is None:
            self.backend = BlockBackend(self.forward, self.replica_id, tp, self.prompts, geometry=self.serving['geometry'],
                                        block_tokens=self.serving['block_tokens'], segment_tokens=self.serving['segment_tokens'],
                                        max_delta_steps=self.serving['max_delta_steps'], max_sequence_length=args.max_position_embeddings,
                                        dtype=args.params_dtype, device=self.runtime.device, attention=self.serving['attention'],
                                        history_source=self.history_source)
            if context is not None:
                self.backend.bind(tp, context=context)
        else:
            self.backend.bind(tp, context=context)
        self.backend.switcher_factory = lambda run, direction: self.switcher_class(self, run, SwitchSpec(direction))
        return self.backend

    def start_serving(self, model, tp, *, context=None, step=0, rows=(), histories=None, last=None):
        """The Driver of this rank's domain on a freshly bound model (initial TP4 load or a victim's expand target)."""
        self.bind_model(model, tp)
        backend = self.make_backend(tp, context)
        if histories:
            backend.adopt(histories, last)
        ledger = ctl.Ledger(self.current_map())
        ledger.admit([ctl.Running(**row) for row in rows])
        comm = self.domain_comm()
        control = remote_control.RemoteControl(self.bus, self.domain, ledger, comm=comm, histories=backend.histories.get,
                                               publisher=self.slot == self.replica_leader())
        hooks = None
        if self.domain == 'EDP0':  # online: EDP0 keeps its requests through the shrink, into the layout planned before serving
            degraded = (lambda: failstop_adapter.rank_map_from_rows(self.degraded_rows)) if self.online else self.current_map
            hooks = failstop_adapter.FailstopHooks(control, self.current_map, degraded, edp0_replicas=self.replicas, replay=not self.online)
        policy = POLICIES[self.serving['decode_policy']]
        policy = policy() if self.serving['decode_policy'] == 'greedy' else policy({i.seq: self.requests[i.id]['output_token_ids'] for i in self.instances})
        validation = BitwiseValidation(self.config['deterministic_mode'])
        run = DomainRun(self.domain, self.replicas, comm, backend, policy, None,
                        TimelineWriter(self.root / f'timeline_{self.domain}_{self.slot}.jsonl'),
                        SwitchEventWriter(self.root / f'switch_events_{self.domain}_{self.slot}.jsonl'), control, validation,
                        self.switcher_class.growth(self.serving), None if self.online else {'mode': 'evict', 'budget': ctl.SWITCH_BUDGET_BYTES},
                        WallClock(), step=step,
                        layout=tp, hooks=hooks, local_replicas=(self.replica_id,), domain_replicas=lambda ledger, ids=self.replicas: ids)
        self.driver = Driver(run)
        return self.driver

    def rebind_serving(self, model, tp, context):
        """Cutover inside a running Driver: the model, its binding and the backend's layout change in place."""
        self.bind_model(model, tp)
        self.backend.bind(tp, context=context)

    def phase_report(self, round_id, phase, *, empty_receiver=False):
        run = self.driver.run
        memory = {**self.runtime.memory(), 'weight_storage_bytes': self.runtime.weight_storage(self.model),
                  'kv_bytes': self.backend.kv_bytes(self.replica_id), 'pool': self.backend.memory(), 'phase': f'{round_id}/{phase}', 'layout': self.tp}
        self.memory.append(memory)
        pause = self.pauses[-1]['end_local'] - self.pauses[-1]['start_local'] if self.pauses and self.pauses[-1]['end_local'] is not None else None
        report = self.journal.event('phase_ready', slot=self.slot, replica=self.row['replica'], phase=phase, round=round_id,
                                    rank_map=self.session.rank_map(), checks_passed=True, step=run.step, sequence=0,
                                    history=remote_control.digest(run.ledger, self.replicas), empty_receiver=empty_receiver, memory=memory,
                                    layout=self.tp, rejoin_pause_s=pause)
        self.bus.put(f'phase/{round_id}/{phase}/{self.slot}', report)

    def kv_bytes(self):
        return 0 if self.backend is None else self.backend.kv_bytes(self.replica_id)

    def world_transition(self, phase, round_id):
        """Fail (1) or rejoin (2): the real world is aborted and rebuilt, the session follows the control event."""
        message = self.bus.wait([f'transition/{round_id}/{phase}'])[0]
        if self.real:
            c10d._abort_process_group()
            require(not dist.is_initialized(), 'old world still initialized after abort')
            self.runtime.world_destroyed()
            if self.healthy:
                self.pauses.append({'phase': phase, 'start_local': time.monotonic(), 'reason': 'abort/world rebuild', 'end_local': None})
        if phase == 1:
            self.session.fail(self.session.failed, ControlEvent(**message))
        else:
            self.session.rejoin(self.session.failed, ControlEvent(**message))
        self.bus.put(f'aborted/{round_id}/{phase}/{self.slot}', True)
        self.bus.wait([f'build/{round_id}/{phase}'])
        self.adopt(phase)
        if self.driver is not None:  # the serving members of the new world; the full domain after the cutover
            run = self.driver.run
            run.comm = self.domain_comm(serving_only=True)
            run.control.comm = run.comm
        if self.pauses and self.pauses[-1]['end_local'] is None:
            self.pauses[-1]['end_local'] = time.monotonic()

    def serve_until(self, key, decode_fn):
        """Foreground logical steps until the whole EDP sees the key (consensus on the model groups)."""
        deadline = time.monotonic() + self.config['stage_timeout_s']
        while not self.collective_flag(self.bus.has(key)):
            decode_fn(phase='switching')
            require(time.monotonic() < deadline, f'service timeout waiting for {key}')
            require(not self.bus.has('failure'), 'peer failure during service')

    # ---- transfers --------------------------------------------------------------------------------------------
    def plan_key(self, label):
        return f'plan/{ROUND}/{label}'

    # ---- online fail-slow switch (common.online_switch): ready before serving, migrate while serving -------------------
    def prepare_online(self, model):
        """Before serving: the survivors' TP2 model (built now, storage released until the shrink), both weight plans (the
        shrink on the profile with the fault GPUs at 1/(1+k) of their p2p bandwidth, the expand on the frozen profile: the
        co-runners are gone by then) and one exchange over every pair those plans and the KV holders use. No switch then
        builds a model, plans weights or opens a connection."""
        self.tp4_model, degraded = model, failstop_adapter.rank_map_from_rows(self.degraded_rows)
        tp4, tp2 = [s for s, _ in self.weight_sources(model, 4, self.row['etp'])], []
        if not self.healthy and not self.victim:
            self.tp2_coord = next(r['etp'] for r in self.degraded_rows if r['slot'] == self.slot)
            self.tp2_model = self.runtime.build(2, self.prebuilt[1]['target'], load=False)
            tp2 = [s for s, _ in self.weight_sources(self.tp2_model, 2, self.tp2_coord)]
            self.runtime.release(self.tp2_model, [], [])
        fault = {identity(r) for r in self.config['topology'] if r['slot'] in self.session.failed}
        factor = 1 / (1 + LEVELS[self.config['failslow']['level']])
        shrink = self.plan_before_serving('shrink', tp4, tp2, downweighted(self.profile, fault, factor))
        expand = self.plan_before_serving('expand', tp4 if self.healthy else tp2, [] if self.healthy else tp4, self.profile)
        current, heads = self.current_map(), range(self.serving['geometry']['heads'] // self.serving['geometry']['group_heads'])
        kv = {(self.builder.holder(a, replica, g), self.builder.holder(b, replica, g))
              for a, b in ((current, degraded), (degraded, current)) for replica in self.replicas_of('EDP0') for g in heads}
        pairs = {(t.source.rank, t.target.rank) for plan in (shrink, expand) for t in plan['tasks']} | kv
        warmed = self.warm_pairs({(a, b) for a, b in pairs if a != b})
        self.events.append({'event': 'online_ready', 'shrink_tasks': len(shrink['tasks']), 'expand_tasks': len(expand['tasks']),
                            'downweight': {'devices': sorted(fault), 'factor': factor}, 'p2p_pairs_warmed': warmed})

    @staticmethod
    def replicas_of(domain):
        return sorted(i for (name, _), i in REPLICA_IDS.items() if name == domain)

    def shrink_maps(self):
        """Online shrink KV: the serving TP4 layout -> the degraded TP2 layout planned before serving."""
        return self.current_map(), failstop_adapter.rank_map_from_rows(self.degraded_rows)

    def online_plan(self, label):
        """transfer()'s cached plan: the weight plan made before serving, or an empty one for a KV-only delta (online);
        None (plan at the switch) for every other scenario and method."""
        if not self.online:
            return None
        return self.plans.get(label) or {'tasks': [], 'key': f'kv/{label}', 'profile_sha256': None, 'shards': (set(), set()),
                                         'membership': sorted(identity(r) for r in self.session.rank_map() if r['active'])}

    def overlap(self, decode_fn, steps, budget):
        """execute()'s in-flight hook of the online switch (as the live benchmark): every member has enqueued this wave's
        P2P before any rank enters a model collective (Gloo barrier), then this domain decodes max_overlap_steps steps,
        at most `budget` in the whole switch (the open-block zone the delta copies). decode_fn None: barrier only."""
        def run(work):
            dist.barrier(group=self.barrier)
            for _ in range(self.serving['max_overlap_steps']):
                if decode_fn is None or len(steps) >= budget:
                    return
                decode_fn(phase='switching')
                steps.append(1)
        return run

    def moved_kv_bytes(self, moves):
        geometry = self.serving['geometry']
        return sum(m.source_tokens[2] - m.source_tokens[1] for m in moves) * geometry['group_heads'] * geometry['layers'] * \
            (geometry['k_dim'] + geometry['v_dim']) * 2

    def shrink_transfer(self, meta=None, overlap=None):
        """Survivor: an empty TP2 target filled from the residual TP4 shards and EDP1 (weights only; KV is replayed).

        Online (meta = this rank's compacted tables; fail-stop and the other fail-slow methods keep the path above):
        every EDP0 rank, victims included, also sends its A-zone KV; a survivor receives into the TP2 model built before
        serving (storage reallocated) and a TP2 pool; the weights follow the plan made before serving, the waves overlap
        decode. Returns (target, target context, stats, KV bytes)."""
        if meta is None:
            self.tp2_coord = self.coord
            budget = self.admission(2, 0)
            sources = self.weight_sources(self.model, 4, self.row['etp'])
            target = self.runtime.build(2, self.pack['target'], load=False)
            self.publish(ROUND, 'shrink', sources, self.weight_sources(target, 2, self.coord))
            leader = self.slot == min(self.survivors)
            if leader:
                self.bus.put(self.plan_key('shrink'), True)
            stats = self.transfer(ROUND, 1, 'shrink', writer=leader)
            self.events.append({'event': 'shrink', 'budget': budget, 'transfer': stats, 'kv': 'none migrated: requests replay from prompt + committed history'})
            return target, None, stats, 0
        moves = self.kv_moves(meta, ('A',), maps=self.shrink_maps())
        target = target_context = budget = None
        if not self.victim:
            budget = self.admission(2, sum(meta['lengths'].values()) + self.serving['max_delta_steps'])
            target = self.tp2_model
            self.runtime.reallocate(target)
            target_context = self.backend.new_context(2)
            self.clone_tables(target_context, meta)
        kv_sources, kv_targets = self.kv_shards(moves, self.backend.context.pool, None if target_context is None else target_context.pool,
                                                maps=self.shrink_maps())
        sources = self.weight_sources(self.model, 4, self.row['etp']) + kv_sources
        targets = ([] if target is None else self.weight_sources(target, 2, self.tp2_coord)) + kv_targets
        self.publish(ROUND, 'shrink', sources, targets)
        leader = self.slot == min(self.survivors)
        if leader:
            self.bus.put(self.plan_key('shrink'), True)
        stats = self.transfer(ROUND, 1, 'shrink', writer=leader, cached=self.online_plan('shrink'), overlap=overlap)
        kv_bytes = self.moved_kv_bytes(moves)
        self.events.append({'event': 'shrink', 'budget': budget, 'transfer': stats, 'kv_prefix_moves': len(moves), 'kv_prefix_bytes': kv_bytes})
        return target, target_context, stats, kv_bytes

    def sender_transfer(self, label, decode_fn=None, overlap=None):
        """EDP1: publish TP4 weight sources (none for a delta) and run the same waves as the receivers (online: the plan
        made before serving, decode inside the waves)."""
        sources = [] if label in ('delta', 'shrink_delta') else self.weight_sources(self.model, 4, self.row['etp'])
        self.publish(ROUND, label, sources, [])
        if decode_fn is None:
            self.bus.wait([self.plan_key(label)])
        else:
            self.serve_until(self.plan_key(label), decode_fn)
        return self.transfer(ROUND, 1 if label.startswith('shrink') else 2, label, writer=False, cached=self.online_plan(label),
                             overlap=overlap)

    def tables_meta(self, context):
        return {'tables': {str(seq): list(t.blocks) for seq, t in context.tables.items()},
                'lengths': {str(seq): t.length for seq, t in context.tables.items()}}

    def clone_tables(self, context, meta):
        """A fresh pool laid out exactly like the compacted source: same block ids per request."""
        blocks = sorted(b for table in meta['tables'].values() for b in table)
        require(blocks == list(range(len(blocks))), 'compacted tables must occupy block ids 0..n-1')
        context.pool.allocate(len(blocks))
        for seq, table in meta['tables'].items():
            context.tables[int(seq)] = block_kv.BlockTable(int(seq), list(table), meta['lengths'][seq])

    def kv_moves(self, meta, zones, assignment=None, maps=None):
        """KV moves of the given zones ('A' = full blocks, 'B' = open blocks) for the request assignment
        (default: every request stays on this rank's replica) between `maps` = (source, target) rank maps
        (default: the expand's served TP2 layout -> the current TP4 one)."""
        tables = {int(seq): block_kv.BlockTable(int(seq), list(t), meta['lengths'][seq]) for seq, t in meta['tables'].items()}
        assignment = assignment or {seq: (self.replica_id, self.replica_id) for seq in tables}
        tokens = self.builder.zone_tokens_from_tables(tables, assignment, self.serving['block_tokens'], self.serving['max_delta_steps'])
        moves = self.builder.kv_moves(*(maps or (self.tp2_map(), self.current_map())), assignment, tokens)
        return [m for m in moves if m.zone in zones]

    def kv_shards(self, moves, source_pool, target_pool, maps=None):
        dtype = self.runtime.args.params_dtype
        sources, targets = self.builder.as_shards(moves, *(maps or (self.tp2_map(), self.current_map())), self.identities(),
                                                  dtype_name=str(dtype), element_size=torch.empty(0, dtype=dtype).element_size())
        return (kv_shard_views(source_pool, sources, self.rank) if source_pool is not None else [],
                kv_shard_views(target_pool, targets, self.rank) if target_pool is not None else [])

    def expand_prefix(self, meta, source_context, zones, assignment=None, overlap=None):
        """Survivor or victim: the TP4 target (weights + the base-wave KV zones) from the TP2 survivors and EDP1 (online:
        the TP4 model of the initial load, storage reallocated, and the plan made before serving)."""
        tokens = sum(meta['lengths'].values())
        budget = self.admission(4, tokens + self.serving['max_delta_steps'])
        if self.online:
            target = self.tp4_model
            self.runtime.reallocate(target)
        else:
            target = self.runtime.build(4, self.pack['target'], load=False)
        # Survivors keep serving on their TP2 context and receive into a new one; a victim's backend is already
        # bound to an empty TP4 context, which is its target.
        target_context = self.backend.new_context(4) if source_context is not None else self.backend.context
        self.clone_tables(target_context, meta)
        moves = self.kv_moves(meta, zones, assignment)
        kv_sources, kv_targets = self.kv_shards(moves, None if source_context is None else source_context.pool, target_context.pool)
        sources = ([] if source_context is None else self.weight_sources(self.model, 2, self.tp2_coord)) + kv_sources
        targets = self.weight_sources(target, 4, self.coord) + kv_targets
        self.publish(ROUND, 'expand', sources, targets)
        leader = self.slot == min(self.survivors)
        if leader:
            self.bus.put(self.plan_key('expand'), True)
        stats = self.transfer(ROUND, 2, 'expand', writer=leader, cached=self.online_plan('expand'), overlap=overlap)
        kv_bytes = self.moved_kv_bytes(moves)
        self.events.append({'event': 'expand', 'budget': budget, 'transfer': stats, 'kv_prefix_moves': len(moves), 'kv_prefix_bytes': kv_bytes})
        return target, target_context, stats, kv_bytes

    def expand_delta(self, meta, source_context, target_context, zones, assignment=None, maps=None, label='delta'):
        """The delta-wave KV zones (WeaveTP: the open blocks, including tokens written during the base wave; AnchorTP:
        none) in the compaction layout, which the source pool keeps intact until the cutover (releases are deferred);
        the source snapshot feeds the check and the live lengths/releases are handed to the target either way.

        Requests that completed during the base wave are still copied (the layout is fixed) and released on the
        target afterwards; the survivors publish which requests live on and their current lengths. The online shrink
        delta (label 'shrink_delta', maps TP4 -> degraded TP2) has every TP4 coordinate as a source, snapshots under
        'shrink', and a victim with no target only sends."""
        if source_context is not None:
            live = {str(seq): table.length for seq, table in source_context.tables.items()
                    if seq not in (self.backend.deferred_releases or [])}
            if self.slot == self.replica_leader():
                self.bus.put(f'tables/{ROUND}/{label}/{self.replica_id}', live)
        else:
            live = self.bus.wait([f'tables/{ROUND}/{label}/{self.replica_id}'])[0]
        moves = self.kv_moves(meta, zones, assignment, maps=maps)
        kv_sources, kv_targets = self.kv_shards(moves, None if source_context is None else source_context.pool,
                                                None if target_context is None else target_context.pool, maps=maps)
        self.publish(ROUND, label, kv_sources, kv_targets)
        if source_context is not None and self.audits:
            self.save_pool_snapshot(source_context, *((self.coord, 'shrink') if label == 'shrink_delta' else ()))
        leader = self.slot == min(self.survivors)
        if leader:
            self.bus.put(self.plan_key(label), True)
        stats = self.transfer(ROUND, 1 if label == 'shrink_delta' else 2, label, writer=leader, cached=self.online_plan(label))
        for seq in [s for s in (target_context.tables if target_context is not None else ()) if str(s) not in live]:
            target_context.release(seq)
        for seq, length in (live.items() if target_context is not None else ()):
            target_context.tables[int(seq)].length = length
        return stats

    def kv_saved_key(self, coord, label):
        return f'kv_saved/{ROUND}/{self.replica_id}/{coord}' if label == 'expand' else f'kv_saved/{ROUND}/{label}/{self.replica_id}/{coord}'

    def save_pool_snapshot(self, context, coord=None, label='expand'):
        """This source coordinate's dense K/V per request (default: the expand's TP2 coordinate), for the receivers'
        independent head mapping."""
        coord = self.tp2_coord if coord is None else coord
        directory = self.root / 'kv' / f'r{ROUND}' / label / f'replica_{self.replica_id}'  # every source coordinate of both EDP0 replicas
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = {}
        for seq, table in context.tables.items():
            snapshot[seq] = {layer: tuple(t.detach().cpu().clone() for t in attention_paged.gather_request(context.pool, layer, table.blocks, table.length))
                             for layer in range(context.pool.layers)}
        torch.save({'coord': coord, 'lengths': {seq: t.length for seq, t in context.tables.items()}, 'kv': snapshot},
                   directory / f'coord_{coord}.pt')
        self.bus.put(self.kv_saved_key(coord, label), str(directory))

    def check_kv(self, target_context, mode, source_tp=2, target_tp=4, target_coord=None, label='expand'):
        """Every moved request's K/V on the target equals the head mapping of the sources' saved snapshots (default:
        the expand's TP2 survivors -> this rank's TP4 coordinate)."""
        directories = self.bus.wait([self.kv_saved_key(c, label) for c in range(source_tp)])
        snapshot = {}
        for path in sorted(Path(directories[0]).glob('coord_*.pt')):
            saved = torch.load(path, weights_only=False)
            snapshot[saved['coord']] = saved
        require(set(snapshot) == set(range(source_tp)), f'KV snapshots of all {source_tp} source coordinates are required')
        heads, local = self.serving['geometry']['heads'], self.serving['geometry']['heads'] // target_tp
        first = (self.coord if target_coord is None else target_coord) * local
        mismatches, compared = [], 0
        for seq, table in target_context.tables.items():
            length = table.length
            for layer in range(target_context.pool.layers):
                actual = attention_paged.gather_request(target_context.pool, layer, table.blocks, length)
                for kind in (0, 1):
                    pieces = []
                    for head in range(first, first + local):
                        coord, offset = divmod(head, heads // source_tp)
                        pieces.append(snapshot[coord]['kv'][seq][layer][kind][:length, offset:offset + 1])
                    compared += 1
                    if not torch.equal(actual[kind].cpu(), torch.cat(pieces, dim=1)):
                        mismatches.append({'seq': seq, 'layer': layer, 'kind': kind})
                        break
                if mismatches and mismatches[-1]['seq'] == seq:
                    break
        result = {'mode': mode, 'passed': not mismatches, 'requests': len(target_context.tables), 'tensors_compared': compared,
                  'mismatches': mismatches[:3], 'reference': "head mapping of the survivors' saved TP2 block snapshots" if label == 'expand'
                  else f"head mapping of the TP{source_tp} sources' saved block snapshots"}
        self.kv_checks.append(result)
        return result

    # ---- victims -----------------------------------------------------------------------------------------------
    def join_expand(self):
        """A victim (epoch 2, or an emulate victim after its wait) builds its TP4 target and receives weights + KV."""
        meta = self.bus.wait([f'tables/{ROUND}/{self.replica_id}'])[0]
        zones = self.switcher_class.zones
        target, target_context, stats, kv_bytes = self.expand_prefix(meta, None, zones['base'],
                                                                     overlap=self.overlap(None, [], 0) if self.online else None)
        audit = self.validate_weights(target, 4) if self.audits else {'skipped': 'timing'}
        delta = self.expand_delta(meta, None, target_context, zones['delta'])
        check = self.check_kv(target_context, 'bitwise') if self.audits else {'mode': 'bitwise', 'passed': None, 'skipped': 'timing'}
        if self.audits:
            require(check['passed'], f'migrated KV differs from the mapped source snapshots: {check["mismatches"]}')
        handoff = self.bus.wait([f'handoff/{ROUND}/{self.replica_id}'])[0]
        for seq, length in handoff['lengths'].items():
            target_context.tables[int(seq)].length = length
        self.events.append({'event': 'join', 'weights': audit, 'transfer': stats, 'delta': delta, 'kv_check': check})
        return target, target_context, handoff

    def join_as_victim(self):
        self.make_backend(4)  # an empty TP4 context: the expand target
        target, context, handoff = self.join_expand()
        self.start_serving(target, 4, context=context, step=handoff['step'], rows=handoff['rows'], histories=handoff['histories'],
                           last=handoff['last'])
        self.phase_report(ROUND, 2, empty_receiver=True)  # reported in the transition state the gate still holds
        self.session.complete()

    # ---- main flow ---------------------------------------------------------------------------------------------
    def loop(self):
        while True:
            decision = self.driver.step()
            if decision.phase == 'done':
                break
        run = self.driver.run
        run.timeline.close()
        if self.slot == self.replica_leader():  # one timeline per dense replica; the request log is domain-wide
            self.bus.put(f'summary/{self.domain}/{self.replica_id}', {
                'request_log': {str(k): v for k, v in run.request_log.items()},
                'switches': [{'index': s.index, 'direction': s.direction, 'trigger': s.trigger, 'overlap_steps': s.overlap_steps,
                              'evictions': s.evictions, 'bytes': {**s.base['bytes'], **s.delta['bytes']},
                              'validation': s.validation, 'audit': s.audit, 'milestones': {k: v['t'] for k, v in s.milestones.items()}}
                             for s in run.switches],
                'timeline': read_jsonl(run.timeline.path), 'failstop_events': run.failstop_events, 'generated': run.generated,
                'completed': run.completed, 'steps': run.step, 'attention_impl': self.backend.attention_impl, 'kv_checks': self.kv_checks,
                'layout': run.layout, 'replayed_tokens': run.hooks.replayed_tokens if run.hooks is not None else 0,
                # Worker times are run-clock seconds (origin_local on this node's monotonic clock); gate time =
                # t + origin_local + offset_to_gate (the supervisor's ping/pong calibration against SL3060).
                'clock': {'origin_local': run.clock.start, 'offset_to_gate': self.config['clock']['offset'],
                          'error_s': self.config['clock']['error']}})

    def run_worker(self):
        self.initialize()
        if self.initial_epoch == 2:
            self.session.fail(self.session.failed)
            self.session.complete()
            self.session.rejoin(self.session.failed, ControlEvent(**self.bus.wait([f'transition/{ROUND}/2'])[0]))
            self.bus.wait([f'build/{ROUND}/2'])
            self.adopt(2)
            self.runtime.start()
            self.join_as_victim()
        else:
            self.adopt(0)
            self.runtime.start()
            model = self.runtime.build(4, self.pack['target'], load=True)
            self.events.append({'event': 'load', 'weights': self.validate_weights(model, 4) if self.audits else {'skipped': 'timing'}})
            if self.online:
                self.prepare_online(model)
            self.start_serving(model, 4)
            self.phase_report(ROUND, 0)
        self.loop()
        write_json(self.root / f'serving_checks_{self.slot}.json', {
            'slot': self.slot, 'replica': self.row['replica'], 'events': self.events, 'transfers': self.transfers, 'memory': self.memory,
            'pauses': self.pauses, 'kv_checks': self.kv_checks, 'manifest': self.manifest, 'attention_impl': self.backend.attention_impl,
            'step_times': {k: v[:200] for k, v in self.backend.step_times.items()} if self.audits else 'not recorded (timing)',
            'transport': 'synchronous mixed-plan waves with every active member; Step 4 overlap is not measured here'})
        self.bus.wait(['finish'])
        if self.real:
            c10d._abort_process_group()
        else:
            dist.destroy_process_group()
        self.bus.put(f'done/{self.slot}', True)


class ServingSwitcher:
    """The perform_switch protocol on the real worker; the role decides what each phase does (WeaveTP).

    Class attributes are the method's switch semantics: `zones` = which KV zones the expand moves in the base
    wave and in the delta, `growth` = foreground steps per direction while a migration is in flight.
    """

    zones = {'base': ('A',), 'delta': ('B',)}

    @staticmethod
    def growth(serving):
        return dict(serving['growth'])

    def __init__(self, worker, run, spec):
        self.w, self.run, self.spec, self.direction = worker, run, spec, spec.direction
        self.phase = 1 if self.direction == 'shrink' else 2
        self.role = 'healthy' if worker.healthy else ('victim' if worker.victim else 'survivor')
        self.source, self.target, self.target_context, self.meta, self.handoff = worker.model, None, None, None, None
        self.steps = []  # foreground steps decoded inside this switch's waves (online)
        self.online_shrink = worker.online and self.direction == 'shrink'
        worker.bus.put(f'switch_ack/{self.direction}/{worker.slot}', True)

    def compact(self, direction):
        # Online shrink: every EDP0 rank (victims too) sends its KV, so every one fixes its compacted layout.
        if (self.role == 'survivor' and direction == 'expand') or (self.online_shrink and self.role != 'healthy'):
            context = self.w.backend.context
            self.w.backend.hold_releases()  # the compacted layout stays fixed until the cutover
            report = block_kv.compact(context, {seq: self.w.replica_id for seq in context.tables})
            self.meta = self.w.tables_meta(context)
            if direction == 'expand' and self.w.slot == self.w.replica_leader():
                self.w.bus.put(f'tables/{ROUND}/{self.w.replica_id}', self.meta)
            return {'moved_blocks': report['moved_blocks'], 'tokens': {f'{k[0]}/{k[1]}': v for k, v in report['tokens'].items()}}
        return {'moved_blocks': 0}

    def base_wave(self, direction, decode_fn):
        w = self.w
        # Online: the waves overlap decode, at most max_delta_steps - growth steps (the open-block zone the delta copies).
        overlap = w.overlap(decode_fn, self.steps, w.serving['max_delta_steps'] - self.run.growth.get(direction, 0)) if w.online else None
        if self.online_shrink:
            # Online fail-slow WeaveTP: migrate in the phase-0 world, both EDPs decode inside the waves, victims are sources;
            # the world moves to the degraded groups only at the cutover. Fail-stop and the other methods: the path below.
            if self.role == 'healthy':
                stats = w.sender_transfer('shrink', decode_fn, overlap)
                delta = w.sender_transfer('shrink_delta', decode_fn)
                return {'overlap_steps': len(self.steps), 'bytes': {'weights': stats['payload_bytes'], 'kv_prefix': 0}, 'transfer': stats,
                        'delta': delta, 'role': self.role}
            self.target, self.target_context, stats, kv_bytes = w.shrink_transfer(self.meta, overlap)
            received = kv_bytes if self.target is not None else 0
            return {'overlap_steps': len(self.steps), 'bytes': {'weights': stats['payload_bytes'] - received, 'kv_prefix': received},
                    'transfer': stats, 'role': self.role}
        if self.role == 'victim':
            if w.real:
                w.bus.wait(['finish'])
                raise RuntimeError('real victim survived SIGKILL')
            return self.emulate_victim()
        w.world_transition(self.phase, ROUND)
        if direction == 'shrink':
            if self.role == 'survivor':
                self.target, _, stats, _ = w.shrink_transfer()
            else:
                stats = w.sender_transfer('shrink')
            return {'overlap_steps': 0, 'bytes': {'weights': stats['payload_bytes'], 'kv_prefix': 0}, 'transfer': stats, 'role': self.role}
        if self.role == 'survivor':
            self.target, self.target_context, stats, kv_bytes = w.expand_prefix(self.meta, w.backend.context, self.zones['base'],
                                                                                self.spec.assignment, overlap)
            growth = self.run.growth.get('expand', 0)
            for _ in range(growth):
                decode_fn(phase='switching')
            return {'overlap_steps': len(self.steps) + growth, 'bytes': {'weights': stats['payload_bytes'] - kv_bytes, 'kv_prefix': kv_bytes},
                    'transfer': stats, 'role': self.role}
        stats = w.sender_transfer('expand', decode_fn, overlap)
        delta = w.sender_transfer('delta', decode_fn)
        return {'overlap_steps': len(self.steps), 'bytes': {'weights': stats['payload_bytes'], 'kv_prefix': 0}, 'transfer': stats, 'delta': delta,
                'role': self.role}

    def emulate_victim(self):
        """Release, wait out the degraded period as Step 5's emulate victims do, then join the expand as a target."""
        w = self.w
        w.bus.wait([f'emulate_release/{ROUND}'])
        w.binding.close()
        w.runtime.release(w.model, [], [])
        w.model = w.binding = None
        w.backend.bind(4, context=w.backend.new_context(4))  # the failed replica's blocks are gone with its model
        w.bus.put(f'released/{ROUND}/{w.slot}', {'src_none': True, 'dst_none': True, 'data_bytes': 0, 'marker_bytes': 0})
        w.world_transition(1, ROUND)
        w.session.complete()
        w.bus.wait(['spawn'])
        w.bus.put(f'cuda/2/{w.slot}', w.journal.event('empty_target_ready', gpu_uuid=w.row['gpu_uuid'], empty_receiver=True))
        w.bus.wait(['cuda_verified/2'])
        w.world_transition(2, ROUND)
        self.target, self.target_context, self.handoff = self.join()
        return {'overlap_steps': 0, 'bytes': {'weights': 0, 'kv_prefix': 0}, 'role': 'victim'}

    def join(self):
        return self.w.join_expand()

    def audit(self, direction):
        if self.role == 'survivor' and self.w.audits:
            return self.w.validate_weights(self.target, 2 if direction == 'shrink' else 4)
        return {'role': self.role, **({} if self.w.audits else {'skipped': 'timing'})}

    def delta(self, direction):
        if self.online_shrink and self.role != 'healthy':  # the open blocks, TP4 -> degraded TP2; a victim only sends
            stats = self.w.expand_delta(self.meta, self.w.backend.context, self.target_context, self.zones['delta'],
                                        maps=self.w.shrink_maps(), label='shrink_delta')
            return {'bytes': {'kv_delta': stats['payload_bytes']}, 'transfer': stats}
        if self.role == 'survivor' and direction == 'expand':
            stats = self.w.expand_delta(self.meta, self.w.backend.context, self.target_context, self.zones['delta'], self.spec.assignment)
            return {'bytes': {'kv_delta': stats['payload_bytes']}, 'transfer': stats}
        return {'bytes': {'kv_delta': 0}}

    def compare_logits(self, direction, mode):
        if self.online_shrink and self.role == 'survivor':
            return self.w.check_kv(self.target_context, mode, source_tp=4, target_tp=2, target_coord=self.w.tp2_coord, label='shrink') \
                if self.w.audits else {'mode': mode, 'passed': True, 'skipped': 'timing'}
        if self.role == 'survivor' and direction == 'expand':
            return self.w.check_kv(self.target_context, mode) if self.w.audits else {'mode': mode, 'passed': True, 'skipped': 'timing'}
        if self.role == 'victim' and direction == 'shrink':
            return {'mode': mode, 'passed': True, 'note': 'victim: checked by join_expand'}
        return {'mode': mode, 'passed': True, 'note': 'no KV moved on this rank; the A6 replay compares the records offline'}

    def cutover(self, direction):
        w, run = self.w, self.run
        if self.online_shrink:
            # Every member finished the migration and its checks; the gate then releases the victims (which wait out the
            # degraded period and join the expand, as fail-stop emulate victims) and the world moves to the prebuilt
            # degraded groups, which must be the layout planned before serving.
            w.bus.put(f'shrink_ready/{w.slot}', True)
            if self.role == 'victim':
                w.backend.flush_releases()
                self.emulate_victim()
            else:
                w.world_transition(1, ROUND)
                differing = [(a['slot'], key, a[key], b[key]) for a, b in zip(w.session.rank_map(), w.degraded_rows) for key in a if a[key] != b[key]]
                require(not differing, f'the degraded rank map differs from the one planned before serving: {differing[:4]}')
        if self.role == 'victim':
            # The emulate victim joined the expand: adopt the handoff and continue as a TP4 member of the domain.
            w.rebind_serving(self.target, 4, self.target_context)
            run.control.ledger = ctl.Ledger(w.current_map())
            run.control.ledger.admit([ctl.Running(**row) for row in self.handoff['rows']])
            w.backend.adopt(self.handoff['histories'], self.handoff['last'])
            run.step = self.handoff['step']
            return
        if self.role != 'survivor':
            return
        w.backend.flush_releases()  # the old pool goes with the source model; the target released its completed requests
        if direction == 'shrink':
            w.rebind_serving(self.target, 2, self.target_context)  # online: the migrated KV; otherwise None (requests replay)
            if w.slot == min(w.survivors):  # the TP2 coordinates every expand participant plans the KV source with
                w.bus.put(f'tp2_coords/{ROUND}', {str(r['slot']): r['etp'] for r in w.session.rank_map() if r['replica'] == 'EDP0' and r['active']})
        else:
            w.rebind_serving(self.target, 4, self.target_context)
        target_ledger = ctl.relabel(run.ledger, w.current_map())
        spec = {str(r): {'tp': s['tp'], 'ranks': list(s['ranks'])} for r, s in target_ledger.rank_map.replicas.items()}
        run.control.replace_ledger(target_ledger, 'relabel', run.replicas, rank_map=spec)
        if direction == 'expand' and w.slot == w.replica_leader():
            # The domain's whole ledger mirror (both EDP0 replicas) so the joining ranks report the same digest;
            # histories and next tokens only for the replica they will serve.
            mine = list(run.ledger.replica(w.replica_id).running.values())
            everyone = [r for replica in run.replicas for r in run.ledger.replica(replica).running.values()]
            w.bus.put(f'handoff/{ROUND}/{w.replica_id}', {
                'rows': [asdict(r) for r in everyone], 'step': run.step,
                'histories': {str(r.seq): w.backend.histories[r.seq] for r in mine},
                'last': {str(r.seq): w.backend.last_token[r.seq] for r in mine},
                'lengths': {str(seq): t.length for seq, t in self.target_context.tables.items()}})

    def release_source(self, direction):
        if self.role == 'survivor' and self.source is not None:
            self.w.runtime.release(self.source, list(self.target.parameters()), [])
            self.source = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def regroup(self, direction):
        w, run = self.w, self.run
        run.comm = w.domain_comm()
        run.control.comm = run.comm
        run.layout = w.tp
        if direction == 'shrink' and self.role == 'survivor':
            w.bus.put(f'destination/{ROUND}/1/{w.slot}', True)
        # An emulate victim's single switch spans the whole degraded period and ends in the expanded world.
        w.phase_report(ROUND, 2 if self.role == 'victim' else self.phase, empty_receiver=self.role == 'victim')
        w.session.complete()  # after the report (the gate compares rank maps in the transition state); forwards may run again


class AnchorTPSwitcher(ServingSwitcher):
    """AnchorTP semantics on the same protocol (anchortp_schedule.IMPLEMENTATION_REF): the planner takes fixed
    sources (model_worker.PLANNERS['anchortp']), EDP0 runs no foreground step while a migration is in flight
    (overlap_steps=0), the expand moves the A and B zones together at the lengths the compaction froze and has no
    delta, and the cutover checks the target tables still have those lengths. EDP1 serves between the waves and
    is a source, as for WeaveTP; the fault replay is the shared FailstopHooks."""

    zones = {'base': ('A', 'B'), 'delta': ()}

    @staticmethod
    def growth(serving):
        return {'shrink': 0, 'expand': 0}

    def cutover(self, direction):
        if self.role == 'survivor' and direction == 'expand':
            frozen = {int(seq): length for seq, length in self.meta['lengths'].items()}
            changed = [seq for seq, table in self.target_context.tables.items() if table.length != frozen[seq]]
            require(not changed, f'AnchorTP: target KV lengths changed during the paused migration: {changed[:3]}')
        super().cutover(direction)


class RestartSwitcher(ServingSwitcher):
    """Restart inside the same processes, fail-slow and emulate fail-stop (process start and NCCL init are not counted:
    optimistic for restart).
    Both switches stop EDP0 and replay its in-flight requests (shrink: the shared FailstopHooks; expand: requeued at the
    compaction), then reload the target layout from disk with the collective-free loader: TP2 on the healthy ranks,
    TP4 on all eight. Nothing migrates; EDP1 only follows the world transitions and keeps serving."""

    zones = {'base': (), 'delta': ()}

    @staticmethod
    def growth(serving):
        return {'shrink': 0, 'expand': 0}

    def compact(self, direction):
        run, hooks = self.run, self.run.hooks
        if self.role == 'survivor' and direction == 'expand':
            rows = [row for r in hooks.edp0 for row in run.ledger.replica(r).running.values()]
            hooks.replayed_tokens += sum(row.generated(run.step) + row.input_len for row in rows)
            run.control.requeue(rows, run.step)
            hooks.down.update(hooks.edp0)
            run.control.set_down(hooks.down)
        return {'moved_blocks': 0}

    def load(self, tp, direction):
        target = self.w.runtime.build(tp, self.w.pack['target'], load='local')
        self.w.bus.put(f'restart_loaded/{direction}/{self.w.slot}', time.monotonic() + self.w.config['clock']['offset'])
        return target

    def base_wave(self, direction, decode_fn):
        w = self.w
        if self.role == 'victim':
            return self.emulate_victim()
        w.world_transition(self.phase, ROUND)
        if self.role == 'survivor' and direction == 'shrink':
            w.tp2_coord = w.coord
            self.target = self.load(2, direction)
        elif self.role == 'survivor':
            w.binding.close()
            w.runtime.release(w.model, [], [])
            w.model = w.binding = self.source = None
            self.target_context = w.backend.new_context(4)
            self.target = self.load(4, direction)
        return {'overlap_steps': 0, 'bytes': {'weights': 0, 'kv_prefix': 0}, 'role': self.role}

    def join(self):
        target = self.load(4, 'expand')
        audit = self.w.validate_weights(target, 4) if self.w.audits else {'skipped': 'timing'}
        self.w.events.append({'event': 'join', 'weights': audit, 'restart': 'reloaded from disk'})
        return target, self.w.backend.context, self.w.bus.wait([f'handoff/{ROUND}/{self.w.replica_id}'])[0]

    def delta(self, direction):
        return {'bytes': {'kv_delta': 0}}

    def compare_logits(self, direction, mode):
        return {'mode': mode, 'passed': True, 'note': 'restart: no KV moved, EDP0 requests replay'}

    def regroup(self, direction):
        if self.role == 'survivor' and direction == 'expand':
            self.run.hooks.down.clear()
            self.run.control.set_down(self.run.hooks.down)
        super().regroup(direction)


SWITCHERS = {'weavetp': ServingSwitcher, 'anchortp': AnchorTPSwitcher, 'ignore': ServingSwitcher, 'restart': RestartSwitcher}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--slot', type=int, required=True)
    parser.add_argument('--epoch', type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        from model_runtime import MegatronRuntime

        ServingWorker(config, args.slot, args.epoch, MegatronRuntime.from_config(config)).run_worker()
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
