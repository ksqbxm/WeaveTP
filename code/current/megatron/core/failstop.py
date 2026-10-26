"""Opt-in fail-stop membership and replica groups (PP=CP=1).

Device identity and source shard coordinates survive world renumbering. No
model tensor or ParameterMetadata is rewritten by this module.
"""

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import permutations


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def choose_local_order(devices, reusable_bytes):
    """Exact enumeration for one EP: matrix[stable id][target local rank]."""
    stable = sorted(devices)
    require(stable and len(set(stable)) == len(stable), "invalid EP membership")
    require(all(len(reusable_bytes[d]) == len(stable) and
                all(v >= 0 for v in reusable_bytes[d]) for d in stable), "invalid reuse bytes")
    # Sorted enumeration gives a deterministic stable-device tie break.
    return max(permutations(stable), key=lambda order: sum(
        reusable_bytes[d][rank] for rank, d in enumerate(order)))


def overlap_bytes(source, target):
    """Coverage entries are (canonical tensor/expert identity, byte start, byte stop).

    Callers supply actual contiguous tensor spans, including packed/strided
    shards as separate non-overlapping entries. Rank names are not identities.
    """
    return sum(max(0, min(b, d) - max(a, c))
               for name, a, b in source for other, c, d in target if name == other)


@dataclass(frozen=True)
class ControlEvent:
    launch: str
    round: int
    epoch: int
    action: str
    id: str


class FaultBackend:
    """Only the fault mechanics differ. Callbacks must complete their CPU gates.

    real: local supervisor SIGKILL/reclaim/spawn + worker abort/rebuild.
    emulate: release local state/isolated data groups + empty receiving target.
    kill: SIGKILL/reclaim/spawn as real; the degraded world keeps the prebuilt groups (emulate's
    ranks), only the rejoin aborts and rebuilds the world.
    A callback exception leaves the session stopped, never apparently recovered.
    """

    def __init__(self, mode=None, *, fail, rejoin):
        self.mode = mode or os.environ.get("FAILSTOP_MODE", "emulate")
        require(self.mode in ("emulate", "real", "kill"), "FAILSTOP_MODE must be emulate, real or kill")
        self.fail, self.rejoin = fail, rejoin

    def rebuilds(self, phase):
        """Whether the process world of `phase` (0 initial, 1 degraded, 2 rejoined) is built anew."""
        return self.mode == "real" or (self.mode == "kill" and phase == 2)


class EpochResources:
    """Single safe boundary for forward, work disposal, and epoch caches."""

    def __init__(self):
        self.lock = threading.RLock()
        self.plans, self.groups, self.pending = {}, {}, []
        self.dispositions = []
        self.key = None
        self.ready = False
        self.generation = 0

    @contextmanager
    def forward(self, key):
        with self.lock:
            require(self.ready and key == self.key, "forward on stale or unavailable world")
            yield

    def invalidate(self, abort=None):
        with self.lock:
            self.ready = False
            self.generation += 1
            if abort is not None:
                abort()  # Must finish abort before discarding any old Work.
                disposition = "discarded_after_abort"
            else:
                for work in self.pending:
                    require(work.wait() is not False, "old communication work failed")
                disposition = "completed_before_group_release"
            self.dispositions.append({"key": self.key, "count": len(self.pending),
                                      "disposition": disposition})
            self.pending.clear()
            self.plans.clear()
            self.groups.clear()
            self.key = None

    def activate(self, key):
        with self.lock:
            require(not self.pending, "cannot activate with old pending work")
            self.key, self.ready = key, True


class FaultSession:
    def __init__(self, rows, launch, backend, *, resources=None, round_id=0, local_orders=None):
        self.rows = tuple(dict(r) for r in rows)
        require(len(self.rows) == 16 and len({r['gpu_uuid'] for r in rows}) == 16,
                "expected 16 distinct devices")
        require([r['slot'] for r in rows] == list(range(16)), "slot must index initial world")
        for replica, node in (("EDP0", "SL3061"), ("EDP1", "SL3060")):
            selected = [r for r in rows if r['replica'] == replica]
            require(len(selected) == 8 and all(r['node'] == node for r in selected) and
                    {(r['ep'], r['etp']) for r in selected} ==
                    {(ep, tp) for ep in range(2) for tp in range(4)}, "invalid fixed scenario")
        self.launch, self.round, self.epoch = launch, round_id, 0
        self.backend, self.resources = backend, resources or EpochResources()
        self.state, self.seen = "normal", set()
        self.actual_local_ranks = {}
        self.failed = tuple(r['slot'] for r in rows if r['replica'] == 'EDP0' and r['etp'] in (0, 1))
        self.members = tuple(range(16))
        self.previous = self.members
        self.local_orders = local_orders if local_orders is not None else {
            ep: tuple(r['slot'] for r in rows if r['replica'] == 'EDP0' and
                      r['ep'] == ep and r['slot'] not in self.failed) for ep in range(2)}
        require(set(self.local_orders) == {0, 1}, 'invalid local order EP keys')
        for ep, order in self.local_orders.items():
            require(len(order) == 2 and set(order) == {r['slot'] for r in rows if r['replica'] == 'EDP0' and
                                  r['ep'] == ep and r['slot'] not in self.failed}, "invalid local order")
        self.resources.activate(self.cache_key())

    def event(self, action):
        epoch = self.epoch + 1
        return ControlEvent(self.launch, self.round, epoch, action,
                            f"{self.launch}/{self.round}/{epoch}/{action}")

    def _change(self, action, ranks, event):
        with self.resources.lock:
            expected = "normal" if action == "fail" else "degraded"
            require(self.state == expected, f"invalid transition {self.state} -> {action}")
            require(event == self.event(action) and event.id not in self.seen, "stale/foreign control event")
            # fail uses current world ranks; rejoin uses original ranks of absent devices.
            victims = self.failed if action == "rejoin" else tuple(
                r['new_rank'] for r in self.rank_map() if r['slot'] in self.failed)
            require(len(ranks) == 4 and set(ranks) == set(victims), "incorrect fault coordinates")
            self.resources.ready = False
            self.state = "failed/recovering" if action == "fail" else "rejoining/expanding"
            getattr(self.backend, action)(self.failed, event)
            require(not self.resources.pending, "backend left old communication work")
            self.resources.invalidate()
            self.previous = self.members
            if action == "fail":
                self.members = tuple(r['slot'] for r in self.rows if r['replica'] == 'EDP1') + tuple(
                    slot for ep in range(2) for slot in self.local_orders[ep])
            else:
                self.members = tuple(range(16))
            self.epoch = event.epoch
            self.actual_local_ranks.clear()
            self.seen.add(event.id)
        return self.current_world()

    def fail(self, ranks, event=None):
        return self._change("fail", ranks, event or self.event("fail"))

    def rejoin(self, ranks, event=None):
        return self._change("rejoin", ranks, event or self.event("rejoin"))

    def complete(self):
        with self.resources.lock:
            require(self.state in ("failed/recovering", "rejoining/expanding"), "no transition to complete")
            self.resources.activate(self.cache_key())
            self.state = "degraded" if self.state == "failed/recovering" else "normal"

    def next_round(self):
        with self.resources.lock:
            require(self.state == "normal" and self.epoch > 0 and self.resources.ready and not self.resources.pending,
                    "round before full recovery")
            self.round += 1
            self.resources.plans.clear()
            self.resources.groups.clear()
            self.resources.activate(self.cache_key())

    def rank_map(self):
        result = []
        for r in self.rows:
            slot = r['slot']
            active = slot in self.members
            compact = self.backend.mode == 'real'
            rank = self.members.index(slot) if active and compact else slot if active else None
            old = self.previous.index(slot) if slot in self.previous and compact else (
                slot if slot in self.previous else None)
            tp = self.local_orders[r['ep']].index(slot) if active and len(self.members) == 12 and r['replica'] == 'EDP0' else r['etp']
            tp = self.actual_local_ranks.get(slot, tp)
            result.append({**r, 'source_etp': r['etp'], 'etp': tp if active else None,
                           'world_epoch': self.epoch, 'old_rank': old, 'new_rank': rank,
                           'active': active, 'source_eligible': active and not (
                               self.state == 'rejoining/expanding' and slot in self.failed)})
        return result

    def current_world(self):
        return {'epoch': self.epoch, 'membership': self.members,
                'ranks': tuple(r['new_rank'] for r in self.rank_map() if r['active']),
                'process_world_size': len(self.members) if self.backend.mode == 'real' else 16}

    def cache_key(self):
        return (self.launch, self.round, self.epoch, self.members,
                tuple((r['gpu_uuid'], r['new_rank'], r['ep'], r['etp']) for r in self.rank_map()))

    def modules(self, slot, source, target):
        row = self.rank_map()[slot]
        return (source if row['source_eligible'] else None, target if row['active'] else None)

    def replicas(self):
        rows = self.rank_map()
        return [(tuple(r['new_rank'] for r in sorted(rows, key=lambda r: (r['ep'], r['etp'] or 0))
                       if r['replica'] == replica and r['active']),
                 2 if replica == 'EDP0' and len(self.members) == 12 else 4,
                 2 if replica == 'EDP0' and len(self.members) == 12 else 4, 2)
                for replica in ('EDP0', 'EDP1')]


    def adopt_group_order(self, records):
        """Reconcile API sorting before allocating a target or using a cached plan."""
        before = self.cache_key()
        for row in self.rank_map():
            if not row['active']:
                continue
            replica = 0 if row['replica'] == 'EDP0' else 1
            actual = {}
            for field in ('ep', 'expt_tp', 'tp'):
                matches = [r for r in records if r['replica'] == replica and
                           r['field'] == field and row['new_rank'] in r['actual']]
                require(len(matches) == 1, 'missing actual group membership')
                actual[field] = matches[0]['actual'].index(row['new_rank'])
            require(actual['ep'] == row['ep'], 'actual group order changed EP expert ownership')
            require(actual['tp'] == actual['expt_tp'], 'fixed scenario requires TP/ETP alignment')
            self.actual_local_ranks[row['slot']] = actual['expt_tp']
        if self.cache_key() != before:
            self.resources.ready = False
            self.resources.plans.clear()
            self.resources.groups.clear()


def replica_group_specs(replicas):
    """Ordered group plan; explicit input order, TP fastest, PP=CP=1."""
    seen, specs = set(), []
    for replica, (ranks, tp, etp, ep) in enumerate(replicas):
        ranks = tuple(ranks)
        n = len(ranks)
        require(n and min(tp, etp, ep) > 0 and n % tp == n % (etp * ep) == 0,
                "invalid replica dimensions")
        require(len(set(ranks)) == n and not seen.intersection(ranks), "overlapping replica ranks")
        seen.update(ranks)
        dense = [ranks[i:i + tp] for i in range(0, n, tp)]
        expert = [ranks[i:i + etp] for i in range(0, n, etp)]
        # new_group sorts each TP group. Pair DP/EP ranks by those actual
        # coordinates, not by an input order that the backend will discard.
        dense_order = tuple(r for group in dense for r in sorted(group))
        expert_order = tuple(r for group in expert for r in sorted(group))
        groups = {
            'tp': dense, 'cp': [(r,) for r in ranks], 'pp': [(r,) for r in ranks],
            'dp': [dense_order[i::tp] for i in range(tp)], 'mp': dense, 'tp_cp': dense,
            'dp_cp': [dense_order[i::tp] for i in range(tp)], 'tp_dp_cp': [ranks],
            'ep': [tuple(expert_order[b + i + j * etp] for j in range(ep))
                   for b in range(0, n, etp * ep) for i in range(etp)],
            'expt_tp': expert, 'expt_dp': [expert_order[i::etp * ep] for i in range(etp * ep)],
            'tp_ep': [ranks[b:b + etp * ep] for b in range(0, n, etp * ep)],
            'tp_ep_pp': [ranks[b:b + etp * ep] for b in range(0, n, etp * ep)],
            'embd': [(r,) for r in ranks], 'pos_embd': [(r,) for r in ranks],
        }
        specs.extend((replica, name, members) for name, lists in groups.items() for members in lists)
    return specs


def build_replica_pg_collections(replicas, *, backend=None, timeout=None, records=None):
    """Every current WORLD rank calls this, including nonmembers.

    PyTorch 2.11 sorts new_group ranks. Actual order is recorded and used in
    local_ranks; source metadata is independent. No process-group cache spans
    calls/epochs. Identical groups *within* this call share a communicator.
    """
    import torch.distributed as dist

    from megatron.core.process_groups_config import ProcessGroupCollection

    rank, size = dist.get_rank(), dist.get_world_size()
    collections = [None for _ in replicas]
    created = {}
    for replica, name, members in replica_group_specs(replicas):
        require(all(0 <= r < size for r in members), 'replica rank outside current world')
        # Explicit canonicalization models the actual 2.11 API before creation.
        canonical = tuple(sorted(members))
        if canonical not in created:
            kwargs = {'ranks': list(canonical), 'backend': backend}
            if timeout is not None:
                kwargs['timeout'] = timeout
            created[canonical] = dist.new_group(**kwargs)
        group = created[canonical]
        actual = tuple(dist.get_process_group_ranks(group)) if rank in members else canonical
        require(actual == canonical, 'unexpected actual PG order')
        if records is not None:
            records.append({'replica': replica, 'field': name, 'requested': list(members),
                            'actual': list(actual), 'backend': backend or dist.get_backend(),
                            'member': rank in members})
        if rank in members:
            if collections[replica] is None:
                collections[replica] = ProcessGroupCollection()
                collections[replica].local_ranks = {}
            setattr(collections[replica], name, group)
            collections[replica].local_ranks[name] = actual.index(rank)
            require(dist.get_rank(group) == actual.index(rank), 'group local rank mismatch')
    return collections


def make_copy_service(session, slot, group, **kwargs):
    """The one-byte participation ring is restricted to effective membership."""
    import torch.distributed as dist

    from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService

    row = session.rank_map()[slot]
    require(row['active'], 'failed GPU cannot create a data service')
    require(group is not None and set(dist.get_process_group_ranks(group)) ==
            set(session.current_world()['ranks']), 'data group includes absent GPUs or wrong membership')
    return NCCLCopyService(group=group, ensure_all_ranks_participate=True,
                           group_relative_ranks=True, **kwargs)
