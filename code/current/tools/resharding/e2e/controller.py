"""Ledger, rank map, D16 regrouping and the admission controller (W1).

Semantics follow documents/e2e_prep_20261008/sim/e2e_sim.py (Replica, admit_step, regroup,
peak_ok, pre_shrink_cap) so the CPU trajectory equals the simulator's on the same input.
The ledger is keyed by replica number; the rank <-> replica mapping lives only in
`RankMap`, and the D16 formula lives only in `regroup`. Decisions are computed for step
s + 1 from the ledger state at the end of step s, which is exactly the state the
simulator admits with, so pushing them one step ahead costs no semantic lag against it.
"""

import heapq
import math
from dataclasses import dataclass, field

# Scale-up/down constants (decimal bytes), from the W1-W4 plan and e2e_sim.HW.
BLOCK_TOKENS = 64
SEGMENT_TOKENS = 4096
B_MAX = 256
PREFILL_CAP = 4096
KV_BYTES_PER_TOKEN = {2: 138_240, 4: 69_120}
CAPACITY_TOKENS = {2: 153_754, 4: 372_280}
SWITCH_BUDGET_BYTES = 15.9e9
GROWTH_STEPS = {'2->4': 7, '4->2': 23}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def effective_capacity(tp, segment_tokens=SEGMENT_TOKENS):
    """C_eff = C - one segment: admission reserves tokens, memory grows by whole segments."""
    return CAPACITY_TOKENS[tp] - segment_tokens


@dataclass
class Request:
    seq: int
    id: str
    input_len: int
    output_len: int
    kind: str = 'short'
    segment: int = 0
    arrive_s: float = 0.0
    submit_s: float = 0.0
    generated: int = 0  # tokens committed before a replay (fail-stop re-prefill keeps them)

    @property
    def reservation(self):
        return self.input_len + self.output_len


@dataclass
class Running:
    seq: int
    replica: int
    admit_step: int
    finish_step: int
    input_len: int
    output_len: int
    generated_at_admit: int = 0

    def generated(self, step):
        """Tokens produced by the start of `step` (the prefill token counts as the first)."""
        return self.generated_at_admit + max(0, min(step - self.admit_step, self.output_len - self.generated_at_admit))


class ReplicaState:
    """One dense data-parallel replica's slots and reservations (e2e_sim.Replica)."""

    def __init__(self, replica, tp, ranks):
        self.replica, self.tp, self.ranks = replica, tp, tuple(ranks)
        self.running = {}
        self.heap = []
        self.reserved_in = self.reserved_out = self.admit_sum = 0

    @property
    def n(self):
        return len(self.running)

    def reserved(self):
        return self.reserved_in + self.reserved_out

    def used(self, step):
        """KV tokens occupied at the start of `step`: inputs plus tokens generated so far."""
        return self.reserved_in + self.n * step - self.admit_sum

    def add(self, row):
        self.running[row.seq] = row
        heapq.heappush(self.heap, (row.finish_step, row.seq))
        self.reserved_in += row.input_len
        self.reserved_out += row.output_len
        self.admit_sum += row.admit_step

    def remove(self, seq):
        row = self.running.pop(seq)
        self.reserved_in -= row.input_len
        self.reserved_out -= row.output_len
        self.admit_sum -= row.admit_step
        self.heap = [entry for entry in self.heap if entry[1] != seq]
        heapq.heapify(self.heap)
        return row

    def pop_done(self, step):
        done = []
        while self.heap and self.heap[0][0] <= step:
            _, seq = heapq.heappop(self.heap)
            done.append(self.running.pop(seq))
            self.reserved_in -= done[-1].input_len
            self.reserved_out -= done[-1].output_len
            self.admit_sum -= done[-1].admit_step
        return done


@dataclass(frozen=True)
class RankMap:
    """rank -> (replica, tp coordinate); the only place that knows which ranks form a replica."""

    replicas: dict  # replica -> {'tp': int, 'ranks': tuple}

    def replica_of(self, rank):
        for replica, spec in self.replicas.items():
            if rank in spec['ranks']:
                return replica, spec['ranks'].index(rank)
        raise KeyError(rank)

    def head_groups(self, rank, heads=16, group_heads=4):
        """Head groups (of `group_heads` heads) this rank holds: TP2 ranks hold two, TP4 ranks one."""
        replica, coord = self.replica_of(rank)
        per_rank = heads // self.replicas[replica]['tp'] // group_heads
        return tuple(range(coord * per_rank, (coord + 1) * per_rank))

    @staticmethod
    def dense(world_size, tp):
        """Scale-up/down layout: rank = tp_coord + tp * replica (HyperCommGrid order)."""
        return RankMap({i: {'tp': tp, 'ranks': tuple(range(i * tp, (i + 1) * tp))} for i in range(world_size // tp)})


class Ledger:
    def __init__(self, rank_map):
        self.rank_map = rank_map
        self.replicas = {r: ReplicaState(r, spec['tp'], spec['ranks']) for r, spec in rank_map.replicas.items()}

    def replica(self, replica):
        return self.replicas[replica]

    def row(self, seq):
        for state in self.replicas.values():
            if seq in state.running:
                return state.running[seq]
        raise KeyError(seq)

    def completions(self, step):
        return [row for state in self.replicas.values() for row in state.pop_done(step)]

    def snapshot(self, step):
        return {r: {'n': s.n, 'reserved': s.reserved(), 'used': s.used(step), 'tp': s.tp, 'ranks': list(s.ranks)}
                for r, s in self.replicas.items()}


def regroup(ledger, to_tp, world_size=None):
    """D16, the only implementation: 2->4 merges replicas 2i and 2i+1; 4->2 hands requests in
    seq order to the child with the smaller reservation. Returns a new Ledger with its RankMap."""
    old = ledger.rank_map
    replicas = sorted(old.replicas)
    if to_tp == 4:
        specs = {i: {'tp': 4, 'ranks': old.replicas[replicas[2 * i]]['ranks'] + old.replicas[replicas[2 * i + 1]]['ranks']}
                 for i in range(len(replicas) // 2)}
        new = Ledger(RankMap(specs))
        for i in range(len(replicas) // 2):
            rows = list(ledger.replica(replicas[2 * i]).running.values()) + list(ledger.replica(replicas[2 * i + 1]).running.values())
            for row in rows:
                new.replica(i).add(Running(row.seq, i, row.admit_step, row.finish_step, row.input_len, row.output_len, row.generated_at_admit))
        return new
    specs = {}
    for i, replica in enumerate(replicas):
        ranks = old.replicas[replica]['ranks']
        half = len(ranks) // 2
        specs[2 * i] = {'tp': 2, 'ranks': ranks[:half]}
        specs[2 * i + 1] = {'tp': 2, 'ranks': ranks[half:]}
    new = Ledger(RankMap(specs))
    for i, replica in enumerate(replicas):
        for row in sorted(ledger.replica(replica).running.values(), key=lambda r: r.seq):
            kids = (new.replica(2 * i), new.replica(2 * i + 1))
            child = 2 * i if kids[0].reserved() <= kids[1].reserved() else 2 * i + 1
            new.replica(child).add(Running(row.seq, child, row.admit_step, row.finish_step, row.input_len, row.output_len, row.generated_at_admit))
    return new


def relabel(ledger, rank_map):
    """Fail-stop: the same replicas change TP size and rank set (shrink/expand); requests stay put."""
    new = Ledger(rank_map)
    for replica, state in ledger.replicas.items():
        for row in state.running.values():
            new.replica(replica).add(row)
    return new


def used_bytes(tokens, tp, segment_tokens=SEGMENT_TOKENS):
    """Physical KV bytes grow by whole segments."""
    return math.ceil(tokens / segment_tokens) * segment_tokens * KV_BYTES_PER_TOKEN[tp]


def peak_check(ledger, step, direction, growth, budget=SWITCH_BUDGET_BYTES, segment_tokens=SEGMENT_TOKENS):
    """Per-card used-plus-growth peak of source and destination KV (e2e_sim.peak_ok, bytes by segment)."""
    replicas = sorted(ledger.replicas)
    if direction == '2->4':
        for i in range(len(replicas) // 2):
            a, b = ledger.replica(replicas[2 * i]), ledger.replica(replicas[2 * i + 1])
            ua, ub = a.used(step) + a.n * growth, b.used(step) + b.n * growth
            if max(used_bytes(ua, 2, segment_tokens), used_bytes(ub, 2, segment_tokens)) + used_bytes(ua + ub, 4, segment_tokens) > budget:
                return False
        return True
    kids = regroup(ledger, 2)
    for i, replica in enumerate(replicas):
        source = ledger.replica(replica)
        us = source.used(step) + source.n * growth
        uk = max(kids.replica(c).used(step) + kids.replica(c).n * growth for c in (2 * i, 2 * i + 1))
        if used_bytes(us, 4, segment_tokens) + used_bytes(uk, 2, segment_tokens) > budget:
            return False
    return True


def pre_shrink_cap(growth, b_max=B_MAX, cap_tp2=None, budget=SWITCH_BUDGET_BYTES):
    """Reservation cap on each TP4 replica after the last long request is admitted."""
    cap_tp2 = effective_capacity(2) if cap_tp2 is None else cap_tp2
    return min(int(0.95 * 2 * cap_tp2), int(0.95 * budget / (2 * KV_BYTES_PER_TOKEN[4])) - b_max * (growth + 1))


@dataclass
class Decision:
    for_step: int
    admits: list = field(default_factory=list)          # (seq, replica, input_len)
    prefill_tokens: dict = field(default_factory=dict)  # replica -> tokens prefilled this step
    completions: list = field(default_factory=list)     # seqs finishing at the end of this step
    phase: str = 'normal'
    head_block_reason: str = 'none'
    need_prefill_forward: bool = False
    sleep_until: float | None = None
    evictions: list = field(default_factory=list)       # seqs pushed back to the queue head


class AdmissionController:
    """Strict FCFS admission over a ledger; one instance owns the whole request queue."""

    def __init__(self, ledger, queue, *, capacity, b_max=B_MAX, prefill_cap=PREFILL_CAP, barrier=False):
        self.ledger, self.queue = ledger, list(queue)
        self.capacity, self.b_max, self.prefill_cap, self.barrier = capacity, b_max, prefill_cap, barrier
        self.pointer = 0
        self.segment_left = {}
        for request in self.queue:
            self.segment_left[request.segment] = self.segment_left.get(request.segment, 0) + 1
        self.done = set()

    def complete(self, step, replicas=None):
        """Requests of `replicas` (default all) whose finish step is `step`; once at the end of every domain step."""
        rows = [row for r in (self.ledger.replicas if replicas is None else replicas) for row in self.ledger.replica(r).pop_done(step)]
        for row in rows:
            self.done.add(row.seq)
            self.segment_left[self.queue_index[row.seq].segment] -= 1
        return rows

    @property
    def queue_index(self):
        if not hasattr(self, '_index'):
            self._index = {r.seq: r for r in self.queue}
        return self._index

    def requeue(self, rows, step):
        """Fail-stop replay and peak eviction: rows return to the queue head in FCFS order,
        keeping their original submit time and the tokens already generated."""
        rows = sorted(rows, key=lambda r: r.seq)
        for row in rows:
            request = self.queue_index[row.seq]
            request.generated = row.generated(step)
            self.ledger.replica(row.replica).remove(row.seq)
        self.queue[self.pointer:self.pointer] = [self.queue_index[r.seq] for r in rows]

    def decide(self, step, now_s, *, replicas=None, freeze_seq=None, frozen=False, throttled_cap=None, phase='normal'):
        """Admissions for `step` into `replicas` (a step domain; default all) from the current ledger
        (end of step - 1), e2e_sim.admit_step. Strict FCFS: the queue head waits for any replica of this
        domain and nothing behind it is considered."""
        replicas = sorted(self.ledger.replicas if replicas is None else replicas)
        decision = Decision(for_step=step, phase=phase)
        decision.prefill_tokens = {r: 0 for r in replicas}
        if frozen:
            decision.phase, decision.head_block_reason = phase if phase != 'normal' else 'paused_admission', 'anchor'
            return self.finish(decision, step, replicas)
        while self.pointer < len(self.queue):
            request = self.queue[self.pointer]
            if request.seq == freeze_seq:
                decision.head_block_reason = 'anchor'
                break
            if request.arrive_s > now_s + 1e-9:
                decision.head_block_reason = 'arrival'
                if not any(self.ledger.replica(r).n for r in replicas):
                    decision.sleep_until = request.arrive_s
                break
            if self.barrier and request.segment > 0 and self.segment_left.get(request.segment - 1, 0) > 0:
                decision.head_block_reason = 'barrier'
                break
            need = request.input_len + request.output_len - request.generated
            best, kv_short = None, False
            for replica in replicas:
                state = self.ledger.replica(replica)
                cap = self.capacity(replica)
                if throttled_cap is not None:
                    cap = min(cap, throttled_cap)
                if state.n >= self.b_max:
                    continue
                if state.reserved() + need > cap:
                    kv_short = True
                    continue
                if decision.prefill_tokens[replica] and decision.prefill_tokens[replica] + request.input_len > self.prefill_cap:
                    continue
                if best is None or state.reserved() < self.ledger.replica(best).reserved():
                    best = replica
            if best is None:
                decision.head_block_reason = 'kv' if kv_short else 'bmax'
                break
            row = Running(request.seq, best, step, step + request.output_len - request.generated - 1,
                          request.input_len + request.generated, request.output_len, request.generated)
            self.ledger.replica(best).add(row)
            decision.admits.append((request.seq, best, row.input_len))
            decision.prefill_tokens[best] += row.input_len
            self.pointer += 1
        return self.finish(decision, step, replicas)

    def finish(self, decision, step, replicas):
        decision.need_prefill_forward = bool(decision.admits)
        decision.completions = sorted(row.seq for r in replicas for row in self.ledger.replica(r).running.values()
                                      if row.finish_step == step)
        return decision

    def idle(self, replicas=None):
        replicas = self.ledger.replicas if replicas is None else replicas
        return self.pointer >= len(self.queue) and not any(self.ledger.replica(r).n for r in replicas)

    def evict_for_peak(self, step, direction, growth, budget=SWITCH_BUDGET_BYTES):
        """Peak shortfall degrades to recorded eviction: newest admissions first, until the budget fits."""
        evicted = []
        while not peak_check(self.ledger, step, direction, growth, budget):
            rows = [row for state in self.ledger.replicas.values() for row in state.running.values()]
            require(rows, 'peak budget cannot be met even with every request evicted')
            victim = max(rows, key=lambda r: (r.admit_step, r.seq))
            self.requeue([victim], step)
            evicted.append(victim.seq)
        return evicted
