"""GPU mechanism fixture. Explicit slices are not the Step 3 source planner.

Weights and KV references are independently generated on CPU. No reference GPU
model is resident during a measured run. The native DeepSeek model is Step 5.
"""

import time
from functools import lru_cache

import torch
import torch.distributed as dist
from async_state import PeakBudget, require

from megatron.core.failstop import make_copy_service
from tools.resharding.standby_weights import StandbyWeights, release_standby, storage_bytes

HEADS, LAYERS, DIM = 8, 2, 8


def weight_reference(ep, tp, size, elements):
    offset = tp * (elements // size)
    return ((torch.arange(offset, offset + elements // size, dtype=torch.int64) % 17) + ep).float()


def kv_reference(tokens, tp, size):
    # [K/V, layer, head, token, channel]; only current valid tokens are allocated.
    kind = torch.arange(2).reshape(2, 1, 1, 1, 1) * 10000
    layer = torch.arange(LAYERS).reshape(1, LAYERS, 1, 1, 1) * 1000
    head = torch.arange(tp * HEADS // size, (tp + 1) * HEADS // size).reshape(1, 1, -1, 1, 1) * 100
    token = torch.tensor(tokens, dtype=torch.int64).reshape(1, 1, 1, -1, 1) * 10
    channel = torch.arange(DIM).reshape(1, 1, 1, 1, DIM)
    return (kind + layer + head + token + channel).contiguous()


@lru_cache(maxsize=4)
def reference_weight_sum(elements):
    return int(weight_reference(1, 0, 1, elements).to(torch.int64).sum())


def reference_logit(tokens, elements):
    return reference_weight_sum(elements) + int(kv_reference(tokens, 0, 1).sum())


class Model(torch.nn.Module):
    def __init__(self, weight, groups, tp, size):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.groups, self.tp, self.size = groups, tp, size

    def forward(self, kv, empty):
        value = self.weight.to(torch.int64).sum() + kv.sum()
        if empty:
            value.zero_()
        # The empty dense-DP branch still enters exactly the same EP collective.
        dist.all_reduce(value, group=self.groups.tp)
        dist.all_reduce(value, group=self.groups.ep)
        return value


def memory_sample(model, kv, phase, budget, retired=()):
    free, total = torch.cuda.mem_get_info()
    theoretical = sum(p.numel() * p.element_size() for p in model.parameters())
    actual = storage_bytes(model.parameters())
    require(actual == theoretical, "steady state does not contain one layout")
    require(all(s.nbytes() == 0 for s in retired), "retired weight storage still allocated")
    return {"phase": phase, "weight_storage_bytes": actual, "theoretical_weight_bytes": theoretical,
            "retired_storage_bytes": [s.nbytes() for s in retired],
            "kv_used_bytes": kv.numel() * kv.element_size(),
            "kv_valid_length": kv.shape[3], "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(), "device_used": total - free,
            "peak_allocated": torch.cuda.max_memory_allocated(), "admission_budget": budget}


def check_peak(source, kv, target_size, tokens, elements, limit):
    target_weights = elements // target_size * 4
    target_kv = 2 * LAYERS * (HEADS // target_size) * tokens * DIM * 8
    source_kv = 0 if kv is None else kv.numel() * kv.element_size()
    source_weights = 0 if source is None else storage_bytes(source.parameters())
    # Snapshot and append coexist until transport completes; integer reduction
    # scratch, NCCL receive scratch and safety margin are explicitly reserved.
    plan = PeakBudget(source_weights, target_weights, source_kv * 2,
                      target_kv * 2, target_kv, elements * 8, 64 << 20)
    plan.check(limit, torch.cuda.mem_get_info()[0], torch.cuda.memory_allocated())
    return dict(vars(plan), total=plan.total)


def release(model, protected, services):
    standby = StandbyWeights(model, protected_tensors=protected)
    storages = tuple(standby.groups)
    audit = release_standby(standby, services)
    require(all(s.nbytes() == 0 for s in storages), "standby release did not clear storage")
    return storages, audit


class Transfer:
    """Own preparation stream, source views and completion until cutover.

    A production Step 3 adapter can supply the same done()/wait() contract.
    Fixture selection is fixed: shrink from healthy TP4, expand from EDP0 TP2.
    """

    def __init__(self, session, slot, group, anchor, stream):
        self.session, self.slot, self.group = session, slot, group
        self.anchor, self.stream = anchor, stream
        self.service = make_copy_service(session, slot, group, p2p_order="task-round")
        self.service.producer_stream = stream
        self.rows = session.rank_map()
        self.ranks = dist.get_process_group_ranks(group)
        self.task, self.bytes, self.intervals = 0, [], []
        self.completed = False

    def piece(self, src_slot, dst_slot, source, target, count, identity):
        size = count
        if self.slot == src_slot:
            self.service.submit_send(source, self.ranks.index(self.rows[dst_slot]["new_rank"]), self.task)
        if self.slot == dst_slot:
            self.service.submit_recv(target, self.ranks.index(self.rows[src_slot]["new_rank"]), self.task)
        if self.slot in (src_slot, dst_slot):
            self.bytes.append({"src": src_slot, "dst": dst_slot, "bytes": size,
                               "identity": identity, "completion_granularity": "batch"})
        self.task += 1

    def weights(self, source, target, phase, elements, old_coords):
        source_size, target_size = (4, 2) if phase == 1 else (2, 4)
        sources = [r for r in self.rows if r["replica"] == ("EDP1" if phase == 1 else "EDP0") and
                   r["slot"] in old_coords]
        for dst in self.rows:
            if dst["replica"] != "EDP0" or not dst["active"]:
                continue
            start, end = dst["etp"] * elements // target_size, (dst["etp"] + 1) * elements // target_size
            for src in sources:
                if src["ep"] != dst["ep"]:
                    continue
                src_start = old_coords[src["slot"]] * elements // source_size
                lo, hi = max(start, src_start), min(end, src_start + elements // source_size)
                if lo < hi:
                    self.piece(src["slot"], dst["slot"],
                               source.weight[lo-src_start:hi-src_start] if self.slot == src["slot"] else None,
                               target.weight[lo-start:hi-start] if self.slot == dst["slot"] else None,
                               (hi-lo)*4, "weight")

    def kv(self, source, target, old_coords, start, stop):
        # Both dense DP branches carry their own request identity; EP never changes.
        for dst in self.rows:
            if dst["replica"] != "EDP0" or not dst["active"]:
                continue
            d0, d1 = dst["etp"] * HEADS // 4, (dst["etp"] + 1) * HEADS // 4
            for src in self.rows:
                if src["replica"] != "EDP0" or src["slot"] not in old_coords or src["ep"] != dst["ep"]:
                    continue
                s0 = old_coords[src["slot"]] * HEADS // 2
                lo, hi = max(d0, s0), min(d1, s0 + HEADS // 2)
                if lo < hi and stop > start:
                    for kind in range(2):
                        for layer in range(LAYERS):
                            for head in range(lo, hi):
                                self.piece(src["slot"], dst["slot"],
                                    source[kind, layer, head-s0, start:stop, :] if self.slot == src["slot"] else None,
                                    target[kind, layer, head-d0, start:stop, :] if self.slot == dst["slot"] else None,
                                    (stop-start)*DIM*8, f"kv/{kind}/{layer}/{head}/{start}:{stop}")

    def launch(self):
        self.submitted_local = time.monotonic()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.handle = self.service.launch()
        # Copy event references before wait clears the handle's timing records.
        pairs = self.handle._timing_stages
        self.events = [pair for name in ("nccl", "local") for pair in pairs.get(name, [])]
        return self

    def done(self):
        return self.handle.done()

    def wait(self):
        if self.completed:
            return True
        self.handle.wait()
        if self.events:
            self.intervals = [(self.anchor.elapsed_time(a), self.anchor.elapsed_time(b)) for a, b in self.events]
            self.events.clear()
        self.completed_local = time.monotonic()
        elapsed = self.completed_local - self.submitted_local
        for link in self.bytes:
            link.update(submitted_local=self.submitted_local, completed_local=self.completed_local,
                        seconds=elapsed, effective_bytes_per_second=link["bytes"] / elapsed)
        self.completed = True
        return True
