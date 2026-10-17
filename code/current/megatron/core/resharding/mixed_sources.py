"""Slice coverage extension of the L/S/D bandwidth-aware greedy planner.

Coordinates are global tensor coordinates, never reconstructed from world rank.
Packed/non-contiguous layouts describe each rectangular view separately. The
tensor dictionary supplied to the executor binds these views without flattening.
"""

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass, replace

from .live import ResidualBandwidthWaveScheduler, TransferTask


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Shard:
    device: str
    node: str
    rank: int | None
    kind: str
    tensor: str
    name: str
    expert: int | None
    dtype: str
    element_size: int
    shape: tuple
    start: tuple
    stop: tuple
    layout: str
    active: bool = True

    def __post_init__(self):
        if (self.kind not in ('gpu', 'host') or self.element_size <= 0 or
                not len(self.shape) == len(self.start) == len(self.stop) or
                not all(0 <= a < b <= n for a, b, n in zip(self.start, self.stop, self.shape))):
            raise ValueError('invalid shard coverage')

    @property
    def identity(self):
        return self.name, self.expert, self.dtype, self.element_size, self.shape

    def covers(self, start, stop):
        return all(a <= x < y <= b for a, b, x, y in zip(self.start, self.stop, start, stop))

    def index(self, start, stop):
        return tuple(slice(x - a, y - a) for a, x, y in zip(self.start, start, stop))


@dataclass(frozen=True)
class Task:
    id: int
    wave: int
    source: Shard
    target: Shard
    start: tuple
    stop: tuple
    payload_bytes: int

    @property
    def source_kind(self):
        return 'local' if self.source.device == self.target.device else self.source.kind


def seconds(entry, size):
    return size / entry['bytes_per_s'] + entry['latency_s']


def increments(source, target, size, profile):
    """Local: no network load. Host: D/M, no GPU sender S."""
    if source.device == target.device:
        return {}
    if source.kind == 'host':
        duration = seconds(profile['h2d'][target.device], size)
        return {('D', target.device): duration,
                ('M', target.node): seconds(profile['host_capacity'][target.node], size)}
    duration = seconds(profile['p2p'][source.device][target.device], size)
    return {('L', source.device, target.device): duration,
            ('S', source.device): duration, ('D', target.device): duration}


def build_mixed_plan(sources, targets, profile, *, membership, epoch, rank_map):
    """Greedy projected L/S/D/M load, with exact atom coverage and stable ties.

    Missing profiles are errors, including for candidates not ultimately chosen.
    Only target-node host shards are eligible. Empty/failed modules pass None.
    EpochResources owns the returned cache key and invalidates it at transitions.
    """
    sources = sorted((s for s in sources if s is not None and s.active),
                     key=lambda s: (s.device, s.tensor, s.start))
    targets = sorted((t for t in targets if t is not None and t.active),
                     key=lambda t: (t.name, -1 if t.expert is None else t.expert, t.device, t.start))
    devices = set(membership)
    for shard in sources + targets:
        if shard.kind == 'gpu' and (shard.device not in devices or
                                   rank_map.get(shard.device) != shard.rank):
            raise ValueError('shard is outside current membership/rank map')
    # Reject overlapping destination declarations before any writes are scheduled.
    for i, target in enumerate(targets):
        for other in targets[:i]:
            if (target.device == other.device and target.identity == other.identity and
                    all(max(a, c) < min(b, d) for a, b, c, d in
                        zip(target.start, target.stop, other.start, other.stop))):
                raise ValueError('overlapping target coverage')
    tasks, loads = [], {}
    for target in targets:
        candidates = [s for s in sources if s.identity == target.identity and
                      (s.kind == 'gpu' or s.node == target.node)]
        bounds = [sorted({a, b} | {v for s in candidates for v in (s.start[d], s.stop[d]) if a < v < b})
                  for d, (a, b) in enumerate(zip(target.start, target.stop))]
        for atom in itertools.product(*(list(zip(b, b[1:])) for b in bounds)):
            start, stop = tuple(a for a, _ in atom), tuple(b for _, b in atom)
            size = math.prod(b - a for a, b in atom) * target.element_size
            options = []
            for source in candidates:
                if source.covers(start, stop):
                    inc = increments(source, target, size, profile)
                    cost = max((loads.get(k, 0) + v for k, v in inc.items()), default=0)
                    options.append((cost, source.device, source.tensor, source.start, source, inc))
            if not options:
                raise ValueError(f'uncovered target: {target.device} {target.name}/{target.expert} {atom}')
            _, _, _, _, source, inc = min(options, key=lambda x: x[:4])
            for key, value in inc.items():
                loads[key] = loads.get(key, 0) + value
            tasks.append(Task(len(tasks), 0, source, target, start, stop, size))
    # Keep the existing bounded wave assignment algorithm. Stable endpoint IDs
    # keep scheduling independent of world renumbering; host endpoints are CPU
    # resources, not GPU ranks. No online feedback/replan/packing is introduced.
    endpoints = {d: i for i, d in enumerate(sorted({s.device for s in sources + targets}))}

    class FixedProfileWaves(ResidualBandwidthWaveScheduler):
        def task_seconds(self, task):
            item = tasks[task.task_id]
            return max(increments(item.source, item.target, item.payload_bytes, profile).values(), default=0)

        def _wave_count(self, items):
            return min(self.max_waves, max(1, math.ceil(len(items) / 64)))

    waves = FixedProfileWaves(None, max_waves=4).schedule([
        TransferTask(t.id, endpoints[t.source.device], endpoints[t.target.device], t.payload_bytes, t.target.name)
        for t in tasks])
    wave_by_id = {task_id: wave for wave, ids in enumerate(waves) for task_id in ids}
    tasks = [replace(t, wave=wave_by_id[t.id]) for t in tasks]
    key = digest({'membership': sorted(membership), 'epoch': epoch, 'rank_map': rank_map,
                  'sources': [asdict(s) for s in sources], 'targets': [asdict(t) for t in targets],
                  'profile_hash': digest(profile)})
    return tasks, key, loads


def normalized(tasks):
    """Compare modes without their temporary ranks or physical host backing paths."""
    rows = []
    for task in tasks:
        row = asdict(task)
        for field in ('source', 'target'):
            row[field].pop('rank')
        rows.append(row)
    return rows
