"""AnchorTP fixed-source plan over the Step 3 shard/task types.

The reproduction (see anchortp_schedule.IMPLEMENTATION_REF) plans a switch as the fixed intersection
of the source and target layouts: every target slice comes from a rank that holds the identical slice,
no bandwidth-aware re-routing, no host daemon, no packing; the executor orders the tasks by static link
bandwidth and runs them as paused batches. This module is that planner on the fail-stop worker's
inputs so the result runs through `mixed_transport.execute` unchanged:

- atoms: the same exact coverage split as `build_mixed_plan`; sources and targets are flat shard
  lists in global coordinates, so the layouts on either side are free (a replica may be split or
  merged later without changing this interface);
- source choice per atom: a GPU shard that covers it, by static distance (the target device itself,
  then its node, then another node) and then a stable order; failed GPUs are not in the membership and
  host shards are never candidates;
- waves: `anchortp_schedule.schedule` over the profile's static p2p bandwidth (local copies at the
  profile's diagonal rate), every batch one wave.

The planner interface is `plan(sources, targets, profile, *, membership, epoch, rank_map) ->
(tasks, cache_key, record)`, the same as the WeaveTP planner wrapper in model_worker.
"""

import itertools
import math
from dataclasses import asdict, replace

from anchortp_schedule import CONSTRAINTS, IMPLEMENTATION_REF, schedule

from megatron.core.resharding.live import TransferTask
from megatron.core.resharding.mixed_sources import Task, digest

LOCAL_BYTES_PER_S = 1000e9 / 8  # the profile's diagonal (1000 Gbps, never measured), see source_profile.collect
PATHS = ('local_copy', 'gpu_same_node', 'network')


def distance(source, target):
    if source.device == target.device:
        return 0
    return 1 if source.node == target.node else 2


def path_kind(source, target):
    return PATHS[distance(source, target)]


def static_bandwidth(source, target, profile):
    if source.device == target.device:
        return LOCAL_BYTES_PER_S
    return profile['p2p'][source.device][target.device]['bytes_per_s']


def build_fixed_plan(sources, targets, profile, *, membership, epoch, rank_map, max_tasks=0, max_bytes=0):
    sources = sorted((s for s in sources if s is not None and s.active and s.kind == 'gpu'),
                     key=lambda s: (s.device, s.tensor, s.start))
    targets = sorted((t for t in targets if t is not None and t.active),
                     key=lambda t: (t.name, -1 if t.expert is None else t.expert, t.device, t.start))
    devices = set(membership)
    for shard in sources + targets:
        if shard.kind != 'gpu' or shard.device not in devices or rank_map.get(shard.device) != shard.rank:
            raise ValueError('shard is outside current membership/rank map')
    for i, target in enumerate(targets):
        for other in targets[:i]:
            if (target.device == other.device and target.identity == other.identity and
                    all(max(a, c) < min(b, d) for a, b, c, d in zip(target.start, target.stop, other.start, other.stop))):
                raise ValueError('overlapping target coverage')
    tasks = []
    for target in targets:
        candidates = [s for s in sources if s.identity == target.identity]
        bounds = [sorted({a, b} | {v for s in candidates for v in (s.start[d], s.stop[d]) if a < v < b})
                  for d, (a, b) in enumerate(zip(target.start, target.stop))]
        for atom in itertools.product(*(list(zip(b, b[1:])) for b in bounds)):
            start, stop = tuple(a for a, _ in atom), tuple(b for _, b in atom)
            size = math.prod(b - a for a, b in atom) * target.element_size
            options = [(distance(s, target), s.device, s.tensor, s.start, s) for s in candidates if s.covers(start, stop)]
            if not options:
                raise ValueError(f'uncovered target: {target.device} {target.name}/{target.expert} {atom}')
            source = min(options, key=lambda x: x[:4])[4]
            tasks.append(Task(len(tasks), 0, source, target, start, stop, size))
    endpoints = {d: i for i, d in enumerate(sorted({s.device for s in sources + targets}))}
    bandwidth = {(endpoints[t.source.device], endpoints[t.target.device]): static_bandwidth(t.source, t.target, profile)
                 for t in tasks}
    batches = schedule([TransferTask(t.id, endpoints[t.source.device], endpoints[t.target.device], t.payload_bytes, t.target.name)
                        for t in tasks], bandwidth, max_tasks, max_bytes)
    wave_by_id = {task_id: wave for wave, ids in enumerate(batches) for task_id in ids}
    tasks = [replace(t, wave=wave_by_id[t.id]) for t in tasks]
    paths = {name: {'tasks': 0, 'bytes': 0} for name in PATHS}
    for task in tasks:
        stats = paths[path_kind(task.source, task.target)]
        stats['tasks'] += 1
        stats['bytes'] += task.payload_bytes
    key = digest({'planner': 'anchortp', 'membership': sorted(membership), 'epoch': epoch, 'rank_map': rank_map,
                  'sources': [asdict(s) for s in sources], 'targets': [asdict(t) for t in targets],
                  'profile_hash': digest(profile)})
    record = {'planner': 'anchortp', 'implementation_ref': IMPLEMENTATION_REF, 'constraints': CONSTRAINTS,
              'batches': len(batches), 'paths': paths, 'bandwidth_source': 'frozen profile p2p (static, bytes_per_s)'}
    return tasks, key, record
