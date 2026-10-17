"""Measured single/4/8 H2D and all directed P2P links; no fallback rates."""

import math
import statistics
import time

import torch
import torch.distributed as dist

from megatron.core.resharding.mixed_sources import digest


def fit(samples):
    if any(not r['elapsed_s'] or any(not math.isfinite(v) or v <= 0 for v in r['elapsed_s']) for r in samples):
        raise ValueError('profile samples must be finite and positive')
    points = [(r['bytes'], statistics.median(r['elapsed_s'])) for r in samples]
    if len({x for x, _ in points}) < 2 or any(x <= 0 or not math.isfinite(y) or y <= 0 for x, y in points):
        raise ValueError('profile requires positive samples at two or more sizes')
    xbar, ybar = statistics.mean(x for x, _ in points), statistics.mean(y for _, y in points)
    slope = sum((x - xbar) * (y - ybar) for x, y in points) / sum((x - xbar) ** 2 for x, _ in points)
    # Nonnegative least-squares boundary, then conservative intercept envelope.
    if slope <= 0:
        slope = min(y / x for x, y in points)
    latency = max(0, max(y - slope * x for x, y in points))
    return {'bytes_per_s': 1 / slope, 'latency_s': latency, 'samples': samples,
            'fit': 'positive_linear_slope_with_upper_median_intercept', 'unit': 'bytes/s'}


def collect(config, rows, slot, group, host, journal):
    device = lambda r: f"{r['node']}/{r['gpu_uuid']}"
    sizes = [config['source_width'] // 2, config['source_width'], config['source_width'] * 2]
    repeats = 2 if config['quick_mode'] else 5
    warmup = 1 if config['quick_mode'] else 2
    source = torch.arange(max(sizes), dtype=torch.uint8, device='cuda')
    target = torch.empty_like(source)
    # Use the actual registered backing and a contiguous byte view, no pin_memory copy.
    cpu = next(iter(host.tensors.values())).view(torch.uint8).reshape(-1)
    if cpu.numel() < max(sizes):
        raise ValueError('profile payload exceeds host shard capacity')
    stream = torch.cuda.Stream()
    profile = {'version': 'failstop-sources-v1', 'h2d': {}, 'host_capacity': {}, 'h2d_concurrency': {},
               'p2p': {device(r): {} for r in rows}, 'warmup': warmup, 'repeats': repeats,
               'sizes_bytes': sizes, 'commit': config['commit'], 'torch': str(torch.__version__),
               'cuda': torch.version.cuda, 'nccl': torch.cuda.nccl.version(), 'ranks': rows,
               'shared_gpu_mode': config['shared_gpu_mode'], 'timing_valid': not config['shared_gpu_mode'],
               'mps_isolation': 'verified_by_supervisor_and_worker_context',
               'gpu_state_evidence': ['preflight_raw.jsonl', 'mps_after_cuda_epoch0.json'],
               'clock': config['clock'], 'timing': 'maximum_selected_endpoint_local_wall_duration'}

    def measure(selected, operation):
        samples = []
        for size in sizes:
            elapsed = []
            for repetition in range(warmup + repeats):
                dist.barrier(group=group)
                torch.cuda.synchronize()
                started = time.monotonic()
                if slot in selected:
                    operation(size)
                    torch.cuda.synchronize()
                duration = time.monotonic() - started if slot in selected else 0.0
                maximum = torch.tensor(duration, dtype=torch.float64, device='cuda')
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
                if repetition >= warmup:
                    elapsed.append(maximum.item())
            samples.append({'bytes': size, 'elapsed_s': elapsed})
        return fit(samples)

    def h2d(size):
        with torch.cuda.stream(stream):
            target[:size].copy_(cpu[:size], non_blocking=True)

    for row in rows:
        profile['h2d'][device(row)] = measure([row['slot']], h2d)
        if slot == row['slot'] and not torch.equal(target.cpu(), cpu[:max(sizes)]):
            raise RuntimeError('single GPU H2D profile bitwise mismatch')
    for node in sorted({r['node'] for r in rows}):
        local = [r['slot'] for r in rows if r['node'] == node]
        profile['h2d_concurrency'][node] = {}
        for count in (4, 8):
            result = measure(local[:count], h2d)
            if slot in local[:count] and not torch.equal(target.cpu(), cpu[:max(sizes)]):
                raise RuntimeError('concurrent H2D profile bitwise mismatch')
            # Fit aggregate bytes against the observed concurrent completion.
            aggregate = fit([{**r, 'bytes': r['bytes'] * count} for r in result['samples']])
            aggregate['participants'] = local[:count]
            profile['h2d_concurrency'][node][str(count)] = aggregate
        # Conservative measured node ceiling; never single-GPU rate times eight.
        profile['host_capacity'][node] = min(profile['h2d_concurrency'][node].values(),
                                            key=lambda x: x['bytes_per_s'])
    for src in rows:
        for dst in rows:
            if src['slot'] == dst['slot']:
                continue
            source.fill_(src['slot'] * 16 + dst['slot'])
            target.fill_(255)

            def p2p(size):
                operation = (dist.P2POp(dist.isend, source[:size], dst['slot'], group) if slot == src['slot']
                             else dist.P2POp(dist.irecv, target[:size], src['slot'], group))
                for work in dist.batch_isend_irecv([operation]):
                    work.wait()

            profile['p2p'][device(src)][device(dst)] = measure([src['slot'], dst['slot']], p2p)
            if slot == dst['slot'] and not bool((target == src['slot'] * 16 + dst['slot']).all()):
                raise RuntimeError('P2P profile bitwise mismatch')
    # Existing consumers can use the decimal Gbps matrix; local diagonal is not measured.
    profile['unit'] = 'Gbps'
    profile['diagonal'] = {'gbps': 1000.0, 'measured': False, 'used_for_local_cost': False}
    profile['matrix_gbps'] = [[1000.0 if a == b else profile['p2p'][device(a)][device(b)]['bytes_per_s'] * 8 / 1e9
                              for b in rows] for a in rows]
    journal.event('profile_complete', profile_sha256=digest(profile), directed_pairs=240)
    return profile
