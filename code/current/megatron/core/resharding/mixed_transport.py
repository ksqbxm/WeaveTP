"""Pinned H2D and the existing NCCLCopyService share a single completion boundary."""

import itertools
import time
from dataclasses import asdict

import torch


def execute(tasks, *, device, sources, targets, host, service_factory, rank_to_group, labels, record, resources, overlap=None):
    """Run every wave of a mixed plan on this device and return its byte totals.

    All members iterate identical waves and launch the copy service each time,
    so the one-byte participation ring never waits on an absent rank. `overlap(work)`, when given, runs between
    each wave's launch and its wait (the caller's ordering barrier and foreground decode); without it the waves
    are synchronous.
    """
    h2d = torch.cuda.Stream()
    stats = {'payload_bytes': 0, 'sent_bytes': 0, 'control_bytes': 0, 'reuse_bytes': 0}
    for wave in sorted({t.wave for t in tasks}):
        batch = f"{labels['event']}/{labels['round']}/{labels['epoch']}/{wave}"
        service = service_factory()
        rows, pending, keepalive = [], [], []
        producer = torch.cuda.current_stream()
        h2d.wait_stream(producer)
        for task in (t for t in tasks if t.wave == wave):
            source, target = task.source, task.target
            here = target.device == device
            sending = source.kind == 'gpu' and source.device == device
            if not (here or sending):
                continue
            submitted = time.monotonic()
            dest = targets[target.tensor][target.index(task.start, task.stop)] if here else None
            reuse = False
            if source.kind == 'host':
                src = host.tensors[source.tensor][source.index(task.start, task.stop)]
                if src.stride(-1) != 1:
                    raise ValueError('host manifest must expose contiguous innermost spans')
                # Row copies from pinned rows into a contiguous staging buffer stay
                # asynchronous; a strided host/destination copy_ would pack through
                # unpinned CPU memory.
                staged = torch.empty_like(dest, memory_format=torch.contiguous_format)
                keepalive.append(staged)
                with torch.cuda.stream(h2d):
                    for index in itertools.product(*(range(n) for n in src.shape[:-1])):
                        staged[index].copy_(src[index], non_blocking=True)
                    dest.copy_(staged)
            elif sending and here:
                src = sources[source.tensor][source.index(task.start, task.stop)]
                reuse = src.data_ptr() == dest.data_ptr() and src.stride() == dest.stride()
                if not reuse:
                    with torch.cuda.stream(h2d):
                        dest.copy_(src, non_blocking=True)
            elif sending:
                packed = sources[source.tensor][source.index(task.start, task.stop)].contiguous()
                keepalive.append(packed)
                service.submit_send(packed, rank_to_group[target.rank], task.id)
                stats['sent_bytes'] += task.payload_bytes
            else:
                received = torch.empty_like(dest, memory_format=torch.contiguous_format)
                keepalive.append(received)
                service.submit_recv(received, rank_to_group[source.rank], task.id)
                pending.append((dest, received))
            if here:
                stats['reuse_bytes' if reuse else 'payload_bytes'] += task.payload_bytes
                rows.append({**asdict(task), **labels, 'source_kind': task.source_kind, 'batch_id': batch,
                             'submitted_local': submitted, 'payload_bytes': 0 if reuse else task.payload_bytes,
                             'reuse_bytes': task.payload_bytes if reuse else 0, 'control_bytes': 0,
                             'host_backing': host.report['backing'] if source.kind == 'host' else None,
                             'host_owner_pid': host.report['owner_pid'] if source.kind == 'host' else None})
        service.producer_stream = producer
        work = service.launch()
        resources.pending.append(work)
        if overlap is not None:
            overlap(work)
        # H2D is already in flight while NCCL launches and waits on its own streams.
        work.wait()
        for dest, received in pending:
            dest.copy_(received)
        producer.synchronize()
        h2d.synchronize()
        resources.pending.remove(work)
        completed = time.monotonic()
        for row in rows:
            elapsed = completed - row['submitted_local']
            row.update(completed_local=completed, elapsed_s=elapsed, completion_granularity='batch',
                       effective_bytes_per_s=row['payload_bytes'] / elapsed if row['payload_bytes'] and elapsed > 0 else None)
            record(row)
        markers = service.last_launch_stats['participation_marker_bytes']
        stats['control_bytes'] += markers
        record({**labels, 'batch_id': batch, 'source_kind': 'control', 'device': device,
                'payload_bytes': 0, 'control_bytes': markers, 'completion_granularity': 'batch'})
    return stats
