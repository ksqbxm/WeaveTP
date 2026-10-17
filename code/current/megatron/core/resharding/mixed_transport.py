"""Pinned H2D and existing NCCLCopyService share a single completion boundary."""

import itertools
import time
from dataclasses import asdict

import torch


def execute(tasks, *, device, sources, targets, host, service_factory, rank_to_group,
            event, round_id, epoch, record, resources):
    h2d = torch.cuda.Stream()
    stats = {'payload_bytes': 0, 'sent_bytes': 0, 'control_bytes': 0, 'reuse_bytes': 0}
    for wave in sorted({t.wave for t in tasks}):
        batch = f'{event}/{round_id}/{epoch}/{wave}'
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
            src = None
            reuse = False
            if source.kind == 'host' and here:
                src = host.tensors[source.tensor][source.index(task.start, task.stop)]
                # Explicit contiguous host rows avoid implicit unpinned CPU
                # packing for strided host/destination views in Tensor.copy_.
                staged = torch.empty_like(dest, memory_format=torch.contiguous_format)
                keepalive.append(staged)
                if src.stride(-1) != 1:
                    raise ValueError('host manifest must expose contiguous innermost spans')
                with torch.cuda.stream(h2d):
                    for index in itertools.product(*(range(n) for n in src.shape[:-1])):
                        staged[index].copy_(src[index], non_blocking=True)
                    dest.copy_(staged)
            elif sending or here:
                if sending:
                    src = sources[source.tensor][source.index(task.start, task.stop)]
                if sending and here:
                    reuse = src.data_ptr() == dest.data_ptr() and src.stride() == dest.stride()
                    if not reuse:
                        with torch.cuda.stream(h2d):
                            dest.copy_(src, non_blocking=True)
                else:
                    if sending:
                        packed = src.contiguous()
                        keepalive.append(packed)
                        service.submit_send(packed, rank_to_group[target.rank], task.id)
                        stats['sent_bytes'] += task.payload_bytes
                    if here:
                        received = torch.empty_like(dest, memory_format=torch.contiguous_format)
                        keepalive.append(received)
                        service.submit_recv(received, rank_to_group[source.rank], task.id)
                        pending.append((dest, received))
            if here:
                stats['reuse_bytes' if reuse else 'payload_bytes'] += task.payload_bytes
                rows.append({**asdict(task), 'source_kind': task.source_kind,
                             'event': event, 'round': round_id, 'epoch': epoch, 'batch_id': batch,
                             'submitted_local': submitted, 'payload_bytes': 0 if reuse else task.payload_bytes,
                             'reuse_bytes': task.payload_bytes if reuse else 0, 'control_bytes': 0,
                             'host_backing': host.report['backing'] if source.kind == 'host' else None,
                             'host_owner_pid': host.report['owner_pid'] if source.kind == 'host' else None})
        service.producer_stream = producer
        work = service.launch()
        resources.pending.append(work)
        # H2D is already in flight while NCCL launches/waits on its own streams.
        work.wait()
        for dest, received in pending:
            dest.copy_(received)
        producer.synchronize()
        h2d.synchronize()
        resources.pending.remove(work)
        completed = time.monotonic()
        for row in rows:
            elapsed = completed - row['submitted_local']
            row.update(completed_local=completed, elapsed_s=elapsed,
                       effective_bytes_per_s=row['payload_bytes'] / elapsed if row['payload_bytes'] and elapsed > 0 else None,
                       completion_granularity='batch')
            record(row)
        markers = service.last_launch_stats['participation_marker_bytes']
        stats['control_bytes'] += markers
        record({'event': event, 'round': round_id, 'epoch': epoch, 'batch_id': batch,
                'source_kind': 'control', 'device': device, 'payload_bytes': 0,
                'control_bytes': markers, 'completion_granularity': 'batch'})
    return stats
