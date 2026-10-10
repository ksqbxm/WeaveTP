"""CPU-only checks of a complete centralized plan, before any P2P is submitted."""

import math


def slice_shape(shape, index):
    if len(index) != len(shape):
        raise ValueError(f"Slice dimensionality mismatch: {shape}, {index}")
    result = []
    for size, part in zip(shape, index):
        if not isinstance(part, slice):
            raise ValueError(f"Expected a slice, got {part!r}")
        start = 0 if part.start is None else part.start
        stop = size if part.stop is None else part.stop
        step = 1 if part.step is None else part.step
        if not 0 <= start < stop <= size or step <= 0:
            raise ValueError(f"Out-of-bounds or empty slice: {shape}, {index}")
        result.append(len(range(start, stop, step)))
    return tuple(result)


def validate_source_metadata(selected, actual, rank):
    """A TP representative is not necessarily the owner of the transmitted slice."""
    if actual.owner_rank != rank or actual.resolved_name != selected.resolved_name:
        raise ValueError(f"Source owner/logical name mismatch on rank {rank}")
    for field in ("shape", "dtype", "element_size", "is_tp", "is_ep", "partition_dim",
                  "partition_stride", "partition_sizes", "tensor_parallel_group_ranks",
                  "global_expert_index"):
        if getattr(actual, field) != getattr(selected, field):
            raise ValueError(f"Source {selected.resolved_name} on rank {rank}: {field} mismatch")
    ep, actual_ep = selected.expert_parallel_group_ranks, actual.expert_parallel_group_ranks
    if ep is not None and (actual_ep is None or len(ep) != len(actual_ep)
                          or ep.index(selected.owner_rank) != actual_ep.index(rank)):
        raise ValueError(f"Source {selected.resolved_name} on rank {rank}: EP-local mismatch")


def validate_reshard_plans(plans, src_params, dst_params):
    """Check paired FIFO/task IDs, logical names, actual owners and slice geometry.

    Metadata must come from each rank's named_parameters(), not the chosen
    representative. Same-rank tasks are local copies, never remote P2P tasks.
    """
    sources = {rank: {} for rank in plans}
    for entries in src_params.values():
        for metadata in entries:
            sources[metadata.owner_rank][metadata.name] = metadata
    targets = {rank: {m.name: m for m in entries.values()}
               for rank, entries in dst_params.items()}
    sends, recvs = {}, {}
    seen = [set(), set()]
    local_tasks = remote_tasks = remote_bytes = 0
    for rank, plan in plans.items():
        for sending, operations, params, pairs, ids in (
            (True, plan.send_ops, sources[rank], sends, seen[0]),
            (False, plan.recv_ops, targets[rank], recvs, seen[1]),
        ):
            for op in operations:
                label = f"rank={rank} task={op.task_id} param={op.param_name}"
                if op.is_send != sending or op.task_id is None or op.task_id in ids:
                    raise ValueError(f"Invalid direction/duplicate task: {label}")
                ids.add(op.task_id)
                metadata = params.get(op.param_name)
                if metadata is None or metadata.owner_rank != rank:
                    raise ValueError(f"Parameter absent on actual owner: {label}")
                shape = slice_shape(metadata.shape, op.my_slice)
                pair = (rank, op.peer_rank) if sending else (op.peer_rank, rank)
                # Canonicalize slices by direction so both endpoint records match.
                source_slice, target_slice = ((op.my_slice, op.peer_slice) if sending
                                              else (op.peer_slice, op.my_slice))
                row = (op.task_id, metadata.resolved_name, metadata.global_expert_index,
                       metadata.dtype, shape, math.prod(shape) * metadata.element_size,
                       source_slice, target_slice)
                pairs.setdefault(pair, []).append(row)
                if sending:
                    if pair[0] == pair[1]:
                        local_tasks += 1
                    else:
                        remote_tasks += 1
                        remote_bytes += row[5]
    if sends.keys() != recvs.keys():
        raise ValueError("Send/recv peer pairs differ")
    for pair in sends:
        if sends[pair] != recvs[pair]:
            raise ValueError(f"Send/recv FIFO, task, logical name or slice mismatch: {pair}")
    per_rank = [dict(rank=rank,
                     remote_sends=sum(op.peer_rank != rank for op in plan.send_ops),
                     remote_recvs=sum(op.peer_rank != rank for op in plan.recv_ops))
                for rank, plan in plans.items()]
    return dict(local_tasks=local_tasks, remote_tasks=remote_tasks, remote_bytes=remote_bytes,
                remote_pairs=len([pair for pair in sends if pair[0] != pair[1]]), per_rank=per_rank)
