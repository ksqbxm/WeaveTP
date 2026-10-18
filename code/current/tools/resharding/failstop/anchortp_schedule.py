"""AnchorTP task ordering, copied from the reproduction branch.

Source: `codex/anchortp-faithful` commit eba3bc8a (2026-10-14), file
`code/current/megatron/core/resharding/anchortp.py` (implementation `anchortp-fixed-paths-v1`):
`schedule()` is copied verbatim (lines 81-107); CONSTRAINTS lists what `configure()` (lines 27-45)
switches off on that branch, so the fail-stop adapter records the same semantics instead of
re-deriving them. Nothing else of that module is used here: its `execute()` is bound to the
benchmark's ReshardPlan/launch_reshard_plan path, and the fail-stop worker transports through
`mixed_transport.execute` with the tasks `anchortp_plan` builds.
"""

import math

IMPLEMENTATION_REF = 'codex/anchortp-faithful@eba3bc8a megatron/core/resharding/anchortp.py (anchortp-fixed-paths-v1)'

# configure(): every WeaveTP-side optimisation the reproduction binds off, independently of shell options.
CONSTRAINTS = {
    'live_scheduler_mode': 'anchortp',
    'live_disable_source_reroute': True,
    'disabled': ('allow_aware_shrink', 'online_replan', 'adaptive_hybrid', 'hybrid_fast_path',
                 'online_migration_first_guard', 'persistent_pack_buffers', 'pack_rerouted_only',
                 'force_bandwidth_routing', 'allow_nonlocal_reroute', 'emulate_noncollocated_sources'),
    'live_pack_target_bytes': 0, 'live_pack_max_item_bytes': 0, 'live_unpack_chunk_bytes': 0,
    'live_p2p_order': 'task-round', 'live_max_waves': 0,
    'live_min_overlap_steps': 0, 'live_max_overlap_steps': 0,
}


def schedule(tasks, bandwidth, max_tasks=0, max_bytes=0):
    """Stable descending static bandwidth; caps never discard the final tasks.

    A single indivisible slice larger than max_bytes occupies its own batch.
    """
    if max_tasks < 0 or max_bytes < 0:
        raise ValueError("AnchorTP batch limits must be non-negative")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("AnchorTP requires globally unique task IDs")
    for task in tasks:
        if task.task_id < 0:
            raise ValueError("AnchorTP task IDs must be non-negative")
        value = bandwidth[task.src_rank, task.dst_rank]
        if not math.isfinite(value) or value <= 0:
            raise ValueError("AnchorTP static bandwidth must be finite and positive")
    ordered = sorted(tasks, key=lambda task: (-bandwidth[task.src_rank, task.dst_rank], task.task_id))
    batches, batch, size = [], [], 0
    for task in ordered:
        if batch and ((max_tasks and len(batch) >= max_tasks)
                      or (max_bytes and size + task.num_bytes > max_bytes)):
            batches.append(batch)
            batch, size = [], 0
        batch.append(task.task_id)
        size += task.num_bytes
    if batch:
        batches.append(batch)
    return batches
