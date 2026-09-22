# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional

import torch
import torch.distributed as dist

from .base import CopyHandle, CopyService

logger = logging.getLogger(__name__)


def _least_stream_priority() -> int:
    """Return the lowest CUDA stream priority supported by this PyTorch build."""
    get_priority_range = getattr(torch.cuda, "get_stream_priority_range", None)
    if get_priority_range is None:
        return 0
    least_priority, _greatest_priority = get_priority_range()
    return int(least_priority)


def _block_current_stream_on_requests(requests) -> None:
    """Make the active CUDA stream wait for every asynchronous P2P work item."""
    for request in requests:
        block_current_stream = getattr(request, "block_current_stream", None)
        if block_current_stream is not None:
            block_current_stream()
        else:
            # Older PyTorch releases implement wait() as a nonblocking CUDA
            # stream dependency when no timeout is supplied.
            request.wait()


@dataclass
class SendOp:
    """Simple container describing a single send operation."""

    task_id: int | None
    tensor: torch.Tensor
    dest_rank: int


@dataclass
class RecvOp:
    """Simple container describing a single receive operation."""

    task_id: int | None
    tensor: torch.Tensor
    src_rank: int


class NCCLCopyHandle(CopyHandle):
    """Own NCCL requests and CUDA events until one asynchronous batch completes."""

    def __init__(
        self,
        requests,
        events: list[torch.cuda.Event],
        timing_groups: list[list[tuple[torch.cuda.Event, torch.cuda.Event]]],
        timing_stages: Mapping[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ],
        keepalive: list[torch.Tensor],
        on_complete: Callable[[], None],
    ) -> None:
        self._requests = list(requests)
        self._events = list(events)
        self._timing_groups = [list(group) for group in timing_groups]
        self._timing_stages = {
            str(name): list(pairs) for name, pairs in timing_stages.items()
        }
        self._keepalive = keepalive
        self._on_complete = on_complete
        self._completed = False
        self._elapsed_s: Optional[float] = None
        self._stage_elapsed_s: dict[str, float] = {}
        self._lock = threading.Lock()

    def _finish_once(self) -> None:
        with self._lock:
            if self._completed:
                return
            group_seconds = [
                sum(start.elapsed_time(end) for start, end in group) / 1.0e3
                for group in self._timing_groups
            ]
            self._stage_elapsed_s = {
                name: sum(start.elapsed_time(end) for start, end in pairs) / 1.0e3
                for name, pairs in self._timing_stages.items()
            }
            self._elapsed_s = max(group_seconds, default=0.0)
            self._completed = True
            self._requests.clear()
            self._events.clear()
            self._timing_groups.clear()
            self._timing_stages.clear()
            self._keepalive.clear()
            self._on_complete()

    def done(self) -> bool:
        if self._completed:
            return True
        requests_done = all(request.is_completed() for request in self._requests)
        events_done = all(event.query() for event in self._events)
        # Completion polling must remain a pure query. In particular, it must
        # never enqueue or synchronize receive unpacking on the decode thread.
        return requests_done and events_done

    def wait(self) -> None:
        if self._completed:
            return
        # Launch records these events only after the comm stream has been made
        # dependent on every NCCL Work. Synchronizing the final event therefore
        # covers direct receives and any queued unpacking.
        for event in self._events:
            event.synchronize()
        for request in self._requests:
            request.wait()
        self._finish_once()

    def elapsed_seconds(self) -> Optional[float]:
        return self._elapsed_s if self._completed else None

    def stage_elapsed_seconds(self) -> Optional[dict[str, float]]:
        if not self._completed:
            return None
        return dict(self._stage_elapsed_s)


class NCCLCopyService(CopyService):
    """
    Thin wrapper around torch.distributed batch_isend_irecv to submit and execute
    a batch of point-to-point sends and recvs.
    """

    def __init__(
        self,
        group=None,
        p2p_order: Optional[str] = None,
        pack_target_bytes: int = 0,
        pack_max_item_bytes: int = 0,
        persistent_pack_buffers: bool = False,
        pack_rerouted_only: bool = False,
        unpack_chunk_bytes: int = 64 << 20,
        ensure_all_ranks_participate: bool = False,
    ):
        self.group = group
        # Use group.rank()/size() to support cross-cluster ProcessGroups
        self.rank = group.rank() if group is not None else dist.get_rank()
        self.world_size = group.size() if group is not None else dist.get_world_size()
        self.send_ops: List[SendOp] = []
        self.recv_ops: List[RecvOp] = []
        # Dedicated stream for local (same-rank) copies to avoid unnecessary
        # serialization with work on the default stream.
        self._copy_stream = torch.cuda.Stream()
        self._comm_stream = torch.cuda.Stream()
        self._unpack_stream = torch.cuda.Stream(priority=_least_stream_priority())
        self._launch_callback = None
        self._inflight: Optional[NCCLCopyHandle] = None
        self.pack_target_bytes = max(int(pack_target_bytes), 0)
        self.pack_max_item_bytes = max(int(pack_max_item_bytes), 0)
        if self.pack_target_bytes and not self.pack_max_item_bytes:
            self.pack_max_item_bytes = self.pack_target_bytes
        self.persistent_pack_buffers = bool(persistent_pack_buffers)
        self.pack_rerouted_only = bool(pack_rerouted_only)
        self.unpack_chunk_bytes = max(int(unpack_chunk_bytes), 0)
        self.ensure_all_ranks_participate = bool(ensure_all_ranks_participate)
        self._participation_send = None
        self._participation_recv = None
        if self.ensure_all_ranks_participate and self.world_size > 1:
            self._participation_send = torch.zeros(
                1,
                dtype=torch.uint8,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            self._participation_recv = torch.empty_like(self._participation_send)
        self._pack_task_ids: frozenset[int] | None = None
        self._cacheable_pack_task_ids: frozenset[int] | None = None
        self._send_pack_cache: dict[tuple, torch.Tensor] = {}
        self._recv_pack_cache: dict[tuple, torch.Tensor] = {}
        self.last_launch_stats: dict[str, object] = {}
        self.p2p_order = p2p_order.lower() if p2p_order is not None else None
        valid_orders = {
            "send-recv",
            "nccl-round",
            "task-round",
            "small-first",
            "large-first",
            "peer-size-asc",
            "peer-size-desc",
        }
        if self.p2p_order is not None and self.p2p_order not in valid_orders:
            raise ValueError(
                f"Unsupported NCCL P2P order {self.p2p_order!r}; "
                f"expected one of {sorted(valid_orders)}"
            )
        if self.rank == 0:
            logger.info(f"NCCLCopyService initialized with {self.world_size} ranks")

    def set_launch_callback(self, callback) -> None:
        self._launch_callback = callback

    def configure_plan(self, plan) -> None:
        """Select task IDs eligible for packing in the next launch."""
        plan_ops = list(plan.send_ops) + list(plan.recv_ops)
        self._cacheable_pack_task_ids = frozenset(
            int(op.task_id)
            for op in plan_ops
            if op.task_id is not None and not op.param_name.startswith("kv::")
        )
        if not self.pack_rerouted_only:
            self._pack_task_ids = None
            return
        rerouted_task_ids = getattr(plan, "rerouted_task_ids", None)
        self._pack_task_ids = (
            frozenset() if rerouted_task_ids is None else frozenset(rerouted_task_ids)
        )

    def invalidate_persistent_pack_cache(self) -> None:
        """Discard inference-only packed snapshots after source weights mutate."""
        if self._inflight is not None:
            raise RuntimeError("cannot invalidate packing buffers during an in-flight batch")
        self._send_pack_cache.clear()
        self._recv_pack_cache.clear()

    def _notify_launched(self) -> None:
        if self._launch_callback is not None:
            self._launch_callback()

    def submit_send(self, src_tensor: torch.Tensor, dest_rank: int, task_id: Optional[int] = None):
        self.send_ops.append(SendOp(task_id=task_id, tensor=src_tensor, dest_rank=dest_rank))

    def submit_recv(self, dest_tensor: torch.Tensor, src_rank: int, task_id: Optional[int] = None):
        self.recv_ops.append(RecvOp(task_id=task_id, tensor=dest_tensor, src_rank=src_rank))

    def _build_send_recv_p2p_ops(self, remote_sends: List[SendOp], remote_recvs: List[RecvOp]):
        p2p_ops = []
        for op in remote_sends:
            p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group))
        for op in remote_recvs:
            p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group))
        return p2p_ops

    def _op_num_bytes(self, op) -> int:
        return int(op.tensor.numel() * op.tensor.element_size())

    def _task_sort_key(self, task_id: Optional[int], fallback_index: int):
        if task_id is None:
            return (1, fallback_index)
        return (0, int(task_id))

    def _partition_pack_groups(self, ops, peer_attr: str):
        """Group small, task-addressable tensors into deterministic peer buckets."""
        if not self.pack_target_bytes:
            return [[op] for op in ops]

        peer_queues = {}
        for index, op in enumerate(ops):
            peer = int(getattr(op, peer_attr))
            peer_queues.setdefault(peer, []).append((index, op))

        groups = []
        for peer in sorted(peer_queues):
            ordered = sorted(
                peer_queues[peer],
                key=lambda item: self._task_sort_key(item[1].task_id, item[0]),
            )
            direct_groups = []
            packable_ops = []
            selected_task_ids = getattr(self, "_pack_task_ids", None)
            for index, op in ordered:
                num_bytes = self._op_num_bytes(op)
                packable = (
                    op.task_id is not None
                    and (selected_task_ids is None or op.task_id in selected_task_ids)
                    and 0 < num_bytes <= self.pack_max_item_bytes
                    and num_bytes <= self.pack_target_bytes
                )
                if packable:
                    packable_ops.append((index, op))
                else:
                    direct_groups.append([op])

            peer_groups = list(direct_groups)
            current = []
            current_bytes = 0
            current_cacheable = None
            cacheable_task_ids = getattr(self, "_cacheable_pack_task_ids", None)

            def flush_current():
                nonlocal current, current_bytes, current_cacheable
                if current:
                    peer_groups.append(current)
                    current = []
                    current_bytes = 0
                    current_cacheable = None

            for _index, op in packable_ops:
                num_bytes = self._op_num_bytes(op)
                cacheable = (
                    cacheable_task_ids is None or op.task_id in cacheable_task_ids
                )
                if current and (
                    current_bytes + num_bytes > self.pack_target_bytes
                    or current_cacheable != cacheable
                ):
                    flush_current()
                current.append(op)
                current_bytes += num_bytes
                current_cacheable = cacheable
            flush_current()
            peer_groups.sort(
                key=lambda group: self._task_sort_key(group[0].task_id, 0)
            )
            groups.extend(peer_groups)
        return groups

    @staticmethod
    def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_contiguous():
            raise ValueError("NCCL packing requires contiguous submitted tensors")
        return tensor.view(torch.uint8).reshape(-1)

    @staticmethod
    def _foreach_byte_copy(
        destinations: list[torch.Tensor], sources: list[torch.Tensor]
    ) -> None:
        if not destinations:
            return
        if len(destinations) != len(sources):
            raise ValueError("NCCL packing source/destination count mismatch")
        foreach_copy = getattr(torch, "_foreach_copy_", None)
        if foreach_copy is not None:
            foreach_copy(destinations, sources)
            return
        for destination, source in zip(destinations, sources):
            destination.copy_(source)

    def _foreach_byte_copy_bounded(
        self, destinations: list[torch.Tensor], sources: list[torch.Tensor]
    ) -> None:
        """Submit bounded foreach groups to avoid one monolithic unpack kernel."""
        if not destinations:
            return
        if len(destinations) != len(sources):
            raise ValueError("NCCL packing source/destination count mismatch")
        if not self.unpack_chunk_bytes:
            self._foreach_byte_copy(destinations, sources)
            return

        chunk_destinations = []
        chunk_sources = []
        chunk_bytes = 0
        for destination, source in zip(destinations, sources):
            num_bytes = int(source.numel() * source.element_size())
            if chunk_sources and chunk_bytes + num_bytes > self.unpack_chunk_bytes:
                self._foreach_byte_copy(chunk_destinations, chunk_sources)
                chunk_destinations = []
                chunk_sources = []
                chunk_bytes = 0
            chunk_destinations.append(destination)
            chunk_sources.append(source)
            chunk_bytes += num_bytes
        self._foreach_byte_copy(chunk_destinations, chunk_sources)

    def _pack_remote_ops(self, remote_sends: List[SendOp], remote_recvs: List[RecvOp]):
        """Coalesce small remote tensors and return deferred receive writebacks."""
        send_groups = self._partition_pack_groups(remote_sends, "dest_rank")
        recv_groups = self._partition_pack_groups(remote_recvs, "src_rank")
        packed_sends = []
        packed_recvs = []
        recv_writebacks = []
        pack_destinations = []
        pack_sources = []
        packed_send_messages = 0
        packed_send_items = 0
        packed_send_bytes = 0
        packed_recv_messages = 0
        packed_recv_items = 0
        send_cache_hits = 0
        send_cache_misses = 0
        recv_cache_hits = 0
        recv_cache_misses = 0
        reused_send_bytes = 0
        mutable_send_bytes = 0
        persistent = bool(getattr(self, "persistent_pack_buffers", False))
        send_cache = getattr(self, "_send_pack_cache", None)
        recv_cache = getattr(self, "_recv_pack_cache", None)
        if send_cache is None:
            send_cache = {}
            self._send_pack_cache = send_cache
        if recv_cache is None:
            recv_cache = {}
            self._recv_pack_cache = recv_cache

        def cache_key(group, peer_attr: str) -> tuple:
            device = group[0].tensor.device
            return (
                int(getattr(group[0], peer_attr)),
                device.type,
                device.index,
                tuple(
                    (
                        int(op.task_id),
                        self._op_num_bytes(op),
                        int(op.tensor.data_ptr()),
                    )
                    for op in group
                ),
            )

        for group in send_groups:
            if len(group) == 1:
                packed_sends.append(group[0])
                continue
            total_bytes = sum(self._op_num_bytes(op) for op in group)
            key = cache_key(group, "dest_rank")
            cacheable_task_ids = getattr(self, "_cacheable_pack_task_ids", None)
            cacheable = cacheable_task_ids is None or all(
                op.task_id in cacheable_task_ids for op in group
            )
            use_send_cache = persistent and cacheable
            buffer = send_cache.get(key) if use_send_cache else None
            if buffer is None:
                buffer = torch.empty(
                    total_bytes, dtype=torch.uint8, device=group[0].tensor.device
                )
                offset = 0
                for op in group:
                    source = self._byte_view(op.tensor)
                    num_bytes = source.numel()
                    pack_destinations.append(buffer[offset : offset + num_bytes])
                    pack_sources.append(source)
                    offset += num_bytes
                if use_send_cache:
                    send_cache[key] = buffer
                    send_cache_misses += 1
                elif persistent:
                    mutable_send_bytes += total_bytes
            else:
                send_cache_hits += 1
                reused_send_bytes += total_bytes
            packed_sends.append(
                SendOp(task_id=group[0].task_id, tensor=buffer, dest_rank=group[0].dest_rank)
            )
            packed_send_messages += 1
            packed_send_items += len(group)
            packed_send_bytes += self._op_num_bytes(packed_sends[-1])

        for group in recv_groups:
            if len(group) == 1:
                packed_recvs.append(group[0])
                continue
            total_bytes = sum(self._op_num_bytes(op) for op in group)
            key = cache_key(group, "src_rank")
            buffer = recv_cache.get(key) if persistent else None
            if buffer is None:
                buffer = torch.empty(
                    total_bytes, dtype=torch.uint8, device=group[0].tensor.device
                )
                if persistent:
                    recv_cache[key] = buffer
                    recv_cache_misses += 1
            else:
                recv_cache_hits += 1
            offset = 0
            for op in group:
                num_bytes = self._op_num_bytes(op)
                recv_writebacks.append((op.tensor, buffer, offset, num_bytes))
                offset += num_bytes
            packed_recvs.append(
                RecvOp(task_id=group[0].task_id, tensor=buffer, src_rank=group[0].src_rank)
            )
            packed_recv_messages += 1
            packed_recv_items += len(group)

        self._foreach_byte_copy(pack_destinations, pack_sources)

        stats = {
            "original_remote_sends": len(remote_sends),
            "original_remote_recvs": len(remote_recvs),
            "coalesced_remote_sends": len(packed_sends),
            "coalesced_remote_recvs": len(packed_recvs),
            "packed_send_messages": packed_send_messages,
            "packed_recv_messages": packed_recv_messages,
            "packed_send_items": packed_send_items,
            "packed_recv_items": packed_recv_items,
            "unpack_items": len(recv_writebacks),
            "unpack_bytes": sum(item[3] for item in recv_writebacks),
            "packed_send_bytes": packed_send_bytes,
            "persistent_pack_buffers": persistent,
            "send_pack_cache_hits": send_cache_hits,
            "send_pack_cache_misses": send_cache_misses,
            "recv_pack_cache_hits": recv_cache_hits,
            "recv_pack_cache_misses": recv_cache_misses,
            "reused_send_bytes": reused_send_bytes,
            "mutable_send_bytes": mutable_send_bytes,
            "pack_task_count": (
                -1
                if getattr(self, "_pack_task_ids", None) is None
                else len(self._pack_task_ids)
            ),
        }
        return packed_sends, packed_recvs, recv_writebacks, stats

    def _build_task_round_p2p_ops(self, remote_sends: List[SendOp], remote_recvs: List[RecvOp]):
        """Build P2P ops in global task-id order without changing tensor storage.

        Each reshard task has the same task_id on the sender and receiver.  Sorting
        both sends and receives by that id keeps each peer-pair FIFO-compatible,
        while avoiding the all-sends-then-all-recvs phase split from the default
        order.  This is intentionally zero-copy: tensors are submitted exactly as
        received by the service.
        """
        ordered_ops = []
        for index, op in enumerate(remote_sends):
            ordered_ops.append(
                (
                    self._task_sort_key(op.task_id, index),
                    0,
                    index,
                    dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group),
                )
            )
        send_count = len(remote_sends)
        for index, op in enumerate(remote_recvs):
            ordered_ops.append(
                (
                    self._task_sort_key(op.task_id, send_count + index),
                    1,
                    index,
                    dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group),
                )
            )
        ordered_ops.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in ordered_ops]

    def _build_size_round_p2p_ops(
        self,
        remote_sends: List[SendOp],
        remote_recvs: List[RecvOp],
        *,
        reverse: bool,
    ):
        """Build P2P ops by tensor size, using task_id as a deterministic tie-breaker."""
        ordered_ops = []
        for index, op in enumerate(remote_sends):
            num_bytes = self._op_num_bytes(op)
            ordered_ops.append(
                (
                    num_bytes,
                    self._task_sort_key(op.task_id, index),
                    0,
                    index,
                    dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group),
                )
            )
        send_count = len(remote_sends)
        for index, op in enumerate(remote_recvs):
            num_bytes = self._op_num_bytes(op)
            ordered_ops.append(
                (
                    num_bytes,
                    self._task_sort_key(op.task_id, send_count + index),
                    1,
                    index,
                    dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group),
                )
            )
        ordered_ops.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=reverse)
        return [item[4] for item in ordered_ops]

    def _build_peer_size_p2p_ops(
        self,
        remote_sends: List[SendOp],
        remote_recvs: List[RecvOp],
        *,
        reverse: bool,
    ):
        """Keep send-recv phases, but sort each peer FIFO by tensor size."""

        def _sort_peer_queues(ops, peer_attr: str):
            queues = {}
            for index, op in enumerate(ops):
                peer = int(getattr(op, peer_attr))
                queues.setdefault(peer, []).append((index, op))
            ordered = []
            for peer in sorted(queues):
                peer_ops = queues[peer]
                peer_ops.sort(
                    key=lambda item: (
                        self._op_num_bytes(item[1]),
                        self._task_sort_key(item[1].task_id, item[0]),
                        item[0],
                    ),
                    reverse=reverse,
                )
                ordered.extend(op for _, op in peer_ops)
            return ordered

        p2p_ops = []
        for op in _sort_peer_queues(remote_sends, "dest_rank"):
            p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group))
        for op in _sort_peer_queues(remote_recvs, "src_rank"):
            p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group))
        return p2p_ops

    def _build_nccl_round_p2p_ops(self, remote_sends: List[SendOp], remote_recvs: List[RecvOp]):
        """Build P2P ops in an NCCL-like peer-round order.

        NCCL queues P2P tasks per peer and schedules them by repeatedly walking
        a rank-local send/recv peer round.  Preserve each peer FIFO queue, but
        interleave sends and receives by cyclic peer distance so both sides of a
        peer pair enter the same group round.
        """
        send_queues: dict[int, List[SendOp]] = {}
        recv_queues: dict[int, List[RecvOp]] = {}
        for op in remote_sends:
            send_queues.setdefault(int(op.dest_rank), []).append(op)
        for op in remote_recvs:
            recv_queues.setdefault(int(op.src_rank), []).append(op)

        p2p_ops = []
        while send_queues or recv_queues:
            progressed = False
            for distance in range(1, max(self.world_size, 1)):
                send_peer = (self.rank + distance) % self.world_size
                recv_peer = (self.rank - distance) % self.world_size

                send_queue = send_queues.get(send_peer)
                if send_queue:
                    op = send_queue.pop(0)
                    p2p_ops.append(
                        dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group)
                    )
                    progressed = True
                    if not send_queue:
                        del send_queues[send_peer]

                recv_queue = recv_queues.get(recv_peer)
                if recv_queue:
                    op = recv_queue.pop(0)
                    p2p_ops.append(
                        dist.P2POp(
                            dist.irecv, op.tensor, op.src_rank, group=self.group
                        )
                    )
                    progressed = True
                    if not recv_queue:
                        del recv_queues[recv_peer]

            if progressed:
                continue

            # Fallback for non-contiguous or non-standard peer ids. Preserve
            # FIFO order rather than spinning forever.
            for peer in sorted(send_queues):
                op = send_queues[peer].pop(0)
                p2p_ops.append(dist.P2POp(dist.isend, op.tensor, op.dest_rank, group=self.group))
                if not send_queues[peer]:
                    del send_queues[peer]
                break
            for peer in sorted(recv_queues):
                op = recv_queues[peer].pop(0)
                p2p_ops.append(dist.P2POp(dist.irecv, op.tensor, op.src_rank, group=self.group))
                if not recv_queues[peer]:
                    del recv_queues[peer]
                break

        return p2p_ops

    def launch(self) -> CopyHandle:
        if self._inflight is not None:
            raise RuntimeError("NCCLCopyService already has an in-flight copy batch")

        total_ops = len(self.send_ops) + len(self.recv_ops)
        if self.rank == 0:
            logger.info(
                "Executing batched communication: %d sends + %d recvs = %d ops",
                len(self.send_ops),
                len(self.recv_ops),
                total_ops,
            )

        local_sends = [op for op in self.send_ops if op.dest_rank == self.rank]
        remote_sends = [op for op in self.send_ops if op.dest_rank != self.rank]
        local_recvs = [op for op in self.recv_ops if op.src_rank == self.rank]
        remote_recvs = [op for op in self.recv_ops if op.src_rank != self.rank]

        completion_events: list[torch.cuda.Event] = []
        timing_groups: list[list[tuple[torch.cuda.Event, torch.cuda.Event]]] = []
        timing_stages: dict[
            str, list[tuple[torch.cuda.Event, torch.cuda.Event]]
        ] = {}
        if local_sends or local_recvs:
            local_sends_by_id = {op.task_id: op for op in local_sends}
            if None in local_sends_by_id:
                raise RuntimeError(
                    "NCCLCopyService: local (same-rank) transfer requires a task_id "
                    "to match sends with recvs"
                )
            local_recvs_by_id = {op.task_id: op for op in local_recvs}
            if None in local_recvs_by_id:
                raise RuntimeError(
                    "NCCLCopyService: local (same-rank) transfer requires a task_id "
                    "to match sends with recvs"
                )
            if len(local_sends_by_id) != len(local_sends) or len(local_recvs_by_id) != len(
                local_recvs
            ):
                raise RuntimeError(
                    f"NCCLCopyService: unmatched local ops on rank {self.rank}: "
                    f"{len(local_sends)} local sends vs {len(local_recvs)} local recvs"
                )
            local_start = torch.cuda.Event(enable_timing=True)
            local_event = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                with torch.cuda.stream(self._copy_stream):
                    local_start.record(self._copy_stream)
                    for task_id, recv_op in local_recvs_by_id.items():
                        send_op = local_sends_by_id.get(task_id)
                        if send_op is None:
                            raise RuntimeError(
                                f"NCCLCopyService: missing local send for task_id={task_id} "
                                f"on rank {self.rank}"
                            )
                        recv_op.tensor.copy_(send_op.tensor)
                    local_event.record(self._copy_stream)
            completion_events.append(local_event)
            timing_groups.append([(local_start, local_event)])
            timing_stages["local"] = [(local_start, local_event)]

        original_remote_tensors = [op.tensor for op in remote_sends]
        original_remote_tensors.extend(op.tensor for op in remote_recvs)
        pack_start = None
        pack_end = None
        if self.pack_target_bytes and (remote_sends or remote_recvs):
            producer_stream = torch.cuda.current_stream()
            pack_start = torch.cuda.Event(enable_timing=True)
            pack_end = torch.cuda.Event(enable_timing=True)
            pack_start.record(producer_stream)
            remote_sends, remote_recvs, recv_writebacks, pack_stats = self._pack_remote_ops(
                remote_sends, remote_recvs
            )
            pack_end.record(producer_stream)
        else:
            recv_writebacks = []
            pack_stats = {
                "original_remote_sends": len(remote_sends),
                "original_remote_recvs": len(remote_recvs),
                "coalesced_remote_sends": len(remote_sends),
                "coalesced_remote_recvs": len(remote_recvs),
                "packed_send_messages": 0,
                "packed_recv_messages": 0,
                "packed_send_items": 0,
                "packed_recv_items": 0,
                "packed_send_bytes": 0,
                "unpack_items": 0,
                "unpack_bytes": 0,
            }
        # A residual-aware wave may have no model transfers on one rank.  If
        # that rank skips batch_isend_irecv, NCCL ProcessGroup sequence numbers
        # diverge and a later wave can deadlock.  A one-byte ring exchange keeps
        # every rank in the same P2P group without changing model state or the
        # model-transfer byte counters above.
        if self.ensure_all_ranks_participate and self.world_size > 1:
            marker_task_id = -1
            remote_sends.append(
                SendOp(
                    marker_task_id,
                    self._participation_send,
                    (self.rank + 1) % self.world_size,
                )
            )
            remote_recvs.append(
                RecvOp(
                    marker_task_id,
                    self._participation_recv,
                    (self.rank - 1) % self.world_size,
                )
            )
            pack_stats["participation_marker_bytes"] = 2
        else:
            pack_stats["participation_marker_bytes"] = 0
        self.last_launch_stats = pack_stats

        p2p_order = self.p2p_order or os.environ.get(
            "NCCL_COPY_P2P_ORDER", "send-recv"
        ).lower()
        if p2p_order == "nccl-round":
            p2p_ops = self._build_nccl_round_p2p_ops(remote_sends, remote_recvs)
        elif p2p_order == "task-round":
            p2p_ops = self._build_task_round_p2p_ops(remote_sends, remote_recvs)
        elif p2p_order == "small-first":
            p2p_ops = self._build_size_round_p2p_ops(remote_sends, remote_recvs, reverse=False)
        elif p2p_order == "large-first":
            p2p_ops = self._build_size_round_p2p_ops(remote_sends, remote_recvs, reverse=True)
        elif p2p_order == "peer-size-asc":
            p2p_ops = self._build_peer_size_p2p_ops(remote_sends, remote_recvs, reverse=False)
        elif p2p_order == "peer-size-desc":
            p2p_ops = self._build_peer_size_p2p_ops(remote_sends, remote_recvs, reverse=True)
        else:
            p2p_ops = self._build_send_recv_p2p_ops(remote_sends, remote_recvs)

        reqs = []
        remote_event = None
        remote_group = []
        if p2p_ops:
            remote_start = torch.cuda.Event(enable_timing=True)
            remote_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._comm_stream):
                if pack_end is not None:
                    self._comm_stream.wait_event(pack_end)
                remote_start.record(self._comm_stream)
                reqs = dist.batch_isend_irecv(p2p_ops)
                _block_current_stream_on_requests(reqs)
                remote_event.record(self._comm_stream)
            if pack_stats["packed_send_messages"] or pack_stats["packed_recv_messages"]:
                remote_group.append((pack_start, pack_end))
                timing_stages["pack"] = [(pack_start, pack_end)]
            remote_group.append((remote_start, remote_event))
            timing_stages["nccl"] = [(remote_start, remote_event)]

        if recv_writebacks:
            if remote_event is None:
                raise RuntimeError("packed receives require a remote completion event")
            unpack_start = torch.cuda.Event(enable_timing=True)
            unpack_end = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                with torch.cuda.stream(self._unpack_stream):
                    self._unpack_stream.wait_event(remote_event)
                    unpack_start.record(self._unpack_stream)
                    destinations = []
                    sources = []
                    for destination, packed, offset, num_bytes in recv_writebacks:
                        destinations.append(self._byte_view(destination))
                        sources.append(packed[offset : offset + num_bytes])
                    self._foreach_byte_copy_bounded(destinations, sources)
                    unpack_end.record(self._unpack_stream)
            completion_events.append(unpack_end)
            remote_group.append((unpack_start, unpack_end))
            timing_stages["unpack"] = [(unpack_start, unpack_end)]
        elif remote_event is not None:
            completion_events.append(remote_event)

        if remote_group:
            timing_groups.append(remote_group)

        keepalive = [op.tensor for op in self.send_ops]
        keepalive.extend(op.tensor for op in self.recv_ops)
        keepalive.extend(original_remote_tensors)
        keepalive.extend(op.tensor for op in remote_sends)
        keepalive.extend(op.tensor for op in remote_recvs)
        self.send_ops.clear()
        self.recv_ops.clear()

        handle = NCCLCopyHandle(
            reqs,
            completion_events,
            timing_groups,
            timing_stages,
            keepalive,
            on_complete=self._mark_batch_complete,
        )
        self._inflight = handle
        self._notify_launched()
        if not reqs and not completion_events:
            handle.wait()
        return handle

    def _mark_batch_complete(self) -> None:
        self._inflight = None
        if self.rank == 0:
            logger.info("Batched communication completed")

    def run(self):
        self.launch().wait()
