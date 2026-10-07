# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from megatron.core.resharding import async_execution
from megatron.core.resharding.async_execution import launch_reshard_plan
from megatron.core.resharding.copy_services.base import CompletedCopyHandle
from megatron.core.resharding.copy_services.nccl_copy_service import (
    NCCLCopyHandle,
    NCCLCopyService,
    RecvOp,
    SendOp,
    _block_current_stream_on_requests,
    _least_stream_priority,
)
from megatron.core.resharding.utils import ReshardPlan, TransferOp


def test_temporary_send_address_reuse_does_not_reuse_payload(monkeypatch):
    service = _packing_service(target_bytes=128, max_item_bytes=64)
    service.persistent_pack_buffers = True
    service.send_ops, service.recv_ops = [], []
    model = torch.nn.Module()
    for name in ("a", "b"):
        model.register_parameter(name, torch.nn.Parameter(torch.arange(16.).reshape(2, 8)))
    buffers = [torch.empty(2, 4), torch.empty(2, 4)]
    submitted = []
    calls = 0

    def pooled_contiguous(tensor):
        nonlocal calls
        buffer = buffers[calls % 2]
        calls += 1
        buffer.copy_(tensor)
        return buffer

    def launch():
        packed, _, _, stats = service._pack_remote_ops(service.send_ops, [])
        submitted.append((packed[0].tensor.clone(), stats))
        service.send_ops.clear()
        return CompletedCopyHandle()

    monkeypatch.setattr(torch.Tensor, "contiguous", pooled_contiguous)
    monkeypatch.setattr(service, "launch", launch)
    for start in (0, 4):
        section = (slice(None), slice(start, start + 4))
        plan = ReshardPlan(
            send_ops=[TransferOp(name, 1, True, section, section, i)
                      for i, name in enumerate(("a", "b"))], recv_ops=[])
        launch_reshard_plan(plan, model, None, service).wait().commit()
    expected = torch.cat([p[:, 4:].reshape(-1) for p in model.parameters()]).view(torch.uint8)
    assert torch.equal(submitted[1][0], expected)
    assert submitted[1][1]["send_pack_cache_hits"] == 0


@pytest.mark.parametrize("kind", ["stable", "strided", "dequantized", "mxfp8", "transformed"])
def test_launcher_classifies_send_storage_before_configuring_service(monkeypatch, kind):
    model = torch.nn.Linear(4, 4, bias=False)
    service = Mock()
    service.last_launch_stats = {}
    full = (slice(None), slice(None))
    index = (slice(None), slice(0, 2)) if kind == "strided" else full
    plan = ReshardPlan([TransferOp("weight", 1, True, index, index, 7)], [])
    if kind in {"dequantized", "mxfp8"}:
        wire = model.weight.detach().clone()
        monkeypatch.setattr(async_execution, "_ensure_sendable", lambda p: wire)
    if kind == "mxfp8":
        monkeypatch.setattr(async_execution, "_is_mxfp8_tensor", lambda _: True)
        monkeypatch.setattr(model.weight, "untyped_storage", Mock(side_effect=AssertionError("wrapper has no storage")))
    transform = None
    if kind == "transformed":
        transform = Mock()
        transform.should_transform.return_value = True
        transform.prepare_send.return_value = [model.weight.detach().clone()]
    launch_reshard_plan(plan, model, None, service, transform=transform)
    service.configure_plan.assert_called_once_with(plan, uncacheable_send_task_ids=set() if kind == "stable" else {7})
    assert [call[0] for call in service.method_calls] == ["submit_send", "configure_plan", "launch"]


def test_temporary_eligibility_does_not_change_peer_bucket_geometry():
    full = (slice(None),)
    sends = [TransferOp(name, 1, True, full, full, tid)
             for tid, name in enumerate(("weight.a", "weight.b", "kv::k", "kv::v"))]
    recvs = [TransferOp(op.param_name, 0, False, full, full, op.task_id) for op in sends]
    sender = _packing_service(target_bytes=32, max_item_bytes=8)
    receiver = _packing_service(target_bytes=32, max_item_bytes=8)
    sender.persistent_pack_buffers = receiver.persistent_pack_buffers = True
    sender.configure_plan(ReshardPlan(sends, []), uncacheable_send_task_ids={1})
    receiver.configure_plan(ReshardPlan([], recvs), uncacheable_send_task_ids=set())
    send_ops = [SendOp(op.task_id, torch.zeros(4, dtype=torch.uint8), 1) for op in sends]
    recv_ops = [RecvOp(op.task_id, torch.zeros(4, dtype=torch.uint8), 0) for op in recvs]
    send_groups = sender._partition_pack_groups(send_ops, "dest_rank")
    recv_groups = receiver._partition_pack_groups(recv_ops, "src_rank")
    assert [[op.task_id for op in group] for group in send_groups] == [[0, 1], [2, 3]]
    assert [[op.task_id for op in group] for group in recv_groups] == [[0, 1], [2, 3]]
    for _ in range(2):
        packed, _, _, stats = sender._pack_remote_ops(send_ops, [])
        assert len(packed) == 2 and stats["send_pack_cache_hits"] == 0
        assert not sender._send_pack_cache


def _packing_service(*, target_bytes: int, max_item_bytes: int) -> NCCLCopyService:
    service = object.__new__(NCCLCopyService)
    service.pack_target_bytes = target_bytes
    service.pack_max_item_bytes = max_item_bytes
    service.persistent_pack_buffers = False
    service.pack_rerouted_only = False
    service._pack_task_ids = None
    service._cacheable_pack_task_ids = frozenset()
    service._uncacheable_send_task_ids = frozenset()
    service._send_pack_cache = {}
    service._recv_pack_cache = {}
    service._inflight = None
    service.unpack_chunk_bytes = 0
    return service


class _ReadyEvent:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def test_least_stream_priority_falls_back_for_older_torch(monkeypatch):
    monkeypatch.delattr(torch.cuda, "get_stream_priority_range", raising=False)

    assert _least_stream_priority() == 0


def test_block_current_stream_prefers_nonblocking_work_dependency():
    class Work:
        def __init__(self):
            self.block_calls = 0
            self.wait_calls = 0

        def block_current_stream(self):
            self.block_calls += 1

        def wait(self):
            self.wait_calls += 1

    requests = [Work(), Work()]

    _block_current_stream_on_requests(requests)

    assert [request.block_calls for request in requests] == [1, 1]
    assert [request.wait_calls for request in requests] == [0, 0]


def test_copy_handle_done_is_a_pure_query_until_wait():
    event = _ReadyEvent()
    completions = []
    keepalive = [torch.tensor(1)]
    handle = NCCLCopyHandle(
        requests=[],
        events=[event],
        timing_groups=[],
        timing_stages={},
        keepalive=keepalive,
        on_complete=lambda: completions.append(True),
    )

    assert handle.done()
    assert handle.done()
    assert not completions
    assert event.synchronize_calls == 0
    assert handle.elapsed_seconds() is None

    handle.wait()

    assert completions == [True]
    assert event.synchronize_calls == 1
    assert handle.elapsed_seconds() == 0.0


def test_bounded_unpack_splits_foreach_submission_by_bytes(monkeypatch):
    service = _packing_service(target_bytes=16, max_item_bytes=8)
    service.unpack_chunk_bytes = 6
    calls = []
    monkeypatch.setattr(
        service,
        "_foreach_byte_copy",
        lambda destinations, sources: calls.append(sum(source.numel() for source in sources)),
    )
    sources = [torch.zeros(4, dtype=torch.uint8) for _ in range(3)]
    destinations = [torch.empty_like(source) for source in sources]

    service._foreach_byte_copy_bounded(destinations, sources)

    assert calls == [4, 4, 4]


def test_remote_packing_coalesces_by_peer_and_preserves_mixed_dtypes():
    service = _packing_service(target_bytes=16, max_item_bytes=8)
    sources = [
        torch.tensor([1.5, -2.0], dtype=torch.float16),
        torch.tensor([123456], dtype=torch.int32),
        torch.arange(12, dtype=torch.uint8),
    ]
    destinations = [torch.zeros_like(tensor) for tensor in sources]
    sends = [
        SendOp(task_id=task_id, tensor=tensor, dest_rank=1)
        for task_id, tensor in zip((7, 3, 11), sources)
    ]
    recvs = [
        RecvOp(task_id=task_id, tensor=tensor, src_rank=1)
        for task_id, tensor in zip((7, 3, 11), destinations)
    ]

    packed_sends, packed_recvs, writebacks, stats = service._pack_remote_ops(sends, recvs)

    assert len(packed_sends) == 2
    assert len(packed_recvs) == 2
    assert stats["packed_send_messages"] == 1
    assert stats["packed_send_items"] == 2
    assert stats["original_remote_sends"] == 3
    assert stats["coalesced_remote_sends"] == 2

    sends_by_id = {op.task_id: op.tensor for op in packed_sends}
    for recv in packed_recvs:
        recv.tensor.copy_(sends_by_id[recv.task_id])
    for destination, packed, offset, num_bytes in writebacks:
        service._byte_view(destination).copy_(packed[offset : offset + num_bytes])

    for source, destination in zip(sources, destinations):
        assert torch.equal(source, destination)


def test_remote_packing_leaves_large_and_unaddressed_tensors_direct():
    service = _packing_service(target_bytes=16, max_item_bytes=8)
    operations = [
        SendOp(task_id=1, tensor=torch.zeros(9, dtype=torch.uint8), dest_rank=2),
        SendOp(task_id=None, tensor=torch.zeros(4, dtype=torch.uint8), dest_rank=2),
    ]

    groups = service._partition_pack_groups(operations, "dest_rank")

    assert len(groups) == 2
    assert groups[0][0] is operations[0]
    assert groups[1][0] is operations[1]


def test_remote_packing_only_coalesces_rerouted_tasks():
    service = _packing_service(target_bytes=16, max_item_bytes=8)
    service._pack_task_ids = frozenset({1, 3})
    operations = [
        SendOp(task_id=1, tensor=torch.zeros(4, dtype=torch.uint8), dest_rank=2),
        SendOp(task_id=2, tensor=torch.zeros(4, dtype=torch.uint8), dest_rank=2),
        SendOp(task_id=3, tensor=torch.zeros(4, dtype=torch.uint8), dest_rank=2),
    ]

    groups = service._partition_pack_groups(operations, "dest_rank")

    assert len(groups) == 2
    assert [[op.task_id for op in group] for group in groups] == [[1, 3], [2]]


def test_persistent_send_bucket_is_reused_after_warmup():
    service = _packing_service(target_bytes=16, max_item_bytes=8)
    service.persistent_pack_buffers = True
    full = (slice(None),)
    service.configure_plan(ReshardPlan([
        TransferOp(name, 2, True, full, full, tid) for tid, name in ((1, "a"), (2, "b"))
    ], []), uncacheable_send_task_ids=set())
    sends = [
        SendOp(task_id=1, tensor=torch.tensor([1, 2], dtype=torch.uint8), dest_rank=2),
        SendOp(task_id=2, tensor=torch.tensor([3, 4], dtype=torch.uint8), dest_rank=2),
    ]

    first, _, _, first_stats = service._pack_remote_ops(sends, [])
    second, _, _, second_stats = service._pack_remote_ops(sends, [])

    assert first[0].tensor.data_ptr() == second[0].tensor.data_ptr()
    assert first_stats["send_pack_cache_misses"] == 1
    assert second_stats["send_pack_cache_hits"] == 1
    assert second_stats["reused_send_bytes"] == 4

    service.invalidate_persistent_pack_cache()
    assert not service._send_pack_cache


def test_persistent_send_bucket_never_caches_mutable_kv_tasks():
    service = _packing_service(target_bytes=16, max_item_bytes=8)
    service.persistent_pack_buffers = True
    service.pack_rerouted_only = True
    full = (slice(None),)
    plan = ReshardPlan(
        send_ops=[
            TransferOp("kv::key", 2, True, full, full, 1),
            TransferOp("kv::value", 2, True, full, full, 2),
        ],
        recv_ops=[],
        rerouted_task_ids=frozenset({1, 2}),
    )
    service.configure_plan(plan, uncacheable_send_task_ids=set())
    sends = [
        SendOp(task_id=1, tensor=torch.tensor([1, 2], dtype=torch.uint8), dest_rank=2),
        SendOp(task_id=2, tensor=torch.tensor([3, 4], dtype=torch.uint8), dest_rank=2),
    ]

    first, _, _, first_stats = service._pack_remote_ops(sends, [])
    sends[0].tensor.fill_(9)
    second, _, _, second_stats = service._pack_remote_ops(sends, [])

    assert first[0].tensor.data_ptr() != second[0].tensor.data_ptr()
    assert second_stats["send_pack_cache_hits"] == 0
    assert second_stats["mutable_send_bytes"] == 4
