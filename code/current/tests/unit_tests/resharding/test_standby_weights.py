"""Real storage/planner/transaction tests; CUDA ordering doubles are labelled as such."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from tools.resharding.correctness.harness import (
    Bundle,
    finish_all,
    fixture_tensors,
    launch_all,
    param,
    plans_for,
)
from tools.resharding.correctness.reference import Layout, StateError, exact_chunks, markers
from tools.resharding.standby_weights import (
    StandbyWeights,
    _chunks,
    release_standby,
    storage_bytes,
    validate_release_mode,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_shared_storage_and_offset_survive_repeated_release(dtype):
    model = torch.nn.Module()
    backing = torch.arange(48, dtype=dtype).reshape(6, 8)
    model.a = param(backing[1:5, 1:7:2])
    model.b = param(backing[1:5, 1:7:2].T)
    model.tied = model.a
    model.register_buffer("expert_backing", backing)
    model.a.partition_dim, model.a.partition_stride = 1, 2
    kv = torch.arange(30, dtype=dtype)
    before = [(id(p), p.shape, p.stride(), p.storage_offset()) for p in model.parameters()]
    weights = StandbyWeights(model, protected_tensors=[kv])
    assert weights.num_bytes == backing.numel() * backing.element_size()
    assert storage_bytes([model.a, model.b, backing]) == weights.num_bytes
    for _ in range(6):
        weights.release()
        assert model.a.untyped_storage().nbytes() == model.b.untyped_storage().nbytes() == 0
        assert before == [(id(p), p.shape, p.stride(), p.storage_offset()) for p in model.parameters()]
        weights.allocate_and_poison()
        assert torch.isnan(model.a).all() and torch.isnan(model.b).all()
        assert not weights.finite_flag()
        with torch.no_grad():
            model.a.fill_(7)
        assert torch.equal(model.b, model.a.T)
        assert weights.finite_flag()
        assert model.a.partition_dim == 1 and model.a.partition_stride == 2
        torch.testing.assert_close(kv, torch.arange(30, dtype=dtype), rtol=0, atol=0)


@pytest.mark.parametrize("alias", ["active", "kv"])
def test_protected_alias_is_rejected_before_storage_mutation(alias):
    tensor = torch.randn(8, 8)
    model = Bundle({"weight::w": param(tensor[:, ::2])})
    with pytest.raises(ValueError, match="alias"):
        StandbyWeights(model, protected_tensors=[tensor])
    assert tensor.untyped_storage().nbytes() == 256


def test_invalid_transitions_and_changed_storage_fail_without_repair():
    model = Bundle({"w": param(torch.ones(4))})
    weights = StandbyWeights(model, protected_tensors=[])
    with pytest.raises(RuntimeError, match="transition"):
        weights.allocate_and_poison()
    weights.release()
    with pytest.raises(RuntimeError, match="transition"):
        weights.release()
    with pytest.raises(RuntimeError, match="transition"):
        weights.finite_flag()
    weights.allocate_and_poison()
    model.entries["w"].data = torch.zeros(4)
    with pytest.raises(RuntimeError, match="changed"):
        weights.release()
    assert torch.equal(model.entries["w"], torch.zeros(4))


@pytest.mark.parametrize("shape", [(), (0,), (33,), (3, 29), (2, 3, 17)])
def test_finite_chunks_are_bounded_views(shape):
    tensor = torch.zeros(shape)
    parts = list(_chunks(tensor, limit=7))
    assert sum(p.numel() for p in parts) == tensor.numel()
    assert all(p.numel() <= 7 and p.untyped_storage() is tensor.untyped_storage() for p in parts)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_four_real_reshards_after_nan_reallocation(dtype, noncontiguous, stride):
    full = markers((12, 48), dtype=dtype)
    layouts = [Layout((0, 1), 1, stride), Layout((0, 1, 2, 3), 1)]
    name = "weight::experts.w"
    source, target, _ = fixture_tensors(full, *layouts, name=name, noncontiguous=noncontiguous)
    banks = [source, target]
    managers = [{r: StandbyWeights(m, protected_tensors=[]) for r, m in bank.items()} for bank in banks]
    for manager in managers[1].values():
        manager.release()
    for turn in range(4):
        src, dst = turn % 2, 1 - turn % 2
        for manager in managers[dst].values():
            manager.allocate_and_poison()
        plans = plans_for(full.shape, layouts[src], layouts[dst], name=name, dtype=dtype)
        source_before = {r: next(m.parameters()).detach().clone() for r, m in banks[src].items()}
        transactions, mailbox = launch_all(plans, banks[src], banks[dst])
        assert all(not t.done() for t in transactions.values())
        # Real foreground reads while the transport double is pending.
        for r, module in banks[src].items():
            exact_chunks(next(module.parameters()), source_before[r])
        finish_all(transactions)
        assert mailbox.finished
        for r, module in banks[dst].items():
            assert managers[dst][r].finite_flag()
            exact_chunks(next(module.parameters()), layouts[dst].shard(full, r))
        for manager in managers[src].values():
            manager.release()


@pytest.mark.parametrize("fault", ["entire_weight", "slice", "cold_expert"])
def test_missing_transfer_cannot_pass_on_stale_allocator_contents(fault):
    name = "weight::experts.cold" if fault == "cold_expert" else "weight::w"
    full = markers((4, 8))
    src_layout, dst_layout = Layout((0, 1), 1), Layout((0, 1, 2, 3), 1)
    source, target, expected = fixture_tensors(full, src_layout, dst_layout, name=name)
    managers = {}
    for r, model in target.items():
        with torch.no_grad():
            next(model.parameters()).copy_(expected[r])  # Previously valid, stale target.
        managers[r] = StandbyWeights(model, protected_tensors=[])
        managers[r].release()
        managers[r].allocate_and_poison()
    plans = plans_for(full.shape, src_layout, dst_layout, name=name)
    omitted = {op.task_id for op in plans[0].recv_ops}
    if fault != "slice":
        omitted = {op.task_id for plan in plans.values() for op in plan.recv_ops}
    for plan in plans.values():
        plan.send_ops = [op for op in plan.send_ops if op.task_id not in omitted]
        plan.recv_ops = [op for op in plan.recv_ops if op.task_id not in omitted]
    transactions, _ = launch_all(plans, source, target)
    finish_all(transactions)
    assert not managers[0].finite_flag()
    with pytest.raises(StateError, match="NaN/Inf"):
        exact_chunks(next(target[0].parameters()), expected[0])


def test_finite_but_wrong_values_require_independent_numeric_oracle():
    model = Bundle({"w": param(torch.ones(4))})
    weights = StandbyWeights(model, protected_tensors=[])
    weights.release()
    weights.allocate_and_poison()
    with torch.no_grad():
        next(model.parameters()).fill_(9)
    assert weights.finite_flag()
    with pytest.raises(StateError, match="mismatch"):
        exact_chunks(next(model.parameters()), torch.ones(4))


@pytest.mark.parametrize("expert_count,pressure_count", [(2, 1), (1, 2), (2, 2)])
def test_multiple_phases_rejected_even_if_groups_equal(expert_count, pressure_count):
    args = SimpleNamespace(live_active_expert_phases_tuple=[(0,)] * expert_count,
                           live_pressure_rank_phases_tuple=[()] * pressure_count,
                           live_method_variant="moetp++")
    with pytest.raises(ValueError, match="single"):
        validate_release_mode(args)


@pytest.mark.parametrize("variant", ["baseline", "moetp++", "moetp++-hybrid", "flying-serving-proxy", "llumnix-proxy"])
def test_mode_validation(variant):
    args = SimpleNamespace(live_active_expert_phases_tuple=[(0, 1)],
                           live_pressure_rank_phases_tuple=[()], live_method_variant=variant)
    if variant in {"flying-serving-proxy", "llumnix-proxy"}:
        with pytest.raises(ValueError, match="full weight"):
            validate_release_mode(args)
    else:
        validate_release_mode(args)


def test_release_accounts_weights_separately_from_packing():
    weights = StandbyWeights(Bundle({"w": param(torch.ones(4))}), protected_tensors=[])
    service = Mock()
    with patch("tools.resharding.standby_weights.cuda_memory", side_effect=[
        {"allocated": 100, "reserved": 200}, {"allocated": 80, "reserved": 200},
        {"allocated": 64, "reserved": 200},
    ]):
        record = release_standby(weights, [service])
    service.invalidate_persistent_pack_cache.assert_called_once_with()
    assert record["packing_allocated_freed_bytes"] == 20
    assert record["weight_allocated_freed_bytes"] == record["weight_storage_bytes"] == 16


@pytest.mark.parametrize("enabled", [False, True])
def test_actual_nccl_local_launch_orders_producer_without_host_sync(enabled):
    events, waits = [], []
    current = [None]

    class Event:
        def __init__(self, **_):
            events.append(self)
        def record(self, stream=None):
            self.stream = stream or current[0]
        def query(self):
            return True
        def elapsed_time(self, _):
            return 0.0
        def synchronize(self):
            raise AssertionError("launch must not host-synchronize")

    class Stream:
        def __init__(self, **_):
            pass
        def wait_event(self, event):
            waits.append((self, event))

    @contextmanager
    def stream_context(stream):
        old, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = old

    with patch("torch.cuda.Stream", Stream), patch("torch.cuda.Event", Event), \
            patch("torch.cuda.stream", stream_context), patch("torch.cuda.current_stream", lambda: current[0]):
        service = NCCLCopyService(group=SimpleNamespace(rank=lambda: 0, size=lambda: 1))
        current[0] = Stream()
        service.producer_stream = current[0] if enabled else None
        source, target = torch.arange(4.), torch.full((4,), float("nan"))
        service.submit_send(source, 0, task_id=0)
        service.submit_recv(target, 0, task_id=0)
        handle = service.launch()
        assert torch.equal(target, source)
        assert handle.done()
        assert waits == ([(service._copy_stream, events[0]), (service._comm_stream, events[0])] if enabled else [])


def test_wrong_submission_stream_rejected_before_any_copy():
    service = object.__new__(NCCLCopyService)
    service._inflight, service.producer_stream = None, object()
    with patch("torch.cuda.current_stream", return_value=object()), pytest.raises(RuntimeError, match="preparation stream"):
        service.launch()


def test_poison_waits_for_allocation_stream_without_host_sync():
    weights = StandbyWeights(Bundle({"w": param(torch.ones(4))}), protected_tensors=[])
    weights.release()
    producer, allocation = Mock(), Mock()
    @contextmanager
    def poison_stream(stream):
        assert stream is producer
        producer.wait_stream.assert_called_once_with(allocation)
        yield
    with patch("torch.cuda.current_stream", return_value=allocation), patch("torch.cuda.stream", poison_stream), \
            patch("torch.cuda.synchronize", side_effect=AssertionError("no device fence")):
        weights.allocate_and_poison(producer)
    assert torch.isnan(next(iter(weights.parameters))).all()
    producer.synchronize.assert_not_called()
