"""Raw-bit checksum coverage, including inactive expert corruption."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tools.resharding.standby_weights import StandbyWeights, audit_weight_checksums


def test_chunk_positions_are_unique_and_scratch_is_bounded(monkeypatch):
    from tools.resharding import standby_weights

    chunks = standby_weights._chunks
    monkeypatch.setattr(standby_weights, "_chunks", lambda tensor, limit=7: chunks(tensor, 7))
    model = torch.nn.Linear(10, 10, bias=False)
    model.weight.data.copy_(torch.arange(100.).reshape(10, 10))
    weights = StandbyWeights(model, protected_tensors=())
    views = list(standby_weights._chunks(model.weight))
    assert all(view.numel() <= 7 for view in views)
    bits = torch.cat([view.contiguous().view(torch.int32).reshape(-1).to(torch.int64) for view in views])
    reference = torch.stack([bits.sum(), (bits * torch.arange(1, 101)).sum()])
    assert torch.equal(weights.bitwise_checksums()["weight"], reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bitwise_audit_distinguishes_signed_zero(dtype):
    model = torch.nn.Module()
    model.w = torch.nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)
    weights = StandbyWeights(model, protected_tensors=())
    reference = weights.bitwise_checksums()["w"]
    model.w.fill_(-0.0)
    assert model.w.item() == 0.0
    assert not torch.equal(weights.bitwise_checksums()["w"], reference)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("corruption", [None, "cold_expert", "swap_halves", "one_ulp"])
def test_bitwise_checksum_detects_migration_corruption(dtype, strided, corruption):
    model = torch.nn.Module()
    data = torch.arange(1, 33, dtype=dtype).reshape(4, 8)
    backing = torch.empty(4, 16, dtype=dtype)
    target = backing[:, ::2] if strided else torch.empty_like(data)
    target.copy_(data)
    model.register_parameter("cold_expert", torch.nn.Parameter(target, requires_grad=False))
    storage = StandbyWeights(model, protected_tensors=())
    expected = storage.bitwise_checksums()
    storage.release()
    storage.allocate_and_poison()
    if corruption != "cold_expert":
        target.copy_(data)
    if corruption == "swap_halves":
        target.copy_(data.roll(4, dims=1))
    elif corruption == "one_ulp":
        target[0, 0] = torch.nextafter(target[0, 0], torch.tensor(float("inf"), dtype=dtype))
    actual = storage.bitwise_checksums()
    assert torch.equal(actual["cold_expert"], expected["cold_expert"]) == (corruption is None)
    if corruption == "swap_halves":
        assert actual["cold_expert"][0] == expected["cold_expert"][0]
        assert actual["cold_expert"][1] != expected["cold_expert"][1]


@pytest.mark.parametrize("bad_rank", [None, 0, 1])
def test_audit_coordinates_local_and_remote_failures(bad_rank):
    expected = {"expert.w": torch.tensor([1, 2], dtype=torch.int64)}
    actual = {"expert.w": torch.tensor([1, 3] if bad_rank == 0 else [1, 2], dtype=torch.int64)}
    weights = SimpleNamespace(bitwise_checksums=lambda: actual)
    def gather(out, local, **kwargs):
        assert kwargs["group"] == "control"
        out[:] = [local, {"remote.w": {"expected": [3, 4], "actual": [5, 6]}} if bad_rank == 1 else {}]
    with patch("tools.resharding.standby_weights.dist.get_world_size", return_value=2), \
            patch("tools.resharding.standby_weights.dist.all_gather_object", side_effect=gather) as vote:
        if bad_rank is None:
            audit_weight_checksums(weights, expected, "control")
        else:
            with pytest.raises(AssertionError, match=r"expected.*actual"):
                audit_weight_checksums(weights, expected, "control")
        vote.assert_called_once()
