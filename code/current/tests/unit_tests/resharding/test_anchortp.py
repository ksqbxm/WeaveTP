"""Scheduling and full-plan allocation tests; CUDA execution is tested separately."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.resharding import anchortp
from megatron.core.resharding.async_execution import launch_reshard_plan, prepare_reshard_receives
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.live import TransferTask
from megatron.core.resharding.utils import ReshardPlan, TransferOp


def task(i, src=0, dst=1, size=4):
    return TransferTask(i, src, dst, size, "w")


def test_static_order_ties_caps_and_all_tasks():
    tasks = [task(7, 2), task(3), task(1), task(9, size=30)]
    rates = {(0, 1): 10, (2, 1): 100}
    assert anchortp.schedule(tasks, rates, max_tasks=2, max_bytes=8) == [[7, 1], [3], [9]]
    assert [t.task_id for t in tasks] == [7, 3, 1, 9]
    with pytest.raises(ValueError, match="unique"):
        anchortp.schedule(tasks + [tasks[0]], rates)
    with pytest.raises(ValueError, match="positive"):
        anchortp.schedule(tasks, {(0, 1): float("nan"), (2, 1): 100})


def test_hardware_paths_do_not_infer_xgmi_from_bandwidth():
    hosts = ["node-a", "node-a", "node-b"]
    assert anchortp.path_kind(task(0, 0, 0), hosts) == "local_copy"
    assert anchortp.path_kind(task(0), hosts) == "gpu_pcie"
    assert anchortp.path_kind(task(0, 0, 2), hosts) == "network"
    links = [["local", "xgmi", "network"], ["xgmi", "local", "network"], ["network"] * 3]
    assert anchortp.path_kind(task(0), hosts, links) == "gpu_xgmi"
    assert anchortp.path_kind(task(0, 0, 2), hosts, links) == "network"


def test_explicit_source_search_conflicts(monkeypatch):
    from megatron.core.resharding import planner

    monkeypatch.setattr(planner, "_source_search_config", lambda: {"algorithm": "greedy"})
    with pytest.raises(ValueError, match="source-search"):
        anchortp.configure(SimpleNamespace())


def test_transport_preserves_scheduler_order(monkeypatch):
    monkeypatch.setattr(torch.distributed, "P2POp",
                        lambda op, tensor, peer, group: SimpleNamespace(tensor=tensor))
    service = object.__new__(NCCLCopyService)
    service.group = None
    service.set_task_order([9, 2, 5])
    send = [SimpleNamespace(task_id=i, tensor=torch.zeros(1), dest_rank=1) for i in [2, 9]]
    recv = [SimpleNamespace(task_id=5, tensor=torch.zeros(1), src_rank=1)]
    ops = service._build_task_round_p2p_ops(send, recv)
    assert [id(op.tensor) for op in ops] == [id(send[1].tensor), id(send[0].tensor), id(recv[0].tensor)]
    assert service._task_sort_key(-1, 0) < service._task_sort_key(9, 0)


class Bundle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.full((3, 4), float("nan")), requires_grad=False)


def plans():
    slices = [(slice(None), slice(i, i + 1)) for i in range(4)]
    ops = [TransferOp("w", 0, False, s, s, i) for i, s in enumerate(slices)]
    return ReshardPlan([], ops)


class Transport:
    producer_stream = None

    def __init__(self, source, events):
        self.source, self.events, self.recvs = source, events, []

    def set_task_order(self, ids):
        self.ids = ids

    def submit_recv(self, tensor, peer, task_id=None):
        self.recvs.append((tensor, peer, task_id))

    def launch(self):
        self.events.append(("launch", tuple(self.ids)))
        for buffer, _peer, tid in self.recvs:
            buffer.copy_(self.source.w[:, tid:tid + 1])
        self.recvs.clear()
        return SimpleNamespace(wait=lambda: None, elapsed_seconds=lambda: 0.01)


def environment(monkeypatch, tasks, events, remote_failure=False):
    def gather(out, value, **_):
        events.append(("ready", value))
        out[:] = [value, "rank 1: OOM" if remote_failure else None]

    monkeypatch.setattr(anchortp, "dist", SimpleNamespace(
        get_rank=lambda: 0, get_world_size=lambda **_: 2,
        broadcast_object_list=lambda *_a, **_kw: None, all_gather_object=gather,
        barrier=lambda **_: events.append(("barrier",)), all_reduce=lambda *_a, **_kw: None,
        ReduceOp=SimpleNamespace(MAX=0, SUM=1),
    ))
    monkeypatch.setattr(anchortp, "collect_transfer_tasks", lambda *_a, **_kw: tasks)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("gpu_prepared",)))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))


@pytest.mark.parametrize("idle", [False, True])
def test_all_receives_precede_first_transfer_and_survive_every_batch(monkeypatch, idle):
    events = []
    source, target = Bundle(), Bundle()
    source.w.data.copy_(torch.arange(12.).reshape(3, 4))
    plan = ReshardPlan([], []) if idle else plans()
    tasks = [task(i, dst=0, size=12) for i in range(4)]
    environment(monkeypatch, tasks, events)
    real_empty = torch.empty

    def allocate(*a, **kw):
        assert not any(e[0] == "launch" for e in events)
        events.append(("allocate",))
        return real_empty(*a, **kw)

    monkeypatch.setattr(torch, "empty", allocate)
    result = anchortp.execute(
        plan, source, None if idle else target, Transport(source, events), bandwidth={(0, 0): 100},
        hosts=["a"], max_tasks=1, prealloc_started=anchortp.time.perf_counter() - 0.02,
    )
    first = next(i for i, e in enumerate(events) if e[0] == "launch")
    assert sum(e[0] == "allocate" for e in events[:first]) == (0 if idle else 4)
    assert ("gpu_prepared",) in events[:first] and ("barrier",) in events[:first]
    assert [e[1] for e in events if e[0] == "launch"] == [(0,), (1,), (2,), (3,)]
    assert 0.02 <= result["prealloc_s"] <= result["wall_s"]
    assert result["reload_bytes"] == result["overlap_steps"] == 0
    if not idle:
        torch.testing.assert_close(target.w, source.w)


def test_remote_preallocation_failure_aborts_before_transport(monkeypatch):
    events = []
    environment(monkeypatch, [task(0, dst=0)], events, remote_failure=True)
    with pytest.raises(RuntimeError, match="rank 1: OOM"):
        anchortp.execute(plans(), Bundle(), Bundle(), Transport(Bundle(), events),
                         bandwidth={(0, 0): 10}, hosts=["a"])
    assert not any(e[0] == "launch" for e in events)


def test_contiguous_receive_aliases_target_and_missing_preallocation_submits_nothing():
    target = Bundle()
    s = (slice(0, 1), slice(None))
    plan = ReshardPlan([], [TransferOp("w", 1, False, s, s, 0)])
    prepared = prepare_reshard_receives(plan, target)
    assert prepared[0][0].data_ptr() == target.w.data_ptr()
    assert prepared[0][1] == ("direct",)
    service = Mock()
    with pytest.raises(ValueError, match="missing preallocated"):
        launch_reshard_plan(plan, Bundle(), target, service, prepared_receives={})
    assert not service.mock_calls


def _document_module(relative):
    root = Path(__file__).resolve().parents[5]
    spec = importlib.util.spec_from_file_location("anchor_summary_test", root / "documents/e2e_prep_20261008" / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_does_not_relabel_legacy_and_rejects_missing_reload(tmp_path):
    summary = _document_module("tools/cycle/summarize_cycle.py")
    folder = tmp_path / "run"
    folder.mkdir()
    payload = {"method_variant": "anchortp-proxy", "switches": []}
    (folder / "result.json").write_text(json.dumps(payload))
    result = summary.one(str(tmp_path))
    assert result["implementation"] == "legacy-weavetp-proxy"
    assert "旧代理（含 WeaveTP residual 与改道）" in result["method_label"]
    payload.update(implementation=anchortp.IMPLEMENTATION, switches=[{"overlap_steps": 0, "delta_tokens": 0}])
    (folder / "result.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing/nonzero"):
        summary.one(str(tmp_path))


def test_calibration_rejects_mixed_anchor_versions(tmp_path, monkeypatch):
    calibrate = _document_module("sim/calibrate.py")
    path = tmp_path / "cycle.json"
    path.write_text(json.dumps([
        {"method": "anchortp-proxy", "switches": []},
        {"method": "anchortp-proxy", "implementation": anchortp.IMPLEMENTATION, "switches": []},
    ]))
    monkeypatch.setattr("sys.argv", ["calibrate.py", "--cycle", str(path)])
    with pytest.raises(ValueError, match="AnchorTP"):
        calibrate.main()
