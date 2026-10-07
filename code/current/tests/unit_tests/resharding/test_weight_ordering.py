"""Event-interval semantics and real two-process Gloo failure coordination."""

import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.resharding.correctness import gpu_weight_ordering as probe
from tools.resharding.correctness.gpu_weight_ordering import completions_during_transfer, make_plan

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("nan", [False, True])
def test_failure_diagnostic_distinguishes_stale_half_and_nan(nan):
    expected = torch.arange(8.)
    target = expected.repeat(2, 1)
    target[:, 4:] = float("nan") if nan else expected[:4]
    diagnostic = probe.mismatch_diagnostic(target, expected)
    assert diagnostic["mismatch_count"] == 8
    assert diagnostic["first_mismatch_index"] == [0, 4]
    assert diagnostic["has_nan"] == nan
    assert diagnostic["mismatches_equal_other_half"] == (not nan)
    json.dumps(diagnostic, allow_nan=False)


@pytest.mark.parametrize("transfer,foreground,expected", [
    ([(10, 20)], [(11, 12)], 1),
    ([(10, 20)], [(5, 12)], 1),
    ([(10, 20)], [(5, 10), (20, 21)], 0),
    ([(10, 20)], [(11, 20), (11, 21)], 0),
    ([(10, 20)], [(12, 12)], 0),
    ([(10, 20), (15, 30)], [(16, 17)], 1),
    ([(10, 20), (30, 40)], [(11, 12), (21, 22), (31, 32)], 2),
    ([], [(11, 12)], 0),
])
def test_overlap_requires_completion_while_transport_is_pending(transfer, foreground, expected):
    assert completions_during_transfer(transfer, foreground) == expected


def test_chunk_plans_match_peers_and_cover_target_once():
    for start, end in ((0, 8), (8, 16)):
        plans = [make_plan(rank, start, end) for rank in range(2)]
        for rank, plan in enumerate(plans):
            for send in plan.send_ops:
                recv = next(op for op in plans[send.peer_rank].recv_ops if op.task_id == send.task_id)
                assert recv.peer_rank == rank and recv.param_name == send.param_name
                assert recv.my_slice == send.peer_slice == (slice(None), slice(start, end))


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("corrupt", [False, True])
def test_ordering_probe_executes_chunks_and_checks_data_with_cpu_transport(tmp_path, monkeypatch, packed, rank, corrupt):
    """Execute the probe itself; only device/transport are CPU test doubles."""
    tick = [0]
    launches = [0]
    producer = SimpleNamespace(wait_stream=lambda *_: None, wait_event=lambda *_: None, synchronize=lambda: None)

    class Event:
        def __init__(self, **_):
            self.time = 0
        def record(self, *_):
            tick[0] += 1
            self.time = tick[0]
        def elapsed_time(self, other):
            return other.time - self.time

    class Service:
        def __init__(self, **kwargs):
            assert bool(kwargs["pack_target_bytes"]) == packed
            self.sends, self.recvs = [], []
        def submit_send(self, tensor, peer, task_id=None):
            self.sends.append((tensor, peer, task_id))
        def submit_recv(self, tensor, peer, task_id=None):
            self.recvs.append((tensor, peer, task_id))
        def invalidate_persistent_pack_cache(self):
            pass
        def launch(self):
            launches[0] += 1
            sends = {tid % 3: tensor for tensor, _, tid in self.sends}
            for tensor, peer, tid in self.recvs:
                tensor.copy_(sends[tid % 3] + (peer - rank) * 4)
                if corrupt and launches[0] == 2:
                    tensor[0, 0] = -7
            self.sends.clear()
            self.recvs.clear()
            start, end = Event(), Event()
            start.record()
            end.record()
            stages = {"local": [(start, end)], "nccl": [(start, end)]}
            return SimpleNamespace(_timing_stages=stages, wait=stages.clear)

    for name in ("empty", "arange", "randn"):
        factory = getattr(torch, name)
        def cpu_factory(*args, _factory=factory, **kwargs):
            kwargs.pop("device", None)
            return _factory(*args, **kwargs)
        monkeypatch.setattr(torch, name, cpu_factory)
    empty_like = torch.empty_like
    pools = [torch.empty(16, 8) for _ in range(3)]
    allocated = [0]
    def reuse(template, **kwargs):
        if template.shape != (16, 8):
            return empty_like(template, **kwargs)
        tensor = pools[allocated[0] % 3]
        allocated[0] += 1
        return tensor
    monkeypatch.setattr(torch, "empty_like", reuse)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda *_: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: producer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "_sleep", lambda *_: None)
    monkeypatch.setattr("tools.resharding.standby_weights.cuda_memory", lambda: {"allocated": 0, "reserved": 0})
    monkeypatch.setattr(probe, "NCCLCopyService", Service)
    monkeypatch.setattr(probe.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(probe.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(probe.dist, "barrier", lambda: None)
    monkeypatch.setattr(probe.dist, "all_gather_object", lambda out, value: out.__setitem__(slice(None), [value, value]))
    records = []
    with torch.inference_mode():
        if corrupt:
            with pytest.raises(AssertionError, match="coordinated ordering acceptance failure"):
                probe.run_probe(None, producer, packed, "ordering", records, tmp_path)
        else:
            probe.run_probe(None, producer, packed, "ordering", records, tmp_path)
    assert len(records) == (1 if corrupt else 3)
    assert all(row["receive_storage_reused"] for row in records)
    assert records[-1]["checks"]["exact_remote"] == (not corrupt)
    if not corrupt:
        assert records[-1]["checks"]["receive_storage_reuse_exercised"]


@pytest.mark.parametrize("failed_rank", [-1, 0, 1])
@pytest.mark.parametrize("check_kind", ["ordering", "bitwise"])
def test_all_ranks_agree_before_next_batch_and_save_failures(tmp_path, failed_rank, check_kind):
    # These workers use the actual completion function and actual Gloo. A failed
    # rank must prevent BOTH workers from submitting their simulated next batch.
    script = '''
import ast, json, sys
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from tools.resharding.correctness.gpu_weight_ordering import complete_case
from tools.resharding.standby_weights import StandbyWeights, audit_weight_checksums
rank, failed = map(int, sys.argv[1:3])
output = Path(sys.argv[3])
dist.init_process_group("gloo", init_method=sys.argv[4], rank=rank, world_size=2,
                        timeout=timedelta(seconds=15))
try:
    # Test the production MIN reduction too: both ranks must decode while
    # either one is pending, then agree on readiness and failed validity.
    tree = ast.parse(Path("examples/rl/benchmark_live_moe_tp.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_all_ready")
    ns = dict(torch=torch, dist=dist)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "readiness", "exec"), ns)
    votes = [ns["_all_ready"](value, None) for value in (rank == 0, rank == 1, True, rank != failed)]
    (output / f"votes{rank}.json").write_text(json.dumps(votes))
    if sys.argv[5] == "bitwise":
        model = torch.nn.Linear(2, 2, bias=False)
        weights = StandbyWeights(model, protected_tensors=())
        reference = weights.bitwise_checksums()
        if rank == failed:
            model.weight.data.fill_(float("nan"))
        try:
            audit_weight_checksums(weights, reference, dist.group.WORLD)
        except AssertionError as error:
            (output / f"audit{rank}.txt").write_text(str(error))
            raise
    complete_case({"checks": {"exact": rank != failed, "finite": True}}, [], output)
    (output / f"next_batch{rank}").write_text("submitted")
finally:
    dist.destroy_process_group()
'''
    rendezvous = (tmp_path / "rendezvous").as_uri()
    processes, logs = [], []
    deadline = time.monotonic() + 30
    try:
        for rank in range(2):
            log = (tmp_path / f"worker{rank}.log").open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, "-B", "-c", script, str(rank), str(failed_rank), str(tmp_path), rendezvous, check_kind],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1"},
            ))
        codes = [p.wait(timeout=max(0.01, deadline - time.monotonic())) for p in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    diagnostics = "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.glob("worker*.log"))
    assert all((code == 0) == (failed_rank == -1) for code in codes), diagnostics
    for rank in range(2):
        assert (tmp_path / f"next_batch{rank}").exists() == (failed_rank == -1), diagnostics
        if check_kind == "bitwise" and failed_rank != -1:
            error = (tmp_path / f"audit{rank}.txt").read_text()
            assert f"{failed_rank}:" in error and "weight" in error and "expected" in error and "actual" in error
            continue
        receipt = json.loads((tmp_path / f"ordering_rank{rank}.json").read_text())
        assert receipt["passed"] == (failed_rank == -1)
        assert receipt["cases"][0]["failures"] == ([] if failed_rank == -1 else [{"rank": failed_rank, "failed_checks": ["exact"]}])
        assert json.loads((tmp_path / f"votes{rank}.json").read_text()) == [False, False, True, failed_rank == -1]
