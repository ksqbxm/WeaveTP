"""Forced-plan endpoint checks and tiny EP2 execution through the real launcher."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from test_source_search import replay

from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService, RecvOp, SendOp
from megatron.core.resharding.plan_validation import (
    validate_reshard_plans,
    validate_source_metadata,
)
from megatron.core.resharding.utils import extract_param_metadata
from tools.resharding.correctness.harness import finish_all, launch_all
from tools.resharding.correctness.source_search_gpu import (
    CASES,
    fixture,
    planner_kwargs,
    rank_groups,
)


def cpu_fixture():
    sources, targets, expected = {}, {}, {}
    source_metadata, target_metadata = {}, {}
    with patch("torch.distributed.get_process_group_ranks", side_effect=list), \
            patch("torch.distributed.get_world_size", return_value=8):
        for rank in range(8):
            topology = {}
            for tp in (2, 4):
                groups = {kind: next(g for g in groups if rank in g)
                          for kind, groups in rank_groups(tp).items()}
                topology[tp] = SimpleNamespace(**groups, expt_tp=groups["tp"], pp=None)
            sources[rank], targets[rank], expected[rank] = fixture(rank, "cpu", topology)
            for name, tensor in sources[rank].named_parameters():
                m = extract_param_metadata(tensor, name, rank, topology[2], num_experts=2)
                source_metadata.setdefault(m.resolved_name, []).append(m)
            target_metadata[rank] = {}
            for name, tensor in targets[rank].named_parameters():
                m = extract_param_metadata(tensor, name, rank, topology[4], num_experts=2)
                target_metadata[rank][m.resolved_name] = m
    kwargs = planner_kwargs("dfs")
    kwargs.pop("source_search_config")
    return dict(src_params=source_metadata, dst_params=target_metadata, kwargs=kwargs), sources, targets, expected


@pytest.mark.parametrize("case", CASES)
def test_ep2_forced_plan_payload_and_peer_fifo(case):
    data, sources, targets, expected = cpu_fixture()
    algorithm = "ilp" if case.startswith("ilp_") else case
    plans = dict(enumerate(replay(data, algorithm=algorithm,
                                 gap=1e-4 if case == "ilp_gap1e-4" else 0.)))
    stats = plans[0].source_route_stats
    assert stats["global_gate_accepted"] and stats["accepted"] > 0
    assert stats["source_search"]["status"] == "complete"
    audit = validate_reshard_plans(plans, data["src_params"], data["dst_params"])
    assert audit["remote_tasks"] > 0 and audit["local_tasks"] > 0
    # Exercise the actual peer-size-desc ordering on actual prepared tensor views.
    ordered_sends, ordered_recvs = {}, {}
    with patch("torch.distributed.P2POp", side_effect=lambda op, tensor, peer, **kw: (op, tensor, peer)):
        for rank, plan in plans.items():
            sends = [SendOp(op.task_id, sources[rank].entries[op.param_name][op.my_slice], op.peer_rank)
                     for op in plan.send_ops if op.peer_rank != rank]
            recvs = [RecvOp(op.task_id, targets[rank].entries[op.param_name][op.my_slice], op.peer_rank)
                     for op in plan.recv_ops if op.peer_rank != rank]
            ids = {id(op.tensor): op.task_id for op in sends + recvs}
            service = object.__new__(NCCLCopyService)
            service.group = None
            wire = service._build_peer_size_p2p_ops(sends, recvs, reverse=True)
            for op, tensor, peer in wire:
                assert rank != peer
                sending = op is torch.distributed.isend
                pair = (rank, peer) if sending else (peer, rank)
                rows = ordered_sends if sending else ordered_recvs
                rows.setdefault(pair, []).append((ids[id(tensor)], tensor.shape,
                                                   tensor.numel() * tensor.element_size()))
    assert ordered_sends == ordered_recvs
    before = {r: {n: t.detach().clone() for n, t in src.entries.items()} for r, src in sources.items()}
    transactions, mailbox = launch_all(plans, sources, targets)
    finish_all(transactions)
    assert mailbox.remote > 0
    for rank in plans:
        for name, truth in expected[rank].items():
            torch.testing.assert_close(targets[rank].entries[name], truth, rtol=0, atol=0)
            torch.testing.assert_close(sources[rank].entries[name], before[rank][name], rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "reorder", "task", "name", "slice", "dtype", "owner"])
def test_plan_validation_rejects_bad_endpoints(fault):
    data, _, _, _ = cpu_fixture()
    plans = dict(enumerate(replay(data, algorithm="dfs")))
    rank, op = next((r, op) for r, plan in plans.items() for op in plan.send_ops if op.peer_rank != r)
    if fault == "missing":
        plans[rank].send_ops.remove(op)
    elif fault == "reorder":
        # Two transfers on the same pair must keep FIFO/task correspondence.
        extra = copy.copy(op)
        extra.task_id = 10000
        plans[rank].send_ops.append(extra)
        recv = next(o for o in plans[op.peer_rank].recv_ops if o.task_id == op.task_id)
        extra = copy.copy(recv)
        extra.task_id = 10000
        plans[op.peer_rank].recv_ops.insert(0, extra)
    elif fault == "task":
        op.task_id = 10000
    elif fault == "name":
        op.param_name = "absent"
    elif fault == "slice":
        op.my_slice = (slice(0, 10000),) * len(op.my_slice)
    else:
        m = next(m for entries in data["src_params"].values() for m in entries
                 if m.owner_rank == rank and m.name == op.param_name)
        if fault == "dtype":
            m.dtype = torch.float32
        else:
            m.owner_rank = (rank + 1) % 8
    with pytest.raises(ValueError):
        validate_reshard_plans(plans, data["src_params"], data["dst_params"])


def test_sender_local_name_is_not_representative_name():
    data, _, _, _ = cpu_fixture()
    # Metadata resolves the same logical expert despite different local names.
    for entries in data["src_params"].values():
        for m in entries:
            m.name = f"rank{m.owner_rank}." + m.name
    plans = dict(enumerate(replay(data, algorithm="dfs")))
    validate_reshard_plans(plans, data["src_params"], data["dst_params"])
    assert all(op.param_name.startswith(f"rank{rank}.")
               for rank, plan in plans.items() for op in plan.send_ops)


def test_selected_tp_representative_checks_actual_expert_owner():
    data, _, _, _ = cpu_fixture()
    selected, actual = data["src_params"]["experts.weight0"][:2]
    validate_source_metadata(selected, actual, actual.owner_rank)
    wrong = copy.copy(actual)
    wrong.global_expert_index = 1
    with pytest.raises(ValueError, match="global_expert_index"):
        validate_source_metadata(selected, wrong, wrong.owner_rank)
    wrong = copy.copy(actual)
    wrong.expert_parallel_group_ranks = list(reversed(actual.expert_parallel_group_ranks))
    with pytest.raises(ValueError, match="EP-local"):
        validate_source_metadata(selected, wrong, wrong.owner_rank)
