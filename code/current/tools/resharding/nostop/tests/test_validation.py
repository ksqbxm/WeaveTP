"""CPU bit corruption, TP/head mapping, audit policy and failure artifacts."""

import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.resharding.nostop import benchmark, synthetic_kv, validation
from tools.resharding.nostop.tests.test_synthetic_kv import context, prefix
from tools.resharding.standby_weights import StandbyWeights


def model(tp, rank):
    m = torch.nn.Linear(3, 2, bias=False).bfloat16()
    m.config = NS(tensor_model_parallel_size=tp)
    m.pg_collection = NS(tp=rank % tp)
    return m


def audit_fixture():
    models, contexts, rows = [], [], []
    real = [prefix(), prefix()]
    dist = NS(get_rank=lambda group: group, get_world_size=lambda group: 8)
    for rank in range(8):
        pair = [model(tp, rank) for tp in (2,4)]
        caches = [context(4 // tp) for tp in (2,4)]
        for m, c in zip(pair, caches):
            synthetic_kv.restore(c, real[rank//4], m.pg_collection.tp, rank//4, torch)
        models.append(pair)
        contexts.append(caches)
        rows.append(dict(rank=rank, request_domain=rank//4, kv={
            role: validation.kv_snapshot(m,c,dist,torch)
            for role,m,c in zip(("source","target"),pair,caches)}, kv_reference_mismatches={}, weights={}))
    def gather(out, local, *, group):
        out[:] = copy.deepcopy(rows)
        out[0] = local
    dist.all_gather_object = gather
    d = NS(rank=0,index=-1,args=NS(live_request_domain_id=0), dist=dist, torch=torch,
           b=NS(StandbyWeights=StandbyWeights))
    audit = validation.CutoverAudit(d)
    args = (models[0][0],contexts[0][0],models[0][1],contexts[0][1],"gloo")
    audit.check(*args, stage="initial")
    d.index = 2
    return audit, args, rows


def test_global_heads_match_replicas_domains_and_both_switch_directions():
    audit, args, rows = audit_fixture()
    assert not validation.compare_kv(rows)
    for row in rows:
        row["kv"]["source"],row["kv"]["target"] = row["kv"]["target"],row["kv"]["source"]
    assert not validation.compare_kv(rows)
    reverse = (args[2],args[3],args[0],args[1],args[4])
    record = audit.check(*reverse,stage="cutover")
    assert record["kv_passed"] and record["weights_passed"]


def test_weight_reference_survives_original_storage_release_and_rebuild():
    audit, args, _ = audit_fixture()
    target = args[2]
    expected = target.weight.detach().clone()
    storage = StandbyWeights(target, protected_tensors=())
    storage.release()
    storage.allocate_and_poison()
    with torch.no_grad():
        target.weight.copy_(expected)
    assert audit.check(*args, stage="cutover")["weights_passed"]


def test_prefix_audit_ignores_appended_decode_tail_but_compares_raw_zero_bits():
    audit, args, _ = audit_fixture()
    tensor = args[3].key_value_memory_dict[1][0]
    tensor[2048:].fill_(3)
    assert audit.check(*args, stage="cutover")["kv_passed"]
    tensor[0,0,0,0] = 0
    before = validation.kv_snapshot(args[2],args[3],audit.d.dist,torch)
    tensor[0,0,0,0] = -0.0
    assert tensor[0,0,0,0] == 0  # Numeric equality does not imply bit equality.
    after = validation.kv_snapshot(args[2],args[3],audit.d.dist,torch)
    assert before["1:K:0"]["sha256"] != after["1:K:0"]["sha256"]


@pytest.mark.parametrize("damage", ["head_swap","one_bit","remote_replica","missing_head"])
def test_kv_corruption_is_fatal_and_retained(damage):
    audit, args, rows = audit_fixture()
    if damage == "head_swap":
        tensor = args[1].key_value_memory_dict[1][0]
        tensor[:2048] = tensor[:2048].flip(2)
    elif damage == "one_bit":
        args[3].key_value_memory_dict[1][1].view(torch.int16)[2047,3,0,1] ^= 1
    elif damage == "remote_replica":
        rows[2]["kv"]["source"]["1:K:0"]["sha256"] = "corrupt"
    else:
        del rows[1]["kv"]["target"]["1:K:1"]
    with pytest.raises(AssertionError, match="bit-content"):
        audit.check(*args,stage="cutover")
    assert not audit.records[-1]["kv_passed"]


def test_restore_reference_catches_identically_corrupted_layouts():
    audit, args, rows = audit_fixture()
    # Even equal layouts are insufficient: the canonical initialization is immutable.
    for role in ("source","target"):
        audit.kv_reference[2 if role=="source" else 4] = copy.deepcopy(audit.kv_reference[2 if role=="source" else 4])
        audit.kv_reference[2 if role=="source" else 4]["1:K:0"]["sha256"] = "different initial prefix"
    with pytest.raises(AssertionError,match="prefix reference"):
        audit.check(*args,stage="restore",weights=False)
    record = audit.records[-1]
    assert not record["cross_layout_mismatches"] and record["reference_failures"]
    assert record["weights_passed"] is None


@pytest.mark.parametrize("remote", [False,True])
def test_weight_raw_bit_audit_is_collective_and_fatal(remote):
    audit,args,rows = audit_fixture()
    if remote:
        rows[7]["weights"] = dict(target=dict(mismatches={"weight":"corrupt"}))
    else:
        args[2].weight.data.view(torch.int16)[0,0] ^= 1
    with pytest.raises(AssertionError,match="weights="):
        audit.check(*args,stage="cutover")
    assert audit.records[-1]["kv_passed"] and not audit.records[-1]["weights_passed"]


@pytest.mark.parametrize("kv_ok,weight_ok", [(True,True),(True,False),(False,True)])
def test_observed_logits_only_relaxed_after_both_hard_audits(kv_ok,weight_ok):
    args=NS(live_logit_max_nrmse=.4,live_logit_min_cosine=.93,live_logit_min_top1_agreement=0)
    metrics=dict(nrmse=.411,cosine=.9156,top1_agreement=0,max_abs=18.94,reference_max_abs=40.5)
    audit=dict(kv_passed=kv_ok,weights_passed=weight_ok)
    if kv_ok and weight_ok:
        assert not validation.synthetic_logit_check(metrics,args,audit)
    else:
        with pytest.raises(AssertionError,match="require passed"):
            validation.synthetic_logit_check(metrics,args,audit)


def test_real_kv_original_validator_still_rejects_observed_metrics():
    path=Path(benchmark.CODE)/"examples/rl/benchmark_live_moe_tp.py"
    tree=ast.parse(path.read_text(encoding="utf-8"))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_validate_cutover")
    node.decorator_list=[]
    module=ast.Module(body=ast.parse("from __future__ import annotations").body+[node],type_ignores=[])
    env=dict(torch=torch,_decode=lambda *a,**kw:(torch.ones(1),12),_all_max=lambda x,g:x,_all_ready=lambda x,g:x,
             _comparison_metrics=lambda *a:dict(nrmse=.411,cosine=.9156,top1_agreement=0))
    exec(compile(module,"real_kv_validation","exec"),env)
    args=NS(live_logit_atol=0,live_logit_rtol=0,live_logit_validation_mode="bf16-relative",
            live_logit_max_nrmse=.4,live_logit_min_cosine=.93,live_logit_min_top1_agreement=0)
    with pytest.raises(AssertionError,match="logits mismatch"):
        env["_validate_cutover"](None,None,None,None,args,None,100)


def test_failed_audit_writes_rank_trace_without_cuda_event_serialization(tmp_path):
    audit,args,rows=audit_fixture()
    args[3].key_value_memory_dict[1][0].view(torch.int16)[0,0,0,0] ^= 1
    with pytest.raises(AssertionError):
        audit.check(*args,stage="cutover")
    d=benchmark.Diagnostic.__new__(benchmark.Diagnostic)
    d.args,d.rank,d.audit=NS(live_json_output=str(tmp_path/"result.json")),0,audit
    d.memory=NS(finish=lambda:None,document=lambda **kw:dict(kw))
    d.logit_metrics,d.marks,d.states,d.waves=[],[],[],[]
    d.decodes=[dict(step=1,complete_host_ns=123,events=(object(),object()))]
    d.write_failure(AssertionError("KV mismatch"))
    trace=json.loads((tmp_path/"trace_rank0.json").read_text())
    assert not trace["complete"] and not trace["validation_audits"][-1]["kv_passed"]
    assert trace["decodes"]==[dict(step=1,complete_host_ns=123)]
    assert not json.loads((tmp_path/"memory_rank0.json").read_text())["complete"]
