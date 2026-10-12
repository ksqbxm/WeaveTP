"""CPU regression for torch stage evidence, context subtraction and admission."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))
from tools.resharding.nostop import artifacts, common, cuda_memory, formal_plan, memory_budget


def write_memory_case(case, method="W2", mode="cycle", allocated=20, context=.5, reserved=31):
    (case / "run").mkdir(parents=True, exist_ok=True)
    artifacts.write_json(case / "config.json", common.method_env(method, mode, "", case / "run"))
    (case / "exit_code.txt").write_text("0")
    count = 3 if mode == "cycle" else 1
    for rank in range(8):
        stages = ["startup", "prefill", "plan_setup"] + [f"{phase}_{i}" for i in range(count)
                    for phase in ("prepare_switch", "switch", "post_switch")]
        samples = [dict(where=where, valid=True, reserved_bytes=int(reserved*1024**3),
                        nvml_process_bytes=int((reserved+context)*1024**3), context_bytes=int(context*1024**3),
                        gpu_uuid=f"GPU-{rank}") for where in ["after_prefill_validation"] + [f"after_switch_{i}_validation" for i in range(count)]]
        artifacts.write_json(case / f"run/memory_rank{rank}.json", dict(rank=rank, complete=True,
            workload=common.kv_workload(), device_total_bytes=int(31.36*1024**3), context_samples=samples,
            phases=[dict(stage=stage, max_memory_allocated=int(allocated*1024**3),
                         max_memory_reserved=int(reserved*1024**3)) for stage in stages]))
    return case


def write_memory_run(root, allocated=20, context=.5, reserved=31):
    write_memory_case(root / "smoke_L0_W2", allocated=allocated, context=context, reserved=reserved)
    artifacts.write_json(root / "manifest.json", dict(workload=common.kv_workload(),
        cases=[["smoke_L0_W2", "L0", "W2", "cycle"]], checkpoint="checkpoint", profile_sha256="profile"))
    artifacts.write_json(root / "machine.json", dict(memory_used_mib={f"GPU-{i}":56 for i in range(8)},
        compute_processes=[dict(memory_used_mib=56)]))
    return root


def test_reserved_cache_and_nvml_peak_are_not_live_demand(tmp_path):
    root = write_memory_run(tmp_path)
    budget = memory_budget.from_runs(root, "0:2,1:3,4:6,5:7")
    assert budget["allocated_peak_gib"] == 20 and budget["context_gib"] == .5
    assert budget["reserved_peak_gib"] == 31 and budget["min_free_gib"] == 22
    assert budget["counts"] == dict(L0=0,L1=1,L2=3,L3=7)
    assert budget["estimated_total_gib"]["7"] == pytest.approx(26.064453125)
    assert budget["budgeted_total_gib"]["7"] == pytest.approx(27.564453125)
    # Changing cached reserved (and corresponding NVML) does not change demand.
    write_memory_run(root, reserved=22)
    changed = memory_budget.from_runs(root, "0:2,1:3,4:6,5:7")
    assert changed["min_free_gib"] == budget["min_free_gib"]


def test_budget_uses_all_l0_phases_but_no_l3_allocations(tmp_path):
    write_memory_run(tmp_path)
    path = tmp_path / "smoke_L0_W2/run/memory_rank3.json"
    record = artifacts.read_json(path)
    record["phases"][1]["max_memory_allocated"] = 24 * 1024**3
    artifacts.write_json(path, record)
    write_memory_case(tmp_path / "L3_W2_cycle", allocated=30)
    manifest = artifacts.read_json(tmp_path / "manifest.json")
    manifest["cases"].append(["L3_W2_cycle","L3","W2","cycle"])
    artifacts.write_json(tmp_path / "manifest.json", manifest)
    budget = memory_budget.from_runs(tmp_path,"0:2,1:3,4:6,5:7")
    assert budget["allocated_peak_gib"] == 24 and budget["min_free_gib"] == 26
    assert memory_budget.l3_count(budget,None) == 6
    assert memory_budget.l3_count(budget,4) == 4
    with pytest.raises(ValueError,match="exceeds"):
        memory_budget.l3_count(budget,7)


@pytest.mark.parametrize("peak,count", [(20.5,7),(24.88,5),(25.5,4),(28.52,0),(31,None)])
def test_largest_count_and_proportional_levels(peak,count):
    budget = memory_budget.estimate(peak-.5,.5,"0:2,1:3,4:6,5:7",56)
    assert budget["max_l3_processes"] == count
    assert budget["min_free_gib"] == peak+1.5
    if count in (4,5):
        assert budget["counts"] == dict(L0=0,L1=1,L2=2,L3=count)


@pytest.mark.parametrize("damage", ["old_workload","missing_rank","missing_stage","invalid_context","failed"])
def test_old_or_incomplete_evidence_is_rejected(tmp_path,damage):
    root = write_memory_run(tmp_path)
    p = root / ("manifest.json" if damage=="old_workload" else "smoke_L0_W2/run/memory_rank0.json")
    row=artifacts.read_json(p)
    if damage=="old_workload": row.pop("workload")
    elif damage=="missing_rank": p.unlink()
    elif damage=="missing_stage": row["phases"].pop()
    elif damage=="invalid_context": row["context_samples"][0]["valid"]=False
    else: row["complete"]=False
    if damage!="missing_rank": artifacts.write_json(p,row)
    with pytest.raises(ValueError): memory_budget.from_runs(root,"0:2")


def test_phase_reset_and_context_subtraction_add_no_cuda_sync(monkeypatch):
    cuda=NS(current_device=lambda:0,get_device_properties=lambda _:NS(total_memory=32*1024**3),
        reset_peak_memory_stats=Mock(),max_memory_allocated=lambda:20*1024**3,
        max_memory_reserved=lambda:31*1024**3,memory_reserved=lambda:31*1024**3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","3,2,1,0")
    monkeypatch.setattr(cuda_memory.os,"getpid",lambda:42)
    capture=Mock(return_value=NS(stdout="42,GPU-3,32256"))
    monkeypatch.setattr(cuda_memory.subprocess,"run",capture)
    collector=cuda_memory.StageMemory(NS(cuda=cuda),0)
    collector.begin("prefill")
    collector.observe_context("after_prefill_validation")
    collector.finish()
    doc=collector.document(True)
    assert cuda.reset_peak_memory_stats.call_count==2
    assert [p["stage"] for p in doc["phases"]]==["startup","prefill"]
    assert doc["context_samples"][0]["context_bytes"]==.5*1024**3
    assert capture.call_args.args[0][2]=="3"
    # No synchronize or empty_cache API is provided by the fake CUDA object.


def test_cpu_budget_cli_and_frozen_fields(tmp_path):
    root=write_memory_run(tmp_path/"run")
    out=tmp_path/"out"
    subprocess.run([sys.executable,str(CODE/"tools/resharding/nostop/memory_budget.py"),str(root),"--out",str(out)],
                   check=True,capture_output=True)
    config=formal_plan.frozen_config(out/"frozen.env")
    assert not config["ready"] and config["pressure"]["min_free_gib"]==22
    p=out/"frozen.env"
    p.write_text(p.read_text().replace("MIN_FREE_GIB=22.0","MIN_FREE_GIB=21"))
    with pytest.raises(ValueError,match="ALLOCATED_PEAK_GIB"):
        formal_plan.frozen_config(p)


def test_prefill_already_projects_last_hidden_state_without_changing_kv():
    # Execute the real context constructor and GPT postprocess, without importing
    # CUDA/NCCL-dependent Megatron modules or allocating full-prompt logits.
    class Base:
        def __init__(self, inference_config):
            self.config = inference_config

        def is_static_batching(self):
            return True

        def is_dynamic_batching(self):
            return False

    context_path = CODE / "megatron/core/inference/contexts/static_context.py"
    context_class = next(n for n in ast.parse(context_path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef))
    namespace = dict(BaseInferenceContext=Base, InferenceConfig=NS)
    exec(compile(ast.Module(body=[context_class], type_ignores=[]), str(context_path), "exec"), namespace)
    context = namespace["StaticInferenceContext"](4, 5248)
    context.enable_prefill_mode()
    assert context.config.materialize_only_last_token_logits
    context.sequence_len_offset = 8
    kv = torch.randn(8, 4, 2, 8)
    context.key_value_memory_dict = {1: (kv, kv.clone())}
    before = [(t.data_ptr(), t.clone()) for t in context.key_value_memory_dict[1]]

    model_path = CODE / "megatron/core/models/gpt/gpt_model.py"
    cls = next(n for n in ast.parse(model_path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "GPTModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_postprocess")
    namespace = dict(has_config_logger_enabled=lambda _: False)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(model_path), "exec"), namespace)
    hidden, weight = torch.randn(8, 4, 8), torch.randn(8, 51200)
    seen = []
    def head(states, **kwargs):
        seen.append(tuple(states.shape))
        return states @ weight, None
    model = NS(training=False, share_embeddings_and_output_weights=False, post_process=True,
               config=NS(mtp_num_layers=0), output_layer=head, _scale_logits=lambda x: x)
    logits = namespace["_postprocess"](model, hidden, None, None, None, None, None, None,
                                       inference_context=context, runtime_gather_output=True)
    assert seen == [(1, 4, 8)] and logits.shape == (4, 1, 51200)
    torch.testing.assert_close(logits, (hidden[-1:] @ weight).transpose(0, 1))
    assert context.sequence_len_offset == 8
    for tensor, (pointer, contents) in zip(context.key_value_memory_dict[1], before):
        assert tensor.data_ptr() == pointer
        torch.testing.assert_close(tensor, contents)
