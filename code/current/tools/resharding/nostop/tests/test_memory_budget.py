"""CPU regression for measured admission and the existing last-token logits path."""

import ast
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))

from tools.resharding.nostop import formal_plan, memory_budget


def write_memory_logs(root, peak_mib=20992):
    root.mkdir(parents=True, exist_ok=True)
    (root / "telemetry.log").write_text("timestamp, index, uuid, memory.total [MiB]\n" + "".join(
        f"2026/10/11 10:00:00, {i}, GPU-{i}, 32607 MiB\n" for i in range(8)))
    (root / "process_memory.log").write_text(
        "pid, process_name, gpu_uuid, used_gpu_memory [MiB]\n" + "".join(
            f"42, nvidia-cuda-mps-server, GPU-{i}, 56 MiB\n"
            f"{100+i}, /env/python, GPU-{i}, {peak_mib-i} MiB\n"
            f"{200+i}, /env/python, GPU-{i}, 806 MiB\n" for i in range(8)))
    return root


def test_peak_uses_single_process_including_initialization_not_sum_of_pid_peaks(tmp_path):
    write_memory_logs(tmp_path, 31922)
    budget = memory_budget.from_logs(tmp_path, "0:2,1:3,4:6,5:7")
    assert budget["model_peak_gib"] == 31.173828125
    assert budget["min_free_gib"] == 32.673828125
    assert budget["gpus"]["0"]["peak"]["pid"] == 100
    assert budget["max_l3_processes"] is None  # Even zero stressors cannot leave 2 GiB.
    assert not budget["feasible"] and budget["counts"] == dict(L0=0, L1=0, L2=0, L3=0)
    assert budget["estimated_total_gib"]["7"] == pytest.approx(36.73828125)
    assert len(budget["files_sha256"]) == 2


@pytest.mark.parametrize("peak,count", [(20.5, 7), (24.88, 5), (25.5, 4), (28.52, 0), (31.173828125, None)])
def test_largest_count_and_proportional_levels(peak, count):
    budget = memory_budget.estimate(peak, "0:2,1:3,4:6,5:7", 56)
    assert budget["max_l3_processes"] == count
    assert budget["min_free_gib"] == peak + 1.5
    if count is not None:
        assert 31.36 - budget["estimated_total_gib"][str(count)] >= 2
        if count < 7:
            assert 31.36 - budget["estimated_total_gib"][str(count + 1)] < 2
    if count in (4, 5):
        assert budget["counts"] == dict(L0=0, L1=1, L2=2, L3=count)


def test_shared_gpu_links_and_lower_physical_capacity_are_counted(tmp_path):
    budget = memory_budget.estimate(24.88, "0:1,0:2", 56)
    assert budget["max_pairs_per_gpu"] == 2 and budget["max_l3_processes"] == 2
    write_memory_logs(tmp_path)
    path = tmp_path / "telemetry.log"
    path.write_text(path.read_text().replace("32607", "24576"))
    assert memory_budget.from_logs(tmp_path, "0:2")["capacity_gib"] == 24


@pytest.mark.parametrize("damage", ["missing_gpu", "invalid_memory", "changed_uuid"])
def test_incomplete_memory_evidence_is_rejected(tmp_path, damage):
    write_memory_logs(tmp_path)
    path = tmp_path / ("telemetry.log" if damage == "changed_uuid" else "process_memory.log")
    content = path.read_text()
    if damage == "missing_gpu":
        content = "\n".join(line for line in content.splitlines() if "GPU-7" not in line)
    elif damage == "invalid_memory":
        content = content.replace("20992 MiB", "[N/A]")
    else:
        content += "2026/10/11 10:00:01, 0, GPU-OTHER, 32607 MiB\n"
    path.write_text(content)
    with pytest.raises(ValueError):
        memory_budget.from_logs(tmp_path, "0:2")


def test_cpu_report_and_frozen_env_remain_blocked_for_actual_peak(tmp_path):
    source = write_memory_logs(tmp_path / "l0", 31922)
    out = tmp_path / "report"
    subprocess.run([sys.executable, str(CODE / "tools/resharding/nostop/memory_budget.py"), str(source),
                    "--out", str(out)], check=True, capture_output=True)
    frozen = formal_plan.frozen_config(out / "frozen.env")
    assert not frozen["ready"] and frozen["pressure"]["min_free_gib"] == 32.673828125
    # Editing READY cannot turn an impossible memory budget into an executable plan.
    path = out / "frozen.env"
    path.write_text(path.read_text().replace("DIAGNOSTICS_READY=0", "DIAGNOSTICS_READY=1"))
    assert not formal_plan.frozen_config(path)["ready"]
    path.write_text(path.read_text().replace("MIN_FREE_GIB=32.673828125", "MIN_FREE_GIB=22"))
    with pytest.raises(ValueError, match="measured MODEL_PEAK_GIB"):
        formal_plan.frozen_config(path)


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
    context.sequence_len_offset = 2036
    kv = torch.randn(2036, 4, 2, 8)
    context.key_value_memory_dict = {1: (kv, kv.clone())}
    before = [(t.data_ptr(), t.clone()) for t in context.key_value_memory_dict[1]]

    model_path = CODE / "megatron/core/models/gpt/gpt_model.py"
    cls = next(n for n in ast.parse(model_path.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "GPTModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_postprocess")
    namespace = dict(has_config_logger_enabled=lambda _: False)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(model_path), "exec"), namespace)
    hidden, weight = torch.randn(2036, 4, 8), torch.randn(8, 51200)
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
    assert context.sequence_len_offset == 2036
    for tensor, (pointer, contents) in zip(context.key_value_memory_dict[1], before):
        assert tensor.data_ptr() == pointer
        torch.testing.assert_close(tensor, contents)
