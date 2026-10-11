"""CPU-only evidence/ownership tests; never initialize a CUDA context."""

import ast
import json
import signal
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

CODE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(CODE))

from tools.resharding.nostop import benchmark, common, diag_report, diagnostics


def traces(gap_excess=0, host_excess=0, mode="cycle", stw=False):
    result = []
    measured = 2 if mode == "cycle" else 0
    for rank in range(8):
        rows, waves = [], []
        done = 1_000_000_000 + rank * 100
        rows.append(dict(rank=rank, step=0, switch=measured, phase="normal", wave=None,
                         kv_offset=2047, host_start_ns=done - 50_000_000, complete_host_ns=done))
        for wave in range(3):
            done += 200_000_000 + (gap_excess if wave == 0 else 0)
            start = done - 50_000_000
            waves.append(dict(switch=measured, wave=wave,
                              submit_host_ns=start - 10_000_000 - (host_excess if wave == 0 else 0)))
            if not stw:
                rows.append(dict(rank=rank, step=wave+1, switch=measured, phase="base", wave=wave,
                                 kv_offset=2048+wave, host_start_ns=start, complete_host_ns=done))
        for i in range(10):
            rows.append(dict(rank=rank, step=4+i, switch=measured, phase="post", wave=None,
                             kv_offset=2051+i, host_start_ns=done+i*100_000_000,
                             complete_host_ns=done+(i+1)*100_000_000))
        result.append(dict(rank=rank, measured_switch=measured, stall_pause=stw,
                           clock=dict(monotonic=True, boot_id="same", time_namespace="same"),
                           decodes=rows, waves=waves, state_checks=[dict(measured=True, kv_tokens=2048,
                           target_weight_storage_bytes=0, target_weight_bytes_required=1000)],
                           independent_warmup=dict(scratch_bytes=128) if mode == "independent" else None))
    return result


@pytest.mark.parametrize("gap,host,status", [(0,0,"passed"), (100_000_000,50_000_000,"passed"),
    (100_000_001,0,"failed"), (0,50_000_001,"failed"), (-50_000_000,0,"passed")])
def test_cold_thresholds_are_strict_nanosecond_boundaries(gap, host, status):
    assert diag_report.cold_check(traces(gap, host))["status"] == status


@pytest.mark.parametrize("damage", ["missing_rank", "clock", "logical_step", "one_wave", "reverse_time"])
def test_cold_incomplete_or_incomparable_evidence_never_passes(damage):
    raw = traces()
    if damage == "missing_rank":
        raw.pop()
    elif damage == "clock":
        raw[3]["clock"]["boot_id"] = "other"
    elif damage == "logical_step":
        raw[3]["decodes"][1]["kv_offset"] += 1
    elif damage == "one_wave":
        for t in raw:
            t["decodes"] = t["decodes"][:2]
            t["waves"] = t["waves"][:1]
    else:
        for t in raw:
            t["decodes"][1]["complete_host_ns"] = 0
    assert diag_report.cold_check(raw)["status"] == "inconclusive"


def test_slowest_rank_completion_and_rank_local_host_blocking():
    raw = traces()
    raw[7]["decodes"][1]["complete_host_ns"] += 110_000_000
    report = diag_report.cold_check(raw)
    assert report["status"] == "failed"
    assert report["wave_max_gap_s"][0] == pytest.approx(.310)
    assert report["wave_submit_to_decode_host_s"][0] == .010


def probe(speed, candidate_speed=100):
    return dict(complete=True, links=[dict(src=src, dst=dst, samples=[dict(gbps=value)]*5)
                for src, dst, value in ((0,2,speed), (1,3,candidate_speed))])


DEFAULT = [dict(src=0,dst=2,kind="weight",bytes=100), dict(src=0,dst=0,kind="weight",bytes=1000)]
CANDIDATE = [dict(src=1,dst=3,kind="weight",bytes=100)]


@pytest.mark.parametrize("speed,status", [(40,"passed"),(40.01,"failed"),(20,"passed")])
def test_pressure_uses_measured_bandwidth_not_process_count(speed, status):
    result = diag_report.pressure_check(probe(100), probe(speed), DEFAULT, CANDIDATE)
    assert result["status"] == status
    assert result["b0_weighted_slowdown"] == pytest.approx(100/speed)
    assert result["links"][1]["slowdown"] == 1  # legal alternative source needn't slow
    assert result["l0_effective_gbps"] == 100


@pytest.mark.parametrize("damage", ["incomplete", "missing_link", "nan", "no_samples"])
def test_bad_probes_are_inconclusive(damage):
    high = probe(20)
    if damage == "incomplete":
        high["complete"] = False
    elif damage == "missing_link":
        high["links"].pop()
    elif damage == "nan":
        high["links"][0]["samples"] = [dict(gbps=float("nan"))]*5
    else:
        high["links"][0].pop("samples")
    assert diag_report.pressure_check(probe(100), high, DEFAULT, CANDIDATE)["status"] == "inconclusive"


def test_probe_variability_preserved_and_flagged():
    high = probe(20)
    high["links"][0]["samples"] = [dict(gbps=v) for v in (10,20,30,40,50)]
    result = diag_report.pressure_check(probe(100), high, DEFAULT, CANDIDATE)
    assert result["links"][0]["variability_flag"]
    assert result["links"][0]["l3_cv"] > .15
    assert len(high["links"][0]["samples"]) == 5


def test_complete_report_has_only_two_decisions_and_frozen_recommendation(tmp_path):
    diagnostics.write_json(tmp_path / "manifest.json", dict(execution_complete=True, stress_links="0:2", l3_processes=7,
        stress_bytes=256, stress_sync_every=16, profile="/data/profile.json"))
    for name, _, method, mode in diagnostics.diagnostic_cases():
        case = tmp_path / name
        (case / "run").mkdir(parents=True)
        (case / "exit_code.txt").write_text("0\n")
        for t in traces(mode=mode, stw=method == "B0"):
            diagnostics.write_json(case / f"run/trace_rank{t['rank']}.json", t)
    diagnostics.write_json(tmp_path / "probe_routes.json", dict(default=DEFAULT, candidate=CANDIDATE))
    diagnostics.write_json(tmp_path / "probe_L0.json", probe(100))
    diagnostics.write_json(tmp_path / "probe_L3.json", probe(30))
    result = diag_report.make_report(tmp_path)
    assert set(result) == {"cold_start", "pressure"}
    assert result["cold_start"]["recommended_mode"] == "cycle"
    frozen = (tmp_path / "frozen.env").read_text()
    assert "export DIAGNOSTICS_READY=1" in frozen and "export SWITCHES=3" in frozen
    assert "export MAX_WAVES=64" in frozen and "export MAX_POSITION_EMBEDDINGS=5248" in frozen
    assert "export NCCL_RUNTIME_CONNECT=" not in frozen
    # A completed probe file cannot override a later pressure/monitor failure.
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["execution_complete"] = False
    diagnostics.write_json(tmp_path / "manifest.json", manifest)
    result = diag_report.make_report(tmp_path)
    assert result["pressure"]["status"] == "inconclusive"
    assert "export DIAGNOSTICS_READY=0" in (tmp_path / "frozen.env").read_text()
    # A failed smoke or absent probe cannot recommend a ready configuration.
    (tmp_path / "smoke_L0_B0/exit_code.txt").write_text("1\n")
    (tmp_path / "probe_L3.json").unlink()
    result = diag_report.make_report(tmp_path)
    assert result["cold_start"]["status"] == result["pressure"]["status"] == "inconclusive"
    assert "export DIAGNOSTICS_READY=0" in (tmp_path / "frozen.env").read_text()


@pytest.mark.parametrize("damage", ["kv", "resident_weights", "no_decode", "no_post", "no_warmup"])
def test_state_gates(damage):
    raw = traces(mode="independent")
    if damage == "kv":
        raw[0]["state_checks"][0]["kv_tokens"] = 2049
    elif damage == "resident_weights":
        raw[0]["state_checks"][0]["target_weight_storage_bytes"] = 1
    elif damage == "no_decode":
        raw[0]["decodes"] = [r for r in raw[0]["decodes"] if r["phase"] != "base"]
    elif damage == "no_post":
        raw[0]["decodes"].pop()
    else:
        raw[0]["independent_warmup"] = None
    assert diag_report.launch_gate(raw, "independent", False)


def test_capacity_accounts_for_two_warmup_switches_and_validation():
    budget = common.capacity_budget()
    assert budget["warmup_peak_tokens"] == 4130
    assert budget["measured_peak_tokens"] == 3098
    assert budget["high_water_tokens"] == 4130
    assert budget["benchmark_worst_case_tokens"] == 5163
    assert budget["max_position_embeddings"] == 5248
    envs = [common.method_env(m,c,"p","o") for m in ("W2","B0") for c in ("cycle","independent")]
    assert {e["MAX_POSITION_EMBEDDINGS"] for e in envs} == {"5248"}
    assert {e["PROMPT_TOKENS"] for e in envs} == {"2036"}
    assert all("NCCL_RUNTIME_CONNECT" not in e for e in envs)
    assert envs[0]["ADAPTIVE_MIN_PREDICTED_E2E_GAIN_PCT"] == "5.0"
    assert envs[0]["HYBRID_MAX_FOREGROUND_PRESSURE"] == "0.20"
    assert [e["EXPANSION_MAX_WAVE_TASKS"] for e in envs] == ["4096","4096","2048","2048"]


@pytest.mark.parametrize("method", ["W2", "B0"])
@pytest.mark.parametrize("mode", ["cycle", "independent"])
def test_capacity_passes_original_benchmark_guard(method, mode):
    # Execute the production formula AND guard, without importing CUDA/Megatron.
    tree = ast.parse((CODE / "examples/rl/benchmark_live_moe_tp.py").read_text(encoding="utf-8"))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_live_benchmark")
    start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "worst_case_tokens" for t in n.targets))
    guard = ast.Module(body=run.body[start:start + 3], type_ignores=[])
    env = common.method_env(method, mode, "p", "o")
    args = NS(live_prompt_tokens=int(env["PROMPT_TOKENS"]), live_switches=int(env["SWITCHES"]),
              live_max_waves=int(env["MAX_WAVES"]), live_max_overlap_steps=int(env["MAX_OVERLAP_STEPS"]),
              live_release_standby_weights=env["RELEASE_STANDBY_WEIGHTS"] == "1",
              max_position_embeddings=int(env["MAX_POSITION_EMBEDDINGS"]))
    namespace = dict(args=args)
    exec(compile(guard, "production_capacity_guard", "exec"), namespace)
    assert namespace["worst_case_tokens"] < args.max_position_embeddings
    assert env["SEQ_LENGTH"] == env["MAX_POSITION_EMBEDDINGS"] == "5248"
    assert args.live_max_waves == 64


@pytest.mark.parametrize("links", ["0:0","0:8","-1:2","0","a:b"])
def test_invalid_pressure_links(links):
    with pytest.raises(ValueError):
        common.parse_links(links)


def test_stress_parameters_and_launch_order():
    args = NS(python="python3", links="2:0,0:2,1:3", processes_per_pair=3, stress_bytes=256, stress_sync_every=16)
    commands = diagnostics.stress_commands(args)
    assert len(commands) == 6 and all("--bidirectional" in c for c in commands)
    assert diagnostics.diagnostic_cases() == [
        ("smoke_L0_W2","L0","W2","cycle"), ("smoke_L0_B0","L0","B0","cycle"),
        ("L0_W2_cycle","L0","W2","cycle"), ("L0_W2_independent","L0","W2","independent"),
        ("L3_W2_cycle","L3","W2","cycle"), ("L3_W2_independent","L3","W2","independent")]


@pytest.mark.parametrize("mode", ["machine","diag"])
def test_dry_run_never_creates_output_or_invokes_gpu(tmp_path, mode):
    out = tmp_path / "must_not_exist"
    result = subprocess.run([sys.executable, str(CODE / "tools/resharding/nostop/diagnostics.py"),
        mode, "--dry-run", "--out", str(out)], capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert not out.exists() and not plan["automatic_formal_launch"]
    assert len(plan["cases"]) == (6 if mode == "diag" else 0)


def fake_diagnostic(monkeypatch, stall=False):
    calls = []
    class Event:
        def __init__(self, **_kw):
            pass
        def record(self):
            calls.append("record")
        def synchronize(self):
            calls.append("event_sync")
        def elapsed_time(self, _end):
            return 12.0
    cuda = NS(Event=Event, current_device=lambda: 0)
    torch = NS(cuda=cuda, inference_mode=nullcontext, full=lambda *a, **kw: None,
               full_like=lambda *a, **kw: None, device=lambda *a: None, long="long")
    args = NS(live_switches=3, live_prompt_tokens=2036, live_release_standby_weights=True,
              live_repeat_forward=False, live_weight_bitwise_audit=False, max_position_embeddings=5248, seq_length=5248,
              live_max_waves=64, micro_batch_size=4, padded_vocab_size=100, live_kv_request_identity=False)
    monkeypatch.setenv("NOSTOP_COLD_MODE", "cycle")
    monkeypatch.setenv("STALL_PAUSE", "1" if stall else "0")
    b = NS(torch=torch, dist=NS(get_rank=lambda:0), _apply_hotspot_pressure=lambda *a:None)
    d = benchmark.Diagnostic(b,args)
    d.origin, d.index, d.phase, d.wave = Event(), 2, "base", 0
    context = NS(sequence_len_offset=2048, max_sequence_length=5248, enable_decode_mode=lambda:None)
    model = Mock(return_value="logits")
    model.config.tensor_model_parallel_size = 2
    def now():
        calls.append("clock")
        return len(calls) * 1_000_000
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", now)
    return d, context, model, calls


def test_decode_reuses_event_sync_and_timestamps_immediately_after(monkeypatch):
    d, context, model, calls = fake_diagnostic(monkeypatch)
    logits, ms = d.decode(None, model, context, d.args, token_value=7)
    assert (logits,ms) == ("logits",12)
    assert calls == ["clock","record","record","event_sync","clock"]
    assert context.sequence_len_offset == 2049
    row = d.decodes[0]
    assert (row["rank"],row["step"],row["tp"],row["phase"],row["wave"],row["kv_offset"]) == (0,0,2,"base",0,2048)


@pytest.mark.parametrize("phase", ["base","weight_check","delta","prepare"])
def test_stw_forbids_decode_entire_migration(monkeypatch, phase):
    d, context, model, calls = fake_diagnostic(monkeypatch, stall=True)
    d.phase = phase
    with pytest.raises(AssertionError, match="STW"):
        d.decode(None, model, context, d.args, token_value=7)
    model.assert_not_called()
    assert not calls


def test_event_overlap_allows_decode_to_finish_after_transfer():
    assert common.intervals_overlap(3,8,1,5)
    assert not common.intervals_overlap(5,8,1,5)
    assert not common.intervals_overlap(7,8,1,5)


def test_capture_excludes_participation_marker_from_model_overlap(monkeypatch):
    d, _, _, _ = fake_diagnostic(monkeypatch)
    service = NS(_inflight=NS(_timing_stages={"nccl":[("s","e")]}),
                 last_launch_stats=dict(original_remote_sends=0,original_remote_recvs=0))
    d.capture_transfer(service)
    assert not d.transfers[0]["model_payload"]
    assert d.transfers[0]["events"] == ("s","e")


def test_cleanup_signals_only_owned_group_before_reaping(monkeypatch, tmp_path):
    monkeypatch.setattr(signal,"SIGKILL",9,raising=False)  # model Linux on Windows CPU CI
    events = []
    process = NS(pid=54321, returncode=None)
    def wait(**_kw):
        events.append("reap")
        process.returncode = 0
    process.wait = wait
    monkeypatch.setattr(diagnostics.os,"killpg",lambda pid,sig: events.append((pid,sig)),raising=False)
    manager = diagnostics.Processes(tmp_path)
    monkeypatch.setattr(manager,"exited",lambda _p:True)
    log = (tmp_path / "child.log").open("w")
    record = dict(name="child",pid=54321,exit_code=None)
    manager.children.append((process,log,record))
    manager.records.append(record)
    manager.stop(process)
    first_end = record["ended_ns"]
    manager.cleanup()  # idempotent; no reused PID signals
    assert events == [(54321,signal.SIGTERM),(54321,signal.SIGKILL),"reap"]
    assert (tmp_path / "child.exit_code.txt").read_text() == "0\n"
    assert record["ended_ns"] == first_end


def test_benchmark_hook_skips_both_wave_and_weight_check_loops():
    tree = ast.parse((CODE / "examples/rl/benchmark_live_moe_tp.py").read_text(encoding="utf-8"))
    wave = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == "_run_async_waves")
    weight_loop = next(n for n in ast.walk(tree) if isinstance(n,ast.For) and "diagnostic.stall_pause" in ast.unparse(n.iter))
    for stall, expected in ((True,False),(False,True)):
        diagnostic = NS(stall_pause=stall,check_wave_count=Mock(),wave_begin=Mock(),wave_end=Mock())
        args = NS(live_max_overlap_steps=16, live_min_overlap_steps=1, live_scheduler_mode="baseline",
                  live_max_wave_tasks=2, live_max_waves=64,live_active_experts_tuple=(),
                  live_release_standby_weights=True,live_foreground_pressure_cap=.9,_nostop_diagnostic=diagnostic)
        launches = []
        tasks = [NS(task_id=i,num_bytes=100) for i in range(5)]
        def launch(plan,*a,**kw):
            launches.append(plan)
            return NS(metadata={},done=lambda:True,wait=lambda:None,commit=lambda:None,
                      transport_elapsed_s=.1,transport_stage_elapsed_s={})
        decode = Mock(return_value=(None,10))
        env = dict(torch=NS(inference_mode=lambda:lambda f:f,
                            cuda=NS(stream=lambda s:nullcontext(),current_stream=lambda:NS(synchronize=lambda:None))),
                   dist=NS(get_rank=lambda:0,broadcast_object_list=lambda *a,**kw:None,barrier=lambda **kw:None),
                   time=time,statistics=statistics,nullcontext=nullcontext,
                   collect_transfer_tasks=lambda *a,**kw:tasks, _tasks_for_ids=lambda t,ids:[t[i] for i in ids],
                   filter_plan_by_task_ids=lambda p,ids:ids, launch_reshard_plan=launch,_decode=decode,
                   _all_max=lambda value,g:value,_all_ready=lambda value,g:value,_percentile=lambda values,q:0)
        module = ast.Module(body=ast.parse("from __future__ import annotations").body+[wave],type_ignores=[])
        exec(compile(module,"test","exec"),env)
        result,token = env["_run_async_waves"](plan=None,src_bundle=None,dst_bundle=NS(named_parameters=lambda:[]),
            active_model=None,active_context=None,service=NS(producer_stream=None),
            tracker=NS(observe_tpot_pressure=lambda *a,**kw:0,observe_wave=lambda *a:{}),scheduler=None,
            reshard_group=None,control_group=None,args=args,token_counter=100,foreground_baseline_tpot_ms=10,
            foreground_links=set())
        assert launches == [[0,1],[2,3],[4]] and result["waves"] == 3
        assert decode.call_count == (3 if expected else 0)
        assert token == 100 + decode.call_count
        env.update(diagnostic=diagnostic,args=args,weight_checked=NS(query=lambda:False),control_group=None,
                   source_model=None,source_context=None,token_counter=0,check_tpot_ms=[])
        decode.reset_mock()
        exec(compile(ast.Module(body=[weight_loop],type_ignores=[]),"test","exec"),env)
        assert decode.call_count == (16 if expected else 0)
        assert len(env["check_tpot_ms"]) == decode.call_count


def test_wave_budget_refuses_overflow(monkeypatch):
    d,_,_,_ = fake_diagnostic(monkeypatch)
    d.check_wave_count(63)
    with pytest.raises(RuntimeError,match="budget"):
        d.check_wave_count(64)


def test_third_switch_restores_real_prefix_and_2048_tokens(monkeypatch):
    d, _, model, _ = fake_diagnostic(monkeypatch)
    d.torch = NS(inference_mode=torch.inference_mode,
                 cuda=NS(current_stream=lambda:NS(synchronize=lambda:None),reset_peak_memory_stats=lambda:None))
    source = NS(sequence_len_offset=2037, key_value_memory_dict={1:(torch.ones(5248,1),torch.ones(5248,1))},
                enable_decode_mode=lambda:None)
    target = NS(sequence_len_offset=2037, key_value_memory_dict={1:(torch.ones(5248,1),torch.ones(5248,1))},
                enable_decode_mode=lambda:None)
    weights = {tp:NS(_check_storage=Mock(),num_bytes=900) for tp in (2,4)}
    def decode(_model, context, _args, *, token_value):
        for tensor in context.key_value_memory_dict[1]:
            tensor[context.sequence_len_offset] = token_value
        context.sequence_len_offset += 1
    d.b._decode = decode
    def run(index):
        return d.before_switch(index,2,model,source,model,target,weights,{},None,None,None,None,999)
    first_token = run(0)
    reference = [t[:2048].clone() for t in source.key_value_memory_dict[1]]
    # Warmup migrations may write beyond the original prompt but not within it.
    for context in (source,target):
        context.sequence_len_offset = 4000
        for tensor in context.key_value_memory_dict[1]:
            tensor[2036:] = -7
    assert run(2) == first_token
    assert source.sequence_len_offset == target.sequence_len_offset == 2048
    assert all(torch.equal(t[:2048],r) for t,r in zip(source.key_value_memory_dict[1],reference))
    assert d.states[-1]["measured"] and d.states[-1]["real_prefill_tokens"] == 2036
    assert weights[4]._check_storage.call_count == 4


def test_post_cutover_completes_target_before_numerical_validation(monkeypatch):
    d,source,model,_ = fake_diagnostic(monkeypatch)
    target = NS(sequence_len_offset=2048)
    order = []
    def decode(_model,context,_args,*,token_value):
        order.append((d.phase,context))
        context.sequence_len_offset += 1
        return torch.tensor([float(token_value)]), 12.0
    d.b._decode = decode
    d.b._comparison_metrics = lambda *a:dict(nrmse=0,cosine=1,top1_agreement=1)
    d.b._all_ready = lambda valid,_group:valid
    d.b._all_max = lambda value,_group:value
    d.args.live_logit_max_nrmse,d.args.live_logit_min_cosine,d.args.live_logit_min_top1_agreement = .4,.93,0
    output = d.validate(None,model,source,model,target,d.args,None,100)
    assert order == [("post",target)]*10 + [("validation",source)]*10
    assert source.sequence_len_offset == target.sequence_len_offset == 2058
    assert output == (0.0,110,12.0,12.0)


def test_unreaped_status_poll_uses_wnowait(monkeypatch):
    for name,value in (("P_PID",1),("WEXITED",4),("WNOHANG",1),("WNOWAIT",0x1000000)):
        monkeypatch.setattr(diagnostics.os,name,value,raising=False)
    waitid = Mock(return_value=None)
    monkeypatch.setattr(diagnostics.os,"waitid",waitid,raising=False)
    p = NS(pid=99,returncode=None)
    assert not diagnostics.Processes.exited(p)
    assert waitid.call_args.args == (1,99,4|1|0x1000000)
    waitid.return_value = NS(si_status=0)
    assert diagnostics.Processes.exited(p)
    assert p.returncode is None


@pytest.mark.parametrize("installed,sample_error", [(False, False), (True, False), (False, True)])
def test_machine_records_optional_nsys_and_requires_all_gpu_samples(monkeypatch, tmp_path, installed, sample_error):
    commands = []
    gpu_samples = 0
    def run(command, **_kwargs):
        nonlocal gpu_samples
        commands.append(command)
        code, output = 0, "ok\n"
        if command == ["which", "nsys"]:
            code, output = (0, "/usr/bin/nsys\n") if installed else (1, "")
        elif command == ["nsys", "--version"]:
            if not installed:
                raise FileNotFoundError("nsys")
            output = "NVIDIA Nsight Systems version test\n"
        elif any(arg.startswith("--query-compute-apps") for arg in command):
            output = ""
        elif any(arg.startswith("--query-gpu") for arg in command):
            gpu_samples += 1
            output = "\n".join(f"{i},GPU-{i},RTX 5090,32000,0,0,30,210,405,P8,15,580" for i in range(8))
            if sample_error and gpu_samples == 2:
                code, output = 1, "GPU query failed"
        return subprocess.CompletedProcess(command, code, stdout=output)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: 0)
    monkeypatch.setattr(diagnostics.time, "sleep", lambda s: None)
    result = diagnostics.machine_check(tmp_path, 1 if sample_error else 0, "python")
    assert result["idle_state"] == ("unknown" if sample_error else "idle")
    assert result["checks"]["runtime"]["exit_code"] == 0
    assert result["checks"]["nsys"]["status"] == ("available" if installed else "missing")
    assert ["which", "nsys"] in commands and ["nsys", "--version"] in commands
    assert (tmp_path / "nsys_which.txt").exists() and (tmp_path / "nsys_version.txt").exists()


def test_supervisor_checks_monitors_even_when_child_already_exited(monkeypatch, tmp_path):
    manager = diagnostics.Processes(tmp_path)
    process = NS(returncode=None)
    monitor = NS(returncode=None)
    manager.required = [monitor]
    monkeypatch.setattr(manager, "start", lambda *a: process)
    monkeypatch.setattr(manager, "exited", lambda p: True)
    stopped = Mock()
    monkeypatch.setattr(manager, "stop", stopped)
    with pytest.raises(RuntimeError, match="pressure/telemetry"):
        manager.run(["benchmark"], "case")
    stopped.assert_called_once_with(process)


def test_diagnostic_workers_use_callable_torchrun_entrypoint():
    from torch.distributed.run import config_from_args, parse_args, run_script_path

    # Callable entries use multiprocessing workers, not torchrun's subprocess
    # handler (which calls Popen(start_new_session=True)). No workers are started.
    for script in ("benchmark.py", "probe.py"):
        path = CODE / "tools/resharding/nostop" / script
        _, entrypoint, argv = config_from_args(parse_args([
            "--standalone", "--nproc-per-node=8", "--run-path", str(path)]))
        assert entrypoint is run_script_path and argv == [str(path)]
