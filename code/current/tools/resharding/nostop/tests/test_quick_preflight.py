"""CPU tests for quick's explicit utilization exception and matmul admission."""

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.resharding.nostop import artifacts, quick, quick_preflight


def samples():
    return [dict(gpu=i,uuid=f"GPU-{i}",tflops=speed,clock_sample={"clocks.current.sm":"2010" if i in (2,3,6) else "2800"})
            for i,speed in enumerate((240,229,166,166,229,240,166,230))]


@pytest.mark.parametrize("value", ["1,1","8","-1","1,","x","0,1,2,3,4,5,6,7"])
def test_bad_ignore_list_is_rejected_without_gpu_work(value,tmp_path):
    with pytest.raises(SystemExit) as error:
        quick.main(["--out",str(tmp_path/"absent"),"--ignore-util-gpus",value,"--dry-run"])
    assert error.value.code==2 and not (tmp_path/"absent").exists()


def test_dry_run_and_default_are_explicit(tmp_path,capsys):
    for extra,expected in (([],[]),(["--ignore-util-gpus","5,1,4"],[1,4,5])):
        assert quick.main(["--out",str(tmp_path/"absent"),"--dry-run",*extra])==0
        data=json.loads(capsys.readouterr().out)
        assert data["ignore_util_gpus"]==expected
        assert data["matmul_preflight"]==dict(dtype="bf16",size=8192,warmup=5,iterations=50)
    assert not (tmp_path/"absent").exists()


@pytest.mark.parametrize("speed,passed", [(228,True),(227.999,False),(229,True)])
def test_gate_uses_maximum_unignored_not_average_and_allows_locked_cards(speed,passed):
    rows=samples()
    rows[1]["tflops"]=speed
    result=quick_preflight.throughput_gate(rows,[1,4,5])
    assert result["reference_max_tflops"]==240 and result["minimum_tflops"]==228
    assert result["passed"] is passed
    assert result["failed_gpus"]==([] if passed else [1])
    assert quick_preflight.throughput_gate(rows,[])["passed"]  # Locks alone never reject.


@pytest.mark.parametrize("bad", [0,-1,float("nan"),float("inf")])
def test_invalid_throughput_cannot_pass(bad):
    rows=samples()
    rows[0]["tflops"]=bad
    with pytest.raises(ValueError,match="eight positive finite"):
        quick_preflight.throughput_gate(rows,[1,4,5])


def test_measure_enqueues_exact_bf16_work_and_gpu_event_timing():
    operations=[]
    class Event:
        def __init__(self,**kw): assert kw==dict(enable_timing=True)
        def record(self): operations.append("event")
        def synchronize(self): operations.append("end_sync")
        def elapsed_time(self,end): return 250
    def allocate(shape,**kw):
        assert shape==(8192,8192) and kw==dict(device="cuda:0",dtype="bf16")
        return "a"
    torch=NS(cuda=NS(device_count=lambda:1,set_device=lambda n:None,Event=Event,
                    synchronize=lambda:operations.append("warmup_sync")),
             manual_seed=lambda seed:None,inference_mode=nullcontext,bfloat16="bf16",
             randn=allocate,randn_like=lambda a:"b",empty_like=lambda a:"c",
             mm=lambda a,b,**kw:operations.append((a,b,kw)))
    result=quick_preflight.measure(torch,lambda:operations.append("clocks"))
    mm=("a","b",dict(out="c"))
    assert operations==[mm]*5+["warmup_sync","event"]+[mm]*50+["event","clocks","end_sync"]
    assert result["step_ms"]==5
    assert result["tflops"]==pytest.approx(2*8192**3*50/.25/1e12)


@pytest.mark.parametrize("failure", [None,"busy","slow","child","cleanup"])
def test_preflight_owns_isolated_workers_saves_results_and_refuses_failure(monkeypatch,tmp_path,failure):
    args=NS(out=tmp_path,python="python",mps_owner="yiwei",ignore_util_gpus=[1,4,5])
    actions=[]
    def idle(root,owner,ignored):
        assert owner=="yiwei" and ignored==[1,4,5]
        actions.append("idle")
        if failure=="busy": raise RuntimeError("busy")
        return dict(gpu_uuids={i:f"GPU-{i}" for i in range(8)})
    class Processes:
        def __init__(self,root): self.root=root
        def run(self,command,name,env,timeout):
            gpu=int(command[command.index("--gpu")+1])
            assert env["CUDA_VISIBLE_DEVICES"]==f"GPU-{gpu}" and env["CUDA_MPS_PIPE_DIRECTORY"]=="existing"
            assert command[command.index("--uuid")+1]==f"GPU-{gpu}"
            actions.append(gpu)
            if failure=="child" and gpu==2: raise RuntimeError("worker failed")
            row=samples()[gpu]
            if failure=="slow" and gpu==1: row["tflops"]=220
            artifacts.write_json(Path(command[-1]),row)
        def cleanup(self):
            actions.append("cleanup")
            return ["CPU fixture"] if failure=="cleanup" else []
    monkeypatch.setattr(quick_preflight.supervisor,"Processes",Processes)
    monkeypatch.setattr(quick_preflight.machine,"require_idle",idle)
    monkeypatch.setattr(quick_preflight.machine,"ComputeMonitor",lambda *a: NS(start=lambda:None,close=lambda:None))
    monkeypatch.setattr(quick_preflight.time,"sleep",lambda seconds:actions.append(("settle",seconds)))
    manifest=dict(ignore_util_gpus=args.ignore_util_gpus)
    if failure:
        with pytest.raises(RuntimeError): quick_preflight.run(args,dict(CUDA_MPS_PIPE_DIRECTORY="existing"),manifest)
    else:
        quick_preflight.run(args,dict(CUDA_MPS_PIPE_DIRECTORY="existing"),manifest)
        assert actions==["idle",*range(8),("settle",2),"idle","cleanup"]
    saved=artifacts.read_json(tmp_path/"manifest.json")["matmul_preflight"]
    assert saved["state"]==("failed" if failure else "passed")
    assert saved["reported_clock_locks"]==dict(host="SL3060",gpus=[2,3,6],sm_mhz=2010,source="user_report")
    assert actions[-1]=="cleanup"
    assert len(saved["gpus"])==(0 if failure=="busy" else 2 if failure=="child" else 8)


def test_failed_preflight_stops_before_any_method_and_sets_exit_status(monkeypatch,tmp_path):
    # Same CPU host accommodation as the existing quick main subprocess tests.
    monkeypatch.setattr(quick,"os",NS(name="posix",environ={},getpid=lambda:123))
    monkeypatch.setattr(Path,"is_relative_to",lambda *a:True)
    profile=tmp_path/"profile.json"
    profile.write_text("{}")
    method=Mock(side_effect=AssertionError("must not load model"))
    monkeypatch.setattr(quick,"run_method",method)
    monkeypatch.setattr(quick.quick_preflight,"run",Mock(side_effect=RuntimeError("slow ignored GPU")))
    out=tmp_path/"run"
    code=quick.main(["--out",str(out),"--profile",str(profile),"--checkpoint",str(tmp_path),"--ignore-util-gpus","1,4,5"])
    assert code==1 and artifacts.read_json(out/"status.json")["exit_code"]==1
    method.assert_not_called()
