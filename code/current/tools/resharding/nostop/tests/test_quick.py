"""CPU-only quick schedule/handshake and standard-reader compatibility."""

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from tools.resharding.nostop import (
    artifacts,
    common,
    pressure,
    quick,
    quick_benchmark,
    quick_protocol,
    summarize,
)
from tools.resharding.nostop.tests.test_analysis import synthetic_traces
from tools.resharding.nostop.tests.test_pressure_resume import memory_queries


def request(row):
    return dict(level=row["level"], switch=row["switch"], ranks=[dict(rank=i, gpu_uuid=f"GPU-{i}",
        allocated_peak_bytes=20*1024**3, context_peak_bytes=512*1024**2,
        reserved_bytes=10*1024**3, context_bytes=512*1024**2) for i in range(8)])


def test_dry_run_configs_and_reset_bounded_host_loop(tmp_path, capsys):
    for order in ("desc","asc"):
        assert quick.main(["--out",str(tmp_path/"absent"),"--order",order,"--dry-run"]) == 0
        plan=json.loads(capsys.readouterr().out)
        assert plan["runtime_switches_per_launch"]==18
        for method, config in plan["configs"].items():
            expected=common.method_env(method,"cycle",config["PROFILE"],config["OUT_DIR"])
            assert {k:v for k,v in config.items() if k in expected and k!="ENTRYPOINT_OVERRIDE"} == {
                k:v for k,v in expected.items() if k!="ENTRYPOINT_OVERRIDE"}
            assert config["ENTRYPOINT_OVERRIDE"].endswith("quick_benchmark.py")
            assert config["QUICK_ORDER"]==order
    assert not (tmp_path/"absent").exists()
    q=quick_benchmark.Quick.__new__(quick_benchmark.Quick)
    q.schedule,q.prefill_count,q.memory=quick_protocol.schedule(2,"desc"),0,Mock()
    args=NS(live_switches=3,live_prompt_tokens=8)
    q.prefill(MagicMock(),None,None,args)
    assert args.live_switches==18
    assert [i for i in range(18) if q.is_measured(i)]==list(range(2,18,2))


@pytest.mark.parametrize("order,levels", [("desc", ["L3","L2","L1","L0"]), ("asc", ["L0","L1","L2","L3"])])
def test_order_restores_each_pair_and_empties_cache_before_every_request(tmp_path, order, levels):
    rows = quick_protocol.schedule(2, order)
    assert [r["switch"] for r in rows] == list(range(2,18,2))
    assert [r["level"] for r in rows if r["change_level"]] == levels
    q = quick_benchmark.Quick.__new__(quick_benchmark.Quick)
    q.root, q.method, q.rank, q.timeout = tmp_path, "W2", 0, 10
    events = []
    q.torch = NS(cuda=NS(empty_cache=lambda: events.append("empty_cache"), max_memory_allocated=lambda: 20*1024**3))
    q.memory = NS(phases=[], context_samples=[], observe_context=lambda where: q.memory.context_samples.append(
        dict(valid=True, gpu_uuid="GPU-0", reserved_bytes=10*1024**3, context_bytes=512*1024**2)))
    def gather(out, sample, *, group):
        out[:] = [dict(sample,rank=i,gpu_uuid=f"GPU-{i}") for i in range(8)]
    q.dist = NS(barrier=lambda **kw: events.append("barrier"), all_gather_object=gather)
    for row in (r for r in rows if r["change_level"]):
        path = tmp_path / "methods/W2/handshake" / row["level"]
        path.mkdir(parents=True)
        artifacts.save_json(path / "go.json", dict(go=True, level=row["level"], switch=row["switch"]))
        q.change_level(row, "gloo")
        actual = artifacts.read_json(path / "request.json")
        assert actual["switch"] == row["switch"]
        assert quick_protocol.admission(actual)[0] == 22
    assert events == ["barrier","empty_cache","barrier","barrier"] * 4
    with pytest.raises(ValueError, match="mismatch"):
        quick_protocol.acknowledge(tmp_path / "bad_go.json", request(rows[0]), rows[-1])
    assert not (tmp_path / "bad_go.json").exists()


@pytest.mark.parametrize("order,failure", [("desc",None),("asc",None),("desc","gate"),
    ("desc","early_exit"),("desc","last_exit"),("desc","cleanup")])
def test_supervisor_go_only_after_pressure_memory_gate_and_owned_cleanup(monkeypatch, tmp_path, order, failure):
    rows = quick_protocol.schedule(1,order)
    args = NS(out=tmp_path,k=1,order=order,timeout=100,python="python",mps_owner="yiwei")
    actions, children = [], []
    class Manager:
        def __init__(self,path): self.root,self.required,self.children=path,[],[]
        def start(self,command,name,env):
            p=NS(name=name,pid=len(children)+1,done=False,returncode=None)
            children.append(p)
            (self.root/f"{name}.log").write_text("CPU fixture")
            if name=="benchmark":
                for row in rows:
                    artifacts.save_json(self.root/"handshake"/row["level"]/"request.json",request(row))
            return p
        def exited(self,p): return p.done
        def stop(self,p):
            p.done,p.returncode=True,2 if p.name=="benchmark" and failure in ("early_exit","last_exit") else 0
            self.required=[x for x in self.required if x is not p]
            actions.append(("stop",p.name))
        def cleanup(self):
            for p in children: self.stop(p)
            return ["fixture cleanup failure"] if failure=="cleanup" else []
    monkeypatch.setattr(quick.supervisor,"Processes",Manager)
    monkeypatch.setattr(quick.machine,"ComputeMonitor",lambda *a: NS(start=lambda:None,close=lambda:None))
    monkeypatch.setattr(quick.machine,"require_idle",lambda *a: actions.append(("idle","W2")))
    monkeypatch.setattr(quick,"check_method_outputs",lambda *a: None)
    def gate(manager,stress,specs,prefix,minimum,*,resident):
        assert minimum==22 and len(resident)==8
        actions.append(("gate",prefix))
        if failure=="gate": raise RuntimeError("memory gate")
    def start(manager,config,env,prefix,*,resident):
        assert config.processes_per_pair==quick_protocol.COUNTS[prefix]
        gate(manager,[],[],prefix,config.min_free_gib,resident=resident)
        return [manager.start([],prefix,env)]
    monkeypatch.setattr(quick.pressure,"start_stress",start)
    monkeypatch.setattr(quick.pressure,"check_pressure_memory",gate)
    ack=quick_protocol.acknowledge
    def acknowledge(path,req,row):
        assert actions[-1]==("gate",row["level"])
        actions.append(("go",row["level"]))
        ack(path,req,row)
        if row==rows[-1] or failure=="early_exit":
            next(p for p in children if p.name=="benchmark").done=True
    monkeypatch.setattr(quick_protocol,"acknowledge",acknowledge)
    if failure:
        with pytest.raises(RuntimeError): quick.run_method(args,"W2",rows,{})
        if failure=="gate": assert not list(tmp_path.rglob("go.json"))
    else:
        quick.run_method(args,"W2",rows,{})
        assert [v for a,v in actions if a=="go"]==[r["level"] for r in rows]
        assert all((tmp_path/f"{quick_protocol.case_name('W2',r)}.log").read_text()=="CPU fixture" for r in rows)
    assert all(p.done for p in children)
    assert (tmp_path/"methods/W2/exit_code.txt").read_text().strip() == ("1" if failure else "0")
    assert {(tmp_path/quick_protocol.case_name("W2",r)/"exit_code.txt").read_text().strip() for r in rows} == {
        "1" if failure else "0"}


@pytest.mark.parametrize("failed_method,error", [("W2","RuntimeError('GPU failed')"),
    ("B0","RuntimeError('GPU failed')"),("W2","SystemExit(0)"),("W2","SystemExit(143)")])
def test_quick_process_exit_matches_status_and_stops_after_failed_method(tmp_path, failed_method, error):
    profile=tmp_path/"profile.json"
    profile.write_text("{}")
    out=tmp_path/"run"
    script="""
import os, sys
from pathlib import Path
from types import SimpleNamespace
from tools.resharding.nostop import quick
# Exercise production main/exit handling on a CPU host with a temporary output.
quick.os=SimpleNamespace(name='posix',environ=os.environ,getpid=os.getpid)
Path.is_relative_to=lambda self,other: True
def run(args,method,rows,env):
    with (args.out/'methods_called.txt').open('a') as f: f.write(method+'\\n')
    if method==FAILED_METHOD: raise FAILURE
quick.run_method=run
quick.write_figures=lambda root: (_ for _ in ()).throw(AssertionError('must not summarize failure'))
raise SystemExit(quick.main(sys.argv[1:]))
""".replace("FAILED_METHOD",repr(failed_method)).replace("FAILURE",error)
    p=subprocess.run([sys.executable,"-B","-c",script,"--out",str(out),"--checkpoint",str(tmp_path),
                      "--profile",str(profile)],cwd=quick.supervisor.CODE,
                     capture_output=True,text=True)
    status=artifacts.read_json(out/"status.json")
    assert p.returncode != 0 and status["state"]=="failed"
    assert status["exit_code"]==p.returncode==int((out/"exit_code.txt").read_text())
    assert (out/"methods_called.txt").read_text().splitlines()==(["W2"] if failed_method=="W2" else ["W2","B0"])


def test_resident_gate_keeps_raw_free_and_rejects_shortfall(monkeypatch,tmp_path):
    row=quick_protocol.schedule(1,"desc")[0]
    minimum,resident=quick_protocol.admission(request(row))
    memory_queries(monkeypatch,free=12*1024)
    result=pressure.check_pressure_memory(NS(root=tmp_path),[],[],"ok",minimum,resident=resident)
    assert result["gpus"][-1]["free_mib"]==12*1024
    assert result["gpus"][-1]["available_with_resident_gib"]==22.5
    memory_queries(monkeypatch,free=11*1024)
    with pytest.raises(RuntimeError,match="memory gate"):
        pressure.check_pressure_memory(NS(root=tmp_path),[],[],"bad",minimum,resident=resident)


def write_quick(root):
    root.mkdir()
    rows=quick_protocol.schedule(1,"desc")
    artifacts.write_json(root/"manifest.json",dict(cases=[[quick_protocol.case_name(m,r),r["level"],m,"cycle"]
        for m in ("W2","B0") for r in rows],profile_sha256="synthetic",checkpoint="synthetic",
        stress_links=quick_protocol.LINKS,stress_bytes=common.STRESS_BYTES,stress_sync_every=16,
        processes_per_pair=quick_protocol.COUNTS))
    for method in ("W2","B0"):
        for row in rows:
            case=root/quick_protocol.case_name(method,row)
            (case/"run").mkdir(parents=True)
            artifacts.write_json(case/"config.json",common.method_env(method,"cycle","p",case/"run"))
            (case/"exit_code.txt").write_text("0")
            for raw in synthetic_traces(stw=method=="B0"):
                raw.update(memory=dict(phases=[],complete=True),transfers=[],
                           validation_audits=[dict(switch=i,kv_passed=True,weights_passed=True)
                                              for i in (0,1,row["switch"],row["switch"]+1)],
                           logit_metrics=[dict(switch=i,metrics=dict(nrmse=.411,cosine=.9156))
                                          for i in (0,1,row["switch"],row["switch"]+1)])
                for field in ("decodes","waves","milestones","state_checks","plan_records"):
                    for r in raw[field]:
                        if r["switch"]==2: r["switch"]=row["switch"]
                result=dict(switches=[dict(index=i,direction="2->4" if i%2==0 else "4->2",
                    base=dict(wave_records=[dict(transport_s=1)])) for i in (0,1,row["switch"])],nostop_diagnostic={})
                before=copy.deepcopy(raw)
                trace,selected=quick_protocol.project(raw,result,row,f"{root.name}:{method}")
                assert raw==before and trace["quick"]["original_switch"]==row["switch"]
                assert [r["switch"] for r in trace["validation_audits"]] == [0,1,2]
                assert [r["switch"] for r in trace["logit_metrics"]] == [0,1,2]
                artifacts.write_json(case/f"run/trace_rank{raw['rank']}.json",trace)
                artifacts.write_json(case/f"run/memory_rank{raw['rank']}.json",trace["memory"])
                artifacts.write_json(case/"run/result.json",selected)
                artifacts.write_json(case/"run/routes.json",{})


def test_quick_csv_and_multiple_runs_are_readable_without_reader_adapters(tmp_path):
    for name in ("launch_a","launch_b"):
        root=tmp_path/name
        write_quick(root)
        for method in ("W2","B0"):
            quick.check_method_outputs(root,method,quick_protocol.schedule(1,"desc"))
        quick.write_figures(root)
        with (root/"downtime.csv").open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
        assert len(rows)==8 and {r["n"] for r in rows}=={"1"}
        assert {float(r["b0_migration_bandwidth_gbps"]) for r in rows}=={800/1e9}
        assert len((root/"step_latency.csv").read_text().splitlines())==9
    merged=summarize.collect([tmp_path/"launch_a",tmp_path/"launch_b"])
    assert not merged["errors"] and len(merged["launches"])==16
    assert len({l["launch_id"] for l in merged["launches"]})==16
    assert len({l["process_launch_id"] for l in merged["launches"]})==4
    assert {r["n"] for r in summarize.aggregate(merged["launches"]) if r["metric"]=="raw_max_gap_s"}=={2}


@pytest.mark.parametrize("missing", ["result.json","trace_rank7.json","memory_rank7.json","routes.json"])
def test_zero_exit_without_complete_outputs_is_failure(tmp_path,missing):
    root=tmp_path/"fixture"
    write_quick(root)
    (root/"W2_L3_R1/run"/missing).unlink()
    with pytest.raises((OSError,ValueError)):
        quick.check_method_outputs(root,"W2",quick_protocol.schedule(1,"desc"))
