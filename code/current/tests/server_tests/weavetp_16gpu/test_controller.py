"""T04: actual two-host SSH, /proc, exit receipts, resume and signals; CPU workers only.

Uses a small, explicitly synthetic code copy under this test's OUT_DIR. A CPU
profile fixture lets this run before T08. Requires idle GPUs because the unchanged
production runner still executes its GPU preflight. Never changes deployed code.
"""

import argparse
import copy
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from common import ROOT, read, require, run, write

import compare_weavetp_16gpu as compare
import profile_weavetp_16gpu as profile

# The remote bootstrap only creates this bounded JSON file set; no tar extraction,
# shell interpolation of file contents, installs, recursive deletes, or GPU work.
BOOTSTRAP = """
import json,os,subprocess,sys
from pathlib import Path
data=json.load(sys.stdin)
root=Path(data['root']).resolve()
if not root.is_relative_to('/data'): raise ValueError('test root must be under /data')
root.mkdir(parents=True,exist_ok=False)
for name,text in data['files'].items():
    target=(root/name).resolve()
    if not target.is_relative_to(root): raise ValueError('file escaped test root')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf-8') as stream: stream.write(text)
(root/'checkpoint').mkdir()
subprocess.run(['git','init','-q',str(root)],check=True,timeout=15)
subprocess.run(['git','-C',str(root),'-c','core.autocrlf=false','add','-A'],check=True,timeout=15)
dates={**os.environ,'GIT_AUTHOR_DATE':'2000-01-01T00:00:00+0000',
       'GIT_COMMITTER_DATE':'2000-01-01T00:00:00+0000'}
subprocess.run(['git','-C',str(root),'-c','user.name=Fixture',
                '-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false',
                'commit','-q','-m','CPU test fixture'],env=dates,check=True,timeout=15)
print(json.dumps({'created':str(root)}))
"""


def check(args, out, env):
    require(socket.gethostname().split(".")[0].lower() == "sl3060", "run this test only on SL3060")
    if args.profile:
        profile.load_profile(args.profile)
    # Reuse the existing synthetic result factory, not the historical/performance outputs.
    sys.path.insert(0, str(ROOT / "tests/unit_tests/resharding"))
    from test_summarize_weavetp_formal import FormalSummaryTests
    from test_profile_weavetp_16gpu import fixture as profile_fixture

    previous = Path.cwd()
    os.chdir(out)
    fixture = FormalSummaryTests()
    try:
        fixture.setUp()
        fixture.build_fixture(world_size=16)
        result = fixture.read_run(0)
    finally:
        fixture.doCleanups()
        os.chdir(previous)
    result.update(checkpoint_loaded=True, final_active_tp=2, server_test_fixture=True)
    for switch in result["switches"]:
        switch["validation_max_diff"] = 0.1
    test_code_files = (compare.HELPER, compare.WRAPPER, "tools/resharding/run_live_moe_tp_benchmark.sh",
                       "tools/resharding/profile_weavetp_16gpu.py",
                       "tools/resharding/summarize_weavetp_formal.py",
                       "examples/rl/benchmark_live_moe_tp.py")
    files = {name: (ROOT / name).read_text(encoding="utf-8") for name in test_code_files}
    files["examples/rl/benchmark_live_moe_tp.py"] = (
        Path(__file__).with_name("fixtures") / "cpu_worker.py").read_text(encoding="utf-8")
    files[compare.WRAPPER] = ('#!/usr/bin/env bash\nset -euo pipefail\n'
                              'exec "$PYTHON" -B examples/rl/benchmark_live_moe_tp.py --out-dir "$OUT_DIR"\n')
    files["fixture_result.json"] = json.dumps(result)
    files["profile.json"] = (Path(args.profile).read_text(encoding="utf-8") if args.profile
                             else json.dumps(profile_fixture()))
    code = out / "synthetic_code"
    for rank, python in enumerate((sys.executable, str(args.node1_python))):
        bundle = copy.deepcopy(files)
        if args.scenario == "code-mismatch" and rank == 1:
            bundle[compare.WRAPPER] += "# deliberate node1 test mismatch\n"
        argv = [python, "-B", "-c", BOOTSTRAP]
        if rank == 1:
            argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    args.peer_ssh, shlex.join(argv)]
        proc = subprocess.run(argv, input=json.dumps({"root": str(code), "files": bundle}),
                              capture_output=True, text=True, encoding="utf-8", timeout=30, env=env)
        write(out / f"bootstrap.node{rank}.json", {"exit_code": proc.returncode,
                                                  "stdout": proc.stdout, "stderr": proc.stderr})
        require(proc.returncode == 0, f"node{rank} bootstrap failed; no workers started")
    settings = {"root_out": str(out / "synthetic_cases"), "profile": str(code / "profile.json"),
                "checkpoint": str(code / "checkpoint"), "master_port": str(args.master_port),
                "master_addr": args.master_addr, "mps_owner": args.mps_owner,
                "nodes": [{"hostname": "SL3060", "code_dir": str(code), "python": sys.executable},
                          {"hostname": "SL3061", "code_dir": str(code), "python": str(args.node1_python),
                           "ssh": args.peer_ssh}]}
    request = next(compare.make_cases(settings))
    request["env"]["SERVER_TEST_SCENARIO"] = args.scenario
    write(out / "test_request.json", request)
    case_dir = Path(request["out_dir"])
    if args.scenario == "partial-result":
        case_dir.mkdir(parents=True)
        write(case_dir / "result.json", result)

    def call(request, rank, action):
        if args.scenario == "node1-failure" and rank == 1 and action == "run":
            # Ensure the real local CPU worker exists before injecting peer failure.
            deadline = time.monotonic() + 10
            while not (case_dir / "cpu_worker_started.node0").is_file():
                require(time.monotonic() < deadline, "local CPU worker did not start")
                time.sleep(0.05)
        return compare.rpc(request, rank, action)
    if args.scenario == "success":
        compare.run_case(request)  # Real RPC, process launch, wait and exit inspection.
        marker = (case_dir / "complete.json").read_bytes()
        compare.run_case(request)  # Must verify/skip; exclusive files forbid a second launch.
        require((case_dir / "complete.json").read_bytes() == marker, "resume modified completion evidence")
    else:
        stop_interrupt = threading.Event()

        def interrupt_when_running():
            deadline = time.monotonic() + 10
            while not stop_interrupt.wait(0.05) and time.monotonic() < deadline:
                if (case_dir / "cpu_worker_started.node0").is_file():
                    os.kill(os.getpid(), signal.SIGINT)  # This test coordinator only.
                    return

        interrupter = None
        if args.scenario == "controller-interrupt":
            interrupter = threading.Thread(target=interrupt_when_running)
            interrupter.start()
        try:
            compare.run_case(request, call)
        except (ValueError, RuntimeError, KeyboardInterrupt) as exc:
            write(out / "expected_failure.json", {"type": type(exc).__name__, "error": str(exc)})
        else:
            raise ValueError("injected failure was accepted")
        finally:
            stop_interrupt.set()
            if interrupter:
                interrupter.join()
        require(not (case_dir / "complete.json").exists(), "failed case was marked complete")
        states = [compare.rpc(request, rank, "inspect") for rank in (0, 1)]
        if args.scenario in ("node1-failure", "controller-interrupt"):
            if args.scenario == "node1-failure":
                require(states[1]["exit"]["exit_code"] == 7, "did not reach the intended node1 worker failure")
            else:
                require(read(out / "expected_failure.json")["type"] == "KeyboardInterrupt", "SIGINT was not exercised")
            require(states[0]["exit"] is not None, "peer exit receipt missing after cleanup")
            require(states[0]["exit"]["exit_code"] != 0, "local worker finished naturally instead of being stopped")
            require(all(s["exit"] is None or s["exit"]["quiescent"] for s in states), "a host did not return to its baseline")
            failures = list(case_dir.glob("failure.*.json"))
            require(len(failures) == 1 and all(r["ok"] for r in read(failures[0])["cleanup"]),
                    "both real cleanup RPCs must confirm success")
            require(not compare.tagged_processes(request["out_dir"]), "local CPU worker leaked")
        else:
            require(all(s["exit"] is None for s in states), "a worker launched despite a preflight mismatch")
            expected = "hashes differ" if args.scenario == "code-mismatch" else "incomplete evidence"
            require(expected in read(out / "expected_failure.json")["error"], "wrong failure was exercised")
        write(out / "states_after.json", states)
    return {"scenario": args.scenario, "real_ssh_and_processes": True, "gpu_allocated": False,
            "synthetic_results_only": True, "test_code_copy": str(code)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", type=Path, help="optional real profile; otherwise an explicitly synthetic CPU fixture")
    parser.add_argument("--node1-python", type=Path, required=True)
    parser.add_argument("--peer-ssh", required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--mps-owner", help="expected existing MPS owner on node 0")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--scenario", choices=("success", "node1-failure", "controller-interrupt",
                                              "code-mismatch", "partial-result"), required=True)
    raise SystemExit(run(parser.parse_args(), check))
