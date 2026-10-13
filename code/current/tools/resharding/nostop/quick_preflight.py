"""Quick-only matmul admission; isolated GPU children exit before model loading."""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.nostop import artifacts, machine, supervisor

MATMUL = dict(dtype="bf16", size=8192, warmup=5, iterations=50)
# Supplied experimental context, not a throughput-based inference or admission exemption.
REPORTED_CLOCK_LOCKS = dict(host="SL3060", gpus=[2, 3, 6], sm_mhz=2010, source="user_report")


def parse_ignored(value):
    if not value.strip():
        return []
    gpus = [int(v) for v in value.split(",")]
    if len(set(gpus)) != len(gpus) or not set(gpus) < set(range(8)):
        raise ValueError("use distinct GPU indices 0..7 and leave at least one reference GPU")
    return sorted(gpus)


def throughput_gate(rows, ignored):
    if len(rows) != 8 or {r["gpu"] for r in rows} != set(range(8)) or any(
            not math.isfinite(r["tflops"]) or r["tflops"] <= 0 for r in rows):
        raise ValueError("matmul admission requires eight positive finite GPU measurements")
    reference = max(r["tflops"] for r in rows if r["gpu"] not in ignored)
    failed = [r["gpu"] for r in rows if r["gpu"] in ignored and r["tflops"] < .95 * reference]
    return dict(passed=not failed, reference_max_tflops=reference, minimum_tflops=.95 * reference,
                max_slowdown_fraction=.05, failed_gpus=failed)


def measure(torch, clock_sample):
    if torch.cuda.device_count() != 1:
        raise ValueError("matmul worker must see exactly its assigned GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    n = MATMUL["size"]
    with torch.inference_mode():
        a = torch.randn((n,n), device="cuda:0", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        c = torch.empty_like(a)
        for _ in range(MATMUL["warmup"]):
            torch.mm(a,b,out=c)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(MATMUL["iterations"]):
            torch.mm(a,b,out=c)
        end.record()
        # Sample after all work is queued; no host query between GPU timing events.
        clocks = clock_sample()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
    return dict(elapsed_ms=elapsed_ms, step_ms=elapsed_ms / MATMUL["iterations"],
                tflops=2 * n**3 * MATMUL["iterations"] / (elapsed_ms / 1000) / 1e12,
                clock_sample=clocks)


def worker(gpu, uuid, output):
    import torch

    def clocks():
        fields = "index,uuid,clocks.current.sm,clocks.current.memory,clocks.max.sm,pstate,temperature.gpu"
        query = supervisor.capture(["nvidia-smi", "-i", uuid, f"--query-gpu={fields}",
                                    "--format=csv,noheader,nounits"], output.parent, f"matmul_gpu{gpu}_clocks")
        if query["exit_code"]:
            raise RuntimeError(f"matmul clock/identity query failed: {query}")
        values, = list(csv.reader(query["output"].splitlines(), skipinitialspace=True))
        sample = dict(zip(fields.split(","), (v.strip() for v in values)))
        if int(sample["index"]) != gpu or sample["uuid"] != uuid:
            raise ValueError("matmul GPU identity mismatch")
        return sample
    result = dict(gpu=gpu, uuid=uuid, **MATMUL, **measure(torch, clocks),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda)
    artifacts.write_json(output, result)


def run(args, env, manifest):
    root = args.out / "preflight"
    root.mkdir()
    report = manifest["matmul_preflight"] = dict(state="running", **MATMUL, gpus=[],
        reported_clock_locks=REPORTED_CLOCK_LOCKS, clock_sample_timing="after timed submissions; may finish after GPU execution")
    manager = supervisor.Processes(root)
    monitor = machine.ComputeMonitor(manager, args.mps_owner)
    try:
        info = machine.require_idle(root, args.mps_owner, args.ignore_util_gpus)
        monitor.start()
        for gpu in range(8):
            uuid = info["gpu_uuids"][gpu]
            output = root / f"matmul_gpu{gpu}.json"
            manager.run([args.python, "-B", str(Path(__file__).resolve()), "--gpu", str(gpu),
                         "--uuid", uuid, "--output", str(output)], f"matmul_gpu{gpu}",
                        {**env, "CUDA_VISIBLE_DEVICES": uuid}, timeout=120)
            result = artifacts.read_json(output)
            if result["gpu"] != gpu or result["uuid"] != uuid:
                raise ValueError("matmul result does not match assigned GPU")
            report["gpus"].append(result)
            artifacts.save_json(args.out / "manifest.json", manifest)
        report["gate"] = throughput_gate(report["gpus"], args.ignore_util_gpus)
        if not report["gate"]["passed"]:
            raise RuntimeError(f"ignored-util GPU matmul throughput below 95% of reference: {report['gate']}")
        # Our own short matmul can remain in NVML's recent-utilization window.
        # One fixed settling interval, never a retry/wait loop for a busy machine.
        time.sleep(2)
        machine.require_idle(root / "postflight", args.mps_owner, args.ignore_util_gpus)
        report["state"] = "passed"
    except BaseException as error:
        report.update(state="failed", error=str(error))
        raise
    finally:
        try:
            errors = manager.cleanup()
            monitor.close()
            if errors:
                raise RuntimeError(f"matmul owned-process cleanup failed: {errors}")
        except BaseException as error:
            report.update(state="failed", error=str(error))
            raise
        finally:
            artifacts.save_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(args.gpu, args.uuid, args.output)
