"""SL3060 default-allocator recheck; --dry-run does not access CUDA or create files."""

import argparse
import os
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[2] / "code/current"
ACCEPTANCE = Path("/data/ubuntu/lxh/weavetp/acceptance")


def commands(root):
    # Use the existing acceptance matrix as the single source of model settings.
    output = subprocess.check_output(
        ["bash", "tools/resharding/run_standby_weight_acceptance.sh", "benchmark", str(root), "--dry-run"],
        cwd=CODE, text=True, env={**os.environ, "PYTHON": sys.executable},
    )
    originals = {}
    for line in output.splitlines():
        command = shlex.split(line)
        name = next(x.split("=", 1)[1].rsplit("/", 1)[1] for x in command if x.startswith("OUT_DIR="))
        originals[name] = command
    cases = []
    for name, source, bitwise, pending in (
        ("default_synthetic_off", "default_synthetic_off", 0, 0),
        ("default_deepseek_off", "default_deepseek_off", 0, 0),
        ("default_synthetic_on", "default_synthetic_on", 1, 0),
        ("default_deepseek_on", "default_deepseek_on", 1, 0),
        ("default_deepseek_on_bitwise_r2", "default_deepseek_on", 1, 0),
        ("default_deepseek_on_bitwise_r3", "default_deepseek_on", 1, 0),
        ("default_pending_check", "default_synthetic_on", 0, 1),
    ):
        replacements = {"OUT_DIR": str(root / name), "WEIGHT_BITWISE_AUDIT": str(bitwise),
                        "WEIGHT_CHECK_AUDIT": str(pending)}
        command = [f"{key}={replacements[key]}" if (key := word.split("=", 1)[0]) in replacements else word
                   for word in originals[source]]
        cases.append((name, command))
    cases.append(("default_ordering", [
        "env", "-u", "PYTORCH_ALLOC_CONF", "-u", "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_VISIBLE_DEVICES=0,1", "CUDA_DEVICE_MAX_CONNECTIONS=8", sys.executable,
        "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        "tools/resharding/correctness/gpu_weight_ordering.py", "--output", str(root / "default_ordering"),
    ]))
    return cases


def check_idle(xml, control):
    """Only idle MPS servers with a verified empty client list may remain."""
    gpus = ET.fromstring(xml).findall("gpu")
    if len(gpus) != 8:
        raise ValueError("expected GPU 0-7")
    servers = set()
    for gpu in gpus:
        if gpu.findtext("utilization/gpu_util").strip() != "0 %":
            raise ValueError(f"GPU {gpu.attrib['id']} is busy")
        for process in gpu.findall("processes/process_info"):
            pid, name = process.findtext("pid"), process.findtext("process_name")
            if Path(name).name != "nvidia-cuda-mps-server":
                raise ValueError(f"disallowed GPU process: pid={pid} name={name}")
            servers.add(pid)
    if servers:
        known = set(control("get_server_list").split())
        if not servers <= known:
            raise ValueError(f"cannot inspect MPS servers {sorted(servers - known)} with current control socket")
        for pid in sorted(servers):
            clients = control(f"get_client_list {pid}").strip()
            if clients:
                raise ValueError(f"MPS server {pid} has clients or returned an error: {clients}")


def run_cases(root, cases):
    statuses = []
    for name, command in cases:
        out = root / name
        out.mkdir()
        (out / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        with (out / "console.log").open("w", encoding="utf-8") as log:
            status = subprocess.run(command, cwd=CODE, stdout=log, stderr=subprocess.STDOUT).returncode
        (out / "exit_code.txt").write_text(f"{status}\n", encoding="utf-8")
        statuses.append((name, status))
        print(f"{name}: exit={status} ({out})", flush=True)
    summary = "case\texit_code\n" + "".join(f"{name}\t{status}\n" for name, status in statuses)
    (root / "summary.tsv").write_text(summary, encoding="utf-8")
    print(summary)
    print(shlex.join(["tar", "czf", str(root) + ".tar.gz", "-C", str(root.parent), root.name]))
    return int(any(status for _, status in statuses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = ACCEPTANCE / f"standby_fix2_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}"
    cases = commands(root)
    if args.dry_run:
        print("\n".join(shlex.join(command) for _, command in cases))
        return 0
    root.mkdir()  # Refuse to reuse an existing result directory.
    (root / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE, text=True))
    gpus = subprocess.check_output(["nvidia-smi"], text=True)
    (root / "gpus.txt").write_text(gpus)
    xml = subprocess.check_output(["nvidia-smi", "-i", "0,1,2,3,4,5,6,7", "-q", "-x"], text=True)
    (root / "gpus.xml").write_text(xml)
    def control(command):
        result = subprocess.check_output(["nvidia-cuda-mps-control"], input=command + "\n",
                                         text=True, stderr=subprocess.STDOUT, timeout=15)
        with (root / "mps.txt").open("a") as log:
            log.write(f"{command}\n{result}\n")
        return result
    try:
        check_idle(xml, control)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(gpus)
        print(f"GPU preflight rejected: {error}", file=sys.stderr)
        return 2
    runtime = subprocess.check_output([sys.executable, "-c",
        "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.nccl.version())"], text=True)
    (root / "runtime.txt").write_text(runtime)
    print(f"Output: {root}", flush=True)
    return run_cases(root, cases)


if __name__ == "__main__":
    sys.exit(main())
