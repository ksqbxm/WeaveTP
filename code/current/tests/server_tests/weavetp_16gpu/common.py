"""Small helpers for explicit server acceptance commands, never pytest collection."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/resharding"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def caches(out):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TORCH_HOME",
                "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
                "TRANSFORMERS_CACHE", "PIP_CACHE_DIR"):
        path = out / key.lower()
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    return env


def command(argv, out, name, *, cwd=None, env=None, timeout=120):
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                            text=True, encoding="utf-8", timeout=timeout)
    write(out / f"{name}.json", {"command": argv, "exit_code": result.returncode,
                               "stdout": result.stdout, "stderr": result.stderr})
    require(result.returncode == 0, f"{name} failed; inspect {name}.json")
    return result.stdout


def run(args, check):
    out = Path(args.output_dir).resolve()
    require(sys.platform == "linux", "server test requires Linux; not verified on this host")
    require(out.is_relative_to("/data"), "all new test evidence must be under /data")
    protected = [Path("/data/models")]
    protected += [Path(value).resolve() for key in ("batch", "batch_root", "checkpoint", "peer_receipts")
                  if (value := getattr(args, key, None))]
    require(not any(out.is_relative_to(path) for path in protected), "output must be outside read-only inputs")
    out.mkdir(parents=True, exist_ok=False)
    env = caches(out)
    try:
        details = check(args, out, env)
        status = "blocked" if details.get("pending") else "passed"
        result = {"status": status, "test": Path(sys.argv[0]).name, "details": details}
    except Exception as exc:
        result = {"status": "failed", "test": Path(sys.argv[0]).name,
                  "error": f"{type(exc).__name__}: {exc}"}
    write(out / "acceptance.json", result)
    print(json.dumps({"status": result["status"], "evidence": str(out)}, ensure_ascii=False))
    return {"passed": 0, "failed": 1, "blocked": 2}[result["status"]]
