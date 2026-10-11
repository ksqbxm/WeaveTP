"""Validate frozen inputs and persist the immutable formal experiment plan."""

import hashlib
import json
import math
import os
import re
import shlex
from pathlib import Path

if __package__:
    from . import artifacts, common, machine
else:
    import artifacts
    import common
    import machine


FIXED = dict(SWITCHES="3", MAX_WAVES="64", SEQ_LENGTH="5248", MAX_POSITION_EMBEDDINGS="5248",
             MICRO_BATCH_SIZE="4", CYCLE_KV_TOKENS="2048")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen_config(path):
    raw = Path(path).read_bytes()
    values = {}
    for line in raw.decode("utf-8").splitlines():
        words = shlex.split(line, comments=True)
        if words[:1] == ["export"]:
            words = words[1:]
        if not words:
            continue
        if len(words) != 1 or not re.fullmatch(r"[A-Z][A-Z0-9_]*=.*", words[0]):
            raise ValueError("frozen.env accepts literal KEY=value assignments only, never shell code")
        key, value = words[0].split("=", 1)
        if key in values or any(c in value for c in ("$", "`", "\n")):
            raise ValueError(f"duplicate key or shell expansion in frozen.env: {key}")
        values[key] = value
    for key, expected in FIXED.items():
        if values.get(key, expected) != expected:
            raise ValueError(f"frozen {key} must remain {expected}")
    links = common.parse_links(values["STRESS_LINKS"])
    counts = {f"L{i}": int(values.get(f"L{i}_PROCESSES_PER_PAIR", default)) for i, default in enumerate((0, 1, 3, 7))}
    pressure = dict(links=",".join(f"{a}:{b}" for a, b in links), counts=counts,
                    stress_bytes=int(values["STRESS_BYTES"]), min_free_gib=float(values["MIN_FREE_GIB"]),
                    stress_sync_every=int(values.get("STRESS_SYNC_EVERY", "16")))
    if (counts["L0"] != 0 or min(counts.values()) < 0 or min(pressure["stress_bytes"], pressure["stress_sync_every"]) <= 0
            or not math.isfinite(pressure["min_free_gib"]) or pressure["min_free_gib"] <= 0):
        raise ValueError("invalid frozen pressure configuration (L0 must have zero stressors)")
    return dict(sha256=sha(raw), values=values, pressure=pressure,
                ready=values.get("DIAGNOSTICS_READY") == "1" and values.get("NOSTOP_COLD_MODE") == "cycle")


def matrix():
    rows = []
    for repeat in (1, 2, 3):
        for level in (range(3, -1, -1) if repeat == 2 else range(4)):
            methods = ("W2", "B0") if (repeat - 1 + level) % 2 == 0 else ("B0", "W2")
            for method in methods:
                rows.append(dict(name=f"R{repeat}_L{level}_{method}", block=f"R{repeat}_L{level}",
                                 repeat=repeat, level=f"L{level}", method=method, order=len(rows)))
    return rows


def prepare_manifest(root, frozen_path, frozen, *, checkpoint, mps_owner, python, timeout):
    values = frozen["values"]
    evidence = Path(values["DIAGNOSTIC_EVIDENCE"])
    source = artifacts.read_json(evidence / "manifest.json")
    machine_info = artifacts.read_json(evidence / "machine.json")
    profile = Path(values["PROFILE"]).read_bytes()
    routes = (evidence / "probe_routes.json").read_bytes()
    if sha(profile) != source["profile_sha256"]:
        raise ValueError("profile differs from diagnostic evidence")
    route_data = json.loads(routes)
    pairs = {(r["src"], r["dst"]) for r in route_data["default"] + route_data["candidate"] if r["src"] != r["dst"]}
    if (not pairs or pairs != {(r["src"], r["dst"]) for r in route_data["links"]}
            or any(not (0 <= s < 8 and 0 <= t < 8) for s, t in pairs)):
        raise ValueError("probe routes must be the fixed diagnostic migration-link union")
    checkpoint = checkpoint or source["checkpoint"]
    if checkpoint != source["checkpoint"] or not Path(checkpoint).is_dir():
        raise ValueError("checkpoint must be the same existing directory used in diagnostics")
    pressure = frozen["pressure"]
    manifest = dict(schema_version=1, protocol="nostop_formal_v1", frozen_sha256=frozen["sha256"],
                    frozen_values=values, checkpoint=checkpoint, profile_sha256=sha(profile), routes_sha256=sha(routes),
                    diagnostic_machine=dict(host=machine_info["host"], gpu_uuids=sorted(machine_info["memory_used_mib"])),
                    mps_owner=mps_owner, python=python, timeout_s=timeout,
                    environment={k: os.environ.get(k) for k in machine.ENV_NAMES if k != "MPS_OWNER"},
                    stress_links=pressure["links"], l3_processes=pressure["counts"]["L3"],
                    processes_per_pair=pressure["counts"], stress_bytes=pressure["stress_bytes"],
                    stress_sync_every=pressure["stress_sync_every"], min_free_gib=pressure["min_free_gib"],
                    order=matrix(), cases=[[r["name"], r["level"], r["method"], "cycle"] for r in matrix()],
                    drift_flag_threshold=.20, measured_switch=2)
    if (root / "manifest.json").exists():
        if artifacts.read_json(root / "manifest.json") != manifest:
            raise ValueError("resume inputs/order/environment differ from the immutable manifest")
        for name, digest in (("frozen.env", frozen["sha256"]), ("bandwidth_profile.json", sha(profile)), ("probe_routes.json", sha(routes))):
            if sha((root / name).read_bytes()) != digest:
                raise ValueError(f"saved input changed: {name}")
    else:
        frozen_bytes = frozen_path.read_bytes()
        if sha(frozen_bytes) != frozen["sha256"]:
            raise ValueError("frozen.env changed while preparing the manifest")
        artifacts.save_json(root / "manifest.json", manifest)
        (root / "frozen.env").write_bytes(frozen_bytes)
        (root / "bandwidth_profile.json").write_bytes(profile)
        (root / "probe_routes.json").write_bytes(routes)
    return manifest
