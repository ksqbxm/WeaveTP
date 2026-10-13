"""Quick mode's fixed schedule, file handshake and standard per-measurement exports."""

import copy
import time

from . import artifacts

COUNTS = dict(L3=7, L2=3, L1=1, L0=0)
LINKS = "0:2,1:3,4:6,5:7"


def schedule(k, order):
    if k < 1 or order not in ("asc", "desc"):
        raise ValueError("quick requires K >= 1 and order asc|desc")
    levels = list(COUNTS) if order == "desc" else list(reversed(COUNTS))
    return [dict(level=level, repeat=repeat + 1, switch=2 + 2 * (i * k + repeat),
                 change_level=repeat == 0)
            for i, level in enumerate(levels) for repeat in range(k)]


def case_name(method, row):
    return f"{method}_{row['level']}_R{row['repeat']}"


def wait_json(path, timeout, check=lambda: None):
    deadline = time.monotonic() + timeout
    while not path.exists():
        check()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"quick handshake timed out: {path}")
        time.sleep(.2)
    check()
    return artifacts.read_json(path)


def admission(request):
    rows = request["ranks"]
    if len(rows) != 8 or {r["rank"] for r in rows} != set(range(8)) or len({r["gpu_uuid"] for r in rows}) != 8:
        raise ValueError("quick admission needs eight rank/GPU memory samples")
    keys = ("allocated_peak_bytes", "context_peak_bytes", "reserved_bytes", "context_bytes")
    if any(not isinstance(r[k], int) or r[k] < 0 for r in rows for k in keys):
        raise ValueError("invalid quick memory sample")
    a, c = (max(r[k] for r in rows) for k in keys[:2])
    return (a + c) / 1024**3 + 1.5, {r["gpu_uuid"]: r["reserved_bytes"] + r["context_bytes"] for r in rows}


def acknowledge(path, request, expected):
    if request["switch"] != expected["switch"] or request["level"] != expected["level"]:
        raise ValueError("quick handshake order/index mismatch")
    artifacts.save_json(path, dict(level=request["level"], switch=request["switch"], go=True))


def project(raw, result, row, launch_id):
    """Keep original clocks/steps; explicitly map the selected expansion to cycle index 2.

    Every sample shares the process's ONE initial warmup cycle. It is not an
    independent GPU launch. Complete process artifacts retain every transition.
    """
    index = row["switch"]
    mapping = {0: 0, 1: 1, index: 2}
    metadata = dict(process_launch_id=launch_id, level=row["level"], repeat=row["repeat"],
                    original_switch=index, switch_mapping=mapping, shared_warmup=True)
    trace = copy.deepcopy(raw)
    trace.update(measured_switch=2, quick=metadata)
    for field in ("decodes", "transfers", "waves", "milestones", "state_checks", "plan_records"):
        trace[field] = [dict(r, original_switch=r["switch"], switch=mapping[r["switch"]])
                        for r in trace[field] if r["switch"] in mapping]
    for state in trace["state_checks"]:
        state["measured"] = state["switch"] == 2
    # Memory remains the full process record: all stages, pressure levels, resets.
    trace["memory"]["quick"] = dict(metadata, scope="entire process, original switch indices")
    selected = copy.deepcopy(result)
    selected["switches"] = [dict(r, original_index=r["index"], index=mapping[r["index"]])
                            for r in result["switches"] if r["index"] in mapping]
    selected["quick"] = metadata
    selected["quick"]["other_result_metadata_scope"] = "process-wide; switches below are the selected warmup/measurement projection"
    selected["final_active_tp"] = 4
    selected["nostop_diagnostic"]["measured_switch"] = 2
    if "standby_weight_storage" in selected:
        for rank in selected["standby_weight_storage"]["ranks"]:
            rank["switches"] = [dict(r, original_index=r["index"], index=mapping[r["index"]])
                                for r in rank["switches"] if r["index"] in mapping]
    return trace, selected
