"""Independent control plane for one launch; node 0 never spawns node 1 workers."""

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

from common import (
    NODES,
    Journal,
    connect,
    failed_slots,
    join_timing,
    membership,
    prefix,
    rank_map,
    require,
    topology,
    write_json,
)
from safety import Children, parent_death_signal

HERE = Path(__file__).resolve().parent


def validate_epoch(rows, epoch, reports, initial, returned):
    members = membership(rows, epoch)
    require(len(reports) == len(members), "missing worker reports")
    identities = [[s, (returned if s in returned else initial)[s]["pid"]] for s in members]
    for slot, report in zip(members, reports):
        expected = returned.get(slot, initial[slot])
        require(report["slot"] == slot and report["identities"] == identities,
                "actual PG membership/PIDs differ from registered workers")
        require(report["pid"] == expected["pid"] and report["tensor"] == expected["tensor"],
                "PID or resident tensor identity changed across epochs")
        require(report["world_size"] == len(members) and report["new_rank"] == members.index(slot),
                "non-contiguous or incorrect new world ranks")


def control(config, bus, journal, check):
    root = Path(config["output"])

    def wait(keys, timeout=None):
        return bus.wait(keys, timeout, check)

    def publish(key, **fields):
        event = journal.event(key, **fields)
        bus.put(key, event)
        return event["t"]

    calibrations = {}
    for node in NODES:
        candidates = []
        for index in range(5):
            sent = time.monotonic()
            bus.put(f"ping/{node}/{index}", sent)
            remote = wait([f"pong/{node}/{index}"], config["startup_timeout_s"])[0]
            received = time.monotonic()
            candidates.append({"offset": (sent + received) / 2 - remote,
                               "error": (received - sent) / 2})
        chosen = min(candidates, key=lambda row: row["error"])
        bus.put(f"clock/{node}", chosen)
        calibrations[node] = {"chosen": chosen, "samples": candidates,
                              "method": "TCPStore midpoint; constant monotonic offset within launch"}
    write_json(root / "clock_calibration.json", calibrations)
    preflight = wait([f"preflight/{n}" for n in NODES], config["startup_timeout_s"])
    require(len({r["host"] for r in preflight}) == 2, "two distinct physical nodes required")
    for node, report in zip(NODES, preflight):
        require(report["host"].split(".")[0].upper() == node, f"{node} is running on wrong host: {report['host']}")
        for key in ("commit", "launch", "kind", "port", "tensor_elements", "memory_slack_mib",
                    "stable_samples", "sample_s", "reclaim_timeout_s", "stage_timeout_s", "shared_gpu_mode", "repeats"):
            require(report["config"][key] == config[key], f"node configuration differs: {key}")
    rows = topology(dict(zip(NODES, [r["gpus"] for r in preflight])))
    victims = failed_slots(rows)
    print(json.dumps({"fault_coordinates": [rows[s] for s in victims]}, indent=2), flush=True)
    maps = {epoch: rank_map(rows, epoch) for epoch in range(3)}
    write_json(root / "rank_map.json", maps)
    bus.put("topology", rows)
    publish("start")
    wait([f"verified/0/{n}" for n in NODES])
    initial = {s: r for s, r in zip(range(16), wait([f"cuda/0/{s}" for s in range(16)]))}
    publish("cuda_verified/0")
    epoch0 = wait([f"ready/0/{s}" for s in range(16)])
    validate_epoch(rows, 0, epoch0, initial, {})
    publish("arm")
    armed = wait([f"armed/{s}" for s in range(16)])
    require(all(r["pending"] == (config["kind"] in ("inflight", "shrink") and r["slot"] not in victims)
                for r in armed), "in-flight fixture not established")
    t_fail = publish("kill", semantics="controller issues failure command to SL3061 supervisor")
    signals = wait(["kill_sent"])[0]
    # Signal submission precedes publication. Reclaim proceeds concurrently on SL3061.
    publish("transition/1", membership=membership(rows, 1), epoch=1)
    survivors = membership(rows, 1)
    killed = wait(["kill_exited"])[0]
    require(len(killed) == 4 and {r["slot"] for r in killed} == set(victims), "incomplete fault exit evidence")
    require(all(r["pid"] == initial[r["slot"]]["pid"] for r in killed), "fault exit PID mismatch")
    require(all(r["exit"]["returncode"] == -signal.SIGKILL for r in killed), "SIGKILL not confirmed")
    abort1 = wait([f"aborted/1/{s}" for s in survivors])
    build1 = publish("build/1", prefix=prefix(config["launch"], 1))
    epoch1 = wait([f"ready/1/{s}" for s in survivors])
    validate_epoch(rows, 1, epoch1, initial, {})
    ready1 = journal.event("world12_ready")["t"]
    wait([f"serving/{s}" for s in survivors])
    request = publish("rejoin_request", repair_wait_s=0, policy="toy probe: request immediately after world12 validation")
    reclaimed = wait(["reclaimed"], config["reclaim_timeout_s"])[0]
    require(set(reclaimed["cards"]) == {rows[s]["gpu_uuid"] for s in victims}, "incomplete four-GPU reclaim evidence")
    # Use controller receipt as the conservative barrier; never schedule from a remote estimate.
    reclaim_received = journal.event("reclaim_received", evidence=reclaimed)["t"]
    spawn = publish("spawn")
    wait([f"verified/2/{n}" for n in NODES], config["startup_timeout_s"])
    returned = {s: r for s, r in zip(victims, wait([f"cuda/2/{s}" for s in victims]))}
    require(all(returned[s]["pid"] != initial[s]["pid"] for s in victims), "returned worker PID did not change")
    spawned = wait([f"spawned/{n}" for n in NODES])
    replacements = [r for r in spawned[1] if r["slot"] in victims]
    require(len(replacements) == 4, "incomplete replacement process records")
    actual_spawn = min(r["start"]["t"] for r in replacements)
    for record in spawned[1]:
        if record["slot"] in victims:
            require(record["start"]["t_local"] >= max(v["confirmed_t_local"] for v in reclaimed["cards"].values()),
                    "spawn precedes local reclaim confirmation")
    publish("cuda_verified/2")
    # Each survivor must keep making progress during background process startup.
    progress = wait([f"service_after_spawn/{s}" for s in survivors])
    prepared = publish("rejoin_ready", returned_pids=[returned[s]["pid"] for s in victims])
    publish("transition/2", membership=membership(rows, 2), epoch=2)
    abort2 = wait([f"aborted/2/{s}" for s in survivors])
    build2 = publish("build/2", prefix=prefix(config["launch"], 2))
    epoch2 = wait([f"ready/2/{s}" for s in range(16)])
    validate_epoch(rows, 2, epoch2, initial, returned)
    ready2 = journal.event("world16_ready")["t"]
    publish("finish")
    wait([f"done/{s}" for s in range(16)])
    wait([f"clean/{n}" for n in NODES], config["reclaim_timeout_s"] + config["stage_timeout_s"])
    timing = join_timing(t_fail, reclaimed["t_done"], request, actual_spawn, ready2)
    timing["reclaim_receipt_delay_s"] = reclaim_received - reclaimed["t_done"]
    timing.update(t_rejoin_request=request, t_rejoin_start=actual_spawn,
                  t_spawn_command=spawn, t_rejoin_ready=prepared, t_world16_ready=ready2,
                  spawn_to_ready_s=max(r["t_local"] for r in returned.values()) - min(r["start"]["t_local"] for r in replacements),
                  rejoin_preparation_s=prepared - spawn,
                  spawn_control_delay_s=actual_spawn - spawn)
    metrics = {k: v for k, v in timing.items() if k.endswith("_s")}
    for name, aborts, built, ready in (("shrink", abort1, build1, ready1), ("rejoin", abort2, build2, ready2)):
        metrics[f"abort_{name}_s"] = max(r["end"]["t"] for r in aborts) - min(r["start"]["t"] for r in aborts)
        metrics[f"rebuild_and_validate_{name}_s"] = ready - built
    # Use actual last service progress before rejoin, not the world12-ready event.
    pauses = []
    service = {s: r["service"] for s, r in zip(survivors, abort2)}
    for s, report in zip(survivors, progress):
        require(report["pid"] == initial[s]["pid"] and service[s]["last"]["pid"] == initial[s]["pid"],
                "service progress PID mismatch")
        require(service[s]["last"]["step"] >= report["step"] > service[s]["first"]["step"],
                "survivor did not compute during replacement preparation")
    for before, after in ((epoch0, epoch1), ([service[s]["last"] for s in survivors], epoch2)):
        old = {r["slot"]: r for r in before}
        for report in after:
            if rows[report["slot"]]["replica"] == "EDP1":
                previous = old[report["slot"]]
                pauses.append({"slot": report["slot"], "epoch": report["epoch"],
                               "interval": [previous["t"], report["t"]],
                               "seconds": report["t_local"] - previous["t_local"]})
    metrics["edp1_shrink_pause_max_s"] = max(p["seconds"] for p in pauses if p["epoch"] == 1)
    metrics["edp1_rejoin_pause_s"] = max(p["seconds"] for p in pauses if p["epoch"] == 2)
    result = {"status": "passed", "kind": config["kind"], "launch": config["launch"],
              "shared_gpu_mode": config["shared_gpu_mode"],
              "commit": config["commit"], "metrics": metrics, "timing": timing,
              "clock_calibration": calibrations, "gpu_reclaim": reclaimed,
              "kill_signals": signals, "aborts": {1: abort1, 2: abort2},
              "epochs": {0: epoch0, 1: epoch1, 2: epoch2}, "edp1_probe_pauses": pauses,
              "preparation_service": service,
              "edp1_pause_semantics": "rejoin: last validated local service step to validated world16; shrink: world16 to world12 progress gap",
              "prefixes": ["inherited_by_shrink_group" if config["kind"] == "shrink" and e == 1
                           else prefix(config["launch"], e) for e in range(3)],
              "real_integration": "deferred_pending_two_phase_gpu_validation"}
    return result


def run(config):
    root = Path(config["output"])
    journal = Journal(root / "controller.jsonl", config["node"], "controller" if config["node"] == NODES[0] else "node_launcher")
    children = Children(config, root, journal)
    journal.event("process_start")
    bus = None
    result = {"status": "failed", "kind": config["kind"], "launch": config["launch"],
              "shared_gpu_mode": config["shared_gpu_mode"]}
    try:
        argv = [sys.executable, str(HERE / "supervisor.py"), "--config", str(root / "config.json")]
        _, bus = connect(config)
        supervisor, _ = children.spawn(argv, "supervisor")
        if config["node"] == NODES[0]:
            def check():
                for child, record in children.children:
                    require(child.poll() in (None, 0), f"{record['role']} exited {child.returncode}")

            if config.get("probe") == "step02_topology_rebind":
                from topology_probe import control as topology_control

                result = topology_control(config, bus, journal, check)
            else:
                result = control(config, bus, journal, check)
            require(supervisor.wait(timeout=30) == 0, "SL3060 supervisor exit failed")
        else:
            require(supervisor.wait(timeout=config["launch_timeout_s"] - 30) == 0, "SL3061 supervisor failed")
            result["status"] = "passed"
    except BaseException:
        result.update(status="failed", traceback=traceback.format_exc())
        if bus is not None:
            try:
                bus.put("failure", result)
            except Exception:
                pass
    finally:
        try:
            children.cleanup()
        except Exception:
            result.update(status="failed", cleanup_error=traceback.format_exc())
        write_json(root / "control_process_lifecycle.json", [r for _, r in children.children])
        write_json(root / "controller_result.json", result)
        journal.event("process_exit", status=result["status"])
    return 0 if result["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    parent_death_signal()

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
