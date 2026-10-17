"""Per-launch local process ownership, MPS evidence, and stable GPU reclamation."""

import argparse
import json
import os
import signal
import socket
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

from common import (
    TOPOLOGY_PROBES,
    Journal,
    Reclaim,
    append_json,
    baseline_ranges,
    connect,
    failed_slots,
    require,
    write_json,
)
from safety import (
    Children,
    check_gpu_processes,
    gpu_snapshot,
    mps_snapshot,
    parent_death_signal,
    signal_registered,
)

HERE = Path(__file__).resolve().parent


class Supervisor:
    def __init__(self, config, bus, journal):
        self.config, self.bus, self.journal = config, bus, journal
        self.root = Path(config["output"])
        self.children = Children(config, self.root, journal)
        self.workers, self.killed = {}, {}
        self.foreign = {}
        self.state = "preflight"
        self.host_owner = None

    def transition(self, expected, target):
        require(self.state == expected, f"invalid supervisor transition {self.state} -> {target}")
        self.journal.event("supervisor_state", previous=self.state, current=target)
        self.state = target

    def check_alive(self):
        if self.host_owner is not None:
            require(self.host_owner[0].poll() is None, 'host owner exited before workers finished')
        for slot, (child, record) in self.workers.items():
            if slot not in self.killed:
                require(child.poll() is None, f"unexpected worker exit: slot={slot}, pid={child.pid}, rc={child.returncode}")

    def wait(self, keys, timeout=None):
        return self.bus.wait(keys, timeout=timeout, check=self.check_alive)

    @property
    def task_pids(self):
        return {record["pid"] for _, record in self.children.children}

    def inspect(self, name, expected=None):
        raw = self.root / f"{name}_raw.jsonl"
        snapshot = gpu_snapshot(raw)
        mps = mps_snapshot(raw, snapshot, self.config.get("mps_pipe"), self.config["shared_gpu_mode"])
        foreign = check_gpu_processes(snapshot, mps, expected, self.config["shared_gpu_mode"], self.task_pids)
        if name == "preflight":
            self.foreign = foreign
        new_foreign = {uuid: [p for p in processes if p["pid"] not in
                             {old["pid"] for old in self.foreign.get(uuid, [])}]
                       for uuid, processes in foreign.items()}
        write_json(self.root / f"{name}.json", {"gpu": snapshot, "mps": mps,
                   "shared_gpu_mode": self.config["shared_gpu_mode"], "foreign": foreign, "new_foreign": new_foreign})
        return snapshot, mps

    def start_workers(self, slots, epoch):
        isolation = self.root / f"absent_mps_pipe_{self.config['launch']}_{self.config['node']}"
        require(not isolation.exists(), "worker MPS isolation path must not exist")
        for slot in slots:
            row = self.config["topology"][slot]
            require(row["node"] == self.config["node"], "cannot spawn a remote node's worker")
            previous = self.workers.get(slot)
            if epoch == 2:
                require(self.state == "reclaimed" and slot in self.killed, "rejoin before reclaim")
                require(previous[0].poll() == -signal.SIGKILL, "target did not exit by SIGKILL")
            env = {"CUDA_VISIBLE_DEVICES": row["gpu_uuid"], "CUDA_MPS_PIPE_DIRECTORY": str(isolation),
                   "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0", "TORCH_NCCL_ENABLE_MONITORING": "0",
                   "TORCH_NCCL_BLOCKING_WAIT": "0", "TORCH_NCCL_DESYNC_DEBUG": "0",
                   "NCCL_DEBUG": "INFO", "NCCL_DEBUG_FILE": str(self.root / "nccl_%h_%p.log"),
                   "TORCH_FR_DUMP_TEMP_FILE": str(self.root / f"flight_{slot}_"),
                   "TORCH_NCCL_TRACE_BUFFER_SIZE": "10000"}
            child, record = self.children.spawn(
                [sys.executable, str(HERE / ("topology_worker.py" if self.config.get("probe") in
                                            TOPOLOGY_PROBES.values() else "worker.py")),
                 "--config", str(self.root / "worker_config.json"),
                 "--slot", str(slot), "--epoch", str(epoch)],
                "worker", row["gpu_uuid"], env,
                replacement_for=previous[0].pid if previous else None)
            record.update(slot=slot, initial_epoch=epoch)
            if previous:
                require(child.pid != previous[0].pid, "replacement reused old PID; case cannot prove new PID")
                previous[1]["replacement_pid"] = child.pid
            self.workers[slot] = child, record
            self.killed.pop(slot, None)

    def verify_contexts(self, slots, epoch):
        records = self.wait([f"cuda/{epoch}/{s}" for s in slots])
        for slot, report in zip(slots, records):
            child, record = self.workers[slot]
            require(report["pid"] == child.pid and report["gpu_uuid"] == record["gpu_uuid"],
                    "worker registration identity mismatch")
        expected = {child.pid: record["gpu_uuid"] for child, record in self.workers.values()
                    if child.poll() is None}
        self.inspect(f"mps_after_cuda_epoch{epoch}", expected)
        self.bus.put(f"verified/{epoch}/{self.config['node']}", self.journal.event("mps_verified", epoch=epoch))

    def kill_and_reclaim(self, baselines, allowed):
        victims = failed_slots(self.config["topology"])
        require(all(self.config["topology"][s]["node"] == self.config["node"] for s in victims),
                "fault targets are not local")
        snapshot = gpu_snapshot(self.root / "kill_gpu_raw.jsonl")
        contexts = {p["pid"]: g["uuid"] for g in snapshot["gpus"] for p in g["processes"]}
        self.transition("running", "killing")
        signals = {}
        for slot in victims:
            child, record = self.workers[slot]
            require(child.poll() is None, "fault target already exited")
            signals[slot] = self.journal.event("kill_sent", slot=slot, target_pid=child.pid,
                                              gpu_uuid=record["gpu_uuid"])
            require(signal_registered(record, signal.SIGKILL, contexts), "fault target exited before signal")
            self.killed[slot] = True
        self.bus.put("kill_sent", signals)
        for slot in victims:
            child, record = self.workers[slot]
            require(child.wait(timeout=15) == -signal.SIGKILL, "fault target exit was not SIGKILL")
            self.children.exited(child, record, "injected_SIGKILL")
        self.bus.put("kill_exited", [self.workers[s][1] for s in victims])
        self.transition("killing", "reclaiming")
        selected = {self.config["topology"][s]["gpu_uuid"]: baselines[self.config["topology"][s]["gpu_uuid"]]
                    for s in victims}
        tracker = Reclaim(selected, allowed, self.config, self.task_pids)
        deadline = time.monotonic() + self.config["reclaim_timeout_s"]
        while time.monotonic() < deadline:
            self.check_alive()
            sample = gpu_snapshot(self.root / "reclaim_gpu_raw.jsonl")
            done = tracker.update(sample["gpus"], True, sample["end_local"])
            append_json(self.root / "gpu_reclaim_samples.jsonl", {**sample,
                        "t": sample["end_local"] + self.journal.offset,
                        "clock_error_s": self.journal.error, "streak": tracker.streak,
                        "confirmed_local": tracker.confirmed, "task_usage": tracker.task_usage})
            if done:
                details = {}
                for slot in victims:
                    uuid = self.config["topology"][slot]["gpu_uuid"]
                    details[uuid] = {"baseline": selected[uuid], "kill": signals[slot],
                                     "confirmed_t_local": tracker.confirmed[uuid],
                                     "confirmed_t": tracker.confirmed[uuid] + self.journal.offset,
                                     "gpu_reclaim_s": tracker.confirmed[uuid] - signals[slot]["t_local"]}
                result = {"cards": details, "t_done": max(tracker.confirmed.values()) + self.journal.offset,
                          "clock_error_s": self.journal.error, "shared_gpu_mode": self.config["shared_gpu_mode"],
                          "task_pids": sorted(self.task_pids), "task_usage": tracker.task_usage}
                write_json(self.root / "gpu_reclaim.json", result)
                self.bus.put("reclaimed", result)
                self.transition("reclaiming", "reclaimed")
                return
            time.sleep(self.config["sample_s"])
        raise TimeoutError("GPU memory/process reclamation did not become stable")

    def rejoin_preflight(self, baselines):
        victims = failed_slots(self.config["topology"])
        expected = {p.pid: r["gpu_uuid"] for s, (p, r) in self.workers.items() if s not in victims}
        before_spawn, _ = self.inspect("rejoin_pre_spawn", expected)
        for gpu in before_spawn["gpus"]:
            if not self.config["shared_gpu_mode"] and gpu["uuid"] in {self.config["topology"][s]["gpu_uuid"] for s in victims}:
                require(gpu["used_mib"] <= baselines[gpu["uuid"]]["threshold_mib"],
                        "reclaimed GPU memory rose before spawn")

    def run(self):
        node = self.config["node"]
        # Controller midpoint clock calibration, all on CPU TCPStore.
        for index in range(5):
            self.bus.wait([f"ping/{node}/{index}"], timeout=self.config["startup_timeout_s"])
            self.bus.put(f"pong/{node}/{index}", time.monotonic())
        calibration = self.bus.wait([f"clock/{node}"])[0]
        self.journal.offset, self.journal.error = calibration["offset"], calibration["error"]
        self.config["clock"] = calibration
        snapshot, mps = self.inspect("preflight")
        samples = [snapshot]
        for _ in range(self.config["stable_samples"] - 1):
            time.sleep(self.config["sample_s"])
            sample = gpu_snapshot(self.root / "baseline_raw.jsonl")
            check_gpu_processes(sample, mps, shared_gpu_mode=self.config["shared_gpu_mode"])
            samples.append(sample)
        baselines = baseline_ranges(samples, self.config)
        write_json(self.root / "baseline.json", {"samples": samples, "ranges": baselines})
        self.bus.put(f"preflight/{node}", {"gpus": snapshot["gpus"], "host": socket.gethostname(),
                                          "pid": os.getpid(), "config": self.config})
        self.config["topology"] = self.wait(["topology"])[0]
        write_json(self.root / "worker_config.json", self.config)
        if self.config.get('probe') == TOPOLOGY_PROBES[3]:
            self.host_owner = self.children.spawn(
                [sys.executable, str(HERE / 'host_owner.py'), '--config', str(self.root / 'worker_config.json')],
                'host_owner')
        slots = [r["slot"] for r in self.config["topology"] if r["node"] == node]
        self.wait(["start"])
        self.transition("preflight", "running")
        self.start_workers(slots, 0)
        self.verify_contexts(slots, 0)
        if not self.config.get('profile_only'):
            if node == "SL3061":
                self.wait(["kill"], timeout=self.config["launch_timeout_s"])
                self.kill_and_reclaim(baselines, mps["servers"])
            self.wait(["spawn"], timeout=self.config["launch_timeout_s"])
            if node == "SL3061":
                victims = failed_slots(self.config["topology"])
                self.rejoin_preflight(baselines)
                self.start_workers(victims, 2)
                self.transition("reclaimed", "rejoined")
                self.verify_contexts(victims, 2)
            else:
                expected = {p.pid: r["gpu_uuid"] for p, r in self.workers.values()}
                self.inspect("mps_after_cuda_epoch2", expected)
                self.bus.put(f"verified/2/{node}", self.journal.event("mps_verified", epoch=2))
            self.bus.put(f"spawned/{node}", [r for _, r in self.workers.values()])
        self.wait(["finish"], timeout=self.config["launch_timeout_s"])
        for child, record in self.workers.values():
            require(child.wait(timeout=self.config["stage_timeout_s"]) == 0,
                    f"worker failed at exit: {child.pid}")
            self.children.exited(child, record, "normal")
        if self.host_owner is not None:
            self.bus.put(f'host_stop/{node}', True)
            child, record = self.host_owner
            require(child.wait(timeout=30) == 0, 'host owner cleanup failed')
            self.children.exited(child, record, 'normal')
            self.host_owner = None
        # Cleanup evidence covers all eight GPUs, including the returned workers.
        tracker = Reclaim(baselines, mps["servers"], self.config, self.task_pids)
        deadline = time.monotonic() + self.config["reclaim_timeout_s"]
        while time.monotonic() < deadline:
            sample = gpu_snapshot(self.root / "cleanup_gpu_raw.jsonl")
            done = tracker.update(sample["gpus"], True, sample["end_local"])
            append_json(self.root / "cleanup_gpu_samples.jsonl", {**sample, "task_usage": tracker.task_usage,
                        "streak": tracker.streak, "confirmed_local": tracker.confirmed})
            if done:
                self.inspect("final_mps")
                self.bus.put(f"clean/{node}", self.journal.event("cleanup_verified"))
                return
            time.sleep(self.config["sample_s"])
        raise TimeoutError("final worker/GPU cleanup unconfirmed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=("store", "supervisor"), default="supervisor")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    parent_death_signal()
    root = Path(config["output"])
    journal = Journal(root / f"{args.role}.jsonl", config["node"], args.role)
    journal.event("process_start")
    if args.role == "store":
        import torch.distributed as dist

        store = dist.TCPStore(config["master"], config["port"], world_size=None, is_master=True,
                              timeout=timedelta(seconds=10), wait_for_workers=False)
        journal.event("store_listening", port=config["port"])
        deadline = time.monotonic() + config["launch_timeout_s"]
        while time.monotonic() < deadline and not store.check(["store_stop"]):
            time.sleep(0.1)
        journal.event("process_exit")
        return
    _, bus = connect(config)
    if config.get("probe") in TOPOLOGY_PROBES.values() and config["mode"] == "emulate":
        from topology_probe import EmulateSupervisor

        supervisor = EmulateSupervisor(config, bus, journal)
    else:
        supervisor = Supervisor(config, bus, journal)
    try:
        supervisor.run()
    except Exception:
        error = {"node": config["node"], "role": "supervisor", "state": supervisor.state,
                 "traceback": traceback.format_exc()}
        write_json(root / "supervisor_error.json", error)
        bus.put("failure", error)
        raise
    finally:
        try:
            supervisor.children.cleanup()
        finally:
            write_json(root / "process_lifecycle.json", [r for _, r in supervisor.children.children])
            journal.event("process_exit")


if __name__ == "__main__":
    main()
