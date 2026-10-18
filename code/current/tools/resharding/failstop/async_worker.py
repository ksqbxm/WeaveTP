"""Step 4 replica-local CUDA service and migration mechanism runner."""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from async_state import ReplicaDriver, Request, interference, interval_overlap
from common import Journal, connect, require, write_json
from safety import parent_death_signal, verify_worker_context

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.distributed as dist
from async_fixture import (
    Model,
    Transfer,
    check_peak,
    kv_reference,
    memory_sample,
    reference_logit,
    release,
    weight_reference,
)
from async_topology import AsyncSession as FaultSession
from topology_fixture import toy_local_orders
from torch.distributed import distributed_c10d as c10d

from megatron.core.failstop import (
    ControlEvent,
    FaultBackend,
    build_replica_pg_collections,
)


class Worker:
    def __init__(self, config, slot, initial_epoch):
        self.config, self.slot, self.initial_epoch = config, slot, initial_epoch
        self.root, self.row = Path(config["output"]), config["topology"][slot]
        self.store, self.bus = connect(config)
        self.journal = Journal(self.root / f"async_{slot}_{initial_epoch}.jsonl",
                               config["node"], "worker", **config["clock"])
        self.real, self.healthy = config["mode"] == "real", self.row["replica"] == "EDP1"
        self.session = FaultSession(config["topology"], config["launch"],
                                    FaultBackend(config["mode"], fail=lambda *_: None, rejoin=lambda *_: None),
                                    local_orders=toy_local_orders(config["topology"]))
        self.victim = slot in self.session.failed
        self.driver = ReplicaDriver(self.row["replica"])
        self.model = self.kv = None
        self.samples, self.retired, self.memory, self.pauses = [], [], [], []
        self.transfers, self.overlaps, self.compute_events = [], [], []
        self.switch_memory, self.migration_bytes = [], 0
        self.request = Request("mechanism-r0", 0, 0.0, (2, 3, 5, 7))
        self.elements, self.budget = config["weight_elements"], config["memory_budget_bytes"]
        self.last_end = None
        self.prebuilt = {}

    def initialize(self):
        parent_death_signal()
        require(str(torch.__version__).startswith("2.11.") and tuple(torch.cuda.nccl.version()[:2]) == (2, 28),
                "Step 4 requires the validated PyTorch 2.11 / NCCL 2.28 runtime")
        require(not Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"]).exists(), "MPS isolation path exists")
        torch.cuda.set_device(0)
        require(torch.cuda.device_count() == 1, "worker must see only its assigned GPU")
        require(str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-") ==
                self.row["gpu_uuid"].removeprefix("GPU-"), "GPU UUID mismatch")
        context_tensor = torch.empty(1, device="cuda")
        context = verify_worker_context(self.row["gpu_uuid"], self.root / f"context_{self.slot}_{self.initial_epoch}.jsonl")
        self.bus.put(f"cuda/{self.initial_epoch}/{self.slot}", self.journal.event("cuda_ready",
            slot=self.slot, gpu_uuid=self.row["gpu_uuid"], epoch=self.initial_epoch,
            empty_receiver=self.initial_epoch == 2, context_verification=context))
        self.bus.wait([f"cuda_verified/{self.initial_epoch}"])
        del context_tensor
        self.stream = torch.cuda.Stream()
        self.anchor = torch.cuda.Event(enable_timing=True)
        self.anchor.record()
        self.anchor.synchronize()
        if not self.real:
            dist.init_process_group("gloo", store=dist.PrefixStore(
                f"step04/{self.config['launch']}/control/", self.store),
                rank=self.slot, world_size=16, timeout=timedelta(seconds=self.config["stage_timeout_s"]))
            stage = FaultSession(self.config["topology"], self.config["launch"], self.session.backend,
                                 local_orders=toy_local_orders(self.config["topology"]))
            self.prebuilt[0] = self.build(stage, 0)
            stage.fail(stage.failed)
            self.prebuilt[1] = self.build(stage, 1)
            stage.complete()
            stage.rejoin(stage.failed)
            self.prebuilt[2] = self.build(stage, 2)
            # A healthy emulate replica uses the same service communicator throughout.
            for phase in (1, 2):
                self.prebuilt[phase]["target"] = self.prebuilt[0]["target"] if self.healthy else self.prebuilt[phase]["target"]
        torch.cuda.reset_peak_memory_stats()

    def build(self, session, phase):
        if self.real:
            members = session.current_world()["membership"]
            dist.init_process_group("nccl", store=dist.PrefixStore(
                f"step04/{self.config['launch']}/epoch/{session.epoch}/", self.store),
                rank=members.index(self.slot), world_size=len(members),
                timeout=timedelta(seconds=self.config["stage_timeout_s"]), device_id=torch.device("cuda", 0))
        records = []
        groups = build_replica_pg_collections(session.replicas(), backend="nccl",
            timeout=timedelta(seconds=self.config["stage_timeout_s"]), records=records)
        data = dist.new_group(list(sorted(session.current_world()["ranks"])), backend="nccl",
                              timeout=timedelta(seconds=self.config["stage_timeout_s"]))
        service2 = None
        if phase == 2:
            # Recreated TP2 service groups in the new real world16; targets have TP4 groups.
            ranks = tuple(r["new_rank"] for r in session.rank_map()
                          if r["replica"] == "EDP0" and r["slot"] not in session.failed)
            service2 = build_replica_pg_collections([(ranks, 2, 2, 2)], backend="nccl",
                timeout=timedelta(seconds=self.config["stage_timeout_s"]))[0]
        return {"target": groups[1 if self.healthy else 0], "data": data, "records": records, "service2": service2}

    def adopt(self, phase):
        pack = self.build(self.session, phase) if self.real else self.prebuilt[phase]
        self.session.adopt_group_order(pack["records"])
        self.pack = pack
        if self.model is not None and (self.healthy or phase == 2):
            # Only a model that keeps serving is rebound; a shrink source is transport-only.
            pointer = self.model.weight.data_ptr()
            self.model.groups = pack["target"] if self.healthy else pack["service2"]
            require(self.model.weight.data_ptr() == pointer, "rebind changed survivor pointer")
            if not self.healthy:
                require(self.model.groups.local_ranks["expt_tp"] == self.model.tp,
                        "rejoin changed the live TP2 shard coordinate")
            self.journal.event("serving_model_rebound", pointer=pointer, phase=phase,
                               tp=self.model.tp, size=self.model.size)
        return pack

    def collective_flag(self, flag):
        value = torch.tensor(int(flag), dtype=torch.int64, device="cuda")
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.model.groups.tp_ep)
        return bool(value.item())

    def step(self, phase, commit=True, tokens=None):
        tokens = self.request.tokens if tokens is None else tuple(tokens)
        ticket = self.driver.begin_step(self.request) if commit else None
        cache = self.kv if commit else kv_reference(tokens, self.model.tp, self.model.size).to("cuda")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        submitted = time.monotonic()
        start.record()
        value = self.model(cache, self.row["ep"] == 0)
        end.record()
        actual = int(value.item())  # Only this replica's forward completion.
        completed = time.monotonic()
        require(actual == reference_logit(tokens, self.elements), "logits differ bitwise from independent CPU reference")
        if commit:
            self.driver.commit(ticket, self.request, actual % 127)
            # Append only a committed token; the frozen prefix storage remains immutable.
            tail = kv_reference((actual % 127,), self.model.tp, self.model.size).to("cuda")
            self.kv = torch.cat((self.kv, tail), dim=3)
            if phase == "expand" and not self.healthy:
                self.compute_events.append((start, end))
            if self.pauses and self.pauses[-1]["end_local"] is None:
                self.pauses[-1]["end_local"] = submitted  # service resumed on the rebuilt world
            self.last_end = completed
        sample = {"phase": phase, "context": len(tokens), "tokens": list(tokens), "layout": self.model.size,
                  "seconds": completed - submitted, "gpu_ms": start.elapsed_time(end),
                  "step": self.driver.step, "sequence": self.driver.collective_sequence,
                  "epoch": self.session.epoch, "submitted_local": submitted, "completed_local": completed,
                  "clock_error_s": self.config["clock"]["error"], "migration_bytes": self.migration_bytes}
        self.samples.append(sample)
        if commit:
            report = self.journal.event("service_commit", **sample, slot=self.slot, replica=self.row["replica"])
            self.bus.put(f"committed/{self.row['replica']}/{self.slot}", {
                "request": asdict(self.request), "step": self.driver.step,
                "sequence": self.driver.collective_sequence, "report": report})
        return sample

    def serve_until(self, key, phase, progress_key=None):
        deadline = time.monotonic() + self.config["stage_timeout_s"]
        while True:
            self.step(phase)
            if progress_key is not None:
                self.bus.put(progress_key, True)
            if self.collective_flag(self.bus.has(key)):
                return
            require(time.monotonic() < deadline, f"service timeout waiting for {key}")
            require(not self.bus.has("failure"), "peer failure during service")
            time.sleep(.005)

    def phase_report(self, round_id, phase):
        require(torch.equal(self.kv.cpu(), kv_reference(self.request.tokens, self.model.tp, self.model.size)),
                "steady KV differs bitwise from committed history")
        self.memory.append(memory_sample(self.model, self.kv, f"{round_id}/{phase}", self.budget, self.retired))
        require(self.memory[-1]["peak_allocated"] <= self.budget, "observed peak exceeds admission budget")
        report = self.journal.event("phase_ready", slot=self.slot, replica=self.row["replica"],
            phase=phase, round=round_id, rank_map=self.session.rank_map(), checks_passed=True,
            pointer=self.model.weight.data_ptr(), step=self.driver.step,
            sequence=self.driver.collective_sequence, history=list(self.request.tokens),
            empty_receiver=phase == 2 and self.victim, memory=self.memory[-1],
            rejoin_pause_s=self.pauses[-1]['end_local'] - self.pauses[-1]['start_local'] if self.pauses else None)
        self.bus.put(f"phase/{round_id}/{phase}/{self.slot}", report)

    def transition(self, phase, round_id):
        """Members of the old world stop, abort it (real) and move the session; replacements never enter."""
        message = self.bus.wait([f"transition/{round_id}/{phase}"])[0]
        if self.real:
            c10d._abort_process_group()
            require(not dist.is_initialized(), "old world still initialized after abort")
            if self.healthy:
                self.pauses.append({"phase": phase, "start_local": self.last_end,
                                    "reason": "abort/world rebuild", "end_local": None})
        if phase == 1:
            self.session.fail(self.session.failed, ControlEvent(**message))
            if not self.healthy:
                self.driver.fail(self.session.epoch)
        else:
            self.session.rejoin(self.session.failed, ControlEvent(**message))
            self.driver.epoch = self.session.epoch
        self.bus.put(f"aborted/{round_id}/{phase}/{self.slot}", True)
        key = f"build/{round_id}/{phase}"
        if not self.real and self.healthy:
            self.serve_until(key, "shrink" if phase == 1 else "expand")
        else:
            self.bus.wait([key])

    def report_transfer(self, transfer, phase):
        self.transfers.append({"phase": phase, "links": transfer.bytes, "intervals_ms": transfer.intervals,
            "batch_seconds": transfer.handle.elapsed_seconds(),
            "marker_bytes": transfer.service.last_launch_stats["participation_marker_bytes"]})

    def migrate(self, phase, round_id):
        pack, target_size = self.pack, 2 if phase == 1 else 4
        old_coords = {r["slot"]: r["etp"] for r in self.config["topology"] if r["replica"] == "EDP1"} if phase == 1 else {
            int(slot): coord for slot, coord in self.bus.wait([f"service2_coords/{round_id}"])[0].items()}
        snapshot = None
        source_kv = self.kv
        if phase == 2:
            leader = min(old_coords)
            if self.slot == leader:
                self.bus.put(f"snapshot/{round_id}", {"request": asdict(self.request), "step": self.driver.step,
                                                    "sequence": self.driver.collective_sequence})
            key = f"snapshot/{round_id}"
            if self.healthy:
                self.serve_until(key, "expand")
            record = self.bus.wait([key])[0]
            if not self.healthy:
                if self.victim:
                    self.request = Request.from_dict(record["request"])
                    self.driver.join(self.session.epoch)
                    self.driver.follow(record["step"], record["sequence"])
                require(self.request == Request.from_dict(record["request"]), "replica snapshot disagreement")
                snapshot = self.request.snapshot()
        target = target_kv = None
        if not self.healthy:
            budget = check_peak(self.model, self.kv, target_size, len(self.request.tokens) + 4,
                                self.elements, self.budget)
            self.switch_memory.append({"round": round_id, "phase": phase, "budget": budget})
            coord = self.session.rank_map()[self.slot]["etp"]
            with torch.cuda.stream(self.stream):
                target = Model(torch.full((self.elements // target_size,), float("nan"), device="cuda"),
                               pack["target"], coord, target_size)
                if phase == 2:
                    shape = kv_reference(self.request.tokens, coord, 4).shape
                    target_kv = torch.full(shape, -1, dtype=torch.int64, device="cuda")
        self.stream.wait_stream(torch.cuda.current_stream())
        transfer = Transfer(self.session, self.slot, pack["data"], self.anchor, self.stream)
        transfer.weights(self.model, target, phase, self.elements, old_coords)
        if phase == 2:
            prefix_length = len(record["request"]["prompt"]) + len(record["request"]["generated"])
            transfer.kv(source_kv, target_kv, old_coords, 0, prefix_length)
        transfer.launch()
        self.migration_bytes = sum(link["bytes"] for link in transfer.bytes)
        if not self.healthy:
            self.driver.start_transfer(transfer, target_layout=target_size, snapshot=snapshot)
        begin_event = len(self.compute_events)
        if phase == 2 and not self.healthy and not self.victim:
            deadline = time.monotonic() + self.config["stage_timeout_s"]
            steps = 0
            while True:
                self.step("expand")
                steps += 1
                if not self.collective_flag(steps < 4 or not transfer.done()):
                    break
                check_peak(self.model, self.kv, 4, len(self.request.tokens) + 1, self.elements, self.budget)
                require(time.monotonic() < deadline, "expansion transport timeout while serving TP2")
            if self.slot == min(old_coords):
                self.bus.put(f"delta/{round_id}", {"request": asdict(self.request), "step": self.driver.step,
                                                 "sequence": self.driver.collective_sequence})
        if self.healthy:
            # Continue through both EDP0 downtime and expansion.
            key = f"delta/{round_id}" if phase == 2 else f"destination_ready/{round_id}/1"
            self.serve_until(key, "expand" if phase == 2 else "shrink")
        transfer.wait()
        self.report_transfer(transfer, f"{round_id}/{phase}/prefix")
        delta = None
        if phase == 2:
            current = self.bus.wait([f"delta/{round_id}"])[0]
            if not self.healthy:
                if self.victim:
                    self.request = Request.from_dict(current["request"])
                    self.driver.follow(current["step"], current["sequence"])
                require(self.request == Request.from_dict(current["request"]), "final history disagreement")
                new_length = len(self.request.tokens)
                target_kv = torch.cat((target_kv, torch.full(
                    (*target_kv.shape[:3], new_length - prefix_length, target_kv.shape[4]),
                    -1, dtype=torch.int64, device="cuda")), dim=3)
            self.stream.wait_stream(torch.cuda.current_stream())
            delta = Transfer(self.session, self.slot, pack["data"], self.anchor, self.stream)
            delta.kv(self.kv, target_kv, old_coords, prefix_length,
                     len(current["request"]["prompt"]) + len(current["request"]["generated"]))
            delta.launch()
            delta.wait()
            self.report_transfer(delta, f"{round_id}/2/delta")
        if not self.healthy:
            if phase == 1:
                # Fault invalidates ALL EDP0 in-flight KV; full prefill rebuilds it from input only.
                self.kv = None
                target_kv = kv_reference(self.request.tokens, target.tp, target.size).to("cuda")
            def validate():
                require(torch.equal(target.weight.cpu(), weight_reference(
                    self.row["ep"], target.tp, target.size, self.elements)), "target weights differ bitwise")
                require(torch.equal(target_kv.cpu(), kv_reference(self.request.tokens, target.tp, target.size)),
                        "target KV prefix/delta differs bitwise")
                require(torch.cuda.max_memory_allocated() <= self.budget, "observed switch peak exceeds budget")
                self.switch_memory[-1].update(peak_allocated=torch.cuda.max_memory_allocated(),
                                             allocated_before_release=torch.cuda.memory_allocated(),
                                             reserved_before_release=torch.cuda.memory_reserved())
                self.bus.put(f"validated/{round_id}/{phase}/{self.slot}", True)
                self.bus.wait([f"validated/{round_id}/{phase}/{r['slot']}" for r in self.session.rank_map()
                               if r["replica"] == "EDP0" and r["active"]])
            def release_source():
                if self.model is not None:
                    retired, audit = release(self.model, [target.weight, target_kv],
                                              [transfer.service] + ([delta.service] if delta else []))
                    self.retired.extend(retired)
                    self.journal.event("source_released", phase=phase, audit=audit)
            self.driver.cutover(self.request, validate=validate, release=release_source, delta_handle=delta)
            self.model, self.kv = target, target_kv
            if phase == 1:
                self.bus.put(f"destination/{round_id}/1/{self.slot}", True)
            elif not self.victim:
                compute = [(self.anchor.elapsed_time(a), self.anchor.elapsed_time(b))
                           for a, b in self.compute_events[begin_event:]]
                overlap = interval_overlap(compute, transfer.intervals)
                self.overlaps.append({"round": round_id, "compute_ms": compute,
                                      "transfer_ms": transfer.intervals, "overlap_ms": overlap,
                                      "committed_before_completion_fence": steps})
                if not self.config["correctness_only"]:  # shared GPUs: the overlap is recorded, not asserted
                    require(overlap > 0, "no measured CUDA service/transfer overlap")
        source_kv = None
        self.migration_bytes = 0
        self.step("recovered" if phase == 2 else "degraded")
        self.phase_report(round_id, phase)
        self.session.complete()
        self.compute_events.clear()

    def run(self):
        self.initialize()
        if self.initial_epoch == 2:
            # A replacement has no old world to abort; it joins the session at the rejoin event.
            self.session.fail(self.session.failed)
            self.session.complete()
            self.session.rejoin(self.session.failed, ControlEvent(**self.bus.wait(["transition/0/2"])[0]))
            self.bus.wait(["build/0/2"])
            self.adopt(2)
            self.migrate(2, 0)
        else:
            for round_id in range(self.config["rounds"]):
                self.adopt(0)
                if self.model is None:
                    self.model = Model(weight_reference(self.row["ep"], self.row["etp"], 4, self.elements).to("cuda"),
                                       self.pack["target"], self.row["etp"], 4)
                    self.kv = kv_reference(self.request.tokens, self.model.tp, 4).to("cuda")
                for _ in range(3):
                    self.step("before")
                self.phase_report(round_id, 0)
                if self.healthy:
                    self.serve_until(f"transition/{round_id}/1", "before",
                                     f"healthy_progress/{round_id}/{self.slot}")
                elif self.victim:
                    if self.real:
                        self.bus.wait(["finish"])
                        raise RuntimeError("real victim survived SIGKILL")
                    self.bus.wait([f"emulate_release/{round_id}"])
                    retired, audit = release(self.model, [], [])
                    self.retired.extend(retired)
                    self.model = self.kv = None
                    self.bus.put(f"released/{round_id}/{self.slot}", {"src_none": True, "dst_none": True,
                                                                    "data_bytes": 0, "marker_bytes": 0})
                self.transition(1, round_id)
                if not self.victim:
                    self.adopt(1)
                    self.migrate(1, round_id)
                    self.serve_until(f"transition/{round_id}/2", "degraded",
                                     f"serving/{round_id}/{self.slot}")
                else:
                    self.session.complete()
                    self.bus.wait(["spawn" if round_id == 0 else f"emulate_rejoin/{round_id}"])
                    if round_id == 0:
                        self.bus.put(f"cuda/2/{self.slot}", self.journal.event("empty_target_ready",
                                     gpu_uuid=self.row["gpu_uuid"], empty_receiver=True))
                        self.bus.wait(["cuda_verified/2"])
                    else:
                        self.bus.put(f"empty/{round_id}/{self.slot}", True)
                self.transition(2, round_id)
                self.adopt(2)
                self.migrate(2, round_id)
                if round_id + 1 < self.config["rounds"]:
                    self.bus.wait([f"next/{round_id}"])
                    self.session.next_round()
        # Replay identical inputs/shapes for an uncontended baseline, with one model.
        self.bus.wait(["baseline"])
        if self.healthy:
            measured = [dict(s) for s in self.samples if s["phase"] in ("before", "shrink", "expand")]
            for sample in measured:
                self.step("baseline", commit=False, tokens=sample["tokens"])
            stats = interference(self.samples)
        else:
            stats = None
        report = {"slot": self.slot, "samples": self.samples, "interference": stats, "memory": self.memory,
                  "overlap": self.overlaps, "transfers": self.transfers, "pauses": self.pauses,
                  "switch_memory": self.switch_memory, "baseline_order": "post_fault_same_input_replay",
                  "prerequisites_verified": False, "transport_fixture": "explicit_slices",
                  "step3_integration": "separate acceptance required"}
        write_json(self.root / f"async_checks_{self.slot}.json", report)
        self.bus.put(f"baseline_done/{self.slot}", True)
        self.bus.wait(["finish"])
        if self.real:
            c10d._abort_process_group()
        else:
            dist.destroy_process_group()
        self.bus.put(f"done/{self.slot}", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--epoch", type=int, choices=(0, 2), required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    try:
        Worker(config, args.slot, args.epoch).run()
    except BaseException:
        error = {"slot": args.slot, "pid": os.getpid(), "traceback": traceback.format_exc()}
        write_json(Path(config["output"]) / f"worker_error_{args.slot}_{args.epoch}.json", error)
        try:
            _, bus = connect(config)
            bus.put("failure", error)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
