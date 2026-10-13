#!/usr/bin/env python3
"""Outer diagnostic entry point. No changes to W2's routing/adaptive thresholds."""

import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "examples/rl"))

from tools.resharding.nostop import synthetic_kv
from tools.resharding.nostop.common import (
    MAX_WAVES,
    WORKLOAD_ID,
    capacity_budget,
    intervals_overlap,
    kv_workload,
)
from tools.resharding.nostop.cuda_memory import StageMemory
from tools.resharding.nostop.validation import CutoverAudit, synthetic_logit_check


def clock_identity():
    info = time.get_clock_info("perf_counter")
    return {"host": socket.gethostname(), "implementation": info.implementation,
            "monotonic": info.monotonic, "adjustable": info.adjustable,
            "resolution_s": info.resolution,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "time_namespace": os.readlink("/proc/self/ns/time")}


class Diagnostic:
    def __init__(self, backend, args):
        self.b, self.args = backend, args
        self.torch, self.dist = backend.torch, backend.dist
        self.mode = os.environ.get("NOSTOP_COLD_MODE", "cycle")
        self.stall_pause = os.environ.get("STALL_PAUSE", "0") == "1"
        self.kv = int(os.environ.get("CYCLE_KV_TOKENS", "2048"))
        self.extra = int(os.environ.get("CYCLE_EXTRA_STEPS", "10"))
        expected = 3 if self.mode == "cycle" else 1
        if (self.mode not in ("cycle", "independent") or args.live_switches != expected
                or self.kv != 2048 or self.extra != 10 or not args.live_release_standby_weights
                or args.live_prompt_tokens != 8 or os.environ.get("NOSTOP_WORKLOAD") != WORKLOAD_ID
                or args.live_repeat_forward or args.live_weight_bitwise_audit):
            raise ValueError("diagnostic requires mode-matched switches, KV=2048, extra=10, release=1, audit=0")
        capacity = capacity_budget()["max_position_embeddings"]
        if args.live_max_waves != MAX_WAVES or args.max_position_embeddings != capacity or args.seq_length != capacity:
            raise ValueError(f"diagnostic requires MAX_WAVES={MAX_WAVES} and both sequence capacities={capacity}")
        self.rank = self.dist.get_rank()
        self.memory = StageMemory(self.torch, self.rank)
        self.prefill_count = 0
        self.prefix = None
        self.index, self.phase, self.wave, self.step = -1, "initial_validation", None, 0
        self.decodes, self.transfers, self.marks, self.waves, self.states = [], [], [], [], []
        self.plan_inputs = []
        self.first_target = set()
        self.origin = None
        self.routes = None
        self.audit = CutoverAudit(self)
        self.logit_metrics = []

    def is_measured(self, index):
        return index == self.args.live_switches - 1

    def mark(self, name):
        self.marks.append({"switch": self.index, "name": name, "host_ns": time.perf_counter_ns()})
        if name in ("base_done", "delta_done", "target_ready"):
            self.wave = None

    def check_wave_count(self, count):
        if count >= self.args.live_max_waves:
            raise RuntimeError("wave budget exhausted; refusing an oversized final wave")

    def wave_begin(self, wave, tasks):
        self.phase, self.wave = "base", wave
        self.waves.append({"switch": self.index, "wave": wave, "tasks": tasks,
                           "submit_host_ns": time.perf_counter_ns()})

    def wave_end(self):
        self.waves[-1]["commit_done_host_ns"] = time.perf_counter_ns()

    def capture_transfer(self, service):
        # Existing launch callback runs after _inflight is assigned and before
        # wait() clears its events. Keep references; never synchronize here.
        handle = service._inflight
        for stage, pairs in handle._timing_stages.items():
            stats = service.last_launch_stats
            model_payload = stage != "nccl" or bool(
                stats["original_remote_sends"] + stats["original_remote_recvs"])
            for start, end in pairs:
                self.transfers.append({"switch": self.index, "phase": self.phase,
                                       "wave": self.wave, "stage": stage, "rank": self.rank,
                                       "model_payload": model_payload, "events": (start, end)})

    def independent_warmup(self, services, weights, contexts, control_group, pairs):
        torch = self.torch
        before = [c.sequence_len_offset for c in contexts]
        checks = [sum(float(t[:c.sequence_len_offset].float().sum().item())
                      for values in c.key_value_memory_dict.values() for t in values) for c in contexts]
        weights[4]._check_storage(False)
        scratch_send = torch.zeros(64 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        scratch_recv = torch.empty_like(scratch_send)
        for name in ("2->4-baseline", "2->4-candidate"):
            service = services[name]
            service.producer_stream.wait_stream(torch.cuda.current_stream())
            for src, dst in pairs:
                for _ in range(2):
                    with torch.cuda.stream(service.producer_stream):
                        if self.rank == src:
                            service.submit_send(scratch_send, dst, task_id=0)
                        if self.rank == dst:
                            service.submit_recv(scratch_recv, src, task_id=0)
                        service.launch().wait()
                    self.dist.barrier(group=control_group)
        weights[4]._check_storage(False)
        after_checks = [sum(float(t[:c.sequence_len_offset].float().sum().item())
                            for values in c.key_value_memory_dict.values() for t in values) for c in contexts]
        if before != [c.sequence_len_offset for c in contexts] or checks != after_checks:
            raise AssertionError("scratch warmup changed KV state")
        return {"scratch_bytes": 2 * scratch_send.numel(), "peer_pairs": pairs,
                "kv_checksum_before": checks, "kv_checksum_after": after_checks,
                "target_weight_storage_bytes": 0, "passes_per_pair": 2}

    def before_switch(self, index, active_tp, src_model, src_context, dst_model, dst_context,
                      weights, services, control_group, default_plan, candidate_plan,
                      dst_bundle, token_counter):
        self.index, self.phase, self.wave = index, "prepare", None
        self.memory.begin(f"prepare_switch_{index}")
        if self.origin is None:
            if self.dist.get_world_size() != 8 or self.args.expert_model_parallel_size != 2:
                raise ValueError("diagnostic requires one host, eight ranks, EP=2")
            self.clock = clock_identity()
            self.clock_anchor = {"perf_counter_ns": time.perf_counter_ns(), "unix_time_ns": time.time_ns()}
            clocks = [None] * 8
            self.dist.all_gather_object(clocks, self.clock, group=control_group)
            if any(c != clocks[0] for c in clocks) or not self.clock["monotonic"]:
                raise RuntimeError("rank perf_counter clocks have no verified common basis")
            # A completed SAME-DEVICE origin only; no GPU/CPU clock calibration.
            self.origin = self.torch.cuda.Event(enable_timing=True)
            self.origin.record()
            self.origin.synchronize()
            local_routes = {}
            for label, plan in (("default", default_plan), ("candidate", candidate_plan)):
                prefix = self.b.restrict_plan_sequence(plan, start=0, end=self.kv, include_non_kv=True)
                local_routes[label] = self.b.observations.receiver_rows(prefix, dst_bundle, self.rank, "kv_prefix")
            gathered = [None] * 8
            self.dist.all_gather_object(gathered, local_routes, group=control_group)
            self.routes = {label: [row for r in gathered for row in r[label]] for label in local_routes}
            self.routes["candidate_route_stats"] = dict(getattr(candidate_plan, "source_route_stats", {}))
            self.warmup_evidence = None
            if self.mode == "independent":
                pairs = sorted({(row["src"], row["dst"]) for label in ("default", "candidate")
                                for row in self.routes[label] if row["src"] != row["dst"]})
                self.warmup_evidence = self.independent_warmup(
                    services, weights, (src_context, dst_context), control_group, pairs)
            for service in services.values():
                service.set_launch_callback(lambda service=service: self.capture_transfer(service))
        if index == 0 or self.is_measured(index):
            if active_tp != 2:
                raise AssertionError("measured expansion must start from TP2")
            weights[4]._check_storage(False)
            for model, context in ((src_model, src_context), (dst_model, dst_context)):
                self.restore_prefix(model, context)
            self.torch.cuda.current_stream().synchronize()  # outside measured window
            self.audit.check(src_model, src_context, dst_model, dst_context, control_group,
                             stage="restore", weights=False)
            token_counter = self.args.live_prompt_tokens + 32
            for i in range(12):
                self.phase = "normal_warmup" if i < 2 else "normal"
                self.b._decode(src_model, src_context, self.args, token_value=token_counter)
                token_counter += 1
            # Normal decodes appended after the canonical 2048-token prefix.
            # Discard only their logical tail: no GPU work after last completion.
            src_context.sequence_len_offset = dst_context.sequence_len_offset = self.kv
        target = 4 if active_tp == 2 else 2
        weights[target]._check_storage(False)
        self.states.append({"switch": index, "active_tp": active_tp,
                            "kv_tokens": (src_context if active_tp == 2 else dst_context).sequence_len_offset,
                            "target_weight_storage_bytes": 0, "target_weight_bytes_required": weights[target].num_bytes,
                            "real_prefill_tokens": 8, "kv_content": kv_workload(),
                            "measured": self.is_measured(index)})
        self.phase = "prepare"
        self.memory.begin(f"switch_{index}")
        return token_counter

    def prefill(self, original, model, context, args):
        if self.prefill_count == 0:
            self.memory.begin("prefill")
        self.prefill_count += 1
        return original(model, context, args)[:, -1:, :].clone()

    def restore_prefix(self, model, context):
        synthetic_kv.restore(context, self.prefix, self.dist.get_rank(model.pg_collection.tp),
                             self.args.live_request_domain_id, self.torch)

    def remember_plan(self, base, delta, bundle):
        # Keep immutable references inside the switch; inspect after timing.
        self.plan_inputs.append((self.index, base, delta, bundle))

    def decode(self, original, model, context, args, *, token_value):
        if self.origin is None:
            return original(model, context, args, token_value=token_value)
        if self.stall_pause and self.phase in ("base", "weight_check", "delta", "prepare"):
            raise AssertionError("STW attempted a migration decode")
        torch = self.torch
        host_start = time.perf_counter_ns()
        offset = context.sequence_len_offset
        if offset >= context.max_sequence_length:
            raise ValueError("decode would exceed allocated KV capacity")
        value = token_value + (args.live_request_domain_id if args.live_kv_request_identity else 0)
        with torch.inference_mode():
            tokens = torch.full((args.micro_batch_size, 1), value % args.padded_vocab_size,
                                device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.long)
            positions = torch.full_like(tokens, offset)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self.b._apply_hotspot_pressure(model, args)
            logits = model(tokens, positions, None, inference_context=context, runtime_gather_output=True)
            end.record()
            end.synchronize()  # Same completion boundary as the original _decode.
            completed = time.perf_counter_ns()
            elapsed = float(start.elapsed_time(end))
            context.sequence_len_offset += 1
            context.enable_decode_mode()
        self.decodes.append({"rank": self.rank, "step": self.step, "switch": self.index,
                             "tp": int(model.config.tensor_model_parallel_size), "phase": self.phase,
                             "wave": self.wave, "kv_offset": offset, "token_value": token_value,
                             "host_start_ns": host_start, "complete_host_ns": completed,
                             "event_ms": elapsed, "events": (start, end)})
        self.step += 1
        if self.phase == "post" and self.index not in self.first_target:
            self.first_target.add(self.index)
            self.marks.append({"switch": self.index, "name": "target_first_decode_complete", "host_ns": completed})
        return logits, elapsed

    def validate(self, src_model, src_context, dst_model, dst_context, args, group, token):
        initial = self.index < 0 or self.origin is None
        if initial:
            # Gloo-only initialization: choose one real TP2 prefix per request
            # domain so both replicas and TP4 start from bit-identical contents.
            local = dict(rank=self.rank, request_domain=args.live_request_domain_id,
                         tp_ranks=self.dist.get_process_group_ranks(src_model.pg_collection.tp),
                         prefix={layer: tuple(t[:8].cpu().clone() for t in pair)
                                 for layer, pair in src_context.key_value_memory_dict.items()})
            gathered = [None] * self.dist.get_world_size(group)
            self.dist.all_gather_object(gathered, local, group=group)
            self.prefix = synthetic_kv.canonical_prefix(gathered, args.live_request_domain_id, self.torch)
            self.restore_prefix(src_model, src_context)
            self.restore_prefix(dst_model, dst_context)
            audit = self.audit.check(src_model, src_context, dst_model, dst_context, group, stage="initial")
        else:
            self.memory.begin(f"post_switch_{self.index}")
        count = self.extra if not initial and self.is_measured(self.index) else 1
        targets, dst_ms, src_ms, diffs = [], [], [], []
        self.phase, self.wave = "initial_validation" if initial else "post", None
        for step in range(count):
            output, ms = self.b._decode(dst_model, dst_context, args, token_value=token + step)
            targets.append(output)
            dst_ms.append(ms)
        self.phase = "validation"
        if not initial:
            # Post decodes only append beyond 2048. Audit the immutable prefix
            # after first-target completion, before comparing logits or releasing source weights.
            audit = self.audit.check(src_model, src_context, dst_model, dst_context, group, stage="cutover")
        for step, target in enumerate(targets):
            source, ms = self.b._decode(src_model, src_context, args, token_value=token + step)
            metrics = self.b._comparison_metrics(source, target, group)
            within = synthetic_logit_check(metrics, args, audit)
            self.logit_metrics.append(dict(switch=self.index, step=step, token_value=token + step,
                                           policy="record_only_after_bit_content_audits",
                                           within_thresholds=within, metrics=metrics))
            diffs.append(metrics["max_abs"])
            src_ms.append(ms)
        self.memory.observe_context("after_prefill_validation" if initial else f"after_switch_{self.index}_validation")
        if initial:
            self.memory.begin("plan_setup")
        return max(diffs), token + count, self.b._all_max(statistics.mean(src_ms), group), self.b._all_max(statistics.mean(dst_ms), group)

    def write_failure(self, error):
        self.memory.finish()
        directory = Path(self.args.live_json_output).parent
        directory.mkdir(parents=True, exist_ok=True)
        memory = self.memory.document(complete=False)
        raw = dict(schema_version=1, complete=False, rank=self.rank, error=str(error),
                   workload=kv_workload(), validation_audits=self.audit.records, logit_metrics=self.logit_metrics,
                   weight_reference=self.audit.weight_reference, memory=memory,
                   decodes=[{k: v for k, v in r.items() if k != "events"} for r in self.decodes],
                   milestones=self.marks, state_checks=self.states, waves=self.waves)
        for name, value in ((f"memory_rank{self.rank}.json", memory), (f"trace_rank{self.rank}.json", raw)):
            (directory / name).write_text(json.dumps(value, indent=2), encoding="utf-8")

    def write(self, original, args, result):
        self.memory.finish()
        memory = self.memory.document(complete=True)
        measured = max(s["switch"] for s in self.states if s["measured"])
        for row in self.decodes + self.transfers:
            start, end = row.pop("events")
            row["gpu_start_ms"] = self.origin.elapsed_time(start)
            row["gpu_end_ms"] = self.origin.elapsed_time(end)
        for row in self.decodes:
            row["event_interval_overlap_stages"] = [x["stage"] for x in self.transfers
                                     if x["switch"] == row["switch"] and x["wave"] == row["wave"]
                                     and x["phase"] == row["phase"] == "base"
                                     and x["stage"] in ("local", "nccl")
                                     and x["model_payload"]
                                     and intervals_overlap(row["gpu_start_ms"], row["gpu_end_ms"],
                                                           x["gpu_start_ms"], x["gpu_end_ms"])]
        raw = {"schema_version": 1, "rank": self.rank, "clock": self.clock, "mode": self.mode,
               "clock_anchor": self.clock_anchor,
               "measured_switch": measured, "stall_pause": self.stall_pause,
               "decodes": self.decodes, "transfers": self.transfers, "waves": self.waves,
               "milestones": self.marks, "state_checks": self.states,
               "validation_audits": self.audit.records, "logit_metrics": self.logit_metrics,
               "weight_reference": self.audit.weight_reference,
               "independent_warmup": self.warmup_evidence,
               "workload": kv_workload(), "memory": memory,
               "peak_allocated": max(p["max_memory_allocated"] for p in memory["phases"]),
               "peak_reserved": max(p["max_memory_reserved"] for p in memory["phases"]),
               "host_time_semantics": "completion observed immediately after CUDA Event synchronization",
               "verified_execution_overlap_steps": None,
               "gpu_time_semantics": "same-device event bounds; NCCL bounds include device dependency waits; intersection is not kernel-concurrency proof",
               "plan_records": []}
        for index, base, delta, bundle in self.plan_inputs:
            record = result["switches"][index]
            raw["plan_records"].append({"switch": index,
                "plan_variant": "default" if record["plan_variant"] == "baseline" else record["plan_variant"],
                "execution_mode": record["base"]["execution_mode"],
                "adaptive_hybrid": record["adaptive_hybrid"],
                "candidate_fallback_reason": record["candidate_fallback_reason"],
                "source_route_stats": record["source_route_stats"],
                "waves": record["base"]["waves"],
                "base_receiver_rows": self.b.observations.receiver_rows(base, bundle, self.rank, "kv_prefix"),
                "delta_receiver_rows": self.b.observations.receiver_rows(delta, bundle, self.rank, "kv_delta")})
        directory = Path(args.live_json_output).parent
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"trace_rank{self.rank}.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
        (directory / f"memory_rank{self.rank}.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
        if self.rank == 0:
            (directory / "routes.json").write_text(json.dumps(self.routes, indent=2), encoding="utf-8")
        result["nostop_diagnostic"] = {"mode": self.mode, "measured_switch": measured,
                                       "capacity_budget": capacity_budget(), "stall_pause": self.stall_pause,
                                       "workload": kv_workload(),
                                       "logit_validation": "record_only_after_KV_and_weight_bit_content_audits",
                                       "memory_files": [f"memory_rank{rank}.json" for rank in range(8)]}
        original(args, result)
        return raw


def main(diagnostic_type=Diagnostic):
    import benchmark_live_moe_tp as b

    original_run, original_decode = b.run_live_benchmark, b._decode
    original_prefill = b._prefill
    original_write = b._write_results
    holder = {}

    def run():
        args = b.get_args()
        holder["diagnostic"] = args._nostop_diagnostic = diagnostic_type(b, args)
        try:
            original_run()
        except BaseException as error:
            holder["diagnostic"].write_failure(error)
            raise

    b.run_live_benchmark = run
    b._prefill = lambda *a, **kw: holder["diagnostic"].prefill(original_prefill, *a, **kw)
    b._decode = lambda *a, **kw: holder["diagnostic"].decode(original_decode, *a, **kw)
    b._validate_cutover = lambda *a, **kw: holder["diagnostic"].validate(*a, **kw)
    b._write_results = lambda *a, **kw: holder["diagnostic"].write(original_write, *a, **kw)
    b.main()


if __name__ == "__main__":
    main()
