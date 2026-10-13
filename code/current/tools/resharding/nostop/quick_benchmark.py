"""One loaded model per method, one warm cycle, then repeated pressure stages."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.resharding.nostop import artifacts, benchmark, common, quick_protocol


class Quick(benchmark.Diagnostic):
    def __init__(self, backend, args):
        super().__init__(backend, args)
        self.schedule = quick_protocol.schedule(int(os.environ["QUICK_K"]), os.environ["QUICK_ORDER"])
        self.root = Path(os.environ["QUICK_OUT"])
        self.method = os.environ["QUICK_METHOD"]
        self.launch_id = os.environ["QUICK_LAUNCH_ID"]
        self.timeout = int(os.environ["QUICK_TIMEOUT"])
        # Each expansion restores 2048; at most one expansion/return pair is live.
        self.kv_pair_peak = self.kv + 2 * (common.MAX_WAVES * args.live_max_overlap_steps + args.live_max_overlap_steps) + self.extra + 1
        if self.kv_pair_peak >= args.max_position_embeddings:
            raise ValueError("quick expansion/return pair exceeds KV capacity")

    def is_measured(self, index):
        return index >= 2 and index % 2 == 0

    def prefill(self, original, model, context, args):
        # The legacy pre-allocation guard assumes KV grows forever. Retain its
        # checked three-switch allocation, then extend ONLY the host loop after
        # that guard. The reset-aware pair bound above replaces its long-run bound.
        args.live_switches = 2 + 2 * len(self.schedule)
        return super().prefill(original, model, context, args)

    def change_level(self, row, control_group):
        self.dist.barrier(group=control_group)
        self.torch.cuda.empty_cache()  # outside all measured windows, for BOTH orders
        self.dist.barrier(group=control_group)
        self.memory.observe_context(f"quick_{row['level']}")
        current = self.memory.context_samples[-1]
        if not current["valid"]:
            raise ValueError(f"quick requires valid resident context measurement: {current}")
        sample = dict(rank=self.rank, gpu_uuid=current["gpu_uuid"],
                      allocated_peak_bytes=max([self.torch.cuda.max_memory_allocated(),
                                                *(p["max_memory_allocated"] for p in self.memory.phases)]),
                      context_peak_bytes=max(p["context_bytes"] for p in self.memory.context_samples if p["valid"]),
                      reserved_bytes=current["reserved_bytes"], context_bytes=current["context_bytes"])
        ranks = [None] * 8
        self.dist.all_gather_object(ranks, sample, group=control_group)
        path = self.root / "methods" / self.method / "handshake" / row["level"]
        if self.rank == 0:
            artifacts.save_json(path / "request.json", dict(level=row["level"], switch=row["switch"], ranks=ranks))
            go = quick_protocol.wait_json(path / "go.json", self.timeout)
            if not go.get("go") or (go["level"], go["switch"]) != (row["level"], row["switch"]):
                raise ValueError("stale/invalid quick go file")
        self.dist.barrier(group=control_group)

    def before_switch(self, index, active_tp, src_model, src_context, dst_model, dst_context,
                      weights, services, control_group, default_plan, candidate_plan, dst_bundle, token_counter):
        if self.is_measured(index):
            row = self.schedule[(index - 2) // 2]
            if row["change_level"]:
                self.change_level(row, control_group)
        return super().before_switch(index, active_tp, src_model, src_context, dst_model, dst_context,
                                     weights, services, control_group, default_plan, candidate_plan, dst_bundle, token_counter)

    def write(self, original, args, result):
        result["quick_process"] = dict(launch_id=self.launch_id, schedule=self.schedule,
                                       runtime_switches=args.live_switches, kv_pair_peak=self.kv_pair_peak,
                                       max_position_embeddings=args.max_position_embeddings)
        raw = super().write(original, args, result)
        for row in self.schedule:
            trace, selected = quick_protocol.project(raw, result, row, self.launch_id)
            directory = self.root / quick_protocol.case_name(self.method, row) / "run"
            artifacts.write_json(directory / f"trace_rank{self.rank}.json", trace)
            artifacts.write_json(directory / f"memory_rank{self.rank}.json", trace["memory"])
            if self.rank == 0:
                artifacts.write_json(directory / "result.json", selected)
                artifacts.write_json(directory / "routes.json", self.routes)


if __name__ == "__main__":
    benchmark.main(Quick)
