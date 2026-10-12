"""Stage peak counters and quiescent process-NVML observations; no CUDA sync."""

import csv
import os
import subprocess
import time

from .common import kv_workload


class StageMemory:
    def __init__(self, torch, rank):
        self.cuda, self.rank = torch.cuda, rank
        self.device_total_bytes = self.cuda.get_device_properties(self.cuda.current_device()).total_memory
        self.phases, self.context_samples = [], []
        self.stage = None
        self.begin("startup")

    def finish(self):
        if self.stage is not None:
            self.phases.append(dict(stage=self.stage, start_ns=self.started, end_ns=time.perf_counter_ns(),
                                    max_memory_allocated=self.cuda.max_memory_allocated(),
                                    max_memory_reserved=self.cuda.max_memory_reserved()))
            self.stage = None

    def begin(self, stage):
        self.finish()
        self.cuda.reset_peak_memory_stats()
        self.stage, self.started = stage, time.perf_counter_ns()

    def observe_context(self, where):
        # Called only during setup or after completed validation, never within
        # the measured switch-to-first-target-completion window. No empty_cache.
        device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[self.cuda.current_device()]
        reserved = self.cuda.memory_reserved()
        record = dict(where=where, host_ns=time.perf_counter_ns(), reserved_bytes=reserved, pid=os.getpid())
        command = ["nvidia-smi", "-i", device, "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]
        try:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15, check=True)
            record["nvml_raw"] = result.stdout
            rows = [r for r in csv.reader(result.stdout.splitlines(), skipinitialspace=True) if int(r[0]) == os.getpid()]
            if len(rows) != 1 or self.cuda.memory_reserved() != reserved:
                raise ValueError("need one process/GPU NVML sample with unchanged reserved bytes")
            _, uuid, used = rows[0]
            overhead = int(used) * 1024**2 - reserved
            if overhead < 0:
                raise ValueError("NVML process memory is below torch reserved; cannot infer context overhead")
            record.update(gpu_uuid=uuid.strip(), nvml_process_bytes=int(used) * 1024**2,
                          context_bytes=overhead, valid=True)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            record.update(valid=False, error=str(error))
        self.context_samples.append(record)

    def document(self, complete):
        return dict(rank=self.rank, workload=kv_workload(), complete=complete,
                    device_total_bytes=self.device_total_bytes,
                    phases=self.phases, context_samples=self.context_samples,
                    nvml_note="process usage includes reserved cache; only NVML minus simultaneous reserved estimates context/non-Torch overhead")
