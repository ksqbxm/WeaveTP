# Source-search microbenchmark

Run from `code/current` on SL3060. These commands do not run automatically.
Keep the usual `PYTHON`, `CHECKPOINT` and idle `PROFILE` settings. All server
results, temporary files and caches below are under `/data`.

## CPU checks

The existing interpreter needs `pytest`, `psutil`, `scipy` (with
`scipy.optimize.milp`) and `matplotlib`. No dependency installation is performed
by the runner. Check the imports before reserving the GPUs:

```bash
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/data/weavetp_search_bench/cpu/tmp
export MPLCONFIGDIR=/data/weavetp_search_bench/cpu/matplotlib
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
"$PYTHON" -B -c 'import pytest, psutil, scipy, matplotlib; from scipy.optimize import milp; print(scipy.__version__)'
CUDA_VISIBLE_DEVICES= "$PYTHON" -B -m pytest --noconftest -o addopts= \
  -p no:cacheprovider --basetemp=/data/weavetp_search_bench/cpu/pytest \
  tests/unit_tests/resharding/test_source_search.py \
  tests/unit_tests/resharding/test_source_search_summary.py \
  tests/unit_tests/resharding/test_source_search_execution.py \
  tests/unit_tests/resharding/test_planner.py \
  tests/unit_tests/resharding/test_live_defaults.py \
  tests/unit_tests/resharding/test_live_storage_lifecycle.py -q
```

`--noconftest` avoids unrelated repository-wide dataset downloads and GPU
fixtures. The default-parity tests load the frozen planner at commit
`7deac20feaf76ee4e6a0c53b2289271a903dd12c`, compare ordered TransferOp hashes and
the original route statistics, and exercise the actual benchmark coordinator
with CPU transport doubles. Keep this Git object available on the server.

## Inspect the launch matrix

```bash
DRY_RUN=1 bash tools/resharding/run_source_search_bench.sh
```

This prints 24 commands without starting workers or creating output directories.
Each repeat rotates the eight-case order by one position. The ILP cases share
`WEAVETP_SOURCE_SEARCH=ilp` and differ only in `WEAVETP_ILP_MIP_REL_GAP` (0 or 1e-4).

## Smoke test and real metadata audit

### First: tiny real-GPU forced-plan execution

Keep the miniconda `PYTHON` from the server's usual `env.sh`. Do not change
Torch/NCCL or run the full checkpoint until this test passes every case:

```bash
SMALL=/data/weavetp_search_bench/tiny_$(date +%Y%m%d_%H%M%S)
mkdir -p "$SMALL/tmp" "$SMALL/cache"
TMPDIR="$SMALL/tmp" XDG_CACHE_HOME="$SMALL/cache" \
CUDA_CACHE_PATH="$SMALL/cache/cuda" TORCH_EXTENSIONS_DIR="$SMALL/cache/torch" \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
"$PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node=8 \
  --log-dir "$SMALL/torchrun" --redirects 3 --tee 3 \
  --module tools.resharding.correctness.source_search_gpu --output "$SMALL"
```

This uses tiny non-contiguous BF16 tensors with TP2->TP4 and EP2, not a model
checkpoint. DFS/DP/Dijkstra/heap/LS and both ILP gaps must each finish search,
pass the real 5% Global Gate, execute a changed plan on a fresh NCCL group and
match exact expected payloads without modifying the source. Each rank writes
its results. Transport is the production peer-size-desc NCCL launch/wait/commit
path, with the same all-rank participation marker as the measured forward waves.
No greedy fallback, transport mock or model repair is used in this GPU test.
The CPU test of the same fixture is not evidence that NCCL passed on SL3060.

### Then: full-model smoke and metadata replay

Run these only after the tiny GPU cases pass and the 8 GPUs are available:

```bash
ROOT_OUT=/data/weavetp_search_bench/smoke_$(date +%Y%m%d_%H%M%S) \
CASES="greedy dfs" REPEATS=1 AUDIT=0 WEAVETP_SEARCH_BUDGET_S=60 \
WEAVETP_SEARCH_MEM_GIB=8 \
bash tools/resharding/run_source_search_bench.sh

AUDIT_ROOT=/data/weavetp_search_bench/audit_$(date +%Y%m%d_%H%M%S)
ROOT_OUT="$AUDIT_ROOT" CASES=greedy REPEATS=1 AUDIT=1 \
bash tools/resharding/run_source_search_bench.sh

CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$PYTHON" -B \
  tests/unit_tests/resharding/test_source_search.py \
  --replay "$AUDIT_ROOT/greedy/r1/metadata.pkl" --items 10 --budget 60 \
  > "$AUDIT_ROOT/cpu_replay.log" 2>&1
```

The audit checks the complete real metadata against the frozen default planner,
then builds full forced plans for all six algorithms and both ILP gaps. Each
candidate is checked BEFORE Global Gate, including rejected candidates. It
checks each (src,dst) FIFO by task_id, count, logical/global expert identity,
actual sender-local parameter name, dtype, bounded slice shape and bytes. The
actual sender metadata must agree with the selected TP representative's layout
and EP-local position; the representative's owner need not be the slice sender.
Same-rank transfers are local copies, excluded from remote P2P. Per-rank remote
send/recv counts are emitted to diagnose ranks that have no remote work.

It also samples 10 decision entries (configurable from 10 to 20, seed 2026). All
other entries retain their default source and contribute fixed background load.
DFS, DP, Dijkstra and gap=0 ILP must agree on the optimum within floating-point
tolerance; LS, heap and both ILP modes must not exceed default H. A small-instance
timeout is a failed verification, not proof of optimality. Audit pickles are
trusted local artifacts only; do not load files from untrusted sources.

Audit serialization is inside the planner call. Its timings are contaminated by
the dump and MUST NOT be included in the formal batch. Use fresh audit directories.
The dump now belongs ONLY to the cached forward live-state plan. The d88a3f3
implementation could instead dump the initial weight-only sync and try to reuse
the same filename later. Do not relabel those historical dumps as live-state data.

### Failure evidence boundary

The d88a3f3 initial-sync DFS timeout exposed an environment-scope bug: the planner
read WEAVETP_SOURCE_SEARCH at every call. Only the cached forward call now receives
an explicit source_search_config; initial refit, reverse plans and online replan
do not. Audit output is also an explicit per-call parameter. There is no implicit
environment fallback in the builder.

The sender-local name formerly came from the representative metadata, although
LCM can select a different rank in its TP group. It now comes from that actual
rank's metadata. CPU regressions exercise differing local names and reject
missing tasks/parameters, reordered peer queues, wrong slices, owners and dtypes.
This fixes a routing correctness risk. In the supplied metadata replay, however,
old and new planners produce identical ordered operations when given the same
DFS assignment, so this name fix does not explain that replay's P2P timeout.

The supplied `srcbench_debug.tgz` was inspected on CPU:

- `audit_20261010_202240/greedy/r1/metadata.pkl` is truncated (6,687,014 bytes,
  SHA256 `8800963f4f656bfc120e102837547e953e50454d1d0c09bfb55d4e74a2acedb8`).
  Normal pickle loading raises EOFError. The policy supplies MappingProxyType
  matrices; the old audit's pickle.dump raises TypeError on those matrices,
  leaving an incomplete file. This failure was reproduced locally. Export now
  converts both matrices to dict and serializes before opening the file. Error
  cleanup no longer enters collective group destruction and masks the exception.
- Forensic inspection recovered all 15,488 source records and 15,446 complete
  destination records. Rank 7 lacks 42 destination records. A 60-second DFS replay
  on only the recovered records passed owner/name/shape/byte/task/FIFO checks,
  including the production peer-size-desc ordering on shape-only tensors:
  16,306 remote tasks on 19 directed pairs, at most 2,660 messages per pair.
  Every rank has remote sends and receives. Predicted H is 0.221365922 seconds.
- A SEPARATE reconstruction inferred the 42 missing records from rank 6's same-TP
  layout and rank 7's recovered topology. It reproduced all logged rank-level
  send/recv counts, 6,232 changes and 15,559,098,368 rerouted bytes, and passed
  the same pairing checks for 16,310 remote tasks. This is NOT a complete actual
  metadata replay. Partial-file recovery is not a supported audit input path.
- The timeout log places rank 0/7's SeqNum=8 ALLREDUCE (barrier) on migration
  PG 33 / GUID 139, the SAME group as rank 1..6's pending COALESCED operation.
  Default PG 0 was reporting a dump, not this pending barrier. Thus neither
  a cross-PG barrier mismatch nor absent P2P participation is established.

The scope leak and audit serialization failure are confirmed and fixed. CPU
replay does NOT establish why the large, single-batch initial P2P failed on
NCCL. Initial sync now uses the original successful greedy path; the measured
forced plan uses the existing bounded forward waves. No speculative transport
or runtime change is included. Run the tiny GPU test and fresh full-model smoke
above before treating the execution issue as resolved. Local CPU validation:
205 tests and 82 subtests passed; no SL3060 GPU execution was performed here.

## Formal batch and resume

```bash
BATCH=/data/weavetp_search_bench/$(date +%Y%m%d_%H%M%S)
ROOT_OUT="$BATCH" WEAVETP_SEARCH_BUDGET_S=300 WEAVETP_SEARCH_MEM_GIB=8 \
bash tools/resharding/run_source_search_bench.sh

# Reuse exactly the same BATCH to resume after interruption.
ROOT_OUT="$BATCH" WEAVETP_SEARCH_BUDGET_S=300 WEAVETP_SEARCH_MEM_GIB=8 \
bash tools/resharding/run_source_search_bench.sh

# Regenerate reports without starting workers.
"$PYTHON" -B tools/resharding/summarize_source_search_bench.py "$BATCH"
```

Valid existing results are checked and skipped. Invalid JSON, different settings,
different code or a changed bandwidth profile are errors, not silent skips.
Formal aggregation requires every case/r1..r3 result. `--partial` is available for
smoke batches and labels the plot as incomplete. Never append to historical
formal experiment directories. The runner retains each launch's `run.log`.

## Interpretation

- Total time is `plan_2_to_4_build_s + switches[0].switch_wall_s`. Planning includes
  metadata gathering, candidate construction/search, assembly and scatter. Reverse
  planning, model initialization and checkpoint loading are outside this metric.
- Forced-plan CPU validation is included in build time and recorded separately
  as `plan_validation_s`. It does not run inside the search RSS interval. Runs
  record `source_search_scope=cached_tp2_to_tp4`; old global-scope results are not
  accepted by the current summarizer and must not be resumed or mixed into a batch.
- Transport is the sum of existing wave `transport_s`; migration wall is existing
  `base.wall_s`. Both overlap switch wall and are not added to it. Existing switch
  wall includes its original validation work; it is not client pause time.
- `predicted_H_s` and `overridden_entries` describe the returned search solution
  BEFORE Global Gate. The cached post-gate H, gate outcome, actual switch
  `plan_variant` and fallback reason are separate evidence. A rejected proposal
  incurs search time but executes the default plan. Adaptive hybrid can also select
  the baseline after the gate accepts the candidate.
- Non-greedy methods bypass per-entry greedy filters, including entries absent
  from the override: these always use the default source. They share the original
  global threshold and execution path. Greedy's selection logic is unchanged.
- `prepass_s` includes candidate construction and solving; `solver_s` isolates the
  solver, including ILP model construction/imports. Greedy has no prepass and its
  inseparable solver time is null. No extra census is added to greedy.
- A search time limit covers the full prepass. LS stops after no improving pass or
  50 passes; it has the RSS cap but no time cap. Python methods check limits at entry,
  every 4096 work steps and exit. `budget` means TLE; `memory` means MLE. Each returns
  its best complete solution, or the default, before the unchanged global gate.
- `search_rss_delta_gib` is maximum sampled current rank-0 RSS minus entry RSS.
  Alternative sampling covers candidate construction and solving, before freeing
  search state. Greedy samples only before/after the planner call; these samples are
  outside its build timer. The distinct sampling scopes are recorded.
- ILP samples before/after its native call, with no callback inside HiGHS. It uses
  remaining time but cannot enforce the RSS cap or observe transient RSS peaks
  during that call. `peak_rss_gib` is process-lifetime diagnostic data only and is
  not used for MLE, summary memory metrics or plots.
- ILP `complete` at gap=1e-4 means the requested tolerance was satisfied, not proof
  of exact optimality. Both modes report the actual solver gap and dual bound when
  available. Infeasibility and solver errors fail the launch.
- CSV aggregation uses launch-level means and sample SD (ddof=1). Total error bars
  are computed from per-launch totals. The plot marks any TLE/MLE in the three runs;
  the aggregated CSV contains their counts. Gate and actual candidate counts are
  in the paper table. No missing value is replaced with zero.

No new environment variables: legacy result schema and plan remain unchanged.
No Directional case is produced.
