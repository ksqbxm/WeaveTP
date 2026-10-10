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

Run these only when ready to launch the 8-GPU benchmark:

```bash
ROOT_OUT=/data/weavetp_search_bench/smoke_$(date +%Y%m%d_%H%M%S) \
CASES="greedy dfs" REPEATS=1 WEAVETP_SEARCH_BUDGET_S=60 \
bash tools/resharding/run_source_search_bench.sh

AUDIT_ROOT=/data/weavetp_search_bench/audit_$(date +%Y%m%d_%H%M%S)
ROOT_OUT="$AUDIT_ROOT" CASES=greedy REPEATS=1 AUDIT=1 \
bash tools/resharding/run_source_search_bench.sh

CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$PYTHON" -B \
  tests/unit_tests/resharding/test_source_search.py \
  --replay "$AUDIT_ROOT/greedy/r1/metadata.pkl" --items 10 \
  > "$AUDIT_ROOT/cpu_replay.log" 2>&1
```

The audit checks the complete real metadata against the frozen default planner,
then samples 10 decision entries (configurable from 10 to 20, seed 2026). All
other entries retain their default source and contribute fixed background load.
DFS, DP, Dijkstra and gap=0 ILP must agree on the optimum within floating-point
tolerance; LS, heap and both ILP modes must not exceed default H. A small-instance
timeout is a failed verification, not proof of optimality. Audit pickles are
trusted local artifacts only; do not load files from untrusted sources.

Audit serialization is inside the planner call. Its timings are contaminated by
the dump and MUST NOT be included in the formal batch. Use fresh audit directories.

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
