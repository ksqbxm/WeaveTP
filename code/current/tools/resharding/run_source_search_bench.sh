#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PYTHON=${PYTHON:?Set PYTHON to the existing server interpreter}
CHECKPOINT=${CHECKPOINT:?Set CHECKPOINT to the Megatron checkpoint}
PROFILE=${PROFILE:?Set PROFILE to the existing idle bandwidth profile}
ROOT_OUT=${ROOT_OUT:-/data/weavetp_search_bench/$(date +%Y%m%d_%H%M%S)}
REPEATS=${REPEATS:-3}
DRY_RUN=${DRY_RUN:-0}
AUDIT=${AUDIT:-0}
BUDGET=${WEAVETP_SEARCH_BUDGET_S:-300}
MEMORY=${WEAVETP_SEARCH_MEM_GIB:-8}
RUNNER=${RUNNER:-tools/resharding/run_deepseek_v2_lite_live_benchmark.sh}
SUMMARY=tools/resharding/summarize_source_search_bench.py
read -r -a cases <<< "${CASES:-greedy ls heap dp dfs dijkstra ilp_gap0 ilp_gap1e-4}"

if [[ "$ROOT_OUT" != /data/* || ! "$REPEATS" =~ ^[1-3]$ || ${#cases[@]} == 0 ]]; then
    echo "ROOT_OUT must be under /data, REPEATS must be 1..3, and CASES must be nonempty" >&2
    exit 2
fi
seen=" "
for name in "${cases[@]}"; do
    case "$name" in
        greedy|ls|heap|dp|dfs|dijkstra|ilp_gap0|ilp_gap1e-4) ;;
        *) echo "Unknown case: $name" >&2; exit 2 ;;
    esac
    if [[ "$seen" == *" $name "* ]]; then
        echo "Duplicate case: $name" >&2
        exit 2
    fi
    seen+="$name "
done

fixed=(
    "PYTHON=$PYTHON" "CHECKPOINT=$CHECKPOINT" "PROFILE=$PROFILE"
    NNODES=1 NODE_RANK=0 SWITCHES=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024
    MAX_WAVES=128 MAX_WAVE_TASKS=2048 MAX_OVERLAP_STEPS=1
    PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 ONLINE_REPLAN=0
    METHOD_VARIANT=moetp++-hybrid SCHEDULER_MODE=residual DISABLE_SOURCE_REROUTE=0
    ADAPTIVE_HYBRID=1 ADAPTIVE_RESIDUAL_MAX_WAVES=4
    EXPANSION_MAX_WAVE_TASKS=4096 SHRINK_MAX_WAVE_TASKS=2048
    ALLOW_AWARE_SHRINK=0 HYBRID_FAST_PATH=0 EMULATE_NONCOLLOCATED_SOURCES=0
    PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 REPEAT_FORWARD=0
    RELEASE_STANDBY_WEIGHTS=0 KV_REQUEST_IDENTITY=0 WEIGHT_BITWISE_AUDIT=0
    WEIGHT_CHECK_AUDIT=0 WEIGHT_STORAGE_AUDIT=0 DIAGNOSE_EQUIVALENCE=0
    ONLINE_MIGRATION_FIRST_GUARD=0 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4
    ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1,2,3,4,5 ACTIVE_EXPERT_PHASES=
    PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 PROMPT_TOKENS=8
    REROUTE_MIN_GAIN_PCT=10 REROUTE_MIN_CONTENTION_GAIN_PCT=0
    REROUTE_MIN_GLOBAL_GAIN_PCT=5 REROUTE_PENALTY_US=20 REROUTE_MIN_BYTES=1048576
    LOGIT_VALIDATION_MODE=bf16-relative LOGIT_MAX_NRMSE=0.4
    LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0 P2P_ORDER=peer-size-desc
    "WEAVETP_SEARCH_BUDGET_S=$BUDGET" "WEAVETP_SEARCH_MEM_GIB=$MEMORY"
    PYTHONDONTWRITEBYTECODE=1
)

if [[ "$DRY_RUN" != 1 ]]; then
    ROOT_OUT=$(realpath -m -- "$ROOT_OUT")
    if [[ "$ROOT_OUT" != /data/* ]]; then
        echo "Resolved output directory escapes /data" >&2
        exit 2
    fi
    mkdir -p "$ROOT_OUT/cache/tmp" "$ROOT_OUT/cache/matplotlib"
    export TMPDIR="$ROOT_OUT/cache/tmp" TMP="$ROOT_OUT/cache/tmp" TEMP="$ROOT_OUT/cache/tmp"
    export XDG_CACHE_HOME="$ROOT_OUT/cache" MPLCONFIGDIR="$ROOT_OUT/cache/matplotlib"
    export TORCH_EXTENSIONS_DIR="$ROOT_OUT/cache/torch_extensions"
    export CUDA_CACHE_PATH="$ROOT_OUT/cache/cuda" PYTHONDONTWRITEBYTECODE=1
    "$PYTHON" -B - "$CHECKPOINT" "$PROFILE" "$BUDGET" "$MEMORY" <<'PY'
import math
import sys
from pathlib import Path
import matplotlib
import psutil
import scipy
from scipy.optimize import milp

checkpoint, profile, budget, memory = sys.argv[1:]
assert Path(checkpoint).is_dir(), checkpoint
assert Path(profile).is_file(), profile
assert math.isfinite(float(budget)) and 0 < float(budget) <= 540
assert math.isfinite(float(memory)) and float(memory) > 0
print(f"CPU preflight: scipy={scipy.__version__} psutil={psutil.__version__} matplotlib={matplotlib.__version__}")
PY
fi

for ((repeat = 1; repeat <= REPEATS; repeat++)); do
    offset=$(((repeat - 1) % ${#cases[@]}))
    for ((index = 0; index < ${#cases[@]}; index++)); do
        name=${cases[$(((index + offset) % ${#cases[@]}))]}
        algorithm=$name
        gap=0
        if [[ "$name" == ilp_gap* ]]; then
            algorithm=ilp
            [[ "$name" != ilp_gap1e-4 ]] || gap=1e-4
        fi
        out="$ROOT_OUT/$name/r$repeat"
        check=("$PYTHON" -B "$SUMMARY" --check-run "$out/result.json" --case "$name"
               --budget "$BUDGET" --memory "$MEMORY" --check-current
               --checkpoint "$CHECKPOINT" --profile "$PROFILE")
        command=(env -u WEAVETP_SEARCH_AUDIT_PATH "${fixed[@]}"
                 "WEAVETP_SOURCE_SEARCH=$algorithm" "WEAVETP_ILP_MIP_REL_GAP=$gap" "OUT_DIR=$out")
        if [[ "$AUDIT" == 1 ]]; then
            command+=("WEAVETP_SEARCH_AUDIT_PATH=$out/metadata.pkl")
        fi
        command+=(bash "$RUNNER")
        if [[ "$DRY_RUN" == 1 ]]; then
            printf 'PLAN repeat=%s case=%s ' "$repeat" "$name"
            printf '%q ' "${command[@]}"
            printf '\n'
            continue
        fi
        if [[ -e "$out/result.json" ]]; then
            "${check[@]}"
            echo "SKIP repeat=$repeat case=$name"
            continue
        fi
        if [[ "$AUDIT" == 1 && -e "$out/metadata.pkl" ]]; then
            echo "An incomplete audit exists at $out; use a fresh audit ROOT_OUT" >&2
            exit 2
        fi
        echo "START repeat=$repeat case=$name"
        "${command[@]}"
        "${check[@]}"
    done
done

if [[ "$DRY_RUN" != 1 ]]; then
    summary_args=()
    if [[ "$REPEATS" != 3 || ${#cases[@]} != 8 ]]; then
        summary_args+=(--partial)
    fi
    "$PYTHON" -B "$SUMMARY" "$ROOT_OUT" "${summary_args[@]}"
    echo "Source-search benchmark complete: $ROOT_OUT"
fi
