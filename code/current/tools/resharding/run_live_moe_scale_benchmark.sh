#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/megatron/bin/python}
PROFILE=${PROFILE:-profiles/refit_4gpu_p2p.json}
EXPERTS=${EXPERTS:-"256 512 768"}
REPS=${REPS:-10}
START_REP=${START_REP:-1}
SWITCHES=${SWITCHES:-22}
WARMUP_SWITCHES=${WARMUP_SWITCHES:-2}
PROMPT_TOKENS=${PROMPT_TOKENS:-8}
MAX_WAVES=${MAX_WAVES:-1}
MAX_OVERLAP_STEPS=${MAX_OVERLAP_STEPS:-8}
REROUTE_MIN_GAIN_PCT=${REROUTE_MIN_GAIN_PCT:-10.0}
REROUTE_MIN_CONTENTION_GAIN_PCT=${REROUTE_MIN_CONTENTION_GAIN_PCT:-0.0}
REROUTE_MIN_GLOBAL_GAIN_PCT=${REROUTE_MIN_GLOBAL_GAIN_PCT:-5.0}
REROUTE_PENALTY_US=${REROUTE_PENALTY_US:-20.0}
REROUTE_MIN_BYTES=${REROUTE_MIN_BYTES:-1048576}
PACK_TARGET_BYTES=${PACK_TARGET_BYTES:-4194304}
PACK_MAX_ITEM_BYTES=${PACK_MAX_ITEM_BYTES:-1048576}
PERSISTENT_PACK_BUFFERS=${PERSISTENT_PACK_BUFFERS:-0}
PACK_REROUTED_ONLY=${PACK_REROUTED_ONLY:-0}
ONLINE_MIGRATION_FIRST_GUARD=${ONLINE_MIGRATION_FIRST_GUARD:-0}
MIN_TRANSPORT_GAIN_PCT=${MIN_TRANSPORT_GAIN_PCT:-1.0}
MAX_TPOT_REGRESSION_PCT=${MAX_TPOT_REGRESSION_PCT:-5.0}
GUARD_TPOT_METRIC=${GUARD_TPOT_METRIC:-p95}
GUARD_EWMA=${GUARD_EWMA:-0.5}
GUARD_ROBUST_WINDOW=${GUARD_ROBUST_WINDOW:-3}
CANDIDATE_WARMUP_EXPANSIONS=${CANDIDATE_WARMUP_EXPANSIONS:-1}
GUARD_MIN_SAMPLES=${GUARD_MIN_SAMPLES:-3}
GUARD_MIN_CANDIDATE_WIN_RATE=${GUARD_MIN_CANDIDATE_WIN_RATE:-0.6666666666666666}
GUARD_MAX_TRANSPORT_REGRESSION_PCT=${GUARD_MAX_TRANSPORT_REGRESSION_PCT:-2.0}
ONLINE_REPLAN=${ONLINE_REPLAN:-0}
REPLAN_MIN_CHANGE_PCT=${REPLAN_MIN_CHANGE_PCT:-5.0}
REPLAN_MAX_STALE_EXPANSIONS=${REPLAN_MAX_STALE_EXPANSIONS:-4}
REPLAN_NOOP_COOLDOWN_EXPANSIONS=${REPLAN_NOOP_COOLDOWN_EXPANSIONS:-8}
AMORTIZATION_HORIZON_EXPANSIONS=${AMORTIZATION_HORIZON_EXPANSIONS:-0}
FOREGROUND_PRESSURE_CAP=${FOREGROUND_PRESSURE_CAP:-0.90}
FOREGROUND_BASELINE_EWMA=${FOREGROUND_BASELINE_EWMA:-0.25}
GUARD_REEVALUATE_EXPANSIONS=${GUARD_REEVALUATE_EXPANSIONS:-0}
GUARD_HYSTERESIS_PCT=${GUARD_HYSTERESIS_PCT:-0.0}
OUT_ROOT=${OUT_ROOT:-outputs/live_moe_scale_$(date +%Y%m%d_%H%M%S)}

if (( WARMUP_SWITCHES < 0 || WARMUP_SWITCHES >= SWITCHES )); then
    echo "WARMUP_SWITCHES must satisfy 0 <= warmup < switches" >&2
    exit 2
fi
if (( (SWITCHES - WARMUP_SWITCHES) % 2 != 0 )); then
    echo "Measured switches must contain complete TP2->TP4->TP2 cycles" >&2
    exit 2
fi
if [[ ! -f "$PROFILE" ]]; then
    echo "Required bandwidth profile not found: $PROFILE" >&2
    exit 2
fi

required_tokens=$((PROMPT_TOKENS + 1 + SWITCHES * (MAX_WAVES * MAX_OVERLAP_STEPS + 1)))
seq_length=1
while (( seq_length < required_tokens )); do
    seq_length=$((seq_length * 2))
done
measured_cycles=$(((SWITCHES - WARMUP_SWITCHES) / 2))

mkdir -p "$OUT_ROOT"
printf '%s\n' "$OUT_ROOT" > outputs/latest_live_moe_scale.txt
if [[ ! -s "$OUT_ROOT/design.txt" ]]; then
    {
        echo "experts=$EXPERTS"
        echo "paired_repetitions=$REPS"
        echo "start_rep=$START_REP"
        echo "switches=$SWITCHES"
        echo "warmup_switches=$WARMUP_SWITCHES"
        echo "measured_bidirectional_cycles_per_run=$measured_cycles"
        echo "sequence_length=$seq_length"
        echo "reroute_min_gain_pct=$REROUTE_MIN_GAIN_PCT"
        echo "reroute_min_contention_gain_pct=$REROUTE_MIN_CONTENTION_GAIN_PCT"
        echo "reroute_min_global_gain_pct=$REROUTE_MIN_GLOBAL_GAIN_PCT"
        echo "reroute_penalty_us=$REROUTE_PENALTY_US"
        echo "reroute_min_bytes=$REROUTE_MIN_BYTES"
        echo "pack_target_bytes=$PACK_TARGET_BYTES"
        echo "pack_max_item_bytes=$PACK_MAX_ITEM_BYTES"
        echo "persistent_pack_buffers=$PERSISTENT_PACK_BUFFERS"
        echo "pack_rerouted_only=$PACK_REROUTED_ONLY"
        echo "online_migration_first_guard=$ONLINE_MIGRATION_FIRST_GUARD"
        echo "min_transport_gain_pct=$MIN_TRANSPORT_GAIN_PCT"
        echo "max_tpot_regression_pct=$MAX_TPOT_REGRESSION_PCT"
        echo "guard_tpot_metric=$GUARD_TPOT_METRIC"
        echo "guard_robust_window=$GUARD_ROBUST_WINDOW"
        echo "candidate_warmup_expansions=$CANDIDATE_WARMUP_EXPANSIONS"
        echo "guard_min_samples=$GUARD_MIN_SAMPLES"
        echo "guard_min_candidate_win_rate=$GUARD_MIN_CANDIDATE_WIN_RATE"
        echo "guard_max_transport_regression_pct=$GUARD_MAX_TRANSPORT_REGRESSION_PCT"
        echo "online_replan=$ONLINE_REPLAN"
        echo "replan_min_change_pct=$REPLAN_MIN_CHANGE_PCT"
        echo "replan_max_stale_expansions=$REPLAN_MAX_STALE_EXPANSIONS"
        echo "replan_noop_cooldown_expansions=$REPLAN_NOOP_COOLDOWN_EXPANSIONS"
        echo "amortization_horizon_expansions=$AMORTIZATION_HORIZON_EXPANSIONS"
        echo "foreground_pressure_cap=$FOREGROUND_PRESSURE_CAP"
        echo "foreground_baseline_ewma=$FOREGROUND_BASELINE_EWMA"
        echo "guard_reevaluate_expansions=$GUARD_REEVALUATE_EXPANSIONS"
        echo "guard_hysteresis_pct=$GUARD_HYSTERESIS_PCT"
        echo "statistical_unit=paired_run_mean"
    } | tee "$OUT_ROOT/design.txt"
else
    echo "$(date -Is) RESUME start_rep=$START_REP reps=$REPS" \
        | tee -a "$OUT_ROOT/progress.log"
fi

run_case() {
    local experts=$1
    local rep=$2
    local mode=$3
    local out="$OUT_ROOT/e${experts}/rep_$(printf '%02d' "$rep")/${mode}"
    local case_online_guard="$ONLINE_MIGRATION_FIRST_GUARD"
    if [[ "$mode" == "baseline" ]]; then
        case_online_guard=0
    fi

    if [[ -s "$out/result.json" ]]; then
        echo "$(date -Is) SKIP experts=$experts rep=$rep mode=$mode"
        return
    fi

    mkdir -p "$out"
    echo "$(date -Is) START experts=$experts rep=$rep mode=$mode" \
        | tee -a "$OUT_ROOT/progress.log"

    OUT_DIR="$out" \
    PROFILE="$PROFILE" \
    SWITCHES="$SWITCHES" \
    PROMPT_TOKENS="$PROMPT_TOKENS" \
    MAX_WAVES="$MAX_WAVES" \
    MAX_OVERLAP_STEPS="$MAX_OVERLAP_STEPS" \
    REROUTE_MIN_GAIN_PCT="$REROUTE_MIN_GAIN_PCT" \
    REROUTE_MIN_CONTENTION_GAIN_PCT="$REROUTE_MIN_CONTENTION_GAIN_PCT" \
    REROUTE_MIN_GLOBAL_GAIN_PCT="$REROUTE_MIN_GLOBAL_GAIN_PCT" \
    REROUTE_PENALTY_US="$REROUTE_PENALTY_US" \
    REROUTE_MIN_BYTES="$REROUTE_MIN_BYTES" \
    PACK_TARGET_BYTES="$PACK_TARGET_BYTES" \
    PACK_MAX_ITEM_BYTES="$PACK_MAX_ITEM_BYTES" \
    PERSISTENT_PACK_BUFFERS="$PERSISTENT_PACK_BUFFERS" \
    PACK_REROUTED_ONLY="$PACK_REROUTED_ONLY" \
    ONLINE_MIGRATION_FIRST_GUARD="$case_online_guard" \
    MIN_TRANSPORT_GAIN_PCT="$MIN_TRANSPORT_GAIN_PCT" \
    MAX_TPOT_REGRESSION_PCT="$MAX_TPOT_REGRESSION_PCT" \
    GUARD_TPOT_METRIC="$GUARD_TPOT_METRIC" \
    GUARD_EWMA="$GUARD_EWMA" \
    GUARD_ROBUST_WINDOW="$GUARD_ROBUST_WINDOW" \
    CANDIDATE_WARMUP_EXPANSIONS="$CANDIDATE_WARMUP_EXPANSIONS" \
    GUARD_MIN_SAMPLES="$GUARD_MIN_SAMPLES" \
    GUARD_MIN_CANDIDATE_WIN_RATE="$GUARD_MIN_CANDIDATE_WIN_RATE" \
    GUARD_MAX_TRANSPORT_REGRESSION_PCT="$GUARD_MAX_TRANSPORT_REGRESSION_PCT" \
    ONLINE_REPLAN="$ONLINE_REPLAN" \
    REPLAN_MIN_CHANGE_PCT="$REPLAN_MIN_CHANGE_PCT" \
    REPLAN_MAX_STALE_EXPANSIONS="$REPLAN_MAX_STALE_EXPANSIONS" \
    REPLAN_NOOP_COOLDOWN_EXPANSIONS="$REPLAN_NOOP_COOLDOWN_EXPANSIONS" \
    AMORTIZATION_HORIZON_EXPANSIONS="$AMORTIZATION_HORIZON_EXPANSIONS" \
    FOREGROUND_PRESSURE_CAP="$FOREGROUND_PRESSURE_CAP" \
    FOREGROUND_BASELINE_EWMA="$FOREGROUND_BASELINE_EWMA" \
    GUARD_REEVALUATE_EXPANSIONS="$GUARD_REEVALUATE_EXPANSIONS" \
    GUARD_HYSTERESIS_PCT="$GUARD_HYSTERESIS_PCT" \
    SCHEDULER_MODE="$mode" \
    NUM_EXPERTS="$experts" \
    SEQ_LENGTH="$seq_length" \
    MAX_POSITION_EMBEDDINGS="$seq_length" \
    bash tools/resharding/run_live_moe_tp_benchmark.sh \
        > "$out/driver.log" 2>&1

    echo "$(date -Is) DONE  experts=$experts rep=$rep mode=$mode" \
        | tee -a "$OUT_ROOT/progress.log"
}

last_rep=$((START_REP + REPS - 1))
for experts in $EXPERTS; do
    for rep in $(seq "$START_REP" "$last_rep"); do
        if (( rep % 2 == 1 )); then
            run_case "$experts" "$rep" baseline
            run_case "$experts" "$rep" residual
        else
            run_case "$experts" "$rep" residual
            run_case "$experts" "$rep" baseline
        fi
    done
done

echo "$(date -Is) ALL COMPLETE $OUT_ROOT" | tee -a "$OUT_ROOT/progress.log"
"$PYTHON" tools/resharding/summarize_live_moe_scale.py \
    "$OUT_ROOT" \
    --warmup-switches "$WARMUP_SWITCHES" \
    | tee "$OUT_ROOT/scale_summary.txt"
