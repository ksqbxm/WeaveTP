#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate megatron

out=outputs/recheck_p2p_order_arg_20260822
mkdir -p "$out"
profile=profiles/refit_4gpu_p2p_gpu4_7.json
common=(
  --src-tp 1 --dst-tp 2 --expert-parallel-size 2
  --num-experts 8 --moe-router-topk 2 --routing-mode fixed-hot
  --active-experts 0,1 --num-layers 1 --hidden-size 256
  --num-attention-heads 8 --num-query-groups 4
  --ffn-hidden-size 512 --moe-ffn-hidden-size 512
  --seq-len 64 --vocab-size 2048 --micro-batch-size 1
  --dirty-profile-steps 4 --live-train-steps 16 --max-waves 4
  --scheduler-min-benefit-pct 0
  --scheduler-prediction-uncertainty-pct 0
  --min-bandwidth-cv 0.0001 --min-link-degradation-pct 0
  --source-reroute-min-gain-pct 0
  --source-reroute-min-contention-gain-pct 0
  --source-reroute-min-global-gain-pct 0 --source-reroute-min-bytes 0
  --no-profile-p2p-bandwidth
  --scheduler-bandwidth-profile "$profile"
  --scheduler-slow-links '1:2=8,1:3=8'
  --scheduler-slow-link-simulate-delay
  --no-audit-migration-payloads --no-validate-next-optimizer-step
)

run_case() {
  local label=$1 scheduler=$2 order=$3 rep=$4
  local prefix="$out/${label}_${rep}"
  echo "===== $label rep=$rep order=$order ====="
  CUDA_VISIBLE_DEVICES=4,5,6,7 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    timeout 300s python -m torch.distributed.run --standalone --nproc_per_node=4 \
    tools/resharding/moe_tp.py "${common[@]}" \
    --scheduler "$scheduler" --scheduler-p2p-order "$order" \
    $(if [[ "$scheduler" == "bandwidth-aware" ]]; then echo --force-bandwidth-routing; fi) \
    --metrics-output "${prefix}.json" 2>&1 | tee "${prefix}.log"
}

for rep in 1 2 3; do
  run_case baseline baseline send-recv "$rep"
  run_case nccl_round bandwidth-aware nccl-round "$rep"
  run_case send_recv bandwidth-aware send-recv "$rep"
done

python - <<'PY'
import json
import pathlib
import statistics

root = pathlib.Path("outputs/recheck_p2p_order_arg_20260822")
for label in ("baseline", "nccl_round", "send_recv"):
    rows = []
    for path in sorted(root.glob(f"{label}_*.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except Exception as exc:
            print(label, path, "LOAD_ERROR", exc)
    print("\n", label, len(rows))
    for key in (
        "base_wall_s",
        "base_migration_s",
        "transition_wall_s",
        "training_disruption_s",
        "overlap_train_step_s",
        "weight_max_diff",
    ):
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            print(
                key,
                "mean",
                statistics.mean(vals),
                "stdev",
                statistics.stdev(vals) if len(vals) > 1 else 0,
                "vals",
                vals,
            )
    print("scheduler_enabled", [row.get("scheduler_enabled") for row in rows])
    print("p2p_order", [row.get("p2p_order") for row in rows])
PY
