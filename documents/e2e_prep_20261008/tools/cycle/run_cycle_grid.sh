#!/usr/bin/env bash
# 第 1 行校准：真实 GPU 上测"带 KV 负载的完整切换周期"（2->4、4->2 各 2 次）。
# 用法（SL3060 单机 8 卡）：nohup bash run_cycle_grid.sh <输出目录> > <输出目录>/grid.log 2>&1 &
# 可用环境变量：METHODS（默认 "weavetp anchortp llumnix"）、KV_LIST（默认 "8 1024 2048 3072"）、MB（默认 16）、SWITCHES（默认 4）
set -uo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
OUT=${1:?输出目录}; mkdir -p "$OUT"
BASE=/data/ubuntu/lxh/weavetp/e2e_prep
METHODS=${METHODS:-"weavetp anchortp llumnix"}
KV_LIST=${KV_LIST:-"8 1024 2048 3072"}
MB=${MB:-16}
SWITCHES=${SWITCHES:-4}
EXTRA=${CYCLE_EXTRA_STEPS:-20}

# 1) 取分支、导出快照（不动仓库工作区）；把包装器放进快照，并让快照里的启动脚本可以换入口
git -C "$REPO" -c http.version=HTTP/1.1 fetch -q origin codex/standby-weight-storage || exit 2
SHA=$(git -C "$REPO" rev-parse FETCH_HEAD); echo "SHA=$SHA" | tee "$OUT/sha.txt"
SNAP=$BASE/cyc_${SHA:0:7}
if [[ ! -f "$SNAP/code/current/examples/rl/cycle_bench.py" ]]; then
  rm -rf "$SNAP"; mkdir -p "$SNAP"
  git -C "$REPO" archive "$SHA" code/current documents/e2e_prep_20261008/tools/cycle | tar -x -C "$SNAP" || exit 2
  cp "$SNAP/documents/e2e_prep_20261008/tools/cycle/cycle_bench.py" "$SNAP/code/current/examples/rl/"
  sed -i 's#^ENTRYPOINT=examples/rl/benchmark_live_moe_tp.py$#ENTRYPOINT=${ENTRYPOINT_OVERRIDE:-examples/rl/benchmark_live_moe_tp.py}#' \
    "$SNAP/code/current/tools/resharding/run_live_moe_tp_benchmark.sh"
  cp -f "$REPO"/code/current/megatron/core/datasets/helpers_cpp*.so "$SNAP/code/current/megatron/core/datasets/" 2>/dev/null || true
fi
grep -q 'ENTRYPOINT_OVERRIDE' "$SNAP/code/current/tools/resharding/run_live_moe_tp_benchmark.sh" || { echo "STOP: 入口替换失败"; exit 2; }
PROFILE=$(ls /home/ubuntu/Megatron-LM-weavetp/profiles/moe_tp_8gpu_idle_p2p_v21.json 2>/dev/null || true)
echo "PROFILE=${PROFILE:-无（使用默认 100 Gbps 矩阵）}" | tee -a "$OUT/sha.txt"

method_env() {
  case "$1" in
    weavetp) echo RELEASE_STANDBY_WEIGHTS=1 METHOD_VARIANT=moetp++-hybrid SCHEDULER_MODE=residual \
      DISABLE_SOURCE_REROUTE=0 ADAPTIVE_HYBRID=1 ADAPTIVE_RESIDUAL_MAX_WAVES=4 ALLOW_AWARE_SHRINK=1 \
      REROUTE_MIN_GAIN_PCT=0 EXPANSION_MAX_WAVE_TASKS=4096 SHRINK_MAX_WAVE_TASKS=2048 MAX_WAVE_TASKS=2048 ;;
    anchortp) echo RELEASE_STANDBY_WEIGHTS=1 METHOD_VARIANT=anchortp-proxy SCHEDULER_MODE=residual \
      DISABLE_SOURCE_REROUTE=1 ADAPTIVE_HYBRID=0 ALLOW_AWARE_SHRINK=0 MAX_WAVE_TASKS=2048 ;;
    llumnix) echo RELEASE_STANDBY_WEIGHTS=0 METHOD_VARIANT=llumnix-proxy SCHEDULER_MODE=baseline \
      DISABLE_SOURCE_REROUTE=1 ADAPTIVE_HYBRID=0 ALLOW_AWARE_SHRINK=0 MAX_WAVE_TASKS=2048 ;;
    flying) echo RELEASE_STANDBY_WEIGHTS=0 METHOD_VARIANT=flying-serving-proxy SCHEDULER_MODE=baseline \
      DISABLE_SOURCE_REROUTE=1 ADAPTIVE_HYBRID=0 ALLOW_AWARE_SHRINK=0 MAX_WAVE_TASKS=2048 ;;
    *) echo "未知方法 $1" >&2; return 1 ;;
  esac
}

port=29531
for KV in $KV_LIST; do
  for M in $METHODS; do
    TAG=${M}_kv${KV}_mb${MB}; D="$OUT/$TAG"; mkdir -p "$D"
    while pgrep -f "torch.distributed.run" >/dev/null; do sleep 20; done; sleep 10
    if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500{b=1} END{exit !b}'; then
      echo "$(date +%T) $TAG: GPU 不空闲，跳过" | tee -a "$OUT/grid.log"; nvidia-smi >> "$D/busy.txt"; continue; fi
    MAXPOS=$(( KV + 200 * (SWITCHES + 1) + 512 ))
    echo "$(date +%T) START $TAG maxpos=$MAXPOS" | tee -a "$OUT/grid.log"
    nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -lms 200 > "$D/smi.csv" &
    SMI=$!
    ( cd "$SNAP/code/current" && env $(method_env "$M") \
        NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
        ENTRYPOINT_OVERRIDE=examples/rl/cycle_bench.py CYCLE_KV_TOKENS=$KV CYCLE_EXTRA_STEPS=$EXTRA \
        CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 ${PROFILE:+PROFILE=$PROFILE} OUT_DIR="$D/run" \
        MICRO_BATCH_SIZE=$MB SWITCHES=$SWITCHES SEQ_LENGTH=$MAXPOS MAX_POSITION_EMBEDDINGS=$MAXPOS \
        MAX_WAVES=128 MAX_OVERLAP_STEPS=1 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 ONLINE_REPLAN=0 \
        timeout 2400 bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh ) 2>&1 \
      | while IFS= read -r l; do printf '%s %s\n' "$(date +%H:%M:%S.%3N)" "$l"; done > "$D/launcher.log"
    rc=${PIPESTATUS[0]}
    kill $SMI 2>/dev/null
    if [[ $rc != 0 ]]; then pkill -f "torch.distributed.run" 2>/dev/null; sleep 30; fi
    echo "$(date +%T) END $TAG exit=$rc" | tee -a "$OUT/grid.log"
    port=$((port + 1))
  done
done
echo "$(date +%T) ALL DONE" | tee -a "$OUT/grid.log"
