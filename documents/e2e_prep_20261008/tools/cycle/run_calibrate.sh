#!/usr/bin/env bash
# 第 3 步：用实测结果校准模拟器并重算各方法的墙钟分解（纯 Python，不用 GPU，几分钟）。
# 用法：bash run_calibrate.sh <命令1输出目录（含 cycle_summary.json）> <微基准 jsonl ...>
# 结果写到 <命令1输出目录>/calib/：calib.json、breakdown_uncalibrated.log、breakdown_calibrated.log，并打包 calib_result.tgz
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
CYC=${1:?命令1输出目录}; shift
[[ $# -ge 1 ]] || { echo "STOP: 至少给一个微基准 jsonl"; exit 1; }
[[ -f "$CYC/cycle_summary.json" ]] || { echo "STOP: $CYC 里没有 cycle_summary.json，先运行 summarize_cycle.py"; exit 1; }
git -C "$REPO" -c http.version=HTTP/1.1 fetch -q origin codex/standby-weight-storage
SHA=$(git -C "$REPO" rev-parse FETCH_HEAD); echo "模拟器版本 $SHA"
W="$CYC/calib"; mkdir -p "$W/sim"
for f in e2e_sim.py breakdown.py calibrate.py; do
  git -C "$REPO" show "$SHA:documents/e2e_prep_20261008/sim/$f" > "$W/sim/$f"
done
export SIM_PROFILE_DIR=/data/ubuntu/lxh/weavetp/e2e_prep/profile
[[ -f $SIM_PROFILE_DIR/lengths.csv && -f $SIM_PROFILE_DIR/lengths_longout.csv ]] || { echo "STOP: $SIM_PROFILE_DIR 缺长度文件"; exit 1; }
cd "$W"
"$PYTHON" -I sim/calibrate.py --microbench "$@" --cycle "$CYC/cycle_summary.json" --out calib.json | tee calibrate.log
"$PYTHON" -I sim/breakdown.py > breakdown_uncalibrated.log 2>&1
SIM_CALIB=calib.json "$PYTHON" -I sim/breakdown.py > breakdown_calibrated.log 2>&1
grep -A9 "计划：own）" breakdown_calibrated.log | grep -v "^  " || tail -30 breakdown_calibrated.log
tar czf calib_result.tgz calib.json calibrate.log breakdown_uncalibrated.log breakdown_calibrated.log
echo "已打包：$W/calib_result.tgz"
