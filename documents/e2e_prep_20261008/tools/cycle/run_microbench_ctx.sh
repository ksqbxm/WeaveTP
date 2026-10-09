#!/usr/bin/env bash
# 命令 2：单步时间随上下文长度的扫描（SL3061，8 卡空闲）。基于服务器上已有的 run_microbench_v2.sh / microbench_ec.py。
# 用法：cd /data/ubuntu/lxh/weavetp/e2e_prep && nohup bash run_microbench_ctx.sh > microbench/ctx_$(date +%m%d_%H%M).nohup 2>&1 &
set -euo pipefail
cd /data/ubuntu/lxh/weavetp/e2e_prep
[[ -f run_microbench_v2.sh && -f microbench_ec.py ]] || { echo "STOP: 缺 run_microbench_v2.sh 或 microbench_ec.py"; exit 1; }
OLD='--mb-decode-batches 1,32,64,96,128,160,192,224,256 --mb-decode-contexts 128,512'
NEW='--mb-decode-batches 16,32,64,128 --mb-decode-contexts 512,2048,4096,8192 --mb-decode-steps 12'
grep -qF -- "$OLD" run_microbench_v2.sh || { echo "STOP: run_microbench_v2.sh 里找不到原网格参数"; grep -n "mb-decode" run_microbench_v2.sh; exit 1; }
sed "s#${OLD}#${NEW}#" run_microbench_v2.sh > run_microbench_ctx_gen.sh
grep -qF -- "$NEW" run_microbench_ctx_gen.sh || { echo "STOP: 替换失败"; exit 1; }
echo "$(date +%T) 开始上下文扫描：$NEW"
bash run_microbench_ctx_gen.sh
echo "$(date +%T) 完成；最新结果：$(ls -t microbench/*.jsonl | head -1)"
