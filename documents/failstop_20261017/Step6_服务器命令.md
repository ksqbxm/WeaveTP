# Step 6 服务器命令（A 部分 fake 快验 + B 部分真实模型分层快验）

实现 `0f1776a0ff0929114d68318f62e7e2c3d957f508`，快验清单 `df7ad5d33364fbd481c82c8164aad093ce984d54`（`releases.json["step6"]`），见[实现与验证](Step6_实现与验证.md)。本批不含任何 GPU 运行：快验在一个节点上用 CPU 跑 fake 后端完整循环（扩缩容 2→4、4→2 与 fail-stop fail→replay→expand）加 `tests/unit_tests/e2e` 的 CPU 测试，`accepted=false`、`timing_valid=false`。它不占 GPU，不依赖 Step 3/4/5 的上机顺序，可随时运行；真实模型的 B 部分（B1 单域固定 TP4 → B2 → B3）等阶段 A 服务器验收后另行交付。


**2026-10-18 B 部分交付**：实现 `ab5a4a90a1088df232eadcce0b5a2dd0ba152dba`，快验清单 `a07bde7a2a357581505f8d72988839d6401dc222`（`releases.json["step6"]`，与 Step 3/4 重钉的清单 `371daf8` 一并打入 bundle `step6b_a07bde7.bundle`），见[实现与验证](Step6_实现与验证.md) B 部分。快验分四层，都是 `quick_check.sh --step 6 --layer <层>`，每层预算 300 s，`accepted=false`：

| 层 | 参数 | 内容 | 节点 |
|---|---|---|---|
| A | `--layer a --single-node`（缺省层） | fake 后端全循环 + e2e CPU 测试 | 任一节点，无 GPU |
| B1 | `--layer b1 --single-node` | 真实模型 TP4 单副本组（8 卡）：block KV 与静态 KV 的 logits 对比（nrmse/cosine）、M1 单步时间、池字节记账；decode 为 torch split-KV | SL3061 |
| B2 | `--layer b2 --single-node` | 真实模型 batched 逐位检查（确定性模式）：两个独立 TP2 进程（GPU 0–3 / 4–7）+ 一个 TP4 进程，变长批、跨块边界、同进程两遍与跨进程逐位 | SL3061 |
| B3 | `--layer b3 --profile-file … --reference-root …`（两节点，不带 `--single-node`，可加 `--mode real`） | 两节点 fail-stop 端到端缩小版：24 条玩具请求，故障 → 缩容 → 重新 prefill → rejoin → 扩容，timeline 自检、`events_actual.json`、全部请求完成 | SL3060 + SL3061 |

A 层命令见第 1 节；B 层见第 1b 节。Triton kernel 本轮不实现，所有层 `attention=torch_splitkv`。

## 0. 前置检查

runner 需要 `pytest`，驱动启动前预检编译好的 Megatron 数据集 helpers（SL3060 曾因继承 base 环境 Python 3.13、缺 pybind11 现场编译失败）。任一节点：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && "$PYTHON" -c "import pytest; print('pytest', pytest.__version__)" && (cd "$REPO/code/current" && "$PYTHON" -c "import megatron.core.datasets.helpers_cpp" && echo helpers_cpp ok)
```

两行都成功才继续；`pytest` 缺失时需先在 env.sh 的环境中安装，再运行第 1 节。

## 1. 单节点快速正确性测试（先完成第 2 节离线导入）

一个节点一条命令，不等待对端，不用 GPU。SL3061 示例，换节点只改 `--node`：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach df7ad5d33364fbd481c82c8164aad093ce984d54 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 6 --node SL3061 --single-node
```

入口内部再次 source env.sh，把 `$(dirname "$PYTHON")` 放到 PATH 最前，预检 `helpers_cpp`，按 releases.json 检出固定实现 `0f1776a0ff0929114d68318f62e7e2c3d957f508`；结束后无论成功失败都切回 `f7d555998b45756998465924d733fd70e553d6b4`。同一检出目录的各步快验共用一把本地锁。预算 300 s（本机实测两次运行加 30 项测试约 13 s）。`--step 6` 只有单节点形式，不带 `--single-node` 即报错。

完成时日志必须包含：

```text
QUICK CHECK step=6 result=PASS (fake backend: controller/kv/anchors/decode/timeline, failstop cycle; runs={'scaling': ['2->4', '4->2'], 'failstop': ['shrink', 'expand']}; tests=30 passed, <警告数> warnings in <秒>s) backend=fake single_node=true accepted=false logs=<目录>
```

receipt.result 为 PASS、exit_code 为 0，`quick_mode=true`、`correctness_only=true`、`shared_gpu_mode=false`（无 GPU）、`timing_valid=false`、`accepted=false`、`single_node=true`、`budget_s=300`。失败标准：任一 fake 运行的 `check.json` 不通过、切换序列不是 `['2->4','4->2']`/`['shrink','expand']`、任一切换校验失败、pytest 非 0 或无 `N passed`、`pytest` 缺失、超时、打包或恢复提交失败。

查看本机最新快验（SL3060 将 NODE 改为 SL3060）：

```bash
NODE=SL3061
ROOT=/data/ubuntu/lxh/weavetp/acceptance/step06_batched_serving
LATEST=$(find "$ROOT/kickoff" -maxdepth 1 -type d -name "${NODE}_*" | sort | tail -n 1)
tail -n 60 "$LATEST/nohup.log"
test ! -f "$LATEST/exit_code" || cat "$LATEST/exit_code"
test ! -f "$LATEST/final.json" || cat "$LATEST/final.json"
```

结果目录形如 `$ROOT/s6q_..._single/<NODE>`：`scaling/` 与 `failstop/`（`inputs/` 玩具负载、`command.txt`、`out/result.json`、`out/check.json`、`out/timeline*.jsonl`、`out/switch_events*.jsonl`）、`pytest.log`、`summary.json`，失败时另有 `error.txt`。回传 `$ROOT/s6q_..._<NODE>.tar.gz` 与同前缀 `.receipt.json`；失败时另带 kickoff 的 nohup.log/final.json 与 `error.txt`。tar 内 `quick_status.result=PENDING_FINALIZATION` 是有意设计。


## 1b. B 层快验（真实模型；先完成第 2 节离线导入）

B1、B2 在 SL3061 一条命令（不等对端）；B3 两节点各一条，60 秒内先后启动。所有层内部按 `releases.json["step6"]` 检出实现 `ab5a4a90a1088df232eadcce0b5a2dd0ba152dba`，结束后切回 `f7d555998b45756998465924d733fd70e553d6b4`。checkpoint 取 env.sh 的 `CHECKPOINT`（缺省 `/data/models/DeepSeek-V2-Lite-megatron-v2`），`--seq-length 512`（rotary 表与静态基线的 max_position_embeddings）。

### B1（SL3061，8 卡，默认模式，1 次加载）

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach a07bde7a2a357581505f8d72988839d6401dc222 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 6 --layer b1 --node SL3061 --single-node
```

显存预检同 Step 5 的 tp4 参考运行（每卡 5.98 GiB）再加块池：TP4 每段 283 MB/卡（4096 token），本层最多 2 段；logits 按 token 物化（`[1, S, 102400]`，S ≤ 200）。探针请求 8 token prompt + 64 步 teacher forcing，另 7 条 prompt 20–130 token 的请求同批（跨 64 token 块边界）。通过标准（`thresholds`）：探针单独与批内两条路径对静态 KV 的逐步 logits `nrmse_max ≤ 0.4`、`cosine_min ≥ 0.93`（W2 验收口径），池物理字节与段记账误差 < 10%，总生成 token 等于请求输出长度之和，`attention_impl.decode = torch_splitkv`。结论行：

```text
QUICK CHECK step=6 layer=b1 result=PASS ({"nrmse_max": …, "cosine_min": …, "top1_agreement": …, "m1_ms": {"static_decode": …, "block_alone_decode": …, "block_batched_decode": …, …}, "attention_impl": {…}, "pool_error": …}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=true accepted=false logs=…
```

M1 数字在共享 GPU 下只作开销评估。结果目录 `$ROOT/s6q_..._single/SL3061/b1/`：`result.json`（每 rank：alone/batched 指标、m1_ms、memory、manifest）、`torchrun.log`、`timeline_*_rank*.jsonl`、`memory_preflight.json`。

### B2（SL3061，8 卡，确定性模式，3 次加载：TP2 两进程并发 + TP4）

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach a07bde7a2a357581505f8d72988839d6401dc222 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 6 --layer b2 --node SL3061 --single-node
```

两个 TP2 进程分别在 GPU 0–3 与 4–7（各卡 7.93 GiB + 池，`--rdzv-endpoint 127.0.0.1:29840/29841`），完成后 TP4 用 8 卡；三次加载是否在 300 s 内未实测，超时即 FAIL（Step 5 单节点六次加载用了 600 s 预算，本层只有三次）。负载：10 条请求，prompt 40/70/100/130/60、输出 70/30/50/66/40 循环，每副本 3 槽，greedy；跨块边界、完成即释放补入、末尾一个 dense 副本空（dummy 前向）。通过标准：三个进程各自两遍（新池）逐位相等（Step 5 `compare_records` 合同：manifest 确定性、同历史、同参数校验和）、两个 TP2 进程跨进程逐位相等、manifests 一致、总生成 token 正确。结论行：

```text
QUICK CHECK step=6 layer=b2 result=PASS ({"in_process_bitwise": {"tp2_a": true, "tp2_b": true, "tp4": true}, "cross_process_tp2_bitwise": true, "forwards": {…}, "crossed_block_boundary": true}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=true accepted=false logs=…
```

结果目录 `…/SL3061/b2/{tp2_a,tp2_b,tp4}/result.json`、`records_rank_*.pt`、`cross_process.json`。

### B3（两节点，emulate 缺省；`--mode real` 为真实 kill/respawn）

前置：冻结的 Step 3 profile（两节点同一路径）与 Step 5 参考 hash（SL3061 上 `…/s5q_20261017T165421_80b77c92_single/SL3061`，含 `tp2/reference` 与 `tp4/reference`；SL3060 传同一参数即可，不读取）。SL3060 先启动：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach a07bde7a2a357581505f8d72988839d6401dc222 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 6 --layer b3 --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_<suite>.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

SL3061 同一命令只改 `--node SL3061`。握手端口 29839，runner 端口 29840 起（与 Step 2–5 相同，不能与其他步同时启动）。流程：16 卡 TP4 加载（EDP0=SL3061，EDP1=SL3060）→ 服务 24 条玩具请求（每副本 4 槽，greedy，决策来自 SL3060 控制节点）→ 完成 6 条时注入故障 → EDP0 幸存卡缩容 TP2（权重来自残余 TP4 + EDP1，在途请求回队首重放）→ fail 后 5 s 且缩容完成即 rejoin → 回归卡/新进程接收 TP4 权重 + 块级 KV 前缀与增量（对照幸存者快照逐位）→ 扩容 cutover → 排空。若负载在 rejoin 前排空，控制节点让两域空步直到扩容完成并把 `failstop.stats_valid=false`（原因写明）。通过标准：控制门 `passed`、`check.json` 通过（Σ gen_tokens、完成数、阶段和）、扩容 KV 检查通过、全部请求完成、16 份阶段报告账本 digest 一致。结论行（SL3060 含指标，SL3061 只有 status）：

```text
QUICK CHECK step=6 layer=b3 result=PASS ({"mode": "emulate", "metrics": {"requests_completed": 24, "check_passed": true, "kv_check_passed": true, "replay_prefill_tokens": …, "shrink_wall_s": …, "expand_wall_s": …, "stats_valid": …}, "files": {…}}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=false accepted=false logs=…
```

结果目录 `$ROOT/s6q_…/<NODE>/<launch>/`：`result.json`、`check.json`、`events_actual.json`（SL3060）、`timeline_<域>_<slot>.jsonl`、`switch_events_*.jsonl`、`serving_checks_<slot>.json`、`kv/r0/expand/replica_*/coord_*.pt`、`source_plan_0_{shrink,expand,delta}.json`、`serving_phase_checks.json`、`rank_map.json`、`controller_result.json`。回传两台的 tar 与 receipt。

## 2. Windows 离线 bundle 生成与上传

本轮已生成并验证：`C:/Users/ksqbx/.codex/worktrees/failstop/step6b_a07bde7.bundle`，1289292 字节，SHA256 `64f8d4f48e78a28fe2ac831561c0f6e90759c94521861c443689a8694b0a9800`，配套 `.sha256` 在同目录。基线严格为 `f7d555998b45756998465924d733fd70e553d6b4`，目标为 B 部分快验清单提交 `a07bde7a2a357581505f8d72988839d6401dc222`，包含 Step 1–6（含 Step 3/4 重钉清单 `371daf8`）与路由确定性检查的全部提交。

（A 部分的包）本轮已生成并验证：`C:/Users/ksqbx/.codex/worktrees/failstop/step6_df7ad5d.bundle`，1216667 字节，SHA256 `f02e68ece3636bf3f7d38b27fca680bd96c4032e00d49ce735fcf89d8b081098`，配套 `.sha256` 在同目录。基线严格为 `f7d555998b45756998465924d733fd70e553d6b4`，目标为快验清单提交 `df7ad5d33364fbd481c82c8164aad093ce984d54`，包含 Step 1–6 与路由确定性检查的全部提交。

上传：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step6b_a07bde7.bundle'
Get-FileHash -LiteralPath $Bundle -Algorithm SHA256
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

等价显式上传命令：

```powershell
scp -P 43061 $Bundle "$Bundle.sha256" ubuntu@hdssh01.casdao.com:/data/ubuntu/lxh/weavetp/
```

如需重新生成，用新文件名，不覆盖已有包：

```powershell
$Repo = 'C:/Users/ksqbx/.codex/worktrees/failstop/WeaveTP_代码与实验资料_20260920'
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step6b_a07bde7_rebuilt.bundle'
if (Test-Path -LiteralPath $Bundle) { throw '文件已存在，请先核对，不覆盖' }
git -C $Repo update-ref refs/heads/codex/step6b-a07bde7 a07bde7a2a357581505f8d72988839d6401dc222
git -C $Repo bundle create $Bundle refs/heads/codex/step6b-a07bde7 '^f7d555998b45756998465924d733fd70e553d6b4'
if ($LASTEXITCODE) { throw 'bundle create failed' }
git -C $Repo bundle verify $Bundle
if ($LASTEXITCODE) { throw 'bundle verify failed' }
$Hash = (Get-FileHash -LiteralPath $Bundle -Algorithm SHA256).Hash.ToLowerInvariant()
$Name = [System.IO.Path]::GetFileName($Bundle)
[System.IO.File]::WriteAllText("$Bundle.sha256", "$Hash  $Name`n", [System.Text.UTF8Encoding]::new($false))
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

### SL3061：校验、导入、转发 SL3060

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step6b_a07bde7.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step6b_a07bde7.bundle"
git -C "$REPO" fetch "$PWD/step6b_a07bde7.bundle" refs/heads/codex/step6b-a07bde7
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = a07bde7a2a357581505f8d72988839d6401dc222
git -C "$REPO" checkout --detach a07bde7a2a357581505f8d72988839d6401dc222
scp step6b_a07bde7.bundle step6b_a07bde7.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

### SL3060：校验、导入

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step6b_a07bde7.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step6b_a07bde7.bundle"
git -C "$REPO" fetch "$PWD/step6b_a07bde7.bundle" refs/heads/codex/step6b-a07bde7
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = a07bde7a2a357581505f8d72988839d6401dc222
git -C "$REPO" checkout --detach a07bde7a2a357581505f8d72988839d6401dc222
```

然后运行第 1 节命令（任一节点即可，CPU）。全流程无需访问 GitHub。

## 3. 正式运行

本批无正式 GPU 运行命令；B1/B2/B3 都是快验层。正式的 fail-stop 端到端（正式负载、45 s rejoin、timing 运行）在 B 层服务器通过后按总计划 §7 交付。此前写的：`examples/rl/run_e2e_scaling.py --e2e-backend block` 在本批直接退出并说明属于 B 部分；B1（fail-stop 稳态段、单域固定 TP4、无切换的真实运行）的命令随 B 部分交付。

## 4. 顺带修复的生效范围

本清单起，`quick_check.sh` 对所有 step 都先把 env.sh 的解释器放到 PATH 最前并预检 `helpers_cpp`，runner 子进程继承同一 PATH；Step 4 worker 在 `correctness_only` 下只记录 CUDA 重叠不断言。这些修复只存在于本清单及之后的提交里：Step 3/4/5 现有服务器命令仍钉在各自的旧清单，若要让 Step 3/4 两节点快验也带上这两处修复，需要用户决定后重钉它们的 `releases.json` 条目。
