# Step 5 服务器命令（第一批）

审阅修正后的交付。实现 `2fe4a9a0265580a47d0753dfd2b839e4bdb01f05`，快验清单 `f414fdc7c834b7935bc78aa51da74c635c37090a`（`releases.json["step5"]`），见[实现与验证](Step5_实现与验证.md)与[审阅与修正](Step5_审阅与修正.md)。此前的 `9405ed6`/`98c498b` 命令与 `step5_98c498b.bundle` 作废，不再使用。本批交付 §2.1 确定性模式合同、显存预检、真实模型的参考/候选/默认模式计时三类 torchrun 与跨模式迁移计划比较工具；**真实模型的 fail→shrink→rejoin→expand、A6/A7、smoke 与多轮尚未实现**，本批 summary 固定 `accepted=false`、`timing_valid=false`、`candidate_kind=fresh_checkpoint_load`、`fail_stop_integration=pending`。上机顺序：先完成 Step 3、Step 4 的快验，再运行本步。2026-10-18 用户回传：两节点快验 suite `s5q_20261017T162447_15950af2` 在 SL3060 被显存预检拦下（`tp4 run GPU 7 free 2.72 GiB < required 5.98 GiB`，被其他用户占用，非代码问题）；当前批次只需一个节点的 8 张卡，因此新增单节点入口（第 1 节），两节点入口保留为第 1b 节。本轮没有服务器 GPU 结论。

## 1. 单节点快速正确性测试（先完成第 2 节离线导入）

一个节点一条命令，不等待对端。该节点依次运行 tp2（前 4 卡）与 tp4（8 卡）各 3 个真实模型 torchrun：确定性模式参考（`--deterministic-mode` + `NCCL_ALGO=Ring` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`）、确定性模式候选（独立重新加载，与参考逐步逐位比较）、默认模式计时（不设上述参数与环境变量，只记录逐步 GPU 耗时）。共 6 次 59 GB checkpoint 加载，预算 600 秒（两节点模式每节点 3 次加载、300 秒；单节点把两节点的工作串行到一台机器，不放宽任何检查项）。SL3061 示例，换节点只改 `--node`：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach f414fdc7c834b7935bc78aa51da74c635c37090a && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 5 --node SL3061 --single-node
```

入口内部再次 source env.sh，按 releases.json 检出固定实现 `2fe4a9a0265580a47d0753dfd2b839e4bdb01f05`；结束后无论成功失败都切回 `f7d555998b45756998465924d733fd70e553d6b4`。同一检出目录的各步快验共用一把本地锁；torchrun 使用 `--standalone` 本机 rendezvous，本节点不能同时有其他 `--standalone` torchrun。checkpoint 路径取 env.sh 的 `CHECKPOINT`，缺省 `/data/models/DeepSeek-V2-Lite-megatron-v2`。

单节点版每卡显存需求（max_seq 128，余量 2 GiB；两布局顺序执行，不叠加）：

| 卡 | 承担 | 需求/卡 | 组成 |
|---|---|---|---|
| GPU 0–3 | tp2 三次运行 + tp4 三次运行 | **9.94 GiB** | TP2 权重 7.93 + KV 0.02 + 余量 2 |
| GPU 4–7 | 仅 tp4 三次运行 | **5.98 GiB** | TP4 权重 3.97 + KV 0.01 + 余量 2 |

启动后先打印显存预检表并以 `this launch (tp2, tp4) requires per GPU: GPU 0 9.94 GiB, …, GPU 7 5.98 GiB` 给出逐卡需求；任一卡不足即在任何 torchrun 之前失败并指出卡号。16 卡 fail-stop 场景行（存活卡 21.85 GiB 等）同时记录在 `memory_preflight.json`，单节点快验不以其为拒绝条件。完成时日志必须包含：

```text
QUICK CHECK step=5 result=PASS (memory preflight, deterministic reference/candidate ['tp2', 'tp4'], step_time_ratio={'tp2': <比值>, 'tp4': <比值>}) shared_gpu_mode=true single_node=true fail_stop_integration=pending
```

receipt.result 为 PASS、exit_code 为 0，quick_mode=true、correctness_only=true、shared_gpu_mode=true、timing_valid=false、accepted=false、single_node=true、budget_s=600，两个布局都有正有限的 `deterministic_step_time_ratio`。失败标准与第 1b 节相同（manifest 非确定性模式、参数校验和或历史不一致、任一步 logits 位差异、timing 非默认模式、超时、打包或恢复提交失败）。结果目录为 `$ROOT/s5q_..._single/<NODE>`，查看命令与回传文件同第 1b 节（只有本节点一份 tar 与 receipt）。

## 1b. 两节点快速正确性测试（真实模型 fail-stop 接入后的形式）

两节点各一条命令，60 秒内先后启动（谁先都可以），脚本自动协调同一 suite。每节点依次运行 3 个真实模型 torchrun：确定性模式参考（`--deterministic-mode` + `NCCL_ALGO=Ring` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`）、确定性模式候选（独立重新加载，与参考逐步逐位比较）、默认模式计时（不设上述参数与环境变量，只记录逐步 GPU 耗时）。快验布局分配：SL3060 跑 tp4（8 卡），SL3061 跑 tp2（前 4 卡）；两布局合计覆盖，每节点 3 次 59 GB checkpoint 加载。总预算 300 秒；路由确定性检查两次加载实测 123 秒，本入口是否在预算内尚未实测，超时即 FAIL。

SL3060：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach f414fdc7c834b7935bc78aa51da74c635c37090a && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 5 --node SL3060
```

SL3061：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach f414fdc7c834b7935bc78aa51da74c635c37090a && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 5 --node SL3061
```

入口内部再次 source env.sh，按 releases.json 检出固定实现 `2fe4a9a0265580a47d0753dfd2b839e4bdb01f05`；结束后无论成功失败都切回 `f7d555998b45756998465924d733fd70e553d6b4`。握手端口 29839 与 Step 2–4 共用，同一检出目录的各步快验共用一把本地锁，不能同时启动；torchrun 使用 `--standalone` 本机 rendezvous，本节点不能同时有其他 `--standalone` torchrun。checkpoint 路径取 env.sh 的 `CHECKPOINT`，缺省 `/data/models/DeepSeek-V2-Lite-megatron-v2`。快验磁盘门槛仍为 1 GB（`--min-free-disk-gb N` 可调）。

启动后先打印显存预检全表（16 卡角色、约束窗口、需求、本机空闲、是否足够），本次运行所用卡显存不足即在任何 torchrun 之前失败并指出卡号与需求。完成时日志必须包含：

```text
QUICK CHECK step=5 result=PASS (memory preflight, deterministic reference/candidate ['tp4'], step_time_ratio={'tp4': <比值>}) shared_gpu_mode=true fail_stop_integration=pending
```

SL3061 一行中布局为 `['tp2']`。两边最终 receipt.result 都必须为 PASS、exit_code 都为 0、quick_mode=true、correctness_only=true、shared_gpu_mode=true、timing_valid=false、accepted=false，且每个布局都有正有限的 `deterministic_step_time_ratio`。参考或候选任一 rank 的 manifest 不是确定性模式、参数校验和或历史不一致、任一步 logits 位不同、timing 进程不是默认模式、超时、打包或恢复提交失败、对端失败，均不能通过。耗时比只作确定性模式开销评估，不是性能结论。此快验不作为正式验收。

查看本机最新快验（SL3061 将 NODE 改为 SL3061）：

```bash
NODE=SL3060
ROOT=/data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness
LATEST=$(find "$ROOT/kickoff" -maxdepth 1 -type d -name "${NODE}_*" | sort | tail -n 1)
tail -n 80 "$LATEST/nohup.log"
test ! -f "$LATEST/exit_code" || cat "$LATEST/exit_code"
test ! -f "$LATEST/final.json" || cat "$LATEST/final.json"
```

结果目录形如 `$ROOT/s5q_.../SL3060`：`memory_preflight.json`（全表与缺口）、`<layout>/<role>/command.txt`、`torchrun.log`、`result.json`（每 rank manifest、参数校验和、逐步耗时、峰值/预留显存；candidate 另有逐步比较与首个差异）、`<layout>/reference|candidate/rank_<r>.pt`（原 dtype logits 位记录）、`model_checks_<layout>.json`、`summary.json`。回传每台的 `$ROOT/s5q_..._<NODE>.tar.gz` 与同前缀 `.receipt.json`；失败时另带 kickoff 的 nohup.log/final.json 与 `error.txt`。tar 内 `quick_status.result=PENDING_FINALIZATION` 是有意设计。

## 2. Windows 离线 bundle 生成与上传

本轮已生成并验证：`C:/Users/ksqbx/.codex/worktrees/failstop/step5_f414fdc.bundle`，1027179 字节，SHA256 `811e14d5a96162190c1db6f63915cb48f2c6964a754ffd93c8de9462b16c9e5c`，配套 `.sha256` 在同目录。基线严格为 `f7d555998b45756998465924d733fd70e553d6b4`，目标为快验清单提交 `f414fdc7c834b7935bc78aa51da74c635c37090a`，包含 Step 1–5 与路由确定性检查的全部提交。

上传：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step5_f414fdc.bundle'
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
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step5_f414fdc_rebuilt.bundle'
if (Test-Path -LiteralPath $Bundle) { throw '文件已存在，请先核对，不覆盖' }
git -C $Repo update-ref refs/heads/codex/step5-f414fdc f414fdc7c834b7935bc78aa51da74c635c37090a
git -C $Repo bundle create $Bundle refs/heads/codex/step5-f414fdc '^f7d555998b45756998465924d733fd70e553d6b4'
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
sha256sum -c step5_f414fdc.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step5_f414fdc.bundle"
git -C "$REPO" fetch "$PWD/step5_f414fdc.bundle" refs/heads/codex/step5-f414fdc
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = f414fdc7c834b7935bc78aa51da74c635c37090a
git -C "$REPO" checkout --detach f414fdc7c834b7935bc78aa51da74c635c37090a
scp step5_f414fdc.bundle step5_f414fdc.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

### SL3060：校验、导入

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step5_f414fdc.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step5_f414fdc.bundle"
git -C "$REPO" fetch "$PWD/step5_f414fdc.bundle" refs/heads/codex/step5-f414fdc
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = f414fdc7c834b7935bc78aa51da74c635c37090a
git -C "$REPO" checkout --detach f414fdc7c834b7935bc78aa51da74c635c37090a
```

然后运行第 1 节命令。SL3060 全流程无需访问 GitHub。

## 3. 每卡显存需求（真实模型，预检表口径）

预检按与 `model_provider` 相同形状公式给出精确参数字节，KV 为 max_seq 128 的静态缓存，余量 2 GiB；接收/发送缓冲取"一个波次全部远程"的上界。cuBLAS 工作区、NCCL 缓冲、MoE all-to-all 临时量、注意力分数、allocator 预留不作解析估算，由 `result.json` 的 memory 字段运行后记录。

| 节点 / 卡 | 角色 | 约束窗口 | 需求/卡 |
|---|---|---|---|
| SL3061 GPU 2,3,6,7（EDP0 存活，etp 2,3） | TP4↔TP2 双布局共存 | 缩容：TP4 源 3.97 + TP2 目标 7.93 + KV + 接收 ≤7.93 + 余量 2 | **21.85 GiB**（扩容窗口 17.89） |
| SL3061 GPU 0,1,4,5（EDP0 故障/回归，etp 0,1） | 空 TP4 目标 | 扩容：3.97 + KV + 接收 ≤3.97 + 余量 2 | 9.94 GiB |
| SL3060 GPU 0–7（EDP1） | TP4 稳态源 | 3.97 + KV + 发送打包 ≤3.97 + 余量 2 | 9.94 GiB |
| 本批参考/候选/计时运行 | tp2 4 卡 / tp4 8 卡，顺序执行 | 7.93 + 0.02 + 2 / 3.97 + 0.01 + 2 | 9.94 / 5.98 GiB |

本批快验只强制本次运行所用布局的参考运行需求（SL3060 tp4 8 卡各 5.98 GiB；SL3061 tp2 前 4 卡各 9.94 GiB）；双布局共存行在 `memory_preflight.json` 的 `scenario` 与 `scenario_sufficient_on_this_node` 中记录，供集成真实模型 fail-stop 时作为拒绝条件。

## 4. 正式运行（本批范围：三类 torchrun，两布局）

默认独占 GPU。SL3061 依次跑 tp2、tp4（6 次加载），SL3060 跑 tp4；每节点独立，不需对端。两段只在 NODE 不同。本批无论结果 `accepted` 固定为 false。

SL3060：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" checkout --detach 2fe4a9a0265580a47d0753dfd2b839e4bdb01f05
export NODE=SL3060 SUITE=s5_formal_f414fdc_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_model.py"
export REPO PYTHON
test ! -e "$RUN"
mkdir -p "$RUN"
nohup bash -c 'rc=1; finish() { git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4 || rc=1; tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")" || rc=1; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"; }; trap finish EXIT; "$PYTHON" -B "$ENTRY" --node "$NODE" --suite-id "$SUITE" --output "$RUN"; rc=$?' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

SL3061：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" checkout --detach 2fe4a9a0265580a47d0753dfd2b839e4bdb01f05
export NODE=SL3061 SUITE=s5_formal_f414fdc_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_model.py"
export REPO PYTHON
test ! -e "$RUN"
mkdir -p "$RUN"
nohup bash -c 'rc=1; finish() { git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4 || rc=1; tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")" || rc=1; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"; }; trap finish EXIT; "$PYTHON" -B "$ENTRY" --node "$NODE" --suite-id "$SUITE" --output "$RUN"; rc=$?' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

查看和回传：

```bash
tail -n 80 "$RUN/$NODE.nohup.log"
test ! -f "$RUN/$NODE.exit_code" || cat "$RUN/$NODE.exit_code"
test ! -f "$RUN/$NODE/summary.json" || cat "$RUN/$NODE/summary.json"
printf '%s\n' "${RUN}_${NODE}.tar.gz"
```

预期退出码 0，summary.all_checks_passed=true、quick_mode=false、accepted=false、timing_valid=false，`layouts.<layout>.checks` 全为 true，`deterministic_step_time_ratio.<layout>` 含 prefill/decode 中位数比值。失败标准：任一 manifest 非确定性模式、参数校验和/历史不一致、任一步 logits 位差异（`first_mismatch` 给出 rank、步、位置、位模式）、timing 非默认模式、显存预检缺口、超时。

## 5. 迁移计划跨模式一致性（待 Step 3 GPU 通过后运行）

§2.1 第 2 条的服务器验证需要两次使用同一冻结 profile 的 Step 3 launch，一次默认环境、一次确定性环境（Step 3 玩具 runner 没有 Megatron 模型，模式差异即 `NCCL_ALGO`/`CUBLAS_WORKSPACE_CONFIG` 环境；真实模型集成后再加 `--deterministic-mode`）。步骤：

1. 用 Step 3 正式入口加 `--profile-only` 采集一次 profile（两节点同 SUITE），得到 `$RUN/<NODE>/<SUITE>_emulate/profile.json` 与 `profile.sha256`。
2. 两节点各复制该 profile.json 到 `/data/ubuntu/lxh/weavetp/profiles/step3_<SUITE>.json`（两机内容必须相同，`sha256sum` 核对）。
3. 默认环境 launch：Step 3 正式命令加 `--profile-file /data/ubuntu/lxh/weavetp/profiles/step3_<SUITE>.json`，SUITE 记为 `s3_plan_default_r1`。
4. 确定性环境 launch：同上，SUITE 记为 `s3_plan_det_r1`，两节点启动前 `export NCCL_ALGO=Ring CUBLAS_WORKSPACE_CONFIG=:4096:8`。
5. 在写出计划文件的节点（SL3060，slot 0）比较：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
ROOT=/data/ubuntu/lxh/weavetp/acceptance/step03_sources_and_transport
"$PYTHON" -B "$REPO/code/current/tools/resharding/failstop/run_model.py" --compare-plans \
  "$ROOT/s3_plan_default_r1/SL3060/s3_plan_default_r1_emulate" "$ROOT/s3_plan_det_r1/SL3060/s3_plan_det_r1_emulate"
```

退出码 0 且输出 `"equal": true` 表示每个 `source_plan_<round>_<phase>.json` 的归一化任务（稳定源/目标身份、切片、字节、波次）逐项相同、profile hash 相同、输入快照相同；`first_difference` 指出首个不同任务与字段。real 模式 launch 同法比较 `<SUITE>_real_00` 目录。此检查结果连同两次 launch 的 tar 一起回传；本批快验不包含此项，summary 标记 `plan_consistency.status=pending`。

## 6. 失败与下一关口

本批通过只表示：预检表与真实空闲显存一致、Megatron `--deterministic-mode` 下 DeepSeek-V2-Lite 两次独立加载在同输入同历史下逐位一致、默认模式耗时基线可得。它不是 A6/A7 通过，不改变 Step 5 "停在 A6/A7 用户确认关口" 的要求。下一批在 Step 3/4 GPU 通过后接入真实模型的 fail→shrink→rejoin→expand，把本批的参考记录与 `compare_records` 用于每个 A6 检查点。
