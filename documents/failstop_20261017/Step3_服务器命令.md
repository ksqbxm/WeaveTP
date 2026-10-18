# Step 3 服务器命令

审阅修正后的交付。实现 `7e61596114b003bf3ecefdd8045a6b10aeeda65b`，快验清单 `50b4a781451277cbd472e94ae1253b1331a4dfba`；问题与修正见[审阅与修正](Step3_审阅与修正.md)。此前的 `bcc61ae`/`1c5dea2` 命令与 `step3_1c5dea2.bundle` 作废，不再使用。

2026-10-18 重钉（用户指示）：快验清单改为 `371daf851f19b21cc044a8e04c4bcc897d068b3d`，`releases.json["step3"]` 改指实现 `0f1776a0ff0929114d68318f62e7e2c3d957f508`（含 Step 6 A 部分一并提交的两处顺带修复：Step 4 worker 在 correctness_only 下只记录 CUDA 重叠不断言；`quick_check.sh` 与各 runner 把 env.sh 解释器放到 PATH 最前并预检 `megatron.core.datasets.helpers_cpp`，见 [Step6_实现与验证.md](Step6_实现与验证.md)）。下文全部命令已按新清单与新实现改写，Step 3/4 的检查项、入口与 PASS 标准不变；此前的清单 `50b4a78` 命令与 `step3_50b4a78.bundle` 作废。Step 3/4 共用一个 bundle `step34_371daf8.bundle`（Step 6 的 `step6_df7ad5d.bundle` 不含本清单）。Step 5 仍钉在 `241fb75`/`eae09c2`，未改。

## 1. 快速正确性测试（先完成第 2 节离线导入）

每台一条，先后在 60 秒内启动；也支持先 SL3061。自动协调同一个新 suite，默认共享 GPU，仅用于正确性。两种模式各做一次缩容/扩容，同时保留全部 profile 类别、三类来源、非连续/专家切片、逐位比较、身份、清理和主机库检查。总预算 300 秒，超时 FAIL；本次尚未在服务器实测该预算。

SL3060：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 371daf851f19b21cc044a8e04c4bcc897d068b3d && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 3 --node SL3060
```

SL3061：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 371daf851f19b21cc044a8e04c4bcc897d068b3d && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 3 --node SL3061
```

入口会内部 source env.sh，并按 releases.json 检出实现 `0f1776a0ff0929114d68318f62e7e2c3d957f508`。清单与实现分开，避免 SHA 自引用。后台使用 nohup；启动命令返回只表示启动，不表示测试完成。结束后（包括失败）恢复 `f7d555998b45756998465924d733fd70e553d6b4`。同一检出目录的 Step 2/3 快验共用一把本地锁，不能同时启动。

预期最终单行格式：

```text
QUICK CHECK step=3 result=PASS (emulate shrink/expand, real shrink/expand) shared_gpu_mode=true edp1_rejoin_pause_s=<本次实测非负数值> logs=<本次目录>
```

两机 exit_code 均须为 0，receipt.result 均须为 PASS；quick_mode=true、correctness_only=true、shared_gpu_mode=true、timing_valid=false、accepted=false。任一模式、对端、打包、恢复或清理失败都不能 PASS。指标缺失/NaN 也失败（自 `28ab6dd320f64d9b595e691252fc8ab56ccbbc70` 起所有 step 统一：指标只由 SL3060 计算与校验，SL3061 的结论行打印 None；本页固定的 Step 3 清单 `371daf8` 仍是两节点都携带该指标的旧路径）；不以停顿时长阈值判定正确性。`edp1_rejoin_pause_s` 是 real 的 EDP1 从重建前最后一次正确服务到新 16 rank 组验证后第一次正确服务的最大间隔，排除后台启动/CUDA 初始化。完整同步迁移探针间隙另见 events_actual.json 的 `edp1_rejoin_validation_gap_s`。

查看最新结果（SL3061 使用 NODE=SL3061）：

```bash
NODE=SL3060
ROOT=/data/ubuntu/lxh/weavetp/acceptance/step03_sources_and_transport
LATEST=$(find "$ROOT/kickoff" -maxdepth 1 -type d -name "${NODE}_*" | sort | tail -n 1)
tail -n 60 "$LATEST/nohup.log"
test ! -f "$LATEST/exit_code" || cat "$LATEST/exit_code"
test ! -f "$LATEST/final.json" || cat "$LATEST/final.json"
```

日志会打印实际 tar.gz 和 receipt 路径，形如 `$ROOT/s3q_..._SL3060.tar.gz` 与同前缀 `.receipt.json`；另一机后缀 SL3061。**回传两台各自的 tar 和 receipt**，失败时另带 kickoff/nohup.log、final.json。包内 quick_status.result=PENDING_FINALIZATION 是设计如此，最终以包外 receipt、对端终态和 exit_code 为准。profile.json、profile.sha256、source_plan、links、host_library/host_integrity/host_worker、进程生命周期、CUDA/MPS 原文及逐位结果均在各 launch 子目录。worker 的 `host_registration_preflight` 事件记录共享文件页注册探针；`host_library.json` 的 `memlock_bytes` 只是证据，不构成拒绝条件。

## 2. Windows 离线 bundle 与两节点导入

本轮已生成并验证：`C:/Users/ksqbx/.codex/worktrees/failstop/step34_371daf8.bundle`，1220144 字节，SHA256 `6d06fa5a2a5eec59358c1641e816d5d25d2214be5ca7d5311c56a334914a75e2`，配套 `.sha256` 在同目录。基线严格为 `f7d555998b45756998465924d733fd70e553d6b4`，目标为 Step 3/4 共用的快验清单提交 `371daf851f19b21cc044a8e04c4bcc897d068b3d`，包含 Step 1–6 与路由确定性检查的全部提交。

直接上传已有包（本轮只生成本地包，尚未替用户上传或运行服务器任务）：

```powershell
$TaskBundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step34_371daf8.bundle'
Get-FileHash -LiteralPath $TaskBundle -Algorithm SHA256
scp $TaskBundle "$TaskBundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

如需独立重建包，下面使用新文件名以保留原包；后续校验/转发命令中的文件名也改成 `_rebuilt`：

```powershell
$TaskRepo = 'C:/Users/ksqbx/.codex/worktrees/failstop/WeaveTP_代码与实验资料_20260920'
$TaskBundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step34_371daf8_rebuilt.bundle'
if (Test-Path -LiteralPath $TaskBundle) { throw '文件已存在，不覆盖' }
git -C $TaskRepo update-ref refs/heads/codex/step34-371daf8 371daf851f19b21cc044a8e04c4bcc897d068b3d
git -C $TaskRepo bundle create $TaskBundle refs/heads/codex/step34-371daf8 '^f7d555998b45756998465924d733fd70e553d6b4'
if ($LASTEXITCODE) { throw 'bundle create failed' }
git -C $TaskRepo bundle verify $TaskBundle
if ($LASTEXITCODE) { throw 'bundle verify failed' }
$TaskHash = (Get-FileHash -LiteralPath $TaskBundle -Algorithm SHA256).Hash.ToLowerInvariant()
$TaskName = [System.IO.Path]::GetFileName($TaskBundle)
[System.IO.File]::WriteAllText("$TaskBundle.sha256", "$TaskHash  $TaskName`n", [System.Text.UTF8Encoding]::new($false))
scp $TaskBundle "$TaskBundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

SL3061 校验、导入，再转发 SL3060：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step34_371daf8.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step34_371daf8.bundle"
git -C "$REPO" fetch "$PWD/step34_371daf8.bundle" refs/heads/codex/step34-371daf8
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = 371daf851f19b21cc044a8e04c4bcc897d068b3d
git -C "$REPO" checkout --detach 371daf851f19b21cc044a8e04c4bcc897d068b3d
scp step34_371daf8.bundle step34_371daf8.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

SL3060 校验、导入，不依赖 GitHub：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step34_371daf8.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step34_371daf8.bundle"
git -C "$REPO" fetch "$PWD/step34_371daf8.bundle" refs/heads/codex/step34-371daf8
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = 371daf851f19b21cc044a8e04c4bcc897d068b3d
git -C "$REPO" checkout --detach 371daf851f19b21cc044a8e04c4bcc897d068b3d
```

SL3061 也可从个人远端补取对象：`git -C "$REPO" fetch origin codex/failstop`，随后仍检出上述固定 SHA；SL3060 继续使用 bundle。不要使用浮动分支代替交付 SHA。

## 3. 正式 Step 3 验收

先完成快验，按 Step 2 real 的 GPU 回传关口安排 real 验收。以下命令默认独占，emulate 一个 launch 连续 5 轮、real 一个独立 launch；端口 29840/29841，不与快验同时启动。入口自动启动 TCPStore、控制器、两节点各自的监管器和 CPU host owner，并执行 /data、MPS、worker 身份、RAM、/dev/shm 预算与共享文件页注册探针预检。默认 allocator；不会停止常驻 MPS 或其他任务。

默认 `--host-library-mode on-demand`；完整库检查时两机都改成 `--host-library-mode full`，并使用新的共同 SUITE。这里的库是 Step 3 小型切片 checkpoint，不代表 native DeepSeek 整模型内存已验收。需要共享多轮正确性测试时两机都加 `--correctness-only`（Step 2/3 唯一的测试专用共享 GPU 开关），此时 accepted=false、timing_valid=false，不能作正式性能数据。

SL3060：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" checkout --detach 0f1776a0ff0929114d68318f62e7e2c3d957f508
export NODE=SL3060 SUITE=s3_formal_371daf8_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step03_sources_and_transport/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_sources.py"
export REPO PYTHON
test ! -e "$RUN"
mkdir -p "$RUN"
nohup bash -c 'rc=1; finish() { git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4 || rc=1; tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")" || rc=1; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"; }; trap finish EXIT; "$PYTHON" -B "$ENTRY" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --mode both --rounds 5 --real-launches 1 --host-library-mode on-demand; rc=$?' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

SL3061：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" checkout --detach 0f1776a0ff0929114d68318f62e7e2c3d957f508
export NODE=SL3061 SUITE=s3_formal_371daf8_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step03_sources_and_transport/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_sources.py"
export REPO PYTHON
test ! -e "$RUN"
mkdir -p "$RUN"
nohup bash -c 'rc=1; finish() { git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4 || rc=1; tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")" || rc=1; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"; }; trap finish EXIT; "$PYTHON" -B "$ENTRY" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --mode both --rounds 5 --real-launches 1 --host-library-mode on-demand; rc=$?' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

每次重跑两机共同使用新 SUITE，已有目录拒绝覆盖。先只验 emulate 时，两机均改 `--mode emulate`；real 结果保持 pending。只验 real 时用 `--mode real`，不伪造两模式完整验收。

查看及回传：

```bash
tail -n 60 "$RUN/$NODE.nohup.log"
test ! -f "$RUN/$NODE.exit_code" || cat "$RUN/$NODE.exit_code"
test ! -f "$RUN/$NODE/summary.json" || cat "$RUN/$NODE/summary.json"
printf '%s\n' "${RUN}_${NODE}.tar.gz"
```

两节点退出码须为 0；完整独占矩阵才可 accepted=true。核对每轮 bitwise、normalized_replay_equal、host_integrity.unchanged；新 PID 的 initial_h2d_bytes/sent_bytes=0；逐链路接收 payload+reuse 总量与独立目标字节一致；每个 profile 均有 16 卡 H2D、两节点 4/8 并发及 240 条 P2P。owner PID/backing/inode 跨故障不变，正常完成后每 worker 注销、owner unlink。失败卡无数据/ring 参与、无旧 work、无残留登记进程。任何缺项、预算不足、注册失败、内容不等、挂起/超时或清理未确认都失败。

## 4. 独立 profile 采集

在上方两节点正式命令中，同步使用 `SUITE=s3_profile_371daf8_r1`，ENTRY 不变，只把 Python 参数改成下面对应一行，保留相同 nohup、打包和恢复 wrapper：

SL3060 的 runner 参数：

```bash
"$PYTHON" -B "$ENTRY" --node SL3060 --suite-id s3_profile_371daf8_r1 --output /data/ubuntu/lxh/weavetp/acceptance/step03_sources_and_transport/s3_profile_371daf8_r1 --mode emulate --profile-only --port-base 29860 --host-library-mode on-demand
```

SL3061 的 runner 参数：

```bash
"$PYTHON" -B "$ENTRY" --node SL3061 --suite-id s3_profile_371daf8_r1 --output /data/ubuntu/lxh/weavetp/acceptance/step03_sources_and_transport/s3_profile_371daf8_r1 --mode emulate --profile-only --port-base 29860 --host-library-mode on-demand
```

`--profile-only` 只采集初始 world16 profile，不 kill、不缩扩容，轮数固定为 1，accepted=false、profile_only=true、edp1_rejoin_pause_s=null。在 profile.json 所在目录执行 `sha256sum -c profile.sha256`；两机对应 launch 的 hash 必须相同。GPU 共享采样须加 `--correctness-only`，不能作为独占性能 profile。

实现与 CPU 证据见 [Step3_实现与验证.md](Step3_实现与验证.md) 和 [审阅与修正](Step3_审阅与修正.md)。本次没有替用户执行服务器命令；Step 4 仍等待 fix2 GPU 回传确认。
