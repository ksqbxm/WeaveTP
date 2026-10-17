# Step 2 服务器命令

当前为审阅修正版，见[修正报告](Step2_审阅与修正.md)。先用下方默认共享 GPU 的快验；本轮尚无服务器 GPU 通过结论。

## 1. 快速正确性测试（先完成下方离线导入）

SL3060 执行一行，随后 60 秒内在 SL3061 执行对应一行。也支持先启动 SL3061。每次自动生成新 suite，无须手改名称。默认开启测试专用 `--correctness-only`、允许共享 GPU；两种模式各一次缩容和扩容，仍执行全部逐位、身份和清理检查。

SL3060：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach df4d0096243024ca1a1f92447181362e1411f885 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 2 --node SL3060
```

SL3061：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach df4d0096243024ca1a1f92447181362e1411f885 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 2 --node SL3061
```

入口内部再次 source env.sh，并按 releases.json 检出固定实现 `fa152b134107f3efdf4335c2a99752d02b61bcf0`。清单提交与实现提交分开，避免提交 SHA 自引用。结束后无论成功失败都切回 `f7d555998b45756998465924d733fd70e553d6b4`；再次快验使用上面同一行即可。

启动命令打印实际 nohup.log 路径。启动命令返回只表示后台任务已创建；测试退出码保存在同目录 exit_code。完成时日志必须包含：

```text
QUICK CHECK step=2 result=PASS (emulate shrink/expand, real shrink/expand) shared_gpu_mode=true
```

两边最终 receipt.result 都必须为 PASS，exit_code 都为 0，quick_mode=true、correctness_only=true、shared_gpu_mode=true、timing_valid=false、accepted=false；超时、包生成、恢复提交、任一模式/对端失败均不能通过。含等待和启动总预算 300 秒；服务器实际能否在预算内完成尚待本轮回传，超时不算通过。此快验不作为正式验收和性能数据。

查看本机最新快验（SL3061 将 NODE 改为 SL3061）：

```bash
NODE=SL3060
ROOT=/data/ubuntu/lxh/weavetp/acceptance/step02_topology_rebind
LATEST=$(find "$ROOT/kickoff" -maxdepth 1 -type d -name "${NODE}_*" | sort | tail -n 1)
tail -n 60 "$LATEST/nohup.log"
test ! -f "$LATEST/exit_code" || cat "$LATEST/exit_code"
test ! -f "$LATEST/final.json" || cat "$LATEST/final.json"
```

结果目录由同一 suite 协调，例如 `$ROOT/s2q_.../SL3060` 和 `$ROOT/s2q_.../SL3061`。每台回传日志中 archive 和 receipt 指出的两个文件：`$ROOT/s2q_..._SL3060.tar.gz`、同前缀 `.receipt.json`（另一台后缀 SL3061）。receipt 含打包、清理、恢复和对端终态，必须与 tar 一起回传。tar 内 `quick_status.result=PENDING_FINALIZATION` 是有意设计，不是最终结论；避免包创建之前过早宣称通过。失败时同时回传 kickoff 中的 nohup.log/final.json。

## 2. Windows 离线 bundle 生成与上传

本轮已生成并验证：`C:/Users/ksqbx/.codex/worktrees/failstop/step2_df4d009.bundle`，807816 字节。配套 `.sha256` 在同目录。bundle 基线严格为原正式提交 f7d555998b45756998465924d733fd70e553d6b4。

已有包可直接上传：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step2_df4d009.bundle'
Get-FileHash -LiteralPath $Bundle -Algorithm SHA256
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

SSH 别名 5090-SL3061 来自本地已存在配置，对应 ubuntu@hdssh01.casdao.com:43061。等价显式上传命令：

```powershell
scp -P 43061 $Bundle "$Bundle.sha256" ubuntu@hdssh01.casdao.com:/data/ubuntu/lxh/weavetp/
```

如需重新生成，在独立的新文件名生成，保留已有证据：

```powershell
$Repo = 'C:/Users/ksqbx/.codex/worktrees/failstop/WeaveTP_代码与实验资料_20260920'
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step2_df4d009_rebuilt.bundle'
if (Test-Path -LiteralPath $Bundle) { throw '文件已存在，请先核对，不覆盖' }
git -C $Repo update-ref refs/heads/codex/step2-df4d009 df4d0096243024ca1a1f92447181362e1411f885
git -C $Repo bundle create $Bundle refs/heads/codex/step2-df4d009 '^f7d555998b45756998465924d733fd70e553d6b4'
if ($LASTEXITCODE) { throw 'bundle create failed' }
git -C $Repo bundle verify $Bundle
if ($LASTEXITCODE) { throw 'bundle verify failed' }
$Hash = (Get-FileHash -LiteralPath $Bundle -Algorithm SHA256).Hash.ToLowerInvariant()
$Name = [System.IO.Path]::GetFileName($Bundle)
[System.IO.File]::WriteAllText("$Bundle.sha256", "$Hash  $Name`n", [System.Text.UTF8Encoding]::new($false))
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

重建包使用 `_rebuilt` 文件名时，下面导入/转发命令中的文件名相应替换。正常使用已生成原包，其 SHA256 为 `86ceb7234aeb1b10409f8ccadaf60311f2ae21d13e22a66f6628f967b243a7d4`。

### SL3061：校验、导入、转发 SL3060

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step2_df4d009.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step2_df4d009.bundle"
git -C "$REPO" fetch "$PWD/step2_df4d009.bundle" refs/heads/codex/step2-df4d009
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = df4d0096243024ca1a1f92447181362e1411f885
git -C "$REPO" checkout --detach df4d0096243024ca1a1f92447181362e1411f885
scp step2_df4d009.bundle step2_df4d009.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

### SL3060：校验、导入

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step2_df4d009.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step2_df4d009.bundle"
git -C "$REPO" fetch "$PWD/step2_df4d009.bundle" refs/heads/codex/step2-df4d009
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = df4d0096243024ca1a1f92447181362e1411f885
git -C "$REPO" checkout --detach df4d0096243024ca1a1f92447181362e1411f885
```

然后运行第 1 节命令。SL3060 全流程无需访问 GitHub。

## 3. 正式 Step 2 验收

默认独占 GPU，emulate 一次 launch 连续 5 轮，real 一次独立 launch；这是 Step 2 拓扑验收，不是 Step 7 正式吞吐矩阵。每次重跑两机共同使用新 SUITE，旧结果不得覆盖。预检包含 /data 空间、MPS 原始查询、实际 worker 独立 GPU context、PID 登记及两节点配置一致性；busy GPU 会按默认独占规则失败。需要多轮共享正确性测试时，两机的 Python runner 参数都加 `--correctness-only`；此时强制 accepted=false、timing_valid=false，即使全部通过也不能作正式验收或耗时数据。Step 2 原共享参数已删除，不保留兼容别名。

先 SL3060，再 SL3061。两段只在 NODE 不同；端口 29840/29841（不能与快验同时启动）。

SL3060：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" checkout --detach fa152b134107f3efdf4335c2a99752d02b61bcf0
export NODE=SL3060 SUITE=s2_formal_df4d009_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step02_topology_rebind/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_topology.py"
export REPO PYTHON
test ! -e "$RUN"
mkdir -p "$RUN"
nohup bash -c 'rc=1; finish() { git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4 || rc=1; tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")" || rc=1; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"; }; trap finish EXIT; "$PYTHON" -B "$ENTRY" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --mode both --rounds 5 --real-launches 1; rc=$?' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

SL3061：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" checkout --detach fa152b134107f3efdf4335c2a99752d02b61bcf0
export NODE=SL3061 SUITE=s2_formal_df4d009_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step02_topology_rebind/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_topology.py"
export REPO PYTHON
test ! -e "$RUN"
mkdir -p "$RUN"
nohup bash -c 'rc=1; finish() { git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4 || rc=1; tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")" || rc=1; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"; }; trap finish EXIT; "$PYTHON" -B "$ENTRY" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --mode both --rounds 5 --real-launches 1; rc=$?' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

查看和回传：

```bash
tail -n 60 "$RUN/$NODE.nohup.log"
test ! -f "$RUN/$NODE.exit_code" || cat "$RUN/$NODE.exit_code"
test ! -f "$RUN/$NODE/summary.json" || cat "$RUN/$NODE/summary.json"
printf '%s\n' "${RUN}_${NODE}.tar.gz"
```

预期两节点退出码 0，summary.all_cases_passed=true、quick_mode=false，emulate/real 全部 launch 各自 passed；独占完整矩阵 accepted=true，使用 --correctness-only 则 accepted=false；SL3060 的 topology_checks.json 包含每轮三阶段实际 TP/EP/DP 顺序、逐位检查、指针、数据/ring 字节及回归身份。回传两个节点各自 tar.gz 和 exit_code。任何超时、错组、内容或指针变化、失败 GPU 参与数据、回归者提前充当源、清理不确认均失败；原始证据保留。
