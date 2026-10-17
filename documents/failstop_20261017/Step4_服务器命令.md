# Step 4 服务器命令

实现 `28ab6dd320f64d9b595e691252fc8ab56ccbbc70`，快验清单 `02e7bce29bf36ef9707853ef863ae20d85accc79`，见[实现与验证](Step4_实现与验证.md)和[审阅与修正](Step4_审阅与修正.md)。2026-10-18 服务器结果（用户回传，清单 `7c9aaf20fbda914eaa2969ad67fecb70ecb2f471`，suite `s4q_20261017T160602_33f20a9d`）：两节点 emulate 与 real_00 的 result.json、controller_result.json 均 passed，SL3060 报 `edp1_rejoin_pause_s=6.09`（共享 GPU，仅参考）；但 SL3061 的套件汇总报 `KeyError: 'metrics'`，两节点 QUICK CHECK=FAIL。原因：metrics 只存在于 SL3060 控制器的结果中，Step 2/3 曾通过 bus 把 `topology_metrics` 交给 SL3061，Step 4 没有这条交接。修正（实现 `28ab6dd320f64d9b595e691252fc8ab56ccbbc70`，清单 `02e7bce29bf36ef9707853ef863ae20d85accc79`）：删除这条交接，所有 step 统一只由 SL3060 计算并校验 `edp1_rejoin_pause_s`，SL3061 的 summary/receipt 不携带该指标；CPU 测试覆盖对端 summary 缺指标（Step 2/3/4 均 PASS）和汇总函数对缺少 metrics 字段的对端结果（SL3061 返回 None，SL3060 缺失即 KeyError）。137 项 failstop 单测通过。 此前的 `7c9aaf20fbda914eaa2969ad67fecb70ecb2f471` 命令与 `step4_merged_7c9aaf2.bundle` 作废。2026-10-17 合并：`codex/failstop-step4` 已合入 `codex/failstop`（合并提交 `28ab6dd320f64d9b595e691252fc8ab56ccbbc70`，快验清单 `02e7bce29bf36ef9707853ef863ae20d85accc79`）。Step 4 接入 Step 3 的 probe 注册表（`common.ASYNC_PROBE/STEP_PROBES`），`run_async.py` 复用 `run_topology.main(step=4)`，快验沿用 Step 3 的单一握手端口 29839、共享本地锁和必填 `edp1_rejoin_pause_s`（Step 4 real 由 EDP1 阶段 2 报告的 abort/rebuild 暂停给出），并保留 1 GB 快验磁盘门槛与 runner 错误透出；删除 async_probe 中与 topology_probe 重复的 EmulateSupervisor。合并后 117 项 failstop CPU 测试通过；此后统一在 `codex/failstop` 开发。此前的 `e09d7bc` 命令与 `step4_e09d7bc.bundle` 作废。本版包含审阅修正（real 替代进程不再对不存在的进程组调用 abort 等）。此前一版修正首次服务器快验的失败原因：SL3060 `/data` 仅剩 4.88 GB，低于 runner 默认 5 GB 门槛，预检拒绝运行。快验门槛改为默认 1 GB（`quick_check.sh --min-free-disk-gb N` 可调），正式验收保持 5 GB；预检失败时 receipt/nohup 直接写出子进程真实错误（如 `insufficient /data space`），并保存完整 runner 输出到 kickoff 目录的 runner.log。先用下方默认共享 GPU 的快验；本轮尚无服务器 GPU 通过结论。fix2 GPU 复验证据仍未登记，runner 结果固定 `prerequisites_verified=false`。

## 1. 快速正确性测试（先完成下方离线导入）

SL3060 执行一行，随后 60 秒内在 SL3061 执行对应一行；也支持先启动 SL3061。每次自动生成新 suite。默认开启 `--correctness-only`、允许共享 GPU；两种模式各一次缩容和扩容，执行全部逐位、单套权重、EP 序列、重叠和清理检查。

SL3060：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 02e7bce29bf36ef9707853ef863ae20d85accc79 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 4 --node SL3060
```

SL3061：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 02e7bce29bf36ef9707853ef863ae20d85accc79 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 4 --node SL3061
```

入口内部再次 source env.sh，并按 releases.json 检出固定实现 `28ab6dd320f64d9b595e691252fc8ab56ccbbc70`；结束后无论成功失败都切回 `f7d555998b45756998465924d733fd70e553d6b4`。握手端口 29839、runner 端口 29840/29841 与 Step 2/3 共用，同一检出目录的各步快验共用一把本地锁，不能同时启动。

启动命令打印实际 nohup.log 路径；退出码保存在同目录 exit_code。完成时日志必须包含：

```text
QUICK CHECK step=4 result=PASS (emulate shrink/expand, real shrink/expand) shared_gpu_mode=true
```

两边最终 receipt.result 都必须为 PASS，exit_code 都为 0，quick_mode=true、correctness_only=true、shared_gpu_mode=true、timing_valid=false、accepted=false；`edp1_rejoin_pause_s`（real 的 EDP1 abort/rebuild 暂停）只出现在 SL3060 的结论行与 receipt，SL3061 打印 None 属正常；超时、包生成、恢复提交、任一模式/对端失败均不能通过。含等待和启动总预算 300 秒；服务器能否在预算内完成尚待回传，超时不算通过。此快验不作为正式验收和性能数据。

查看本机最新快验（SL3061 将 NODE 改为 SL3061）：

```bash
NODE=SL3060
ROOT=/data/ubuntu/lxh/weavetp/acceptance/step04_async_driver_and_memory
LATEST=$(find "$ROOT/kickoff" -maxdepth 1 -type d -name "${NODE}_*" | sort | tail -n 1)
tail -n 60 "$LATEST/nohup.log"
test ! -f "$LATEST/exit_code" || cat "$LATEST/exit_code"
test ! -f "$LATEST/final.json" || cat "$LATEST/final.json"
```

结果目录例如 `$ROOT/s4q_.../SL3060` 和 `$ROOT/s4q_.../SL3061`。每台回传日志中 archive 和 receipt 指出的两个文件：`$ROOT/s4q_..._SL3060.tar.gz` 与同前缀 `.receipt.json`（另一台后缀 SL3061）。tar 内 `quick_status.result=PENDING_FINALIZATION` 是有意设计，不是最终结论。失败时同时回传 kickoff 中的 nohup.log/final.json，以及 `s4q_.../<NODE>/<launch>/` 下的 `worker_error_*.json`、`controller_result.json`。

## 2. Windows 离线 bundle 生成与上传

本轮已生成并验证：`C:/Users/ksqbx/.codex/worktrees/failstop/step4_metrics_02e7bce.bundle`，1032019 字节，SHA256 `625d7148b7c153026dfb94b03b6fd75d7d5b85f28ec2ae0fa417e2ada9aa0381`。配套 `.sha256` 在同目录。bundle 基线严格为 `f7d555998b45756998465924d733fd70e553d6b4`，包含 Step 1–4 与路由确定性检查的全部提交。

上传：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step4_metrics_02e7bce.bundle'
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
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step4_metrics_02e7bce_rebuilt.bundle'
if (Test-Path -LiteralPath $Bundle) { throw '文件已存在，请先核对，不覆盖' }
git -C $Repo update-ref refs/heads/codex/step4-metrics-02e7bce 02e7bce29bf36ef9707853ef863ae20d85accc79
git -C $Repo bundle create $Bundle refs/heads/codex/step4-metrics-02e7bce '^f7d555998b45756998465924d733fd70e553d6b4'
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
sha256sum -c step4_metrics_02e7bce.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step4_metrics_02e7bce.bundle"
git -C "$REPO" fetch "$PWD/step4_metrics_02e7bce.bundle" refs/heads/codex/step4-metrics-02e7bce
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = 02e7bce29bf36ef9707853ef863ae20d85accc79
git -C "$REPO" checkout --detach 02e7bce29bf36ef9707853ef863ae20d85accc79
scp step4_metrics_02e7bce.bundle step4_metrics_02e7bce.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

### SL3060：校验、导入

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step4_metrics_02e7bce.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step4_metrics_02e7bce.bundle"
git -C "$REPO" fetch "$PWD/step4_metrics_02e7bce.bundle" refs/heads/codex/step4-metrics-02e7bce
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = 02e7bce29bf36ef9707853ef863ae20d85accc79
git -C "$REPO" checkout --detach 02e7bce29bf36ef9707853ef863ae20d85accc79
```

然后运行第 1 节命令。SL3060 全流程无需访问 GitHub。

## 3. 正式 Step 4 验收

默认独占 GPU，emulate 一次 launch 连续 5 轮，real 一次独立 launch。每次重跑两机共同使用新 SUITE。预检与 Step 2 相同；busy GPU 会按默认独占规则失败。需要共享多轮正确性测试时两机都加 `--correctness-only`，此时 timing_valid=false。无论哪种方式，本步 summary 的 `accepted` 固定为 false（fix2 前置与 Step 3 集成另行验收），独占完整矩阵的机制结论写在 `mechanism_matrix_passed`。

先 SL3060，再 SL3061。两段只在 NODE 不同；端口 29840/29841 与 Step 2/3 正式验收相同（不能与快验或其他步同时启动）。

SL3060：

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" checkout --detach 28ab6dd320f64d9b595e691252fc8ab56ccbbc70
export NODE=SL3060 SUITE=s4_formal_02e7bce_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step04_async_driver_and_memory/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_async.py"
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
git -C "$REPO" checkout --detach 28ab6dd320f64d9b595e691252fc8ab56ccbbc70
export NODE=SL3061 SUITE=s4_formal_02e7bce_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step04_async_driver_and_memory/$SUITE"
export ENTRY="$REPO/code/current/tools/resharding/failstop/run_async.py"
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

预期两节点退出码 0，summary.all_cases_passed=true、quick_mode=false、emulate/real 均 passed、prerequisites_verified=false、accepted=false；独占完整矩阵 mechanism_matrix_passed=true。每个 launch 目录 `$RUN/$NODE/<launch>/` 含：`async_checks_<slot>.json`（逐步服务样本、EDP1 对无故障基线的干扰分布、各阶段内存样本、扩容期间 CUDA 计算/传输区间与重叠、逐链路传输、real 暂停区间、切换峰值预算）、`async_phase_checks.json`（每阶段 PID/映射/EP 序列/历史审计）、`rank_map.json`、`async_<slot>_<epoch>.jsonl`、`controller_result.json`。回传两个节点各自 tar.gz 和 exit_code。

失败标准：任何错序、死锁或超时、错误 token（逐位不一致）、预算越界、稳态双套权重或释放后 storage 仍非零、无测量到的计算/传输重叠、real 回归后 TP2 坐标变化、静默暂停、清理不确认。EDP1 性能分布变化单独报告，由用户判断，不写成“性能无影响”。
