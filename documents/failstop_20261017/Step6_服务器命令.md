# Step 6 服务器命令（A 部分：fake 后端快验，CPU）

实现 `0f1776a0ff0929114d68318f62e7e2c3d957f508`，快验清单 `df7ad5d33364fbd481c82c8164aad093ce984d54`（`releases.json["step6"]`），见[实现与验证](Step6_实现与验证.md)。本批不含任何 GPU 运行：快验在一个节点上用 CPU 跑 fake 后端完整循环（扩缩容 2→4、4→2 与 fail-stop fail→replay→expand）加 `tests/unit_tests/e2e` 的 CPU 测试，`accepted=false`、`timing_valid=false`。它不占 GPU，不依赖 Step 3/4/5 的上机顺序，可随时运行；真实模型的 B 部分（B1 单域固定 TP4 → B2 → B3）等阶段 A 服务器验收后另行交付。

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

## 2. Windows 离线 bundle 生成与上传

本轮已生成并验证：`C:/Users/ksqbx/.codex/worktrees/failstop/step6_df7ad5d.bundle`，1216667 字节，SHA256 `f02e68ece3636bf3f7d38b27fca680bd96c4032e00d49ce735fcf89d8b081098`，配套 `.sha256` 在同目录。基线严格为 `f7d555998b45756998465924d733fd70e553d6b4`，目标为快验清单提交 `df7ad5d33364fbd481c82c8164aad093ce984d54`，包含 Step 1–6 与路由确定性检查的全部提交。

上传：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step6_df7ad5d.bundle'
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
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/step6_df7ad5d_rebuilt.bundle'
if (Test-Path -LiteralPath $Bundle) { throw '文件已存在，请先核对，不覆盖' }
git -C $Repo update-ref refs/heads/codex/step6-df7ad5d df7ad5d33364fbd481c82c8164aad093ce984d54
git -C $Repo bundle create $Bundle refs/heads/codex/step6-df7ad5d '^f7d555998b45756998465924d733fd70e553d6b4'
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
sha256sum -c step6_df7ad5d.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step6_df7ad5d.bundle"
git -C "$REPO" fetch "$PWD/step6_df7ad5d.bundle" refs/heads/codex/step6-df7ad5d
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = df7ad5d33364fbd481c82c8164aad093ce984d54
git -C "$REPO" checkout --detach df7ad5d33364fbd481c82c8164aad093ce984d54
scp step6_df7ad5d.bundle step6_df7ad5d.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

### SL3060：校验、导入

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step6_df7ad5d.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/step6_df7ad5d.bundle"
git -C "$REPO" fetch "$PWD/step6_df7ad5d.bundle" refs/heads/codex/step6-df7ad5d
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = df7ad5d33364fbd481c82c8164aad093ce984d54
git -C "$REPO" checkout --detach df7ad5d33364fbd481c82c8164aad093ce984d54
```

然后运行第 1 节命令（任一节点即可，CPU）。全流程无需访问 GitHub。

## 3. 正式运行

本批无正式 GPU 运行命令。`examples/rl/run_e2e_scaling.py --e2e-backend block` 在本批直接退出并说明属于 B 部分；B1（fail-stop 稳态段、单域固定 TP4、无切换的真实运行）的命令随 B 部分交付。

## 4. 顺带修复的生效范围

本清单起，`quick_check.sh` 对所有 step 都先把 env.sh 的解释器放到 PATH 最前并预检 `helpers_cpp`，runner 子进程继承同一 PATH；Step 4 worker 在 `correctness_only` 下只记录 CUDA 重叠不断言。这些修复只存在于本清单及之后的提交里：Step 3/4/5 现有服务器命令仍钉在各自的旧清单，若要让 Step 3/4 两节点快验也带上这两处修复，需要用户决定后重钉它们的 `releases.json` 条目。
