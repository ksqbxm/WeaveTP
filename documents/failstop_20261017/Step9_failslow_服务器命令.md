# fail-slow 快验（Step 9）：服务器命令

交付：实现 A `1ccac7804105af6b9d70ee93df78f066a8b32e37`，清单 B `141050fe79708a848b7eeb9371c84911337d9a0f`（`releases.json` 的 `step6` 与 `failslow` 都指向 A）。本文件在 B 的子提交 C 里，C 随 bundle `failslow_141050f.bundle` 交付（基线 `f7d555998b45756998465924d733fd70e553d6b4`，ref `refs/heads/codex/failslow-141050f` 指向 C），bundle 的 SHA256 在同名 `.sha256` 文件里。改动见 [09_failslow主图.md](09_failslow主图.md) §9；CPU 回归 295 项（294 通过 + 1 项仅 Linux 的测试跳过，0 失败）。服务器 GPU 未运行，以下命令未在服务器执行过。

## ① 导入 bundle（Windows → SL3061 → SL3060；只 fetch 和校验，不检出主仓库）

Windows：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failslow/failslow_141050f.bundle'
Get-FileHash -LiteralPath $Bundle -Algorithm SHA256
Get-Content "$Bundle.sha256"
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

两行 SHA256 应相同。SL3061：校验、fetch 到本地分支、转发 SL3060：

```bash
set -e
cd /data && mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c failslow_141050f.bundle.sha256
git -C "$REPO" bundle verify "$PWD/failslow_141050f.bundle"
git -C "$REPO" fetch "$PWD/failslow_141050f.bundle" refs/heads/codex/failslow-141050f:refs/heads/codex/failslow-141050f
test "$(git -C "$REPO" rev-parse refs/heads/codex/failslow-141050f~1)" = 141050fe79708a848b7eeb9371c84911337d9a0f
for key in step6 failslow; do test "$(git -C "$REPO" show 141050fe79708a848b7eeb9371c84911337d9a0f:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$key")" = 1ccac7804105af6b9d70ee93df78f066a8b32e37; done
git -C "$REPO" rev-parse refs/heads/codex/failslow-141050f
scp failslow_141050f.bundle failslow_141050f.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`rev-parse refs/heads/codex/failslow-141050f` 打印的是 C（本文件所在提交），请记入回传。SL3060：同上，去掉最后一行 `scp`。导入只新增分支 `codex/failslow-141050f`，主仓库的检出和工作区不变。

## ② quick_check --step 9（两节点 60 s 内先后启动，SL3060 先）

`quick_check.sh` 直接从清单 B 读出运行，不需要先检出：它按 B 的 `releases.json["failslow"]` 找到实现 A，把 A 的 `quick_driver.py` 取到 kickoff 目录运行；`quick_driver` 在 quick 锁内检查已跟踪文件干净、检出 A，结束后切回 `f7d5559`。运行期间主仓库被检出为 A，请避开其他正在使用主仓库工作区的任务。

内容：四次 launch，共享 GPU、correctness 模式。前三次是 fail-slow：weavetp、anchortp、restart，W 级（SL3061 的 GPU 0/1/4/5 每卡 1 个 `failslow_hog.py`），warmup 20 s、D 30 s、tail 20 s、检测 10 s，toy 请求池 2000 条，到截止时还有请求未完成。第四次是 emulate fail-stop 的 restart：64 条 toy 请求全部完成，故障后 4 个存活 rank 从磁盘加载 TP2，rejoin 后 8 个 rank 加载 TP4，两次加载都对参考 hash 逐参数检查。实际约 15 分钟；超时上限按公式取，不会提前截断。

SL3060：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 141050fe79708a848b7eeb9371c84911337d9a0f:code/current/tools/resharding/failstop/quick_check.sh) --manifest 141050fe79708a848b7eeb9371c84911337d9a0f --step 9 --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

SL3061：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 141050fe79708a848b7eeb9371c84911337d9a0f:code/current/tools/resharding/failstop/quick_check.sh) --manifest 141050fe79708a848b7eeb9371c84911337d9a0f --step 9 --node SL3061 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

每个节点的 `nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step09_failslow/kickoff/<NODE>_*/`）末行应为 `QUICK CHECK step=9 result=PASS ...`。

各次 launch 的检查（汇总在 `<run>/<node>/summary.json` 的 `launches`：`weavetp`、`anchortp`、`restart`、`failstop_restart`）：
- launch passed，本节点没有残留的 `failslow_hog.py`（`leftover_hogs` 为空）。
- SL3060 另查网关文件：
  - fail-slow 三次：`<run>/SL3060/<suite>_<method>_failslow/failslow_result.json` 的事件顺序正确、截止时确有未完成请求、4 个 hog 都 ready 后才开始注入、8 个 EDP0 rank 扩回或重新加载 TP4 后逐参数 sha256 与本次注入前一致（`tp4_hash`）；
  - fail-stop restart：`<run>/SL3060/<suite>_restart_failstop/` 的 `check.json` 通过，`result.json` 的 `failstop.restart_loaded_s` 有 shrink 与 expand，EDP0 两次切换迁移字节为 0。

失败时先看 `<run>/<node>/error.txt` 与 `summary.json`。请回传两节点的 `<suite>_<NODE>.tar.gz` 与 `<suite>_<NODE>.receipt.json`（`/data/ubuntu/lxh/weavetp/acceptance/step09_failslow/` 下）。
