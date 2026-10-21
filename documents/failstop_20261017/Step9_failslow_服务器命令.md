# fail-slow 快验（Step 9）：服务器命令

交付：实现 A `0fe0a3e3b182d8b200789c9d1a483ee7e66ba6a0`，清单 B `f884ff9ec2412ede58c6528d3fe366d0252c4d49`（`releases.json` 的 `step6` 与 `failslow` 都指向 A）。本文件在 B 的子提交 C 里，C 随 bundle `failslow_f884ff9.bundle` 交付（基线 `f7d555998b45756998465924d733fd70e553d6b4`，ref `refs/heads/codex/failslow-f884ff9` 指向 C），bundle 的 SHA256 在同名 `.sha256` 文件里。改动见 [09_failslow主图.md](09_failslow主图.md) §9；CPU 回归 293 项（292 通过 + 1 项仅 Linux 的测试跳过；1 个与本次改动无关的子测试因本机电池供电降频超时，见 09 §8）。服务器 GPU 未运行，以下命令未在服务器执行过。

## ① 导入 bundle（Windows → SL3061 → SL3060；只 fetch 和校验，不检出主仓库）

Windows：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failslow/failslow_f884ff9.bundle'
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
sha256sum -c failslow_f884ff9.bundle.sha256
git -C "$REPO" bundle verify "$PWD/failslow_f884ff9.bundle"
git -C "$REPO" fetch "$PWD/failslow_f884ff9.bundle" refs/heads/codex/failslow-f884ff9:refs/heads/codex/failslow-f884ff9
test "$(git -C "$REPO" rev-parse refs/heads/codex/failslow-f884ff9~1)" = f884ff9ec2412ede58c6528d3fe366d0252c4d49
for key in step6 failslow; do test "$(git -C "$REPO" show f884ff9ec2412ede58c6528d3fe366d0252c4d49:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$key")" = 0fe0a3e3b182d8b200789c9d1a483ee7e66ba6a0; done
git -C "$REPO" rev-parse refs/heads/codex/failslow-f884ff9
scp failslow_f884ff9.bundle failslow_f884ff9.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`rev-parse refs/heads/codex/failslow-f884ff9` 打印的是 C（本文件所在提交），请记入回传。SL3060：同上，去掉最后一行 `scp`。导入只新增分支 `codex/failslow-f884ff9`，主仓库的检出和工作区不变。

## ② quick_check --step 9（两节点 60 s 内先后启动，SL3060 先）

`quick_check.sh` 直接从清单 B 读出运行，不需要先检出：它按 B 的 `releases.json["failslow"]` 找到实现 A，把 A 的 `quick_driver.py` 取到 kickoff 目录运行；`quick_driver` 在 quick 锁内检查已跟踪文件干净、检出 A，结束后切回 `f7d5559`。运行期间主仓库被检出为 A，请避开其他正在使用主仓库工作区的任务。

内容：四次 launch，共享 GPU、correctness 模式。前三次是 fail-slow：weavetp、anchortp、restart，W 级（SL3061 的 GPU 0/1/4/5 每卡 1 个 `failslow_hog.py`），warmup 20 s、D 30 s、tail 20 s、检测 10 s，toy 请求池 2000 条，到截止时还有请求未完成。第四次是 emulate fail-stop 的 restart：64 条 toy 请求全部完成，故障后 4 个存活 rank 从磁盘加载 TP2，rejoin 后 8 个 rank 加载 TP4，不做参考 hash 比对（结果标 `reference_audit="skipped"`）。不需要 `--reference-root`。实际约 15 分钟；超时上限按公式取，不会提前截断。

SL3060：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show f884ff9ec2412ede58c6528d3fe366d0252c4d49:code/current/tools/resharding/failstop/quick_check.sh) --manifest f884ff9ec2412ede58c6528d3fe366d0252c4d49 --step 9 --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

SL3061：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show f884ff9ec2412ede58c6528d3fe366d0252c4d49:code/current/tools/resharding/failstop/quick_check.sh) --manifest f884ff9ec2412ede58c6528d3fe366d0252c4d49 --step 9 --node SL3061 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

每个节点的 `nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step09_failslow/kickoff/<NODE>_*/`）末行应为 `QUICK CHECK step=9 result=PASS ...`。

Step 9 只检查（汇总在 `<run>/<node>/summary.json` 的 `launches`：`weavetp`、`anchortp`、`restart`、`failstop_restart`）：
- 每次 launch 跑完且 passed，本节点没有残留的 `failslow_hog.py`（`leftover_hogs` 为空）；
- SL3060 上 fail-slow 三次的 `<run>/SL3060/<suite>_<method>_failslow/failslow_result.json`：事件顺序正确，截止时确有未完成请求。

失败时先看 `<run>/<node>/error.txt` 与 `summary.json`。请回传两节点的 `<suite>_<NODE>.tar.gz` 与 `<suite>_<NODE>.receipt.json`（`/data/ubuntu/lxh/weavetp/acceptance/step09_failslow/` 下）。
