# 快验：fail-slow Step 9 与 fail-stop inflight（服务器命令）

交付：实现 A `073a50f41e70ee54551f3f2e2613c60203697ad1`，清单 B `7d3573dd0ab1ee43d72fb22d9766e7b3b93be30a`（`releases.json` 的 `step6` 与 `failslow` 都指向 A）。本文件在 B 的子提交 C 里，C 随 bundle `failslow_7d3573d.bundle` 交付（基线 `f7d555998b45756998465924d733fd70e553d6b4`，ref `refs/heads/codex/failslow-7d3573d` 指向 C），bundle 的 SHA256 在同名 `.sha256` 文件里。本次修正上机暴露的两个缺陷（worker 的参数解析、控制门的导入），见 [09_failslow主图.md](09_failslow主图.md) §11；CPU 回归 297 项测试（294 通过、1 项仅 Linux 的测试跳过；2 项与 2 个子测试因本机电池供电降频超出 CPU 模拟的等待上限，见 09 §8）。以下命令未在服务器执行过。

## ① 导入 bundle（Windows → SL3061 → SL3060；只 fetch 和校验，不检出主仓库）

Windows：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failslow/failslow_7d3573d.bundle'
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
sha256sum -c failslow_7d3573d.bundle.sha256
git -C "$REPO" bundle verify "$PWD/failslow_7d3573d.bundle"
git -C "$REPO" fetch "$PWD/failslow_7d3573d.bundle" refs/heads/codex/failslow-7d3573d:refs/heads/codex/failslow-7d3573d
test "$(git -C "$REPO" rev-parse refs/heads/codex/failslow-7d3573d~1)" = 7d3573dd0ab1ee43d72fb22d9766e7b3b93be30a
for key in step6 failslow; do test "$(git -C "$REPO" show 7d3573dd0ab1ee43d72fb22d9766e7b3b93be30a:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$key")" = 073a50f41e70ee54551f3f2e2613c60203697ad1; done
git -C "$REPO" rev-parse refs/heads/codex/failslow-7d3573d
scp failslow_7d3573d.bundle failslow_7d3573d.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`rev-parse refs/heads/codex/failslow-7d3573d` 打印的是 C（本文件所在提交），请记入回传。SL3060：同上，去掉最后一行 `scp`。导入只新增分支 `codex/failslow-7d3573d`，主仓库的检出和工作区不变。

两个快验都直接从清单 B 读出 `quick_check.sh` 运行，不需要先检出：它按 B 的 `releases.json` 找到实现 A，把 A 的 `quick_driver.py` 取到 kickoff 目录运行；`quick_driver` 在 quick 锁内检查已跟踪文件干净、检出 A，结束后切回 `f7d5559`。运行期间主仓库被检出为 A，请避开其他正在使用主仓库工作区的任务。每次两节点 60 s 内先后启动，SL3060 先；SL3061 的命令只把 `--node SL3060` 改成 `--node SL3061`。

## ② fail-slow：quick_check --step 9

四次 launch，共享 GPU、correctness 模式：fail-slow 的 weavetp、anchortp、restart（W 级，SL3061 的 GPU 0/1/4/5 每卡 1 个 `failslow_hog.py`；warmup 20 s、D 30 s、tail 20 s、检测 10 s；toy 请求池 2000 条，到截止时还有请求未完成），然后 emulate fail-stop 的 restart（64 条 toy 请求全部完成，不做参考 hash 比对，结果标 `reference_audit="skipped"`）。实际约 15 分钟。

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 7d3573dd0ab1ee43d72fb22d9766e7b3b93be30a:code/current/tools/resharding/failstop/quick_check.sh) --manifest 7d3573dd0ab1ee43d72fb22d9766e7b3b93be30a --step 9 --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

`nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step09_failslow/kickoff/<NODE>_*/`）末行应为 `QUICK CHECK step=9 result=PASS ...`。只检查：每次 launch 跑完且 passed、本节点没有残留的 `failslow_hog.py`；SL3060 上 fail-slow 三次的 `failslow_result.json` 事件顺序正确、截止时确有未完成请求（汇总在 `<run>/<node>/summary.json` 的 `launches`）。

## ③ fail-stop：quick_check --step 6 --layer inflight（先 emulate，再 real）

扩容时有在途请求的 fail-stop 路径：200 条 toy 请求（输出 96–288 token），共享 GPU、correctness 模式，预算 600 s。先 emulate：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 7d3573dd0ab1ee43d72fb22d9766e7b3b93be30a:code/current/tools/resharding/failstop/quick_check.sh) --manifest 7d3573dd0ab1ee43d72fb22d9766e7b3b93be30a --step 6 --layer inflight --method weavetp --mode emulate --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

emulate 通过后，把 `--mode emulate` 改成 `--mode real` 再跑一次。`nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step06_batched_serving/kickoff/<NODE>_*/`）的 SL3060 结论行：

```text
QUICK CHECK step=6 layer=inflight method=weavetp result=PASS ({"mode": "emulate", "method": "weavetp", "metrics": {"requests_completed": 200, "check_passed": true, "kv_check_passed": true, …}, "files": {…}, "inflight_at_expand": <≥1>, "analysis": {"exit_code": 0, "rejected": [], …}}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=false accepted=false logs=…
```

通过标准：控制门 passed（200 条请求全部完成、`check.json` 通过、扩容 KV 逐位检查通过），`inflight_at_expand ≥ 1`，`analyze_failstop_e2e` 接受门的 `result.json`/`events_actual.json`（退出码 0，`rejected` 为空）。SL3061 的结论行同样 `result=PASS`，`metrics` 为 null。若失败信息是 `inflight_error`（负载在扩容前排空）或 `quick launch exceeded 600 seconds`，请回传。

## 回传

失败时先看 `<run>/<node>/error.txt` 与 `summary.json`。请回传每次快验两节点的 `<suite>_<NODE>.tar.gz` 与 `<suite>_<NODE>.receipt.json`（在 `acceptance/step09_failslow/` 与 `acceptance/step06_batched_serving/` 下）。
