# Step 1 共享 GPU 正确性测试命令

本页保留已通过的 d16b5e8 验收命令。当前两阶段回归优化请使用 [新版 N=1 快验命令](Step1_两阶段回归服务器命令.md)。

固定实现提交：`d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c`（`codex/failstop`，个人仓库 `ksqbxm/WeaveTP`）。本文件是实现后的交付说明，后续文档提交不改变该实现 SHA。

50 项 CPU 测试通过；**此实现已获用户确认：s1_shared_20261017_d16b5e8_r2 的 20/20 共享 GPU 正确性验收通过**。本次只跑玩具探针，不需要 checkpoint。两节点各八卡，允许其他任务占用 GPU；预检与运行均加 `--allow-foreign-gpu-processes`。共享模式允许 foreign MPS 客户端，无法定位实际管道的 daemon 记录后跳过查询；本测试 worker 必须在内部证明独立 GPU context，已定位管道查询报错仍失败，不修改服务。服务器需现有 PyTorch 2.11 / NCCL 2.28 构建，不在此安装或升级。

本次包含通知连接隔离、前台 MPS 识别、登记失败回收和两节点最终清理确认修正，见 [审阅报告](Step1_审阅与修正.md)。旧版命令固定的 `a609198` 不再用于本次验收。store 由最外层 launcher 持有，完成两节点清理终态交换后关闭；最终结果以 `result.json` 为准。

本页显式开启共享模式，结果标记 `shared_gpu_mode=true`。**与其他任务共享 GPU，仅用于正确性和可行性判断，各阶段耗时不作为正式数据。** 此模式的 accepted=true 仅表示正确性/可行性通过。默认独占口径见 [原服务器命令](Step1_服务器命令.md)，实现语义见 [共享模式说明](Step1_共享GPU正确性模式.md)。

主测试固定 **10 idle + 10 inflight，共 20 次独立 launch**。默认控制地址 `10.60.14.1`，TCPStore 端口 `29750–29769`；只在两台机器执行各自一份命令，不需要 SSH 互相拉起。先执行 SL3060，再在 10 分钟内执行 SL3061。两处 `SUITE` 必须一致；以下是首次运行名，复跑必须在两处改成新的同名值，保留旧结果。

先完成 [SL3061 拉取和 bundle 分发](Step1_bundle分发.md)，两节点均已导入固定实现提交后再启动。磁盘预检默认要求 `/data` 可用空间至少 5 GB；可在预检和运行两处同时加 `--min-free-disk-gb 5` 调整门槛。

## SL3060：预检并启动 store、控制器与本地监管器

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git cat-file -e d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c^{commit}
git checkout --detach d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c
export NODE=SL3060
export SUITE=s1_shared_20261017_d16b5e8_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/$SUITE"
export PROBE="$REPO/code/current/tools/resharding/failstop/run_probe.py"
export PYTHON
mkdir -p "$RUN"
"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN/preflight" --preflight-only --allow-foreign-gpu-processes
nohup bash -c '"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --allow-foreign-gpu-processes; rc=$?; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

首轮的 `SL3060/${SUITE}_idle_00/store.jsonl` 会记录 `store_listening`，监管器日志为同目录 `supervisor.jsonl`。启动协调期间不要求先有任何 GPU worker；两个节点预检和配置匹配后才启动 16 个 worker。

## SL3061：预检并启动本地监管器

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git cat-file -e d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c^{commit}
git checkout --detach d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c
export NODE=SL3061
export SUITE=s1_shared_20261017_d16b5e8_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/$SUITE"
export PROBE="$REPO/code/current/tools/resharding/failstop/run_probe.py"
export PYTHON
mkdir -p "$RUN"
"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN/preflight" --preflight-only --allow-foreign-gpu-processes
nohup bash -c '"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --allow-foreign-gpu-processes; rc=$?; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

MPS 管道自动从原环境、默认路径或可读取的 daemon 环境定位，并匹配 PID 文件。若实际服务管道无法定位，记录 `skipped_daemons`、`query_complete=false` 后继续；这不表示没有客户端。本测试每个 worker 分配 CUDA 张量后核对自身 PID/UUID 的独立计算 context。已知真实服务路径时可对预检和运行同时使用 `--mps-pipe /实际服务路径`；它只用于只读查询，不改变 worker 隔离策略。

## 查看状态与验收

分别在上述两个 shell 使用：

```bash
tail -n 60 "$RUN/$NODE.nohup.log"
"$PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["RUN"])/os.environ["NODE"]/"progress.json"; print(json.dumps(json.loads(p.read_text()),indent=2) if p.exists() else "First launch is still running; inspect its logs")'
```

终态文件出现后读取：

```bash
cat "$RUN/$NODE.exit_code"
"$PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["RUN"])/os.environ["NODE"]/"summary.json"; s=json.loads(p.read_text()); print(json.dumps({"accepted":s["accepted"],"shared_gpu_mode":s["shared_gpu_mode"],"timing_note":s["timing_note"],"groups":s["groups"],"launches":[{"launch":r["launch"],"status":r["status"]} for r in s["launches"]]},indent=2))'
```

预期两节点退出码均为 0；SL3060 的 `shared_gpu_mode=true`、`accepted=true`，idle/inflight 各 `count=passed=10`。SL3061 的 `accepted=false` 是设计行为：它只报告本机退出和证据，全局共享正确性结论由 SL3060 计算。任何 timeout、crash、张量错误、MPS 查询失败、回收失败、清理未确认或 `not_run` 均不能放行。耗时只作诊断；本测试 PID、张量、计算、collective、完整 20/20 及清理确认仍须通过。失败后先检查失败轮 `result.json`、`controller.jsonl`、`supervisor_error.json`、对应 worker 日志及最后成功阶段，不覆盖原目录重跑。

## 打包回传并切回原正式实验提交

两节点都在 runner 已退出后执行；即使测试失败也回传完整包：

```bash
test -f "$RUN/$NODE.exit_code"
tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")"
printf '%s\n' "${RUN}_${NODE}.tar.gz"
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git checkout --detach f7d555998b45756998465924d733fd70e553d6b4
```

首次运行的回传路径分别为：

- SL3060：`/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/s1_shared_20261017_d16b5e8_r1_SL3060.tar.gz`
- SL3061：`/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/s1_shared_20261017_d16b5e8_r1_SL3061.tar.gz`

