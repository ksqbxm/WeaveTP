# Step 1 服务器测试命令

固定实现提交：`ca72cd23d2e6a92cb58e38b6b3ea8b91936ca254`（`codex/failstop`，个人仓库 `ksqbxm/WeaveTP`）。本文件是实现后的交付说明，后续文档提交不改变该实现 SHA。

审阅修正后 31 项 CPU 测试通过；**服务器 GPU 尚未执行**。本次只跑玩具探针，不需要 checkpoint。要求两节点各八卡空闲，原 MPS 服务保留；查询权限不足、存在客户端或不能证明独立 GPU context 时，程序失败并留证据，不修改服务。服务器需现有 PyTorch 2.11 / NCCL 2.28 构建，不在此安装或升级。

本次包含通知连接隔离、前台 MPS 识别、登记失败回收和两节点最终清理确认修正，见 [审阅报告](Step1_审阅与修正.md)。旧版命令固定的 `a609198` 不再用于本次验收。store 由最外层 launcher 持有，完成两节点清理终态交换后关闭；最终结果以 `result.json` 为准。

主测试固定 **10 idle + 10 inflight，共 20 次独立 launch**。默认控制地址 `10.60.14.1`，TCPStore 端口 `29750–29769`；只在两台机器执行各自一份命令，不需要 SSH 互相拉起。先执行 SL3060，再在 10 分钟内执行 SL3061。两处 `SUITE` 必须一致；以下是首次运行名，复跑必须在两处改成新的同名值，保留旧结果。

## SL3060：预检并启动 store、控制器与本地监管器

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
test -z "$(git status --porcelain)"
git fetch https://github.com/ksqbxm/WeaveTP.git codex/failstop
git checkout --detach ca72cd23d2e6a92cb58e38b6b3ea8b91936ca254
export NODE=SL3060
export SUITE=s1_20261017_ca72cd2_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/$SUITE"
export PROBE="$REPO/code/current/tools/resharding/failstop/run_probe.py"
export PYTHON
mkdir -p "$RUN"
"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN/preflight" --preflight-only
nohup bash -c '"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN"; rc=$?; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
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
test -z "$(git status --porcelain)"
git fetch https://github.com/ksqbxm/WeaveTP.git codex/failstop
git checkout --detach ca72cd23d2e6a92cb58e38b6b3ea8b91936ca254
export NODE=SL3061
export SUITE=s1_20261017_ca72cd2_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/$SUITE"
export PROBE="$REPO/code/current/tools/resharding/failstop/run_probe.py"
export PYTHON
mkdir -p "$RUN"
"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN/preflight" --preflight-only
nohup bash -c '"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN"; rc=$?; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

MPS 管道自动从原环境、默认路径或可读取的 daemon 环境定位，并匹配 PID 文件。若实际服务管道无法定位，停止并回传 `preflight`，不能通过给查询程序设置不存在的 worker 管道来绕过检查。已知真实服务路径时可对预检和运行同时使用 `--mps-pipe /实际服务路径`；它只用于只读查询，不改变 worker 隔离策略。

## 查看状态与验收

分别在上述两个 shell 使用：

```bash
tail -n 60 "$RUN/$NODE.nohup.log"
"$PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["RUN"])/os.environ["NODE"]/"progress.json"; print(json.dumps(json.loads(p.read_text()),indent=2) if p.exists() else "First launch is still running; inspect its logs")'
```

终态文件出现后读取：

```bash
cat "$RUN/$NODE.exit_code"
"$PYTHON" -c 'import json,os,pathlib; p=pathlib.Path(os.environ["RUN"])/os.environ["NODE"]/"summary.json"; s=json.loads(p.read_text()); print(json.dumps({"accepted":s["accepted"],"groups":s["groups"],"launches":[{"launch":r["launch"],"status":r["status"]} for r in s["launches"]]},indent=2))'
```

预期两节点退出码均为 0；SL3060 的 `accepted=true`，idle/inflight 各 `count=passed=10`。SL3061 的 `accepted=false` 是设计行为：它只报告本机退出和证据，全局验收由 SL3060 计算。任何 timeout、crash、张量错误、MPS 查询失败、回收失败、清理未确认或 `not_run` 均不能放行。失败后先检查失败轮 `result.json`、`controller.jsonl`、`supervisor_error.json`、对应 worker 日志及最后成功阶段，不覆盖原目录重跑。

## 打包回传并切回原正式实验提交

两节点都在 runner 已退出后执行；即使测试失败也回传完整包：

```bash
test -f "$RUN/$NODE.exit_code"
tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")"
printf '%s\n' "${RUN}_${NODE}.tar.gz"
cd "$REPO"
test -z "$(git status --porcelain)"
git checkout --detach f7d555998b45756998465924d733fd70e553d6b4
```

首次运行的回传路径分别为：

- SL3060：`/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/s1_20261017_ca72cd2_r1_SL3060.tar.gz`
- SL3061：`/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/s1_20261017_ca72cd2_r1_SL3061.tar.gz`

## 单独可选的 shrink 能力探针

主路径完成并清理后，可在两节点重新 checkout 上述实现 SHA，仍按 SL3060→SL3061 顺序执行下段。保留各自 NODE、PROBE、PYTHON 环境；新的 `SUITE` 两节点相同。它额外占用一次独立 launch，使用端口 29770，实际调用 `shrink_group(..., shrink_flags=SHRINK_ABORT)`，不计入主矩阵或替代主路径。

```bash
cd "$REPO"
test -z "$(git status --porcelain)"
git checkout --detach ca72cd23d2e6a92cb58e38b6b3ea8b91936ca254
export SUITE=s1_shrink_20261017_ca72cd2_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/$SUITE"
mkdir -p "$RUN"
nohup bash -c '"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --optional-shrink; rc=$?; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

按前节同样打包、切回正式实验提交。查看 `runtime_*` 的 `backend_supports_shrink`、`SHRINK_ABORT`、`shrink_actual_call` 和逐 worker 错误，不以 Python 属性存在宣称后端可用。该项不支持或失败时单列说明；主路径仍必须 20/20，real 集成仍需用户确认。
