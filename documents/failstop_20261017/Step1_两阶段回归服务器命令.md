# Step 1 两阶段回归：共享 GPU 快验命令

固定实现 SHA：`6bbd25326eb7311688ebb057d35e648471f0e5ae`，分支 `codex/failstop`，个人仓库 `ksqbxm/WeaveTP`。54 项 CPU 测试通过；此优化版本的 GPU 测试尚未运行。上一版 d16b5e8 的共享 20/20 关口已获用户确认。本次不推进 Step 2/task2 或模型集成。

先分发，再先启动 SL3060、10 分钟内启动 SL3061。两边 suite、源码 SHA、repeats 必须相同。N=1 执行 idle_00、inflight_00 两次独立 launch，端口 29750、29751；仍用 PyTorch 2.11 / NCCL 2.28 现有环境。重跑必须两边一起改新 suite 名，不能覆盖证据。

## 1. SL3061 拉取、打包、传给 SL3060

bundle 以两节点已经验收过的 d16b5e8 为基线，只传新增内容。保留旧 bundle；SL3060 无需访问 GitHub。scp 默认使用当前用户名，若不同需在地址前加实际用户名。

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
SHA=6bbd25326eb7311688ebb057d35e648471f0e5ae
BASE=d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c
git fetch https://github.com/ksqbxm/WeaveTP.git codex/failstop
git merge-base --is-ancestor "$BASE" "$SHA"
git checkout --detach "$SHA"
git update-ref refs/heads/codex/step1-6bbd253 "$SHA"
cd /data/ubuntu/lxh/weavetp
BUNDLE=step1_6bbd253.bundle
test ! -e "$BUNDLE"
git -C "$REPO" bundle create "$PWD/$BUNDLE" refs/heads/codex/step1-6bbd253 "^$BASE"
git -C "$REPO" bundle verify "$PWD/$BUNDLE"
sha256sum "$BUNDLE" > "$BUNDLE.sha256"
ls -lh "$BUNDLE" "$BUNDLE.sha256"
scp "$BUNDLE" "$BUNDLE.sha256" 10.60.14.1:/data/ubuntu/lxh/weavetp/
```

## 2. SL3060 校验并导入

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step1_6bbd253.bundle.sha256
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git bundle verify /data/ubuntu/lxh/weavetp/step1_6bbd253.bundle
git fetch /data/ubuntu/lxh/weavetp/step1_6bbd253.bundle refs/heads/codex/step1-6bbd253
git checkout --detach 6bbd25326eb7311688ebb057d35e648471f0e5ae
```

## 3. SL3060 启动

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git checkout --detach 6bbd25326eb7311688ebb057d35e648471f0e5ae
export NODE=SL3060
export SUITE=s1_twophase_20261017_6bbd253_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/$SUITE"
export PROBE="$REPO/code/current/tools/resharding/failstop/run_probe.py"
export PYTHON
mkdir -p "$RUN"
"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN/preflight" --preflight-only --repeats 1 --allow-foreign-gpu-processes
nohup bash -c '"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --repeats 1 --allow-foreign-gpu-processes; rc=$?; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

## 4. SL3061 启动

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git checkout --detach 6bbd25326eb7311688ebb057d35e648471f0e5ae
export NODE=SL3061
export SUITE=s1_twophase_20261017_6bbd253_r1
export RUN="/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/$SUITE"
export PROBE="$REPO/code/current/tools/resharding/failstop/run_probe.py"
export PYTHON
mkdir -p "$RUN"
"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN/preflight" --preflight-only --repeats 1 --allow-foreign-gpu-processes
nohup bash -c '"$PYTHON" -B "$PROBE" --node "$NODE" --suite-id "$SUITE" --output "$RUN" --repeats 1 --allow-foreign-gpu-processes; rc=$?; printf "%s\n" "$rc" > "$RUN/$NODE.exit_code"; exit "$rc"' > "$RUN/$NODE.nohup.log" 2>&1 < /dev/null &
echo $! > "$RUN/$NODE.runner.pid"
```

`/data` 门槛默认 5 GB，若调整则在预检/运行两处都加 `--min-free-disk-gb 值`。未跟踪 bundle/结果文件不阻止运行，已跟踪文件的修改仍拒绝。MPS 与进程回收规则沿用 d16b5e8。

## 5. 查看与判断

两节点分别查看各自 nohup 日志；完成后读取退出码及 summary：

```bash
tail -n 60 "$RUN/$NODE.nohup.log"
cat "$RUN/$NODE.exit_code"
"$PYTHON" -c 'import json,os,pathlib; s=json.loads((pathlib.Path(os.environ["RUN"])/os.environ["NODE"]/"summary.json").read_text()); print(json.dumps({k:s[k] for k in ("accepted","all_cases_passed","quick_mode","repeats","shared_gpu_mode","groups")},indent=2))'
```

快验预期：两节点 exit=0，quick_mode=true、repeats=1、all_cases_passed=true，idle/inflight 各 count=passed=1。**accepted=false 是 N<10 的预期行为**，不表示快验失败。仍需原张量/地址、计算、新 world collective、存活 PID、回收和清理全部正确。

在 SL3060 的每轮 result.json / summary.groups.metrics 对比：

- spawn_to_ready_s：后台新进程启动耗时；这段期间 EDP1 的 preparation_service 步数应增加，worker JSONL 有连续 service_progress。
- edp1_rejoin_pause_s：实际重建窗口内的最大 EDP1 计算间隙，应接近 rebuild_and_validate_rejoin_s 加 abort/协调开销。
- rejoin_join_s：完整回归墙钟，仍包含后台初始化；它没有变短不代表 EDP1 仍停顿同样久。

controller.jsonl 应满足 spawn → rejoin_ready → transition/2 → build/2；初始化期间不能提前 transition/2。时间仍是共享 GPU 诊断数据，不作正式性能数据，不按固定 1.7–1.9 s 自动判通过。

## 6. 回传与恢复

两节点 runner 退出后执行，即使失败也保留完整证据：

```bash
test -f "$RUN/$NODE.exit_code"
tar -czf "${RUN}_${NODE}.tar.gz" -C "$(dirname "$RUN")" "$(basename "$RUN")"
printf '%s\n' "${RUN}_${NODE}.tar.gz"
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git checkout --detach f7d555998b45756998465924d733fd70e553d6b4
```

回传路径为 `/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/s1_twophase_20261017_6bbd253_r1_SL3060.tar.gz` 与同名前缀 `_SL3061.tar.gz`。后续需要 20 轮验证时，两边改新 suite，并将预检和运行的 repeats 同时改为 10（或删去参数），顺序为 idle/inflight 交替。
