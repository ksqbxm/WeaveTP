# TP2→TP4 Decode 连续性：第 1 步诊断

第 2 步 CPU 汇总、Restart 估计及 L600 请求回放见 [ANALYSIS.md](ANALYSIS.md)，实现 SHA 为 `7511ff32d61ab11cafeb1a5aee19841835a32cee`。以下保留第 1 步诊断说明；第 2 步不更改 GPU 执行路径。

实现 SHA：`661167faa4dacca7ffc4937409133a507cdff427`（下方命令固定到该实现提交；本 README 随后的文档提交不改变脚本）。本目录只交付诊断，不运行 24 次正式实验，不做请求回放、最终指标汇总或绘图。GPU 尚未实测；CPU 测试不能证明显存足够、NCCL 不死锁或诊断门槛通过。

## 固定协议

`diag.sh` 在本机依次执行以下 6 个独立 Launch，再运行两次 8-rank 链路探针。诊断启动前及每次 Launch/Probe 前必须通过同一套空闲检查，任何检查不通过、Launch 失败、超时、压力进程退出都会停止后续步骤，保留日志。不会自动重跑或删除不理想的运行。

空闲规则沿用 16 卡正式实验：全部 8 卡 `memory.used ≤ 200 MiB` 且 `utilization.gpu = 0`；计算进程只允许 `MPS_OWNER` 指定属主的 `nvidia-cuda-mps-server`（精确匹配可执行文件 basename，属主由 `/proc/PID` 的 UID 查询）。`--mps-owner` 优先于环境变量 `MPS_OWNER`；未设置或设为空时，存在计算进程就拒绝启动。MPS 只读检查，不关闭、不发送信号，不修改 `CUDA_MPS_PIPE_DIRECTORY`。每次 L3 Launch/Probe 各自启动相同配置的压力进程，结束后停止本次压力，再检查下一次任务的空闲状态。

诊断结束（包括失败或中断）后，先清理本次启动的进程，再按 GPU UUID 检查显存不高于诊断启动前最后一个空闲采样的基线 +16 MiB。保存 `postflight/machine.json` 和 `memory_recovery.json`；超标或无法确认恢复时退出失败，不能生成 `DIAGNOSTICS_READY=1`。不会为等待 GPU 空闲或显存恢复而重试，也不会关闭原有 MPS 来降低显存。

| 顺序 | case | 压力 | 方法 | 冷启动模式 |
|---|---|---|---|---|
| 1 | smoke_L0_W2 | L0 | W2 | cycle |
| 2 | smoke_L0_B0 | L0 | B0 | cycle |
| 3 | L0_W2_cycle | L0 | W2 | cycle |
| 4 | L0_W2_independent | L0 | W2 | independent |
| 5 | L3_W2_cycle | L3 | W2 | cycle |
| 6 | L3_W2_independent | L3 | W2 | independent |

共同配置：8 卡、EP=2、micro batch=4、seed=1234、DeepSeek-V2-Lite **已转换的真实 Megatron checkpoint**、释放备用权重、Packing/Online Replanning 关闭、MAX_WAVES=64、目标接管后 Decode 10 步。W2 预期约 6 波、B0 约 12 波，以实际记录为准。B0 波上限 2048 是本实验基线设置，不是 NVIDIA 原生 Refit 的 Wave Size。

W2 保留当前选源算法和自适应门槛：请求 residual，允许候选选源，自适应 residual 上限 4 波，扩容波上限 4096，MAX_OVERLAP_STEPS=16。5% 的预测 E2E 门槛、0.20 的 foreground pressure 门槛不降低。B0 请求 baseline/FIFO、禁用选源改写，扩容波上限 2048、STALL_PAUSE=1。实际 candidate/default、调度方式、回退原因、选源计数、每波任务和字节都保留在原结果及逐 rank 轨迹中。发生回退也保留该结果。

冷启动模式由 `NOSTOP_COLD_MODE=cycle|independent` 切换：

- **cycle / SWITCHES=3**：前两次扩/缩容用于预热，只诊断索引 2 的第三次扩容。每次目标存储都必须为 0 字节，重建后仍完整迁移。正式触发前恢复不可变的 2036-token 真实 Prefill 前缀，清除其后 KV，再执行相同的 2 步正常预热和 10 步正常 Decode，保证有效长度恰为 2048。恢复的是内容和 token 输入，不能只重设 offset。没有复制 CPU 权重备份。
- **independent / SWITCHES=1**：使用原 migration NCCL group 和既有 service，在 default/candidate 链路并集上做两轮独立 64 MiB scratch 通信；每 rank 分配发送/接收缓冲各一份。保存 KV 前后校验和、offset、备用权重 0 字节证据。随后同样恢复前缀和 12 步 Decode。它可能不能预热正式接收缓冲的全部分配路径，必须依实测判定，不能先声称已隔离。

两种方法、两种模式统一分配 `SEQ_LENGTH=MAX_POSITION_EMBEDDINGS=5248`，只保留这一配置路径。每次迁移最多推进 `64×16 + 16=1040` tokens（Base 与权重检查），两次预热及各一次验证的高水位为 `2048+2×(1040+1)=4130`，正式阶段为 `2048+1040+10=3098`。同时，原 `run_live_benchmark` 的检查不感知前缀恢复，最严格组合 W2 cycle 的 `worst_case_tokens = 2036+1+3×(64×16+2)+3×16 = 5163`。因此容量是 `align_up_128(max(4130, 5163+1)) = 5248`，不绕过或修改原入口检查。越界、超波数或备用权重未释放直接失败。真实长 Prefill 的显存能否满足需要服务器验证；OOM 不会被替换成零填充的“有效 KV”。

## 时间与证据边界

外层 `benchmark.py` 复用原 `_decode` 的两个 CUDA Events 和 `end.synchronize()` 完成边界，紧接其后读取 `perf_counter_ns()`，不添加逐步 `torch.cuda.synchronize()`。原 benchmark 只加可选诊断回调，普通入口行为不变。STALL_PAUSE 同时禁止 Base 和权重检查中的 Decode，目标真正就绪后先完成目标 Decode，再额外重放源模型做数值比较，避免校验污染目标首步时刻。

每 rank 的 `trace_rankN.json` 保存逻辑步、TP、阶段、wave、KV offset、token 输入、主机开始/完成时间、Event 耗时、GPU 区间、切换触发/Base 完成/Delta 完成/目标就绪/目标首步完成等。预热切换也保留原始数据，但 `measured_switch` 唯一指定被测扩容。rank 必须在相同 hostname、boot ID、Linux time namespace 和 monotonic clock 实现上，报告才对齐逻辑步并取最慢 rank 的主机完成时刻。这里是 CPU 观察到完成的时刻，包含线程被唤醒的延迟。GPU Event 只在**同一 GPU**的共同原点上比较，不能跨 GPU 相减，也没有 1 ms 校准作废规则。

已有 CopyService 的 Event 引用在 launch callback 中保留，到安全结束点再读取。`event_interval_overlap_stages` 使用严格区间相交，允许 Decode 晚于 Transfer 完成，排除只有 participation marker、没有模型 payload 的 NCCL 区间。**这些是 GPU stream Event 包围区间，NCCL 区间含依赖等待，尚不能证明内核实际同时执行。**因此 `verified_execution_overlap_steps=null`，不把已提交次数或区间交集伪装为已证实的真实重叠次数。**kernel 级重叠留待配置冻结后，单独用 1 次 nsys run 验证**；等诊断结果回来、`frozen.env` 确定后，采集和解析作为独立任务。本步仅记录 nsys 是否安装及其版本，不采集 GPU 执行轨迹。

原 `result.json` 的旧 `switch_wall_s` 包含验证，不能用于下一步论文墙钟指标；`base.wave_records[].transport_s` 是原有 Event 传输耗时。新的 milestones 已提供不含额外校验的窗口端点。本步不混用单步 Event latency 与相邻完成间隔，也不计算最终 Downtime、SLO 或请求分位数。

## 两项判定

`diag_report.py` 的顶层输出只有 `cold_start` 与 `pressure`，另写建议 `frozen.env`。

1. 每个 W2 诊断 Launch，仅在被测扩容中按逻辑步取 8 rank 的最晚完成时刻，将进入本波的相邻完成间隔归到本波。第一波的最大间隔减去其他波最大间隔的中位数，**严格大于 100 ms**，或第一波“提交开始到首个 Decode 主机开始”的 rank 最大耗时减去其他波该耗时中位数，**严格大于 50 ms**，判为未隔离。后者专门测主机提交阻塞，不把 GPU Decode 时间计入。只有一波、缺 rank、时钟不可比、状态不符、数值验证/冒烟失败均不能通过。cycle 在 L0/L3 都通过时优先建议它；只有 independent 两档均通过且保存独立预热证据时，才允许建议 SWITCHES=1。
2. 探针测试 B0 默认路由及所有 W2 候选路由实际出现的有向链路并集，使用 NCCL、64 MiB payload，每条链路 2 次预热、5 组原始样本（每组 8 次传输），保存每 rank 耗时。L0 与 L3 使用相同 probe。主要链路在看带宽结果之前按 B0 Base 接收字节降序选取，覆盖至少 80% 远程字节；以 `sum(bytes) / sum(bytes / median_gbps)` 定义固定的等效带宽。要求 `L3 ≤ L0/2.5`。这是字节加权串行链路探针等效值，不能当作并发 Base 的整体吞吐；实际 `transport_s` 另存。每条链路的速率、CV、减速倍数和首尾变化全部保留，W2 合法替代来源不必都降速 2.5×。

探针在运行前固定标记阈值：CV>0.15 或末两样本均值相对首两样本均值变化绝对值>20% 标记波动，不删除样本或事后放宽。这些是波动标记，不自动否定真实带宽波动。缺失、不完整、非正数/非有限带宽证据为 inconclusive。只有全部 Launch、探针、压力/监控存活检查、进程清理及最终显存恢复检查通过后，manifest 的 `execution_complete` 才设为 true；中途失败即使留下完整探针文件，也不能生成可用的冻结配置。`DIAGNOSTICS_READY=1` 仅表示这两项诊断通过，**不是正式性能结论或内核重叠验证通过**。

压力参数：`--links '0:2,1:3,4:6,5:7' --processes-per-pair 7 --stress-bytes 268435456 --stress-sync-every 16`。链路按无向对去重，压力双向，探针保留方向。默认链路仅为待校准的起始配置，报告会列出实际主要链路。已有 `p2p_link_stressor.py` 每方向有一个 Stream，但没有“单进程 k 个压力 Stream”的接口，故保留每对 k 个进程，不能假设 k 等于带宽减速倍数。记录进程 PID/CPU/RSS 和显存开销。若压力不足，先检查 topology、实际路由和 START/退出日志，在新目录用统一新参数重新诊断；不得为了正式结果换链路。

## 可复制的服务器命令

优先登录 SL3061，再对 SL3060 执行同一套机器检查；本地脚本不持有 SSH 凭据、不自动跨服务器操作。SSH 入口沿用工作区已有记录，实际连通性需用户服务器环境确认：

```bash
# SL3061
ssh -p 43061 ubuntu@hdssh01.casdao.com
# 另一个终端检查 SL3060
ssh -p 46060 ubuntu@hdssh01.casdao.com
```

在**两台服务器分别**执行。先 source 服务器现有环境文件，使用其中的 `$REPO` 和 `$PYTHON`，不重新克隆或猜测解释器。更新代码时若 Git 因现有改动拒绝 checkout，应先保留这些改动，不能强制覆盖。带宽画像必须是该服务器实际测得的 8 卡画像，仓库不附带或伪造该文件；CHECKPOINT/PROFILE 如与下面路径不同，应设置为实际绝对路径。

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export REPO PYTHON
# SL3060 已确认 MPS 属主为 yiwei；SL3061 需按实际属主设置，不能照搬。
case "$(hostname -s)" in SL3060|sl3060) export MPS_OWNER=yiwei ;; esac
export BASE=/data/ubuntu/lxh/weavetp/nostop
mkdir -p "$BASE"
export IMPLEMENTATION_SHA=661167faa4dacca7ffc4937409133a507cdff427
git -C "$REPO" fetch origin codex/nostop-diagnostics
git -C "$REPO" checkout --detach "$IMPLEMENTATION_SHA"
cd "$REPO/code/current"
export CHECKPOINT="${CHECKPOINT:-/data/models/DeepSeek-V2-Lite-megatron-v2}"
export PROFILE="${PROFILE:-/home/ubuntu/Megatron-LM-weavetp/profiles/moe_tp_8gpu_idle_p2p_v21.json}"
test -x "$PYTHON"
git rev-parse HEAD

bash tools/resharding/nostop/check_machine.sh --dry-run
CHECK_JOB="$BASE/check_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$CHECK_JOB"
nohup bash tools/resharding/nostop/check_machine.sh \
  --python "$PYTHON" --sample-seconds 30 --out "$CHECK_JOB/run" \
  >"$CHECK_JOB/console.log" 2>&1 < /dev/null &
echo $! > "$CHECK_JOB/launcher.pid"
printf '检查目录：%s\n' "$CHECK_JOB"
```

查看两台各自的 `run/machine.json`、`run/gpu_samples.json`、`run/gpu_processes.txt`、`run/topology.txt` 与 `console.log`。记录包括显存/利用率、温度、SM/显存时钟范围、驱动报告的 CUDA、nvcc Toolkit、PyTorch 构建 CUDA、NCCL、进程、Git SHA 和运行环境。`machine.json` 的 `compute_processes` 逐卡记录 MPS 的属主、PID、GPU UUID 和显存；另有 `mps_owner`、`cuda_mps_pipe_directory`、`idle_errors`。每次采样都查询计算进程，全部原始查询和属主证据保存到 `gpu_samples.json`；任一次采样或属主核验失败均不能判为空闲。另记录 `which nsys` 和 `nsys --version`（`nsys_which.txt` / `nsys_version.txt`）；找不到记为 `missing`，不影响机器检查或诊断的退出码。空闲时钟范围不代表负载时稳定。选择空闲且状态稳定的一台；不修改锁频设置。

然后只在选定服务器继续下面命令。诊断本身再次检查空闲；以下过程较长，使用 nohup。`PROFILE` 必须存在，诊断不会落入原 launcher 的虚构 100 Gbps 默认画像。

```bash
cd "$REPO/code/current"
test -d "$CHECKPOINT" && test -f "$PROFILE"
bash tools/resharding/nostop/diag.sh --dry-run \
  --python "$PYTHON" --checkpoint "$CHECKPOINT" --profile "$PROFILE" \
  --links '0:2,1:3,4:6,5:7' --processes-per-pair 7

export DIAG_JOB="$BASE/diag_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$DIAG_JOB"
nohup bash tools/resharding/nostop/diag.sh \
  --python "$PYTHON" --checkpoint "$CHECKPOINT" --profile "$PROFILE" \
  --links '0:2,1:3,4:6,5:7' --processes-per-pair 7 \
  --stress-bytes 268435456 --stress-sync-every 16 \
  --timeout 2400 --out "$DIAG_JOB/run" \
  >"$DIAG_JOB/console.log" 2>&1 < /dev/null &
echo $! > "$DIAG_JOB/launcher.pid"
printf '诊断目录：%s\n' "$DIAG_JOB"
```

结束后查看（不启动正式实验）：

```bash
cat "$DIAG_JOB/run/status.json"
cat "$DIAG_JOB/run/exit_code.txt"
cat "$DIAG_JOB/run/memory_recovery.json"
cat "$DIAG_JOB/run/diag_report.json"
cat "$DIAG_JOB/run/frozen.env"
# 需要重新生成同一份诊断报告时：
"$PYTHON" -B tools/resharding/nostop/diag_report.py "$DIAG_JOB/run" \
  > "$DIAG_JOB/report_refresh.log" 2>&1
```

`--out` 必须是尚不存在的绝对 `/data/...` 目录。`--dry-run` 不创建目录、不导入 CUDA、不运行任何 GPU 命令。`diag.sh` 退出码 0 表示完成且两项通过，4 表示完整诊断结束但有不通过/不确定项，1 表示执行失败，130/143 表示中断；`check_machine.sh` 的 0 只表示检查执行完成，是否空闲看 machine.json。前置检查失败时可能没有 manifest/report，但一定保留 status 与日志；其后失败则尽可能输出 inconclusive 与 `DIAGNOSTICS_READY=0`。

`pid.txt` 是 supervisor PID；`processes.json` 保存每个子进程的独立 session/PGID、命令、退出码和时间，另有每进程 `*.exit_code.txt`。诊断 benchmark 与 probe 统一使用 torchrun `--run-path` 的 callable worker，继承启动进程组，避免默认 subprocess worker 新建 session 而逃出组清理。supervisor 收到 TERM/INT 或异常时只清理它启动的进程组，保留 leader 到清理完成才回收 PID；重复 cleanup 不重写退出时刻，不使用全局 pkill。普通 benchmark 入口保持原启动方式。超时默认每 Launch/Probe 2400 秒，压力启动等待 120 秒。若强制 SIGKILL supervisor，任何用户态清理器都无法保证执行 finally，应避免该方式。

每个 case 保存 `preflight/machine.json`、`config.json`、`environment.json`、`command.txt`、`exit_code.txt`、`run/result.json`、8 个 rank trace 和 `routes.json`。探针前的空闲检查分别保存在 `probe_L0_preflight/` 与 `probe_L3_preflight/`。根目录保存完整控制台/NCCL 初始化日志、画像副本/哈希、逐进程显存及 CPU 日志、200 ms GPU 时钟/温度采样、全部带宽 probe 样本和候选/默认路由。trace 的 `clock_anchor` 可将主机单调时间映射到 UTC telemetry，以核查正常与迁移期间时钟档位变化；本步只保留证据，不自动删除异常时钟运行。

## CPU 验证

在仓库 `code/current` 目录执行，无需服务器或 CUDA：

```bash
python -m pytest -q -o addopts='' --noconftest \
  tools/resharding/nostop/tests tests/unit_tests/resharding/test_live_defaults.py
bash -n tools/resharding/nostop/check_machine.sh
bash -n tools/resharding/nostop/diag.sh
bash -n tools/resharding/run_live_moe_tp_benchmark.sh
```

本地已通过 79 项诊断 CPU 测试和 7 项默认入口回归，共 86 项；两个 shell 入口的 dry-run 与 shell 语法检查通过。新增测试覆盖指定属主 MPS、其他属主/非 MPS 进程、200/201 MiB 边界、5% 利用率、未设置属主、环境变量/参数优先级、每次 Launch/Probe 前复检、按 UUID 核验 +16/+17 MiB，以及失败或中断后的恢复检查。容量回归直接执行原 benchmark 的 `worst_case_tokens` 公式与检查，对 `{W2,B0}×{cycle,independent}` 四种组合验证通过。其余测试覆盖完成边界、STW 全窗口、冷启动恢复、判定边界、nsys 缺失、GPU 采样失败、探针样本、失败后禁止冻结，以及进程清理和 callable worker 配置；不声称 Linux 实际进程组或 GPU NCCL 已实测。
