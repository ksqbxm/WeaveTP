# 第 2 步：CPU 汇总与回放

实现 SHA：`7511ff32d61ab11cafeb1a5aee19841835a32cee`。130 项 CPU 测试通过（其中新增 27 项）；以下文档提交不改变实现。

三个入口只读诊断文件，不导入 PyTorch、不启动 GPU 作业，不改 `benchmark.py`、`benchmark_live_moe_tp.py`、诊断和压力执行路径。真实 L0 trace 和 L600 长度文件尚未回传时，只能验证合成测试，不能称为实测端到端验证。

## 输入与统计口径

`summarize.py RUN [RUN ...] --output OUT` 读取每个 run 的 `manifest.json` 和各 case 的 `config.json`、`exit_code.txt`、`run/trace_rank0.json` 至 `trace_rank7.json`。`run/result.json` 用于原有 Base `transport_s`；缺失时该项为 null。支持 `--cases smoke_L0_W2,smoke_L0_B0,L0_W2_cycle` 筛选，默认保留全部 case，包括失败记录。诊断中止不影响已经成功完成的 L0 case。失败或证据不完整的 case 写入 `summary.json.errors`，不记作零性能，也不自动重试；只要存在这些错误，退出码为 1，但仍保存可用结果。

- 只取各 Launch 的 `measured_switch`。排除数值校验、前两次预热切换；被测前正常段不包括 `normal_warmup`。
- 验证 8 rank 的主机、boot ID、time namespace、monotonic clock 相同，逻辑步字段一致。每步完成时间取最晚 rank 的 `perf_counter_ns`，不跨 GPU 相减 Event。缺 rank、缺逻辑步、状态检查失败、STW 在迁移中 Decode 均拒绝统计。
- 窗口包含“最后一次正常完成 → 迁移中所有完成 → TP4 首次完成”。`raw_max_gap_s` 为相邻完成差的最大值；`normal_tbt_s` 为同 Launch、同压力、被测切换前正常 TP2 完成间隔中位数；`excess_downtime_s=max(0,raw-normal)`。固定切换代价仍包含在 raw 中。
- 窗口 TBT 的 Mean/P50/P99 及六个间隔 SLO 比例都使用这些间隔，含两端跨越间隔。达标为 `间隔 ≤ 阈值`，阈值 1.5/2/3/5/10/30 秒。此处是**间隔级**比例，区别于回放的**请求级**比例。
- 单步 latency 使用每个逻辑步各 rank 的最大 `event_ms`，分别统计 normal 和 migration 的 P50/P99/Mean，单位毫秒。migration 包括 Base、权重检查及其他迁移阶段中真实完成的 Decode。STW 无迁移样本，因此 latency 为 null，不填零。
- 每波间隔是“结束于本波 Decode”的相邻完成差；STW 每波间隔列表为空，同时保存各波主机跨度，不能把整段停机重复分配到每一波。保留实际波数、原始 execution mode、FIFO/residual 显示名、候选回退原因、各 rank 自适应原因、选源与接收字节表。Base/Delta 字节仅计接收侧一次，区分含本地拷贝的总量和 remote 字节。
- `switch_wall_s` 使用 trigger → target_ready，不使用旧 result 中包含校验的墙钟；TP4 首步另记录。`verified_execution_overlap_steps` 保持 null，kernel 级重叠验证仍是配置冻结后的独立 nsys 任务。

输出先逐 Launch，再按 method × level × cold_mode × cohort 汇总；不同机器、带宽画像、压力参数和实验配置不混合。复用目录中内容相同的 8-rank trace 按 SHA256 去重，重复来源单列。每个 Launch 等权；不将各 Launch 的步混在一起计算总分位数。分位数采用 `(n−1)×p` 线性插值；跨 Launch 标准差使用样本标准差，只有一个样本时为 null。每个聚合行都有均值、std、n、Launch 总数和带 Launch ID 的原始点。

| 文件 | 内容 |
|---|---|
| `summary.json` | 逐 Launch 完整指标、波、接收表、完成序列、错误、来源和聚合 |
| `launches.csv` | 逐 Launch 指标和实际计划/调度/回退信息 |
| `aggregates.csv` | 全部数值指标的跨 Launch 聚合 |
| `downtime.csv` | 方法 × 压力；主纵轴选 `excess_downtime_s`，并保留 raw 和 normal |
| `step_latency.csv` | W2 的 normal / migration × 压力；正常段来自同一次 W2 Launch，B0 正常值保留在完整汇总 |

## Restart 参考

`restart_estimate.py --summary OUT/summary.json --output RESTART_OUT` 默认查找诊断根目录的 `CASE.log` 中明确的 Megatron `load-checkpoint` 毫秒计时行；支持 max 或 (min,max) 格式，取最大 rank。也可用 `--load-log /实际/TP4加载.log` 指定实测加载日志。多个加载计时、缺单位或非法值会报错，不猜测目标项；显式指定日志缺失/无计时也报错。默认日志不存在或没有加载计时时使用 **35 s 的已有记录**，明确标为 recorded_reference。计时行的路径、行号、原文写入来源；加载布局以原日志为准，不能把 TP2 的初始化加载声称为实测 TP4 Restart。

KV 重建按 `e2e_sim.py:REPREFILL_RATE[4]`（当前 4400 tokens/s/副本）估计；本 microbenchmark 每目标副本为 `MICRO_BATCH_SIZE × CYCLE_KV_TOKENS`，即 4×2048，不累加并行副本耗时。TP4 首步取实测 `target_ready → 首次完成` 的主机墙钟，包含就绪后控制开销，**不是 Event latency**。保留前一正常完成到 trigger 的间隔：

`raw = pre_trigger + load + KV_rebuild + TP4_first_step`

`excess = max(0, raw − 本 Launch normal_tbt)`

输出 `restart.json/csv`，逐 Launch 标注每项来源和 estimated reference。即使加载与首步来自实测，合计仍是 Restart 估计，不是 Restart GPU 实验。

## L600 回放

导入仓库 `documents/e2e_prep_20261008/sim/e2e_sim.py`；沿用其长度过滤、seed=7、不放回抽样、4000 短 + 600 长 + 4000 短请求、`sky_t1_17k` 长请求池、B_max=256。按本实验设为 8 GPU；所有请求 t=0 到达，使用模拟器的 FCFS、prefill 和窗口外 Decode 速率。不是新的 Serving 实测。

每个 Launch 独立回放一次：在首个长请求接纳前执行唯一 TP2→TP4，之后保持 TP4。各轨迹使用相同的切换前容量/零增长峰值条件，必要时先按模拟器排空；不使用历史切换时长、历史 overlap 步数或 AnchorTP。把实测“前一正常完成 → TP4 首次完成”序列原样放入一次切换窗口，每个完成点让当前活动请求各推进一个 token；窗口内不接纳新请求。完成请求正常离开，窗口结束后恢复模拟器时间推进。

这是把 microbatch=4、KV=2048 的实测节奏**不缩放地投影**到模拟请求上，不声称不同 batch/KV 下性能相同，也不验证模拟负载的实际 GPU 显存峰值。输出明确标注 **“模拟请求时间线 + 实测切换轨迹”**。CPU 适配层同时确保到达锚点的那一批 prefill 先完成，再切换，避免原模拟器在一次 admit 同时接纳请求和碰到锚点时漏掉首 token；不修改模拟器源文件。

逐请求计算其完整生成期间最大相邻完成间隔，排除 TTFT；只有一个输出 token 的请求没有 TBT，不计入分位数/SLO 分母。L600 的有效请求均有至少两个输出 token。`replay.json/csv` 给出每轨迹的 P50/P99 和六阈值请求级比例；`requests_LAUNCH_SHA.csv` 保留所有请求最大间隔、输出计数、首末 token 时刻和是否在切换时活动。内部按完成事件压缩正常 Decode 块，无需保存数千万个 token 时间戳；每个请求的输出 token 数必须与输入要求相等。

以下依赖不在仓库，必须提供原文件，缺失时输出 `required_paths`、退出 2，且不生成替代负载或回放结果：

```text
/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/lengths.csv
/tmp/claude-0/-home-user-WeaveTP/37dfa7bc-e9fa-5a64-bc42-6012d0eeb203/scratchpad/data_in/profile/lengths_longout.csv
```

可以用 `--lengths`、`--longout` 指向迁移后的真实文件。保存输入和模拟器 SHA256；样本不足直接失败，不补样本。

## 服务器 CPU 命令

先更新到交付的实现 SHA，然后执行以下命令。`RUN` 填本次或续跑后的完整诊断目录；续跑目录已有复用的 L0 文件。不要重新运行 GPU 诊断。输出均放 `/data/ubuntu/lxh/weavetp/nostop/`。

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export REPO PYTHON
export IMPLEMENTATION_SHA=7511ff32d61ab11cafeb1a5aee19841835a32cee
git -C "$REPO" fetch origin codex/nostop-diagnostics
git -C "$REPO" checkout --detach "$IMPLEMENTATION_SHA"
export RUN=/data/ubuntu/lxh/weavetp/nostop/实际诊断目录
export OUT=/data/ubuntu/lxh/weavetp/nostop/cpu_$(date +%Y%m%d_%H%M%S)
export ANALYSIS="$REPO/code/current/tools/resharding/nostop"
mkdir -p "$OUT"
git -C "$REPO" rev-parse HEAD > "$OUT/implementation.sha"
# L0 实测试跑：筛选已完成的四个 case，不隐藏失败的 L3 原始记录。
nohup bash -c '
  "$PYTHON" -B "$ANALYSIS/summarize.py" "$RUN" \
    --cases smoke_L0_W2,smoke_L0_B0,L0_W2_cycle,L0_W2_independent --output "$OUT/summary"
  status=$?
  if [ "$status" -eq 0 ]; then
    "$PYTHON" -B "$ANALYSIS/restart_estimate.py" --summary "$OUT/summary/summary.json" --output "$OUT/restart"
    status=$?
  fi
  printf "%s\n" "$status" > "$OUT/exit_code.txt"
  exit "$status"
' > "$OUT/analysis.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/pid.txt"
```

具备两份真实长度文件且上一步成功后执行回放（路径改为真实路径）：

```bash
export LENGTHS=/实际数据目录/lengths.csv
export LONGOUT=/实际数据目录/lengths_longout.csv
nohup bash -c '
  "$PYTHON" -B "$ANALYSIS/replay.py" --summary "$OUT/summary/summary.json" \
    --lengths "$LENGTHS" --longout "$LONGOUT" --output "$OUT/replay"
  status=$?
  printf "%s\n" "$status" > "$OUT/replay_exit_code.txt"
  exit "$status"
' > "$OUT/replay.log" 2>&1 < /dev/null &
echo "$!" > "$OUT/replay_pid.txt"
```

只读 CPU 作业，无压力进程或 GPU 子进程，不调用清理/kill。全量汇总时去掉 `--cases`；多个 run 可作为多个位置参数传入，重复轨迹会去重。

本地 CPU 测试：

```bash
"$PYTHON" -m pytest -q -o addopts='' --noconftest \
  "$REPO/code/current/tools/resharding/nostop/tests" \
  "$REPO/code/current/tests/unit_tests/resharding/test_live_defaults.py"
```

合成数据仅在 pytest 临时目录生成，用于核对公式和端到端文件接口，不作为任何实测指标或默认负载。
