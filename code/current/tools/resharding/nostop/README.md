# TP2→TP4 Decode 连续性实验

## 2026-10-13：合成 KV 切换审计与失败退出

针对 SL3060 的 `nrmse=0.411 / cosine=0.9156`，本次增加实际位内容审计；尚无该失败运行的 KV checksum，不能提前断言它只是数值噪声。下方 quick 命令固定到本次修复版本。

- 每个 rank 的 `trace_rankN.json` 增加 `validation_audits`：按请求域、层、K/V、全局 head，对 `[0,2048)` 的全部 batch 元素计算原始字节 SHA256，比较 TP2/TP4 及同域全部副本，同时核对初始化基准。保留本 rank 的全部 checksum、跨 rank 失败项、审计开始/完成时间。
- `stage=initial` 检查首次合成内容与 head 分片是否一致；`restore` 检查 cycle 恢复；`cutover` 检查每次扩容、回切之后的内容。初始化/恢复通过而 cutover 目标不一致时，应沿迁移路径排查，不能通过放松 logits 阈值掩盖。
- 权重复用 `StandbyWeights.bitwise_checksums()` 的逐参数原始位双 int64 checksum，与首次加载的本 rank、同 TP 布局基准比较，源、目标都审计。基准保存在 `weight_reference`，失败记录包含参数名和 expected/actual。它沿用仓库现有的 checksum 审计语义，并非保存整份 CPU 权重进行逐元素比较。
- 合成 KV 只有在全 rank KV 与权重审计都通过后，logits 才采用仅记录策略。`logit_metrics` 保存 nrmse、cosine、top1、max_abs、reference_max_abs 等原指标及 `within_thresholds`，不改原阈值。真实 KV 的原始 validator 仍然断言；`benchmark_live_moe_tp.py` 相对 `8ae22e7` 和 `4a9d1fa` 均为零 diff。
- 切换审计在目标 post Decode 完成后、源侧 logits 比较前执行，检查不会被 post Decode 改写的原始 2048 前缀；恢复审计在正常 Decode 段之前。审计增加窗口外校验耗时，不能将这部分计入 downtime。Decode、Transfer、cutover 路径及其同步点保持不变，没有在迁移窗口增加校验通信。
- 失败时方法目录仍写出 `complete=false` 的 rank trace 和 memory，包含已取得的审计证据；不伪造成功 result。成功时各测量目录按原有 switch 映射保留对应审计与 logits 指标，现有汇总/回放读取格式保持可用。
- quick 仅在子进程成功、完整输出检查和本任务清理均成功后写退出码 0。缺文件、提前退出、末档失败、清理失败均返回非零；W2 失败后不启动 B0。根目录 `status.json.exit_code`、`exit_code.txt` 与 quick 进程退出码一致；子进程原始退出码另外保存在 `processes.json` / `benchmark.exit_code.txt`。`quick.sh` 继续通过 `exec` 传递 Python 的退出码。

本次全套 **240 项 CPU 测试通过**，包括原始位单 bit/正负零差异、全局 head 错位、远端副本和权重损坏、原权重释放重建、cycle 基准、审计先于 logits 比较、真实 KV 严格校验、失败 trace、quick 进程实际退出码与状态一致及标准文件读取。14 段 README 命令和 3 个相关 shell 脚本通过 Bash 语法检查；decode/prefill/transfer 包装函数 AST 与 `4a9d1fa` 相同。GPU 上须重新运行确认 KV/权重审计结果；原来的 logits 失败记录不足以证明迁移内容正确。

## 2026-10-13：quick 最小出数入口

实现 SHA：`4a9d1fa7526e014823a09041f4e2c6643141eb03`。

CPU 验证：全套 221 项通过，其中 quick 新增 8 项覆盖 desc/asc 顺序、每档 empty_cache、握手与显存门槛、进程清理、dry-run、标准文件读取及多次运行合并。两个相关 Bash 脚本和本 README 的 14 段命令通过语法检查；尚未在服务器执行 GPU 实测。

`quick.sh` 独立启动 `quick.py`，不调用 diag/formal。W2 然后 B0，各一个 torchrun 进程、各加载一次模型。保留 `method_env` 的 W2/B0 参数、8-token 真实 prefill、确定性合成 KV、固定画像、5248 容量和权重释放规则。加载后先执行一个无压力预热周期 2→4→2，然后按 `--order desc|asc` 换档；默认 desc 为 L3→L2→L1→L0，asc 为相反顺序。

`run_live_moe_tp_benchmark.sh` 仅为 quick 入口启用已有 `--run-path` 策略，使 torchrun worker 保持在本任务的进程组中，供 supervisor 清理。

每档固定链路 `0:2,1:3,4:6,5:7`，每对 L3/L2/L1/L0 使用 7/3/1/0 个隔离设备的 stressor、64 MiB buffer。每次换档（包括 asc）先在 GPU 进程中 barrier → `empty_cache()` → barrier，读取本次进程的 torch 峰值/context，写 request；外层停止上一档自己的 stressor，启动新档，检查显存，再原子写 go。全部发生在计时窗口之外。门槛不足就停止，不自动调低进程数。

模型已常驻，因此准入采用已确认的口径：**`NVML free + 本 rank 当前 reserved + 当前 context ≥ 全 rank 历史最大 allocated + 全 rank 历史最大 context + 1.5 GiB`**，避免重复扣算本模型。每档保存原始 free、增加的本模型可用预算、A/C 和进程显存；A/C 每次换档更新，不使用旧 L0 文件。

每档测 `--k` 次（默认 2）：恢复同一 2048 KV → 12 次正常 Decode（前 2 次预热）→ 被测 2→4 → 10 次目标 Decode及校验 → 不计入指标的 4→2。默认每个进程实际执行 18 次切换，其中 8 次扩容计入指标；整个 quick 共两次进程启动、16 次测量。CPU 外层在原有容量检查及分配完成后扩展 host 循环，并按“每对切换前恢复 KV”的真实高水位检查 5248 容量；不按累计切换数膨胀 KV 分配。`benchmark_live_moe_tp.py` 保持零 diff，Decode/Transfer/cutover 执行逻辑未改。

自动输出：

- `downtime.csv`：方法 × 档位，Excess Downtime 的 mean/min/max/n，单位秒。
- `step_latency.csv`：W2 normal/migration × 档位，先算每次测量的最慢 rank 单步 GPU Event 耗时均值，再给 mean/min/max/n，单位 ms；n 是测量次数，不是 token 数。
- 两张表的 `x_label` / `b0_migration_bandwidth_gbps` 使用同档 B0 的平均实测迁移带宽。带宽 = **远端 Base 接收字节 / Base transport_s / 1e9**，是整个迁移路径的聚合吞吐，包含通信与并发 Decode 的影响，不是裸链路探针。Base/Delta、本地/远端字节分别保存。
- `measurements.csv/json`：每次测量的 raw gap、normal TBT、excess downtime、step latency、字节、transport、带宽和计划。`summary/` 是现有 `summarize.py` 的完整输出，可直接交给 `replay.py` 和 `restart_estimate.py`，缺失的模拟器输入仍须用户提供。

`methods/W2|B0/run/` 保留完整进程的 `trace_rank0..7.json`、`memory_rank0..7.json`、`result.json`、routes 和日志。每次测量另有 `W2_L3_R1/run/` 等标准目录，含同名全部文件；其 trace 将共享的真实预热周期映射为 0/1、被测扩容映射为 2，保留原始时钟/逻辑步及 `original_switch`。result 的 switches 和权重生命周期也按相同映射选择，其余进程级遥测的范围在 quick 元数据中明确标注。memory 文件保留完整进程的全部原始阶段。`process_launch_id` 标明两次真实进程启动；同一进程内 K 次测量共享模型与预热，不能当作 K 次独立 launch。原始完整轨迹不丢弃。

每个 case 同时提供 `CASE.log`，与所属方法的日志共用文件内容，便于现有 Restart 读取器发现 checkpoint 加载计时；没有可识别计时才沿用 35 秒参考值。

仅保留 MPS 空闲检查、本任务进程组清理和外部进程只读监测；每方法保存 200 ms GPU 采样、1 s compute-apps、外部进程时间段、environment/config、PID、退出状态与完整日志。外部进程记录不触发中断/重跑。没有探针、带宽门槛、frozen.env、失败重试或污染重跑。输出目录必须新建；失败后另选目录重新执行。

服务器复制执行（下方为一个 nohup 启动命令，默认 K=2、desc）：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export REPO PYTHON MPS_OWNER=yiwei
git -C "$REPO" fetch origin codex/nostop-diagnostics
git -C "$REPO" checkout --detach 4a9d1fa7526e014823a09041f4e2c6643141eb03
export NOSTOP="$REPO/code/current/tools/resharding/nostop"
export QUICK_OUT=/data/ubuntu/lxh/weavetp/nostop/quick_$(date +%Y%m%d_%H%M%S)
mkdir -p "$QUICK_OUT"
nohup bash "$NOSTOP/quick.sh" --out "$QUICK_OUT/run" --k 2 --order desc > "$QUICK_OUT/quick.log" 2>&1 < /dev/null &
echo "$!" > "$QUICK_OUT/nohup_pid.txt"
tail -f "$QUICK_OUT/quick.log"
```

改成 `--order asc` 即为升档，同样每档 empty_cache 并重验门槛。进程结束后：

```bash
cat "$QUICK_OUT/run/status.json" "$QUICK_OUT/run/exit_code.txt"
cat "$QUICK_OUT/run/downtime.csv" "$QUICK_OUT/run/step_latency.csv"
tar -C "$(dirname "$QUICK_OUT")" -czf "${QUICK_OUT}.tar.gz" "$(basename "$QUICK_OUT")"
# 多次 quick 运行一起合并；分别传每次的 run 目录，不传 methods 子目录。
"$PYTHON" "$NOSTOP/summarize.py" \
  /data/ubuntu/lxh/weavetp/nostop/quick_第一次/run \
  /data/ubuntu/lxh/weavetp/nostop/quick_第二次/run \
  --output /data/ubuntu/lxh/weavetp/nostop/quick_merged
```

可先使用 `bash "$NOSTOP/quick.sh" --out "$QUICK_OUT/run" --dry-run` 查看顺序和配置，不查询 GPU、不启动进程、不创建输出目录。默认每个 torchrun 的总超时为 7200 秒；两次加载共 36 次切换的真实机时须由服务器日志确定。

## 2026-10-12：8-token prefill、合成 KV 与 torch 显存预算

实现 SHA：`8d22890be3e4639165571e15382c2eff656202a5`。

撤回 30772ad 中从 NVML 进程峰值推导的“无合法 L3 进程数”结论，并删除对应冻结建议。NVML 包含 caching allocator 保留的缓存，旧 L0 的 31.17 GiB 不能作为实际需求。本次不推算新的实测峰值；必须重新跑新工作负载的 L0。

本次 **213 项 CPU 测试通过**，README 的 12 段 Bash 命令通过语法检查。覆盖 TP2/TP4 内容逐 bit 一致、初始验证输入、cycle 恢复、全局 RNG 不变、各阶段 torch 峰值/context 预算、拒绝旧工作负载证据、无旧文件的 L0 启动及后续 L3 准入。未执行新的 GPU 测量。

工作负载标识为 `short8-synthetic2048-v1`，结果、trace、memory 文件及 manifest 均标注 **“合成 KV 内容，长度 2048 真实”**：

- TP2、TP4 各真实 prefill **8 tokens**。在初始化的 Gloo control group 中一次性收集这 8 个位置的 TP2 CPU KV，按请求域选同一 TP2 副本作为参考，重排到两个布局，使真实前缀也逐 bit 相同。这是初始化通信，发生在切换计时与独立 NCCL 预热之前；不会进入正式迁移字节数。
- 位置 8–2047 按固定 seed `20261012`、请求域、层、K/V、全局 attention head 生成 CPU `N(0, 0.02)` 数据，再转换为 bf16 写入实际 KV 存储。batch 维每条请求使用各自的随机元素；不使用全局 RNG，不依赖 TP shard 大小或方法。逐 head 生成，不在 GPU 上额外分配整份 KV 副本。
- cycle 第 0/2 次扩容前恢复同一真实前缀并重新生成同一合成内容；正常段在 offset 2048–2059 执行 12 次 Decode（前 2 次预热），随后仅将有效长度回退到 2048，丢弃新追加的尾部。原 2048 个位置不被这些 Decode 改写，因此最后一次正常 Decode 完成之后不再执行 GPU 填充。被测切换仍释放、重建并完整迁移目标权重及 2048-token KV。
- `benchmark_live_moe_tp.py` 相对 `8ae22e7` / `30772ad` 零 diff。外层 `Diagnostic.decode` 的执行/同步逻辑不变，迁移和 cutover 代码不改；prefill 与前缀初始化/恢复在外层实现。保持 `MAX_WAVES=64`、`SEQ_LENGTH=MAX_POSITION_EMBEDDINGS=5248`、原 W2/B0 策略，不启用 `expandable_segments`，不设置 `NCCL_RUNTIME_CONNECT`。

每个 rank 输出独立的 `run/memory_rankN.json`（trace 中也保存同一记录）：启动、prefill、计划准备、每次切换前准备、两次预热切换、被测切换及各自的 post 阶段。每阶段开始前 `reset_peak_memory_stats`，结束时记录 `max_memory_allocated` / `max_memory_reserved`。不调用 `empty_cache`，不增加 CUDA 同步；切换后的 post 阶段包含目标 Decode、数值验证和随后释放。

初始验证和各次切换验证完成后，在本 rank 已有 Decode Event 完成点之后查询静止进程的 NVML 与同时的 torch reserved。`context = NVML_process_bytes - memory_reserved()`，记录原始输出和 GPU UUID；缺样、reserved 在查询中变化或差值为负时标记不可用，不能凭 MPS server 用量伪造 context。该差值包含 CUDA/NCCL 等非 PyTorch allocator 开销。查询在首次目标 Decode 已完成之后，不进入主指标窗口。

新预算使用全部已成功 L0 的 `A = max(各 rank / 各阶段 allocated)` 与 `C = max(有效静止点 context)`，而非 reserved 或 NVML 峰值：

- 启动门槛 `MIN_FREE_GIB = A + C + 1.5`。reserved、NVML 只作参考，原始阶段样本及文件 hash 保留。
- L3 每对最多 7 个进程。按每张卡涉及的固定链路数计入 stressor，取可留出 **2 GiB** 余量的最大整数；这 2 GiB 已覆盖 1.5 GiB 碎片余量，不重复相加。L1/L2 按 1:3:7 比例缩放，正档至少 1 个。
- 对默认每卡一对链路、单进程约 806 MiB、MPS 56 MiB，L3 单卡预计占用为 `A + C + k×806/1024 + 56/1024 GiB`；含碎片预算再加 1.5。若新 L0 允许 `k=7`，则分别为 **`A+C+5.564453125`** 与 **`A+C+7.064453125 GiB`**。以 31.36 GiB 容量计，7 进程要求 **`A+C ≤ 23.795546875 GiB`**。目前没有新 L0 实测，不能用旧 OOM 时的 22.14 GiB 代替新峰值或提前宣布 7 进程可行。

诊断默认依次执行 `smoke_L0_W2`、`smoke_L0_B0`、`L0_W2_cycle`、`L3_W2_cycle`，然后运行 L0/L3 路由探针。`--memory-from` **可选**：不提供时，首次 L0 在已通过空闲检查的整机上测量；其余 L0 及 L3 根据本次累计的新 torch 证据准入。三个 L0 完成后确定 L3 进程数，正式实验前还须通过原带宽门槛。`--routes-from` 仅用于可选的后续同版本局部重跑，不是默认启动条件；旧工作负载的 L0、路由、显存证据不能复用。独立预热仍可通过 `--cases` 显式选择。

后续同工作负载局部重跑若指定 `--routes-from` 且不指定 `--memory-from`，会同时读取该目录成功 L0 的新 torch 显存证据；默认首次诊断不需要这两个参数。

`frozen.env` 使用 `NOSTOP_WORKLOAD`、`ALLOCATED_PEAK_GIB`、`CONTEXT_GIB`、`MIN_FREE_GIB` 及实际 L0 预算推荐的进程数。正式入口核对这些值与诊断证据一致。没有完整新证据时 `DIAGNOSTICS_READY=0`，不自动启动正式实验。kernel 级重叠留待配置冻结后，单独用 1 次 nsys run 验证；`verified_execution_overlap_steps` 仍为 null。

服务器直接运行（不传旧 L0 路径，四个 case 全部重新测量；长任务的输出、PID、退出状态都在 `/data`）：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export REPO PYTHON MPS_OWNER=yiwei
git -C "$REPO" fetch origin codex/nostop-diagnostics
git -C "$REPO" checkout --detach 8d22890be3e4639165571e15382c2eff656202a5
export NOSTOP="$REPO/code/current/tools/resharding/nostop"
export DIAG_OUT=/data/ubuntu/lxh/weavetp/nostop/diag_synthetic_$(date +%Y%m%d_%H%M%S)
bash "$NOSTOP/check_machine.sh" --dry-run
bash "$NOSTOP/diag.sh" --dry-run --out "$DIAG_OUT/run"
mkdir -p "$DIAG_OUT"
nohup bash "$NOSTOP/diag.sh" --out "$DIAG_OUT/run" \
  --links '0:2,1:3,4:6,5:7' --stress-bytes 67108864 \
  > "$DIAG_OUT/supervisor.log" 2>&1 < /dev/null &
echo "$!" > "$DIAG_OUT/nohup_pid.txt"
tail -f "$DIAG_OUT/supervisor.log"
# supervisor 退出后检查并打包。
cat "$DIAG_OUT/run/status.json" "$DIAG_OUT/run/exit_code.txt"
cat "$DIAG_OUT/run/frozen.env"
tar -C "$(dirname "$DIAG_OUT")" -czf "${DIAG_OUT}.tar.gz" "$(basename "$DIAG_OUT")"
```

单独 CPU 重算新 run 的预算（需要轻量的 memory_rank 文件、config、manifest、machine，不需要完整 trace）：

```bash
"$PYTHON" "$NOSTOP/memory_budget.py" "$DIAG_OUT/run" \
  --out /data/ubuntu/lxh/weavetp/nostop/memory_recheck_$(date +%Y%m%d_%H%M%S)
```

第 2 步 CPU 汇总、Restart 估计及 L600 请求回放见 [ANALYSIS.md](ANALYSIS.md)，实现 SHA 为 `7511ff32d61ab11cafeb1a5aee19841835a32cee`。以下保留第 1 步诊断说明；第 2 步不更改 GPU 执行路径。

## 模块职责与维护边界

本次重构解决入口脚本兼任公共库、配置准备隐式改写 argparse 参数、压力统计回写原始输入的问题。CLI 参数、实验顺序、冻结协议、结果字段、目录结构和失败处理规则保持不变；原公共函数已迁至所属模块，不保留旧入口的兼容转发。

| 模块 | 职责 |
|---|---|
| `diagnostics.py` / `formal.py` | 分别编排诊断与正式实验，处理 CLI、信号和整体退出状态 |
| `supervisor.py` | 子进程组所有权、日志、超时、回收和独立缓存环境 |
| `machine.py` | 空闲与 MPS 规则、机器信息、显存恢复、每秒外部计算进程观测；不结束外部进程 |
| `pressure.py` | stressor 设备隔离、初始化与剩余显存门槛、前后探针漂移比较 |
| `memory_budget.py` | 纯 CPU 解析分阶段 torch 显存证据、计算准入与压力档位预算；不修改 GPU 执行或生成替代测量数据 |
| `synthetic_kv.py` / `cuda_memory.py` | 分别负责初始化/恢复确定性 KV 和阶段显存采样；不修改 Decode、迁移或 cutover 算法 |
| `formal_plan.py` | 解析冻结配置、生成 24 次顺序、校验并保存不可变输入；显式返回 checkpoint，不修改调用者参数 |
| `attempts.py` | 尝试记录、成功判定、重试/污染重跑决策与幂等进度写入；不启动任务 |
| `trace_metrics.py` | 共享 trace 状态验证、rank 对齐和单次 Launch 指标；纯 CPU，无 CLI/执行器依赖 |
| `diag_report.py` / `summarize.py` | 分别输出诊断判定与跨 Launch 汇总，调用共享指标模块 |
| `simulator.py` | 只读加载已有模拟器；`replay.py` 与 `restart_estimate.py` 独立使用，不互相导入 |
| `artifacts.py` | JSON/CSV 读写；需要原子替换的 supervisor 状态使用 `save_json` |

依赖方向为入口 → 职责模块。两个执行入口直接使用同一个 supervisor、机器规则和压力模块；正式入口不导入诊断或汇总 CLI。诊断整批执行与正式实验逐 attempt 执行的生命周期不同，因此保留各自编排，不引入通用工作流框架。共享模块不在 import 时查询 GPU 或启动进程。

在 `bb60cf9` 架构重构时，`benchmark.py`、`benchmark_live_moe_tp.py`、`probe.py`、`p2p_link_stressor.py` 和原 benchmark 启动脚本相对 `8ae22e7` 均无修改。后续 `benchmark.py` 的 prefill/初始化/显存采样及 quick 的启动脚本变更见上文；200 ms GPU 采样和 1 s compute-apps 采样频率保持不变。`verified_execution_overlap_steps` 仍为 null，kernel 级重叠留待配置冻结后，单独用 1 次 nsys run 验证。

重构后 188 项 CPU 测试通过（重构前 174 项）。另外对照重构前实现，确认 24 次顺序、四档 stressor 命令、四种配置的单 Launch 指标、冷启动/压力判定、288 种重试状态、dry-run JSON、跨 Launch 汇总及 Restart 估计相同。尚未重新执行 GPU 实验，CPU 验证不代表服务器性能实测。

CPU 验证命令（仓库根目录）：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
"$PYTHON" -m pytest -q -o addopts='' --noconftest \
  code/current/tools/resharding/nostop/tests \
  code/current/tests/unit_tests/resharding/test_live_defaults.py
```

## 第 3 步：冻结配置后的正式实验入口

前一版重构实现 SHA：`bb60cf9e4eaa52a944aab93b3d43b95994eb231a`，当时 188 项 CPU 测试通过。当前实现见上方显存准入修正，第 3 步最初实现为 `e9348a2e64c18f2ebc5f9efaa82d184bd5d3447a`。

`formal.sh` → `formal.py` 复用第 1 步的 `Processes` supervisor、MPS 空闲规则、stressor 设备隔离、显存门槛及 `probe.py`；不修改 benchmark/GPU 执行路径。`benchmark_live_moe_tp.py` 相对 `8ae22e7` 零 diff；当前外层 prefill 改动见上文。本次交付仅完成脚本和 CPU 测试，不代表已执行 24 次 GPU 实验。

`--frozen` 必填。脚本将文件作为字面量键值读取，**不执行 source/eval**，记录 SHA256 并保存原文件。`DIAGNOSTICS_READY=1`、`NOSTOP_COLD_MODE=cycle` 及可行的实测显存预算都满足才允许真实执行，否则只能 `--dry-run`。读取 `STRESS_LINKS`、`L0_PROCESSES_PER_PAIR` 至 `L3_PROCESSES_PER_PAIR`（缺省按实测预算计算，L0 必须 0）、`STRESS_BYTES`、`MIN_FREE_GIB` 和 `STRESS_SYNC_EVERY`（缺省 16）；新的显存字段见上文，旧的固定 22 GiB 冻结文件不再兼容。`PROFILE` 与 `DIAGNOSTIC_EVIDENCE` 指向诊断画像及证据目录，正式探针使用其中 `probe_routes.json` 的 B0 默认/W2 候选迁移链路并集；背景压力使用冻结的 `STRESS_LINKS`，两者不按正式结果重新选择。checkpoint 必须与诊断相同。画像及路由文件同时保存 hash 和副本。

两种方法直接调用同一 `method_env`，固定 cycle、SWITCHES=3、MAX_WAVES=64、SEQ_LENGTH=MAX_POSITION_EMBEDDINGS=5248、micro batch=4、KV=2048，唯一被测切换索引为 2。不改 W2 门槛，不设置 `NCCL_RUNTIME_CONNECT`，不改锁频。启动前生成全部 24 次顺序并写入不可变 manifest：

| 轮次 | 档位和该档先跑的方法（另一个方法紧接其后） |
|---|---|
| 1 | L0 W2 → L1 B0 → L2 W2 → L3 B0 |
| 2 | L3 W2 → L2 B0 → L1 W2 → L0 B0 |
| 3 | L0 W2 → L1 B0 → L2 W2 → L3 B0 |

共 12 次进入档位，每次在方法对之前/之后各做一次探针，正常完整运行共 24 次探针。探针也经过空闲、压力初始化/显存门槛和结束显存恢复检查。每条有向链路保留全部原始样本，以样本带宽中位数计算 `(after/before−1)`；绝对值 >20% 只标记，不删除或替换数据。重试或续跑的探针独立保存，不覆盖旧数据；如果仅后置探针失败，续跑只补后置探针，不重复已成功的 Launch。补测的时间间隔可由对应 outcome 时间戳复核。

每个 Launch 单独执行：立即检查空闲 → 启动本档压力并确认 START → 每卡剩余显存门槛 → benchmark → 清理自己创建的进程组 → 确认显存 ≤ 本次启动基线+16 MiB。L0 同样检查剩余显存。空闲不通过立即退出，不后台等候；清理或显存恢复失败也停止后续作业。MPS 只读，不关闭、不 kill；外部进程从不加入清理列表。

Launch 期间每秒查询 compute-apps，保存 PID、属主、进程名、每卡显存和采样时间。只允许本 supervisor 持有的未回收子进程组及 `MPS_OWNER` 属主的精确 `nvidia-cuda-mps-server` 名称；其他进程标记 contaminated，记录首次/末次看到及下一次未看到的时间，**不打断当前 benchmark**。时间段是 1 秒采样的观测边界；MPS 未暴露的客户端显存不推算。查询/监测失败会在本次 GPU 执行结束后报告监测不完整，不把未知状态当作干净数据。

同一 `--out` 的并发 supervisor 被非阻塞文件锁拒绝。续跑要求 frozen hash、配置、环境、机器/GPU 身份和顺序不变。退出 0、结果/路由齐全且 8-rank trace 通过状态与时间对齐检查的 Launch 自动跳过，包括成功但 contaminated 的 Launch。失败最多自动重试一次，重试预算跨 supervisor 重启保留；耗尽后停止。`--rerun-contaminated` 只选被标记的 Launch，不看 raw gap、计划或性能，也不启动从未执行的 Launch；每次显式污染重跑是新的尝试组，仍最多一次失败自动重试。不会在一次命令中因结果再次 contaminated 而无限重跑。

目录约定：`attempts/Rn_Lk_METHOD/attempt_NNN/` 保存每次尝试的配置、环境、命令、退出码、stdout/NCCL 日志、`run/result.json`、8 rank trace、routes、压力进程显存、`compute_apps.jsonl`、`external_processes.json` 和显存恢复结果。根目录 `Rn_Lk_METHOD` 是指向最新尝试的相对符号链接，便于第 2 步工具直接读取；旧尝试一直保留。`intent.json` 保存 generation/retry/污染重跑来源；`outcome.json` 保存成功/失败及污染标记。根目录另外保存不可变 manifest、冻结输入、status、PID、`progress.tsv`、`probes/` 全部探针、`sessions/` 每次启动的版本/时钟/温度检查和 200 ms GPU 采样。每完成一次尝试（含失败）追加 progress 一行，失败无指标时留空，污染和漂移从不用于按性能筛选。

**预计机时：**没有正式 Launch 日志时仅能估算。若每次 Launch 3–6 分钟、每次探针 1–3 分钟，24+24 次加检查与压力初始化约 **2–4 小时单机时间**（8 卡约 16–32 GPU·小时）。这不是实测承诺；可根据诊断真实耗时重算 `24×Launch + 24×Probe + 检查/初始化`。失败重试和显式污染重跑另计；每个 GPU 子任务缺省超时 2400 秒。

服务器命令（先使用交付的实现 SHA 更新代码；冻结文件必须来自通过诊断的 run）：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export REPO PYTHON
export MPS_OWNER=yiwei
export FORMAL_IMPLEMENTATION_SHA=8d22890be3e4639165571e15382c2eff656202a5
git -C "$REPO" fetch origin codex/nostop-diagnostics
git -C "$REPO" checkout --detach "$FORMAL_IMPLEMENTATION_SHA"
export NOSTOP="$REPO/code/current/tools/resharding/nostop"
export FROZEN=/data/ubuntu/lxh/weavetp/nostop/实际诊断目录/frozen.env
export FORMAL_OUT=/data/ubuntu/lxh/weavetp/nostop/formal_$(date +%Y%m%d_%H%M%S)
# dry-run 不创建输出目录，不查询 GPU，不启动压力或 benchmark。
bash "$NOSTOP/formal.sh" --frozen "$FROZEN" --out "$FORMAL_OUT" --dry-run
mkdir -p "$FORMAL_OUT"
nohup bash "$NOSTOP/formal.sh" --frozen "$FROZEN" --out "$FORMAL_OUT" \
  > "$FORMAL_OUT/supervisor_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
echo "$!" > "$FORMAL_OUT/nohup_pid.txt"

# 查看进度（首次 Launch 完成后 progress.tsv 才出现）。
cat "$FORMAL_OUT/status.json"
tail -f "$FORMAL_OUT/progress.tsv"
```

普通续跑使用相同的 FROZEN、FORMAL_OUT、环境和命令；仅在旧 supervisor 已退出后执行，不删除旧目录。`FORMAL_OUT` 若在新终端中丢失，重新 export 原路径，不生成新的时间戳：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export REPO PYTHON MPS_OWNER=yiwei
export NOSTOP="$REPO/code/current/tools/resharding/nostop"
export FROZEN=/data/ubuntu/lxh/weavetp/nostop/实际诊断目录/frozen.env
export FORMAL_OUT=/data/ubuntu/lxh/weavetp/nostop/原正式实验目录
nohup bash "$NOSTOP/formal.sh" --frozen "$FROZEN" --out "$FORMAL_OUT" \
  > "$FORMAL_OUT/resume_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
echo "$!" > "$FORMAL_OUT/nohup_pid.txt"
```

需要显式重跑污染结果时，保持上述环境变量，用下面命令**替代**普通续跑命令；不要同时启动：

```bash
nohup bash "$NOSTOP/formal.sh" --frozen "$FROZEN" --out "$FORMAL_OUT" --rerun-contaminated \
  > "$FORMAL_OUT/rerun_contaminated_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
echo "$!" > "$FORMAL_OUT/nohup_pid.txt"
```

确认 supervisor 退出后打包。tar 保留相对符号链接，实际 attempt 数据也在同一包中；不要使用 `--dereference` 重复打包：

```bash
tar -C "$(dirname "$FORMAL_OUT")" -czf "${FORMAL_OUT}.tar.gz" "$(basename "$FORMAL_OUT")"
sha256sum "${FORMAL_OUT}.tar.gz" > "${FORMAL_OUT}.tar.gz.sha256"
```

下面保留旧版本第 1 步的历史记录（长 prefill 已由上方唯一的新路径取代），不要用其中旧 SHA 或命令启动当前诊断。

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
