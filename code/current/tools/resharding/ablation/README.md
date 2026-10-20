# WeaveTP 8 卡组件消融（TP2→TP4，EP=2）

只采数据（result.jsonl / summary.csv），不出图。进程由人在服务器上启动：
- 每行一个脚本：`run_row_a.sh`、`run_row_b.sh`、`run_row_c.sh`；
- 冒烟用 `run_smoke.sh`。

服务器命令见 `documents/ablation_8gpu_20261020/服务器命令.md`。

## 开关
- `WEAVETP_ABLATION_LEVEL` 列出要跑的级别，例如 `L1,L2,L3,L4`（脚本默认值），或单个 `L3`（调试用）。
- 不设置这个变量时，driver 直接报错退出。核心库不读这个变量。
- `planner.py` 新增三个 kwarg，默认值下行为与 beda9f1 一致（由 `test_ablation.py` 对比 beda9f1 的 planner 保证）：
  - `source_accumulate_load`
  - `source_global_gate`
  - `source_allowed_ranks`
- `async_execution.py`、`live.py`、`utils.py`、NCCL copy service、`examples/rl/benchmark_live_moe_tp.py` 都没有改动。

## 四个级别在代码里的分叉点
所有分叉都在 `driver.level_kwargs` 里（未知级别直接 KeyError），执行器完全共用：
- `B._run_async_waves`；
- 同一个 candidate 配置的 `NCCLCopyService`（peer-size-desc，无 packing，带 participation marker）；
- 同一套波次规则；
- 同一个 delta 流程。

| 级别 | 规划调用 | 选源在哪里分叉 |
|---|---|---|
| L1 | `build_centralized_reshard_plan(prefer_local_source=False)`，不传带宽 | `select_src_metadata_balanced` 的 DP 轮询分支，不做本地优先；另加补字节（见下） |
| L2 | `build_centralized_reshard_plan()`，默认参数 | 本地优先选组（`utils.select_src_metadata_balanced` 的 local 分支）。实验 A 的默认计划就是它，也是 L4 被门槛拒绝时退回的计划 |
| L3 | 正式 policy kwargs，加 `source_accumulate_load=False`、`source_global_gate=False` | `planner.py` 中两处 `*decision_loads`：贪心选候选的代价和 θ 判定，都不带已分配负载；全局门槛跳过（`global_gate_accepted` 只看有没有改路） |
| L4 | 正式 policy kwargs（θ10%、δ20µs、B_min 1MiB、λ5µs、门槛 5%） | 现有贪心，参数全部不变 |

- **行 c**：L3、L4 再加 `source_allowed_ranks`，取值是两个 NUMA 节点交替排列后的前 m 张卡。它只在贪心候选处生效：候选列表裁剪成默认源，加上实际发送卡全部在集合内的候选。
- **行 c 的 L1/L2**：计划与 m 无关，只在 m=8 跑；summary 里 m=4/2/1 的 L1/L2 引用 m=8 那一组（`ref_point=m=8`）。
- **L3/L4 的 source_search_config**：两者都额外传 `{"algorithm": "greedy"}`，只为让现有代码记录 `predicted_H_s`，不改变选源。

### L1 补字节
L1 的要求是实际搬运的字节包含不需要的部分。做法：
- 对每个 (src, dst, src 参数)，补字节数 = 源分片字节数 − 所需字节数。KV 的源分片按 `[0, P)` 行计算。
- 规划后做一次 gloo `all_gather_object`，所有 rank 得到同一份列表。每条记录锚定在本组最小的 task_id 上，自己用 −2、−3… 作为 task_id（−1 是 participation marker）。
- 执行时 driver 在进程内包装 `B.launch_reshard_plan`：wave 含锚定 task 时，在同一个 batch 里补发和补收。
- 缓冲：所有发送共用一块只读 tx 缓冲，所有接收复用同一块 rx 缓冲（数据不读）。大小取单条记录的最大值。
- 同卡的补字节走 service 现有的本地拷贝路径。
- delta 阶段不补。
- **补字节不计入调度器的 task 字节**：residual 调度器和 wave 切分只看计划里的 task。result 里另外记录每个 wave 的 `wave_pad_bytes` 和 `wave_bytes`（task 字节 + 补字节）。

## 执行器参数（正式 WeaveTP case）
- 扩容 wave 上限 4096，`MAX_WAVES=128`。
- adaptive hybrid 只保留波次规则：`ceil(tasks/4096) ≤ 4` 用 residual，否则用 FIFO。计划从不被替换成 baseline 变体；adaptive 本来是否会退回，记录在 `adaptive_would_fallback/adaptive_reason`。
- packing、online replan、guard 关闭，`RELEASE_STANDBY_WEIGHTS=0`。
- decode 连续：`live_min_overlap_steps=1`，`live_max_overlap_steps=256`。
- 和正式 B 的差异：正式 B 在 `ADAPTIVE_HYBRID=1` 时，默认计划走的是不带 participation marker 的 baseline 服务；这里所有级别共用 candidate 服务。

## 测点与流程
- 模型：DeepSeek-V2-Lite 结构，随机初始化（`--seed 1234`），不加载 checkpoint；行 a 用 `--num-layers n`。
- torchrun 进程数 = `nvidia-smi -L` 看到的 GPU 数；rank r 用 PCI 地址顺序的第 r 张卡（`CUDA_DEVICE_ORDER=PCI_BUS_ID`）。
- 规划：每个（级别 × 测点）只规划一次。L1/L2 整次 launch 只规划一次；L3/L4 按（灌流配置, 允许集合）缓存。规划耗时只记录在 `plan_build_s`，不计入任何指标。
- 每次迁移前只把 src/dst 的 KV 回卷到 P（prefill 后的长度），并重置 token 计数；每次迁移新建 tracker 和 scheduler，用该配置的画像初始化。
- 每个测点跑 1 轮预热 + 3 轮计时（环境变量 `REPEATS` 可改，例如 `REPEATS=2`），每轮用固定种子打乱级别顺序。
- **带宽画像**：
  - 复用 `tp_build_src_dst_demo.profile_p2p_bandwidth_matrix`（64 MB、2 次迭代），调 3 次取平均。
  - 每种灌流配置测一次，同配置的测点共用；L3/L4 用这份画像规划。
  - 每次 launch 结束时复测最后一种配置，`ratio_mean` 偏差超过 15% 时，该配置的所有行标记 `unstable=true`，不自动重跑。
- **TPOT 基线**：每种（层数, 灌流）配置跑 10 步不切换的 decode，取中位数。
- **灌流**：rank0 用 `Popen` 启动 `cpu_flood.py local <node> <idx> <TAG>`，等待全部打印 `ready`。
  - `cpu_flood.py` 是服务器脚本的原样副本，sha256 `f7659988…d8c9ea`。
  - 切换配置或退出时 terminate；bash 在 torchrun 结束后再执行 `pkill -f $TAG`。
- **绑核**：只读 sysfs（NVIDIA 显示类 PCI 设备按地址排序后的 `numa_node`；`node*/cpulist`），在 import torch 之前 `sched_setaffinity`。每个 rank 绑到本节点 cpulist 前段的 `ABLATION_RANK_CPUS` 个核（默认 2），灌流用 `cpus(node)[-1-idx]`，两者重叠时直接报错。

## 指标口径
- **`transfer_ms`（主指标）**：每个 wave 的设备侧时长之和。
  - 每个 wave 的设备侧时长：通过 `NCCLCopyService.set_launch_callback` 拿到该 wave 的 nccl/local stage event，在每个 rank 上取 max(end) − min(start)，再在各 rank 之间取最大值。
  - 不包含 wave 之间等 decode 发现完成的空档（实测一个 decode 步约 0.32 s）。
- **`transfer_span_ms`（辅助）**：从第一个 start 到最后一个 end。所有 event 都相对原点 event 计算，原点在窗口前 `cuda.synchronize` 加 gloo barrier 之后打。
- **`goodput_gbps`**：所需字节 × 8 ÷ `transfer_ms`。所需字节 = L2 计划的全部 task 字节（远端 + 本地），四个级别共用这个分子。
- **wave 级**：
  - `wave_dev_ms`：设备侧时长。summary 的"最大 wave 时长"和"最后一个 wave 的时长"都用它。
  - `wave_wall_ms`：主机侧 wall。
  - `wave_bytes`：task 字节 + 补字节；`wave_pad_bytes`：其中的补字节。
- **`switch_wall_ms`**：从窗口开始，到 delta 提交、目标就绪为止。
- **decode**：
  - `tpot_ms`：窗口内每个 decode 步的设备时间，取各 rank 最大值。
  - `token_interval_ms`：真实 token 间隔。依次包含窗口起点到第一个 token 的间隔、wave 之间的 exposed 间隔，以及最后一段（含 delta）到目标模型上第一个 token 的间隔。
- **字节**：
  - `network_bytes`：远端字节，含补字节；`local_bytes`：本地拷贝字节，含本地补字节。
  - `pad_bytes`：补字节（远端 + 本地）。
  - `sender_bytes`：每个发送端的远端字节（含补字节），附 max/mean。只统计字节，不测忙时。
- **规划**：
  - `predicted_H_s`：只有 L3/L4 有，来自 planner。
  - `projected_global_gain_pct`、`global_gate_accepted`；L3/L4 的 `global_gate_would_reject` = `projected_global_gain_pct < 5`。
- **带宽差**：
  - `ratio_mean` = 同节点卡对平均带宽 ÷ 跨节点卡对平均带宽；
  - `ratio_0to1` / `ratio_1to0` = 同节点平均 ÷ 该方向跨节点平均。

## 正确性
只用现有的切换后数值校验 `B._validate_cutover`（bf16-relative：nrmse ≤ 0.4，cosine ≥ 0.93）。smoke 里每个级别各做一次，开灌流后再做一次。正式行不做校验。

## 输出
- `$ROOT/<行>/<launch>/` 下：`result.jsonl`（每次迁移一行）、`launch.json`（画像矩阵、带宽差、复测结果、TPOT 基线、L2 计划网络字节中跨 socket 的占比、绑核、耗时）、`run.log`、`gpu_before.txt` / `gpu_after.txt`（nvidia-smi 快照）。
- `summarize.py $ROOT` 生成 `$ROOT/summary.csv`。
- 预检：`/data` 剩余 <1 GB 拒绝运行；正式行要求连续 10 s 所有 GPU 利用率为 0%。
- 每次 launch 后检查输出（不含 `cache/`）是否超过 200 MB。

## 已知影响
本分支改了 `planner.py`，旧的 source-search 批次在 `summarize_source_search_bench.py --check-current` 下会因 SHA 不同被拒绝。这类批次请在 beda9f1 上复核。
