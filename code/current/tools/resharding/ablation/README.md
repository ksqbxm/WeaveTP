# WeaveTP 组件消融（TP2→TP4，EP=2）

只采数据，不出图。两个测量入口共用 `driver.py`，用 `--decode-mode` 区分：

- **pause**（当前主线）：迁移窗口内不跑前向，测纯数据搬运时间 T_data。
- **overlap**：与 run_r1 的行为相同，迁移期间连续 decode。脚本先保留，暂不运行。

| 入口 | 脚本 |
|---|---|
| pause | `run_pause_smoke.sh`、`run_pause_row_b.sh`、`run_pause_row_a.sh`、`run_pause_xnode.sh` |
| overlap | `run_overlap_row_b.sh`、`run_overlap_row_a.sh`、`run_overlap_xnode.sh` |

所有脚本共用 `common.sh`。服务器上的完整命令见 `documents/ablation_8gpu_20261020/服务器命令.md`。

## 开关
- `WEAVETP_ABLATION_LEVEL` 列出要跑的级别（脚本默认 `L1,L2,L3,L4`）。不设置时 driver 直接报错退出；核心库不读这个变量。
- `planner.py` 有三个默认关闭的 kwarg：`source_accumulate_load`、`source_global_gate`、`source_allowed_ranks`。不设置时，规划结果与 beda9f1 一致，由 `test_ablation.py` 保证。`source_allowed_ranks` 目前保留但不再使用。
- 以下文件均未改动：B（`examples/rl/benchmark_live_moe_tp.py`）、16 卡文件、`async_execution.py`、`live.py`、`utils.py`、NCCL copy service。

## 四个级别
所有分叉都在 `driver.level_kwargs` 里，未知级别直接 KeyError。四个级别共用同一套执行器：

- `B._run_async_waves`；
- 同一个 `NCCLCopyService`（peer-size-desc，不打包，带 participation marker）；
- 固定 FIFO，每个 wave 最多 4096 个 task；
- 同一个 delta 流程。

| 级别 | 规划调用 |
|---|---|
| L1 | 不传带宽，`prefer_local_source=False`（DP 轮询选源），另加补字节 |
| L2 | 默认参数（现有默认计划，本地优先） |
| L3 | 正式 policy kwargs，另加 `source_accumulate_load=False`、`source_global_gate=False` |
| L4 | 正式 policy kwargs（θ10%、δ20µs、B_min 1MiB、λ5µs、门槛 5%），即现有贪心 |

L3/L4 额外传 `{"algorithm": "greedy"}` 的 source_search_config。它不改变选源，只是让 planner 记录 `predicted_H_s` 和 `gate_outcome`。

**L1 补字节**：
- 每个 (src, dst, src 参数) 的补字节数 = 源分片字节 − 所需字节；KV 按 `[0, P)` 行计算。
- 补字节锚定在本组最小的 task 上，用负的 task_id（-1 留给 participation marker）。
- 在锚定 task 所在的 wave 里，用同一个 batch 收发。
- 发送端共用一块只读 tx 缓冲，接收端复用同一块 rx 缓冲，收到的数据不读。
- 补字节不计入调度器的 task 字节，但计入每个 wave 每个 rank 的字节统计。

## 测点与流程
| 入口 | 测点 |
|---|---|
| row_b | 27 层，一次 launch，依次测 `none`、`0:12,1:12`、`0:16,1:16`、`0:20,1:20`、`0:24,1:24` |
| row_a | n = 3 / 7 / 14，每个 n 一次 launch，不灌流。n=27 在 summary 里引用 row_b 的 none 点 |
| xnode | 两台机器各用 GPU0–3。SL3061 跑 `NODE_RANK=0`（master，10.60.14.2:29531），SL3060 跑 `NODE_RANK=1`；static rendezvous。NCCL 网络设置与 16 卡正式脚本相同。不灌流；"跨域"指跨机 |
| smoke | pause 模式，27 层，只测 `0:24,1:24` |

每个测点的流程：
1. 必要时切换灌流。
2. 新的灌流配置测 2 遍画像取平均（`tp_build_src_dst_demo.profile_p2p_bandwidth_matrix`，64 MB，2 次迭代）。overlap 模式下再测 10 步 TPOT 基线。
3. 构建 L1–L4 计划，按 (级别, 测点) 缓存；L1/L2 每次 launch 只规划一次。
4. **生效检查**：
   - 计算 L3 相对 L2 的改路字节，以及 L4 相对 L3 的计划差异字节。
   - 在灌流点或 xnode 点上，如果 L3 改路字节为 0，或 L4 的 `gate_outcome` 不是 accepted，就标记 `inactive` 并打印警告，照常测完。
5. **预热**：只在每次 launch 的第一个测点做，每个级别 1 次。
6. **计时**：固定 3 轮，每轮用固定种子打乱级别顺序。
7. **数值校验**：每次 launch 中每个级别的第一次计时迁移，跑切换后的那一步 decode 并调用 `B._validate_cutover`。失败会 raise，进程以非零码退出。
8. launch 结束时复测最后一种配置的画像，`ratio_mean` 偏差超过 15% 时，该配置的所有行标记 `unstable`，不重跑。

每次迁移前只回卷 KV，并新建 tracker 和 scheduler。

**pause 模式**：
- patch 后的 `_decode` 直接返回，`live_max_overlap_steps=1`。每个 wave 发出后立即 wait，KV 不推进，所以 delta 为空。
- 切换后的那一步 decode 只在做校验的迁移上执行。

**smoke 额外做的事**：
- 先做一次不计时的独立规划：L4 的 kwargs 加 `source_search_audit_path`。这次规划只用来把 planner 元数据导出到 `--ablation-metadata`（smoke 输出目录下的 `metadata.pkl`），留给以后的离线工具。
- 结束时每个级别打印：T_data 中位数、T_data 的 CV、wave 设备时长 CV 的最大值、改路字节、发送端 max/mean、瓶颈 rank 分布。
- 另外打印峰值显存，以及 L3/L2、L4/L3 两个 T_data 中位数比值。

## 指标口径（result.jsonl，每次迁移一行）
- **时间原点**：sync 加 gloo barrier 之后，在 current stream 上记一个 event。各阶段的 event 来自 `set_launch_callback` 拿到的 nccl/local stage。
- `wave_rank[k][r]`：第 k 个 wave 在 rank r 上的数据：
  - `nccl` 和 `local` 两个阶段的 `[start, end]`，单位 ms，从原点算起，没有该阶段时为 null；
  - `send_bytes`、`recv_bytes`、`local_bytes`、`pad_bytes`、`tasks`。
- **每个 wave**：
  - `wave_dev_ms`：设备时长，取各 rank（该 wave 最晚 end − 最早 start）的最大值；
  - `wave_net_ms`、`wave_local_ms`：nccl 和 local 阶段时长，各取各 rank 的最大值；
  - `wave_bottleneck_rank`：最晚结束的 rank；
  - `wave_wall_ms`：主机侧 wall。
- **整次迁移**：
  - `T_data_ms` = Σ wave_dev_ms；`T_data_net_ms`、`T_data_local_ms` 同理求和；
  - `span_ms` = 最后一个 end − 第一个 start；
  - `gap_ms` = span − T_data。
- **切换分解**：
  - `switch_wall_ms` 从窗口开始计到 delta 完成，不含切换后的那一步 decode；
  - 分解为 `T_data_ms` + `gap_ms` + `commit_ms` + `other_ms`（其余全部）。
- **计划层面**（同时写入 `plans.jsonl`，那里还有 8×8 的 `link_bytes` 矩阵）：
  - `network_bytes`、`local_bytes`、`pad_bytes`、`required_bytes`（等于 L2 的 task 字节）；
  - `cross_bytes`：跨 socket 字节，xnode 时为跨机字节；
  - `reroute_sender_bytes`（换发送方）和 `reroute_local_to_net_bytes`（本地拷贝改走网络），都相对 L2 计算；
  - `sender_bytes_max/mean`；
  - L3/L4 另有 `predicted_H_s`、`gate_outcome`。
- **overlap 额外**：`tpot_ms`、`token_interval_ms`、`baseline_tpot_ms`。

## 输出
- 每次 launch 一个目录，内含：`result.jsonl`、`plans.jsonl`、`launch.json`（画像矩阵、带宽差、复测结果、TPOT 基线、域划分、耗时）、`run.log`、`gpu_before.txt`、`gpu_after.txt`。smoke 目录里还有 `metadata.pkl`。
- `summarize.py ROOT` 生成 `ROOT/summary.csv`：
  - 每个 (mode, row, 层数, flood, level) 一行；
  - T_data、T_data_net、T_data_local、span、gap、最大 wave、最后 wave、switch_wall 这几项，各给出 median / min / max；
  - 附带 n、计划层面的字节类列、预测 H、inactive、unstable、ref_point。
- 预检：`/data` 剩余不足 1 GB 时拒绝运行；正式脚本要求本机使用的 GPU 连续 10 s 利用率为 0。
- 每次 launch 后检查输出（不含 `cache/`）是否超过 200 MB。

## 已知影响
本分支改了 `planner.py`，旧的 source-search 批次在 `summarize_source_search_bench.py --check-current` 下会因 SHA 不同而被拒绝。这类批次请在 beda9f1 上复核。
