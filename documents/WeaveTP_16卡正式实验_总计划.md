# WeaveTP 双机 16 卡正式实验总计划

状态：方案已确认，本文档不代表代码、部署、历史复算或 GPU 实验已经完成。

配套文档：[分步任务](./WeaveTP_16卡正式实验_分步任务.md)。项目规则：[CODEX.md](../CODEX.md)；涉及实验源码时还须阅读 [code/current/AGENTS.md](../code/current/AGENTS.md)。论文依据：[WeaveTP PDF](../WeaveTP(1).pdf)。

> 每次开始、继续或修改任何分步 task 的代码前，必须重新完整阅读本总计划和项目根目录的 CODEX.md，不能仅依赖聊天记录或上次阅读记忆。涉及 code/current 时还必须阅读适用的 AGENTS.md。先确认本任务范围和验收条件，再修改代码；不得擅自扩大实验范围或调整固定配置。

## 1. 唯一目标与实验范围

只在 SL3060 + SL3061 的 16 卡上运行论文正式实验的 Fixed、Directional、WeaveTP 三组，各 3 次独立 launch，每次执行 2→4、4→2、2→4、4→2。

计算本批次内 WeaveTP 相对两个 baseline 的提升百分比，再与论文 8 卡历史相对提升比较，回答：**扩到 16 卡后，WeaveTP 的相对提升是否扩大。**

- 不重跑任何 8 卡 GPU 实验；历史 JSON 复算是 CPU 数据处理，不是重跑实验。
- 不比较 8 卡与 16 卡绝对秒数，不增加固定总工作量对照。
- 不跑附录五方法代理、图 3b、aware shrink、门槛为零、就近优先、TCP 等额外配置。
- 不修改选源、执行、KV 算法，不调整门槛来取得正收益。
- 只允许多机启动、网卡环境、global batch、指定 wave 上限、16×16 画像、并行组检查、计划统计和结果汇总等适配。

## 2. 固定配置

### 2.1 并行布局与工作量

保持 TP/ETP=2↔4、EP=2、PP=CP=1，每副本 micro-batch=1。本次 global batch=8。DP 翻倍、总迁移量增加属于预期，不称为固定总工作量实验。

| 规模 | TP/ETP | dense DP | EDP |
|---|---:|---:|---:|
| 历史 8 卡 | 2 | 4 | 2 |
| 历史 8 卡 | 4 | 2 | 1 |
| 本次 16 卡 | 2 | 8 | 4 |
| 本次 16 卡 | 4 | 4 | 2 |

启动后从实际 ProcessGroup 读取 dp、expt_dp 的成员与大小，逐 rank 断言与表中 16 卡配置一致。记录 hostname、global/local rank、GPU UUID，以及 TP、ETP、EP、DP、EDP 组成员。确认 TP/ETP/EP 均留在机内；DP/EDP 可以跨机。断言失败停止，不仅记录按公式计算的预期值。

### 2.2 三组配置

| 配置 | METHOD_VARIANT | SCHEDULER_MODE | DISABLE_SOURCE_REROUTE | ADAPTIVE_HYBRID | 扩容上限 | 收缩上限 |
|---|---|---|---:|---:|---:|---:|
| Fixed | baseline | baseline | 1 | 0 | 4096 | 4096 |
| Directional | baseline | baseline | 1 | 0 | 8192 | 4096 |
| WeaveTP | moetp++-hybrid | residual | 0 | 1 | 8192 | 4096 |

WeaveTP 固定 ADAPTIVE_RESIDUAL_MAX_WAVES=4。三组共用：

```bash
NPROC_PER_NODE=8
NNODES=2
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
EXPERT_PARALLEL_SIZE=2
ROUTER_MODE=fixed-hot
ACTIVE_EXPERTS=0,1,2,3,4,5
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024
SWITCHES=4
MAX_WAVE_TASKS=4096
MAX_WAVES=128
MAX_OVERLAP_STEPS=1
PACK_TARGET_BYTES=0
PACK_MAX_ITEM_BYTES=0
ONLINE_REPLAN=0
ONLINE_MIGRATION_FIRST_GUARD=0
ALLOW_AWARE_SHRINK=0
NCCL_DEBUG=WARN
```

使用 DeepSeek-V2-Lite、64 专家；其余参数沿用 wrapper 默认值，包括 PROMPT_TOKENS=8、P2P_ORDER=peer-size-desc。选源门槛保留 REROUTE_MIN_GAIN_PCT=10、REROUTE_MIN_CONTENTION_GAIN_PCT=0、REROUTE_MIN_GLOBAL_GAIN_PCT=5、REROUTE_PENALTY_US=20、REROUTE_MIN_BYTES=1048576。在线 guard、hybrid fast path 等额外开关保持关闭，不继承外部 shell 的实验残留设置。

全部 9 次正式 launch（包括首轮冒烟）两机显式固定 NCCL_DEBUG=WARN，并记录实际环境值。画像/连通性检查可单独使用 INFO；不得把 INFO 带入正式 launch。

### 2.3 wave 上限与失败规则

任务上限按全局任务数计算。16 卡将原 wave 上限乘 2，以尽量保持每卡每波任务量和 wave 数，避免 wave 翻倍及每 wave 一步前台 decode 稀释提升比例。

预期扩容 Fixed 约 12 波，Directional/WeaveTP 约 6 波，收缩约 22 波。报告核对实际值，不为匹配预期自行调参。**8192 卡住时人工确认后立即停止并报告，不回退到原上限，不自动重试。**

## 3. 机器、资源与硬性规则

### 3.1 已知资源，执行时仍需只读核查

| 项目 | 配置 |
|---|---|
| SL3060 | ${MASTER_ADDR}，node_rank=0，master，已配环境 |
| SL3061 | ${NODE1_ADDR}，node_rank=1，两机可以互相 SSH |
| 网卡 | eno1np0，25 GbE |
| RDMA | mlx5_0，RoCE v2，已知 IPv4 GID 条目为 3 |
| 两机工作根目录 | WORK=/data/${SERVER_USER}/lxh/weavetp |
| SL3060 基准代码 | /home/${SERVER_USER}/Megatron-LM-weavetp；来自 <source-repository> commit 2b96fcc 的 code/current，含 async_execution 补丁；目录本身不是 git 仓库 |
| SL3060 checkpoint | /data/models/DeepSeek-V2-Lite-megatron-v2，约 59 GB，torch_dist 格式 |
| SL3060 Python | /home/${SERVER_USER}/miniconda3/envs/megatron/bin/python |
| 已知软件版本 | Python 3.12.13、torch 2.11.0+cu128、CUDA 12.8、NCCL 2.28.9 |
| 历史 8 卡画像 | /home/${SERVER_USER}/Megatron-LM-weavetp/profiles/moe_tp_8gpu_idle_p2p_v21.json，sha256 前缀 3f23ea8b；只读保留，不作为本次 16 卡画像 |

### 3.2 存储、安装与进程规则

- 所有服务器新数据仅写 /data，包括代码副本、conda 环境、checkpoint、输出、报告、缓存和 TMPDIR。两机没有共享存储。
- 不修改或删除 /home/${SERVER_USER}/Megatron-LM-nlh_tp 及其 outputs，不写任何历史 outputs；基准代码和历史画像只读。
- 先检查 /data 空间。SL3060 曾接近 99% 使用率，不假定现在有足够空间，不自行删除数据腾空间。
- SL3060 直接读取已有 /data/models checkpoint，避免复制第二份。SL3061 如缺资源，先列明源、目标、大小与剩余空间。
- **安装任何东西、或复制超过 10 GB 前必须停下征求用户同意。** 不拆分复制绕过限制。不得升级或安装 PyTorch/CUDA/NCCL/TE/Apex。
- 使用既有解释器时关闭向原环境写入 Python 字节码，将 CUDA、Torch 扩展、Triton、HF、pip 等缓存及临时目录重定向到 /data。
- 每次 GPU 任务前检查两机全部 16 卡：无他人 GPU 作业、每卡显存接近约 65 MiB 的既有 MPS 基线才算空闲。发现占用立即停止报告，不等待重试，不挂守候脚本。
- SL3060 上 指定 MPS 用户 的 nvidia-cuda-mps-server 必须保留，不关闭 MPS，不 kill 他人进程。
- 每次失败或中断后，仅清理命令行包含本次 OUT_DIR 的任务进程，确认本任务已退出且显存恢复运行前基线。这里不是要求关闭 MPS 后的物理零显存。不能确认进程归属时不清理，报告剩余问题。

## 4. 最小实现与观测接口

### 4.1 双机启动与协调

通用 launcher 支持 NNODES、NODE_RANK、MASTER_ADDR、MASTER_PORT。双机使用静态 rendezvous，master 为 ${MASTER_ADDR}，默认端口 29500；预检确认可用，不终止占用端口的其他进程。单机启动路径保持兼容，但本次不执行 8 卡实验。

global batch 默认按总 worker 数派生，本次显式为 8。SL3060 唯一驱动 compare，向两机传递相同配置、统一 run ID 和 OUT_DIR。SL3061 不独立运行 compare，也不根据其本地 result.json 决定跳过。

两机分别保存日志，global rank 0 保存结果 JSON。只有两端退出成功、结果完整且配置匹配才标记完成；续跑由 SL3060 统一决定，不覆盖历史或失败证据。所有 GPU 任务命令行均带本次输出路径标签，以便安全定位清理。两机都从 GitHub 检出同一明确 commit，运行前确认 HEAD 相同且 `git status --porcelain` 为空；这取代六个入口文件的 hash 一致性检查，画像文件仍核对 hash。

### 4.2 网络与 16×16 画像

绑定 NCCL_SOCKET_IFNAME/GLOO_SOCKET_IFNAME 到 eno1np0，NCCL_IB_HCA 到 mlx5_0，核查 IPv4 RoCE v2 条目。按已批准方案，NCCL 2.28.9 使用自动 GID 选择，不照搬旧版强制 GID 配置；参见 [NVIDIA 网络排障说明](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html)。

新增轻量画像入口，沿用仓库已有测量方式：

- 先绑定 LOCAL_RANK 对应 CUDA 设备并完成全 rank 通信初始化，再逐对测量。
- 测量 240 个有向非对角对；64 MiB、1 次预热，每轮 3 次传输，3 轮取中位数。
- 带宽用 bytes × 8 / seconds / 1e9，单位为十进制 Gbps；不混用 GB、GiB。
- 对角线沿用现有 profiler 的 1000 Gbps 估计值，并明确为非实测。
- 保存 world_size=16、matrix_gbps、rank/主机/GPU 映射及测量参数；两机使用同一文件并校验 hash，保留原始 NCCL 日志。画像日志/当前 GPU UUID 逐一配对核查暂缓，不作为正式启动阻断项。
- profile 缺失、尺寸错误、非有效带宽或回退默认 100 Gbps 时禁止正式运行。NET/Socket 数据通路也应停止报告，不增加 TCP 对照。
- 画像单独保存 INFO 日志以验证 NET/IB；9 次正式 launch 统一 WARN。

### 4.3 实际执行路径与候选全局筛选

不改变原有选源、执行、KV、自动回退或计时边界。每次 WeaveTP 切换记录：

- base.execution_mode 的实际 FIFO/residual 路径，不能用请求参数 SCHEDULER_MODE=residual 代替。
- 请求计划、实际采用计划、候选可用性、adaptive 决策与回退原因。
- 全局门槛状态：通过、被全局门槛否决、没有形成有效改道候选、未启用/不适用。
- 预测全局收益、全局门槛、global_gate_accepted、rejected_global。

从缓存候选计划保留筛选信息，避免切到 baseline 后丢失原因。这是初始化阶段的筛选结果，不声称每次切换重新筛选。global_gate_accepted=False 本身不能证明门槛否决：需要区分无有效改道、未启用和真正被全局门槛撤销。收缩关闭候选选源，标记“不适用”。候选被拒绝时保留结果，不降门槛、不强制采用候选。

### 4.4 计划流量统计

只从接收操作单侧计数，每个逻辑传输计一次；排除同卡复制后统计远程流量。记录：

- 两个迁移方向的默认计划、候选计划及实际采用计划身份。
- SL3060→SL3061、SL3061→SL3060 字节及跨机总字节。
- 每卡远程发送/接收量，最大值及对应 rank。
- 权重、KV 前缀与增量的分别统计，使用实际有效序列范围，不把容量 1024 当已填充长度。

这是逻辑计划字节，不是网卡实测流量。新增统计在原计时区间外执行，不在迁移热路径增加同步。若被否决前的候选任务表未保留，明确其流量不可取得，不把恢复后的默认计划称为被否决候选。

## 5. 正式运行前：历史 JSON 复算门禁

使用本次新写、随后用于 16 卡的同一个汇总脚本，只读复算：

```text
/data/${SERVER_USER}/moetp_experiments/deepseek_v2_lite_directional_wave_formal_3runs_20260901
```

明确选定正式三配置各 3 份 JSON，核对配置、输入身份/hash、四次切换顺序。不能随意取目录中最晚的 9 个文件，不能混入失败或重复记录。

### 5.1 统计定义

- switch wall = 每次切换的 switch_wall_s。
- migration wall = 每次切换的 base.wall_s。
- Transport = Σ_wave MAX_rank(T_rank,wave)。现有 wave_records[].transport_s 写入前已经跨 rank 取 MAX，因此逐 wave 累加该字段。禁止先对各 rank 求总和再取 MAX；不能以各 stage 时间之和替代完整 Transport。
- 每 launch 内先平均两次扩容、两次收缩、全部四次切换，再对 3 个 launch 求均值与样本标准差；n=3，ddof=1。不丢弃 warmup switch，不将 rank、wave 或单次切换当独立重复。
- 三个时间范围重叠，不相加；switch wall 包含数值检查，不等于客户端停顿。

### 5.2 表 2 验收参考

单位均为秒，单元格为均值 ± 样本标准差。下表是论文显示值，不是本次新结果。

| 范围 | 配置 | migration wall | Transport | switch wall |
|---|---|---:|---:|---:|
| 扩容 | Fixed | 6.398 ± 0.165 | 2.176 ± 0.043 | 7.156 ± 0.122 |
| 扩容 | Directional | 3.888 ± 0.022 | 1.865 ± 0.005 | 4.645 ± 0.036 |
| 扩容 | WeaveTP | 3.589 ± 0.057 | 1.753 ± 0.022 | 4.386 ± 0.081 |
| 收缩 | Fixed | 11.337 ± 0.348 | 3.675 ± 0.148 | 12.130 ± 0.398 |
| 收缩 | Directional | 11.407 ± 0.131 | 3.304 ± 0.085 | 12.183 ± 0.128 |
| 收缩 | WeaveTP | 11.474 ± 0.289 | 3.640 ± 0.186 | 12.291 ± 0.333 |
| 循环 | Fixed | 8.867 ± 0.253 | 2.926 ± 0.095 | 9.643 ± 0.256 |
| 循环 | Directional | 7.647 ± 0.070 | 2.585 ± 0.041 | 8.414 ± 0.076 |
| 循环 | WeaveTP | 7.531 ± 0.155 | 2.696 ± 0.094 | 8.339 ± 0.198 |

核对全部 27 个均值及对应 27 个标准差。每项容差为 0.0005 s + 浮点误差（实现使用 1e-9 s），对应三位小数的舍入区间。

任何超差先分析：输出逐 launch、逐切换、必要时逐 wave 差异，核对批次、字段、单位、分组、重复计数、MAX/累加顺序及标准差。检查原 benchmark 的 Transport fallback：GPU Transport 非正时会采用 wave wall；历史记录不足以确认时说明证据限制，不伪造 rank 时长。

**未复现且差异未解释清楚，不启动正式 16 卡实验。** 不修改参考值或放宽容差来取得通过。分析后若仍不能复现，向用户报告并等待进一步决定。复算只读历史目录，输出写本次 WORK，不启动 8 卡任务。

## 6. 执行、验收与失败清理

| 轮次 | 配置执行顺序 |
|---|---|
| r1 | Fixed → Directional → WeaveTP |
| r2 | Directional → WeaveTP → Fixed |
| r3 | WeaveTP → Fixed → Directional |

首轮三个 launch 兼作实际配置冒烟；成功即作为正式 r1，不额外增加性能 launch，也不因耗时不理想丢弃首轮。

正式前完成 shell 语法、双机参数与协调检查；CPU 测试覆盖统计、逐 wave MAX、流量去重及候选状态分类。完成历史复算和画像门禁后，检查每次 GPU 启动前的双机空闲状态，再执行 r1。

每次核查真实权重加载、实际 DP/EDP、机内 TP/ETP/EP、8192 上限、双向迁移、实际路径、wave 数及数值检查。wave 偏离预期或候选正常回退应记录，不自行调参。无性能收益也是有效结果。

- 卡住、OOM、通信或数值失败立即停止后续队列，不自动重试。
- 单次 launch 超过 60 分钟时，保存日志和 GPU 状态，再按 OUT_DIR 标签清理并报告。运行中由操作者人工监控每次切换；取消单次切换 10 分钟的自动硬限制。
- 清理后核对两机本任务进程已退出、显存恢复 MPS 基线；不能安全清理时停止报告。
- 正式验收要求 9 个完整结果、36 次切换通过原 BF16-relative 检查：NRMSE≤0.4、cosine≥0.93、top-1 门槛为 0。通过不代表逐 token 或逐元素完全相同。

## 7. 报告、历史相对提升与结论

每个阶段结束后集中更新一次 WORK/reports/trial_report.md，不每一步重写。阶段包括资源核查、历史复算、代码验收、部署/画像、首轮验收、正式完成；遇到阻断也记录该阶段的最终状态。

最终报告为 WORK/reports/16gpu_vs_8gpu.md，保留两机日志和原始 JSON，包含：

1. 历史 9 份 JSON 的复算验收、误差和分析。
2. 16 卡三指标 × 三范围的均值±样本标准差，逐 launch 与逐切换原始值，以及相对两个 baseline 的 18 项提升。
3. 8 卡与 16 卡提升并排表、百分点差；按显示到 0.1 个百分点后的数值标记变大、变小、基本不变。这是描述性标签，不是统计显著性判断。
4. 实际 DP/EDP、NCCL_DEBUG、wave 数、WeaveTP 逐次 FIFO/residual 路径、全局候选筛选和其他回退记录。
5. **扩容与收缩两个方向的默认计划跨机字节及双向分布**，连同 WeaveTP 实际计划、权重/KV、收发热点和 wave 数，解释缩容耗时及方向差异。不把全部时间变化都归因于跨机流量。
6. 数值校验、异常、失败、配置偏差及资源清理结果。

提升百分比 = 100 × (baseline 均值 − WeaveTP 均值) / baseline 均值。先求均值再算比值，不平均逐 launch 的百分比。正数表示 WeaveTP 更快。

以下 8 卡历史百分比按用户给出的论文口径直接引用；缺少的三行由表 2 显示均值补算并注明舍入，不用新数据覆盖历史值：

| 指标 | 范围 | 提升 vs Fixed | 提升 vs Directional |
|---|---|---:|---:|
| switch wall | 扩容 | 38.7% | 5.6% |
| switch wall | 收缩 | −1.3% | −0.9% |
| switch wall | 循环 | 13.5% | 0.9% |
| migration wall | 扩容 | 43.9% | 7.7% |
| Transport | 扩容 | 19.4% | 6.0% |
| Transport | 收缩 | 1.0% | −10.2% |

重点解读 WeaveTP 相对 Directional：两者 wave 上限相同，差异主要用于观察选源收益；相对 Fixed 同时混入 wave 大小影响。核对实际执行路径后再解释，不能仅凭方法名称归因。

结论只回答“16 卡相对提升是否扩大”。必须写明：历史来自 09-01 批次，代码与机器状态不同，只比较各自批次内相对提升；收缩未启用候选选源，没有提升属于预期行为，不是故障。聚合流量与计时不能证明唯一硬件瓶颈。
