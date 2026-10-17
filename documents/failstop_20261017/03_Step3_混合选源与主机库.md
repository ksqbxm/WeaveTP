# Step 3：本地代价修复、主机库、profile 与混合布局选源

状态：公共/emulate/real 独立切片测试路径已实现，CPU 已验证；服务器 GPU 待验证。专用 GPU 测试标识：`step03_sources_and_transport`。

## 1. 必读与前置

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 每次修改前核对本 Step、适用 AGENTS.md；Python imports 变化执行 `uv run isort`。出现容量、API 或现有代码不匹配时，先报告具体差异并确认方案。

Step 1 交付后即推进本步模式无关的本地拷贝修复、主机库、profile 采集和混合布局规划，并接入 Step 2 emulate。real 集成等 Step 1 用户放行及 Step 2 real 通过。此安排是与用户服务器测试重叠推进，并不授权占用其正在测试的 GPU。

## 2. 输入及修改边界

检查 `megatron/core/resharding/planner.py`、`copy_services/nccl_copy_service.py` 及现有 metadata/profile 入口，保持原 WeaveTP 贪心选源和分波框架。新增真实覆盖与主机候选，不移植 AnchorTP，不改 NCCL SHM，不附加新的在线 replan/packing。

配置仍为 EDP0=SL3061、EDP1=SL3060，TP4→TP2→TP4。稳定设备身份、world epoch、源分片实际布局来自 Step 2，不从当前全局 rank 推断旧 shard。

## 3. 实施任务

### 3.1 本地拷贝代价修复

定位 `_source_metadata_transfer_seconds` 等代价累加入口：`src==dst` 不进入 L/S/D 网络负载。本地真实需要的拷贝仍执行并计字节、耗时；视图复用不虚报拷贝字节。用可手算的小例子验证去除本地负载改变的是网络竞争估计，不误删目标覆盖或依赖。

### 3.2 每节点一份共享主机权重库

- 每节点独立 CPU owner 从 checkpoint 加载一次，持有一份共享 mmap 和元数据；worker attach 同一份 backing，各自 cudaHostRegister 注册 pinned，不各自持有完整权重。
- owner 不属于 worker kill 范围，worker 退出不 unlink 全局共享库。显式记录 owner PID、node、backing 标识、checkpoint hash、总字节、加载峰值和每个 worker attach/register/unregister 生命周期。
- 加载前核对 RAM、共享空间、memlock/注册能力及临时加载峰值。避免先完整加载多份再宣称共享，预算不足时报错并报告，不静默改为逐 worker 加载。
- 加载前实测 `/dev/shm` 可用字节并与本次 backing、元数据和加载临时峰值比较；SL3061 用户核查仅余约 43 GB，不能按标称容量估算。提供完整库与按需分片两种显式模式，按需模式只加载可能需要的分片（例如故障副本可能缺失的部分），加载前确定覆盖清单，不先完整加载再裁剪。报告各模式预计/实际 backing、峰值与注册字节；预算不足在读取权重前拒绝。CPU 测试覆盖空间不足、按需清单覆盖和无完整库临时副本。此项在 Step 3 实现，Step 1 不加载主机库。
- 库内切片按 dtype、形状、参数名、global expert id、分片范围索引；保持权重只读。跨机不共享同一个 mmap，目标使用目标节点的 host source。
- 新 real worker 只读结构元数据创建 TP4 空目标，权重填充经过统一 planner 和传输任务；记录初始化/H2D 实际字节，不能遗漏预加载流量。

### 3.3 profile 采集

提供独立采集入口，兼容既有 profile 格式并扩展主机聚合容量。包含：每卡 H2D 单独测试、每节点 4 卡及 8 卡并发 H2D、机内 P2P、两方向跨机 P2P；测试数据量覆盖实际迁移分片范围，保存 warmup/重复次数、原始样本和拟合方法。

输出 bw、latency、每节点 M 上限，使用一致字节单位；保存硬件身份、运行时版本、源码 SHA、MPS 隔离状态和 profile SHA256。不得用不存在的条目零成本补齐，也不得把 single-GPU H2D×8 当作节点带宽。采集使用 Step 1 的预检和进程隔离要求；时钟/GPU 状态写入元数据。

### 3.4 混合覆盖与主机代价

- 每个目标切片列出所有真正覆盖该切片的存活源。缩容允许 EDP0 残余旧 TP4 + EDP1 TP4 + 主机；扩容允许 EDP0 TP2 + EDP1 TP4 + 主机。原算法按完整布局挑选源的假设需替换为切片覆盖判断。
- 保留 source metadata 的旧 TP/ETP 含义，不因 world rank 压缩而变形。专家包含 global expert id；不能误把另一 EP 的同名参数当作相同专家。
- 主机任务成本 `t=bytes/bw+latency`，计入目标 D 和节点 M，不计 GPU 发送 S；GPU 路径保持既有负载语义，本地路径遵守 3.1。
- src/dst 为 None 的设备不贡献覆盖。失败 GPU 永不作源/目标；回归空目标也不得作源。同一目标可由 GPU 与 host 分别写互不重叠片段，整体必须完整且恰好覆盖。
- plan cache 包含实际 membership、world epoch、rank map、源布局/coverage、profile hash。对同一固定输入给出稳定可重复的输出。

### 3.5 H2D 执行与逐链路观测

HostTask 在目标 rank 的专用 CUDA stream 提交异步 H2D，与 NCCL 接收共同遵守目标切片依赖、生命周期和完成通知。切换前必须等待所有相关流；共享主机库在最后引用结束前不能注销。不同任务不得重叠写相同片段。

逐链路记录至少包括：event/round/epoch、task/batch id、source_kind、源/目标稳定设备身份与当前 rank、参数/专家/切片、实际 payload bytes、控制 bytes、提交/完成时刻、elapsed_s、effective_bytes_per_s、completion_granularity。H2D 的源身份为 node+host owner/backing，不伪装 GPU rank。

有效吞吐定义为 **实际字节 / 从提交到完成的区间**，包含排队；若只能观察批级完成，所有相关链路标记 `completion_granularity=batch`，保留共用 batch id，不能把相同批耗时累加成总事件时间或解释为独占 wire 带宽。无实际拷贝的本地视图复用单独计 reuse bytes，零耗时吞吐记 null 而非无穷大。

## 4. 本地验证

手工小切片 oracle 验证覆盖无洞/无重叠、global expert 不混淆、部分 TP4 + TP2 混合、host/GPU 同目标不同片段、None/失败源排除、local L/S/D 为零及 host D/M 约束。独立枚举验证 local rank 复用字节最优。

用固定快照、同 profile、同随机种子生成两故障模式计划，归一化临时 rank 后逐项比较源身份、目标身份、切片、字节、依赖/波次。不能只比总字节。真实运行不同 KV 历史时不强求动态计划一致。主机库测试检查单份 backing 和 owner 存活，不以虚拟地址相同判断物理共享。

## 5. 专用服务器 GPU 验收

先 emulate，再在 real 放行后补测同场景，使用独立 checkpoint/切片 oracle：

1. 复现 local copy 场景：L/S/D 修复正确，GPU 输出逐位不变。
2. GPU-only、host-only、混合来源三类目标；覆盖同一目标 host/GPU 互补写入、非连续切片和专家分片，逐位比较。
3. 16→12→16 中主机 owner PID/backing 保持，新 worker 可 attach/register，源死亡后 host 数据仍正确，无每 worker 全权重复制。
4. H2D 与 NCCL 并发，所有目标完整后方可切换；无死锁、缺失写或重复写。失败 GPU 无数据通信。
5. 单卡/4 卡/8 卡 H2D 与机内/跨机 profile 完整；逐链路实际字节合计与任务执行及独立目标覆盖相符，时间区间有效。
6. 相同快照回放下两模式规范化迁移计划逐项相同；如 real 暂未放行，该项标记 pending。

通过标准：所有对应模式切片逐位一致、内存预算通过、host owner 存活、无挂起，profile/链路日志完整。缺失 profile 或 GPU 不可用属于未验收，不能代入历史数字判通过。

## 6. 交付与关口

交付代价修复、主机库/HostTask、profile runner、混合 planner、链路日志和专用 GPU runner；输出 `/data/.../step03_sources_and_transport/<mode>/<launch_id>/`。报告给出 profile hash、各来源实际字节、逐位结果、内存与注册统计、进程位置和生命周期。

按总计划提供完整 SHA、逐节点 source/fetch/checkout/nohup/打包/恢复 `f7d5559` 命令。公共与 emulate 完成可以交付，real 结果独立等待；即使本步通过，Step 4 仍需 fix2 GPU 回传通过。

## 7. 验收登记

实现 SHA：`bcc61aeb86ddc1bcce403785493ac38ad59486db`。99 项 failstop CPU、56 项 packing/选源回归、4 进程 Gloo 通过。profile/emulate/real GPU 与 300 秒服务器预算：未运行。本轮用户明确开始 Step 3 并要求 real 两阶段核查，因此实现 real 测试路径；Step 2 real GPU 通过结论仍待回传，不越过验收关口。详见 [实现与验证](Step3_实现与验证.md)。

2026-10-17 审阅修正：当前实现 `7e61596114b003bf3ecefdd8045a6b10aeeda65b`，99/56/4 项 CPU 检查通过，GPU 仍未运行；问题与修正见[审阅与修正](Step3_审阅与修正.md)。本步 §3.2 的 "memlock/注册能力" 按修正后的语义执行：memlock 只记录，注册能力由共享文件页的真实注册探针在读权重前证明。
