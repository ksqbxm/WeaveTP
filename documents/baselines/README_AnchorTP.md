# AnchorTP：状态保留/弹性恢复机制代理

## 原论文与为什么选

[AnchorTP: Resilient LLM Inference with State-Preserving Elastic Tensor Parallelism](https://arxiv.org/abs/2511.11617) 面向 GPU 故障后的弹性 TP 恢复，涉及状态保留、非等宽分片和最小迁移等机制。论文主页注明 DATE 2026。它与 WeaveTP 都关注参数/KV 以及弹性布局，但触发事件和具体协议不同。

## 本项目实际完成了什么

本分支的 **AnchorTP-proxy** 实现标识为 `anchortp-fixed-paths-v1`，起点为 `codex/standby-weight-storage` 的 `beda9f120c4711763ea5caeb8d5861ee753b753d`。初始化同步、扩容和缩容统一执行固定源的交集计划，按静态链路带宽降序、同带宽按原任务编号排序。Python 入口强制关闭 WeaveTP 改道、aware shrink、residual、online replan、adaptive/hybrid、guard 与 packing；显式 source-search 配置直接报错。

每次切换冻结解码和 KV offset，先重建并 poison 目标权重（启用权重释放时），再一次性准备整个计划的接收/非连续写回缓冲。各 rank 完成 GPU 准备并同步后才提交第一批传输。任务提交保留批量上限，波数上限不截断任务；后续批次不再分配接收缓冲，OOM 不退回分批分配。迁移及权重检查期间无解码和 KV 增量追赶，接管后保留正确性校验，完成后释放旧权重。

执行器统一绑定当前发送准备 stream，每批同卡复制和远程传输都等待非连续源切片准备完成；初始化及 `RELEASE_STANDBY_WEIGHTS=0/1` 使用同一执行路径。AnchorTP 的波数上限及最小/最大 overlap 步数统一置 0（不限制波数、无重叠解码）。KV 容量仅按 prefill、每次切换后校验及每个不同负载阶段的首次校验计算，不再使用旧的每波/权重检查解码预算。

GPU→GPU PCIe、Host→GPU、高速 GPU 互联分别统计；同卡复制和跨机网络独立统计。同机当前 RTX 5090 拓扑按 PCIe 处理；其他硬件需在静态带宽 profile 的 `link_types` 矩阵提供实际 `local/pcie/nvlink/xgmi/network` 类型，不能由带宽大小猜测 XGMI。跨机始终归为 network，XGMI/NVLink 标签需与 ROCm/CUDA 后端一致。

Host 接口接收显式 CPU 张量，在整体预分配时准备 pinned staging，使用独立 stream 异步上传，并保持资源有效直到完成。当前 GPU 状态完整的扩缩容不会生成 Host 任务，也不增加 CPU 权重/KV 备份。仍未实现故障检测、CMM 缺口生成、daemon、非等宽 TP、KV 重算和 EPLB。

每次切换记录 `prealloc_s`（含 GPU 准备等待的跨 rank 最大值）、`pause_s`、实际调度模式、各路径任务数/字节数、实现版本和代码提交。预分配计入 `base.wall_s`、暂停和 `switch_wall_s`；后者仍含接管后验证。初始化记录在 `initial_sync`，不混入正式切换。`reload_bytes` 来自真实接收侧 Host 字节的跨 rank 求和；正常切换逐次要求 `reload_bytes=overlap_steps=delta_tokens=0`。停服期间没有 TPOT 样本，兼容字段的 0 不代表零延迟。

设备与设置：8 × RTX 5090、DeepSeek-V2-Lite、TP2↔TP4；历史批次 5 次独立 launch，每次 4 次切换。

## 已有结果

以下仅为 **旧代理（含 WeaveTP residual 与改道）** 的历史结果；本分支尚无新的 GPU 性能结果。历史数值与结论不得归给新实现或合并求均值。

| 指标 | AnchorTP-proxy：旧代理（含 WeaveTP residual 与改道） | WeaveTP（同一历史批次） |
|---|---:|---:|
| Base wall | 19.472 ± 0.345 s | 8.787 ± 0.135 s |
| Transport | 2.250 ± 0.162 s | 3.092 ± 0.084 s |
| Switch wall | 20.255 ± 0.351 s | 9.585 ± 0.156 s |

WeaveTP 相对旧代理的 switch wall 低约 52.7%，Transport 却高约 37.4%。该历史分析只对应旧代理；并不证明 WeaveTP 对新实现或官方 AnchorTP 更快。不能把故障恢复与预先安排的 TP 切换当成同一工作负载。

## 继续复现的步骤

直接使用现有 selected-baselines 或周期入口；`METHOD_VARIANT=anchortp-proxy` 在 Python 层绑定上述行为。周期脚本默认导出 `codex/anchortp-faithful`；未来可用 `ANCHORTP_REF` 指向已经合入此分支的实验分支。导出无 .git 的快照时传递 `ANCHORTP_CODE_COMMIT`（周期脚本已传）；同时保存文件哈希，不能把修改后的代码仅凭提交号当成原样代码。

本分支兼容 `RELEASE_STANDBY_WEIGHTS=0/1`。合入 dp-sweep 时再解决 TP 参数化和 `INIT_SYNC_MAX_TASKS`：初始化整体预分配不分批，提交仍受初始化上限约束。当前不合并其他分支，不重跑历史批次。

CPU 验收命令（在 code/current 下）：`python -B -m pytest -q -o addopts= --confcutdir=tests/unit_tests/resharding tests/unit_tests/resharding/test_anchortp.py tests/unit_tests/resharding/test_anchortp_streams.py tests/unit_tests/resharding/test_live_storage_lifecycle.py`。stream 队列模型覆盖同卡/远程传输读取未完成源切片的反例，协调器测试覆盖容量边界及负载阶段去重。真实 GPU 专项：`torchrun --standalone --nproc-per-node=3 tests/unit_tests/resharding/test_anchortp_cuda.py`；跨机需改用对应 rendezvous 参数。覆盖独立 Host 上传、P2P 混合、空任务 rank、非连续接收及释放后重建，调用方不再手动补上 producer stream。CPU 验收不能替代 GPU 异步、重叠和跨机验证，本机没有 GPU，后者尚未实测。

以后编写故障容错时，将现有交集算法接成 AnchorTP 可用的 CMM 缺口生成接口，再向 Host 执行器提供实际缺失张量；本次不生成缺口或备份。

历史证据仍是历史五轮的 5 份旧代理记录，复现需检出对应旧提交；当前入口已修正，不能用于声称复现历史行为。原始数值与证据编号保留。
