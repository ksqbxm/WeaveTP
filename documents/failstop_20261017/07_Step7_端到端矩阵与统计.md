# Step 7：端到端矩阵、恢复指标与统计报告

状态：待 Step 6 正确性通过。专用 GPU 测试标识：`step07_e2e_acceptance`。

## 1. 必读与前置

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 每次代码修改前核对本 Step、适用 AGENTS.md；Python imports 变化执行 `uv run isort`。统计定义变化必须同步总计划并报告，不事后选择有利口径。

前置：两模式阶段 A/B 正确性通过、fix2 和 real 关口均放行、用户确定 ShareGPT 子集、完整 profile、内存槽位预算和配置冻结。初始无故障 TP4 标定选出约 108 s 工作量，记录请求 hash；正式实验所有组使用相同请求 ID/token/顺序和预热规则。

## 2. 正式 GPU 矩阵

| 组 | 次数与启动方式 | 场景 |
|---|---|---|
| WeaveTP emulate | 5 轮，一次 launch，`--rounds=5` | 每轮正常→故障缩容→回归扩容；轮前恢复 TP4 并校验权重 |
| WeaveTP real | 3 轮，每轮独立 launch | 真实四 PID kill，12/16 world 重建；每轮 MPS/回收检查 |
| 无故障 TP4 | 5 次 | 两副本正常固定布局，全量同请求 |
| 降级标定 | 3 次 | EDP0=SL3061 TP2×DP2/ETP2×EP2，EDP1=SL3060 TP4 固定布局，无故障事件 |

提供 `run_failstop_e2e.sh` 和汇总入口，记录真实运行顺序、launch/round id、时间、环境、代码与数据 hash。连续 emulate 轮次与独立 real 启动的差异须在报告中说明。所有正式轮均保留，包括失败和重跑；重跑有新 id，不覆盖旧证据。

SL3060 MPS 服务不操作，worker 启动隔离并核验；每节点一个监管器，TCPStore 位于 SL3060，回归 worker 由 SL3061 拉起。GPU 使用冲突预检失败时停止该轮，不杀别的任务。

## 3. 测量窗口与基础计数

- 总完成时间从固定请求集提交到最后请求完成，排除模型初始加载和预热，包含排队、full prefill、故障恢复与 decode。单列启动/预热开销，不混入服务完成时间。
- 逐秒保存各副本和全局新增 token 数，以及正在服务布局、active slots、队列长度、KV_used、事件状态。只计成功提交的新 token，replay/padding/empty forward 不计。
- request submit 时间不可因故障回队重置；TTFT=首个有效新 token−首次提交。逐请求 TPOT 用首 token 到末 token 的时间除以其后有效 token 数，包含停顿；仅一个输出 token 的请求该指标记 null 并说明。
- 提供全程和 fail→full recovery 窗口的 P50/P99 时延：窗口 TTFT 样本按首 token 落在窗口的请求选取，仍从原始提交计时；窗口 TPOT 按相邻新 token 间隔与窗口相交的请求，保留完整间隔后计算每请求均值再分位数。记录样本数，不能将停顿跨窗口的间隔截断。
- 控制器统一事件时基，GPU 本地计时单独保存；未校准跨机时间不直接相减。所有派生指标必须能由原始日志复算。

## 4. 恢复指标与参考吞吐

固定恢复判据：**5 s trailing mean 连续 3 s ≥ 参考吞吐的 95%**。每秒窗口需要完整有效样本，不以短尾窗代替完整窗口；记录首次满足时刻。

| 名称 | 起点 → 终点 / 公式 |
|---|---|
| shrink TFS | fail 命令 → EDP0 新 TP2 首个正确 token |
| shrink TTP | 上述首 token → 全局吞吐达到固定降级参考的恢复判据 |
| expand time | 实际 rejoin 启动 → EDP0 TP4 ready |
| expand TTP | TP4 ready → 全局吞吐达到故障前 10 s 平均吞吐的恢复判据 |
| full recovery | fail → 扩容后达到上述正常参考判据，包含 45 s 等待 |
| EDP0 downtime | 每次事件无可提交有效 token 的服务暂停区间，分缩容、扩容输出 |
| EDP1 pause | real abort/rebuild 导致健康副本暂停的区间，分事件和总量 |
| completion overhead | 本轮总完成时间 − 无故障五次有效运行总完成时间均值 |
| normalized throughput | 本轮有效 token/本轮完成时间 ÷ 无故障参考吞吐均值 |
| rejoin_delay_s | 实际 rejoin 启动 − fail 后 45 s 到期，列出 shrink/reclaim 等原因与重叠 |
| pre_expand_throttle_s | 实际扩容前限流区间；未实现限流则 0，不能把等待/停顿改名隐藏 |
| estimated lost tokens | max(0, R_ref×窗口秒数−窗口实际新 token)；不同重叠窗口不相加 |

降级参考：三次固定 TP2+TP4 运行，各自在 completed_count 的 20%–80% 区间筛选两副本槽位均饱和的秒样本，先报每轮有效秒数及均值，再给组均值作为 R_degraded。相关吞吐窗口所需完整秒数不足时标无效，不能用排空期低值降低门槛。

扩容和 full recovery 参考为同一轮 fail 前连续 10 s 的全局均值，保存完整原始样本。缩容 estimated lost tokens 使用 R_degraded，扩容/full recovery 使用故障前参考。它们是参考吞吐下的估计损失，不声称是精确反事实。

缩容在扩容开始前未恢复记 censored；其他阶段到工作结束仍未恢复也记 censored。故障前不足 10 s、无有效降级样本、队列提前排空不能写 0；相关指标标记 invalid/censored、原因及观测截止时刻。不能额外跑空请求等待凑恢复。

## 5. real 分项与显存回收

分别保存 fail 阶段和 rejoin 阶段的每进程 abort/rebuild 开始结束及控制器汇总墙钟：

- `abort_s`：对应 epoch 旧 communicator abort 区间。
- `rebuild_s`：该新 world 建组开始到可用；12 和 16 world 分开，不只留一个总数。
- `rejoin_join_s`：rejoin 就绪请求至 16 world ready，包含当时剩余回收等待、spawn、注册与重建。
- `gpu_reclaim_s`：每卡 kill 到无残留本次进程且显存回基线稳定；整体记录四卡最后完成。作为 join 报告的回收子项保留原始区间；落在 join 内的等待另列 `reclaim_wait_s`，早已完成则为 0。

这些区间可重叠，汇总保留 begin/end 与 elapsed，不把 gpu_reclaim_s、rebuild_s、rejoin_join_s 简单相加。重建前后的 EDP1 暂停必须计入其服务时间线。

统一采用总计划时刻：`t_rejoin_request=max(t_rejoin_due,t_shrink_done)`，实际 spawn/回归开始为 `t_rejoin_start`。join 墙钟从 request 算，expand time 从实际 start 算；二者不能混用同名起点。回收完整延迟按 kill 起算，其 join 内等待按 `max(0,t_reclaim_done-t_rejoin_request)` 计算。

emulate 不存在的真实 abort/rebuild/reclaim 指标记 not_applicable，不填测量值 0 后混入均值。报告可比较公共恢复指标的 real − emulate 差值，但不能声称它精确分离了通信器重建的因果成本。

## 6. 汇总与图表

`summary.csv` 包含每组每指标 mean、sample_std（ddof=1）、n_valid、n_failed、n_censored、单位及定义版本；不足两个有效样本时 std=null。附逐轮数据及排除原因。

额外输出 `real_minus_emulate`：对两组共有且定义相同的有效指标，以 real 组均值减 emulate 组均值，列出各自 n 和 std。3 对 5 不构成配对样本，不伪造配对差或显著性。每组吞吐曲线保留逐轮曲线，聚合规则和事件对齐方式明确标注，不用对齐后的时间轴计算原始完成时间。

图表标注 fail、shrink_done、rejoin、expand_done 四个事件，另可标 first token/recovery。至少输出：各副本/全局逐秒吞吐 CSV、请求时延、事件分解、每卡内存、逐链路字节/有效吞吐，以及 real-emulate 对比表。逐链路 batch 时间共享时不得加成虚假的总传输耗时。

## 7. 本地与专用 GPU 验收

本地用手算时间线验证阈值、滑动窗口、删失、未满 10 s、无降级样本、非配对差、样本标准差、回收重叠去重及 token 计数守恒；故意损坏 hashes/丢事件/重复请求的日志必须拒绝纳入正式结果。

服务器执行完整 **5+3+5+3** 矩阵并检查：每轮请求 hash 相同，两个故障事件各一次，回归不早于 due/shrink_done/reclaim，所有完成请求/token 数一致，峰值预算/稳态权重断言通过，MPS/进程生命周期完整，链路字节与迁移任务一致，summary 可由原始记录重复生成。

若失败，不补造样本或只选成功轮发布；保留失败及原因，修复后按相同冻结配置重跑并注明新 SHA。脚本正常退出但关键字段缺失也属于验收失败。

## 8. 交付与完成标准

输出 `/data/.../step07_e2e_acceptance/<matrix_id>/`，包含请求/config/profile/plan manifests、每轮 events_actual、MPS/生命周期/rank maps、链路与逐请求原始数据、summary.csv、时间线 CSV、图表和报告。manifest 保存完整 SHA256 与实际命令。

按总计划提供两节点完整 source/fetch/checkout/nohup 命令、输出路径、打包回传命令与测完切回 `f7d5559`。最终报告区分功能正确性、CPU 证据、GPU 证据、删失指标和仍未解决的问题；只有全部必要检查通过且结果可复算才标记本步完成。

## 9. 验收登记

冻结请求/profile/config hash：待填。实现 SHA 与 CPU：待填。GPU 矩阵：未运行。summary/报告：待生成。最终验收：待回传。
