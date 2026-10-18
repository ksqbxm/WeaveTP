# Step 6：阶段 B 固定槽位服务、请求 KV 与事件注入

状态：A 部分（统一服务驱动的 fake 后端、CPU 测试与 `quick_check.sh --step 6 --single-node`）已实现，实现 `0f1776a0ff0929114d68318f62e7e2c3d957f508`，见 [Step6_实现与验证.md](Step6_实现与验证.md) 与 [Step6_服务器命令.md](Step6_服务器命令.md)；B 部分（真实模型，B1→B2→B3）待阶段 A 服务器验收。专用 GPU 测试标识：`step06_batched_serving`。

2026-10-18 用户指示分两部分推进：A 部分现在开始（统一服务驱动的计划与 CPU/fake 后端实现，不依赖 Step 3–5 上机），B 部分等阶段 A 服务器验收。统一服务驱动的计划稿见 [Step6_服务驱动计划.md](Step6_服务驱动计划.md)，**待用户审阅后才编码**；用户所述扩缩容计划 W1–W4 文档在仓库中未找到，计划稿 §0 给出了与 C1–C11/K1–K7 的映射并列出待确认问题。

## 1. 必读与前置

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 每次代码修改前核对本 Step、适用 AGENTS.md；Python imports 变化执行 `uv run isort`。阶段 A 的 microbatch=1 正确性不能替代本步 batched serving 验收。

输入是已确认的 A6/A7 结果、用户指定的 ShareGPT 子集、冻结的候选 profile 和配置。真实数据子集未明确时可以开发与测试服务逻辑，但不能自行决定正式负载。

## 2. 实施任务：最小批量服务驱动

- 固定请求集一次离线提交，保留 request id、原始提交时间、FCFS 顺序、prompt token、固定输出长度和已提交历史。真实 greedy，达到配置目标长度才结束，忽略 EOS；输出长度≤256，不使用 teacher forcing。
- 全局 admission 按当前两个副本可用容量分配；EDP0 缩容停服时容量为零，新请求交给 EDP1。降级后容量预算反映 TP2，不能只按卡数均分。
- 每副本固定最大 batch slots，容量由完整 full-prefill/切换峰值 profile 验证后冻结；每个 decode step 完成后释放已完成请求 KV 槽，按 FCFS 补入。区分已释放逻辑槽、仍需保留的有效 KV 与 allocator reserved。
- 静态 KV 增加逐槽 position、有效长度、padding mask 的正确管理；异长请求不能共享单个错误的全局 offset。槽位复用不得读到上个请求的 KV。
- Full prefill，无 chunked prefill、无正常运行抢占。若输入长度超出已验收容量，明确过滤规则或报错，不静默截断。阶段 A 后重新核查 full-prefill 的峰值预算。
- 同副本 dense DP 有空分支时按协议进行空前向，保持 EP collective 顺序；两 EDP 独立推进。避免一个 rank 的 OOM/跳过判断造成集体通信分支不一致。
- EDP0 故障请求按原 prompt + 已生成历史重新 prefill，容量不足回原 FCFS 队首；保留最初提交时间，不重复计数完成/新增 token。EDP1 请求历史不重置。
- 扩容继承 Step 4 快照+增量和内存合同，不因批量槽位引入备用最大 KV 常驻或静默停服回退。

## 3. 注入器与 plan.json

采用总计划 §7 的完整 `v2-failstop` schema：fail anchor 为 `completed_count=ceil(0.25*N)`，包含 `replica=EDP0` 与四个坐标；rejoin anchor 为相对 fail 的 `after_event_seconds=45`，`wait_for=shrink_done`。根字段保存请求文件 SHA256，运行前核对一致。

注入器位于 SL3060，通过独立 TCPStore 与每节点监管器协作。先施加故障再通知，不以 watchdog 超时作检测。completed_count 是全局唯一完成请求数，重复上报去重；事件在阈值首次跨越时只触发一次，记录是否因批量完成跨过阈值。

记录 `t_fail` 命令时刻、每 PID kill/exit、通知到达；rejoin due=`t_fail+45s`。到期仍未 shrink_done 则顺延；real 还需目标 GPU 回收。新 worker 只由 SL3061 监管器在所有前置完成后启动。emulate 释放/恢复状态遵循相同事件驱动，不伪造 real 的 abort/reclaim 成本。

将“计划到期”“shrink 就绪”“回收就绪”“rejoin 请求”“spawn”“world ready”“目标布局 ready”分开。`rejoin_delay_s` 保存实际启动相对 due 的顺延，按原因记录区间及重叠；不能把所有阻塞时长简单相加。`gpu_reclaim_s` 完整保存，join 内只累计残余 `reclaim_wait_s`。

实现直接使用总计划公式：`t_rejoin_request=max(t_rejoin_due,t_shrink_done)`；实际 `t_rejoin_start` 在回收也满足后发生。`rejoin_join_s=t_world16_ready-t_rejoin_request`，`reclaim_wait_s=max(0,t_reclaim_done-t_rejoin_request)`，`rejoin_delay_s=t_rejoin_start-t_rejoin_due`。request 只是进入等待流程，不授权提前启动新 worker。

## 4. 事件与请求输出

每轮输出 `events_actual.json`，至少包含：

| 字段组 | 内容 |
|---|---|
| 身份 | schema version、mode、launch/round id、代码 SHA、配置/profile/request/plan SHA256 |
| 计划 | 原始 anchor、coords、replica、计划到期时刻、依赖条件 |
| 实际触发 | 控制器时刻、实际 completed_count、阈值 overshoot、事件唯一 id |
| 内存快照 | 每卡 KV_used/权重/峰值；采样时刻、来源、age、是否最新提交步 |
| 恢复阶段 | fail/notify/abort/rebuild/迁移/KV replay/shrink_done/first token/reclaim/rejoin/spawn/join/cutover 时间区间 |
| 状态映射 | world epoch、稳定设备→rank 映射、每进程节点/PID/生命周期 |
| 延迟 | rejoin_delay_s、各原因区间、gpu_reclaim_s、reclaim_wait_s、abort_s、rebuild_s、rejoin_join_s、EDP1 暂停 |

real 故障卡 KV_used 来自外部 CPU 控制面保存的最后提交步快照，含 snapshot age；无法观测的字段记 null+原因，不造零值。控制器时间为公共时基，本地耗时用单调时钟；跨机对齐保存偏移与误差，未经校准的远端时间不可直接相减。

逐请求记录：初始提交、首次接纳、每次入队/恢复、每个有效新增 token、首 token、完成、replica/layout/slot 迁移、prefill/replay token。逐秒表只计成功提交的新增 token；padding/empty forward/replay 不计入。链路日志沿用 Step 3，所有事件可用 id 关联。

## 5. 本地验证

用小请求清单验证 FCFS、重复完成去重、槽位释放/补入、事件边界 ceil、rejoin 到期但 shrink 未完成、回收晚于到期、回收早于到期不重复计 join、hash 不符拒绝运行。对输出 schema 做结构验证，并从逐请求日志独立重算 completed_count 和新增 token 总量。

模拟暂停与迁移期间的请求状态转移，确保故障重放不重置 TTFT、不少算 TPOT 停顿、不重复完成。小数据 CPU 测试证明调度逻辑，不证明 batched logits 正确。

## 6. 专用服务器 GPU 验收

在两模式均运行以下小规模可审计用例，并用同布局/同历史/同批次参考严格逐位比较 logits：

1. 不同 prompt/输出长度、部分槽为空、完成后当步释放并补入、反复复用槽位。
2. 同副本一个 dense DP 空、另一个非空，EP collective 次序正确；两个 EDP 步数不同仍正确。
3. 故障发生在 batch 完成阈值跨越处，EDP0 全部在途恢复且放不下时按 FCFS 回队，事件只触发一次。
4. 扩容搬运期间请求完成、槽位换请求、KV 长度增长；切换后无旧 request KV 串入新槽。
5. rejoin due 前/后 shrink 完成、real 回收延迟场景；新 PID 不早于回收，顺延与计时可从原始日志重算。
6. 请求提前排空、事件无法触发或不足统计窗口时显式标无效，不循环空前向制造恢复吞吐。

验收成功要求逐位正确、无死锁/漏请求/重复 token、所有输出能够交叉核对、内存峰值在预算内。先通过正确性再做正式速度测量。

## 7. 交付与关口

交付服务驱动、事件计划生成/校验、注入器、逐请求/事件日志、GPU correctness runner，输出 `/data/.../step06_batched_serving/<mode>/<launch_id>/`。提供完整 SHA、逐节点 source/fetch/checkout/nohup、打包回传及恢复 `f7d5559` 命令。

两模式 batched 正确性通过后，进行独立无故障 TP4 初始标定，选择约 108 s 工作量并冻结请求/profile/配置 hash，再进入 Step 7。标定运行单列，不算入正式五次参考。

## 8. 验收登记

A6/A7 用户确认：待填。ShareGPT 子集/请求 hash：待定。实现 SHA、CPU：待填。两模式 batched GPU：未运行。正式冻结配置：待填。
