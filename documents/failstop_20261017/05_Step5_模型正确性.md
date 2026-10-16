# Step 5：阶段 A 全模型正确性与多轮 smoke

状态：待前置验收。专用 GPU 测试标识：`step05_model_correctness`。

## 1. 必读与前置

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 每次修改前核对本 Step 与适用 AGENTS.md；Python imports 变化执行 `uv run isort`。

前置为 Step 2–4 对应模式验收、fix2 GPU 通过及 real 的 Step 1 用户放行。阶段 A 交付后停在 A6/A7 用户 GPU 结果确认关口；不提前进入阶段 B 性能实验。

## 2. 范围与参考合同

使用 DeepSeek-V2-Lite、现有静态 KV 后端，EDP0=SL3061、EDP1=SL3060；microbatch=1。沿用已有 native router 参数，不人为固定专家路由；若必须改 router 实现或模型行为，先报告并确认。

正确性要求两模式都在 **同目标布局、同请求历史、同输入/batch/order** 下 logits 严格逐位相同。TP2 和 TP4 的数值不要求彼此相同。参考权重从独立 checkpoint 路径加载，KV 由切换前快照独立映射；不调用被测 planner 生成期望。参考单独运行，避免多套模型影响被测峰值与时序。

逐位比较保存 dtype、shape、第一处差异位置与值/位模式、参数和历史 hash。只比较 argmax 或误差阈值不算通过；若算子本身不可确定，作为失败原因报告，不自动改成容差。

## 3. 实施任务

- 接好完整 fail→shrink→rejoin→expand 到现有正确性工具，两个模式使用同一断言和参考生成入口。
- 每事件校验目标权重和 logits，native 路由覆盖专家分片；报告实际专家覆盖，不能因未激活专家没参与 logits 就省略其权重逐位审计。
- KV 参考保存请求已提交历史和有效长度；故障后 EDP0 全部在途重新 prefill，EDP1 保留，扩容使用前缀+增量。
- 提供 `run_failstop_smoke.sh` 及独立 A6 正确性 runner。smoke 包括权重迁移、不跑服务的 A7，以及完整 microbatch=1 机制检查；两者分别标明范围，不把权重-only 冒充端到端服务。
- emulate 支持一次启动 `--rounds`，默认 5；每轮开始恢复正常 TP4、清除上一轮请求/事件状态并校验所有权重；保留主机库及可合法复用的预建组，但重置 epoch/round 标识和计划缓存。
- real 每轮独立 launch，沿用两监管器、SL3060 TCPStore、MPS 隔离、GPU 回收门槛和新 PID 记录，不让旧 launcher 状态跨轮污染。

## 4. A6 正确性矩阵

| 检查点 | 参考与断言 |
|---|---|
| 故障前两副本 | TP4 同历史参考；权重/原生路由输入一致 |
| 缩容后 EDP0 | TP2 同历史参考；权重完整；全部在途已重新 prefill |
| 缩容期间/之后 EDP1 | TP4 同历史参考；KV 未被错误清除；real 暂停单独记录 |
| 混合选源 | 残余 TP4、TP2、EDP1 TP4、主机覆盖正确；两来源写同目标不同片段 |
| 扩容尚未切换 | EDP0 TP2 生成 token 对 TP2 同历史参考；目标写入不影响服务源 |
| 扩容切换后 | EDP0 TP4 同历史参考；KV 增量无缺失/重复；EDP1 仍正确 |
| 轮次恢复 | 每轮 TP4 权重逐位正确；稳态单套；无上一轮请求/KV/事件泄漏 |

固定快照+profile 下另比较 emulate 与 real 的规范化迁移计划逐项一致；比较稳定身份、分片、字节、来源和依赖，而非只比较最终 logits。

## 5. 本地验证

测试参考构造独立于 DUT planner、历史 hash/长度错误能够被抓住、逐位比较发现 1 ULP 和切片错位、模式与轮号不串结果。检查 smoke 参数及默认值、每轮 reset、统计不把 replay 算新增 token。只跑受影响的正确性/standby/planner 相关回归，不把 CPU 通过写成 A6 通过。

## 6. 专用服务器 GPU 验收

- emulate 一次 launch 连续 5 轮，逐轮执行完整 A6 检查和正常→故障缩容→回归扩容，轮前权重验证不可省略。
- real 用独立 launch 完成上述相同检查，建议 3 次以覆盖新 PID/epoch 重建；不替代 Step 1 的 20 次可行性压力检查。
- A7 权重-only smoke 两模式各至少一次完整周期，输出每事件 switch_wall_s、EDP0 downtime、各来源本地/机内/跨机/host 实际字节和逐链路有效吞吐。
- 收集 Step 4 重点证据：EDP1 step 分布、两副本 EP 序列、TP2 扩容重叠、稳态/峰值内存、real 暂停与回收。

全部 A6 逐位断言和 A7 迁移审计通过，模式/轮次无缺漏，方可认为阶段 A 已具备交付条件。本步不发布服务吞吐、加速比或正式矩阵结果。

## 7. 交付与关口

输出 `/data/.../step05_model_correctness/<mode>/<launch_id>/`，提供 A6/A7 runner、固定配置、checkpoint/profile hashes、逐位报告、混合计划比较、每轮状态和生命周期日志。按总计划给逐节点完整命令、完整 SHA、回传包及切回 `f7d5559`。

**完成后交给用户跑/审阅 A6 与 A7，等待确认后进入 Step 6。** ShareGPT 子集和请求长度文件由用户在阶段 A 后指定；不得自行换数据集形成性能结论。

## 8. 验收登记

实现 SHA：待填。CPU：未运行。emulate 五轮：未运行。real：未运行。A6/A7 回传与用户确认：待填。
