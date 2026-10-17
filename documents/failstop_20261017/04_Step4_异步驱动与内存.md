# Step 4：副本独立推进、迁移重叠与权重/KV 生命周期

状态：2026-10-17 按用户指示开始实现，已实现并完成 CPU 验证；fix2 GPU 前置仍未登记，服务器 GPU 待运行。专用 GPU 测试标识：`step04_async_driver_and_memory`。

## 1. 必读与硬前置

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 每次修改前核对本 Step 及适用 AGENTS.md；Python imports 变化执行 `uv run isort`。

**用户运行 `run_default_fix2.sh` 并回传、确认 GPU 复验通过后才开始本步。** 不能把 standby 释放或旧 live driver 当作本任务已经验收。还须具备对应模式 Step 2 拓扑和 Step 3 传输结果；real 受 Step 1 关口约束。发现必须改变不间断扩容语义或内存合同，先报告并确认。

## 2. 问题与范围

现有 `benchmark_live_moe_tp.py` 的 `_run_async_waves` 存在跨 rank 同步推进假设。新增“两个副本独立推进、迁移与前向重叠”是未经 GPU 证明的新组件，是阶段 A 的重点验收项。

本步实现 microbatch=1 的机制驱动，不做阶段 B 批量吞吐结论。EDP0=SL3061、EDP1=SL3060；emulate 期间 EDP1 独立服务，real 允许通信器重建时短暂停顿并完整计时。

## 3. 实施任务

### 3.1 执行与通信顺序

- 把每副本服务推进与跨副本迁移协调分开，不在每个 decode step 用全 world barrier 绑定两个副本。
- 同一 EDP 内仍有一致 step/phase 序列；两个 dense DP 分支即使一个为空，也执行匹配的 EP collective。记录 replica step id、EP collective sequence id、epoch 和提交/完成时间。
- 服务组与迁移组具有明确使用范围和创建顺序，GPU stream/event 表达读写依赖；控制通知不能依赖正在挂起的 GPU work。
- fail 后 EDP0 立即停服，确保无错误 token 被提交。emulate 的 EDP1 可继续；real 在 abort/world rebuild 安全点暂停再恢复，记录暂停起止和原因。
- 通信器、布局或权重释放前等待对应已提交 work 的合法终结；禁止使用全局同步掩盖缺失依赖，也禁止过早释放仍被传输使用的 storage。

### 3.2 缩容权重与 KV

- survivor 的旧 TP4 分片保留为源，按 Step 3 计划创建并填充 TP2 目标；完成后重绑定目标组并切换。
- 当前静态 KV 为展开 K/V、按头切分；EDP0 全部在途 KV 不可直接延续。使用原 prompt + 已提交生成历史重新 full prefill，不能把恢复重放算新增 token。
- EDP1 请求历史/KV 不重建。EDP0 放不下的请求依原 FCFS 顺序回队首；阶段 A 小负载也要覆盖恢复容量不足的逻辑。
- 缩容完成后释放旧 TP4 storage 与缓存引用，保存每卡 storage 字节和 allocator/device 显存。理论权重字节按实际 dtype/shape 求和，不能只靠 nvidia-smi 回落判断释放成功。

### 3.3 扩容不停服

- 回归目标为空 TP4，EDP0 TP2 继续服务；权重和 KV 稳定前缀异步迁移，目标 ready 后在一致边界补增量、原子切换到 TP4。
- 快照记录 request id、历史版本、层/头分片和有效长度；切换时验证所有目标 KV 增量与当前已提交历史一致，防止重复/漏 token。
- 前向读取源布局，迁移写目标布局；并发访问共享缓冲需明确依赖。切换后释放旧 TP2，稳态只留 TP4。
- 正式迁移前核对峰值：源/目标权重、源/目标有效 KV、接收缓冲、临时算子和预留余量。预算不足时报错，不静默把扩容改为停服、不引入本轮未授权的新限流算法。

### 3.4 单套权重和观测

缩容及扩容窗口允许短暂源/目标权重共存，稳态每卡权重字节等于单布局理论值。调用经过 fix2 验证的 standby 释放路径，并确认陈旧 planner/transform/反量化引用已失效。备用 KV 不按最大批次长度长期驻留。

输出每阶段权重 storage、torch allocated/reserved、设备显存、KV_used 与峰值；对照固定 admission 预算。观察到缓存分配器保留内存时区分 reserved 和仍有引用的 storage，不能只用 empty_cache 宣称释放。

## 4. 本地验证

验证副本独立状态机、同副本空分支 collective 序列、快照/增量版本、故障重放不重复提交 token、释放顺序及失败时不进入下一布局。选最窄相关既有 standby/correctness 回归；fake stream 测试不替代并发 GPU 验收。

## 5. 专用服务器 GPU 验收

在 16 卡上建立同输入、同前向形状的无故障对照与故障运行，两个模式分别输出：

1. **EDP1 干扰分布：** 逐步耗时原始样本、count/均值/中位数/P95/P99/最大值，并给故障前、EDP0 缩容停服期、扩容重叠期相对无故障的绝对和百分比变化；同步记录 context 长度和迁移字节以解释样本差异。不预设无依据的性能阈值，功能死锁/停顿与性能变化分别报告。
2. **EDP0 恢复后两副本步调：** 两 EDP 允许步数不同；每 EDP 内 EP collective 序号完全一致，无空分支错序；token/request 历史与参考一致。
3. **TP2 与扩容迁移重叠：** CUDA 时间线或事件区间证明服务与迁移实际重叠，不能只展示两个线程存在；重叠期间持续提交正确 token，切换前增量一致，无死锁。
4. **内存生命周期：** 正常、降级、恢复稳态单套权重断言通过；峰值在预留预算内，未保留备用最大 KV；释放对象无悬挂引用导致内存增长。
5. **real 特有：** 旧组 abort/rebuild 的 EDP1 暂停完整计入，存活权重指针仍保留，重建后无旧组调用；新 worker 未连接 MPS且在回收完成后启动。

失败标准：任何错序、死锁、错误 token、预算越界、稳态双套权重或静默暂停回退。性能分布异常需解释并由用户判断，不把“无死锁”写成“性能无影响”。

## 6. 交付与后续

交付驱动、KV 恢复/迁移、布局切换与释放审计、对照 GPU runner、阶段时间线，输出 `/data/.../step04_async_driver_and_memory/<mode>/<launch_id>/`。报告首先列 fix2 的 SHA、原始 GPU 证据与用户确认，再列本步结果。

遵循总计划交付合同补齐完整服务器命令、打包和恢复命令。对应模式通过后进入 Step 5 全模型阶段 A；本步不能替代 DeepSeek native router 正确性。

## 7. 验收登记

fix2 GPU SHA/证据/用户确认：待填；用户 2026-10-17 明确指示在此前置未登记时开始本步，runner 固定输出 `prerequisites_verified=false`。实现 SHA：`28ab6dd320f64d9b595e691252fc8ab56ccbbc70`（控制节点汇总指标的修正；合并提交 `0503710a3abe86e458b2b9ced240d72f38ab793a`；审阅修正版 `b621125876e8ea8ab07c1f3ba210ddd370a0ae60`，首版 `7b33df6591815bb5db8d98d97a6898bc9ab1c0af`，修正 1 `05d942e682e3d96b85c7c7db164ba09a8dd13841`）；快验清单：`02e7bce29bf36ef9707853ef863ae20d85accc79`（分支 `codex/failstop`；合并后首版 `7c9aaf20fbda914eaa2969ad67fecb70ecb2f471`；合并前为 `codex/failstop-step4` 上的 `e09d7bce466fb3f7bbd4afc309860898873c3c2a`）。CPU：合并后 117 项 failstop 单测（含本步 18 项）、16 项 packing 回归及静态检查通过。emulate/real GPU：`7c9aaf20fbda914eaa2969ad67fecb70ecb2f471` 快验两节点全部 case passed，SL3060 edp1_rejoin_pause_s=6.09（共享 GPU，仅参考），因 SL3061 汇总 KeyError 未得 PASS；修正版待重跑。300 秒快验预算：首次服务器快验因 SL3060 /data 剩余 4.88 GB 低于 5 GB 门槛在预检终止，无 GPU 结果。传输为显式切片夹具，Step 3 集成另行验收。审阅修正了 real 替代进程对不存在进程组的 abort、EDP1 暂停区间终点等问题。详见 [实现与验证](Step4_实现与验证.md)、[审阅与修正](Step4_审阅与修正.md) 与 [服务器命令](Step4_服务器命令.md)。
