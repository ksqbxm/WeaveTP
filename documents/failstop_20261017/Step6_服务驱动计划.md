# Step 6 计划（修订版）：统一服务驱动，fail-stop 端到端先跑通，扩缩容只加功能

状态：**第二次修订稿（已吸收 2026-10-18 五条审阅意见），用户已批准开始 A 部分编码。** 2026-10-18 用户发来《第 1 行扩缩容端到端实验：实施计划》修订版（副本：[参考_扩缩容实施计划_W1-W4设计依据.md](参考_扩缩容实施计划_W1-W4设计依据.md)），指示以其中 W1–W4 的设计为准修订本计划；附件只作设计依据与数字来源，其中分支起点、W0、W5–W7 一律忽略；本轮只在 `codex/failstop` 开发，扩缩容以后从本分支拉 `codex/e2e-scaling`。

写本稿前复核了 CODEX.md、[总计划](00_总计划.md) §6.2/§7/§8、[原 Step 6](06_Step6_批量服务与事件.md)、[Step 7](07_Step7_端到端矩阵与统计.md)、附件 W1–W4 全文、`documents/e2e_prep_20261008/sim/e2e_sim.py`（`Req.arrive`、`Replica`、`regroup`、`peak_ok`、`pre_shrink_cap`、`admit_step` 均在本分支）、Step 3/4/5 现有代码（`mixed_sources.Shard`/`build_mixed_plan`、`mixed_transport.execute`、`async_state.PeakBudget/ReplicaDriver/recovery_queue`、`model_shards.py`、`model_worker.py`）。

## 1. 目标与不变量

1. 把 fail-stop 端到端跑通：固定槽位批量服务 + 事件注入 + 缩容/扩容 + 批处理逐位检查。
2. 两个实验共用的服务驱动一次写完整：以后做扩缩容只往 `tools/resharding/e2e/` 上加 `LiveRuntime` 的双布局构建、固定基线、数据集/plan/脚本，不改第 3 节任何模块的结构或接口。

不变量（代码评审以此为检查点）：

- 只有一个 `Driver.step()`；两个实验的差别只在配置与注入的工厂/策略对象，没有 `if experiment == ...` 分叉。
- `Ledger` 以副本编号为键；rank↔副本映射只存于 `rank_map`；D16 公式只在 `controller.regroup` 一处；fail-stop 的映射由 `FaultSession.rank_map()` 适配而来。
- 任何依赖 16 卡、SL3060/SL3061 的常量只出现在校验器（`validators.py`），运行时按 `world_size` 与 `rank_map` 推导。
- 不另写第二套 KV、注意力、计划或结果写入。

## 2. 代码布局

```
code/current/tools/resharding/e2e/
  __init__.py
  requests_io.py      负载/计划读取与校验（W1）
  controller.py       Ledger、rank_map、D16、AdmissionController、Decision（W1）
  decode_policy.py    greedy / teacher_forcing（第 3.8 节）
  anchors.py          四种锚点（第 3.5 节）
  plan_executor.py    PlanExecutor 与切换流程（W3）
  block_kv.py         BlockPool、BlockKVContext、compact、KVPlanBuilder、BlockKVBundle（W2）
  attention_paged.py  Triton kernel、torch 参考、split-KV 回退、自检（W2）
  live_runtime.py     Driver（逐步循环）、LiveRuntime（工厂注入；双布局构建本轮只留接口）（W1/W3）
  factories.py        默认工厂与 fail-stop 工厂（第 3.6 节）
  failstop_adapter.py FaultSession.rank_map() → rank_map、事件 → 锚点、EDP 域（第 4 节）
  timeline.py         TimelineWriter、SwitchEventWriter（W4-b）
  results.py          result.json（W4）
  validators.py       16 卡/节点名/嵌套关系校验（沿用 weavetp_observations.validate_parallel_groups）
  fake_backend.py     CPU 假模型/假 KV/假传输，供全部 CPU 测试与 --step 6 快验
code/current/examples/rl/run_e2e_scaling.py   唯一入口（薄壳）；fail-stop 是同一入口的另一种配置
code/current/tests/unit_tests/e2e/            CPU 测试
```

对 `examples/rl/benchmark_live_moe_tp.py` 的唯一改动：`_run_async_waves(..., decode_fn=None)`，替换其中唯一一次 `_decode(...)` 调用；现有 `test_live_defaults.py`、`test_live_storage_lifecycle.py` 保持通过并补一个 `decode_fn` 用例。Megatron 只加 W2 原文列出的 3 处 `is_block_kv` 钩子。

## 3. 共用架构：本轮实现并测试（W1–W4 的设计与数字照用）

每条先写设计（原文），再写本轮实现程度（A = fake 后端 + CPU 测试；B = 真实模型 GPU），最后写与原文的差异。

### 3.1 逐步循环（W1）

设计照原文：每个逻辑步 rank 0 用接纳控制器算出 `Decision{admits[(seq, replica)], prefill_tokens[], completions[], phase, head_block_reason, need_prefill_forward, sleep_until, switch_trigger}`，`broadcast_object_list` 到 Gloo `control_group`；所有 rank 对同一账本应用同一决策。任一副本有接纳则先 prefill 前向（无接纳副本喂 1 个 dummy token）再 decode 前向，否则只 decode。dummy 前向三不：不写 KV、不计 token、不计入 `t_prefill/t_decode`，耗时记 `t_sync`。每步末 `_all_ready(ok)`（MIN 归约）达成共识；每 10 步一次 `torch.cuda.mem_get_info` 护栏（低于阈值暂停接纳、`head_block_reason=mem`）。开环下队列空且下一条未到时 rank 0 决策带 `sleep_until`，所有 rank 真等（`t_wait`，`head_block_reason=arrival`）。控制流只走 CPU `control_group`，不碰 `NCCLCopyService` 的流。`step_time = t_prefill + t_decode + t_sync + t_ctrl + t_wait`，`t_ctrl` 目标 < 3 ms。

**决策提前一步（审阅意见 3）**：控制节点在第 s 步末按账本末状态算第 s+1 步的 `Decision` 并预推到控制 store；域 rank 0 在第 s+1 步开始时直接取、不等待跨机往返。`Decision` 带 `for_step`，域侧校验步号连续。接纳语义不变，生效晚一步（TTFT 影响 ≤ 1 步，写入字段说明）；锚点判定同样晚一步，overshoot 记录包含这一步。单域（扩缩容）也走预推路径，与 `e2e_sim` 的对照测试把"晚一步生效"作为对照条件。`t_ctrl` 拆为 `t_ctrl_fetch`（取决策）与 `t_ctrl_apply`（账本应用 + 域内广播），fake 循环与冒烟都单独报出，目标合计 < 3 ms。

本轮：A 全部实现并测（fake 后端两进程 Gloo 广播跑完整循环，五项之和断言，dummy 三不断言，开环等待按墙钟，预推决策的步号连续性与晚一步生效）；B 接真实模型。

差异：**步进域**。原文"所有副本同步推进"是单域（16 卡一个 `control_group`）。fail-stop 要求 EDP1 在 EDP0 停服/重建期间继续推进（总计划 §6.2、Step 4 合同），所以 `Driver` 按"步进域"运行：域 = 共享集合通信、必须同步步进的 rank 集合；每个域有自己的 Gloo 控制组和域内 rank 0。扩缩容是一个域，退化为原文设计（与 `e2e_sim` 同轨迹的对照测试就在单域下做）。fail-stop 是两个域（EDP0、EDP1），各自步数独立；全局 FCFS 接纳由控制节点（SL3060，Step 1–5 的控制器进程）上的 `AdmissionController` 持有，各域每步经控制 store（`common.Bus`）取本域 `Decision` 再域内广播。这是对原文的扩展，不改变单域语义。

### 3.2 控制器（W1）

设计照原文：`Ledger`（每副本 running、reserved、used、blocks），`rank_map` 单独存，D16 公式只在一处（`regroup_2to4/4to2`），`AdmissionController.step(now_s, step, freeze_seq, switching, throttled) -> Decision`：严格 FCFS 不跳队首、`< B_max`（256）、按"输入 + 输出"预留、预留最少/编号小者、`C_eff = C − segment_tokens`（TP2 153,754 − 4096 = 149,658，TP4 372,280 − 4096 = 368,184）、每步 prefill ≤ 4096 且超长单条独占一步、分阶段 barrier、开环 `arrive_s`、切换期间冻结、缩容前限流 `min(0.95 × 2 × C_TP2, 0.95 × 预算 ÷ (2 × 69,120) − B_max × (G + 1))`、`drain`、`peak_check(direction, G)`（字节按段取整 `ceil(U'/4096) × 4096 × 每 token 字节`，TP2 138,240 B、TP4 69,120 B，预算 15.9 GB，G 默认 2→4 为 7、4→2 为 23）、`regroup_2to4/4to2`。

本轮：A 全部实现，纯 Python，CPU 测试覆盖每条策略，并与 `documents/e2e_prep_20261008/sim/e2e_sim.py` 在同一输入下逐步比较轨迹（接纳序列、完成步、阶段）相等。

差异：fail-stop 增加三条策略，作为同一控制器的配置而非分支：(1) 副本容量来源可注入，EDP0 停服期间容量 0；(2) 故障时在途请求按原序回队首（复用 `async_state.recovery_queue`），保留原提交时间与已生成历史，重新 prefill 用"原输入 + 已生成"；(3) fail-stop 的容量按布局（TP4/TP2）由 Step 4 `PeakBudget` 口径给出。`FaultSession.rank_map()` 适配器把 `(slot, replica, ep, etp, new_rank, active)` 映射成控制器的 `rank_map`（副本编号 EDP0=0、EDP1=1，副本内 rank 顺序按 `(ep, etp)`），这是"D16 只在一处"之外的另一个映射来源，但都只产出 `rank_map`，账本不感知来源。

### 3.3 块级 KV（W2，K1–K7）

设计照原文：块 64 token、段 4096 token；每布局每 rank 一个 `BlockPool`，段一次 CUDA 分配覆盖全部 27 层，`K[G, 4096, 27, hg=4, 192]`、`V[G, 4096, 27, 4, 128]`，G = 本 rank head 组数（TP2 为 2、TP4 为 1）；`seg[g, t0:t1]` 对所有层连续；段按需分配、空段释放（`release_empty_segments()` + `empty_cache()`）；备用布局池为空；每段以 `nn.Parameter` 暴露为 `kv::seg_{n:04d}.key|value`，由 `BlockKVBundle.named_parameters()` 提供。`BlockKVContext(StaticInferenceContext)`：`is_static_batching()` 为 True，`is_block_kv = True`，`key_value_memory_dict` 保持空，`max_sequence_length = 16384 + 512` 只用于 rotary 表；`prepare_step(prefill=[(seq, len)], decode=[seq])` 生成逐 token 位置、写入槽位表、块表 `[T, max_blocks]`、上下文长度、块对角 mask。Megatron 3 处钩子照原文。压实：对源副本每 rank 把存活请求的块按（目标副本, seq）排 A/B 区（A = 已满块，B = 开口块，空槽 < `max_delta_steps` 时在 B 区预留下一块），cycle-following 原地置换；时机在第 s 步 decode 结束、第 s+1 步开始之前，置换与块表更新为一个原子操作（任一 rank 失败经 `_all_ready` 共识整体中止）；记 `compact_done`，耗时计入 `admission_pause_s`。`KVPlanBuilder` 只依赖 `rank_map` 与账本，每 rank 确定性生成，基础波 = 权重 + A 区连续区间，增量 = B 区连续区间；不再使用 `restrict_plan_sequence`。

本轮：A 实现 `BlockPool`（CPU 张量：分配/释放/段回收/`physical_bytes()`）、`compact`（置换后内容与块表一致、A/B 区连续、原子性）、`KVPlanBuilder` 的映射与任务计数（见下）、`BlockKVContext.prepare_step`（位置、块表、mask；钩子的调用形状用 CPU 假注意力验证）；B 接 Megatron 钩子与真实前向。

**`KVPlanBuilder` 同时覆盖两种映射（审阅重点一）**：构建分两层。第一层 `kv_moves(src_rank_map, dst_rank_map, ledger_assignment) -> [(src_rank, dst_rank, head_group, zone, block_range)]`，只用 head 组归属与块表：head 组 k 在源布局由"持有组 k 的 rank"给出，在目标布局由"需要组 k 的 rank"给出，二者都从 `rank_map` 的 `(replica, tp_coord, tp_size)` 推出（TP2 tp-rank j 持有组 {2j, 2j+1}，TP4 tp-rank k 持有组 k），请求→目标副本的分配来自账本（扩缩容 D16：2→4 合并 2i/2i+1，4→2 按 `seq` 分给预留较少的子副本；fail-stop：EDP0 扩容时四卡都是目标、请求不重分）。第二层是两个渲染器：`as_transfer_ops()` 输出 Megatron `TransferOp(param_name='kv::seg_xxxx.key|value', peer_rank, is_send, my_slice, peer_slice, task_id)`，与缓存的权重 `ReshardPlan` 合并交给 `_run_async_waves`（扩缩容路径）；`as_shards()` 输出 Step 3 `Shard`（全局坐标 `[token, 1, head, dim]` 的 head 段与 token 区间，与 Step 5 `model_shards.kv_shards` 同一坐标约定），交给 `build_mixed_plan` + `mixed_transport.execute`（fail-stop 路径）。CPU 测试：同一账本下，D16 2→4 的 moves 满足"rank 4i+k 从 4i+k//2 与 4i+2+k//2 各收组 k，其一为本地"、4→2 满足"子副本 tp-rank j 从 TP4 rank 4i+2j、4i+2j+1 各收组 2j、2j+1 且只含分给它的请求"、任务数 ≤ 2 × 27 × 2 × 16；fail-stop EDP0 TP2→TP4 的 moves 与 Step 5 `map_kv_heads` 的 head 算术一致，渲染成 `Shard` 后经 `build_mixed_plan` 执行得到的目标与 `map_kv_heads` 逐位相等。两种渲染器的字节总量相等。

差异：(1) fail-stop 缩容不迁 KV，`KVPlanBuilder` 对该方向返回空计划；扩缩容 4→2 迁 KV。**核对依据（审阅意见 1）**：`megatron/core/failstop.py` 的 `FaultSession.failed` 固定为 EDP0 中 etp ∈ {0, 1} 的四张卡（坐标 (0,0),(1,0),(0,1),(1,1)），`replica_group_specs` 的 dense TP 组为连续 4 卡，EDP0 的两个 dense DP 副本恰是两个 EP 半区，**每个 dense 副本都失去 2 个 TP rank**；MLA KV 按 head 切分（K `[S,B,heads/TP,192]`），丢失 head 的 KV 无法由其他卡补回，重算丢失 head 又须跑完整前向。所以"另一个 dense 副本 KV 完好"在固定坐标下不成立，只有故障取走整个 dense 副本（某 EP 的 etp 0–3）时才成立，而那与总计划 §3 的坐标和"每个 EP 保留原专家归属"矛盾。(2) 两条执行路径并存（Megatron 规划器 vs Step 3 混合规划器），由工厂选择，见 3.6。(3) 测试增加：同一账本下 `as_transfer_ops()` 与 `as_shards()` 的任务数同量级，均 ≤ 2 × 27 × 2 × 16，防止 `as_shards()` 退化成按请求出任务。

### 3.4 注意力（W2，K4）

设计照原文：prefill 用显式块对角因果 bool mask `[1,1,S,S]` 走 `DotProductAttention`；decode 走 `paged_decode_attention(q[T, np, 192], pool, block_tables, lengths, scale) -> [T, np, 128]`（Triton：段基址表 + 块表直接读块，在线 softmax，fp32 累加，每层每步一次 launch），`paged_decode_reference`（纯 torch，按请求 gather，只用于测试/校验），回退实现"按段分组的 split-KV 纯 torch"（每段一次按块索引读取 + log-sum-exp 合并）。`--e2e-attention auto`（默认）启动自检：Triton 不可导入、编译失败、或随机块上与参考 max-abs 超阈值（bf16 1e-2 相对）→ 自动切 split-KV，`result.json.attention_impl.decode` 记 `triton | torch_splitkv` 与回退原因，冒烟同测两条路径单步时间写 `attention_impl.m1_step_ms`。

本轮：A 实现参考实现、split-KV 回退、自检逻辑（Triton 缺席路径）与 CPU 测试（参考 vs 朴素注意力逐位相等 fp32；split-KV vs 参考在 fp32 下逐位、bf16 下阈值内；MLA 192/128 不等维；任意块边界与变长批）；B 实现 Triton kernel 并在 sm_120 自检。

差异：无。fail-stop 的逐位检查在确定性模式下要求同一实现、同一分段固定，`attention_impl` 写入 Step 5 的 mode manifest。

### 3.5 计划执行器（W3）

设计照原文：`PlanExecutor` 按序布防 `switches[]`，首次越过只触发一次并记 overshoot；切换流程：冻结接纳 → 峰值等待 → 压实 + 释放空段 → 目标池按序分配 + `allocate_and_poison` → 合并计划 → 基础波 `_run_async_waves(decode_fn=前台步, max_wave_tasks=8192|4096)` → 权重 `finite_flag` + `audit_weight_checksums` → 同步增量 → 可关校验步（`--e2e-validate-cutover on|off`，默认 off；开时同批 token 源/目标各跑一步，逐行 `_comparison_metrics`，`_all_ready` 共识）→ 指针切换 → `release_standby` + 源池清空 → 账本重组 → 恢复接纳；切换期间完成的请求在 cutover 统一结算。

**锚点可插拔（审阅重点二）**：`anchors.py` 定义 `Anchor` 协议 `arm(ledger, step) / check(event) -> Trigger|None`，四个实现：`BeforeAdmit(seq)`、`AfterComplete(seq)`（扩缩容）、`CompletedCount(n)`（fail-stop，`n = ceil(0.25 N)`，唯一完成数去重，记 overshoot）、`AfterEventSeconds(event, seconds, wait_for)`（fail-stop，`t_fail + 45 s` 到期且 `shrink_done`）。`PlanExecutor` 只认协议，plan.json 的 `anchor.kind` 经注册表构造；新增锚点类型只加一个类与一个注册项。

本轮：A 全部实现与测试（四种锚点各自触发/不重复/overshoot，`ceil` 边界，rejoin 到期早于/晚于 shrink_done，切换流程各阶段顺序用 fake 传输验证，校验步开关）；B 接真实切换。

差异：(1) fail-stop 的缩容由故障事件强制触发，不是计划切换；"峰值等待"换成 Step 4 `PeakBudget.check`，"压实"只对扩容执行。**峰值不足的退化（审阅意见 4）**：不 abort，而是按接纳倒序把在跑请求驱逐回队首（保留原提交时间与已生成历史，恢复时重新 prefill），直到预算满足；记 `peak_eviction` 事件（驱逐请求数、token 数、释放字节、所在步）写入 `result.json.failstop.peak_eviction[]` 与 timeline 的 `stall` 行；驱逐不是静默，无事件记录的运行不得通过。总计划 §6.2 的"报错"口径随本稿同步修改。(2) 校验步在 fail-stop 下是 Step 5 的逐位合同（`determinism.compare_records`），所以 `validate_cutover` 是注入的策略对象：扩缩容 `ToleranceValidation`，fail-stop `BitwiseValidation`；**是否开启由运行模式决定，不由默认值决定（审阅意见 2，见 3.8 与 §4）**。

### 3.6 工厂注入（审阅重点三）

`factories.py` 定义 `RuntimeFactories` 数据类，`LiveRuntime` 与 `Driver` 只通过它取对象：

| 工厂 | 扩缩容默认 | fail-stop 注入 |
|---|---|---|
| `process_groups(layout) -> ProcessGroupCollection` | `build_inference_pg_collection` | `build_replica_pg_collections(session.replicas())` |
| `copy_service(group) -> CopyService` | `NCCLCopyService(group=..., **kwargs)` | `make_copy_service(session, slot, group, group_relative_ranks=True)` |
| `build_model(layout, groups) -> model` | `model_provider(pg_collection=...)` | `MegatronRuntime.build`（Step 5） |
| `rebind(model, groups)` | 不需要（备用布局重建） | `ModelBinding.rebind` |
| `planner(weights, kv_moves, rank_maps) -> plan` | 权重 `build_centralized_reshard_plan`（缓存两方向）+ `KVPlanBuilder.as_transfer_ops()` | `model_shards` + `KVPlanBuilder.as_shards()` → `build_mixed_plan` |
| `transport(plan, decode_fn)` | `_run_async_waves` | `mixed_transport.execute`（同步波；重叠接 Step 4 为后续项） |
| `validate_cutover` | `ToleranceValidation`（默认关） | `BitwiseValidation`（默认开） |
| `capacity(replica)` | plan.json `scheduler.capacity_effective_tokens_per_replica` | 布局容量，停服为 0 |

运行时逻辑按 `world_size` 与 `rank_map` 推导；`validators.py` 持有 16 卡、SL3060/SL3061、EDP 嵌套关系校验。本轮 A：工厂接口与 fake 实现；B：fail-stop 列全部接通；扩缩容默认列本轮只写签名与单元级实现（`build_inference_pg_collection`/`NCCLCopyService`/`model_provider` 的调用方式按 benchmark 1441–1569 写好，但 16 卡双布局构建不运行）。

### 3.7 结果（W4、W4-b）

设计照原文：`timeline.jsonl` 每逻辑步一行（`kind(step|stall), t, step, layout, phase, gen_tokens, prefill_tokens, admitted, completed, queue_len, head_block_reason, running[], kv_used_tokens[], kv_bytes_per_card[], step_time, t_prefill, t_decode, t_sync, t_ctrl, t_wait, mem_alloc[]|null, mem_peak[]|null`），rank 0 内存追加、每 200 步落盘，`mem_*` 每 10 步 `all_gather_object`；切换外露等待写 `kind: "stall"`，各阶段时长之和 = 总时长。`SwitchEventWriter` 事件 `switch_trigger | peak_wait_start | peak_wait_end | compact_done | base_wave_end | base_done | delta_done | target_ready | cutover | source_release_done`，`base_wave_end` 带 `wave_records` 字段，字节分类 `weight | kv_prefix | kv_delta`。`result.json` 按原文 schema（`schema_version, method, workload_id, segments_run, totals{...}, switches[...], requests[...], phase_durations_s, attention_impl, config`）。

本轮：A 全部实现，`--check` 自检（Σ gen_tokens = 生成总数；阶段时长和 = 总时长 ±0.1%），CPU 测试从逐请求日志独立重算 completed_count 与新增 token。

差异：fail-stop 字段单独加入 `result.json.failstop{...}`：故障事件（`t_fail`、每 PID kill/exit、通知到达）、恢复阶段区间（abort/rebuild/迁移/replay/shrink_done/first token/reclaim/rejoin/spawn/join/cutover）、重新 prefill 的请求数与 token 数、shrink TFS/TTP、expand time/TTP、EDP0 停机时间、EDP1 暂停、`rejoin_delay_s`、`gpu_reclaim_s`、`reclaim_wait_s`，口径按总计划 §7/§8 与 Step 7；`events_actual.json` 由 `SwitchEventWriter` 的事件与 fail-stop 控制器事件合并产生。

### 3.8 解码与入口

`decode_policy.py`：`DecodePolicy` 协议 `next_tokens(logits, ledger_rows) -> tokens` 与 `is_done(row)`；`Greedy`（argmax，达到固定 `output_len` 结束，忽略 EOS）与 `TeacherForcing`（下一 token 取参考 `output_token_ids`，结束同上）。两者都有 CPU 测试，由 `--e2e-decode-policy greedy|teacher_forcing` 选择，`Driver` 只调用协议。

运行模式（审阅意见 2）：必填参数 `--e2e-run-mode correctness|timing`，无默认值。`correctness`：`--deterministic-mode`、校验步开（fail-stop 逐位 / 扩缩容容差）、Step 5 manifest 记录，所有时间字段 `timing_valid=false`；`timing`：默认模式、校验步关、不物化全部 logits，TFS/TTP/吞吐只从此模式取。`--e2e-validate-cutover` 若显式给出且与运行模式矛盾则报错；`result.json.config.run_mode` 记录。

入口 `examples/rl/run_e2e_scaling.py`，`add_e2e_args` 按原文定义全部 `--e2e-*`：`--e2e-requests/--e2e-workload/--e2e-plan/--e2e-method/--e2e-out-dir/--e2e-segments k..m/--e2e-timeline on|off/--e2e-kv-block-tokens 64/--e2e-kv-segment-tokens 4096/--e2e-attention auto|triton|torch/--e2e-validate-cutover on|off/--e2e-backend block|fake/--e2e-audit-direct-tp4-load`，另加 `--e2e-run-mode correctness|timing`（必填）、`--e2e-decode-policy greedy|teacher_forcing`、`--e2e-experiment scaling|failstop`（后者只是一组默认值：工厂表的 fail-stop 列、`BitwiseValidation` 作为校验策略、greedy、锚点注册表含 fail-stop 两种、步进域两个）。`main()` 照 `live.main()` 的 `args_defaults` 初始化 Megatron。fail-stop 的 Step 1–5 控制器/监管器仍负责进程生命周期与故障注入，worker 进程内运行的是本驱动。

## 4. fail-stop 特有部分（本轮实现）

- EDP0 停服期间容量为 0；缩容后在途请求按"原输入 + 已生成"重新 prefill，保留原提交时间（`recovery_queue`，容量不足回队首）；扩容过程不停服（TP2 继续提交，目标就绪后补增量、校验、原子切换，Step 4 合同）。
- 事件锚点 `completed_count = ceil(0.25 N)`（fail）与 `after_event_seconds = 45, wait_for = shrink_done`（rejoin），`plan.json` 用总计划 §7 的 `v2-failstop` schema。
- 解码 greedy，固定输出长度 ≤ 256，忽略 EOS；请求集为用户指定的 ShareGPT 子集（A 部分用 fixture，正式子集待用户指定）。
- 峰值检查用 Step 4 `PeakBudget`（源/目标权重、源/目标 KV、接收缓冲、临时、余量）。
- 两个步进域、全局接纳在控制节点（3.1）；`FaultSession.rank_map()` 适配器（3.2）；KV 缩容不迁、扩容前缀 + 增量经 Step 3 混合规划器（3.3、3.6）；逐位校验用 Step 5 合同（3.5）。
- 事件锚点选择留余量：`completed_count = ceil(0.25N)` 在正式负载确定时核对切换峰值预算有 ≥ 10% 余量，不满足则在 plan.json 生成时报告（审阅意见 4）。
- **两种运行模式（审阅意见 2）**：correctness 运行用于 A6 逐位验收，确定性模式，每次切换多跑一个校验步并物化全部 logits，时间字段 `timing_valid=false`；timing 运行用于 Step 7 的 TFS/TTP/吞吐与停机时间，默认模式，校验步关，由 `--e2e-run-mode` 显式选择。
- **对重启基线的 TFS 优势来源（审阅意见 1）**：缩容时 EDP0 在途请求全部重新 prefill（KV 不可保留，见 3.3 核对），因此优势不来自 KV 保留，而来自 (a) 权重经 P2P/主机库取得而非从磁盘重载整模型；(b) emulate 不重建进程与 CUDA 上下文、real 只重建 4 个进程；(c) EDP1 持续服务；(d) 重新 prefill 只针对 EDP0 在途请求。`result.json.failstop` 增加 `replay_prefill_tokens` 与 `tfs_breakdown`（权重取得、进程重建、重新 prefill 三段区间），使结果能说明优势来源。

## 5. 扩缩容 GPU 部分：只留接口，不运行

不做：16 卡 TP2×DP8 `LiveRuntime` 双布局构建（`StandbyWeights` 两套、初始 `swap_model_weights`、两方向权重计划缓存）、固定 TP2/TP4 基线、正式运行、W5 数据集、W6 plan.json、W7 脚本。以后加这些时只需：在 `live_runtime.py` 填 `LiveRuntime.build_dual_layout()`（工厂默认列已定义），在 `factories.py` 的扩缩容默认列换成真实实现，新增数据/脚本文件；不改第 3 节任何模块的接口。

B 部分之后的第一个增强项（审阅意见 1）：**完好 dense 副本的 KV 保留**。前提是故障模式改为取走整个 dense 副本（某 EP 的 etp 0–3）或 KV 改为非 head 切分布局，否则无 KV 可保留；实现路径已预留：与扩容同一条 `KVPlanBuilder.as_shards()`，源为残余 TP4 副本。该项需用户先修改总计划 §3 的故障坐标才有意义，本轮不做。

## 6. 交付顺序与验证

1. **本稿**：待审阅。
2. **A 部分**（fake 后端 + 全部 CPU 测试 + 快验）：
   - `tests/unit_tests/e2e/`：控制器全部策略与 `e2e_sim` 同轨迹；`requests_io`；`BlockPool`/压实/`KVPlanBuilder`（两种映射、两种渲染、任务计数、字节一致）；`prepare_step`；注意力参考/回退/自检；锚点与执行器流程；`decode_policy` 两种；timeline/results 与 `--check`；fake 后端两域/单域跑完整循环（含 Gloo 广播，2 进程）与 fail-stop 事件（fail → replay → 回队 → expand）。
   - 快验 `quick_check.sh --step 6 --single-node`：单节点 fake 后端完整循环 + 上述 CPU 测试，5 分钟内，结论行 `QUICK CHECK step=6 result=PASS (fake backend: controller/kv/anchors/decode/timeline, failstop cycle) backend=fake single_node=true`，`accepted=false`。
   - 顺带修复并一起提交：(a) Step 4 `async_worker.py` 的 `require(overlap > 0, ...)` 在 `correctness_only` 下只记录 `overlap_ms`，不作硬断言（共享 GPU 时误判过一次）；(b) `quick_check.sh` 与各 runner 把 `$(dirname "$PYTHON")` 放到 PATH 最前，并在启动前 `"$PYTHON" -c "import megatron.core.datasets.helpers_cpp"` 预检，失败即在任何 GPU 工作之前报错（SL3060 继承 base 环境 Python 3.13、缺 pybind11 导致现场编译失败）。
3. **B 部分**（阶段 A 服务器验收通过后，分三步，审阅意见 5）：
   - B1：用 fail-stop 稳态段（故障前）做一次**单域、固定 TP4、无切换**的真实运行（EDP0 与 EDP1 作为同一域的两个副本锁步推进），验收：各 rank 账本一致、五项步时之和等于 `step_time`、两次吞吐差 < 2%。扩缩容接入时第一次 GPU 运行因此不再是 16 卡双布局。
   - B2：真实模型 + 块级 KV + 分页注意力（Megatron 3 处钩子、Triton 自检），接 Step 3/4 迁移（`as_shards()` 路径，Step 4 重叠随后）与 fail-stop 事件。
   - B3：确定性模式 batched 逐位检查（原 Step 6 §6 六个用例），跑出 fail-stop 端到端第一个结果（correctness 与 timing 各一次）。

## 7. 与 W1–W4 原文的差异汇总

| 处 | 原文 | 本计划 | 原因 |
|---|---|---|---|
| 步进 | 所有副本同步推进，一个 `control_group` | 按步进域；扩缩容单域（等价原文），fail-stop 两域 + 控制节点全局接纳 | EDP1 必须在 EDP0 停服期间推进 |
| 控制器 | 扩缩容策略集 | 同一控制器加容量注入、停服容量 0、恢复队列 | fail-stop 语义 |
| KV 计划 | `TransferOp` 合并进 `_run_async_waves` | 两层：共享 `kv_moves` + 两个渲染器（`TransferOp` / Step 3 `Shard`） | fail-stop 的源覆盖（残余 TP4 + EDP1 + 主机）只有混合规划器能表达 |
| KV 缩容 | 4→2 迁 KV | fail-stop 缩容不迁，重新 prefill | 故障作废在途 KV（Step 5 §3） |
| 切换触发 | 锚点计划 | fail-stop 缩容由故障事件强制 | 总计划 §6.2 |
| 峰值不足 | 暂停接纳只解码（drain） | fail-stop：按接纳倒序驱逐回队首直到预算满足，记录事件 | 审阅意见 4；总计划 §6.2 同步修改 |
| 校验步 | 容差，默认关 | 策略对象（扩缩容容差 / fail-stop 逐位），开关由 `--e2e-run-mode` 决定 | Step 5 §2.1；审阅意见 2 |
| 决策 | 当步算当步广播 | 第 s 步末预推第 s+1 步决策，晚一步生效 | 审阅意见 3 |
| 结果 | 原 schema | 原 schema + `failstop{}` | 总计划 §7/§8 |
| 入口参数 | `--e2e-*` | 同，另加 `--e2e-decode-policy`、`--e2e-experiment` | 两实验共用 |
| 分支/W0/W5–W7 | 新分支、数据集、plan、脚本 | 忽略，本分支开发 | 用户指示 |

## 8. 审阅重点对照

1. **`KVPlanBuilder` 两种映射**：3.3 的两层设计；CPU 测试同时断言 D16 2→4/4→2 与 fail-stop EDP0 TP2→TP4，两种渲染字节相等。
2. **锚点与 `decode_policy` 可插拔**：3.5 的 `Anchor` 协议 + 注册表（四实现），3.8 的 `DecodePolicy` 协议（两实现）；`PlanExecutor`、`Driver` 只依赖协议。
3. **工厂注入**：3.6 的 `RuntimeFactories` 八项，默认列与 fail-stop 列并列；`LiveRuntime`/`Driver` 不直接引用任何具体实现。

## 9. 审阅记录与待确认

2026-10-18 审阅意见与处理：(1) 缩容 KV：核对固定故障坐标后不可保留，文档写明优势来源，KV 保留列为 B 后第一个增强项（需改故障坐标）；(2) 校验步由 `--e2e-run-mode` 决定，correctness/timing 两种运行；(3) 决策提前一步预推，`t_ctrl` 拆两段；(4) 峰值不足改为记录型驱逐，总计划 §6.2 同步，注入时刻留 10% 余量，两种渲染任务数测试；(5) B 部分先做单域固定布局 GPU 运行。用户已批准开始 A 部分编码。

仍待确认：快验 `--step 6` 只跑 fake 后端（CPU），本稿为单节点形式。
