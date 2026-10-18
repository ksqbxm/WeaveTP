# Step 6 实现与验证（A 部分 fake 后端 + B 部分真实模型接入）

实现提交：`0f1776a0ff0929114d68318f62e7e2c3d957f508`（分支 `codex/failstop`；首个实现提交 `9a8a28da41c2a6de0903cc06b2934dd9dd55e5fe`，随后补两进程 Gloo 控制组测试）。快验清单提交 `df7ad5d33364fbd481c82c8164aad093ce984d54`（`releases.json["step6"]` 指向实现提交），服务器命令见 [Step6_服务器命令.md](Step6_服务器命令.md)。计划稿见 [Step6_服务驱动计划.md](Step6_服务驱动计划.md)（已吸收 2026-10-18 五条审阅意见），W1–W4 设计依据见 [参考](参考_扩缩容实施计划_W1-W4设计依据.md)。

本轮写代码前已阅读 CODEX.md、[总计划](00_总计划.md)、[06_Step6](06_Step6_批量服务与事件.md)、计划稿、W1–W4 参考、Step 3/4/5 的实现与服务器命令文档及 code/current/AGENTS.md；Python imports 变动后执行 `python -m isort`（本机无 `uv`）。

状态：共用架构八项（逐步循环、控制器、块级 KV、注意力参考/回退、计划执行器、工厂注入、结果、解码与入口）与 fail-stop 特有部分在 fake 后端上实现并通过 CPU 测试；`quick_check.sh --step 6 --single-node` 入口实现（CPU，不用 GPU）；计划 §6 的两处顺带修复一并提交。**真实模型（`--e2e-backend block`）、Megatron 三处 `is_block_kv` 钩子、Triton kernel、Step 3/4 迁移接入与 B1–B3 均未实现**，`--e2e-backend block` 直接退出并说明；`accepted` 固定 false。服务器尚未运行任何 Step 6 程序。

## 实现

| 文件 | 交付 |
|---|---|
| `tools/resharding/e2e/controller.py` | W1 数字常量（块 64、段 4096、B_MAX 256、prefill 上限 4096、每 token KV 字节、容量、切换预算 15.9 GB、增长步数 2→4 7 / 4→2 23）；`Request`/`Running`/`ReplicaState`/`RankMap`/`Ledger`；`regroup`（D16 的唯一实现）与 `relabel`（fail-stop 同副本换 TP 与 rank 集）；`used_bytes`（按段取整）、`peak_check`、`pre_shrink_cap`；`Decision(for_step, admits, …)`；`AdmissionController.decide` 严格 FCFS（frozen/throttled/phase/freeze_seq）、`complete`、`requeue`（回队首保留提交时间与已生成 token）、`evict_for_peak`、`idle`。CPU 测试与 `documents/e2e_prep_20261008/sim/e2e_sim.py` 同轨迹。 |
| `requests_io.py`、`toy_inputs.py` | 请求/负载/计划读取与校验（请求文件 SHA256 核对、v2 与 v2-failstop）、实例展开、`failstop_fail_count`；`write_toy_inputs` 生成确定性小负载，供测试与快验共用。 |
| `decode_policy.py`、`anchors.py` | `Greedy`（argmax、固定 output_len、忽略 EOS）与 `TeacherForcing`（按 `generated_so_far` 取参考 token）；`Anchor` 协议与四实现（before_admit、after_complete、completed_count、after_event_seconds，含 `wait_for`）、注册表与 `build(spec)`；执行器与驱动只依赖协议。 |
| `block_kv.py` | `BlockPool`（段分配、空段释放、`block_views`、`runs`、`named_parameters`）、`BlockTable`、`BlockKVContext.prepare_step`、`zone_blocks`/`compact`（A/B 区压实，循环跟随 + 一个备用块，表原子切换）、`KVPlanBuilder`（两种映射：D16 与 fail-stop relabel；`kv_moves` 按 head 组出任务；两种渲染 `as_transfer_ops`/`as_shards`，字节一致、任务数不随请求数增长）。 |
| `attention_paged.py` | 块级分页解码的 torch 参考、split-KV 回退与 `select_implementation`（auto：Triton 可用且自检通过才用，否则回退并记录原因；本轮无 Triton kernel，B 部分提供）。 |
| `plan_executor.py` | `PlanExecutor`（锚点轮询、`freeze_seq`、当前切换）、`perform_switch`（冻结 → 峰值检查：drain 或记录型驱逐 → 压实 → 基波（前台逻辑步）→ 审计 → 增量 → 校验 → cutover → 释放源 → regroup → 解冻，里程碑与事件落盘）、`ToleranceValidation`/`BitwiseValidation`、`validation_for(run_mode, experiment, explicit)`（timing 关校验、显式参数矛盾即报错）。 |
| `factories.py`、`failstop_adapter.py` | `RuntimeFactories` 默认列（扩缩容）与 fail-stop 列并列；`rank_map_from_rows`（FaultSession 行 → RankMap，失活卡剔除）、`domains`、`failstop_switches`（fail→shrink、rejoin→expand）、`capacity`（停服副本容量 0）、`FailstopHooks`（缩容前把 EDP0 在途请求回队首并记重放 token，目标 rank map 降级/恢复，事件时刻）。 |
| `timeline.py`、`results.py`、`validators.py` | 逐步行强制六项耗时之和等于 `step_time`（含 `t_ctrl_fetch`/`t_ctrl_apply`）、stall 行、切换事件（含 `peak_eviction`、`validation_done`）；`result.json` 组装、`check`（逐请求重算 token/完成数、阶段和与 timeline 对 duration）；16 卡/嵌套校验。 |
| `live_runtime.py` | `ControlNode`（决策提前一步 push/fetch，域级 frozen/phase/throttle）、`LocalComm`/`GlooComm`、`DomainRun`、`Driver.step()`（待处理触发先执行切换，再跑一步）、`body()`（取决策 → 广播 → 等待 → prefill/decode/空前向 → 完成 → timeline → 锚点轮询 → `all_ready` 共识）、`switch()`、`FakeClock`/`WallClock`、`run_until_idle`。 |
| `fake_backend.py` | 以真实 `BlockPool` 为 KV 的确定性假模型（logits 为历史的函数，KV 行编码层/位置/head/token，错搬或空前向写 KV 都会改变校验和）；`FakeSwitcher` 在域内副本上执行两种切换：扩缩容 D16 重组搬 KV，fail-stop 缩容只重放、扩容搬 KV，搬运量与 `KVPlanBuilder` 计划一致。 |
| `examples/rl/run_e2e_scaling.py` | 唯一入口，`add_e2e_args` 含计划 §3.8 全部 `--e2e-*`（`--e2e-run-mode` 必填）；fake 后端：扩缩容单域，fail-stop EDP0/EDP1 两域共用一个控制节点；写 timeline/事件/`result.json`/`check.json`；`block` 后端为 B 部分。 |
| `examples/rl/benchmark_live_moe_tp.py` | 唯一改动 `_run_async_waves(..., decode_fn=None)`，替换其中唯一一次 `_decode(...)`；`test_live_storage_lifecycle.py` 新增 `decode_fn` 用例，`test_live_defaults.py` 不变。 |
| `failstop/run_serving.py`、`quick_check.sh`、`quick_driver.py`、`common.py` | Step 6 快验 runner：fake 后端扩缩容与 fail-stop 两次完整运行 + `tests/unit_tests/e2e` 测试，写 `summary.json`；`--step 6` 映射 `step06_batched_serving`，仅 `--single-node`，预算 300 s；`SERVING_PROBE`。 |

## 设计要点与本轮调试中的修正

1. **账本归控制器所有**。`DomainRun.ledger` 是 `control.controller.ledger` 的只读属性，cutover 整体替换控制器的账本。此前每个域各持一份引用，EDP0 缩容 relabel 之后 EDP1 的引用过期，新接纳的请求在 EDP1 的 prefill 中找不到。
2. **目标账本在 KV 搬运前重建**。压实先确定每个请求的目标副本（`destination`），基波的前台逻辑步可能完成若干请求，`rebuild_target` 以当前账本和固定的目标分配重建目标账本并丢掉已完成行；否则切换后的账本保留了没有 KV 的请求，下一步 decode 失败。
3. **压实落到有段支撑的块号**。释放空段后块号空间有洞，压实目标必须是最低的、仍有段支撑的块号，`runs` 按这些块号给出。
4. **切换器只触碰本域副本**。fail-stop 两域共享一个 fake 后端，shrink 不搬 KV（故障使 EDP0 的 head 切分 KV 不可恢复，见计划 §4），校验步对重放请求记"重 prefill 后校验"；expand 搬 A 区、增量搬 B 区，`KVPlanBuilder` 规划的 A 区 token × head 组数必须等于实际拷贝量。
5. **决策提前一步与冻结**。第 s 步末触发锚点时先置 frozen 与 `phase=switching`，再 push 第 s+1 步的决策（不再接纳），切换在第 s+1 步开始时执行；基波的前台步是完整逻辑步（取决策、完成、timeline 行都走同一 `body()`），teacher forcing 的历史索引因此始终有效。
6. **失败 rank 保留完整 traceback**。`all_ready` 共识先于报错，`run.failure` 记录 `traceback.format_exc()`，服务器上定位不依赖重现。
7. **峰值不足为记录型驱逐**。`perform_switch` 的 `peak['mode']='evict'` 按接纳倒序驱逐回队首直到 `peak_check` 满足，事件 `peak_eviction` 与 `SwitchOutcome.evictions` 记录；`drain` 模式则先排空。
8. **运行模式决定校验步**。`validation_for`：correctness 下 fail-stop 用 `BitwiseValidation`、扩缩容用 `ToleranceValidation`，timing 下关闭；`--e2e-validate-cutover` 与运行模式矛盾即 `ValueError('contradicts')`，`result.json.config.run_mode`/`validate_cutover` 记录。

## 顺带修复（计划 §6 第 2 条）

- (a) Step 4 `async_worker.py`：`correctness_only` 下只记录 `overlap_ms`，不再 `require(overlap > 0)`；正式模式断言不变。
- (b) `quick_check.sh` 在 `source env.sh` 后把 `$(dirname "$PYTHON")` 放到 PATH 最前，并在启动驱动前执行 `"$PYTHON" -c "import megatron.core.datasets.helpers_cpp"`，失败即在任何 GPU 工作之前退出；`common.PYTHON_FIRST_PATH` 注入所有子进程环境（`safety.Children.spawn`、`run_model.environment`、`run_failstop_model.replay`），`common.require_megatron_helpers()` 在 `run_probe`/`run_topology`（Step 2–4 入口）/`run_model`/`run_failstop_model` 的 GPU 路径开始处调用（`--dry-run` 与 CPU 测试不经过）。
- 这两处修复只在以本次及之后清单检出的代码里生效：Step 3/4/5 现有服务器命令仍钉在各自的旧清单（`releases.json` 的 step3/step4/step5 未改），是否重钉由用户决定。

## 验证

```bash
cd code/current && python -B -X utf8 -m pytest -q -o addopts='' --noconftest -p no:cacheprovider tests/unit_tests/e2e tools/resharding/failstop tools/resharding/correctness/test_router_determinism.py tests/unit_tests/resharding/test_live_storage_lifecycle.py tests/unit_tests/resharding/test_live_defaults.py
```

结果见 [step6_cpu_evidence/cpu_tests.log](step6_cpu_evidence/cpu_tests.log)：217 项通过（e2e 30、failstop 150、路由确定性 10、live 基准 27），其中：

- `test_controller.py`：控制器全部策略与 `e2e_sim` 同轨迹（含容量缩小、晚一步生效）、D16 重组、峰值检查与记录型驱逐。
- `test_block_kv.py`：段分配/释放、压实保内容且区连续、`KVPlanBuilder` 两种映射两种渲染字节一致、任务数 ≤ 2×27×2×16 且不随请求数增长、`prepare_step`、注意力参考/split-KV/自检回退。
- `test_requests_and_policies.py`、`test_executor_and_results.py`：请求文件 hash 核对、两种解码策略、四种锚点、执行器流程（冻结/峰值/校验/事件）、运行模式与校验策略、timeline 六项耗时、`check`。
- `test_driver_fake.py`：fake 后端完整循环——扩缩容 2→4 与 4→2（teacher forcing，世界 8）、空前向不写 KV/不计 token/只进 `t_sync`、fail-stop fail→replay→回队→expand（greedy，两域、重放 token > 0、缩容字节全零、扩容搬 KV 且校验通过、timing 模式显式开校验报错）。
- `test_gloo_comm.py`：两进程 Gloo 控制组，领导者决策到达对端，`all_ready` 为全局 MIN。
- `failstop/test_serving.py`、`test_quick.py`：快验 runner 的 summary 标志、两次运行与测试结果；驱动的 `--step 6` 单节点、预算 300 s、结论行、失败路径与无两节点形式。
- `test_live_storage_lifecycle.py` 新增 `decode_fn` 用例：注入的前台步被每个 overlap 步调用且 `tpot_mean_ms` 取其返回值。

快验 runner 本机直跑（[quick_runner_summary.json](step6_cpu_evidence/quick_runner_summary.json)）：两次运行与 30 项 e2e 测试共 12.5 s，远在 300 s 预算内；扩缩容运行 44 逻辑步、两次切换校验通过（[scaling_result.json](step6_cpu_evidence/scaling_result.json)），fail-stop 运行 28 逻辑步、重放 33 token、`fail → shrink_done → rejoin_start(≥ fail+1 s) → expand_done`（[failstop_result.json](step6_cpu_evidence/failstop_result.json)、[switch_events_EDP0](step6_cpu_evidence/failstop_switch_events_EDP0.jsonl)）。fake 后端的时间是模拟器形状的玩具值，不是性能数据。

## 快验入口

`bash quick_check.sh --step 6 --node <NODE> --single-node`：读 `releases.json["step6"]` 检出实现提交，`quick_driver.py` 以 300 s 预算运行 `run_serving.py`（suite `s6q_<UTC>_<id>_single`），后者在 `<run>/<NODE>/` 下写 `scaling/`、`failstop/`（`inputs/`、`command.txt`、`out/{result,check}.json`、timeline、事件）、`pytest.log` 与 `summary.json`；PASS 要求 `all_checks_passed`、两次运行切换序列与校验正确、pytest 通过、`backend=fake`、`single_node=true`、`gpus_used=0`、`quick_mode=true`、`correctness_only=true`、`shared_gpu_mode=false`、`timing_valid=false`、`accepted=false`。结论行 `QUICK CHECK step=6 result=PASS (fake backend: controller/kv/anchors/decode/timeline, failstop cycle; runs=…; tests=…) backend=fake single_node=true accepted=false logs=…`。服务器 Python 环境需有 `pytest`（runner 启动时检查，缺失即失败并写 `error.txt`）。

## 边界与未完成项

- 真实模型路径全部属于 B 部分：`block` 后端、Megatron `is_block_kv` 三处钩子、Triton 分页解码 kernel（现 `select_implementation` 在无 kernel 时回退 `torch_splitkv` 并记录原因）、Step 3/4 迁移接入（`as_shards()` 渲染已就绪，未接 `mixed_transport`）、B1 单域固定 TP4 GPU 运行、B2、B3。
- 多进程账本复制：当前非领导者 rank 不维护账本副本（单进程 fake 运行里两域共用控制器账本），`GlooComm` 只验证了广播与共识原语；B1 需让每个 rank 把广播来的 `Decision` 应用到本地账本。
- fake 后端不走注意力 kernel（`--e2e-attention` 被接受但不生效），KV 内容校验用块级逐位相等代替 logits 比较。
- 正式负载（ShareGPT 子集）、冻结 profile 与 §4 的 10% 峰值余量核对未做，当前只有 `toy_inputs`。
- 两处顺带修复对 Step 3/4/5 现有清单不生效（见上）。


---

# B 部分：真实模型接入（2026-10-18）

实现提交：`ab5a4a90a1088df232eadcce0b5a2dd0ba152dba`（含 A 部分 `0f1776a0ff0929114d68318f62e7e2c3d957f508` 之后的全部改动），快验清单 `a07bde7a2a357581505f8d72988839d6401dc222`（`releases.json["step6"]`）。用户指示（2026-10-18）：Step 3/4/5 两节点快验已在服务器通过；B 部分一次写完并与 Step 3/4 重钉一起交付；**本轮暂停 Triton paged decode kernel**，decode 固定用 torch split-KV（`--e2e-attention torch`），`auto` 模式与 Triton 接口保留但不实现不调试，B1 去掉 Triton 自检项。

状态：BlockKVContext 的三处 Megatron 钩子、真实模型 BlockBackend、远程控制协议、B3 服务 worker 与控制门、B1/B2 torchrun、`quick_check.sh --step 6 --layer a|b1|b2|b3` 已实现并通过 CPU 测试（含 16 个 CPU actor 的 B3 协议模拟，emulate 与 real 两种模式）；**服务器 GPU 全部未运行**，`accepted=false`。

## 实现

| 文件 | 交付 |
|---|---|
| `megatron/core/transformer/attention.py`、`multi_latent_attention.py` | 三处 `is_block_kv` 钩子（计划 §3.3）：`_adjust_key_value_for_inference` 在 `inference_context is None` 判断之后交给 `context.adjust_key_value(layer_number, q, k, v)` 并返回 `no_mask`；MLA 核心注意力分发到 `context.core_attention(layer_number, q, k, v, mask, dense_attention=self.core_attention, softmax_scale=...)`；MLA 的 rotary 切片改为 `rotary_pos_emb[context.rotary_positions]`（逐 token 位置，yarn 的 mscale 与位置无关）。静态 KV 路径（`StaticInferenceContext`）不受影响。 |
| `tools/resharding/e2e/block_kv.py` | `BlockKVContext` 前向侧：`begin_forward(mode, device)`（prefill / decode / dummy：位置向量、按段分组的写索引、块对角 mask 或块表+长度）、`adjust_key_value`（逐段 index 写入 K/V，dummy 不写）、`core_attention`（decode 走分页实现，prefill 走 Megatron 稠密注意力 + 块对角 mask，dummy 对自身）、`is_decode_only`/`config.materialize_only_last_token_logits=False`（每个 token 的 logits 都要读）。`BlockPool._new_segment` 零填充并优先回填空洞；`compact` 的目标块号恒为 0..n-1（`KVPlanBuilder` 从 0 计数的 token 坐标因此恒成立）。 |
| `tools/resharding/e2e/paged_decode_triton.py` | 仅接口（`KERNEL = None`、`HAVE_TRITON`），本轮不实现；`select_implementation('auto')` 回退 split-KV 并记录原因，`torch` 为 B 部分固定选择。 |
| `tools/resharding/e2e/attention_paged.py` | split-KV 回退的累加器放到 query 所在设备。 |
| `tools/resharding/e2e/block_backend.py` | 真实模型后端 `BlockBackend(forward, replica, tp, prompts, geometry, ...)`：一个 rank 服务一个 dense 副本；prefill 为本步接纳请求的一次块对角前向（输入 `[1, S]`，每请求末位 logits 交给解码策略），decode 为一次分页前向（输入 `[1, T]`，每请求一个 token），dummy 一个 token 不写 KV；histories/last_token 记账，重放请求的历史来自 `history_source`；`bind(tp, context)` 在 cutover 时切换布局；CUDA event 计时。 |
| `tools/resharding/e2e/remote_control.py` | 跨进程决策协议：`ControlService`（控制节点线程，持有 AdmissionController，按 `ctl/<domain>/<n>` 顺序处理 stepped/phase/requeue/down/ledger/evict，推 `decision/<domain>/<step>`，`trigger()` 把切换方向带进下一个决策，`hold` 让域在切换序列完成前不结束）与 `RemoteControl`（域侧：取决策、镜像账本、每 rank 计数消息只由领导者写入、按 dense 副本发布重放历史、驱逐结果域内广播、镜像 digest 一致性检查）；`MemoryChannel` 为测试替身。 |
| `tools/resharding/e2e/live_runtime.py`、`controller.py`、`plan_executor.py` | `Decision` 增加 `rows`/`queue_len`/`switch`；`Ledger.admit`；控制面接口 `apply/set_phase/set_down/replace_ledger/evict_for_peak/peak_ok`（ControlNode 与 RemoteControl 同形）；`DomainRun.local_replicas`（本进程跑前向的副本）；`body()` 的 prefill 带解码策略、完成数按本进程副本计；无执行器时切换由决策的 `switch` 触发；每个 rank 都 `push`（远程控制只在领导者写）；`run_until_done`。 |
| `tools/resharding/failstop/serving_worker.py` | B3 worker（继承 Step 5 `model_worker.Worker` 的世界/组/传输/权重审计）：e2e Driver 在每个 rank 内服务其 dense 副本（block KV，决策来自总线），`ServingSwitcher` 按角色实现 perform_switch 协议——幸存者缩容（TP2 目标，权重来自残余 TP4 + EDP1，不搬 KV，账本 relabel）、幸存者扩容（压实 → TP4 目标 + 按压实表克隆的目标池 → `KVPlanBuilder.as_shards` 的 A 区 KV 与权重同一混合计划 → 前台增长步 → 仍在跑请求的 B 区增量 → 目标 KV 对照幸存者落盘快照的 head 映射逐位检查 → cutover → 向回归卡 handoff 账本/历史/next token/表长度）、EDP1 作为发送方参与各波（扩容期间边轮询边前台步）、emulate 受害者在一次切换内走完释放/等待/加入、real 受害者为 epoch-2 新进程 `join_as_victim`。控制组每个世界 epoch 重建（EDP0 另有仅幸存者的 `EDP0_serving` 组，切换中使用）。 |
| `tools/resharding/failstop/serving_probe.py` | SL3060 控制门：Step 5 的时钟/预检/拓扑/cuda 门 + `ControlService` 线程；`completed_count` 锚点触发后向两域下达 shrink，等全部 switch_ack → kill/release → transition 1 → build → 阶段 1 报告（shrink_done）→ rejoin 到期（`t_fail + seconds`，且 shrink_done）→ reclaim/spawn → expand → transition 2 → build → 阶段 2 报告 → 排空 → 四个 dense 副本的 summary → `result.json`/`check.json`/`events_actual.json`（06_Step6 §4 字段组）。 |
| `tools/resharding/failstop/serving_layers.py` | B1/B2 torchrun 入口（单节点单布局无切换）：所有 rank 在逻辑时钟上跑同一接纳控制器（决策一致，Gloo 只做步末共识）。B1：静态 KV 路径（StaticInferenceContext + `_decode`）与 block KV 路径对同一 teacher-forcing 探针请求的逐步 logits nrmse/cosine（单独与批内），M1 单步时间（静态、block 单请求、block 批量），池字节对照 token/段记账。B2：确定性模式、变长批跨块边界、释放补入、末尾一副本空，同进程两遍逐位记录（`determinism.bitwise_record`），记录落盘供 runner 跨进程比较。 |
| `tools/resharding/failstop/run_serving.py`、`quick_driver.py`、`quick_check.sh` | `--layer a|b1|b2|b3`：a 为原 fake 快验；b1 为 tp4 一次 torchrun（默认模式）；b2 为两个独立 TP2 进程（GPU 0–3 / 4–7，`--rdzv-endpoint` 不同端口）并发后再跑 TP4，runner 跨进程逐位比较；b3 为两节点 launch（`SERVING_PROBE`，`run_probe.run_case`，`--mode emulate|real`，需 `--profile-file` 与 SL3061 上的 `--reference-root`）。预算均 300 s；b3 走两节点握手。 |
| `tools/resharding/failstop/model_worker.py`、`megatron/core/failstop_rebind.py` | Step 5 缺陷修正：服务中的模型在每个世界 epoch 都重绑定；emulate 复用同一组对象时用 `ModelBinding.refresh()` 跟随 generation（此前 emulate 模式下 EDP1 的第一次前向会被判为 stale world；Step 5 CPU 测试的假 runtime 不走 forward hook 故未暴露）。 |
| `tools/resharding/failstop/controller.py`、`supervisor.py` | `SERVING_PROBE` 分发：控制门 `serving_probe.control`，worker `serving_worker.py`，emulate 监管器。 |

## 调试中修正的问题（CPU 模拟暴露）

1. 领导者变更后的消息序号：只有领导者计数 `stepped` 时，故障后 EDP0 换领导者（real 模式 rank 重编号）导致 `ctl/<domain>/<n>` 冲突；现在每个 rank 都计数、只有领导者写。
2. 重放历史按 dense 副本发布：域领导者只持有自己副本的历史，`history/<seq>/<len>` 由每个副本的首个幸存者发布。
3. 切换中的控制组：世界重建后立即重建域通信组，但扩容切换中只含幸存者（`EDP0_serving`），回归卡在 cutover 后加入。
4. 工作负载在 rejoin 之前排空：控制节点 `hold` 域直到扩容完成（空步），`result.json.failstop.stats_valid=false` 并写明原因（06_Step6 §6 第 6 条）。
5. 阶段报告先于 `session.complete()`：控制门在过渡态比较 rank map（`source_eligible`）。
6. 增量计划只含仍在跑的请求：基波前台步中完成的请求退出计划并释放其目标块。
7. 两个 EDP0 副本的 KV 快照目录分开；per rank 的 timeline 只记本进程副本的完成数，控制门按 dense 副本汇总四份 timeline。

## 验证

```bash
cd code/current && python -B -X utf8 -m pytest -q -o addopts='' --noconftest -p no:cacheprovider tests/unit_tests/e2e tools/resharding/failstop tools/resharding/correctness/test_router_determinism.py tests/unit_tests/resharding/test_live_storage_lifecycle.py tests/unit_tests/resharding/test_live_defaults.py
```

结果见 [step6_cpu_evidence/cpu_tests_b.log](step6_cpu_evidence/cpu_tests_b.log)：227 项通过（含新增的 e2e 块上下文/块后端/远程控制测试、`test_serving_protocol.py` 的 emulate 与 real 两次 16-actor 模拟、`test_quick.py` 的分层用例；此前 A 部分为 217 项）。新增：

- `tests/unit_tests/e2e/test_block_context_forward.py`：三处钩子的 CPU 对照——块对角 prefill、跨块边界的分页 decode 与 dummy 前向，对每个请求等于其自身历史上的稠密注意力；Triton 缺席时的回退记录。
- `tests/unit_tests/e2e/test_block_backend.py`：玩具两层注意力模型按 Megatron 的方式调用钩子，批量 block KV 服务提交的 token 与逐请求稠密解码一致；dummy 不写 KV；释放回收段；重放从历史源取 prompt + 已提交 token。
- `tests/unit_tests/e2e/test_remote_control.py`：fake 后端 fail-stop 全循环经远程控制通道（ControlService 线程 + 两个域镜像）完成，镜像 digest 检查通过。
- `tools/resharding/failstop/test_serving_protocol.py`：16 个 CPU actor + 假 runtime（TP all-reduce 的玩具 logits 依赖全部驻留权重）跑 B3 worker 与控制门，emulate 与 real 各一次：12 个请求全部完成、shrink/expand 校验通过、扩容 KV 逐位检查通过、`result.json`/`check.json`/`events_actual.json` 写出、16 份 `serving_checks_*.json`，decode 实现记录为 `torch_splitkv`。
- `test_quick.py`：`--step 6 --layer b1|b2` 单节点、`--layer b3` 两节点（无 `--single-node`），结论行与失败路径。

## 边界与未完成项

- 服务器 GPU 未运行；B1/B2/B3 的服务器通过情况、M1 数字、nrmse/cosine、B3 的 TFS 分解均待回传。
- Triton kernel 按用户指示暂停；`attention_impl` 固定 `torch_splitkv`。
- B3 传输为同步波次（Step 4 重叠未接）；EDP1 在波次执行期间不服务（stall 行记录），这与 Step 5 一致。
- B3 的 `totals.duration_s` 是四份副本 timeline 的步时累加（与 `--check` 自洽），墙钟见 `failstop.serving_wall_s`；`timing_valid=false`。
- 峰值记录型驱逐对真实 worker 走同一条 `evict_for_peak` 路径，但本轮负载不会触发，服务器未验证。
- B3 快验的负载是 `toy_inputs`（24 条 5–7 token 请求），rejoin 在 fail 后 5 s；正式负载与 45 s 仍按总计划 §7。
