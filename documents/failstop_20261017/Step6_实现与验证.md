# Step 6 实现与验证（A 部分：统一服务驱动的 fake 后端、CPU 测试与快验入口）

实现提交：`0f1776a0ff0929114d68318f62e7e2c3d957f508`（分支 `codex/failstop`；首个实现提交 `9a8a28da41c2a6de0903cc06b2934dd9dd55e5fe`，随后补两进程 Gloo 控制组测试）。快验清单提交 `@@MANIFEST@@`（`releases.json["step6"]` 指向实现提交），服务器命令见 [Step6_服务器命令.md](Step6_服务器命令.md)。计划稿见 [Step6_服务驱动计划.md](Step6_服务驱动计划.md)（已吸收 2026-10-18 五条审阅意见），W1–W4 设计依据见 [参考](参考_扩缩容实施计划_W1-W4设计依据.md)。

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
