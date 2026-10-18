# AnchorTP 最小接入：第二恢复方法、正式运行模式与分析器

状态（2026-10-18）：**已实现，CPU 已验证（248 项），服务器 GPU 未运行。** 实现提交见 §8。本文件替代此前的 "Step 7 计划"；总计划 §2 L14「不读取、合并或修改 AnchorTP 实现」与 §9 L170「仅 WeaveTP」按用户 2026-10-18 确认修订（见 §8）。

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 本轮用户确认的设定：重复次数 N=3；AnchorTP 的静态带宽取冻结 profile 的 p2p 矩阵；停服期间 EDP1 继续服务并同时作为源；主机库、故障检测、degraded 场景、多轮 rounds、主机候选源、Triton 一概不做。

## 1. AnchorTP 机制依据（复现代码，`codex/anchortp-faithful@eba3bc8`）

A = 该分支 `code/current/`。本实现只取 `megatron/core/resharding/anchortp.py` 的两样东西：`configure()`（L27-45）关掉的 WeaveTP 优化清单，和 `schedule()`（L81-107）的静态带宽排序与分批；两者原样复制到 `tools/resharding/failstop/anchortp_schedule.py`，文件头注明来源。其余（`execute()` 的 ReshardPlan 执行、`HostTask`）不用：传输仍走 `mixed_transport.execute`。

| 问题 | 复现代码的回答 | 依据 | 本实现 |
|---|---|---|---|
| 故障后布局 | 只有计划内 TP2↔TP4 切换；无故障检测、无非等宽 TP | A README L17；`anchortp.py` docstring | 故障时刻由控制门预先指定，恢复 = 故障时刻触发的计划内切换：缩容 TP4→TP2（EDP0 剩 4 卡），扩容 TP2→TP4 |
| 权重来源 | 固定交集源：每个目标切片取持有同一切片的 rank；禁用改道/aware shrink/packing；任务按静态带宽降序、task_id 并列；主机重载路径存在但无故障时不生成 | `anchortp.py` L27-45、L81-107、L110-152；A `benchmark_live_moe_tp.py` L1550-1567 | `anchortp_plan.build_fixed_plan`：同一原子化，每原子按「本卡 > 同机 > 跨机」稳定选择；候选 = EDP0 幸存卡 + EDP1，故障卡不在成员里；无主机 |
| KV | 无故障切换：冻结 decode 与 KV offset，整体迁移快照内 KV，无增量（`delta_tokens==0` 断言） | A `benchmark_live_moe_tp.py` L2111-2116、L2172-2185、L2372-2373 | 缩容：EDP0 在途请求重 prefill（与 WeaveTP 同，`FailstopHooks`）；扩容：A、B 区一次搬完，无增量；cutover 断言目标块表长度 == 冻结长度。结果字段 `kv_semantics_status='未与论文核对'` |
| 恢复期间是否停服 | 停服：`overlap_steps=0`、`tpot_samples=0`，迁移期间无解码；第二副本不在代码里 | `anchortp.py` L45、L238-245；A L2268-2269 | EDP0 两次迁移期间无前台 step（`growth={'shrink':0,'expand':0}`）；EDP1 照常服务并作源（用户确认） |
| rejoin 后扩回 TP4 | 计划内 2->4 用同一执行器；空白新进程的 rejoin 未实现 | A README L17 | rejoin 卡作空目标，权重 + KV 从 TP2 幸存卡交集取（本拓扑 expand 无跨机任务） |
| 复现配置 | 单机 8×5090、DeepSeek-V2-Lite、TP2↔TP4、4 次切换；新实现无 GPU 性能结果 | A `run_cycle_grid.sh` L3-15、L39-40；A README L21-25 | 本实验：两节点 16 卡 fail-stop 场景，与 WeaveTP 同一服务循环 |

## 2. 方法差异的唯一位置

- `serving_worker.SWITCHERS = {'weavetp': ServingSwitcher, 'anchortp': AnchorTPSwitcher}`，`ServingWorker.switcher_class` 按 `config['method']` 选；`switcher_factory(run, direction)` 构造 `SwitchSpec(direction, assignment=None)`——`assignment` 是请求到（源副本，目标副本）的映射，None 表示请求留在原副本；以后"一个副本拆成两个 / 两个合成一个"的布局变化在这里给显式映射，协议其余部分不假设一对一。
- `model_worker.PLANNERS = {'weavetp': build_mixed_plan 包装, 'anchortp': anchortp_plan.build_fixed_plan}`，`Worker.transfer` 调 `self.planner(sources, targets, profile, membership=, epoch=, rank_map=) -> (tasks, cache_key, record)`；源/目标是全局坐标的平铺分片列表，不限定两侧副本数。
- 切换器类属性：`zones`（扩容 base 波 / delta 波各搬哪些 KV 区）、`growth(serving)`（迁移期间前台步数）。WeaveTP：`{'base': ('A',), 'delta': ('B',)}`、`serving['growth']`；AnchorTP：`{'base': ('A','B'), 'delta': ()}`、全 0。
- 服务循环（`e2e/live_runtime.Driver`）、控制门（`serving_probe.control`）、事件、timeline、结果写入全部共用；`PROBE_KEYS` 含 `method`，两节点配置必须一致。

## 3. 改动清单

| 步 | 内容 | 文件 |
|---|---|---|
| 0（单独提交 `0d4b17acbb2784036dade9db9e84b9478579e782`） | `t_first`/`t_done` 用 `clock.stamp(step_time)`（WallClock 不再重复加 step_time）；重放不覆盖首次 `t_admit`/`t_first`，`admits[]` 记每次接纳；`t_tokens[]` 逐 token 时间戳（只由服务该请求的进程记）；`results.merge_request_logs/request_row` 按 seq 并集合并 | `e2e/live_runtime.py`、`e2e/results.py`、`examples/rl/run_e2e_scaling.py`、`failstop/serving_layers.py`（StepClock.stamp）、`failstop/serving_probe.py` |
| 1 | `DEFAULTS['method']`、`PROBE_KEYS`、`build_result(method=)`、`METHODS` 说明表、`metrics.edp0_switches`、`switcher_class`/`growth` 按方法 | `failstop/common.py`、`failstop/serving_probe.py`、`e2e/methods.py`、`failstop/serving_worker.py` |
| 2 | `anchortp_schedule.py`（复制）、`anchortp_plan.py`（固定选源适配）、`PLANNERS` 与 `transfer` 的 planner 参数、`SwitchSpec`、`kv_moves/expand_prefix/expand_delta` 的 zones+assignment 参数、`AnchorTPSwitcher` | `failstop/anchortp_schedule.py`、`failstop/anchortp_plan.py`、`failstop/model_worker.py`、`failstop/serving_worker.py` |
| 3 | `serving_config.py`（quick 与正式共用的配置构造、timing/correctness、独占预检 `require_exclusive_gpus`）、`run_failstop_e2e.py/.sh`（`--method --scenario --mode --run-mode --launches`，`matrix_manifest`、`progress`）、`nofault` 场景（控制门 `fault_cycle` 分离，监管器不等 kill/spawn）、timing 下不做权重 hash/KV 快照/逐位校验且不写 `kv/*.pt`、`step_times`、结果时基统一到门时钟（`clocks`、`time_base`）、`make_failstop_requests.py`、`e2e/metrics.py` + `analyze_failstop_e2e.py` | `failstop/serving_config.py`、`failstop/run_failstop_e2e.py`、`failstop/run_failstop_e2e.sh`、`failstop/serving_probe.py`、`failstop/supervisor.py`、`failstop/serving_worker.py`、`e2e/make_failstop_requests.py`、`e2e/metrics.py`、`failstop/analyze_failstop_e2e.py` |
| 4 | b3 快验 `--method weavetp|anchortp`：`quick_check.sh`、`quick_driver.py`、`run_serving.layer_b3`（经 `serving_config`，anchortp 通过标准加 `overlap_steps==0` 且 delta 字节 0） | `failstop/quick_check.sh`、`failstop/quick_driver.py`、`failstop/run_serving.py` |

## 4. 结果字段

- `result.json`：`method`；`config.{run_mode, experiment(scenario), method, timing_valid, shared_gpu_mode, validate_cutover}`；`requests[*].admits[{t, step, replica, layout, generated_at_admit}]`、`t_tokens[]`（gate 时基）；`failstop.method{name, source_policy, kv_semantics, kv_semantics_status, implementation_ref}`、`failstop.edp0_switches[{direction, overlap_steps, kv_delta_bytes, kv_prefix_bytes, weights_bytes}]`、`failstop.expand_kv_check_passed`（timing 为 null，`kv_check_note='skipped: timing run'`）；`clocks[domain/replica]{origin_local, offset_to_gate, shift_to_gate_s}`、`time_base`、`serving_wall_s`。nofault 时 `failstop=null`。
- `source_plan_0_{shrink,expand,delta}.json`：`method`、`planner`；anchortp 另有 `implementation_ref`、`constraints`、`batches`、`paths{local_copy, gpu_same_node, network}`、`bandwidth_source`。
- `events_actual.json`：`scenario`、`method`、`run_mode`、`clocks`；fail-stop 字段只在 failstop 场景下出现，`worker_events_s` 为 worker 侧 fail/shrink_done/rejoin_start/expand_done（已移到门时基）。
- `controller_result.json.metrics`：`method`、`scenario`、`run_mode`、`serving_wall_s`、`edp0_switches`。

## 5. 分析器（`analyze_failstop_e2e.py --out DIR <launch 或根目录>...`）

输入 `result.json` + `events_actual.json`；先校验逐请求行（token 数 = output_len、单调、首/末与 `t_first/t_done` 一致、总 token 守恒、哈希一致），不一致的 launch 记入 `rejected.json` 并退出码 1。输出 `metrics_<launch>.json`、`per_second_<launch>.csv`（全局/EDP0/EDP1/各副本逐秒新 token）、`launch_metrics.csv`、`summary.csv`（按 方法×场景×模式 的 mean、ddof=1 std、n_valid、n_censored）。

| 指标 | 定义 |
|---|---|
| shrink TFS / EDP0 缩容停机 | `t_fail` → EDP0 故障后首个提交 token |
| shrink TTP | 该首 token → 全局 5 s 滑动均值连续 3 s ≥ 95% × 缩容参考；参考 = `--degraded-reference`（tok/s），缺省取降级窗口（TP2 首 token→rejoin request）后半段的全局均值并标 `self` |
| expand time | `t_rejoin_start`（spawn）→ EDP0 expand 切换的 `cutover` 里程碑（TP4 ready） |
| expand TTP / full recovery | TP4 ready → 达到故障前 10 s 全局均值的 95%（判据同上）；full = `t_fail` → 该时刻；故障前不足 10 s 记无效，未达到记 censored |
| EDP0 扩容停机 | expand `switch_ack` 前最后一个 EDP0 token → TP4 ready 后首个 EDP0 token |
| P50/P99 TTFT、TPOT | 全程与 fail→full recovery 窗口；TTFT 从 `t_arrive`，TPOT=(末 token−首 token)/(n−1)，单 token 为 null，停顿计入 |
| 重 prefill | `replay_prefill_tokens`（prompt + 已提交历史）与重放请求数 |

与总计划 §8 的差别：无降级标定运行，shrink TTP 的参考按上表口径；每轮独立 launch，无一次 launch 多轮。

## 6. CPU 验证

命令（`code/current`）：

```
python -B -X utf8 -m pytest -q -o addopts='' --noconftest -p no:cacheprovider tests/unit_tests/e2e tools/resharding/failstop tools/resharding/correctness/test_router_determinism.py tests/unit_tests/resharding/test_live_storage_lifecycle.py tests/unit_tests/resharding/test_live_defaults.py
```

248 项通过（第 0 步后 227；日志 [anchortp_cpu_evidence/regression_pytest.log](anchortp_cpu_evidence/regression_pytest.log)）。新增：`test_anchortp_plan.py`（固定路径优先级、无主机源、复制的分批用例）；`test_serving_protocol.py` 16-actor 模拟 weavetp/anchortp × emulate/real（anchortp：`overlap_steps==0`、delta 0、shrink 用跨机源、expand 无跨机）、nofault 场景、timing 模式（无审计、无 `kv/`、`step_times` 不记录）；`test_failstop_e2e.py`（timing/correctness 配置、拒绝项、dry-run 清单、独占预检拒绝忙卡）；`tests/unit_tests/e2e/test_metrics.py`（恢复判据、删失、故障前不足 10 s、重放后 TTFT/TPOT、校验拒收、summary 与 CLI）；`test_make_failstop_requests.py`（筛选、固定输出长度、前缀性质、plan 载入）；`test_driver_fake.py` 的逐 token 戳断言；`test_quick.py` 的 b3 `--method`。本机缺 `pydantic` 时 2 项 Megatron 参数解析测试失败（环境问题，已安装后通过）。

## 7. 已知限制

- 24 条玩具请求在 rejoin 前排空，b3 快验的 `stats_valid=false`（如实记录，正式集不会）。
- timing 模式跳过参考 hash 与 KV 快照比对，正确性只由 correctness 模式证明；两模式的迁移计划由 `source_plan_*.json` 对照。
- 传输仍是同步波次（Step 4 的重叠不在本实验内），两方法相同；EDP1 在波次执行期间不解码（与 B3 相同），只在等待计划键时解码。
- shrink TTP 缺少降级标定参考时用自参考，报告里标出。
- 服务器 GPU 全部未运行；b_max=8、seq_length=2048 等正式参数待首次上机核对显存。

## 8. 交付登记

- 第 0 步日志修正：`0d4b17acbb2784036dade9db9e84b9478579e782`。
- 实现（A）：`16b1617af90eb4ada0bd32d715dcc45ccd5ca91b` + 启动脚本目录修正 `f2a85e9074e11c98605f72ce25f53c29e2f5fa7b`（`releases.json["step6"]` 指向后者；此前的 Step 6 清单 `8a1e576` 作废）。服务器命令（C，含 bundle）：见 [Step6_AnchorTP接入_服务器命令.md](Step6_AnchorTP接入_服务器命令.md)。
