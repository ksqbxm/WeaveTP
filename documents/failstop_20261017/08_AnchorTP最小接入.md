# AnchorTP 最小接入：第二恢复方法、正式运行模式与分析器

状态（2026-10-18）：**已实现，CPU 已验证（248 项），服务器 GPU 未运行。** 实现提交见 §8。2026-10-21 修正两处 fail-stop 缺陷与 b3 两节点配置、新增 `--shared-gpu-ok` 与 inflight 快验，见 §9（CPU 258 项）。本文件替代此前的 "Step 7 计划"；总计划 §2 L14「不读取、合并或修改 AnchorTP 实现」与 §9 L170「仅 WeaveTP」按用户 2026-10-18 确认修订（见 §8）。

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
- `result.json` 另有 `gpu_sharing`（`shared`|`exclusive`，即 `config.shared_gpu_mode`；2026-10-21 起）。
- `launch_result.json`：`run_probe.run_case` 的 launch 结果（状态、清理、pidfd、`timing_note`），两节点各一份；`result.json`/`check.json`/`events_actual.json` 只由 SL3060 控制门写（2026-10-21 之前 run_case 覆盖了它们，§9）。
- `--shared-gpu-ok` 的 timing 运行：`<launch>_preflight_memory.json`（显存检查表）、`<launch>_foreign_gpu_processes.json`（`before`/`after` 两次快照：每个外来 GPU 进程的 `gpu`、`uuid`、`pid`、`user`、`command` 前 120 字符、`used_mib`，及各卡 `gpu_used_mib`）；`progress_*.json` 每条记录与 `matrix_manifest_*.json` 带 `gpu_sharing`。
- `events_actual.json`：`scenario`、`method`、`run_mode`、`clocks`；fail-stop 字段只在 failstop 场景下出现，`worker_events_s` 为 worker 侧 fail/shrink_done/rejoin_start/expand_done（已移到门时基）。
- `controller_result.json.metrics`：`method`、`scenario`、`run_mode`、`serving_wall_s`、`edp0_switches`。

## 5. 分析器（`analyze_failstop_e2e.py --out DIR <launch 或根目录>...`）

输入 `result.json` + `events_actual.json`；先校验逐请求行（token 数 = output_len、单调、首/末与 `t_first/t_done` 一致、总 token 守恒、哈希一致），不一致的 launch 记入 `rejected.json` 并退出码 1。输出 `metrics_<launch>.json`、`per_second_<launch>.csv`（全局/EDP0/EDP1/各副本逐秒新 token）、`launch_metrics.csv`、`summary.csv`（按 方法×场景×模式×`gpu_sharing` 的 mean、ddof=1 std、n_valid、n_censored；`launch_metrics.csv` 也有 `gpu_sharing` 列）。`--shared-gpu-ok` 的 launch 照常接受，自成一组，不与独占数据混合。

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

248 项通过（第 0 步后 227；日志 [anchortp_cpu_evidence/regression_pytest.log](anchortp_cpu_evidence/regression_pytest.log)）。新增：`test_anchortp_plan.py`（固定路径优先级、无主机源、复制的分批用例）；`test_serving_protocol.py` 16-actor 模拟 weavetp/anchortp × emulate/real（anchortp：`overlap_steps==0`、delta 0、shrink 用跨机源、expand 无跨机）、nofault 场景、timing 模式（无审计、无 `kv/`、`step_times` 不记录）；`test_failstop_e2e.py`（timing/correctness 配置、拒绝项、dry-run 清单、独占预检拒绝忙卡）；`tests/unit_tests/e2e/test_metrics.py`（恢复判据、删失、故障前不足 10 s、重放后 TTFT/TPOT、校验拒收、summary 与 CLI）；`test_make_failstop_requests.py`（筛选、固定输出长度、前缀性质、plan 载入）；`test_driver_fake.py` 的逐 token 戳断言；`test_quick.py` 的 b3 `--method`。本机缺 `pydantic` 时 2 项 Megatron 参数解析测试失败（环境问题，已安装后通过）。2026-10-21：258 项通过（250 + §9 的 8 项；日志 [anchortp_cpu_evidence/regression_pytest_inflight.log](anchortp_cpu_evidence/regression_pytest_inflight.log)）。

## 7. 已知限制

- **独占预检的 MPS_OWNER 例外（2026-10-18 用户授权）**：SL3060 有别人的常驻 MPS（`nvidia-cuda-mps-server`，约 56 MiB）。timing 运行前 `export MPS_OWNER=<该用户名>`，`serving_config.require_exclusive_gpus` 与监管器 `inspect` 对属主等于 `MPS_OWNER` 的 server/daemon 放行：管道定位不到或无权限记 `owner_exception` 后继续，该卡须 util 0 且显存 ≤ 200 MiB，MPS 客户端必须为空，其他进程和其他属主仍拒绝（`safety.mps_snapshot/check_gpu_processes` 的 `mps_owner` 参数；T08 `check_idle` 的同一规则）。不设 `MPS_OWNER` 时行为与总计划 §4 原文相同。

- 24 条玩具请求在 rejoin 前排空，b3 快验的 `stats_valid=false`（如实记录，正式集不会）；扩容时有在途请求的路径由 inflight 快验覆盖（§9）。
- timing 模式跳过参考 hash 与 KV 快照比对，正确性只由 correctness 模式证明；两模式的迁移计划由 `source_plan_*.json` 对照。
- 传输仍是同步波次（Step 4 的重叠不在本实验内），两方法相同；EDP1 在波次执行期间不解码（与 B3 相同），只在等待计划键时解码。
- shrink TTP 缺少降级标定参考时用自参考，报告里标出。
- 服务器 GPU 全部未运行；b_max=8、seq_length=2048 等正式参数待首次上机核对显存。

## 8. 交付登记

- 第 0 步日志修正：`0d4b17acbb2784036dade9db9e84b9478579e782`。
- 实现（A）：`16b1617af90eb4ada0bd32d715dcc45ccd5ca91b` + 启动脚本目录修正 `f2a85e9074e11c98605f72ce25f53c29e2f5fa7b`+ 独占预检 MPS_OWNER 例外 `560361c60c6c1ad503b9afda14cd984e011dbd2a`（`releases.json["step6"]` 指向最后者；此前的 Step 6 清单 `8a1e576` 作废）。服务器命令（C，含 bundle）：见 [Step6_AnchorTP接入_服务器命令.md](Step6_AnchorTP接入_服务器命令.md)。
- 2026-10-21 修正（§9）：实现（A）`c4df30a8a2bc2b9435cffde17cb2463be7eea113`，`releases.json["step6"]` 改指它（`560361c` 作废）。服务器命令（C，本次放在 bundle 所含的提交里）：见 [Step6_inflight快验_服务器命令.md](Step6_inflight快验_服务器命令.md)。
- 2026-10-21 fail-slow 交付（`codex/failslow`，[09_failslow主图.md](09_failslow主图.md) §9）：`releases.json["step6"]` 改指 `073a50f41e70ee54551f3f2e2613c60203697ad1`（与 `failslow` 同一实现；新增 emulate fail-stop 的 restart，不做参考 hash 比对；修正 worker 的参数解析与控制门的导入，见 09 §11，此前真实模型 worker 启动即退出；weavetp/anchortp 的 fail-stop 迁移行为同 `c4df30a`）。
- 2026-10-26 fail-stop kill 交付（`codex/failstop-online`，[11_failstop在线化依据.md](11_failstop在线化依据.md)）：`releases.json["step6"]` 改指 `c4b203e0d059700088f4b23fc84b9ab88605e935`（新增 weavetp `--mode kill`，含 `d8b8cd7` 的回放历史修复与峰值驱逐修复；`switch_ack` 改在峰值驱逐之后发，所有模式同一路径；emulate/real 的 weavetp、anchortp、restart 迁移行为不变；首版 `6f2fc84` 作废，见 11 §9）。`073a50f` 带两个回放历史问题，用它跑出的 step6 结果作废。

## 9. 2026-10-21 修正：扩容在途请求、结果文件、b3 两节点配置；`--shared-gpu-ok`；inflight 快验

| 项 | 问题 | 修正 | CPU 测试 |
|---|---|---|---|
| 1 扩容 KeyError | `e2e/live_runtime.Driver.body` 逐 token 记 `run.request_log[seq]['t_tokens']`；扩容时加入的 rank（emulate 受害者在降级期间看不到新接纳，real 替换进程什么都没看到）解码交接来的请求即 `KeyError`，扩容时只要有在途请求就崩。b3 的 24 条玩具请求在 rejoin 前排空，没暴露；正式请求集必然触发 | 取 `codex/failslow@be83967` 的同一行（`setdefault`）。加入 rank 的不完整日志不进结果：summary 只由各副本最低的幸存 slot 发布 | `test_serving_protocol.test_expand_with_requests_in_flight`：60 条、rejoin 1 s，emulate 与 real 各一次；断言加入 rank 确实解码了没见过接纳的请求、全部完成、检查与扩容 KV 检查通过、分析器接受门的文件。未修时两个子测试都以 `peer failure: <seq>`（actor 内 `KeyError`）失败。CPU 模拟的 bus 等待上限 8 s，200 条会超时，故用 60 条 |
| 2 结果覆盖 | `run_probe.run_case` 把 launch 结果写进 `result.json`，SL3060 还写 `events_actual.json`，覆盖门写的 W4 结果与事件：`analyze_failstop_e2e.py` 读不到正式 launch，标定读的 `serving_wall_s` 也不在顶层 | run_case 只写 `launch_result.json` | `test_cleanup`：门的两个文件保持原样；上面的回归测试断言门 `result.json` 的 `serving_wall_s` 等于控制器指标 |
| 3 b3 两节点配置（本次发现） | b3 把玩具输入写在 `<run>/<node>/inputs`，两节点 `config['serving']` 的路径不同，门逐键比较 `PROBE_KEYS`（含 `serving`），上机必报 `node config mismatch: serving`；CPU 模拟两节点共用一个 config，测不到 | 输入写到 `<run>/inputs_<layer>`：两节点同一路径，layer 不同仍被拒 | `test_serving`：b3 与 inflight 两节点的 `PROBE_KEYS` 逐键相等 |
| 4 `--shared-gpu-ok`（用户 2026-10-21 要求） | timing 运行要求 16 卡独占 | `run_failstop_e2e.py --shared-gpu-ok`（只用于 timing；correctness 仍是 `--correctness-only`）：不做独占预检，只查每卡空闲显存够本角色（`model_memory` 场景表，KV 按 `b_max` 条 `seq_length`，`serving_config.require_gpu_memory`）；`shared_gpu_mode=true`（监管器按共享规则记录外来进程、按本任务 PID 确认回收），`timing_valid` 保持 true，`result.json` 的 `gpu_sharing='shared'`；每次 launch 前后记录外来 GPU 进程（`safety.foreign_gpu_processes`，排除本 launch 登记的 PID）；分析器接受并按 `gpu_sharing` 分组 | `test_failstop_e2e`（配置、拒绝、显存检查、前后快照）、`test_shared_gpu`（快照字段）、`test_metrics`（共享 launch 被接受且自成一组） |
| 5 inflight 快验 | 没有覆盖“扩容时有在途请求”的真机入口 | `quick_check.sh --step 6 --layer inflight`：b3 的同一条路径（correctness、共享 GPU），200 条玩具请求、输出 96–288 token（约 2000 个 decode 步），预算 600 s；SL3060 另要求 `inflight_at_expand ≥ 1`（门 `result.json` 中最后一次接纳在 EDP0 扩容 cutover 前、完成在其后的请求数，`run_serving.handed_over`），并对门的文件跑 `analyze_failstop_e2e`（退出码 0、无拒收） | `test_serving`（有/无在途请求两种门文件）、`test_quick`（两节点、600 s、结论行）；`handed_over` 在上面的 16-actor 回归里与加入 rank 实际解码的请求对上 |

限制：inflight 负载按估计的步时定（服务器没有实测）；负载在扩容前排空时以 `inflight_error` 明确失败，需加大条数再交付。快验 tar 包不含 `inputs_<layer>`（玩具输入确定性生成，哈希在 `events_actual.json`）。`--shared-gpu-ok` 的数据是否进论文由用户决定，结果与汇总里都带 `gpu_sharing`。
