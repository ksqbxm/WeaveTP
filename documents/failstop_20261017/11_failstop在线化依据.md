# 11. fail-stop WeaveTP `--mode kill`：降级在预建组上在线迁移，恢复沿用 real 重建

状态（2026-10-26）：**已实现，CPU 已验证，服务器 GPU 未运行。** 分支 `codex/failstop-online`，基于 `codex/failslow-online@1ae7211`（实现 `d8b8cd7`）；先 cherry-pick 了 `codex/failstop` 的纯文档提交 `42f19d3`（00_总计划 的 `--shared-gpu-ok` 登记与 inflight 快验命令，无冲突）。实现（A）`c3d0dfba74ddde136d05c280c811c5b4eeb1e021`（含 §9 的审阅修改与 §10 的初始加载修正；`c4b203e`/`dee4f3b`/`f90c3e8` 与首版 `6f2fc84`/`c71d737`/`c4887e3` 的交付作废）；`releases.json` 的 `step6` 与 `failslow` 都指向 A。只有 fail-stop weavetp 新增 `--mode kill`；fail-stop 的 emulate/real、AnchorTP、restart 与全部 fail-slow 行为不变，峰值驱逐的修复除外（§5）。

## 1. 语义

- **降级（kill）**：4 个 victim（EDP0 的 etp∈{0,1}）被监管器真实 SIGKILL，流程与 real 相同：kill_sent/kill_exited、回收、新进程替换。
  - 存活进程（EDP1 与 4 张存活卡）不 abort world，也不建组。它们沿用启动时预建的 phase-1 组，rank 编号同 emulate，即 rank = slot。
  - 存活卡对开服前建好并释放了存储的 TP2 模型做 reallocate，按开服前算好的缩容计划只迁移权重。
  - 每一波按以下顺序执行：launch → 12 个存活 rank 过 Gloo 栅栏 → EDP1 decode，每波最多 `max_overlap_steps` 步 → wait。每一波都 decode，整次切换不设上限：KV 不迁移，online 那种 open-block 预算在这里没有意义。存活卡只过栅栏，因为它们的 TP4 组里有死卡。
  - EDP0 的在途请求全部 requeue，从 prompt 加已提交 token 重新 prefill；KV 不迁移。
- **降级完成后**（regroup('shrink') 发出 phase-1 报告并 `session.complete()` 之后，即控制门的 shrink_done 之后）：每个存活进程立即对自己所在、成员含 victim 的每个旧组调用 `c10d._abort_process_group(group)`。这些组取自 c10d 注册表，包括预建 phase-0 的 target/data/control 组，以及 `initialize_megatron` 建的模型并行组；默认 world 除外。随后丢掉初始 world 的模型并行状态（`runtime.world_destroyed()`，real 在故障时也这样做）。这一步不建任何组。目的是让 victim GPU 的显存在 spawn 前能回收，独占模式下 `supervisor.py` 的回收等待不超时。各进程的耗时随 phase-2 报告上报，最大值记入 `events_actual.delays.dead_group_abort_s`。
- **恢复（方案 A）**：与 real 相同。先 abort 旧 world（Gloo 默认 world 及其余子组），再以新 store 前缀 `init_process_group('nccl', rank=slot, world_size=16)` 重建 16 rank world，然后在线扩容：KV 前缀 + 增量，expand/delta 在切换时由 planner 计算。只有这里建组；EDP1 的重建停顿照常记录。
- **范围**：`serving_config.require_method` 只接受 `--method weavetp --scenario failstop --mode kill`。kill 只跑 1 轮，Step 1–5 的入口都不开放 kill。
- **GPU 证据（用户提供，本仓库没有原始记录）**：static_group 探针表明，组内成员被 SIGKILL 后，存活组仍能通信；只 abort 含死卡的旧组，耗时 0.34–0.52 s。

## 2. 与原计划前提不符之处（实现前报告，已按批注处理）

| # | 计划原文 | 代码实际情况 | 处理 |
|---|---|---|---|
| P1 | 阶段 2 用 real 的 rank map | kill 的 phase-1 组按 slot 编号（slot 10 是 TP2 坐标 0），real 的 rejoin 把 survivor 排在前面并紧凑编号（`async_topology.py`、`failstop.py:rank_map`），恢复时 slot 11 变成坐标 0，`adopt` 的坐标检查必然失败 | kill 全程沿用 emulate 编号（rank = slot），阶段 2 仍走 real 的 abort 加 `init_process_group`；`rank_map`、`AsyncSession.rejoin` 不改 |
| P2 | FaultBackend 增加 kill | 原来只接受 emulate/real | 接受 kill；新增 `rebuilds(phase)`：real 恒为真，kill 只有 phase 2 为真 |
| P3 | 在 run_worker 里预计算降级计划 | `prepare_online` 有三处不适用：读 `config['failslow']['level']`；victim 也作为源；计划在 phase-0 的 16 rank 会话上算。`warm_pairs` 在 phase-0 的 data 组上，而 copy service 要求组成员等于当前 world | initialize 保留降级态的 stage 会话，计划和 P2P 预热都在它及 `prebuilt[1]['data']` 上做；victim 不参与，所以源天然不含 victim。缺源的 target 由 planner 报 `uncovered target` |
| P4 | `Worker.transition` | 只属于 Step 5，Step 6 用的是 `ServingWorker.world_transition` | 只改 `world_transition`。`Worker.transition` 保持不变：Step 5 不接 kill，它的 `self.real`（现在含 kill）对 emulate/real 的行为不变。计划原写“改用同一判断”，这一处实际未改 |
| P5 | 给 sender_transfer 加 decode_fn | 参数已存在 | 只改调用处 |
| P6 | 在 supervisor 里改选择 | 非 emulate 已经用 real 的 `Supervisor` | 不改 |
| P7 | 入口 | `run_serving` 的 `cases('kill')` 返回空会抛 IndexError；quick_check 的正则只认 emulate/real；`require_method` 原本会放行 anchortp/nofault 的 kill | `run_topology.cases` 让 kill 走单轮分支；入口的 choices 加 kill；只允许 weavetp fail-stop |
| P8 | 从 initialize 到降级完成，0 次建组 | initialize 本身就要建 Gloo world 和预建组（同 emulate） | 计数窗口改为开服后到 rejoin 的 transition |
| P9 | 00_总计划 | 第 166 行附近写着“预建只用于 emulate” | 冲突表加 kill，见 00 §1/§9 |
| P10 | EDP1 全程 decode | EDP1 在 `world_transition(1)` 里等 transition/build 时不 decode（同 emulate），decode 发生在等计划键和各波之间 | 不改 |
| P11 | 峰值驱逐残留 KV | 确实会残留，见 §5 | 单测复现后修复 |

## 3. 实现（修改原函数，不加开关）

| 位置 | kill 怎么走 | 原逻辑保留给谁 |
|---|---|---|
| `failstop.FaultBackend` | 接受 kill；`rebuilds(phase)`；`process_world_size` 只有 real 用成员数 | — |
| `model_worker.Worker.initialize` | 只在 `initial_epoch == 0 and not rebuilds(0)` 时建 Gloo-16 world 和 phase-0/1 预建组；做 `adopt_group_order` 并保存 `degraded_rows`；`self.degraded = stage`；不建 phase 2；healthy 的 phase-1 target 共享 phase-0 的；栅栏为 `sorted(stage.current_world()['ranks'])`（kill 12 人，online 仍是 16 人）；`warm_groups()` | emulate、online fail-slow 的 phase-2 预建；kill 的替换进程不进这里，同 real |
| `Worker.build`/`adopt`、`ServingWorker.world_transition` | 以 `rebuilds(phase)` 判断是否建 NCCL world、abort、记 EDP1 停顿 | real 恒为重建；emulate 恒用预建 |
| `Worker.transfer` | `label in self.plans`（kill 只缓存了 shrink）时用缓存计划并核对 shard | planner 保留给 emulate/real 的全部切换，以及 kill 的 expand/delta |
| `plan_before_serving`/`warm_pairs` | 新增 session/group 参数，kill 传降级会话和 `prebuilt[1]['data']` | 缺省就是 online 原有行为 |
| `ServingWorker.prepare_online` | kill 分支：存活卡建 TP2 后释放；在降级世界、用原始 profile 只计划 shrink；按计划用到的 P2P 对预热；不做 expand 计划，不预热 KV 对 | online fail-slow 的两向计划与 KV 预热 |
| `shrink_transfer(meta=None)` | kill 对预建 TP2 做 reallocate，替代 `runtime.build(2)` | `runtime.build(2)` 保留给 emulate/real 的 weavetp 与 anchortp |
| `ServingSwitcher.base_wave('shrink')` | victim 走 `w.real` 分支，等着被杀。存活方：`world_transition(1)`（不 abort）→ `require_degraded_map()`（与 online cutover 共用）→ 存活卡 `shrink_transfer(None, overlap(None…))`，EDP1 `sender_transfer('shrink', decode_fn, overlap(decode_fn…))`，每波 decode，总步数不设上限；`overlap_steps` 记实际步数 | emulate/real：不带 overlap，`overlap_steps` 为 0 |
| `ServingSwitcher.ack`（`compact` 开头调用；`RestartSwitcher.compact` 同） | `switch_ack` 在峰值驱逐之后才发，控制门收齐后才 kill | 所有模式同一路径（原来在 `__init__` 里发，见 §9） |
| `ServingSwitcher.regroup('shrink')` | 发 phase-1 报告并 `session.complete()` 后调用 `abort_dead_groups()`（耗时随 phase-2 报告上报） | — |
| `serving_probe` | kill_sent/kill_exited、rejoin_request/reclaimed、空接收端、PID 已变、EDP1 停顿：判断从 `mode == 'real'` 改为 `mode != 'emulate'`；新增 `delays.dead_group_abort_s`（取 phase-2 报告） | — |
| 入口 | `run_failstop_e2e.py`/`.sh`、`run_serving.py`、`quick_driver.py`、`quick_check.sh` 接受 `--mode kill` | — |

不变的部分：`FailstopHooks`（kill 走 replay=True）、peak=evict、`online_switch`（仍只是 fail-slow weavetp）、恢复的 `expand_prefix`/`join_expand`/`kv_delta`/`check_kv`、监管器。

## 4. 结果字段（恢复重建的停顿在哪里看）

- `events_actual.delays`：
  - `edp1_pause_s`：EDP1 在阶段 2 的 abort 到采纳重建组之间的停顿；
  - `abort_s["2"]`、`rebuild_s["2"]`：控制门看到的 abort 与重建；
  - `rejoin_join_s`：从 rejoin 请求到 world16 就绪；
  - `gpu_reclaim_s`：每卡从 kill 到回收确认；
  - `reclaim_wait_s`；
  - `dead_group_abort_s`：新字段，kill 专有，见 §1。
- `result.json` 的 `rejoin_pauses`、`metrics.edp1_rejoin_pause_s`，以及 `tfs_breakdown.expand_wall_s`（含 worker 的 NCCL 重建）。
- kill 的 `abort_s["1"]`/`rebuild_s["1"]`/`tfs_breakdown.world_rebuild_s` 只是总线往返，不含 abort 或重建；`null_reason` 为 null，因为有回收记录。

## 5. 峰值驱逐：已复现并修复

- **现象**：peak=evict 把最新准入的请求放回队首（`controller.evict_for_peak` 只改账本），worker 上它们的 KV 留在原副本。之后在同一副本上再次准入时，prefill 报 `seq N already resident`（`block_kv.BlockKVContext.admit`）。
- **会触发的路径**：每个 fail-stop 模式里 EDP1 的切换。EDP1 是发送方，context 不换。kill 下 EDP0 不会触发，因为 requeue 后换了新的 TP2 context。
- **单测**：`tests/unit_tests/e2e/test_block_backend.py::test_a_request_evicted_at_a_switch_peak_prefills_again_on_the_replica_it_left`。用真实 `Driver.switch` 和 `BlockBackend`，切换时保留 KV，预算 1 字节，两条请求被驱逐后回到原副本。修复前报 `RuntimeError: seq 0 already resident`。
- **修复**：`plan_executor.perform_switch` 驱逐后立即调用 `release(seqs)`，由 `Driver.switch` 传入，对本进程服务的副本调用 `backend.release`；在压缩之前执行。
- **未改的已知问题**：跨域驱逐的 KeyError（`remote_control.py` 的镜像删除，见 10 §6）仍在，本次不扩大范围。

## 6. 测试

- **新增 `fs/test_failstop_kill_protocol.py`**：16 actor 的 kill 协议模拟，correctness 与 timing 各一次。模拟的监管器杀掉 victim 之后，任何仍需要 victim 的集合通信都会撞上 8 s 上限而失败。
  - a) 从开服到 rejoin 的 transition 之间：没有 `new_group`、没有 `init_process_group`、没有计划、没有建模型；abort 只针对含死卡的组，覆盖每个存活进程所有含死卡的组，不碰默认 world。恢复时，12 个存活进程各 abort 一次 world、nccl init 一次；4 个替换进程各 init 一次。
  - b) 缩容各波只用 12 人栅栏。
  - c) EDP1 在每个缩容波都 decode（`overlap_steps` = 波数）。测试把 shrink 的 growth 设为整个 open-block 区，原来的上限会让 EDP1 一步也不 decode。EDP0 有 requeue，`replay_prefill_tokens > 0`，12 条请求全部完成，KV 字节为 0。
  - d) 只有 shrink 在开服前计划，用的是原始 profile，源不含 victim；planner 调用开服前 12 次，rejoin 时 2 × 16 次。
  - e) EDP1 的阶段 2 停顿、`dead_group_abort_s` 有值；correctness 下扩容 KV 逐位检查与参考 hash 都通过；控制门的 PID 与空接收端检查通过；分析器接受结果（`rejected` 为空）。
  - 另外：victim 在 shrink ack 之后不参与任何集合通信；每个存活进程的死组 abort 都在它的 phase-1 报告之后，并且只 drop 一次模型并行状态。
- **模拟支架**：假 c10d 按 actor 维护进程组注册表（含默认 world）；假 `_abort_process_group(group=None)` 改为 `Simulation.abort`；假监管器对非 emulate 模式生成替换进程。
- **改动的已有测试**：
  - `test_server_launch`：组合数 28 → 30，替换进程 16 → 24；
  - `test_expand_with_requests_in_flight`：toy 请求 60 → 120（只改测试，加大扩容时的在途余量）；
  - 峰值驱逐单测。
- **回归**（09 §7 全量，AC 供电，无其他忙循环进程）：基线 `1ae7211` 为 303 通过、1 跳过、119 个子测试；首版为 306 通过、1 跳过、123 个子测试；审阅后为 306 通过、1 跳过、123 个子测试。

## 7. 只能上机验证的风险与快验判据

- **R1：abort Gloo 默认 world。** 阶段 2 要 abort 一个挂着 NCCL 子组的 Gloo 默认 world，现有模式没这样用过。torch 文档把 `_abort_process_group` 标为仅限 NCCL 的实验接口；默认后端是 Gloo 时，它不把 NCCL abort 包进 coalescing。本机 torch 2.14 已验证单 rank 的 Gloo world 可以 abort 后重建；服务器 torch 2.11 + NCCL 2.28 未验证。快验判据：控制门 passed（phase 2 的 validate 通过，说明重建成功），`delays.abort_s["2"]`、`rebuild_s["2"]` 有值。
- **R2：回收。** 已由 §1 的死组 abort 处理。快验判据：`delays.gpu_reclaim_s` 每卡有值，没有 `GPU memory/process reclamation did not become stable` 超时；`delays.dead_group_abort_s` 有值。
- **R3：模型并行状态。** 被 abort 的模型并行组和初始 world 的模型并行状态在降级后一起丢掉（`world_destroyed`），所以 kill 降级期间的状态和 real 一样：real 在 phase 1 就 destroy 了模型并行状态，照常服务。推理前向从绑定的 pg_collection 取组，Megatron 的全局取组只出现在训练 recompute/fp8 路径（`moe_layer.py`）。
- **EDP1 decode。** 判据：SL3060 的 `result.json` 中 `switches` 里 domain EDP1 shrink 的 `overlap_steps > 0`。

## 8. step6 改钉

`releases.json["step6"]` 从 `073a50f` 改指本实现 A。`073a50f` 带着两个回放历史问题（`BlockBackend.history_for` 优先用本 rank 的旧副本；跨副本重放时 prompt 拼了两次），二者已在 `d8b8cd7` 修复（见 10 §6）。**此前用 `073a50f` 跑出的 step6 结果作废。**

## 9. 审阅修改（2026-10-26）

1. **`switch_ack` 移到峰值驱逐之后（正确性，real 同样受影响）。**
   - 原来 ack 在 `ServingSwitcher.__init__` 里发，之后 `perform_switch` 才做峰值驱逐。`RemoteControl.evict_for_peak` 无论有无驱逐，都要在整个 EDP0 域的 Gloo 组（含 victim）上广播一次。
   - 控制门收齐 16 个 ack 就 kill。real/kill 下若 victim 在加入这次广播之前被杀，存活卡会卡在广播里，直到 stage 超时。
   - CPU 模拟只在 bus 等待处杀进程，所以原来的测试看不到这个问题。
   - 修复：ack 改由 `compact()` 的第一步发出（`ServingSwitcher.ack`；`RestartSwitcher.compact` 也调用），即 peak 之后的第一个 switcher 调用。所有模式都是这一条路径。
   - 测试：victim 在 ack 之后不参与任何集合通信。修复前这条断言失败（victim 8 在 ack 之后参与了 EDP0 的 `bcast`）。
2. **kill 的 EDP1 每波 decode，不设总上限。** 原来沿用 online 的预算 `max_delta_steps − growth`（正式配置为 32），超过 32 波以后 EDP1 在剩下的波里不再 decode。那个预算是 online 缩容中 open-block 增量区的上限，kill 不迁移 KV，用不上。测试把 shrink 的 growth 设成整个预算，修复前 EDP1 的 `overlap_steps` 是 0，修复后等于波数。
3. **死组 abort 移到降级完成之后，并 drop 模型并行状态。**
   - 原来在 phase-1 报告之前 abort，abort 耗时被算进控制门的 shrink_done 和 `shrink_wall_s`。现在放在报告和 `session.complete()` 之后，耗时随 phase-2 报告上报。
   - 同时调用 `runtime.world_destroyed()`，不让模型并行全局状态指着已 abort 的组；降级期间的状态与 real 一致（R3）。
4. **精简。**
   - `plan_before_serving`、`warm_pairs` 的 session/group 改为必填（调用方都显式传入），去掉 None 缺省分支。
   - 预建组的 healthy target 共享改为遍历全部 phase。
   - prepare 的 kill 分支合成一个推导式。
   - base_wave 的 online 与 kill 分支分开写。
5. **模拟支架。**
   - 假监管器等 `finish` 的上限改为 `launch_timeout_s`，与真实监管器相同（`supervisor.py`）。原来用的是单个协议步的 8 s 上限，而 toy 请求增到 120 条后，扩容后的排空可能超过 8 s：审阅后的第一次全量回归里 emulate 子测试因此失败过一次。
   - 改后在 8–16 个忙循环负载下跑 18 次，未再出现这个失败。另有 1 次（16 个忙循环下 8 次中的 1 次）报 "drained before the expand"，这是基线已有的在途余量问题（10 §6），120 条只是让它更少见；不加负载时全量回归通过。

## 10. 2026-10-27：服务器首次走到初始加载，分片完整性校验失败（实现 `c3d0dfba74ddde136d05c280c811c5b4eeb1e021`）

- **现象。** fail-slow Step 9 的 weavetp、ignore、restart 三项快验（`s9q_20261027T064713_eb030784`、`s9q_20261027T065038_e60e4050`、`s9q_20261027T070202_39d8314a`）都在开服前失败。slot 0 的调用链为 `serving_worker.run_worker` → `MegatronRuntime.build(4, load=True)` → `load_checkpoint` → `dist_checkpointing.load` → `validate_sharding_integrity`。报错有两类：专家权重 `decoder.layers.*.mlp.experts.experts.linear_fc1/fc2.weight` 的 `Invalid access pattern`（访问计数为 2），以及专家 `linear_fc2._extra_state` 的 64 个 ShardedObject 重复（`-64 ShardedObject are missing`）。其余报错（peer failure、TCPStore recv 0 bytes、超时）是连锁反应。10-21 的运行在启动阶段就被参数解析和导入问题挡住，从未走到加载；Step 5 的快验（`run_model.py`）用全局模型并行组建模型，所以也没有暴露。
- **根因（已核实）。** `load_checkpoint` 在 16 rank 的全局 world 上校验分片的访问模式。两个副本中，稠密张量 `replica_id` 的 DP 部分取自全局模型并行的 DP 组（`metadata['dp_cp_group']`），这个组跨两个副本，所以不重复。专家张量则在 `SequentialMLP.sharded_state_dict`（`megatron/core/transformer/moe/experts.py`）里改用模型自身 pg_collection 的 `expt_dp` 组，这个组在一个副本内只有 1 个成员，于是两个副本的专家分片都是 `replica_id=(0,0,0)`，每个分片被计为读了两次（`_extra_state` 同理）。用户提出的根因方向正确，但只限于专家张量：稠密张量本来就带了全局 DP 编号。
- **修复。** 服务 worker 的初始 TP4 加载改用 `MegatronRuntime.load_local`（实现 `c3d0dfba74ddde136d05c280c811c5b4eeb1e021`；fail-slow restart 的重新加载原本就用它）。它的做法：只读本 rank 的分片，偏移取自模型自己的进程组；`dist_checkpointing.load` 关闭全局校验（`validate_access_integrity=False`、`strict='assume_ok_unexpected'`），所以没有 `determine_global_metadata` 这次 all-gather；torch_dist 用 `no_dist=True` 读取；tracker 在本地读，没有 all-reduce，也没有 world barrier。common state（args、iteration、rng）推理用不上，`load_local` 只取 `state['model']`；content metadata 与 `load_checkpoint` 一样从检查点读取。 correctness 模式下已有的 `validate_weights(model, 4)` 把每个参数与参考 load 逐位比对，上机时证明 `load_local` 读出的字节与原来的 `load_checkpoint` 相同（fail-stop weavetp 的 Step 6 kill 快验覆盖；fail-slow 不做参考比对）。

**全仓库的检查点加载点：**

| 位置 | world 与模型 | 是否受影响 |
|---|---|---|
| `serving_worker.run_worker` 的初始 TP4 加载（Step 6、Step 9） | 16 rank、两个副本，模型用各自的 pg_collection | 受影响，已改为 `load_local` |
| fail-slow restart 的重新加载（`RestartSwitcher.load`，`load='local'`） | 同上 | 本来就是 `load_local` |
| 替换进程与恢复（`join_expand`、`expand_prefix`） | 空接收端，不读检查点 | 不涉及 |
| `model_worker.Worker.run_worker:633`（`run_failstop_model.py launch`，两节点 16 rank 的 Step 5 fail-stop 协议） | 16 rank、两个副本、pg_collection，走同一个 `build(4, load=True)` | **同样受影响**。按要求保持不变（计划里写的是"单节点"，实为两节点 launch）；若要上机运行，需同样改为 `load_local` |
| `model_reference.py:164`（`run_model.py`：Step 5 快验与参考 hash） | 用全局模型并行组建模型（`get_training_model`） | 不受影响：专家的 DP 组也是全局的，两个副本的 `replica_id` 不同 |
| `serving_layers.py:301`（Step 6 b1/b2）、`correctness/router_determinism.py:160`、`examples/rl/benchmark_live_moe_tp.py:1446` | 同上，单节点 | 不受影响 |
| `e2e/factories.py:65`（`failstop_factories`） | 没有调用方 | 不可达 |

**测试。**
- 新增 `test_load_local.py`：在 Megatron 真实的 `dist_checkpointing` 上，先用 1 rank 的 Gloo world 存一个 torch_dist 检查点，再用 `load_local` 读 TP2 坐标 1 的行。检查三点：
  - 期间 `torch.distributed` 的全部集合通信计数为 0（存盘时的 gather/broadcast/barrier 也被计到，说明计数有效）；
  - 不调用 `load_checkpoint`；
  - 读出的行正确。
  同一次读取只要打开全局校验，就报同类的 `Invalid access pattern`。
- 模拟中 16 个初始 TP4 加载都走 `load='local'`（kill 与 fail-slow 协议测试）。
- 全量回归：307 通过、1 跳过、123 个子测试（对照 `c4b203e`：306 通过、1 跳过、123 个子测试；多出的 1 项是新单测）。
- CPU 测试不加载真实检查点，测不到 Megatron 模型的分片问题。

**只能在服务器上验证的步骤（快验的检查顺序）：**
1. 初始 TP4 加载（`load_local`）；
2. TP2 预分配（开服前建好并释放，计划缩容）；
3. 组预热（`warm_groups`、`warm_pairs`）；
4. kill 回收（SIGKILL、死组 abort、GPU 回收）；
5. phase-2 重建（在 Gloo 默认 world 上 abort 后重建 16 rank NCCL world）。

每一步看哪里，见 [Step6_failstop_kill_服务器命令.md](Step6_failstop_kill_服务器命令.md) ②；快验依次为 Step 9 的 weavetp、ignore、restart，然后是 Step 6 kill。
