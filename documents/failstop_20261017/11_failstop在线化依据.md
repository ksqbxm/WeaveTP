# 11. fail-stop WeaveTP `--mode kill`：降级在预建组上在线迁移，恢复沿用 real 重建

状态（2026-10-26）：**已实现，CPU 已验证，服务器 GPU 未运行。** 分支 `codex/failstop-online`，基于 `codex/failslow-online@1ae7211`（实现 `d8b8cd7`）；先 cherry-pick 了 `codex/failstop` 的纯文档提交 `42f19d3`（00_总计划 的 `--shared-gpu-ok` 登记与 inflight 快验命令，无冲突）。实现（A）`6f2fc849328c368cd45f9254575129fc32eb2731`；`releases.json["step6"]` 改指 A，`failslow` 仍为 `d8b8cd7`。只有 fail-stop weavetp 新增 `--mode kill`；fail-stop 的 emulate/real、AnchorTP、restart 与全部 fail-slow 行为不变，峰值驱逐的修复除外（§5）。

## 1. 语义

- **降级（kill）**：4 个 victim（EDP0 的 etp∈{0,1}）被监管器真实 SIGKILL，流程与 real 相同：kill_sent/kill_exited、回收、新进程替换。
  - 存活进程（EDP1 与 4 张存活卡）不 abort world，也不建组。它们沿用启动时预建的 phase-1 组，rank 编号同 emulate，即 rank = slot。
  - 存活卡对开服前建好并释放了存储的 TP2 模型做 reallocate，按开服前算好的缩容计划只迁移权重。
  - 每一波按以下顺序执行：launch → 12 个存活 rank 过 Gloo 栅栏 → EDP1 decode，每波最多 `max_overlap_steps` 步 → wait。存活卡只过栅栏，因为它们的 TP4 组里有死卡。
  - EDP0 的在途请求全部 requeue，从 prompt 加已提交 token 重新 prefill；KV 不迁移。
- **降级完成后**（regroup('shrink') 发 phase-1 报告之前）：每个存活进程立即对自己所在、成员含 victim 的每个旧组调用 `c10d._abort_process_group(group)`。这些组取自 c10d 注册表，包括预建 phase-0 的 target/data/control 组，以及 `initialize_megatron` 建的模型并行组；默认 world 除外。这一步不建任何组。目的是让 victim GPU 的显存在 spawn 前能回收，独占模式下 `supervisor.py` 的回收等待不超时。各进程耗时的最大值记入 `events_actual.delays.dead_group_abort_s`。
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
| `ServingSwitcher.base_wave('shrink')` | victim 走 `w.real` 分支，等着被杀。存活方：`world_transition(1)`（不 abort）→ `require_degraded_map()`（与 online cutover 共用）→ 存活卡 `shrink_transfer(None, overlap(None…))`，EDP1 `sender_transfer('shrink', decode_fn, overlap(decode_fn…))`；`overlap_steps` 记实际步数 | emulate/real：不带 overlap，`overlap_steps` 为 0 |
| `ServingSwitcher.regroup('shrink')` | `abort_dead_groups()` 后再发 phase-1 报告（报告带 `dead_group_abort_s`） | — |
| `serving_probe` | kill_sent/kill_exited、rejoin_request/reclaimed、空接收端、PID 已变、EDP1 停顿：判断从 `mode == 'real'` 改为 `mode != 'emulate'`；新增 `delays.dead_group_abort_s` | — |
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
  - c) EDP1 的缩容 `overlap_steps > 0`；EDP0 有 requeue，`replay_prefill_tokens > 0`，12 条请求全部完成，KV 字节为 0。
  - d) 只有 shrink 在开服前计划，用的是原始 profile，源不含 victim；planner 调用开服前 12 次，rejoin 时 2 × 16 次。
  - e) EDP1 的阶段 2 停顿、`dead_group_abort_s` 有值；correctness 下扩容 KV 逐位检查与参考 hash 都通过；控制门的 PID 与空接收端检查通过。
- **模拟支架**：假 c10d 按 actor 维护进程组注册表（含默认 world）；假 `_abort_process_group(group=None)` 改为 `Simulation.abort`；假监管器对非 emulate 模式生成替换进程。
- **改动的已有测试**：
  - `test_server_launch`：组合数 28 → 30，替换进程 16 → 24；
  - `test_expand_with_requests_in_flight`：toy 请求 60 → 120（只改测试，加大扩容时的在途余量）；
  - 峰值驱逐单测。
- **回归**（09 §7 全量，AC 供电，无其他忙循环进程）：基线 `1ae7211` 为 303 通过、1 跳过、119 个子测试；本实现为 306 通过、1 跳过、123 个子测试。

## 7. 只能上机验证的风险与快验判据

- **R1：abort Gloo 默认 world。** 阶段 2 要 abort 一个挂着 NCCL 子组的 Gloo 默认 world，现有模式没这样用过。torch 文档把 `_abort_process_group` 标为仅限 NCCL 的实验接口；默认后端是 Gloo 时，它不把 NCCL abort 包进 coalescing。本机 torch 2.14 已验证单 rank 的 Gloo world 可以 abort 后重建；服务器 torch 2.11 + NCCL 2.28 未验证。快验判据：控制门 passed（phase 2 的 validate 通过，说明重建成功），`delays.abort_s["2"]`、`rebuild_s["2"]` 有值。
- **R2：回收。** 已由 §1 的死组 abort 处理。快验判据：`delays.gpu_reclaim_s` 每卡有值，没有 `GPU memory/process reclamation did not become stable` 超时；`delays.dead_group_abort_s` 有值。
- **R3：模型并行组的元数据查询。** 被 abort 的模型并行组从 c10d 注册表里删掉了。推理前向从绑定的 pg_collection 取组，Megatron 的全局取组只出现在训练 recompute/fp8 路径（`moe_layer.py`）。若仍有元数据查询，会立刻报 `Group … is not registered`，不会静默出错。
- **EDP1 decode。** 判据：SL3060 的 `result.json` 中 `switches` 里 domain EDP1 shrink 的 `overlap_steps > 0`。

## 8. step6 改钉

`releases.json["step6"]` 从 `073a50f` 改指本实现 A。`073a50f` 带着两个回放历史问题（`BlockBackend.history_for` 优先用本 rank 的旧副本；跨副本重放时 prompt 拼了两次），二者已在 `d8b8cd7` 修复（见 10 §6）。**此前用 `073a50f` 跑出的 step6 结果作废。**
