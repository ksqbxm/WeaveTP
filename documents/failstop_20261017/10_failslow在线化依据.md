# fail-slow weavetp 在线切换：依据、探针与删改清单

状态（2026-10-26）：**已实现，CPU 已验证，服务器 GPU 未运行。** 分支 `codex/failslow-online`（自 `codex/failslow@848350f`，实现 073a50f）。只改 fail-slow 场景下的 weavetp（`common.online_switch`）；fail-stop 全部、AnchorTP、restart、ignore、nofault 行为不变（重放 history 的修复除外，见 §5），`releases.json["step6"]` 不动。交付登记见 §6，审阅后的修改见 §7。

## 1. 依据

- **参照实现是 live benchmark，不是 Step 8。** `codex/e2e-scaling` 的扩缩容（Step 8）是同步波次：切换时 `runtime.build`、切换时规划、无预热、无波内重叠（`scaling_worker.py:13-14,295,307-312`）。用户计划要的机制都在 `examples/rl/benchmark_live_moe_tp.py`：
  - 两个方向的计划预先算好（:1601、:1622）；
  - 两套模型用 StandbyWeights 保存，切换时 `allocate_and_poison` 再分配（:1800-1803、:2051）；
  - launch 后过一道 Gloo 栅栏（:1144，"All ranks must enqueue migration P2P before any rank enters active model collectives on another communicator"）；
  - 波内 decode，最多 `live_max_overlap_steps` 步（:1146-1171）。
- **探针。**
  - slow_sender（256 MiB P2P）：慢卡收发带宽相对空载为 W 0.48、M 0.32、S 0.24（1/2/3 个陪跑进程）。缩容计划的降权系数取 1/(1+k)，即 0.5/0.33/0.25，与实测吻合。
  - static_group 探针为 fail-stop 证据，留到 fail-stop 阶段。

## 2. 在线切换

| 阶段 | 做什么 | 位置 |
|---|---|---|
| 启动（initialize） | 按全局全序（成员、phase、位置）对每个预建 NCCL 组做 1 元素 all_reduce 预热；建一个 16 rank 的 Gloo 栅栏组；EDP0 的 phase-2 target/data 组复用 phase-0 的（成员相同）；降级 rank map 用 phase-1 组的实际顺序校正（`adopt_group_order`）后保存 | `model_worker.Worker.initialize`、`warm_groups` |
| 启动（初始加载之后、开始服务之前） | survivor 建 TP2 模型并释放存储（StandbyWeights 留在模型上）；各 rank 发布分片，计算两份权重计划并缓存：缩容用把故障 GPU 的 p2p 进出条目乘 1/(1+k) 的画像，扩回用原始画像（扩回时陪跑已停，victim 已替换）；对两份计划与 KV 持有者对的 (src, dst) 做一次 1 元素 P2P 交换 | `ServingWorker.prepare_online`、`Worker.plan_before_serving`、`warm_pairs`、`model_worker.downweighted` |
| 缩容 | 不重放（hooks `replay=False`）、不先 transition、peak=None。EDP0 全部 rank（含 victim）压实 KV，在 phase-0 世界里作源：本卡 TP4 权重 + KV A 区；survivor 的目标是再分配的 TP2 模型 + TP2 池。权重取缓存计划；KV 每个原子只有一个持有者，用固定配对生成任务（`fixed_tasks`），不调规划器。每波 launch → Gloo 栅栏 → 本域 decode `max_overlap_steps`（默认 1）步 → wait，EDP0（含 victim）与 EDP1 都在服务；整个切换的波内步数不超过 `max_delta_steps − growth`（delta 要搬的开放块区）。之后做 B 区 delta，survivor 用 `check_kv` 逐位校验（4 个 TP4 源坐标 → TP2）。cutover：全部成员发 `shrink_ready`，网关这时才放行 victim（释放权重和 KV、transition、等 spawn）；survivor 与 EDP1 `world_transition(1)` 采用预建组，断言 rank map 等于启动时保存的降级 map，survivor rebind 到 TP2 和迁移来的 KV | `ServingSwitcher.compact/base_wave/delta/compare_logits/cutover`、`shrink_transfer`、`kv_delta('shrink')`、`serving_probe.fault_cycle` |
| 扩回 | 照原扩容路径（先 transition 再迁移）；survivor 与 victim 的 TP4 目标是再分配的初始 TP4 模型，权重取缓存计划，波内重叠同上（victim 只过栅栏）；完成后释放 TP2 存储 | `expand_prefix`、`join_expand`、`release_source` |

`mixed_transport.execute` 加可选 `overlap(work)`，在每波 `launch()` 之后、`wait()` 之前调用；不传时结果与原来逐字节一致（单测比对 stats、目标张量和记录行）。

## 3. 删除的旧逻辑（在线路径上）与保留的调用方

| 旧逻辑 | 在线路径 | 仍走原代码的调用方 |
|---|---|---|
| 缩容时 `runtime.build(2)`、`admission(2, 0)`、只迁权重 | 删除，改用预建 TP2 模型和实迁 KV 量 | fail-stop weavetp/anchortp、fail-slow anchortp 的 `shrink_transfer(meta=None)` |
| 扩回时 `runtime.build(4)` | 删除，改用再分配的初始 TP4 模型 | fail-stop weavetp/anchortp、fail-slow anchortp 的 `expand_prefix` |
| 切换时的规划器调用 | 删除，改用缓存的权重计划 + KV 固定配对 | fail-stop 全部、fail-slow anchortp（`transfer` 的非在线分支） |
| 缩容先 `world_transition(1)` 再迁移 | 删除，transition 移到 cutover | fail-stop、fail-slow anchortp（`base_wave` 下半段） |
| 缩容时 requeue + set_down | 删除（`FailstopHooks(replay=False)`） | fail-stop、fail-slow anchortp/restart（`replay=True`，缺省） |
| victim 迁移前释放 | 删除，victim 在 cutover 才走 `emulate_victim` | fail-stop emulate、fail-slow anchortp/restart（`base_wave` 的 victim 分支） |
| 缩容时 EDP1 停服等待 | 删除，`sender_transfer('shrink', decode_fn, overlap)` | fail-stop、fail-slow anchortp（`sender_transfer('shrink')` 不传 decode_fn） |
| peak=evict | 在线传 None | 其余全部 |

共享函数只有一条路径，不留兼容缺省：
- `shrink_transfer(meta, overlap)`：`meta=None` 是停服缩容（目标现建、只迁权重，事件记录为 `kv_prefix_moves=0`），否则是在线缩容。
- KV 辅助函数由调用方显式传 (源, 目标) rank map（`shrink_maps()` / `expand_maps()`）：`kv_moves(meta, zones, maps, assignment)`、`kv_shards(moves, maps, 源池, 目标池)`。
- `kv_delta(direction, …)`（原 `expand_delta`）、`save_pool_snapshot(context, coord, direction)`、`check_kv(target_context, mode, direction)` 按方向取标签、map、phase 与坐标。
- KV 快照的 bus 键统一为 `kv_saved/0/{direction}/{replica}/{coord}`。`check_kv` 的 reference 文本统一为 "head mapping of the TP{源 TP} sources' saved block snapshots"，fail-stop 扩容记录里的这段文字随之变化。
- `transfer` 在线时取 `Worker.plans[label]`（KV-only 的 delta 没有权重计划），否则调规划器。
- `MegatronRuntime.release` 每次释放都新建 StandbyWeights（保留 protected 别名检查）并留在模型上，`reallocate` 用它再分配。

## 4. 前提核对（用户计划与代码不符之处）

| 计划前提 | 实际 | 处理 |
|---|---|---|
| "扩缩容同款在线切换" | Step 8 是同步波次（§1） | 以 live benchmark 为参照 |
| 第 1 项在 `Worker.initialize` 里建模型 | initialize 在 `runtime.start()` 之前，建不了模型 | 组预热与栅栏组放 initialize；建模、预算计划、P2P 预热放初始加载之后 |
| 第 2/15 项"重新分配存储即可" | `MegatronRuntime.release` 每次新建 StandbyWeights 后丢弃 | 留在模型上，新增 `reallocate` |
| 第 10 项"直接取缓存计划" | `resources.plans` 只写不读；KV 没有独立的任务构造 | 缓存放 `Worker.plans`；KV 用 `fixed_tasks` |
| 第 6 项行号 581–597 | 实为 581–612 | — |
| 第 12/13 项 peak=evict | `evict_for_peak` 可能选中另一域的请求，该域镜像 `KeyError`（`remote_control.py:139-140`） | 在线传 None 即避开；其余路径的潜在缺陷只汇报 |
| （实现中发现）降级 map 用 staged session 即可 | staged session 未经 `adopt_group_order`，survivor 的 TP2 坐标与预建组实际给出的相反（10/11、14/15 互换） | 启动时先对 staged session 采用 phase-1 组的实际顺序再保存；cutover 的断言会在不一致时报出差异 |

## 5. 已有问题（汇报，不在本次范围）

- **restart 循环的 history 竞态。** 电池供电降频时，`test_serving_protocol.test_restart_cycle_reloads_tp2_then_tp4_from_disk` 在未改动的 `848350f` 上也稳定失败：`seq 19: 7 history tokens for an input length of 8`。某个 rank 重新接纳回队请求时用了本地过期的 history。交流供电下通过。fail-stop 与 fail-slow 的 restart 都走这段代码。
  - **已修（实现 `d8b8cd7`：`a7c64d5` 加上 `codex/restart-history` 的 `0c7b250`，cherry-pick 无冲突；只进 `releases.json["failslow"]`，`step6` 仍为 `073a50f`，fail-stop 照旧）。** 根因：`BlockBackend.history_for` 优先用本 rank 自留的 history。请求从副本 R 回队（故障重放、restart 扩回时的回队、peak 驱逐），在另一副本上又提交了 token 后再回队、回到 R 时，R 的 rank（rejoin 后的 emulate victim 或 survivor）手里是上一段的旧副本，比 `input_len` 短。改为重放一律用回队时发布的 `history/<seq>/<len>`。同时发现 `ServingWorker.history_source` 把发布的整段 history（prompt + 已提交）当作"已提交 token"返回，跨副本重放时 prompt 拼了两次（上下文与 greedy 输出错，长度和计时不变），改为去掉 prompt。确定性单测 `test_serving_protocol.ReplayHistoryTests`（回队 → 另一副本 → 再回队 → 回到原副本，每段与不中断的 greedy 结果逐 token 相同）。本机一直是交流供电，用 12–32 个忙循环进程加压代替降频：修复前 restart 模拟 18 次里 2 次报此错（另 1 次是 8 s 等待上限）；修复后 29 次全部通过（`0c7b250` 加压 14 次、不加压 5 次，`d8b8cd7` 加压 10 次），其中 2 次确实走到了旧副本更短的路径。记录见 `failslow_online_cpu_evidence/history_restart_loaded.txt`。
- **`test_expand_with_requests_in_flight`** 的 emulate/real 子测试在电池供电下超出模拟器 8 s 等待上限（基线同样失败，交流供电下通过）。交流供电下 emulate 子测试还会偶发另一种失败："the workload drained before the expand"。60 条 toy 请求中，EDP0 扩容 cutover 时仍在运行的只有 1–4 条，降到 0 条测试就失败。基线 `848350f` 与本分支的分布相同（均值 2.4 / 2.2，最小值都是 1），所以这是测试余量太小，不是代码差异；交替重跑记录见 `failslow_online_cpu_evidence/review_inflight_margin.log`。
- **peak=evict 的跨域驱逐**（§4），只影响非在线路径。

## 6. 验证与交付登记

- CPU 回归（`code/current`，09 §7 的命令）：
  - 基线 `848350f`：交流供电 296 通过 + 1 跳过（仅 Linux），119 subtest；电池供电 295 通过、3 失败（§5 的 3 项）。
  - 基础项（#7/#11/#15）完成后：电池供电下与基线逐项相同。
  - 第一次交付 `9fb58aa`（电池供电）：301 通过、1 跳过、3 失败，失败项与基线电池供电相同。
  - 审阅修改后 `a7c64d5`（交流供电、主机空闲）：302 通过（296 + 新增 6 项）、1 跳过（仅 Linux），118 subtest 通过，1 个 subtest 失败。失败的是 `test_expand_with_requests_in_flight` 的 emulate 子测试（"drained before the expand"），这是 §5 的测试余量问题，与改动无关：单独重复运行时，`848350f` 24/24 通过，第一次交付 `434b104` 9/10，`a7c64d5` 20/24；但三者"cutover 时仍在运行的请求数"分布相同（§5）。
  - 另一次全量在主机被另一会话的 16 个忙循环占满时运行，同一子测试触发 8 s 等待上限。
  - history 修复后 `d8b8cd7`（交流供电、主机空闲）：303 通过（302 + `ReplayHistoryTests`）、1 跳过（仅 Linux），119 subtest；紧接着重跑的 `a7c64d5` 为 302 通过、1 跳过、119 subtest（这两次 in-flight 子测试都没有碰上余量问题）。
  - 日志在 [failslow_online_cpu_evidence/](failslow_online_cpu_evidence/)：`baseline_ac.log`（基线、交流）、`baseline_battery.log`（基线、电池）、`base_items_battery.log`（#7/#11/#15 之后）、`online_battery.log`（第一次交付）、`review_ac.log`（审阅后、交流空闲）、`review_ac_loaded.log`（审阅后、主机满载）、`review_inflight_margin.log`（上述重复运行与余量测量）、`history_ac.log` / `history_review_base_ac.log`（history 修复后 / 同时段的 `a7c64d5`）、`history_restart_loaded.txt`（§5 的加压 restart 模拟）。
- 新增/修改测试：
  - `test_failslow_protocol.test_weavetp_switches_online`：缩容 KV 前缀和 delta 字节 > 0、replayed_tokens = 0、victim 在释放前参与 shrink 与 shrink_delta、survivor 在 transfer 之后才 transition、缩容与扩回的 KV 逐位校验通过、启动后 `runtime.build` 与规划器调用均为 0（只在启动时 16 次 TP4 加载、4 次 TP2 构建、32 次规划）、启动后无 `new_group`、每个 rank 的释放/再分配顺序（稳态只有一套布局驻留）、缩容计划用降权画像而扩回用原始画像。
  - `test_failslow.py`：`execute` 带钩子时钩子在 launch 与 wait 之间、不带钩子结果逐字节一致；栅栏在 decode 之前且步数受预算限制；降权只改故障 device 的进出条目；KV 固定配对；`--method` 单次 launch 与预算；摘要字段（缺失项标"缺失"）。
- 交付（history 修复后，取代审阅后的 A `a7c64d5` / B `9488be4` / C `6748e02`，以及第一次交付 A `9fb58aa` / B `915d08d` / C `434b104`）：
  - 实现（A）`d8b8cd7ef1d59926e3e294a2e6d905e115cdfdfb`（`a7c64d5` + §5 的 history 修复）；
  - 清单（B）为本文件所在提交，`releases.json["failslow"]` 指向 A，`step6` 仍为 `073a50f`；
  - 服务器命令在 B 的子提交（C）里，随 bundle 交付：[Step9_failslow_服务器命令.md](Step9_failslow_服务器命令.md)。
- 行数（相对 `848350f`，不含 releases.json）：实现 +409/−112 行（不计空行、注释和 docstring：314 行）；测试 +216/−14 行（174 行）。审阅修改本身相对第一次交付：实现 +131/−150 行，测试 +1/−1 行。history 修复相对 `a7c64d5`：实现 +8/−10 行，测试 +30 行。

## 7. 审阅修改（2026-10-26，实现 `a7c64d5`）

正确性：
- 单方法快验（`--method`）的结论行仍写着"fail-stop restart: passed"，但这次并没有跑这个 launch。现在只有 4 次组合时才写这一句。
- 不带 `--method` 时，`summary.json` 的 `methods` 从 3 个 fail-slow 方法变成了 4 个 launch 名。已恢复原值。
- `STEP 9 SUMMARY` 在 SL3061 和 fail-stop launch 上各打出一行"缺失"。现在只为有网关记录（SL3060 的 fail-slow launch）的 launch 打印。
- `MegatronRuntime.release` 复用模型上旧的 StandbyWeights，跳过了本次 protected 的别名检查。现在每次释放都新建一个。

简洁性（删掉的兼容与兜底）：
- `transfer(cached=…)`、`online_plan` 删除。成员一致检查也删除：权重分片逐个相等已经蕴含它。
- 构造 hooks 时的降级 map lambda 删除：serving 路径不读 `target_rank_map`。
- `replicas_of` 删除。
- `warm_groups` 改为显式列出成员组，不再用 `__dict__` 探测，也不再有 CUDA 同步兜底。
- `shrink_transfer` 两条路径的重复尾部合并为一份。
- KV 辅助函数改为显式传参：不再靠缺省值区分方向，也没有两种键格式或两种 reference 文本（§3）。
- 切换内步数改用局部列表；`compact` 与 `delta` 共用 `moves_kv`。
