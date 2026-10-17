# Step 6 计划：统一服务驱动（扩缩容端到端与 fail-stop 端到端共用）

状态：**计划稿，待用户审阅；未编码。** 2026-10-18 用户指示：Step 6 分两部分。A 部分现在开始（不依赖 Step 3–5 上机结果）：先写本计划供审阅，再编码控制器、requests_io、BlockPool、分页注意力参考实现与回退实现、timeline/results 写入，并用 fake 后端跑通整个驱动循环，快验入口只用单节点。B 部分等阶段 A（Step 5）服务器验收通过后做：服务驱动与迁移的衔接、真实模型批处理逐位检查。

写本计划前已阅读 CODEX.md、[总计划](00_总计划.md) §6.2/§7/§8、[原 Step 6](06_Step6_批量服务与事件.md)、[Step 7](07_Step7_端到端矩阵与统计.md)、[扩缩容端到端正式前置规划](../e2e_prep_20261008/第1行扩缩容端到端_正式前置规划.md) §2.3/§4/§5/§9、[延后事项与提醒清单](../e2e_prep_20261008/延后事项与提醒清单.md) §2 K1–K7、[最终实验配置](../e2e_prep_20261008/最终实验配置_20261012.md)。

## 0. 必须先确认的来源缺口

用户指示以"扩缩容计划的 W1–W4（逐请求接纳控制器、块级 KV、计划执行器、结果与时间序列；分页注意力先试 Triton，sm_120 不可用时用 split-KV 纯 torch）"为蓝本。**仓库、全部 git 历史、四个工作树与 `~/.codex/plans` 中都没有包含 W1–W4、`requests_io`、`BlockPool`、split-KV、分页注意力或 sm_120 的文档。** 仓库里最接近的是扩缩容规划 §9 的改动清单 C1–C11 与延后事项 §2 的 KV 硬性要求 K1–K7：

| 用户所述 W 项 | 仓库中对应项 | 内容 |
|---|---|---|
| 逐请求接纳控制器 | C1 层 1 + §4 调度规则 | 读 `requests.jsonl`/`workload.json`/`plan.json`；严格 FCFS；预留 = 输入 + 输出；B_max 256；每步 prefill ≤ 4096 token；切换期间暂停接纳 |
| 块级 KV | C2 层 2 + K1–K7 | 物理按块增长并归还；迁移表达为少量连续块；head 切片连续；变长注意力不每步 gather；D16 合并拆分；位置表覆盖 16k；备用布局 KV 不常驻 |
| 计划执行器 | C3 | 锚点布防与触发、逐卡峰值检查、缩容前限流、切换期间暂停接纳 |
| 结果与时间序列 | C10 | 总吞吐、每次切换墙钟/期间步数/停顿/字节、逐请求首末 token 时刻 |
| 分页注意力 Triton / split-KV 回退 | 无 | 仓库只有"Megatron 动态批处理需要 flash-attn，服务器没有"的记录（模拟器核查 §106–107） |

本计划按上表映射编写，并把 W1–W4 的措辞保留为模块名。**请审阅时提供 W1–W4 原文或确认按此映射执行**；若原文含本计划未覆盖的接口或数字，以原文为准修订后再编码。

## 1. 目标与边界

- 一份服务驱动实现，位于 `code/current/tools/resharding/serving/`，被两个实验共用：扩缩容端到端（TP2×DP8 ↔ TP4×DP4，teacher forcing，请求集来自 `make_requests.py`）与 fail-stop 端到端（EDP0/EDP1 TP4，EDP0 TP4→TP2→TP4，真实 greedy，ShareGPT 子集）。两者的差异只能出现在配置（解码策略、容量来源、事件来源、副本拓扑），不能出现在第二套循环或第二套 KV 实现。
- 不实现：chunked prefill、正常运行抢占、teacher forcing 之外的采样、AnchorTP/restart 等基线方法（它们是扩缩容实验的 C5–C8，不属于本步）。
- 本步 A 部分只交付 CPU/fake 后端可验证的部分；真实模型与迁移衔接为 B 部分。CPU 通过不代表 batched logits 正确（原 Step 6 §5）。

## 2. 模块与接口

目录 `code/current/tools/resharding/serving/`：

| 模块 | 职责 | 关键接口（草案） |
|---|---|---|
| `requests_io.py` | 读请求集、负载与计划；校验 sha256；写逐请求日志 | `load_requests(path) -> list[Request]`（字段按扩缩容 §2.3：`id, prompt_token_ids, output_token_ids, input_len, output_len, sha256_text`；fail-stop 用 `output_len` 作固定目标长度，`output_token_ids` 只用于 teacher forcing）；`load_workload(path)`；`load_plan(path, request_sha256)` 同时接受 `v2-failstop` 与扩缩容 v2 的 `rule/anchors`，根字段 sha 不符即拒绝；`RequestLog.append(event)` |
| `admission.py`（W1） | 逐请求接纳控制器 | `Admission(replicas, policy)`：严格 FCFS 不跳过队首；`fits(replica, request)` = 槽位未满 B_max 且剩余预留 ≥ 输入 + 输出（fail-stop 的输出 = 固定目标长度；扩缩容 = 参考输出长度）；并列取编号小者；`freeze()/thaw()` 切换期间暂停；`capacity(replica)` 由后端按布局给出（EDP0 停服为 0，降级后按 TP2 容量）；故障恢复用现有 `async_state.recovery_queue` 把在途请求按原序回队首，保留原提交时间 |
| `block_pool.py`（W2） | 块级 KV | `BlockPool(layers, kv_heads_local, head_dim_k, head_dim_v, block_tokens, capacity_blocks, dtype)`：布局 `[layer][K|V] -> [heads, capacity_blocks*block_tokens, dim]`（head 在外，K3）；`allocate(request, tokens)` 按块增长（K1），`release(request)` 完成即归还；`compact(order)` 切换前按"目标副本 → 请求"把块压实成每个（层, K/V, head 切片, 目标）一段连续区间（K2），返回迁移区间表；`positions(slot)` 逐槽位置表与有效长度（K6 用上限 16k + 余量）；备用布局不建池（K7）；`snapshot()/delta(since)` 给 fail-stop 扩容的前缀 + 增量 |
| `paged_attention.py` | 变长批注意力，直接按块读（K4） | `attention(q, pool, slot_table, valid_lengths)`：参考实现 = 纯 torch split-KV（按块分段算 softmax 分子/分母再合并，支持 MLA 的 qk_dim 192 / v_dim 128 不等维）；`triton_attention` 同签名，启动时探测 Triton 是否可用于当前 sm（服务器为 sm_120），不可用则整轮固定用参考实现并写入 `attention_impl` 字段；所有方法同一实现，一轮内不切换 |
| `executor.py`（W3） | 计划执行器 | 锚点类型：扩缩容 `before_admit(X)`/`after_complete(Y)`、fail-stop `completed_count`/`after_event_seconds + wait_for`；按序布防、首次跨越只触发一次并记录 overshoot；逐卡峰值检查用 Step 4 `PeakBudget` 字段；缩容前限流 `pre_shrink_throttle`；切换期间 `Admission.freeze()`；fail-stop 事件交给现有控制器/注入器执行（总计划 §7 的 fail → 通知 → 恢复顺序与 `t_rejoin_*` 公式不变） |
| `driver.py` | 服务循环 | 每副本独立推进：组批（新接纳请求 full prefill，每步输入合计 ≤ 4096 token，单条超过独占一步）→ 前向 → 逐槽取 token（greedy 或 teacher forcing）→ 提交 → 释放完成请求的块并 FCFS 补入；同副本空 dense DP 分支执行空前向保持 EP collective 顺序；任何 OOM/跳过判断先在副本内所有 rank 达成一致；固定输出长度忽略 EOS |
| `timeline.py`（W4） | 结果与时间序列 | 逐秒表：各副本与全局成功提交的新增 token、布局、活跃槽、队列长度、KV_used、事件状态（padding/空前向/重放不计）；逐请求：提交、首次接纳、每次入队/恢复、每个有效 token、首 token、完成、迁移、prefill/replay token；`events_actual.json` 按总计划 §7 字段组；`summary` 由 Step 7 口径离线重算，本步只写原始数据 |
| `backends.py` | 后端抽象 | `Backend.prefill(batch) / decode(batch) / capacity(layout) / kv_used()`；`FakeBackend`：CPU，确定性玩具 logits（token = f(请求 id, 位置)），模拟块占用与可配置单步时间，用于跑通整个循环与所有协议测试；`MegatronBackend`：B 部分，真实模型 + BlockPool + 分页注意力 |

现有代码复用：`async_state.Request/History/recovery_queue/PeakBudget/ReplicaDriver`（Step 4 状态机与恢复队列）、`failstop` 控制器/监管器/注入器（事件与 PID 生命周期）、Step 3 混合规划器与传输（B 部分迁移）、Step 5 `determinism.py`（B 部分逐位检查合同与模式 manifest）。`model_reference.py` 的单一全局 offset（StaticInferenceContext）正是本步要去掉的限制，不在服务驱动中沿用。

## 3. 规格数字

| 项 | fail-stop 端到端 | 扩缩容端到端 | 来源 |
|---|---|---|---|
| 解码 | 真实 greedy，达到固定输出长度才结束，忽略 EOS，输出 ≤ 256 | teacher forcing（输入取参考 token），忽略 EOS | 原 Step 6 §2；扩缩容 §4 |
| 请求集 | 用户在阶段 A 后指定的 ShareGPT 子集 | `requests.jsonl`（8000 短 + 长段 600/900/1200，或最终配置的 174 条 8k 长请求） | 原 Step 6 §1；扩缩容 §3；最终配置 |
| 槽位上限 | 每副本固定，由 profile 验证后冻结 | B_max = 256 | 原 Step 6 §2；扩缩容 §4 |
| 每步 prefill | ≤ 4096 token/副本（沿用） | ≤ 4096 token/副本 | 扩缩容 §4 |
| 容量 | 按布局的 KV 容量（TP2/TP4 不同），EDP0 停服为 0 | TP2 153,754 / TP4 372,280 token/副本 | 原 Step 6 §2；最终配置 |
| 块大小 | **待定**：文档未固定；建议 64 token（与 Megatron 动态上下文的 64 对齐，便于日后对比），审阅时确认 | 同左 | 本计划提议 |
| 事件 | `completed_count = ceil(0.25N)` fail，`after_event_seconds=45, wait_for=shrink_done` rejoin | `before_admit(X)` 扩容，`after_complete(Y)` 缩容，限流 0.95 | 总计划 §7；扩缩容 §5 |
| 峰值检查 | Step 4 `PeakBudget`（源/目标权重、源/目标 KV、接收缓冲、临时、余量） | 同一字段，常数取扩缩容 §5 `switch_budget` | Step 4；扩缩容 §5 |

## 4. A 部分交付（现在开始，CPU/fake 后端）

1. 上述七个模块的实现，`MegatronBackend` 只留接口不实现。
2. CPU 测试（`tools/resharding/serving/test_*.py`，与 failstop 测试同一发现方式）：
   - 接纳：FCFS 不跳队首、并列取小编号、预留不足等待、切换冻结/解冻、EDP0 停服容量 0、恢复队列保留原序与提交时间。
   - 块池：按块增长与归还、完成即释放、块复用不读到上个请求（用唯一填充值证明）、压实后每个（层, K/V, head 切片, 目标）恰为一段连续区间且内容逐位不变、位置表与有效长度、快照/增量边界。
   - 分页注意力：参考实现对任意块边界与变长批与稠密注意力（同 dtype fp32，CPU）逐位相等；Triton 不可用时的探测与回退记录；MLA 不等维。
   - 执行器：锚点首次跨越只触发一次并记录 overshoot、`ceil` 边界、rejoin due 早于/晚于 shrink_done、回收晚于到期不重复计 join、hash 不符拒绝运行。
   - 时间序列：从逐请求日志独立重算 completed_count 与新增 token 总量，padding/空前向/重放不计；schema 结构校验。
   - 全驱动循环：fake 后端上两副本独立推进、一个 dense DP 空、EDP0 故障 → 在途重放 → 容量不足回队首 → 扩容后槽位复用，运行到请求集排空；输出 `events_actual.json`、逐请求日志、逐秒表。
3. 快验入口：`quick_check.sh --step 6 --single-node` 在单节点运行 fake 后端全循环与上述 CPU 测试，不依赖 Step 3–5；结果标识 `step06_batched_serving`，`accepted=false`，结论行注明 `backend=fake`。
4. 交付文档 `Step6_实现与验证.md` 与服务器命令；CPU 通过明确标注"不证明 batched logits 正确"。

## 5. B 部分（阶段 A 服务器验收通过后）

- `MegatronBackend`：真实 DeepSeek-V2-Lite + BlockPool + 分页注意力替换 StaticInferenceContext 路径；确定性模式下批处理逐位检查（同布局、同历史、同批次与顺序的参考），参考用 Step 5 的记录格式与 `compare_records`。
- 与迁移衔接：切换前冻结接纳；KV 快照（前缀）与切换前增量按块区间交给 Step 3 规划器/传输；缩容后 EDP0 在途请求按"原输入 + 已生成"重新 prefill；扩容不停服（TP2 继续提交，目标就绪后补增量、校验、原子切换，继承 Step 4 合同）。
- 原 Step 6 §6 的六个 GPU 用例。

## 6. 主要风险

1. **分页注意力与 Megatron 本地注意力路径的接合**：服务器没有 flash-attn，Megatron 的动态上下文/分页 KV 依赖 flash-attn 或 flashinfer；本步要在本地 `DotProductAttention` 推理路径上接入自己的块读取注意力，MLA 的 qk/v 维度不同、位置编码（rope 部分 64 维）需在 prefill/decode 两条路径上一致。这是 B 部分最大的技术风险，A 部分先用 CPU 参考实现把语义定死。
2. **块大小与任务数**：块太小则迁移任务数失控（K2），太大则浪费容量；压实步骤是 K2 的关键，A 部分用 CPU 测试证明区间连续性与字节不变。
3. **两个实验的差异渗入循环**：解码策略、容量来源、事件来源都必须是配置项，代码评审时以"只有一个 `driver.step`"为检查点。
4. **W1–W4 原文缺失**：见 §0。

## 7. 审阅问题

1. W1–W4 原文在哪里？若没有文件，是否按 §0 映射执行？
2. 块大小 64 token 是否接受？
3. Triton 在服务器（sm_120，RTX 5090）上的版本与可用性是否已知？不可用时整轮用 split-KV 参考实现是否接受？
4. fail-stop 端到端的 greedy 与扩缩容的 teacher forcing 是否都通过 `decode_policy` 配置在同一驱动实现？
5. 原 Step 6 的 `step06_batched_serving` 标识与输出目录是否沿用于两个实验（扩缩容实验的输出目录另定？）。
