# Claude Code 实施指令：第 1 行扩缩容端到端实验（最终版，只比固定 TP2 / 固定 TP4）

> 把本文件整体作为任务说明交给 Claude Code。仓库：`ksqbxm/WeaveTP`，代码在 `code/current`（Megatron 派生），实验资料在 `documents/e2e_prep_20261008/`。
> 本指令只覆盖扩缩容端到端这一张图所需的全部代码；**不做** Llumnix / Flying / AnchorTP / 重启基线，**不做**容错。

---

## 0. 目标与交付物

对比三条线：**WeaveTP（TP2×DP8 ↔ TP4×DP4 预先规划的在线切换）**、**固定 TP2×DP8**、**固定 TP4×DP4**。所有方法跑同一组请求、相同输入 / 输出 token、相同 16 卡与显存、必须完成相同任务。

两张图（数据定义见 §4）：

| 图 | 横轴 | 纵轴 | 到达方式 |
|---|---|---|---|
| 图 1（主图） | 短/长请求条数比 46 / 92 / 138 / 207 / 276 / 368 / 460 | 吞吐 tok/s（同时记录"输入+输出"与"仅生成"两种口径） | 连续提交（闭环 FCFS） |
| 图 2 | 扩缩容次数 1 / 2 / 3 / 4（单数扩容、偶数缩容） | 吞吐 tok/s | 分阶段（每段等上一段全部完成再提交） |

交付：(a) 能跑三种方法的离线负载驱动器；(b) 数据集生成器与 plan.json 生成器；(c) 运行脚本与结果汇总；(d) 每个点的 `result.json` 与汇总表。预期比值见 §7，用于自检。

---

## 1. 不可更改的事实与常数（来自实测，`进度总览_20261009.md` §3、§5）

| 项 | 值 |
|---|---|
| 模型 / 引擎 | DeepSeek-V2-Lite，Megatron static 推理路径，MLA **解压格式** KV，无 flash-attn |
| 硬件 | 16 × RTX 5090，PCIe、无 NVLink，两机各 8 卡（SL3060 / SL3061），单卡 34.19 GB（十进制） |
| 布局 | TP2×DP8 ↔ TP4×DP4，EP = 2（`EXPERT_PARALLEL_SIZE=2`），ETP = TP，TP / ETP / EP 组都在机内；TP2 副本 2i、2i+1 合并为 TP4 副本 i |
| 每卡 KV | TP2 138,240 B/token，TP4 69,120 B/token |
| 每卡权重 | TP2 8.835 GB，TP4 4.358 GB；非权重开销 2.1 GB；稳态余量 2.0 GB |
| 每副本 KV 容量（预留口径） | TP2 153,754 token，TP4 372,280 token |
| 切换峰值预算 | 每卡"源 KV + 目标 KV" ≤ 15.9 GB（= 34.19 − 8.835 − 4.358 − 2.1 − 2.0 接收缓冲 − 1.0 余量） |
| B_max | 256 条 / 副本，TP2 与 TP4 相同 |
| prefill 上限 | 每副本每步新接纳请求输入合计 ≤ 4096 token；单条超过时独占一步 |
| 已测切换 | WeaveTP 16 卡：2→4 约 5.4 s（6 波）、4→2 约 15.4 s（22 波），切换期间继续解码；扩容每波上限 8192、缩容 4096；带宽感知选源开启 |

仓库规则：遵守 `code/current/AGENTS.md`（改过 import 的文件跑 `uv run isort`）；不要碰与本任务无关的模块；每个工作项单独提交。

---

## 2. 现有代码地图（先读，再动手）

| 路径 | 作用 | 与本任务的关系 |
|---|---|---|
| `code/current/examples/rl/benchmark_live_moe_tp.py` | 现有 live 切换基准：固定 KV、固定 token、按 `--live-switches` 次数切换；含 `_prefill`、`_run_async_waves`、`_validate_cutover`、`_write_results` | **驱动器的起点**。保留其切换流水（备用权重、分波、选源、cutover、逐位审计），把"固定 KV 固定 token"换成按请求推进 |
| `code/current/megatron/core/resharding/`（`planner.py`、`live.py`、`execution.py`、`refit.py`、`transforms.py`） | 切换计划、波调度、带宽感知选源、KV 传输 | 不改算法；需要让 KV 传输任务按"连续块区间"生成（§3 W2 的 K2） |
| `code/current/tools/resharding/standby_weights.py` | 备用权重的分配 / 释放 | 备用布局的 **KV** 也要释放（K7） |
| `code/current/tools/resharding/compare_weavetp_16gpu.py`、`run_deepseek_v2_lite_live_benchmark.sh`、`profile_weavetp_16gpu.py` | 两机 16 卡的启动、环境变量（`NNODES=2`、`NCCL_SOCKET_IFNAME=eno1np0` 等）、带宽 profile | 复用启动与环境；新增的端到端入口沿用同一套包装 |
| `documents/e2e_prep_20261008/tools/cycle/cycle_bench.py` | 仓库外包装器：KV 预填、切换周期、显存峰值 | 校准用（§6 第 0 步），不进端到端 |
| `documents/e2e_prep_20261008/tools/make_requests.py` | S1 + S2 切 token、过滤、写 requests.jsonl / workload / manifest（**旧方案**） | 改成 §4 的定稿规则 |
| `documents/e2e_prep_20261008/sim/e2e_sim.py`、`make_plan.py` | 逻辑步模拟器、plan.json v2 生成 | 生成 plan.json；校准后重算 |
| `documents/e2e_prep_20261008/延后事项与提醒清单.md` §2 | 层 2 KV 布局硬性要求 K1–K7 | W2 的验收标准 |
| `documents/e2e_prep_20261008/第1行扩缩容端到端_正式前置规划.md` §4、§5 | 调度规则、切换时机与 plan.json 规则 | W1、W3 的规格 |

---

## 3. 工作项（按顺序；每项含验收标准）

### W1 层 1：离线负载驱动器（原 C1 + C4）

新文件建议：`code/current/examples/rl/run_e2e_scaling.py`（复用 `benchmark_live_moe_tp.py` 的初始化、模型构建、切换流水；不要复制粘贴整文件，抽公共函数）。

功能：
1. 读 `requests.jsonl`（每行 `id, prompt_token_ids, output_token_ids, input_len, output_len`）、`workload.json`（段列表：`kind: short|long`、`request_ids`、`arrival: phased|continuous|open_loop`、open_loop 时每条请求的 `arrive_s`）、`plan.json`（可为空 = 固定布局）。
2. 全局接纳控制器（rank 0 决策、广播），规则：
   - 严格 FCFS，不跳过队首；
   - 请求放入"未满 B_max 且剩余预留 ≥ 输入 + 输出"的副本中预留最少的一个，并列取编号小者；
   - KV 预留 = 输入 + 参考输出长度（物理上按块增长，见 W2）；
   - 每副本每步 prefill 输入合计 ≤ 4096 token；
   - 分阶段模式：第 k 段的请求在第 k−1 段全部完成前不接纳；开环模式：`arrive_s` 之前不接纳，空闲时等待到下一条到达；
   - 切换期间暂停接纳。
3. 逐步解码循环：所有副本同步步进；**teacher forcing**——每步输入取参考答案 token（`output_token_ids`），路由用 native；请求在生成完 `output_len` 个 token 后结束并释放 KV。
4. 逐请求记录：`t_arrive`、`t_admit`（开始 prefill 的墙钟）、`t_first`（首 token）、`t_done`（末 token）、所在副本、所在布局。
5. 任何"跳过 / OOM / 提前结束"判断先在所有 rank 间达成一致（清单 §7.4 教训）。

验收：固定 TP2 跑一个 2k 条短请求的小负载，总 token 数与 requests.jsonl 一致、逐请求时间单调、无 rank 不一致；同一负载跑两次吞吐差 < 2%。

### W2 层 2：按块 KV（原 C2 + C9，清单 §2 K1–K7 是硬性验收）

- K1 物理 KV 按块随生成增长；请求结束、以及切换前，空闲块真正归还分配器（切换峰值检查按 KV_used 算的前提）。注意 PyTorch 缓存分配器碎片：切换前把存活块压实。
- K2 迁移必须表达成少量大块连续传输：切换前按"目标副本 → 请求"顺序把块压实，使每个（层, K/V, head 切片, 目标副本）对应一段连续区间，任务数与现在同量级（不能变成按请求 × 块的 10⁵ 个任务）。
- K3 head 切片在内存中连续（head 在外层，例如 `[heads, tokens, dim]`），避免接收端临时缓冲吃掉峰值预算。
- K4 变长注意力不做每步整体 gather（按长度分组或按块读）；实现后必须重测单步时间（M1）。
- K5 D16 合并 / 拆分：2→4 时 TP4 副本 i 接收 TP2 副本 2i、2i+1 的请求；4→2 按请求编号依次分给预留较少的子副本；KV 传输按"请求 × head 切片"生成。
- K6 位置表覆盖 16k + 余量，与静态 KV 解耦。
- K7 备用布局的 KV 释放或只为存活请求分配（`codex/standby-weight-storage` 只释放了备用权重）。

验收：每卡 KV 物理占用 ≈ 已用 token × 每 token 字节（误差 < 10%）；切换后 logits 校验与权重逐位审计仍通过（沿用现有 `_validate_cutover` / `--live-weight-bitwise-audit`）；一次 2→4 的 KV 传输任务数不超过现在固定 KV 基准的 2 倍。

### W3 计划执行器（原 C3）

- 读 plan.json v2 的 `switches[]`，按顺序逐个布防：
  - `before_admit(X)`：X 到达队首时冻结接纳并切换；
  - `after_complete(Y)`：Y 完成的那一步末尾切换；布防时 Y 已完成则立即切换。
- 触发前做逐卡峰值检查（`rule.switch_budget`，`check = per_card_used_plus_growth`）：
  - 扩容：`U'_TP2副本 × 138,240 + (U'_2i + U'_2i+1) × 69,120 ≤ 15.9 GB`；
  - 缩容：`U'_TP4副本 × 69,120 + U'_子副本 × 138,240 ≤ 15.9 GB`；
  - `U' = 当前已用 KV + 在跑请求数 × G`，G = 切换期间前台解码步数 + 1。不满足则暂停接纳只解码，直到满足。
- 缩容前限流（`pre_shrink_throttle.factor = 0.95`）：长段最后一条请求接纳之后，每个 TP4 副本预留上限临时降为 `min(0.95 × 2 × C_TP2, 0.95 × 预算 ÷ (2 × 69,120) − B_max × (G + 1))`。
- 切换沿用现有 WeaveTP 流水：备用权重接收、分波、带宽感知选源、稳定前缀 + 增量 KV、token 边界 cutover、切换后释放源布局权重与 KV。
- 固定 TP2 / TP4：plan 为空，永不切换，**不得**产生任何切换类开销。

验收：用 §4 图 2 的 1 次切换负载跑 WeaveTP，切换发生在计划锚点；每次切换的墙钟、期间解码步数、搬运字节写入结果；峰值显存 ≤ 预算 + 0.5 GB。

### W4 结果与日志（原 C10）

`result.json` 必须包含：
- 总时长、总 token（输入 + 输出）、生成 token、两种口径吞吐；
- 每次切换：方向、锚点、开始 / 结束墙钟、期间解码步数、暂停接纳时长、排空等待时长、搬运字节、峰值显存；
- 逐请求：`t_arrive / t_admit / t_first / t_done`、段编号、长 / 短；
- 配置快照：plan.json、workload.json 的 sha256、环境变量、git commit。

另给 `tools/resharding/summarize_e2e_scaling.py`：读多个 `result.json`，输出图 1 / 图 2 的表（均值 ± σ、比值）与 TTFT 统计（离线算，只记录）。

### W5 数据集生成器（仓库外，改 `documents/e2e_prep_20261008/tools/make_requests.py`）

按 §4 的规则生成；输入是服务器上的原始数据（`BASE/data/sharegpt/`、`BASE/data/sky_t1_17k/`、以及 s1K / LIMO / LongWriter-6k / OpenR1-Math 的长输出池，统计见 `sim` 的 `lengths_longout.csv`），tokenizer `DeepSeek-V2-Lite-hf`，不套模板，`add_special_tokens=False`，输入前加 1 个 BOS，只过滤不截断。输出 `requests.jsonl`、`workload_<name>.json`（含 `segments[].arrival`）、`lengths_all.csv`、`manifest.json`（sha256 与逐段统计）。

### W6 plan.json 生成（仓库外，改 `documents/e2e_prep_20261008/sim/make_plan.py`）

- 读新的 `workload_<name>.json`，`BARS = ["weavetp", "fixed_tp2", "fixed_tp4"]`；
- 分阶段负载用 `barrier=True`，开环负载把 `arrive_s` 填进 `Req.arrive`；
- 图 2 奇数点：负载以长段结束，去掉最后一次 4→2；
- 校准（§6 第 0 步）之后 `apply_calib()` 重新生成一次。

### W7 运行脚本

- `tools/resharding/run_e2e_scaling_grid.sh`：遍历 方法 × 负载点 × 种子，沿用 `run_deepseek_v2_lite_live_benchmark.sh` 的两机启动与环境变量；每个 case 一个输出目录，已完成的目录不重跑；
- 支持 `--segments k..m`：只跑负载的某几段（§5 的快照式组合）。

---

## 4. 数据集定稿（直接照此生成，细节见 `最终数据集定稿_20261013.md`、`最终实验配置_20261012.md`）

### 4.1 池

| 池 | 规则 | 规模与统计 |
|---|---|---|
| 长请求池 | s1K / LIMO / LongWriter-6k / Sky-T1 / OpenR1-Math，总长（输入 + 输出）7600–8400、输入 ≤ 800、输出 ≥ 64 | 472 条；输入均值 189（12–799），输出均值 7811，总长均值 8000 |
| 短请求池 P（图 1） | ShareGPT，输入 ≥ 512、输出 ≤ 64 | 9,810 条；输入均值 724、输出均值 21 |
| 短请求池 B1–B4（图 2） | ShareGPT，输出 ≤ 64，输入分别落在 [32,128) / [128,256) / [256,512) / [512,1024] | 2,959 / 3,940 / 11,624 / 3,775 条 |

长请求不放回抽样；短请求有放回（允许重复）。每段、每个种子独立抽样，条数带 ±3% 随机，保证"特征相同、样本不同"。

### 4.2 图 1 负载（7 个点，连续提交）

每点：短段 ns/2 条 → 长段 174 条（= 0.95 × TP4 全集群并发 184，高于 TP2 的 152）→ 短段 ns/2 条。ns = 比值 × 174：

| 短/长比 | 46 | 92 | 138 | 207 | 276 | 368 | 460 |
|---|---|---|---|---|---|---|---|
| 短请求总数 | 8,000 | 16,000 | 24,000 | 36,000 | 48,000 | 64,000 | 80,000 |

切换计划：长段第 1 条接纳前 2→4，最后一条完成后 4→2（2 次）。种子 5 个（7 / 11 / 13 / 17 / 19）。

### 4.3 图 2 负载（4 个点，分阶段）

第 N 点 = S0 + L1 [+ S2 + L3 + S4…]（单数扩容、偶数缩容）；奇数点以长段结束。长段每段 ≈ 174 条（170–179）；短段每点取对应输入档，每个短段输入 token 总量 = 26M × 长段数 ÷ 短段数：

| 切换次数 | 段序列 | 短段输入档 | 每短段条数 |
|---|---|---|---|
| 1 | S0, L1 | [32,128) | 361,309 |
| 2 | S0, L1, S2 | [128,256) | 65,660 |
| 3 | S0, L1, S2, L3 | [256,512) | 69,432 |
| 4 | S0, L1, S2, L3, S4 | [512,1024] | 23,951 |

可选修正：第 4 点输入总量改 30M 以消除轻微回落。种子 3 个（7 / 11 / 13）。

---

## 5. 执行顺序与机时压缩

| 步 | 内容 | 机时 |
|---|---|---|
| 0 | 校准：`tools/cycle/run_cycle_grid.sh`（命令 1）、`run_microbench_ctx.sh`（命令 2）、`run_calibrate.sh` → `calib.json`；用它重生成 plan.json | 两机各 1–2 h，可与 W1–W3 并行 |
| 1 | W1–W4 完成后：小负载冒烟（固定 TP2、WeaveTP 各 1 次，2k 短请求 + 40 条长请求） | < 30 min |
| 2 | **图 2 用快照式组合**：分阶段下各段互不重叠，每种段每种方法只测一次（8k 长段：WeaveTP ≈ 46 min、固定 TP2 ≈ 96 min、固定 TP4 ≈ 50 min；四档短段 10–160 min），切换次数 3、4 的点由测过的段组合（`--segments`）。首尾半段短段按完整段的单位时间折算 | 三方法合计约 15–20 h，两机可分方法并行 |
| 3 | 图 1：7 点 × 3 方法 × 5 种子直接跑会超过 40 h；改为 **3 个种子 + 每点只完整跑 WeaveTP，固定布局用段组合**（固定布局无切换，段之间无耦合） | 约 12 h |
| 4 | 汇总：`summarize_e2e_scaling.py`，与 §7 预期对照 | — |

快照式组合的前提是分阶段到达；图 1 的连续提交下 WeaveTP 必须整点完整跑（短段与长段有重叠）。

---

## 6. 不要做的事

- 不实现运行时自主切换决策；切换只按 plan.json 的锚点。
- 不改 B_max、prefill 上限、显存余量；不装 flash-attn；不改 EP。
- 固定 TP2 / TP4 不得走任何切换代码路径。
- 不截断请求；不丢请求；三种方法的 requests.jsonl 必须 sha256 相同。
- 不把 Llumnix / Flying / AnchorTP / 重启的代理混进本任务。
- 校准常数变了要重生成 plan.json 并记录版本，不要手改锚点。

---

## 7. 自检用的预期值（模拟，§32–§34；实测允许 ±15%，超出要查原因而不是改数据）

图 1（WeaveTP ÷ 固定 TP2 / ÷ 固定 TP4）：46 → 1.69 / 1.15；92 → 1.65 / 1.28；138 → 1.62 / 1.40；207 → 1.55 / 1.54；276 → 1.49 / 1.66；368 → 1.45 / 1.82；460 → 1.40 / 1.95。WeaveTP 单点时长 0.8–1.3 h。

图 2（同上）：1 次 → 1.32 / 1.59；2 次 → 1.48 / 1.43；3 次 → 1.54 / 1.36；4 次 → 1.53 / 1.36。

关键机理（偏差时先查这几项）：
1. 长段里固定 TP2 必须出现**两批**（174 > 152），固定 TP4 一批；否则检查 KV 预留是否按输入 + 输出、容量常数是否对。
2. 短段里固定 TP4 的 prefill 应约为 TP2 的 1/2.7（副本数一半 × 每副本更慢）；否则检查 prefill 路径。
3. WeaveTP 每次切换约 5–15 s，切换相关时间占比 < 3%；否则检查峰值等待 / 限流是否过长。
4. 比值与吞吐口径无关（两种口径的比值应完全一致）。

---

## 8. 汇报格式

每个工作项完成后给：改动文件列表、验收结果（数字）、未达项与原因。全部完成后给图 1 / 图 2 的汇总表（均值 ± σ、比值、时长）、校准前后的 plan.json 差异、与 §7 的偏差说明。
