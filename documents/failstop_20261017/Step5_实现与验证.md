# Step 5 实现与验证（第一批：确定性模式合同、显存预检与真实模型快验入口）

实现提交：`9405ed65c00d573938032e8c94a0d795258427fb`，分支 `codex/failstop`（统一在该分支开发，不再使用 Step 4 的独立分支）。快验清单提交与服务器命令见 [Step 5 服务器命令](Step5_服务器命令.md)。

本轮写代码前已阅读 CODEX.md、[总计划](00_总计划.md)、[Step 5 文档](05_Step5_模型正确性.md)（含 §2.1 确定性模式合同）、[路由确定性检查](路由确定性检查_服务器命令.md)、Step 3/4 的实现与服务器命令文档，以及 code/current/AGENTS.md；Python imports 变动后执行 `python -m isort`（本机无 `uv`）。用户 2026-10-17 指示：Step 3、Step 4 的 GPU 快验因 SL3060 显存被占用尚未通过，Step 5 在 CPU 层面开发与测试，上机顺序仍为 Step 3 → Step 4 → Step 5。本轮没有登录服务器、运行 GPU 或改动其他用户任务。

状态：§2.1 合同、迁移计划跨模式一致性检查、单步耗时比记录、每卡显存预检与 `quick_check.sh --step 5` 入口已实现，CPU 已验证；**真实模型的 fail→shrink→rejoin→expand 接入、A6 矩阵、A7 权重-only smoke、`run_failstop_smoke.sh` 与多轮逻辑尚未实现**，等 Step 3/4 服务器 GPU 通过后进行。服务器 GPU 全部待验证；fix2 GPU 复验仍未登记。

## 实现

| 文件 | 交付 |
|---|---|
| `failstop/determinism.py` | §2.1 合同。`mode_manifest` 记录进程能自证的模式：`--deterministic-mode` 参数、`torch.are_deterministic_algorithms_enabled()`、`NCCL_ALGO=Ring` 与 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，三者齐备才是 `deterministic`。`bitwise_record` 保存原 dtype 的 CPU 位、位 SHA256、manifest、布局、请求历史（request id + token 序列 + hash）与参数校验和；`compare_records` 先要求**双方 manifest 均为确定性模式**，再要求布局、步号、参数校验和、历史 hash 一致，然后整数位视图逐位比较，报告 dtype、shape、首个差异位置与值/位模式（复用路由确定性检查的 `first_mismatch`）。任一侧为默认模式或状态不一致即拒绝比较并说明原因，不退化为容差。`plan_consistency` 去掉临时 rank 后逐项比较两份计划（稳定源/目标身份、切片、字节、波次），报告首个不一致任务及字段；`compare_launch_plans` 比较两次 launch 的 `source_plan_<round>_<phase>.json`，要求 profile hash 相同才可比，并分别报告 membership/epoch/sources/targets 输入是否相同及两次 launch 的模式环境。`step_time_ratio` 要求同布局、同步号、同上下文长度配对，输出逐步、prefill、decode 中位数比值，标注"只作开销评估"。 |
| `failstop/model_memory.py` | 真实模型每卡显存需求。按角色给出 TP2/TP4 双布局共存窗口：EDP0 存活卡缩容窗口（源 TP4 权重 + 目标 TP2 权重 + 作废前的 TP4 KV + 重 prefill 的 TP2 KV + 按波次暂存的接收缓冲 ≤ 目标字节 + 2 GiB 余量）、扩容窗口（源 TP2 + 目标 TP4 + 双方 KV + 接收缓冲 + 余量）；回归卡扩容窗口（空 TP4 目标 + KV + 接收缓冲 + 余量）；EDP1 稳态（TP4 + KV + 发送打包临时量 ≤ 自身分片 + 余量）。公式复用路由确定性检查的精确参数/KV 字节数，窗口字段复用 Step 4 `PeakBudget`，使预检与 runner 的切换 admission 同一口径。`preflight` 查询本机 8 卡 `nvidia-smi` 空闲显存，输出 16 卡全表（远端节点只列需求，空闲显存在该节点查询）、本机角色缺口、本次参考运行所需显存及缺口；未解析估算的 cuBLAS 工作区、NCCL 缓冲、MoE 临时量、注意力分数与 allocator 预留明确列为运行后测量项。 |
| `failstop/model_reference.py` | torchrun 入口，单布局。按路由确定性检查同样的方式构建/加载 DeepSeek-V2-Lite（native router、local 后端）。`--mr-role reference`：在 Megatron `--deterministic-mode` 下 prefill + 4 步 greedy decode，保存每 rank 的逐步 logits 位记录（含每步对应的历史 hash、参数校验和、manifest）与逐步 GPU 耗时；`--mr-role candidate`：独立再次加载，生成同样记录并经 `compare_records` 与参考逐步比较；`--mr-role timing`：默认模式（不传 `--deterministic-mode`、不设 NCCL/cuBLAS 环境）只记录逐步耗时。参数校验和为逐参数位视图整数和的 SHA256，用于证明两进程加载了同一组分片。 |
| `failstop/run_model.py` | Step 5 套件入口。磁盘检查 → 显存预检（写 `memory_preflight.json`、打印全表、本次运行显存不足即拒绝）→ 每个布局依次 reference/candidate/timing 三个 torchrun → `evaluate`（参考与候选均为确定性模式且各 rank manifest 一致、参数校验和一致、历史一致、候选逐位相等、timing 为默认模式、耗时比有限）→ `summary.json`（`accepted=false`、`timing_valid=false`、`candidate_kind=fresh_checkpoint_load`、`plan_consistency.status=pending`、`fail_stop_integration=pending`）。快验布局分配：SL3060 跑 tp4，SL3061 跑 tp2，每节点 3 次 checkpoint 加载以适应 300 秒预算；正式模式 SL3061 跑 tp2+tp4。`--compare-plans A B` 调用 `compare_launch_plans`。 |
| `quick_check.sh`、`quick_driver.py` | `--step 5` 映射 `step05_model_correctness`，入口 `run_model.py`，不传故障矩阵参数；PASS 要求 `all_checks_passed`、quick/shared/timing_valid/accepted 标志正确、每个布局的 `deterministic_step_time_ratio.median_ratio` 为正有限数；最终单行改为 `QUICK CHECK step=5 result=... (memory preflight, deterministic reference/candidate [...], step_time_ratio={...}) shared_gpu_mode=true fail_stop_integration=pending`。Step 2–4 行为不变。 |
| `correctness/router_determinism_driver.py` | 抽出 `model_arguments(layout, checkpoint, prompt_tokens, seq_length)` 供 Step 5 复用，`torchrun_command` 行为不变（10 项既有测试通过）；已固定的 `router_determinism` 清单提交不受影响。 |
| `common.py` | 新增 `MODEL_PROBE`、`DETERMINISTIC_ENVIRONMENT`，`topology()` 拆出 `node_rows()` 供单节点预检复用，输出不变。 |

## Step 3/4 接口问题与修正（本轮直接修改）

1. **Step 3 每次 launch 现场采集 profile，无法满足 §2.1 第 2 条。** 迁移计划的输入含 profile（带宽/延迟决定选源与波次），默认模式与确定性模式两次 launch 若各自实测 profile，计划合法地可能不同，无法逐项比较。修正：`run_sources.py --profile-file <profile.json>` 读取此前 launch 保存的 `profile.json`，配置记录 `profile_file/profile_sha256`，worker 校验 hash 与设备身份后跳过采集并记录 `profile_frozen` 事件；控制器比较两节点 `profile_sha256` 一致；与 `--profile-only` 互斥。两次 launch（一次默认模式、一次确定性模式）使用同一冻结 profile 后，用 `run_model.py --compare-plans` 逐项比较。
2. **Step 3 的 `source_plan_*.json` 只有任务列表，不能离线复核。** 现改为同时保存 membership、epoch、rank_map、sources/targets 快照、`profile_source`（measured/frozen_file）与本进程的模式环境变量，使比较工具能区分"规划器差异"与"快照/环境差异"。
3. **快验驱动把 `edp1_rejoin_pause_s` 当作所有 step 的必填项并硬编码故障矩阵参数。** Step 5 没有故障事件，`quick_driver.py` 改为按 step 选择入口参数与验收字段。

未改动 Step 4 代码；Step 5 预检直接复用其 `PeakBudget` 字段。

## 每卡显存需求（真实模型，max_seq 128，余量 2 GiB）

| 角色 | 卡 | 约束窗口 | 组成 | 需求/卡 |
|---|---|---|---|---|
| EDP0 存活卡（etp 2,3 × ep 0,1，SL3061 GPU 2,3,6,7） | 4 | 缩容（大于扩容的 17.89 GiB） | TP4 源 3.97 + TP2 目标 7.93 + KV 0.03 + 接收缓冲 ≤7.93 + 余量 2 | **21.85 GiB** |
| EDP0 故障/回归卡（etp 0,1，SL3061 GPU 0,1,4,5） | 4 | 扩容 | TP4 空目标 3.97 + KV 0.01 + 接收缓冲 ≤3.97 + 余量 2 | 9.94 GiB |
| EDP1（SL3060 全部 8 卡） | 8 | 稳态 | TP4 3.97 + KV 0.01 + 发送打包 ≤3.97 + 余量 2 | 9.94 GiB |
| 参考运行 TP2（独立、顺序执行） | 4 | — | 7.93 + 0.02 + 2 | 9.94 GiB |
| 参考运行 TP4（独立、顺序执行） | 8 | — | 3.97 + 0.01 + 2 | 5.98 GiB |

接收/发送缓冲取"一个波次全部远程"的上界；实际峰值、allocator 预留与未估算项在 `result.json` 的 memory 字段里按 rank 记录，供回传后与此表对照。

## 验证

证据：[单测日志](step5_cpu_evidence/cpu_tests.log)、[packing/选源回归](step5_cpu_evidence/packing.log)、[dry-run](step5_cpu_evidence/dry_run.log)、[追溯记录](step5_cpu_evidence/validation.json)。Windows、Python 3.13.12、Torch 2.14.0+cpu。

| 检查 | 结果 |
|---|---|
| failstop unittest | 135/135 通过（基线 117 + 新增 18：合同 manifest/记录/计划/耗时比 10 项、显存预检 3 项、驱动与命令 5 项含真实 Megatron `parse_args` 解析入口参数、快验 step 5 路径 1 项、Step 3 冻结 profile 跨模式计划一致 1 项） |
| 原 NCCL packing / 选源回归 | 56/56 通过 |
| 路由确定性检查单测 | 10/10 通过（`model_arguments` 抽取后） |
| isort / py_compile / bash -n / git diff --check / dry-run | 通过 |
| 服务器 GPU、300 秒实际预算、确定性模式下 Megatron 参数校验与模型前向 | 未运行，待回传 |

CPU 测试覆盖：缺参数/缺算法/缺环境变量三类都判为默认模式且拒绝比较时点名原因；候选为默认模式、历史/参数/布局不同均拒绝；1 ULP（bf16 `0x3f80→0x3f81`）与 shape 差异被报告；去 rank 归一化与 Step 3 `normalized` 等价；首个差异任务与字段、任务数不同、profile 不同不可比；耗时比配对约束；角色按故障坐标划分，窗口数值与公式/`PeakBudget` 字段一致；预检 16 卡全表、缺口点名 GPU 与角色、参考运行缺口；命令/环境按角色区分确定性与默认；真实 Megatron parser 接受 `--deterministic-mode` 与 `mr-*` 参数；`evaluate/summarize` 的失败路径；dry-run 布局分配；快验驱动对 step 5 的验收与失败路径；Step 3 协议 CPU 替身在冻结 profile 下不调用采集、两次不同模式环境的 launch 计划逐项相同、篡改字节后被定位。

```powershell
$env:PYTHONPATH = "$PWD/code/current;$env:TEMP/weavetp-step2-deps"
$env:OMP_NUM_THREADS = '1'
python -B -X utf8 -m unittest discover -s code/current/tools/resharding/failstop -p 'test_*.py' -v
python -B -X utf8 -m unittest discover -s code/current/tools/resharding/correctness -p test_router_determinism.py
python -B -X utf8 -m pytest --noconftest code/current/tests/unit_tests/resharding/test_nccl_packing.py code/current/tests/unit_tests/resharding/test_source_search.py -q
```

## 边界与未完成项

- 本批的 candidate 是独立的新加载进程（`candidate_kind=fresh_checkpoint_load`），用于在真实模型上验证跨进程参考/候选比较合同（manifest、参数校验和、历史 hash、逐位）；它不是迁移后的被测模型。A6 各检查点的被测对象要等 Step 3/4 GPU 通过后接入真实模型的 fail→shrink→rejoin→expand。
- 迁移计划跨模式一致性检查已实现为工具（`--compare-plans`）并在 CPU 协议替身上验证；服务器上的实际比较需要两次使用同一 `--profile-file` 的 Step 3 launch（默认模式与确定性模式各一次），本批快验 summary 中该项标记 `pending`。
- 单步耗时比在快验中以共享 GPU 采集，只作开销评估；正式模式同样只是开销估计，不是性能结论。默认模式 timing 进程不设置 NCCL/cuBLAS 环境变量，与正式性能运行配置一致。
- 快验每节点 3 次 59 GB checkpoint 加载；路由确定性检查两次加载实测 123 秒，本入口是否在 300 秒预算内尚未实测。
- 显存表中的接收/发送缓冲为上界，若实测峰值明显低于表值，可在集成时按实测收紧，不在本批调整。
- `run_failstop_smoke.sh`、emulate `--rounds`、real 每轮独立 launch、KV 参考与 A7 权重-only smoke 均未实现。
