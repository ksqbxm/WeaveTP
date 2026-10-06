# 备用权重 GPU storage 释放：实现与验收

## 2026-10-06：验收失败审阅与修复

已读取用户提供的完整 `standby_20261006T045416Z_2293839` 结果，原提交为 `9f084a5974885d43e51f856825d75a2e137125f6`，服务器使用 PyTorch 2.11.0+cu128 / NCCL 2.28.9。本次代码修复尚未经过 GPU 复验，不将 CPU 流模型或旧 GPU 结果当作新版本 GPU 通过证明。

### 根因与证据边界

- **ordering 的后续 NCCL 超时有确定原因：** 单 rank 在 `progress > 0` 断言处失败，另一个 rank 已进入下一批通信，最后卡在 seq3。原测试每 rank 的远端载荷仅约 2 KiB，延迟在重新分配前排入，launch 后还有 barrier，且只在前台完成后查询迁移状态；这些条件可错过重叠窗口。日志不能证明当时绝对没有重叠。
- **expandable DeepSeek ON 尚未确认唯一根因：** 第二次 4→2 的 base 迁移前 8 个 wave 完成，第 9 个 wave 的 PG33 seq36 超时；调用栈是 `_run_async_waves → launch_reshard_plan → batch_isend_irecv`。本轮全权重有限值检查还没有开始，不能将故障归因于该检查期间的 decode；也不能仅凭分配器组合就认定 cuMemMap/NCCL 是根因。需要所有 rank/PG 的 flight recorder。默认分配器实验继续推进，此项非阻塞。
- **独立发现并修复 KV delta 生命周期缺陷：** 释放开启时，非连续接收缓冲在 preparation 流分配，却在默认流异步写回，下一 chunk 可提前复用缓冲。修复将整个 delta 循环的 launch/wait/commit 和最后一次完成等待统一到选定流；关闭时仍是原默认流。没有加逐 chunk 整卡同步、CPU 权重副本、重试或旧路径兼容分支。该问题不是本次 base-wave 卡死的直接证据。

原验收 benchmark 为 7/8，通过的默认分配器为 4/4；memory 为 4/4、160/160 份记录通过，同尺寸复用的 reserved 增量全部为 0，expandable 的 80 份记录均确实观察到 expandable segment。memory 探针带来的同步和额外分配会改变时序，因此其通过不能排除未插探针的 benchmark 问题。[汇总证据](evidence/review_20261006/gpu_archive_review.json)

**备用 KV 仍驻留的字节数：** 原默认分配器 DeepSeek ON 的 rank0，备用 TP4 为 **70,778,880 B**，备用 TP2 为 **141,557,760 B**；其他 rank 的原始记录仍在结果包内。本次修复不释放备用 KV。

### 验收工具改动

- ordering 删除旧的小载荷轮询判据，分为数据顺序与重叠两种 workload，packing 开/关各三轮。顺序测试包括延迟 re-poison、两个不同列区间的 chunk、local/remote、非连续写回及逐元素 oracle，并要求三轮内至少实际复用过接收 storage。
- 重叠测试每个逻辑张量为 4096×4096 FP32（64 MiB），预分配前台输出；用已有 CUDA 事件记录迁移区间与前台开始/结束时间，要求每个 rank 在 NCCL 区间内有前台完成。只在 launch 前做 Gloo rendezvous。事件区间证明前台进展，不能当成 Nsight 的逐 kernel 并行证据。
- 每轮传输、写回完成后，通过 Gloo 汇总检查结果，再决定继续或共同失败。每个 rank 都保存失败检查名称，禁止在对端断言失败后进入下一轮 NCCL。该机制处理正常校验失败，不伪装成 CUDA/NCCL 致命错误的恢复器。
- 新增测试专用 `WEIGHT_CHECK_AUDIT=1`，默认关闭，限 synthetic + release ON 且 memory audit OFF。它延迟 rank1 的有限值检查，运行实际 benchmark，要求观察到有界 decode 并核对对应 KV delta；结果另写 `weight_check_probe.json`，不得用于性能比较。常规验收矩阵显式关闭它。没有新增生产 Python CLI 或改变默认关闭的 result schema。
- 原成功 GPU 结果的 `weight_check_overlap_steps` 全部为 0，新探针专门补足这项覆盖。

### 本地验证

- 最终相关套件 **238 tests + 111 subtests 通过，0 失败、0 跳过**（JUnit 349）。含真实双进程 Gloo：分别注入 rank0/rank1 失败，双方共同退出且不进入下一批；每组进程限时 30 秒。[日志](evidence/review_20261006/cpu_tests.log)、[JUnit](evidence/review_20261006/cpu_tests.xml)、[命令](evidence/review_20261006/cpu_command.json)
- 用冻结旧提交执行同一 delta 回归：两个方向 × 三种类型，释放开启的 **6 个用例按预期失败**，关闭的 6 个通过；修复后 12 个全部通过。这里采用实际 transaction 和确定性的 CPU 流/复用模型，不是 GPU 复现。[旧版反例日志](evidence/review_20261006/old_delta_counterexample.log)、[复现代码与退出码](evidence/review_20261006/old_delta_counterexample.json)
- 默认关闭时，实际 coordinator 的固定时钟完整结果与 main `f7d5559` 相等，也与上次保存的 fixture 相等，释放调用数为 0；标准序列化 SHA256 仍为 `9359fbff7887cd7a72b6562a166df5934da1483a7932c0c9fbdefa374722a2c4`。[证据](evidence/review_20261006/default_parity.json)、[完整结果](evidence/review_20261006/default_result_fixture.json)
- 已执行 `uv run ... isort`、Python/Bash 语法检查、`git diff --check`。两个服务器脚本的 Python 部分已拦截全部进程调用做 dry-run，验证只选择目标 case、保留原参数、正确记录失败并继续剩余验收；本机启动 GPU 进程数为 0。[脚本验证](evidence/review_20261006/replay_dry_run.json)

### SL3060 复验

更新到本次交付 SHA 后执行。两个入口都从原验收目录读取已记录的命令，写入新的 `/data/ubuntu/lxh/weavetp/acceptance/` 子目录，不覆盖旧结果。原验收目录需保留。

```bash
source /data/ubuntu/lxh/weavetp/env.sh
# 默认分配器：synthetic/DeepSeek 开关各一次，再运行 pending-check 和 ordering。
bash "$REPO/documents/standby_weight_storage_20261005/run_default_recheck.sh"

# 非阻塞诊断：只重跑 expandable_deepseek_on，保留原来的四次切换。
bash "$REPO/documents/standby_weight_storage_20261005/rerun_expandable_deepseek.sh"
```

默认复验要求 6 个 case 退出码为 0，原 logits 校验通过，pending-check 的两个检查通过，ordering 两份结果各包含 12 轮且全部通过。expandable 单独记录实际运行提交与原提交，启用 TRACE_BUFFER_SIZE=100000、DUMP_ON_TIMEOUT、CPP_STACK、TIMING、DESYNC_DEBUG，dump 前缀使用 PyTorch 2.11 的 `TORCH_FR_DUMP_TEMP_FILE`，全部写入新目录的 `flight/`。即使单次通过，也不能宣称已经修复此前偶发卡死。请回传整个新目录。

## 2026-10-05：首次交付记录

以下为首次交付时的历史记录；当时尚无 GPU 结果。KV identity 与 storage release 分为独立提交，两个开关均默认关闭。

## 实现范围与审阅结论

- `KV_REQUEST_IDENTITY=1` / `--live-kv-request-identity`：独立提交 `e75e353`，见 [identity 说明](kv_identity.md)。权重释放不依赖此开关。
- `RELEASE_STANDBY_WEIGHTS=1` / `--live-release-standby-weights`：启动完成原有双布局初始化/校验后，释放备用 TP4 权重 storage。以后每次目标布局重新分配同一 storage，立即对所有权重 Parameter 做 `fill_(nan)`；保留 Parameter、形状、stride、storage offset、共享存储关系、分片属性与现有计划。
- 没有 CPU 权重备份、Parameter 替换、按旧地址复用权重内容或异常恢复路径。按唯一 storage 去重释放，禁止与活动布局或 KV 存储交叠。释放后 `memory_allocated` 应下降；缓存分配器的 `memory_reserved` 可以保持不变，释放的空间供本进程复用，不调用 `empty_cache()`。
- 原来的完整权重 + KV prefix 迁移、前台源模型 decode、KV delta、双模型 logits 校验路径保留。全部传输/非连续视图写回及原双模型校验完成后，才清理 persistent packing 缓存并释放最终非活动布局。`repeat-forward` 释放 TP4，普通交替切换释放刚退出的布局。
- storage 在原分配流重建。后台 preparation stream 等待分配流之前的工作，再初始化 NaN、准备源切片。copy/comm 流通过事件等待 preparation；packing 继续使用原 pack-end/unpack 完成依赖。没有新增整卡同步。
- 全权重有限值检查在后台流按至多 1 Mi 元素的视图分块扫描，能发现从未参与本轮 logits 的冷专家漏传。检查期间有界推进源 decode，产生的新 KV 全部纳入 delta；剩余等待计入 cutover。对应增加开启模式的 KV 容量上界。检查不是数值正确性的替代：原 logits 校验与测试中的独立逐元素 oracle 仍保留。
- 开启时，解析后的 `ACTIVE_EXPERT_PHASES` 或 `PRESSURE_RANK_PHASES` 超过一组即拒绝，包括重复的组。`flying-serving-proxy`、`llumnix-proxy` 不走完整权重迁移，因此直接拒绝此组合。没有增加兼容回退分支。

主要生产改动仅在 live benchmark 生命周期、`standby_weights.py` 与 NCCL launch 的可选 producer event 依赖。未修改中央 planner、异步 transaction 实现或既有调度策略。其他新增文件是针对本任务的测试及验收入口。工作区原有的文档移动/删除不包含在提交中。

## 备用 KV 仍驻留的字节数

本次不释放、不重建备用 KV storage。开启时 `result.json` 单列每个 rank 的实际 storage 字节数：

| 字段 | 含义 |
| --- | --- |
| `standby_weight_storage.ranks[r].initial_standby_kv_bytes` | 启动释放 TP4 权重后，备用 TP4 KV 仍驻留的字节数 |
| `standby_weight_storage.ranks[r].switches[i].standby_kv_bytes` | 每轮释放后，实际非活动布局 KV 仍驻留的字节数 |
| `...initial_release.weight_storage_bytes` / `...switches[i].release.weight_storage_bytes` | 去重后本轮释放的权重 storage 字节数 |
| `...release.weight_allocated_freed_bytes` | 清理 packing 后，释放权重本身导致的 allocated 下降量 |
| `...release.packing_allocated_freed_bytes` | 单独记录的 packing 缓存释放量 |

字节数通过真实 tensor storage 统计，未用逻辑切片大小推算。当前无 GPU 实测，不能填写 synthetic 或 DeepSeek 的 GPU 字节数；用户回传结果后再填实际值。关闭时不添加这些字段。

## CPU 验证与默认路径证据

环境：Windows、现有 Python 3.13 / PyTorch 2.14.0+cpu。使用隔离 conftest 的相关 CPU 套件，未运行需要 CUDA 或全仓数据集环境的训练测试。

- **189 项测试 + 111 个子测试通过，0 失败，0 跳过**；JUnit 计数为 300。7 条 warning 来自现有可选 Apex/Transformer Engine/absl 缺失及弃用提示。[原始日志](evidence/cpu_tests.log)、[JUnit](evidence/cpu_tests.xml)、[完整命令](evidence/cpu_command.json)。
- 覆盖 FP32/FP16/BF16、共享/绑定 Parameter、非零 offset、非连续视图、stride 1/2/3、四轮真实 planner + transaction 双向迁移、冷专家/整权重/切片漏传、有限但错误数据的独立 oracle、多 phase 拒绝、释放时点、repeat-forward、两个开关组合、校验新增 KV delta 与容量边界。
- 实际 benchmark 协调函数在 CPU 模型与传输替身上运行，logits 替身同时依赖权重和完整 KV 历史，固定时钟用于精确比较结果。CUDA 流事件使用替身的测试只证明依赖的提交逻辑，不能替代真实 GPU 顺序验收。
- 基准固定为 main `f7d555998b45756998465924d733fd70e553d6b4`。关闭两个开关后，原 token 公式/路径、KV 命名及分片元数据、顶层 result 字典字段与计算 AST 均与基准一致；四轮实际协调函数生成的完整结果字典相等，释放调用数为 0。见 [对照证据](evidence/default_parity.json) 和 [固定计时的完整结果](evidence/default_result_fixture.json)。两份结果序列化 SHA256 同为 `9359fbff7887cd7a72b6562a166df5934da1483a7932c0c9fbdefa374722a2c4`。这不是声称不同 GPU 启动的墙钟数值相同。
- 已按仓库要求执行 `uv run ... isort`；Python 语法、三个 shell 入口语法、验收矩阵 dry-run、`git diff --check` 均通过。[语法证据](evidence/syntax_checks.json)。初次工具准备遇到 uv 默认下载解释器失败，改为显式使用现有解释器后成功；早期 CUDA 流替身缺少 priority 参数的两项测试失败已经修复，最终整套回归通过。

## SL3060 GPU 命令

使用新提交的 `code/current/tools/resharding/run_standby_weight_acceptance.sh`。目录必须是新的绝对 `/data/...` 路径，入口拒绝覆盖旧结果。每个 case 都独立启动进程，记录完整命令、退出码、console.log、benchmark 的 result.json/run.log；根目录记录 commit、GPU 与 torch/CUDA/NCCL 版本。`--dry-run` 只打印命令。

本次是单机 8 卡 TP2↔TP4 功能验收，EP=2，四轮交替切换，identity 固定关闭。带宽画像缺省使用已有 benchmark 的均匀 100 Gbps 矩阵，故这些结果不作为正式调度收益结论。DeepSeek 使用已有 checkpoint 和原来的 BF16-relative logits 阈值；synthetic 使用 allclose。

先将 SL3060 的仓库更新到交付消息中的第二个完整 SHA。若仍需要 SL3061 中转，沿用已有路径：

```bash
# SL3061：取得代码；不启动 GPU
source /data/ubuntu/lxh/weavetp/env.sh
git -C "$REPO" fetch origin codex/standby-weight-storage

# SL3060：经局域网取分支；TARGET 替换为交付消息中的第二个完整 SHA
source /data/ubuntu/lxh/weavetp/env.sh
TARGET='<第二个提交的完整 SHA>'
git -C "$REPO" fetch "ubuntu@10.60.14.2:$REPO" refs/remotes/origin/codex/standby-weight-storage
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = "$TARGET"
git -C "$REPO" checkout --detach "$TARGET"
```

完成代码更新后，在 SL3060 执行：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export PYTHON CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2
unset PROFILE
RUN="/data/ubuntu/lxh/weavetp/acceptance/standby_$(date -u +%Y%m%dT%H%M%SZ)_$$"
SCRIPT="$REPO/code/current/tools/resharding/run_standby_weight_acceptance.sh"

# synthetic、DeepSeek × 开/关 × 默认/expandable 分配器，共 8 次
bash "$SCRIPT" benchmark "$RUN/benchmark"

# 独立进程：真实模型每次释放后的同进程显存复用，共 4 次
bash "$SCRIPT" memory "$RUN/memory"

# 独立 2 GPU 进程：延迟 NaN 初始化、local/remote、非连续切片、packing 开/关
bash "$SCRIPT" ordering "$RUN/ordering"
printf '结果目录：%s\n' "$RUN"
```

三个模式的展开命令：[benchmark 8 组](evidence/benchmark_dry_run.txt)、[memory 4 组](evidence/memory_dry_run.txt)、[ordering 2 组](evidence/ordering_dry_run.txt)。本地只运行了 dry-run，没有执行上述 GPU 命令。

## GPU 通过条件及回传内容

1. 所有 `exit_code.txt` 为 0，benchmark 四次切换全部完成，原 logits 校验通过；ON 的 `all_weights_finite` 全部为 true，并核对逐 rank 释放量、备用 KV 字节数及 decode 的 TPOT/overlap 指标。
2. memory 模式在活动模型、两套 KV 都仍存活的同一进程内，每轮释放后按原各 storage 大小分配 uint8 临时缓冲区，合计字节数严格等于释放字节数；没有 allocator 预热或清缓存。`memory_reserved` 不增长，删除临时缓冲后 allocated 恢复，真实权重 allocated 下降量至少覆盖 storage 字节数。expandable 模式还要求 memory snapshot 实际观察到 expandable segment。每个 case 应有 8 ranks × 5 releases = 40 份 `reuse_rank*_*.json` 且 `passed=true`。
3. 单个“总字节数大小”的大缓冲另作诊断，可能受碎片影响增长或 OOM，不冒充同尺寸 storage 复用的硬验收。其结果完整保留。memory 模式含显式探针同步，其 TPOT/耗时不能当性能结果。
4. ordering 模式两种分配器各两份 `ordering_rank*.json`，每份含 packing 关/开各三轮，目标数值逐元素相等，而且在迁移未完成时确实有前台计算完成。这里的前台是矩阵计算；真实 MoE decode 由 benchmark 负责验证。
5. 回传整个 `$RUN`，包括失败 case 的日志；当前验收只承诺本地 CPU 证据，最终 GPU 正确性、实际显存量与前台延迟结论需据这些真实结果审阅。
