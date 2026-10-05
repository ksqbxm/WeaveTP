# 备用权重 GPU storage 释放：实现与验收

2026-10-05。已按 CODEX.md 实施最小生产路径改动并完成相关 CPU 回归。尚未连接服务器、运行 CUDA/NCCL 或取得 GPU 验收结果。提交分为独立的 KV identity 与 storage release 两个功能，两个开关均默认关闭。

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
