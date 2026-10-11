# DP sweep 初始权重同步分批修复

基线：`397ae279143e1705f05257988b7b8aa0bdac439d`；交付分支：`codex/dp-sweep`。仅修改初始同步，未连接实验服务器，未运行 GPU。完整 CPU 验证通过；这不等于已在 NCCL 实机上复测超时。

## 实现与审阅结论

原初始同步将整个权重计划交给一次 coalesced P2P 提交。新增 `INIT_SYNC_MAX_TASKS` → `--live-init-sync-max-tasks`：正数时，以原有模型解包、专家数和 bandwidth policy 构建同一选源计划，复用 `collect_transfer_tasks` 的全局有序任务表和 `filter_plan_by_task_ids`，连续切片逐批 launch → wait → commit。没有重试、超时兜底、任务丢弃或最后一批突破上限的分支。

每批复用现有 NCCLCopyService 的全 rank 参与标记，避免没有权重任务的 rank 跳过 NCCL 提交而使序号错位；标记不属于模型任务。设置 producer_stream 使用现有就绪事件，保证建模/加载和发送切片准备先于异步通信。transaction 的设备同步保证非连续接收的延迟写回完成，随后在全 rank Gloo control_group 上 barrier，再进入下一批。

默认有效值仍为 0，直接保留原 `swap_model_weights` 调用，不添加日志或结果字段。argparse 使用 SUPPRESS；launcher 在未设置或显式 0 时不生成新 CLI 参数。此原调用是用户明确要求保留的默认语义，启用分批后没有另一条兼容/回退路径。初始同步没有新增 benchmark 计时；后续 TP 输出一致性校验、正式迁移和统计边界保持原样。

| 审阅范围 | 结论与证据 |
|---|---|
| 初始同步任务与提交 | 真实 central planner / metadata / collect / filter / transactions，CPU mailbox 通信；BF16 模型包装正确解包，任务恰好一次，按 id 顺序分批。4,104 个任务得到 2048/2048/8 三批，空任务 rank 也经历提交和 barrier。 |
| 写回及初始数值校验 | 非连续目标由 NaN 恢复为独立参考布局的精确分片；实际协调器先分批同步，再执行原 `_validate_cutover`，随后 standby 释放与四次正式切换校验通过。 |
| 九层结构、TP 键及生命周期 | 全量回归重跑现有实际 Megatron 层规格、TP1↔TP2 各 W 的释放/poison/重建/迁移和缺失分片反例；保留第 0 层 Dense、1–8 层 MoE。 |
| launcher / 8+k / 公共配置 | 两入口的默认和显式 0 命令不变，正数正确透传，非法值在启动 Python 前拒绝。README 公共模板（服务器保存为 dp_sweep_env.sh）增加 `export INIT_SYNC_MAX_TASKS=2048`，32 组 S/E 方法×W 配置通过。 |
| 观测与汇总 | Base transport 的每波跨 rank MAX 累加、Base/Delta 接收侧字节、同卡复制、校验后显存峰值，以及 launch 内/跨 launch 平均和 ddof=1 未修改；相关 CPU 回归通过。 |
| 正式迁移、选源、KV、计时、数值校验 | 整个 benchmark 的既有函数/类 AST 与基线相同，协调器只允许新增参数检查和替换初始同步调用；Megatron 核心、T09 历史检查链和 formal 汇总原文件全部未变。 |

审阅还确认两个**已有边界**，依本次“只改初始同步”的范围保持不变：

- 正式迁移 baseline/FIFO 及 residual 达到 MAX_WAVES 的最后一波时，可能合并全部剩余任务，任务上限不是该边界下的绝对保证。代码位置为 benchmark `_run_async_waves` 和 core/live.py `ResidualBandwidthWaveScheduler.next_wave`。初始同步的新循环不复用这项波数截断逻辑，任何一批都不超过 INIT_SYNC_MAX_TASKS。
- 自适应缩容的 baseline/FIFO 标签与缓存感知计划几何可能不同，且与 megatron_default 的 execution_mode 标签不同；README 和汇总异常检查已明确保留这些差异。不能将波数或执行模式带来的全部收益归因于选源。本次未改标签、算法或门槛。

## 文件与行数

相对 397ae27 的功能、测试和运行文档变更：

| 文件 | 位置 | 增加 / 删除 |
|---|---|---:|
| code/current/examples/rl/benchmark_live_moe_tp.py | 224 参数；1350 分批 helper；1381 参数检查；1608 初始同步调用 | +50 / -10 |
| code/current/tools/resharding/run_live_moe_tp_benchmark.sh | 46 环境变量校验；227 opt-in CLI | +8 / -0 |
| code/current/tests/unit_tests/resharding/test_init_sync.py | 35 真实计划与大批量；148 默认回归；177 校验/释放；215 全函数 AST；234 launcher | +246 / -0 |
| code/current/tools/resharding/README_dp_sweep.md | 28 语义；35 公共配置 export | +3 / -0 |
| documents/dp_sweep_20261011/server_commands.md | 9 更新服务器公共配置的说明（未执行命令） | +2 / -0 |
| documents/dp_sweep_20261011/check_cpu.py | 30 回归注册；231 编译检查 | +2 / -2 |

本报告和 `init_sync_evidence/` 另行保存审阅、原始输出和验证回执，不修改历史证据。

## 验证

- 新增定向回归 **13/13**（包含在以下 420 项中）：精确一次、上限、顺序、空 rank、延迟写回、BF16 包装、原数值校验、默认 0、非法参数、两 launcher 和 AST 范围保护。
- T09 检查链 **181/181**；相关回归 **420 passed，194 subtests passed**，0 失败、0 跳过，104.39 秒。[总回执](init_sync_evidence/validation.json)、[pytest 日志](init_sync_evidence/regressions.stdout.log)、[JUnit](init_sync_evidence/regressions.xml)。
- **21/21 默认原始字节比对一致**：12 组 launcher（两个入口、默认/node0/node1、显式 SRC_TP=2）、8 组实际协调器 CPU console/JSON（synthetic/默认 DeepSeek、release 关闭/开启）、1 组真实 Megatron 参数打印。[原始长度、SHA-256 和结果](init_sync_evidence/default_parity.json)。另有定向回归验证显式 INIT_SYNC_MAX_TASKS=0 的旧调用与原始输出。
- **3/3 T09 dry-run 原始字节一致**：all 34,787 字节、only_weavetp_r1 3,869 字节、r1 11,599 字节；没有换行归一化或 JSON 重序列化。[回执](init_sync_evidence/t09_parity.json)。
- README 配置 **32/32** 和 Bash 语法检查通过。[回执](init_sync_evidence/readme_config.json)。Python 编译、两个 launcher 的 bash -n、uv run isort --check-only、git diff --check 通过。

开发时定向测试先因未实现分批而失败，并修正了测试 fixture 的 num_experts 元数据不一致；最终重跑以上完整检查链全部通过。CPU 环境的 Transformer Engine/Apex/absl 缺失等 warning 不属于测试失败。默认行为的逐字节证据来自 CPU 替身和 dry-run，不声称两次真实 GPU 时序数值能逐字节复现。

复验（输出目录必须在仓库外且尚不存在）：

```powershell
python -B -X utf8 documents/dp_sweep_20261011/check_cpu.py --baseline 397ae279143e1705f05257988b7b8aa0bdac439d --output-dir C:/data/dp-sweep-init-sync-recheck
```

运行时更新 README 所定义的 `dp_sweep_env.sh` 并 source 即可得到 `INIT_SYNC_MAX_TASKS=2048`。main 保持 `f7d555998b45756998465924d733fd70e553d6b4`。提交推送仅到个人 GitHub 分支，交付 SHA 后停止，不发起 GPU 实验。
