# WeaveTP DP 规模敏感性：第二步交付

第二步已完成改动 1：支持 TP1↔TP2 / TP2↔TP4、DeepSeek 前九层随机初始化、实际层类型与逻辑参数量记录，并补全 S/E README 和独立服务器命令。未连接实验服务器，未启动 GPU。

## 提交区间

- 基线 S1：`698703bc344eae53f097bcc86a78096f980ecf80`。
- 功能、测试和运行文档：`014d35c5c4afeae2e93e101ad42b7326fc6f6bac`。
- 第二步独立区间：S1 之后至本报告所在提交；分支 `codex/dp-sweep`。
- main 仍为 `f7d555998b45756998465924d733fd70e553d6b4`；原工作区文件未改动。
- GitHub 分支推送单独执行并核对远端 SHA，不属于实验服务器连接。

## 文件、行号与增删行数

以下统计相对 S1；代码行号对应上述功能提交。

| 文件 | 主要行号 / 内容 | 增加 / 删除 |
|---|---|---:|
| code/current/examples/rl/benchmark_live_moe_tp.py | 219 新参数；543 架构校验；604 元数据；717 请求域；1348 协调流程；1476 方向上限；1531 并行组；1828 权重/KV TP 索引；2277 释放；2513 新结果 | +131 / -74 |
| code/current/megatron/core/resharding/live.py | 114：原方向判断按 TP 大小比较，其他策略不变 | +1 / -1 |
| code/current/tools/resharding/weavetp_observations.py | 47：参数化核查；99：读取实际组；124：方向 gate；后续 packing 方向 | +41 / -17 |
| code/current/tools/resharding/run_live_moe_tp_benchmark.sh | 32 TP 对；110 层数；122 GBS；129 DeepSeek；221 opt-in CLI；312 TP/ETP | +32 / -10 |
| code/current/tools/resharding/run_deepseek_v2_lite_live_benchmark.sh | 24：skip 时允许不存在的 checkpoint；29 继承变量 | +2 / -3 |
| code/current/tests/unit_tests/resharding/test_dp_sweep.py | 15 基线；55 launcher 回归与新配置 | +70 / -17 |
| code/current/tests/unit_tests/resharding/test_dp_sweep_coordinator.py | 22 原始输出；35 默认 DeepSeek 比对；新组观测 CPU 替身 | +10 / -1 |
| code/current/tests/unit_tests/resharding/test_live_defaults.py | 55：仅规范化重命名后的计划局部变量，再检查默认结果 AST | +8 / -1 |
| code/current/tests/unit_tests/resharding/test_live_storage_lifecycle.py | 30：CPU fixture 支持 TP/W、层元数据与初始张量注入 | +17 / -4 |
| code/current/tests/unit_tests/resharding/test_dp_sweep_groups.py | 1：全部布局、8+k、反例、旧 W16 原始字节比对 | +176 / -0 |
| code/current/tests/unit_tests/resharding/test_dp_sweep_tp.py | 37 无效布局；48 策略方向；69 上限/索引；90 生命周期；146 真实 planner；213 真实层规格；245 参数计数；285 随机源同步 | +339 / -0 |
| code/current/tools/resharding/README_dp_sweep.md | 7 S/E 方法；24 生命周期；26 公共配置；82 单节点；120 汇总；139 原始标签限制 | +139 / -0 |
| documents/dp_sweep_20261011/server_commands.md | 7 双节点 8+k；52 E checkpoint；65 CPU 汇总 | +80 / -0 |
| documents/dp_sweep_20261011/check_cpu.py | 42 按阶段保护核心；93 原始比对；新增测试注册与编译检查 | +44 / -5 |
| documents/dp_sweep_20261011/task.md | 47：记录第二步确认与 S1 基线 | +4 / -0 |

功能提交共 15 个文件，+1094 / -133；本报告和 step2_evidence 为独立的证据提交。第一步汇总脚本、观测计时边界和历史 T09 脚本没有修改。

## 重点验收

九层 preset 强制为原模型前缀 `[dense, moe, moe, moe, moe, moe, moe, moe, moe]`。CPU 测试构造实际 Megatron `get_gpt_decoder_block_spec`：9 层为 1 个 MLP + 8 个 MoELayer，27 层为 1 + 26；错误顺序、错误长度、层数越界和未提供 checkpoint 均被拒绝。源模型继续走原 Megatron 建模/初始化调用；skip 时不调用 load_checkpoint，目标仍执行原初始同步。

结果中的层类型读取实际 decoder，参数量只计源模型，按 dense/expert 的 TP 和 DP 组去重，不把备用布局算入。TP1/2 × EP1/2 的 CPU 参数计数用独立期望值验证。

TP1↔TP2 生命周期采用两层证据：

1. 执行实际 benchmark 协调器，CPU 模型和传输替身验证按 TP 键取权重，顺序为释放 TP2 → 重建/poison TP2 → 校验并释放 TP1 → 重建/poison TP1 → 校验并释放 TP2；第二轮恢复 TP1 的精确内容，旧活动权重在数值校验之后释放。
2. 在全部八种 W、FP32/BF16 下使用真实集中式 planner、元数据提取、reshard transactions 和 StandbyWeights，仅把通信换成 CPU mailbox。1→2 的远程接收为零且有同卡复制；2→1 的远程接收非空，已释放并填 NaN 的 TP1 精确恢复。故意删除一个远程分片时 finite 校验失败。这里没有把同卡复制或真实耗时伪造为零。

两个 TP 对的全部卡数组合还验证方向字符串、显式/回退波任务上限、tracker TP 键、源 TP/ETP 一致性、整除约束、嵌套和目标组不跨机。W10/SRC_TP=1 的 GBS=10，且不触发 W16 画像门禁。

## CPU 与静态验证

- 原 T09 完整检查链：**181/181**，10 个套件，0 失败、0 跳过。
- 全部相关回归：**407 passed，194 subtests passed**，0 失败；212.68 秒。[日志](step2_evidence/regressions.stdout.log)、[JUnit](step2_evidence/regressions.xml)。
- 生命周期/层结构定向验收：**93 passed**，0 失败、0 跳过，是上面回归的子集，不重复计数。[记录](step2_evidence/lifecycle/receipt.json)。
- README/服务器文档的 9 个 Bash 段、两个 launcher 的 bash -n、32 个 S/E 方法×W 配置检查通过。[记录](step2_evidence/doc_checks.json)。
- Python 编译、uv run isort --check-only、工作区和暂存区 diff --check 通过；15 个功能/文档文件保留工作区与 Git blob 指纹。[最终静态记录](step2_evidence/final_static.json)。
- 核心保护允许且只允许 live.py 中一个方向表达式的等价推广，采用整文件 AST 精确比较；其他 Megatron 核心文件、历史 T09/T085/T08、原 formal 汇总相对 S1 无变化。选源、执行、KV 算法和所有门槛保持原样。

开发时旧 CPU fixture 未覆盖新增组观测的设备查询，曾在 CPU-only Torch 上报错；补充组观测替身后，实际观测核查由独立 CPU 测试覆盖。另有工作目录错误导致一次文件读取失败、一次 pytest 未收集到目标文件；修正工作目录后复跑通过。CPU 环境的 Transformer Engine/Apex/absl 缺失 warning 不属于测试失败或跳过。

复验命令（输出目录须为仓库外尚不存在的目录）：

```powershell
python -B -X utf8 documents/dp_sweep_20261011/check_cpu.py --baseline 698703bc344eae53f097bcc86a78096f980ecf80 --output-dir C:/data/dp-sweep-s2-recheck
```

[总体回执](step2_evidence/validation.json) 的 ok=true；gpu_started=false、server_connected=false。本地 CPU Gloo 测试不连接实验服务器。

## 与 S1 的逐字节比对

[default_parity.json](step2_evidence/default_parity.json) 的 **21 组全部一致**，保留双方原始 .bin 和 SHA-256：

- 12 组 launcher：两个入口 × 默认/node0/node1 × 未设置新变量/显式 SRC_TP=2。
- 8 组实际协调器 CPU 日志和 _write_results 文件：synthetic/默认 DeepSeek × standby release 关闭/开启 × console/JSON。
- 1 组真实 Megatron 默认 live 参数打印；新增默认项被 SUPPRESS，不进入 Namespace。

[T09 三组 dry-run](step2_evidence/t09_parity.json) 直接比较 stdout 原始字节，未归一化换行、删字段或重新序列化：

| dry-run | cases | 字节 | 双方相同的 SHA-256 |
|---|---:|---:|---|
| all | 9 | 34787 | b4a09eb0e27c2a5ee3721c6fe3bccfbec4bfcec524db3622b3daf6f4f2ed6d75 |
| only_weavetp_r1 | 1 | 3869 | b89623b10ab75cd0529b7925858d611d08835fd53781b0e3eb0f1e8179c8f234 |
| r1 | 3 | 11599 | 3ca20a9903cc993306991a4b1b3623c35be5d6333fd2181bf870965b1e1decc4 |

另有完整 W16 并行组及四次切换观测序列化结果，277,964 字节完全一致，SHA-256 为 `54f79a7b5142059ac459c35f395d94a952dd9eb3f94f3ef2d9b9eb08bc77fbab`。[回执和原始文件](step2_evidence/legacy_observations/receipt.json)。测试还检查旧方向下 16 组 gate/packing 组合的原始序列化字节。

这些证据证明固定 CPU 模型、时钟及 dry-run 下的默认兼容；没有声称真实 GPU 的非确定性计时逐字节相同，也没有生成性能结论。

## 运行示例及解释边界

完整配置见 [README](../../code/current/tools/resharding/README_dp_sweep.md)，可人工运行的双节点命令单列在 [server_commands.md](server_commands.md)。S 三方法、E 两方法，全部显式设置 RELEASE_STANDBY_WEIGHTS=1 和 DP_SWEEP_METRICS=1；保留用户指定的 W2 配置和 T09 门槛。

既有自适应缩容返回 baseline/FIFO，而缓存缩容计划仍可能采用带宽感知选源；default 的 execution_mode 标签为 baseline。这些原始标签会触发汇总中的 candidate/execution_mode 检查。没有为了通过断言重命名标签或修改策略，README 已明确解释。波数、调度模式与选源都有可能影响耗时，不能把差异全部归因于选源。
