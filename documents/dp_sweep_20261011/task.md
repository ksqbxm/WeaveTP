# WeaveTP DP 规模敏感性实验任务

目标：用同一套实验代码评估 DP 增大、可选源副本增多时，带宽感知选源对迁移时间的影响。主指标是 `base.wall_s`，辅助指标是 `transport_s`。结果必须保留不满足实验预期的证据，不能预设性能结论。

## 基线与分步交付

- 已完整阅读 `CODEX.md`、`code/current/AGENTS.md` 和贡献规则。从 `7deac20feaf76ee4e6a0c53b2289271a903dd12c` 创建独立 worktree，分支为 `codex/dp-sweep`；不携带当前工作区改动，`main` 冻结在 `f7d5559`。
- 第一步只做改动 2（launcher）、3（观测）、4（汇总）。提交、推送后单独报告 SHA S1，停止等待用户确认。
- 第二步做改动 1（TP 参数化和随机初始化），基线为 S1；补全 README 和独立服务器命令，提交、推送后报告 SHA S2。
- 全程不连接实验服务器、不运行 GPU。选源、执行、KV 算法和门槛不变。第一步不提前启用 TP1、层数覆盖或跳过 checkpoint。
- 默认不传新增参数或环境变量时，与各步基线的命令、默认日志及结果逐字节一致；CPU 固定时钟替身、T09 检查链和 dry-run 提供证据，不将 CPU 证据称为 GPU 验证。

## 第一步

### Launcher

通用 launcher 使用 `WORLD_SIZE=${WORLD_SIZE_OVERRIDE:-$((NPROC_PER_NODE * NNODES))}` 并校验正整数，用于默认 GBS 和 16 卡画像门禁，保留 torchrun 启动方式。DeepSeek launcher 的 NPROC_PER_NODE、EXPERT_PARALLEL_SIZE 分别可覆盖，默认仍为 8、2。

**SRC_TP、DST_TP、NUM_LAYERS、SKIP_CHECKPOINT 仅透传环境变量，不生成尚未实现的 CLI 参数。** 显式 SRC_TP=2 与不设置新变量时，旧命令行均须逐字节一致。第一步验证 W=10 不触发 16 卡门禁；W=10、SRC_TP=1 时 GBS=10 的组合验收属于第二步。

### 观测

新增 `DP_SWEEP_METRICS=1` / `--live-dp-sweep-metrics`，默认关闭。关闭时不新增输出字段、日志、collective、显存统计调用。新 CLI 默认项不出现在默认参数打印中。

- `transport_s = math.fsum(base.wave_records[*].transport_s)`，沿用已有逐波跨 rank MAX 与 fallback，**只对应 Base 阶段**。
- `base_remote_bytes`、`delta_remote_bytes` 分别统计实际执行 Base、KV Delta 计划的接收侧网络字节；`remote_bytes = base_remote_bytes + delta_remote_bytes`。
- `weight_bytes + kv_bytes = remote_bytes`，二者均排除同卡复制。`local_copy_bytes` 单列，同卡不计入网络字节。
- `cross_node_bytes` 根据实际 hostname 统计；`max_send_bytes_per_rank`、`max_recv_bytes_per_rank` 为整次切换累计网络字节的 rank 最大值。发送镜像不得重复计数。
- `peak_mem_bytes` 使用各 rank `torch.cuda.max_memory_allocated()` 的最大值；`peak_reserved_bytes` 使用各 rank `torch.cuda.max_memory_reserved()` 的最大值。
- 原 switch_start 前 reset_peak_memory_stats，原 switch_wall_s 计算后读取。**区间包括切换后的数值校验**，也包含目标权重重建、迁移及原路径内的释放；不改变任何原计时边界。
- 元数据扫描与跨 rank 汇总放在全部切换之后。逐波 elapsed_s/transport_s/tasks/bytes、base.wall_s、switch_wall_s、plan_variant、base.execution_mode 原样保留。

### 汇总

新增 `tools/resharding/summarize_dp_sweep.py`，接受 --root、--direction（2->1 或 2->4）、--methods（三种已知方法的任意非空子集）。默认 S 三方法，E default 和 weavetp。读取 `<root>/<method>/W*/r*/node0/result.json`，只取指定方向。

每个 launch 内先平均，再跨 launch 等权计算均值和样本标准差 ddof=1。单 launch 标准差未定义；缺字段、无目标方向、零分母不补零。输出按方向命名的 CSV、Markdown，保留要求的所有指标和计划字段，增加 Base/Delta 网络字节及 reserved 峰值。

主表突出 base.wall_s、transport_s、wave_overhead_s、peak_mem_bytes，其中 `wave_overhead_s = base.wall_s - transport_s` 逐切换计算后聚合，不除以 waves，不截断负值。时间、开销和显存指标给出相对 default 的改善；另给 weavetp_2x 相对 weavetp 的比较。公式为 `100 * (参照均值 - 当前均值) / 参照均值`，先求两级均值再计算比值。

按同 W、同名 launch、目标方向切换序号检查原始记录。B/V/X 分别表示 default/weavetp/weavetp_2x 波数：扩容 abs(B-2V)<=1；缩容 abs(B-V)<=1，abs(X-V/2)<=1；若缩容只有 default 和 2x，则 abs(X-B/2)<=1。扩容显式选择 2x 时，其与 weavetp 波数应相同，且应用 default 的二倍关系。

**缩容 default 与 weavetp 的波数差值必须记录，即使在 1 波容差内；超过 1 波标记异常。不得把波数差异造成的性能变化全部归因于选源。** 原始 execution_mode 必须相同，不合并 baseline/fifo 标签。default 的 plan_variant 应为 baseline，另两个应为 candidate；否则醒目标出原始标签、fallback 和 cached-plan 全局门槛证据。缺比较方法为不适用；缺配对记录为无法核验。异常或证据不完整时保留报告并非零退出。

## 第二步

- --live-src-tp 有效默认 2，--live-dst-tp 默认 4，仅支持 (1,2)/(2,4)。源 TP 与启动 TP/ETP 一致，world size 满足 TP/EP 整除。
- 模型、计划、服务、tracker、scheduler、weights 和生命周期状态按实际 TP 索引；日志和方向为实际 src->dst。扩缩容和方向上限按大小比较，辅助方向判断只作等价参数化。
- 并行组验证实际成员、互惠关系、源组嵌套及目标 TP 不跨节点，支持单节点和 8+k；保持默认输出。
- launcher TP/ETP 使用 SRC_TP，GBS 默认 MICRO_BATCH_SIZE*WORLD_SIZE/SRC_TP。
- DeepSeek 支持 SKIP_CHECKPOINT=1/--live-skip-checkpoint 与 NUM_LAYERS，默认 27 层且必须加载真实 checkpoint。源模型用 Megatron 默认随机初始化；目标模型沿用初始同步。
- **九层结构为第 0 层 Dense MLP、第 1–8 层 MoE，保持原始层类型顺序。** 新实验结果记录 weights_random_init、实际层数、按层序的 layer_types、去除 DP/备用布局重复后的参数量。
- **TP1<->TP2 必须验证 standby 权重释放、重建、按 TP 键索引、迁移后数值校验及旧活动权重释放。第二次 2->1 必须重建已释放的 TP1 权重并发生真实数据迁移。** CPU 检查真实协调流程、非空远程计划及恢复后的内容，不仅检查调用次数。

## 第二步 README 实验配置

此处是待实现的实验规范，不是第一步可运行的服务器命令；完整 README 示例第二步再补。

| 方法 | MAX_WAVE_TASKS | EXPANSION_MAX_WAVE_TASKS | SHRINK_MAX_WAVE_TASKS |
|---|---:|---:|---:|
| megatron_default | 256*W | 256*W | 256*W |
| weavetp | 256*W | 512*W | 256*W |
| weavetp_2x | 512*W | 512*W | 512*W |

- default：METHOD_VARIANT=baseline SCHEDULER_MODE=baseline DISABLE_SOURCE_REROUTE=1 ADAPTIVE_HYBRID=0 ALLOW_AWARE_SHRINK=0。
- 两个 WeaveTP：METHOD_VARIANT=moetp++-hybrid SCHEDULER_MODE=residual DISABLE_SOURCE_REROUTE=0 ADAPTIVE_HYBRID=1 ADAPTIVE_RESIDUAL_MAX_WAVES=4 ALLOW_AWARE_SHRINK=1 REROUTE_MIN_GAIN_PCT=0.0。
- T09 门槛：REROUTE_MIN_CONTENTION_GAIN_PCT=0.0 REROUTE_MIN_GLOBAL_GAIN_PCT=5.0 REROUTE_PENALTY_US=20.0 REROUTE_MIN_BYTES=1048576。
- 全部示例显式 RELEASE_STANDBY_WEIGHTS=1；共同 EP=1、SEQ_LENGTH=1024、MAX_POSITION_EMBEDDINGS=1024、PROMPT_TOKENS=8、MAX_WAVES=128、MAX_OVERLAP_STEPS=1、DP_SWEEP_METRICS=1，画像匹配 W 和 rank 排布。
- S：SRC_TP=1 DST_TP=2 SKIP_CHECKPOINT=1 NUM_LAYERS=9 SWITCHES=2，W=2/4/6/8/10/12/14/16，三方法，汇总 2->1。首轮 1->2 的“零传输”指零远程传输，保留本地复制和真实计时，不强制耗时为零。
- E：SRC_TP=2 DST_TP=4 SKIP_CHECKPOINT=0 CHECKPOINT=<真实权重> NUM_LAYERS=27 SWITCHES=1，W=4/8/12/16，只跑 default、weavetp，汇总 2->4。weavetp 与 2x 有效扩容配置相同（通用上限被 512*W 方向上限覆盖），不重复运行 2x。

## 验证与交付要求

每步执行 T09 完整 CPU 测试链及三类 dry-run、相关既有回归、新增功能测试、Python 编译、bash -n、修改导入后的 uv run isort 和 git diff --check。历史保护检查通过新的阶段验收器按本步基线核验，不修改历史证据。

第一步覆盖纯环境透传、显式 SRC_TP=2、override/门禁、Base/Delta 守恒、同卡分离、两种显存峰值及校验区间、三方法子集、两级统计/改善、波数容差边界和所有异常状态。第二步覆盖全部卡数/TP 对、8+k、方向上限、W=10/SRC_TP=1 的 GBS=10、九层顺序、默认真实 27 层、同步和 TP1<->TP2 生命周期。

固定环境/输出路径/CPU 时钟，比较基线和修改版的默认命令、日志、dry-run、序列化结果原始字节，记录双方 SHA-256，不删字段、不排序或忽略差异。每步交付完整 SHA、独立提交区间、逐文件行号和增删行数、CPU 结果、比对证据、推送结果；第一步交付后停止。
