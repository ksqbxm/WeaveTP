# WeaveTP：DP 规模敏感性实验

第二步基线为 `698703bc344eae53f097bcc86a78096f980ecf80`。本分支只完成代码和 CPU 验证；下面是供后续人工运行的 GPU 实验命令，本次没有连接服务器或执行这些命令。

DeepSeek 入口默认不设置新增变量时，仍使用 TP2→TP4、27 层和真实 checkpoint；通用入口默认仍为两层 synthetic 模型。新实验启用 `DP_SWEEP_METRICS=1`；`--live-src-tp` 仅在 SRC_TP=1 时由 launcher 添加，默认参数打印和输出保持兼容。

## 实验与方法

| 实验 | 卡数 W | 源/目标 TP | 权重与层数 | 切换次数 | 汇总方向 |
|---|---|---|---|---:|---|
| S（先做） | 2/4/6/8/10/12/14/16 | 1/2 | 随机初始化，9 层 | 2 | 2->1 |
| E（后做） | 4/8/12/16 | 2/4 | 真实 checkpoint，27 层 | 1 | 2->4 |

EP 固定为 1。S 的第 0 层为 Dense MLP，第 1–8 层为 MoE，保持原始模型前九层的类型顺序。源模型沿用 Megatron 初始化，目标模型仍由初始同步得到。结果记录 `weights_random_init`、`num_layers`、`layer_types`（dense/moe）及去除 DP/备用布局重复后的 `parameter_count`。

| 方法 | MAX_WAVE_TASKS | EXPANSION_MAX_WAVE_TASKS | SHRINK_MAX_WAVE_TASKS |
|---|---:|---:|---:|
| megatron_default | 256×W | 256×W | 256×W |
| weavetp | 256×W | 512×W | 256×W |
| weavetp_2x | 512×W | 512×W | 512×W |

S 跑三个方法。E 中 weavetp 与 weavetp_2x 的有效扩容配置相同：方向上限均为 512×W，覆盖通用上限，因此 E 只跑 megatron_default 和 weavetp。

所有示例显式设置 `RELEASE_STANDBY_WEIGHTS=1`。初始同步、校验后释放 TP(dst) 备用权重；每次切换先重建并 poison 目标存储，迁移和数值校验后释放旧活动权重。S 的首轮 1→2 为零**远程**传输，仍有同卡复制和真实计时；第二轮 2→1 必须重建已释放的 TP1 权重并执行数据迁移。报告只取第二轮，不把首轮的 wall/transport 时间强制设为零。

## 公共配置

以下 Bash 段可保存为 `/data/ubuntu/lxh/weavetp/dp_sweep_20261011/dp_sweep_env.sh`，在每个节点 source。在干净 shell 中运行，PYTHON 指向已有环境；不安装依赖，不更改选源/执行/KV 算法或门槛。配置中的 packing、router、P2P 顺序及验证阈值沿用 T09/W2。

```bash
export EXPERT_PARALLEL_SIZE=1 MICRO_BATCH_SIZE=1
export RELEASE_STANDBY_WEIGHTS=1 DP_SWEEP_METRICS=1
export SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 PROMPT_TOKENS=8
export MAX_WAVES=128 MAX_OVERLAP_STEPS=1
export PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0
export ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1,2,3,4,5
export P2P_ORDER=peer-size-desc HOTNESS_WEIGHT=0.0 HOTSPOT_PRESSURE_BYTES=0
export ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES=
export ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0
export REPEAT_FORWARD=0 EMULATE_NONCOLLOCATED_SOURCES=0
export PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 DIAGNOSE_EQUIVALENCE=0
export REROUTE_MIN_CONTENTION_GAIN_PCT=0.0 REROUTE_MIN_GLOBAL_GAIN_PCT=5.0
export REROUTE_PENALTY_US=20.0 REROUTE_MIN_BYTES=1048576
export LOGIT_VALIDATION_MODE=bf16-relative LOGIT_MAX_NRMSE=0.4
export LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset GLOBAL_BATCH_SIZE

dp_sweep_config() {
    local experiment=$1 method=$2
    export W=$3
    case "$experiment:$W" in
        S:2|S:4|S:6|S:8|S:10|S:12|S:14|S:16)
            export SRC_TP=1 DST_TP=2 SKIP_CHECKPOINT=1 NUM_LAYERS=9 SWITCHES=2 ;;
        E:4|E:8|E:12|E:16)
            export SRC_TP=2 DST_TP=4 SKIP_CHECKPOINT=0 NUM_LAYERS=27 SWITCHES=1
            : "${CHECKPOINT:?Set CHECKPOINT to the real converted checkpoint}" ;;
        *) echo "Unsupported experiment/card count" >&2; return 2 ;;
    esac
    export MAX_WAVE_TASKS=$((256 * W))
    export EXPANSION_MAX_WAVE_TASKS=$((256 * W)) SHRINK_MAX_WAVE_TASKS=$((256 * W))
    case "$method" in
        megatron_default)
            export METHOD_VARIANT=baseline SCHEDULER_MODE=baseline
            export DISABLE_SOURCE_REROUTE=1 ADAPTIVE_HYBRID=0 ALLOW_AWARE_SHRINK=0
            export ADAPTIVE_RESIDUAL_MAX_WAVES=8 REROUTE_MIN_GAIN_PCT=10.0 ;;
        weavetp|weavetp_2x)
            export METHOD_VARIANT=moetp++-hybrid SCHEDULER_MODE=residual
            export DISABLE_SOURCE_REROUTE=0 ADAPTIVE_HYBRID=1 ALLOW_AWARE_SHRINK=1
            export ADAPTIVE_RESIDUAL_MAX_WAVES=4 REROUTE_MIN_GAIN_PCT=0.0
            export EXPANSION_MAX_WAVE_TASKS=$((512 * W))
            if [[ "$method" == weavetp_2x ]]; then
                export MAX_WAVE_TASKS=$((512 * W)) SHRINK_MAX_WAVE_TASKS=$((512 * W))
            fi ;;
        *) echo "Unknown method" >&2; return 2 ;;
    esac
}
```

每个 W 必须提供对应 W×W 的实测带宽画像，行列顺序与实际 global rank 一致；两节点使用同一份文件内容。W=16 继续执行现有严格画像门禁（T05/T09 格式及 sl3060/sl3061 排布），不以默认矩阵替代画像。W≤8 单节点；W>8 使用节点 0 的 8 卡和节点 1 的 W−8 卡，两边均设置 `WORLD_SIZE_OVERRIDE=W`，保证 GBS 和 16 卡门禁使用实际总卡数。TP(dst) 组必须在节点内，TP(src) 组嵌套其中。

## 单节点示例

先 source 公共配置。以下示例从 `code/current` 目录运行；替换路径，每次 r 使用新目录。完整双节点命令见 [server_commands.md](../../../../documents/dp_sweep_20261011/server_commands.md)。

```bash
cd /data/ubuntu/lxh/weavetp/WeaveTP/code/current
source /data/ubuntu/lxh/weavetp/dp_sweep_20261011/dp_sweep_env.sh
export PYTHON=/data/ubuntu/lxh/weavetp/envs/megatron/bin/python
export PROFILE_ROOT=/data/ubuntu/lxh/weavetp/profiles/dp_sweep ROOT_OUT=/data/ubuntu/lxh/weavetp/formal/dp_sweep_20261011/S
export NNODES=1 NODE_RANK=0 MASTER_ADDR=localhost MASTER_PORT=29500
export NCCL_DEBUG=WARN
# 在新 shell 中重复 r1/r2/r3；本例一次 launch。
for METHOD in megatron_default weavetp weavetp_2x; do
    dp_sweep_config S "$METHOD" 8
    export NPROC_PER_NODE="$W" WORLD_SIZE_OVERRIDE="$W"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export PROFILE="$PROFILE_ROOT/W$W/profile.json"
    export OUT_DIR="$ROOT_OUT/$METHOD/W$W/r1/node0"
    test -f "$PROFILE" && test ! -e "$OUT_DIR" || exit 2
    bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
done
```

E 使用真实 checkpoint，示例为 W=8；不运行 weavetp_2x：

```bash
export CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2
export ROOT_OUT=/data/ubuntu/lxh/weavetp/formal/dp_sweep_20261011/E
for METHOD in megatron_default weavetp; do
    dp_sweep_config E "$METHOD" 8
    export NPROC_PER_NODE="$W" WORLD_SIZE_OVERRIDE="$W"
    export PROFILE="$PROFILE_ROOT/W$W/profile.json"
    export OUT_DIR="$ROOT_OUT/$METHOD/W$W/r1/node0"
    test -f "$PROFILE" && test ! -e "$OUT_DIR" || exit 2
    bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
done
```

## 汇总与解释

```bash
"$PYTHON" -B tools/resharding/summarize_dp_sweep.py \
    --root /data/ubuntu/lxh/weavetp/formal/dp_sweep_20261011/S --direction '2->1' \
    --methods megatron_default,weavetp,weavetp_2x
"$PYTHON" -B tools/resharding/summarize_dp_sweep.py \
    --root /data/ubuntu/lxh/weavetp/formal/dp_sweep_20261011/E --direction '2->4' \
    --methods megatron_default,weavetp
```

`--methods` 接受三个方法的任意非空子集；省略时 S 默认为三个、E 默认为前两个。读取 `<root>/<method>/W*/r*/node0/result.json`，先求 launch 内指定方向的均值，再跨 launch 等权求均值及样本标准差（ddof=1）；一个 launch 的标准差未定义。

主指标为 `base.wall_s`；`transport_s` 仅对应 Base，按每波跨 rank MAX 后累加。`wave_overhead_s=base.wall_s-transport_s` 是这些波的累计开销，不除以波数。网络字节仅按接收侧统计、排除同卡复制；`remote_bytes=base_remote_bytes+delta_remote_bytes=weight_bytes+kv_bytes`，同卡复制见 `local_copy_bytes`。显存峰值统计从切换前 reset 到切换后数值校验完成，分别使用跨 rank 最大 `max_memory_allocated()` 和 `max_memory_reserved()`，输出 `peak_mem_bytes`、`peak_reserved_bytes`。

CSV/Markdown 同时报告各方法的时间、累计波开销、显存、波数、网络字节和原始执行/计划标签，并按 `100×(参照均值−当前均值)/参照均值` 给出相对 default 的改善及 weavetp_2x 相对 weavetp 的改善。先平均再计算比值；零分母不补零。

配对检查保留原始差异：扩容要求 default 波数与 2×weavetp 相差≤1；缩容 default 与 weavetp 相差≤1、weavetp_2x 与 weavetp/2 相差≤1。即使在容差内，也记录缩容波数差。三者 execution_mode 应相同；default 的 plan_variant 应为 baseline，WeaveTP 应为 candidate。超出容差、标签不符、全局门槛否决或证据不完整会醒目标出，报告仍保存且非零退出。

**当前算法的已有行为会触发部分标签检查。** 自适应缩容策略以 `direction_without_reroute` 返回 `plan_variant=baseline` 和 FIFO；协调器的缩容 `baseline_full_plan` 指向缓存缩容计划，该计划在 `ALLOW_AWARE_SHRINK=1` 时仍可能含带宽感知选源。原始 baseline/FIFO 标签不能据此解释为“没有感知选源”，也不能改写为 candidate 来让断言通过。megatron_default 的原始 execution_mode=baseline 与 FIFO 不同，汇总会保留并标记此差异。分析必须同时检查实际计划几何、门槛证据、波数和执行模式；不能把波数或调度差异导致的全部性能变化归因于选源。
