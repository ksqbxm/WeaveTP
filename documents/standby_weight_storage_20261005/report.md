# 备用权重 GPU storage 释放：实现与验收

## 2026-10-07：fix2，发送快照缓存、检查探针与按位审计

本轮基于 `e6ba55b3b09663773488a37f2edc3d34b0dc17ad`，遵循 CODEX.md，只处理用户附件列出的事项。**未连接服务器、未运行 GPU 验证；expandable 本轮不处理，以下复验仅使用默认分配器。** 下面的 10-06/10-05 内容是历史记录，后续运行使用本节 fix2 入口。

### 复测证据与根因边界

先读取 `gpu_results_recheck/standby_fix_20261006T115406Z_3218941` 的两份 ordering JSON、pending-check JSON 和完整 result，再修改代码。[提取记录](evidence/review_20261007/gpu_recheck_review.json)

- 两个 rank 首次失败均为 **packed=True / scope=ordering / cycle=0**：local 正确，remote/strided 不正确，finite 与 poison-before-copy 正确。在此之前，packed=False 的 ordering 三轮、overlap 三轮均通过。packed=True 的 overlap 尚未执行，不能报告为通过。用户要求的“packed=False 也失败则停止”条件未触发。
- **陈旧发送缓存是已证实的代码缺陷。** 两个半列计划使用相同 task_id/字节数；非连续源切片被复制到临时 contiguous storage。旧缓存按临时 data_ptr 命中时跳过重新打包，地址被复用就把第一半列当第二半列发送。CPU 回归用同两个缓冲确定性模拟分配器地址复用，经实际 launcher 和实际 packing 代码，旧提交失败、新实现通过。GPU 失败形态与之高度吻合；旧日志没有失败区域的值，因此“该次 GPU 失败确实是另一半列的值”仍待新诊断确认。
- pending-check 的八个 rank 均为 **[0, 1, 1, 1]**，只有第一次切换缺少覆盖；delta 检查全部通过。0 步说明第一次全局就绪检查已通过，不能推导 decode 在检查等待中被阻塞。旧探针在所有 finite-check kernel 之前排入 sleep，测试延迟可能在主机完成入队前耗尽。**启动队列饱和是待实测推断，不是日志已证明的根因。**

### 实现及不变量

1. 采用用户方案 (a)：launcher 标记 contiguous 临时副本、反量化副本及显式 transform 的发送为不可缓存。NCCL 对包含这些发送的整个 bucket 每次重打包，不为其构造地址缓存键。只有直接引用稳定推理权重的发送可复用快照；`kv::` 仍不缓存。量化包装器先按类型特征排除，避免访问其不存在的普通 storage。
2. **快照资格与通信分组分离**：临时发送排除集合只决定是否重打包，不改变收发双方的 weight/KV bucket 边界。保留稳定权重快照复用和接收缓冲复用。删除旧 `configure_plan(plan)` 调用路径及“未配置即全部可缓存”的 None 语义，全部调用使用同一显式接口。已存在的 storage 释放会清空服务缓存；内容发生更新的其他 API 调用者仍必须遵守原推理快照契约并调用 invalidate。
3. ordering 不放宽任何判据。只在失败时记录失败张量的不匹配数量、首个下标、NaN、首个期望/实际值，以及不匹配区域是否等于源的另一半列。传输完成后先 Gloo 协调再共同失败的路径保留。
4. pending-check 将 sleep 移到 finite_flag 返回后、完成事件 record 前，并立即 query，先于任何就绪投票。每个 rank/切换写入 `finite_flag_enqueue_s`、`chunk_count`、`estimated_kernel_count`、`ready_immediately_after_enqueue` 和 `overlap_steps`。kernel 数按 `1 + 3 × chunks` 估计，后端算子分解可能更多，不冒充 CUDA trace。生产只增加 release ON 的 `weight_check_enqueue_s`；有限值检查算法、分块大小和有界 decode 不变。
5. 新增默认关闭的 `WEIGHT_BITWISE_AUDIT=1` / `--live-weight-bitwise-audit`，仅允许 release ON。启动时现有 `swap_model_weights` 从已加载的 TP2 精确拷贝到同 dtype TP4，显式 NCCLCopyService 不启用 packing，没有重新计算或类型转换；原初始双布局校验后、第一次释放前，保存两布局逐参数的 GPU int64 双校验和。每轮全部写回完成后、目标 decode/cutover 和源释放前比对，经现有 control Gloo 汇总后共同失败，错误包含 rank、参数名与两项 expected/actual。NaN 漏传在审计开启时也报告参数名。
6. 校验和将 BF16/FP16 原始位解释为 int16，FP32 为 int32，再转 int64，计算普通和与位置加权和。沿确定的分块遍历赋予每个元素唯一位置，临时开销限定于至多 1 Mi 元素的单个块，参考值每参数仅 16 个逻辑字节、保留在 GPU。**双校验和存在理论碰撞，并非数学上无碰撞的全量逐位比较。** 不保存 CPU 权重、不调用 empty_cache、不改 Parameter/形状、也不释放备用 KV。成功切换记录 `weight_bitwise_audit_s`；audit OFF 不出现该字段，也不计算或分配校验和。审计包含 GPU 归约和协调等待，会增加开启时的 cutover 耗时，不能把开启审计的结果直接当无审计性能。

若新 DeepSeek 实测 `weight_check_enqueue_s` 接近或超过约 0.45 s 的 decode 步，再单独报告减少启动数的方案后实施：例如仅对完全被逻辑参数覆盖的 storage 合并检查，估计从 `1+3C` 降为 `1+3S`（C 为块数、S 为可合并 storage 数），或评估多 tensor 融合遍历与归约。不能扫描未初始化的 storage padding；目前没有实测入队数据，不直接替换生产有限值算法。

### 持久打包影响范围

以下是 `code/current` 中直接启用入口及调用它的包装入口；历史代码快照未改动。

| 入口 | 是否可能使用发送快照 |
| --- | --- |
| `examples/rl/benchmark_live_moe_tp.py --live-persistent-pack-buffers` | 显式开启，且 pack_target_bytes > 0、有多 item bucket 时 |
| `tools/resharding/run_live_moe_tp_benchmark.sh` | 默认 persistent=0；环境可设置为 1，pack 默认 4 MiB |
| `run_live_moe_scale_benchmark.sh` | 默认 persistent=0，允许环境开启并转发 |
| `run_live_moe_bandwidth_ablation.sh` | **pack_only / combined 固定 persistent=1、pack=4 MiB**；original / route_only 关闭 |
| `run_deepseek_v2_lite_live_benchmark.sh` | 经通用 launcher 继承环境中的 persistent，未启用 adaptive 时默认 pack=4 MiB |
| `run_live_moe_dual_objective_benchmark.sh`；`run_live_moe_migration_first_benchmark.sh` | 前者明确固定 persistent=0；后者调用前者 |
| `run_deepseek_v2_lite_directional_wave_compare.sh`；`run_deepseek_v2_lite_selected_baselines.sh` | 固定 pack=0，现有 case 不进入此缺陷路径 |
| `compare_weavetp_16gpu.py` | COMMON_ENV 明确 pack=0、persistent=0，**只读核对，没有修改** |
| `run_standby_weight_acceptance.sh`、fix2 benchmarks | 明确 pack=0、persistent=0；ordering 独立测试 packed 关/开 |
| `correctness/gpu_weight_ordering.py` | packed=True 主动开启 persistent；本次已有受影响复测 |
| `BandwidthAwareRefitPolicy`、`get_or_create_service`、`swap_model_weights`、`launch_swap_model_weights`、`launch_reshard_model_weights` / `NCCLCopyService` | Python API 的 persistent 默认 False；显式设 True 的调用者同样适用 |

**T01–T09 的正式模型实验配置为 pack=0、persistent=0。** T01 文档明确约束；T04/T07/T09 保存的展开配置与当前 COMMON_ENV 一致。T02/T06 是统计/检查实现阶段，T05/T08 是网络画像，没有通过此路径发送权重，不能为其虚构“GPU 权重结果”。T03 的历史结果归档中 9 份 result 也都是 pack=0、persistent=False。只读提取的 127 个配置/结果/测试 fixture 记录来自 45 个文件或归档成员，未发现启用组合；其中 fixture 不视为 GPU 实验。[逐项清单](evidence/review_20261007/packing_inventory.json)

**重跑建议：** 本缺陷不要求重跑上述关闭 packing 的正式数据。带有 pack_only/combined 的历史 ablation，以及其他 `pack_target_bytes>0 && persistent_pack_buffers=True`、可能多次产生临时发送张量的结果，需要用修复版重新做正确性和性能验收，尤其是变化的 wave/切片计划。即使固定计划侥幸发送同一内容，修复后的重打包开销也可能影响性能结论。当前提供的归档未包含这些 ablation 原始结果，不能列出已损坏的具体论文行或宣称全部历史数据已逐一验证。新 GPU 回传将补充缓存特征确认。

### 本地验证证据

- 最终相关 CPU 套件：**303 tests + 111 subtests 通过，0 失败、0 跳过（JUnit 414）**。[完整日志](evidence/review_20261007/cpu_tests.log)、[JUnit](evidence/review_20261007/cpu_tests.xml)、[实际命令](evidence/review_20261007/cpu_command.json)。新增覆盖同地址不同内容、稳定引用/反量化/transform 分类、weight/KV 分组对齐、三种精度及非连续参数的冷专家 NaN/列交换/1 ULP、signed zero、分块位置、四次真实协调流程、默认字段、开关约束、MPS 非空客户端拒绝、八项命令和失败后继续。
- 真实双进程 Gloo 验证 ordering 与 bitwise 两条协调失败路径，各注入 rank0/rank1 错误和全通过场景：任一 rank 失败后双方都不能进入下一批，每组有 30 秒上限。这不是 GPU/NCCL 证明。
- 冻结 `e6ba55b...` 重新运行当前反例：缓存 **1 项按预期失败**，旧探针入队顺序 **5 项按预期失败**；新实现均通过。[旧缓存日志](evidence/review_20261007/old_cache_failure.log)、[旧探针日志](evidence/review_20261007/old_probe_failure.log)、[复现程序](evidence/review_20261007/reproduce_old.py)。复现程序在内存加载冻结代码，不把旧路径留在生产实现中。新审计属于新增验收能力，不将旧代码缺少方法的 AttributeError 冒充算法反例。
- release/audit/identity 默认关闭时，实际 coordinator 的完整固定时钟结果与 main `f7d555998b45756998465924d733fd70e553d6b4` 及上次 fixture 逐字段相等，释放调用 0 次，SHA256 仍为 **`9359fbff7887cd7a72b6562a166df5934da1483a7932c0c9fbdefa374722a2c4`**。[本轮对照](evidence/review_20261007/default_parity.json)、[完整 fixture](evidence/review_20261007/default_result_fixture.json)。这证明计算路径与字段，不是承诺两次真实 GPU 墙钟相等。
- 已执行要求的 `uv run ... isort`，最终 isort check、15 个 Python 文件编译、3 个 Bash 入口 `bash -n` 和 `git diff --check` 均通过。[静态证据](evidence/review_20261007/static_checks.json)。初轮 101 项通过、1 项失败来自测试替身漏接收 Gloo 的 group 参数，已修正；后续定向 106 项通过，整套曾达 294+111。最终数以上方最终日志为准，旧反例的非零退出码为预期。最后将 NaN 首值诊断保存为字符串以保持严格 JSON，再运行相关 10 项测试通过（[日志](evidence/review_20261007/diagnostic_tests.log)、[JUnit](evidence/review_20261007/diagnostic_tests.xml)）。

### SL3060 的唯一 fix2 复验入口

确认 `$REPO` 更新到本次交付的最终完整 SHA 后运行：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
bash "$REPO/documents/standby_weight_storage_20261005/run_default_fix2.sh" --dry-run
bash "$REPO/documents/standby_weight_storage_20261005/run_default_fix2.sh"
```

脚本自身也在开头 source env.sh。实际运行创建新的 `/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_<UTC>_<pid>/`，拒绝覆盖。先检查 GPU 0–7 的利用率、完整进程列表和 MPS server/client 列表；仅确认无客户端的空闲 MPS 可留存，查询权限/管道不匹配则退出并打印占用进程。使用 NVIDIA 的 [get_server_list / get_client_list](https://docs.nvidia.com/deploy/topics/topic_5_1_1.html) 控制接口，保留查询记录；不杀进程、不操作 MPS 服务。

八个 case：synthetic/deepseek OFF 各一次，synthetic ON + bitwise 一次、DeepSeek ON + bitwise 三次，修正后的 synthetic pending-check 一次，ordering 一次。ordering 内部仍是 packed 关/开 × ordering/overlap × 三轮，每 rank 应有 12 条通过记录。每 case 保存 command/console/exit code，失败继续其余 case；根目录保存 commit/gpus/runtime、GPU XML/MPS 查询、summary.tsv，结束打印汇总表和 `tar czf` 命令。无需依赖旧验收包存在；参数取自仓库内默认矩阵，清除两个 allocator 环境变量。

所有新 benchmark 四次切换和原 logits 判据均须通过；ON 各 rank/切换应有 bitwise 比对耗时、无 audit mismatch；pending 两项判据须通过；ordering 两份文件各 12 轮全通过。备用 KV 仍按原字段单列驻留字节数，释放策略未变。请回传整个新目录，届时再审阅真实吞吐/入队耗时及 GPU 正确性。

完整 dry-run 命令见下面展开记录。`standby_fix2_DRYRUN` 只是本地拦截执行用于展示的目录名，实际脚本自动使用 UTC+PID，不创建该展示目录。解释器与 checkpoint 取原 SL3060 验收值；实际值以 env.sh 和每 case command.txt 为准。

```bash
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF PYTHON=/home/ubuntu/miniconda3/envs/megatron/bin/python CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++ KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS=0 WEIGHT_STORAGE_AUDIT=0 WEIGHT_CHECK_AUDIT=0 WEIGHT_BITWISE_AUDIT=0 SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0 MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1 ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE=allclose LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0 EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0 NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4 FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2 PROFILE=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/uniform-100gbps-no-profile.json OUT_DIR=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_synthetic_off bash tools/resharding/run_live_moe_tp_benchmark.sh
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF PYTHON=/home/ubuntu/miniconda3/envs/megatron/bin/python CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++ KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS=0 WEIGHT_STORAGE_AUDIT=0 WEIGHT_CHECK_AUDIT=0 WEIGHT_BITWISE_AUDIT=0 SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0 MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1,2,3,4,5 ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE=bf16-relative LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0 EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0 NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4 FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2 PROFILE=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/uniform-100gbps-no-profile.json OUT_DIR=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_deepseek_off bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF PYTHON=/home/ubuntu/miniconda3/envs/megatron/bin/python CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++ KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS=1 WEIGHT_STORAGE_AUDIT=0 WEIGHT_CHECK_AUDIT=0 WEIGHT_BITWISE_AUDIT=1 SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0 MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1 ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE=allclose LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0 EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0 NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4 FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2 PROFILE=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/uniform-100gbps-no-profile.json OUT_DIR=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_synthetic_on bash tools/resharding/run_live_moe_tp_benchmark.sh
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF PYTHON=/home/ubuntu/miniconda3/envs/megatron/bin/python CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++ KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS=1 WEIGHT_STORAGE_AUDIT=0 WEIGHT_CHECK_AUDIT=0 WEIGHT_BITWISE_AUDIT=1 SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0 MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1,2,3,4,5 ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE=bf16-relative LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0 EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0 NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4 FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2 PROFILE=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/uniform-100gbps-no-profile.json OUT_DIR=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_deepseek_on bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF PYTHON=/home/ubuntu/miniconda3/envs/megatron/bin/python CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++ KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS=1 WEIGHT_STORAGE_AUDIT=0 WEIGHT_CHECK_AUDIT=0 WEIGHT_BITWISE_AUDIT=1 SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0 MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1,2,3,4,5 ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE=bf16-relative LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0 EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0 NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4 FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2 PROFILE=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/uniform-100gbps-no-profile.json OUT_DIR=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_deepseek_on_bitwise_r2 bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF PYTHON=/home/ubuntu/miniconda3/envs/megatron/bin/python CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++ KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS=1 WEIGHT_STORAGE_AUDIT=0 WEIGHT_CHECK_AUDIT=0 WEIGHT_BITWISE_AUDIT=1 SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0 MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1,2,3,4,5 ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE=bf16-relative LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0 EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0 NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4 FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2 PROFILE=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/uniform-100gbps-no-profile.json OUT_DIR=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_deepseek_on_bitwise_r3 bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF PYTHON=/home/ubuntu/miniconda3/envs/megatron/bin/python CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MODEL_PRESET=synthetic EXPERT_PARALLEL_SIZE=2 METHOD_VARIANT=moetp++ KV_REQUEST_IDENTITY=0 RELEASE_STANDBY_WEIGHTS=1 WEIGHT_STORAGE_AUDIT=0 WEIGHT_CHECK_AUDIT=1 WEIGHT_BITWISE_AUDIT=0 SWITCHES=4 REPEAT_FORWARD=0 PROMPT_TOKENS=8 MICRO_BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 MAX_WAVES=128 MAX_WAVE_TASKS=2048 EXPANSION_MAX_WAVE_TASKS=0 SHRINK_MAX_WAVE_TASKS=0 MAX_OVERLAP_STEPS=1 SEQ_LENGTH=1024 MAX_POSITION_EMBEDDINGS=1024 ROUTER_MODE=fixed-hot ACTIVE_EXPERTS=0,1 ACTIVE_EXPERT_PHASES= PRESSURE_RANK_PHASES= HOTSPOT_PRESSURE_BYTES=0 ONLINE_REPLAN=0 ONLINE_MIGRATION_FIRST_GUARD=0 HYBRID_FAST_PATH=0 ADAPTIVE_HYBRID=0 SCHEDULER_MODE=residual ALLOW_AWARE_SHRINK=0 PACK_TARGET_BYTES=0 PACK_MAX_ITEM_BYTES=0 PERSISTENT_PACK_BUFFERS=0 PACK_REROUTED_ONLY=0 P2P_ORDER=peer-size-desc LOGIT_VALIDATION_MODE=allclose LOGIT_MAX_NRMSE=0.4 LOGIT_MIN_COSINE=0.93 LOGIT_MIN_TOP1_AGREEMENT=0.0 EMULATE_NONCOLLOCATED_SOURCES=0 DISABLE_SOURCE_REROUTE=0 DIAGNOSE_EQUIVALENCE=0 NUM_LAYERS=2 HIDDEN_SIZE=512 NUM_ATTENTION_HEADS=8 NUM_QUERY_GROUPS=4 FFN_HIDDEN_SIZE=1024 MOE_FFN_HIDDEN_SIZE=1024 NUM_EXPERTS=8 MOE_ROUTER_TOPK=2 PROFILE=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/uniform-100gbps-no-profile.json OUT_DIR=/data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_pending_check bash tools/resharding/run_live_moe_tp_benchmark.sh
env -u PYTORCH_ALLOC_CONF -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=8 /home/ubuntu/miniconda3/envs/megatron/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 tools/resharding/correctness/gpu_weight_ordering.py --output /data/ubuntu/lxh/weavetp/acceptance/standby_fix2_DRYRUN/default_ordering
```

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
