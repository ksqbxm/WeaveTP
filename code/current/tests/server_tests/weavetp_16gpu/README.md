# WeaveTP 16 卡服务器测试

本目录验证已有实现的真实环境行为。**本次仅新增文件和运行本地 CPU 测试，未 SSH、部署或启动 GPU。当前故障回归为失败，T06 尚未实现，不能据此进入正式 GPU 实验。**

## 文件与证据分离

```text
tests/
  unit_tests/resharding/
    test_summarize_weavetp_formal.py    # T02：统计与异常输入
    test_compare_weavetp_16gpu.py      # T04：配置、mock 协调及 shell argv
    test_profile_weavetp_16gpu.py      # T05：画像、测量控制流、shell 门禁
    test_server_acceptance.py         # 新服务器检查器的 CPU 正反例
    test_weavetp_review_findings.py    # 未解决的生产缺陷，正常失败，不标 expectedFailure
  server_tests/weavetp_16gpu/
    common.py                        # /data、独占输出、hash、子命令证据
    test_cpu.py                      # Linux 解释器上运行上述五组测试
    test_node.py                     # GPU/磁盘/版本/网卡/源码一致性，只读
    test_checkpoint.py               # 已有 checkpoint 分片流式 hash，只读
    test_history.py                  # 固定 9 份真实历史 JSON、54 项表 2 及独立算术
    test_controller.py               # 真实 SSH、进程、信号；只运行 CPU 替身
    test_profile.py                  # 实际 UUID、原始 NCCL 日志、两端画像 hash
    test_groups_worker.py            # torchrun worker；实际 ProcessGroup 与 collective
    test_results.py                  # 正式 9 份结果与两端凭据，只读；完整验收仍 blocked
    fixtures/
      cpu_worker.py                  # 明确标注的替身，不导入 torch
      historical_8gpu_manifest.json  # 仅固定 9 个相对路径与 SHA-256，不选“最新九份”
```

服务器证据使用独立目录，例如 `/data/${SERVER_USER}/lxh/weavetp/acceptance/<唯一批次>/<检查名>/`。正式性能 ROOT_OUT、画像输出、测试替身目录分别存放。已有证据目录会被拒绝，不能覆盖失败后重跑同一路径。每次入口输出 `acceptance.json`；退出码 0=该检查通过，1=失败，2=依赖/完整验收缺项。启动前的平台、输出路径检查失败时不会创建报告。GPU worker 由 torchrun 启动，成功产物为 `groups.json`，另存两端日志和退出码。

`fixtures/historical_8gpu_manifest.json` 仅包含相对路径和 SHA-256。历史复核的来源及当时状态保留在本地阶段报告；此清单用于复核当前汇总脚本。

## 覆盖与执行边界

| 计划/功能 | 测试入口及断言 | 需要的真实证据 / 尚未实现部分 |
|---|---|---|
| T01、T08 空间、机器、版本 | test_node：主机/rank、8 张本地卡、显存/作业、25 GbE、RoCE v2 GID 3、Python/torch/CUDA/NCCL | 两机分别执行；不会安装或删除数据；空闲检查必须紧邻每次 GPU 启动 |
| T01、T07、T08 源码一致 | test_node：Git HEAD 与 `git status --porcelain` | 两机均从 GitHub 检出同一 commit 且工作树为空，再用对端报告比较 |
| T08 权重部署 | test_checkpoint：已有 torch_dist 元数据和全部分片 hash、大小、读期间不变 | 两机报告配对，约 59 GiB 只读 I/O；不复制、不加载 GPU；目录存在不算权重验收 |
| T02 launch 内/间统计 | test_cpu + test_history | 正反例覆盖 n=3、ddof=1、三个范围、无 warmup 丢弃、先逐 wave MAX 再累加；真实 JSON 再独立计算 |
| T02 输入/百分比 | test_cpu + test_results | 缺失、重复、混批、hash、方向、负收益、均值之比、历史引文/补算、0.1 个百分点舍入 |
| T03 表 2 | test_history | 9 个固定 hash，27 均值+27 SD，容差不变；输入前后 hash；不重跑 8 卡 GPU |
| T04 双机参数/环境 | test_cpu | 实际 shell 配合 torchrun 替身检查 9×2 argv、静态 rendezvous、global batch=8、wave、WARN、隔离外部开关 |
| T04 成功/续跑 | test_controller --scenario success | 真实 SSH、两个 Linux 子进程、/proc、退出回执、完成标记；第二次严格跳过，不重新生成标记 |
| T04 节点失败/任务清理 | test_controller --scenario node1-failure | 先确认 master CPU worker 在运行，再让对端退出 7；两端真实清理、退出和显存恢复；保留 MPS |
| T04 主控中断 | test_controller --scenario controller-interrupt | 对测试主控自身注入 SIGINT，检查本任务 worker 清理及回执；未启动的对端可没有 exit，但必须清理确认 |
| T04 身份不一致 | test_controller --scenario code-mismatch | 只改临时副本 node1 wrapper，预检拒绝，两个 worker 都不能启动 |
| T04 只有 master 结果 | test_controller --scenario partial-result | 临时目录只有 result.json 时拒绝跳过、覆盖、重试；两个 worker 都不能启动 |
| T04 清理不可达 | test_weavetp_review_findings | CPU 可控 RPC 反例稳定失败；后续以有限远程命令超时、进程终止和“未确认”状态修复 |
| T04/T09 超时 | 60 分钟 launch wait 与远程命令超时 | 逐切换由操作者人工监控；必须用真实 CPU 阻塞 worker 验证超时后的进程终止和报告 |
| T05 画像算法/格式/拒绝回退 | test_cpu | 240 对、64 MiB、1+3×3、decimal Gbps、中位数、非实测对角线、缺失/NaN/维度/默认画像拒绝 |
| T05/T08 实测网络和设备 | 原 run_weavetp_16gpu_profile.sh + test_profile | 真正的 2×8 测量；保留原始 NCCL 日志及两机画像 hash；逐日志/UUID 配对检查暂缓且不阻断 |
| T06/T08 现有组构建器 | test_groups_worker | 真实 get_process_group_ranks/get_world_size、TP2/4、DP8/4、EDP4/2、TP/ETP/EP 机内、UUID 唯一及各组 all_reduce；不是 benchmark 实际 source/destination hooks 的证明 |
| T06 流量/候选分类/计时 | 尚无实现可调用，见下表待接入案例 | 本次不创建一个只测假实现并宣称通过的测试；正式结果入口明确 blocked |
| T07 集成 | test_cpu、bash -n、本次审阅及文件 hash | 已有功能的测试通过不等于 T06 完成；真实 Linux/SSH/GPU 验证状态分别报告 |
| T09 正式负载/数值 | test_results + 原 compare 的 9 次正式运行 | 9 结果、36 切换、两端 prepare/exit/config/日志/complete；真实 checkpoint_loaded；不允许替身数据，不因慢或 FIFO 回退删结果 |
| T10 汇总 | test_results | 原汇总器、独立均值/SD/18 项提升；wave/实际路径单独输出；流量、候选分类、时间线缺项继续 blocked |

T06 实现后需要接入的案例（不能以请求参数或公式计算的期望值充当实际观测）：

| 功能 | 必要正反例 | 真实服务器验证方式 |
|---|---|---|
| 接收侧去重 | send+recv 只计一次；同卡复制为 0；机内远程计 send/recv 峰值但不计跨机；0→8 与 8→0 分别统计；总量守恒；并列最大 rank | 从已采用任务表导出可复算记录，独立算术对照；不增加性能 launch |
| 权重/KV | 分离权重、前缀、增量；零长度；有效前缀 8 而容量 1024；delta 只计算有效区间 | 对照本次 switch 的有效 sequence 范围和 dtype 字节数 |
| 默认/候选/实际计划 | 扩缩两个方向都有默认计划；候选未保留为 unavailable；候选被拒绝后不能拿恢复的 baseline 冒充候选 | 检查同一缓存计划身份及采用记录；收缩不适用 |
| 全局筛选 | 通过、真正全局否决、无有效改道、关闭/不适用四态；False 不直接等价全局否决；候选通过但 adaptive 回退独立记录 | 对照初始化缓存筛选详情、实际 base.execution_mode，不改门槛制造分支 |
| 不侵入计时 | 统计在原计时区间外，无新增迁移热路径同步，缺字段不推断 | CPU 分支回归 + diff/调用位置审查；正式 r1 验证观测，不加 A/B 性能组 |

## CPU 和只读服务器命令

以下为待执行示例。`CODE` 必须改为实际部署的工作副本（在 `/data`）；`PY` 使用该主机既有解释器；`RUN` 每轮选新的路径。所有命令显式 `-B`，入口为子进程重定向 `/data` 缓存。

```bash
CODE=/data/${SERVER_USER}/lxh/weavetp/code/current
PY=/home/${SERVER_USER}/miniconda3/envs/megatron/bin/python
TESTS="$CODE/tests/server_tests/weavetp_16gpu"
RUN=/data/${SERVER_USER}/lxh/weavetp/acceptance/review_001

"$PY" -B -X utf8 "$TESTS/test_cpu.py" --output-dir "$RUN/cpu"
"$PY" -B -X utf8 "$TESTS/test_history.py" \
  --batch-root /data/${SERVER_USER}/moetp_experiments/deepseek_v2_lite_directional_wave_formal_3runs_20260901 \
  --output-dir "$RUN/history"
```

当前 `test_cpu.py` 预期退出 1：开放缺陷测试失败。不要改为 skip/expectedFailure 或删除它来通过门禁；其余套件日志仍会保存。历史复核只需在 SL3060 执行，调用当前源码，不解包旧 T03 自包含脚本覆盖新代码。

```bash
# 两机各执行一次；SL3061 用 node-rank=1 及实际 Python。
# min-free-gib 按下一阶段真实需求设定，示例 1 仅适用于小型测试证据。
"$PY" -B -X utf8 "$TESTS/test_node.py" --node-rank 0 --expected-gid 3 --min-free-gib 1 \
  --require-idle --output-dir "$RUN/node0"

# 读取整份已有 checkpoint，不复制。两机完成后仅传小型 acceptance.json，
# 再用 --peer-report 比较；不允许同机报告冒充另一台。
"$PY" -B -X utf8 "$TESTS/test_checkpoint.py" \
  --checkpoint /data/models/DeepSeek-V2-Lite-megatron-v2 --output-dir "$RUN/checkpoint0"
```

`test_node` 的 `--peer-report` 比较 commit 和软件；`test_checkpoint` 的 `--peer-report` 比较模型清单。没有对端参数只证明本地检查通过，不能称作两机一致。输入报告需要保持来源和 hash，不能修改其主机字段配对。

## T04 真实协调检查（CPU，不启动 torchrun）

只在 SL3060 驱动。要求两机 SSH、既有解释器、空闲 GPU 和 master 端口可用（沿用生产预检）。使用有限大小的临时代码副本及明确标记的 CPU 画像/结果；每次 worker 最多自然存活约 30 秒，生产清理应更早终止它。测试生成的 dummy checkpoint 只用于目录存在检查，绝不是模型验收。

```bash
"$PY" -B -X utf8 "$TESTS/test_controller.py" \
  --peer-ssh "${SERVER_USER}@${NODE1_ADDR}" --master-addr "${MASTER_ADDR}" \
  --node1-python /data/已批准环境/bin/python \
  --scenario success --output-dir "$RUN/controller_success"
```

分别以 `node1-failure`、`controller-interrupt`、`code-mismatch`、`partial-result` 替换 scenario，每个使用新的 output-dir；一个失败即停止，不用循环继续。默认生成 CPU profile fixture，避免 T07 依赖尚未执行的 T08。可传 `--profile` 使用已有真实画像，但这仍是 CPU 协调测试。任一结果都不能放进正式 ROOT_OUT。临时副本包括测试用 benchmark/wrapper；部署的生产代码不改动。服务器副本和日志保留，不自动删除。

## T05/T08 画像与可选进程组检查（会占用 16 GPU）

先解决开放缺陷并完成 T06/T07、部署/复制批准、两机代码/模型检查。每次 GPU 工作之前重新只读检查两机空闲、磁盘、端口；有占用立即停，不等待空卡。使用生产 `tools/resharding/run_weavetp_16gpu_profile.sh` 的既定 T08 流程，一台一次、同一绝对 OUT_DIR，INFO 仅用于画像。成功后保留两端原始 NCCL 日志，再将同一个 profile.json 同步到对端并固定 hash。

```bash
# 两机分别执行；这是读已有画像和日志，不重新测量。
"$PY" -B -X utf8 "$TESTS/test_profile.py" \
  --profile /data/${SERVER_USER}/lxh/weavetp/profiles/正式画像/profile.json \
  --expected-sha256 实测完整64位sha256 --output-dir "$RUN/profile0"
```

将 node0 的小型报告传到 node1 后，node1 使用同样 profile/hash 并加 `--peer-report <node0报告>`；`paired=true` 才确认两机配对。两机保留各自的 8 份原日志，不把 node0 文件系统当共享磁盘。逐日志/UUID 核验可稍后执行，不作为启动阻断项。

`test_groups_worker.py --out-dir <新目录>` 是可选通信测试 worker，必须使用 `torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0或1 --master_addr=${MASTER_ADDR} --master_port=29500 --rdzv_backend=static --max_restarts=0` 启动。由 T08 操作者准备两端同路径的新目录、重定向本进程及 torchrun 的全部缓存，沿用画像的干净 INFO/eno1np0/mlx5_0 环境。worker 本身还会在导入 torch 前重定向缓存、先 set_device 再建通信组；默认进程组 timeout 为 180 秒，新建组沿用既有构建器的超时设置，这不等于整个测试的退出上限。

该 worker 是测试体，没有独立双机调度器。**当前生产 tagged_processes 的名称白名单不包含 test_groups_worker.py；未接入经验证的 OUT_DIR 清理监督前不要启动这个可选测试**。T08 的必要通信与 UUID 检查仍由正式画像入口覆盖。接入时需为本测试名称补精确标签匹配的正反例，不能泛化到 kill 全部 Python/torchrun。它不加载模型，不产生性能数据，也不能替代正式 benchmark 两套真实组的 T06 断言。

GPU 成功或失败均保存两端日志/退出码，并重新检查进程和显存恢复。只处理命令行明确含本次 OUT_DIR 的本人任务；保留指定用户的 MPS。失联/归属不明时停止报告并标记清理“未确认”。画像文件通过不能代替本步骤。

## T09/T10 正式结果验收

不另外启动性能测试。三配置 r1 既是冒烟也是正式数据；完整九次由原 compare 按固定轮换执行。将 SL3061 的 request/prepared/exit/node1.log 小型证据复制到 master 独立归档，保留 `fixed/r1` 等相对结构（无共享存储）。

```bash
"$PY" -B -X utf8 "$TESTS/test_results.py" \
  --batch /data/${SERVER_USER}/lxh/weavetp/正式ROOT_OUT \
  --peer-receipts /data/${SERVER_USER}/lxh/weavetp/acceptance/节点1凭据归档 \
  --output-dir "$RUN/formal_results"
```

入口检查固定 9 个路径、36 次方向/索引、有效计时、真实权重加载标记、两端环境/源码/画像/退出/清理状态、完整完成标记、原日志存在，拒绝测试替身。用同一汇总器生成统计，再直接从原始 JSON 独立复算 27 格均值/SD 和 18 项提升，写出实际 wave/执行路径。

即便上述检查全部通过，当前版本仍返回 **2 / blocked**，列出尚未实现的 T06 观测、跨 case 时间线和数值原始证据限制；逐切换 10 分钟自动限制已取消。NRMSE/cosine 成功值目前未写入 JSON；只能依据审阅后的生产校验代码和两端成功退出说明原检查通过，不能拿 validation_max_diff 充当 NRMSE。将来应接入真实字段后更新本入口，不预设虚构 schema，不删掉 pending 来宣称完成 T10。
