# Step 5 审阅与修正（第一批）

审阅对象：`codex/failstop` 上 Step 5 第一批实现 `9405ed65c00d573938032e8c94a0d795258427fb`（确定性模式合同、显存预检、真实模型参考/候选/计时入口、Step 3 冻结 profile、快验驱动）。修正实现：`2fe4a9a0265580a47d0753dfd2b839e4bdb01f05`。快验清单与服务器命令以[当前交付说明](Step5_服务器命令.md)为准；此前清单 `98c498b938b587af989802f6a61c2bb879df506b` 与 `step5_98c498b.bundle` 作废。审阅前重新阅读 CODEX.md、总计划、Step 5 文档与 code/current/AGENTS.md。服务器 GPU 仍未运行。

## 审阅发现与修正

| 级别 | 原问题与影响 | 唯一生效路径及验证 |
|---|---|---|
| P1 | 单步耗时比在冷进程上测量：参考与计时进程各只跑一遍 prefill + decode，首个 prefill/decode 包含 CUDA 上下文、cuBLAS 句柄与工作区、NCCL 通信器的懒初始化，且两种模式的初始化开销本身不同（cuBLAS 工作区配置、NCCL 算法固定）。比值反映的是初始化差异而不是用户要求的"单步耗时比"。 | `model_reference.py` 先跑一遍丢弃的完整序列作预热，再跑测量序列；记录与耗时都取自测量序列，`input.warmup_sequences=1` 写入报告。 |
| P1 | `parameter_checksum` 对每个参数执行 `.to(torch.int64).sum()`，会为每个分片临时分配 4 倍字节的 int64 拷贝（TP2 的词表 embedding 分片约 0.84 GiB），虚增该进程的峰值显存并吃掉预检留出的 2 GiB 余量，与预检表的口径冲突。 | 改为 `sum(dtype=torch.int64)` 归约，不再物化拷贝；校验和的定义不变。 |
| P2 | 冻结 profile 的管线重复：`run_topology.py` 复制 `mixed_sources.digest` 的实现算 hash，经配置透传给控制器比较，worker 再核对一次；而控制器在 phase 0 本就要求 16 个 worker 报告同一 `profile_sha256`，计划文件也记录 digest。且 16 个 slot 都读取文件，只有 slot 0 的被发布使用。 | 配置只携带 `profile_file`；控制器要求两节点传同一路径（采集是集合通信，两节点必须同为冻结或同为实测，否则 `collect` 死锁）；slot 0 读取、校验设备身份、记录 `profile_frozen`（含 digest）并发布，其余 slot 等待发布值。删除 `run_topology.py` 的 hashlib 复制与 `profile_sha256` 配置项。 |
| P2 | Step 5 的 torchrun 环境没有像 `run_topology.py` 那样清除 `PYTORCH_ALLOC_CONF`/`PYTORCH_CUDA_ALLOC_CONF`，与总计划"默认 CUDA allocator"不一致。 | `run_model.environment` 统一剔除这两个变量。 |
| P2 | Step 5 快验失败时 `quick_driver.py` 因没有 progress.json 打印"no case started; the runner failed during preflight"，对候选逐位不一致等失败是误导。 | step 5 的提示改为指向 `<run>/<node>/error.txt` 与各 `<layout>/<role>/torchrun.log`；Step 2–4 的提示不变。测试覆盖。 |
| P3 | `run_model.evaluate` 重复校验三个进程自身已强制的事实：参考/候选进程启动即 `require_deterministic`，计时进程 `require_default`，候选的 `compare_records` 已要求参数校验和与历史一致；驱动层再次判定这些布尔项是冗余兜底。 | `evaluate` 只以候选的 `bitwise` 结果判定通过，manifest 与跨 rank 一致性只作证据保留；`model_checks_<layout>.json` 与失败信息直接给出首个差异。 |
| P3 | `compare_launch_plans`、`summarize`、`scenario_table` 对生产者必然写出的字段使用 `.get` 兜底；记录里的 `bits_sha256` 无人消费；`environment(base=None)` 只为测试存在；worker 重复 `mkdir` 驱动已创建的目录。 | 全部直接索引；删除 `bits_sha256`、`base` 参数与重复 `mkdir`；未查询节点的占位行显式携带 `free_bytes/total_bytes=None`。测试改用 `patch.dict(os.environ)`，计划比较的测试夹具改为完整计划文件。 |
| P3 | `--gpus` 只检查 8 个互异整数，不检查是否恰为 0–7。 | `sorted(gpus) == list(range(8))`，保留顺序用于让空闲卡优先承担 tp2。 |

未发现以下问题：§2.1 合同的拒绝条件与报告字段完整；显存窗口数值与公式/`PeakBudget` 字段一致；快验布局分配与 300 秒预算的推算一致；Step 2–4 路径未受影响。

## 验证

| 检查 | 结果与原始证据 |
|---|---|
| failstop unittest | 135/135 通过；[日志](step5_cpu_evidence/cpu_tests_review.log) |
| 路由确定性检查单测 | 10/10 通过 |
| isort / py_compile / bash -n / git diff --check | 通过 |
| 服务器 GPU、300 秒实际预算 | 未运行，待回传 |

修正实现 `2fe4a9a0265580a47d0753dfd2b839e4bdb01f05`。真实模型 fail→shrink→rejoin→expand、A6/A7、smoke 与多轮仍未实现，关口不变。
