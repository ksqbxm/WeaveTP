# Step 3 审阅与修正

实现：`7e61596114b003bf3ecefdd8045a6b10aeeda65b`，分支 `codex/failstop`，审阅起点 `bcc61aeb86ddc1bcce403785493ac38ad59486db`。本轮修改前重新阅读 CODEX.md、总计划、Step 2/3 文档和 code/current/AGENTS.md；imports 变动后执行 isort。快验清单提交与离线 bundle 以[服务器命令](Step3_服务器命令.md)为准。

状态：已实现、CPU 已验证；服务器 CUDA/NCCL、profile 数值与 300 秒快验预算未运行。本轮没有登录服务器、运行 GPU 或改动其他用户任务。Step 2 real GPU 与 Step 3 real GPU 均未登记通过；Step 4 的 fix2 关口不变。

## 审阅发现与修正

| 级别 | 原问题与影响 | 唯一生效路径及验证 |
|---|---|---|
| P1 | 准备期进度检查存在竞态。worker 每步先发布 `serving(N)` 再立即检查回归信号；若控制器恰在该次发布之后读取到 `serving(N)` 并随即发布信号，worker 会以同一步 N 发布 `prepared_service`，控制器的 `step > step` 判定失败，正确运行被判 FAIL。emulate 第 2 轮起控制器读取与发布 `emulate_rejoin` 只相隔一次往返，窗口真实存在。 | worker 观察到信号后再完成一整步服务才发布 `prepared_service`，因此严格晚于控制器在发布信号前能采样到的任何步。CPU 并发协议（emulate 连续两轮、real 替换进程）通过。 |
| P1 | 主机库预算把 `RLIMIT_MEMLOCK` 当作注册上限，`memlock < backing_bytes` 即拒绝。`cudaHostRegister` 由驱动锁页，不走 `mlock`，该限制不约束 CUDA 注册（受 memlock 约束的是 RDMA 的 `ibv_reg_mr`）。服务器常见默认 memlock 为 8 MiB，而正式模式 on-demand backing 约 12.6 MB，会在读权重前被误拒。 | 预算只核对 MemAvailable 与实测 /dev/shm 可用字节不小于 backing；memlock 值仍写入 `host_library.json` 作为证据。注册能力改由下一项真实探针证明。单测改为 RAM 不足拒绝、`memlock=1` 仅记录不拒绝。 |
| P1 | 注册能力预检用匿名 mmap 页做 `cudaHostRegister`，而实际 backing 是 /dev/shm 共享文件映射。匿名页几乎总能注册，不能证明文件映射可 pin，"读权重前核对注册能力"的关口因此没有证明力。 | 探针在 backing 同目录创建一页共享文件，mmap 后注册、注销并删除，记录 `backing_kind=shared_file`。owner 仍等待本节点全部 worker 的探针结果后才读权重。整库注册仍由 `HostAttachment` 在首次使用前完成，失败即停止。 |
| P2 | 扩容选源用硬编码坐标 `source_etp < 2` 排除回归空目标，与 Step 2 的 `source_eligible` 合同重复且可能分叉；缩容时的 TP2 局部序号用 `.get(..., source_etp)` 兜底，缺失时静默取错分片。 | `snapshot` 只按 `rank_map()` 的 `source_eligible` 取源；缩容阶段由 slot 0 广播全部 16 个 slot 的 `shrink_etp`，扩容直接索引，缺失即 KeyError。 |
| P2 | host-only/mixed 的放置规则在 fixture、worker 初始加载、GPU 源快照、host 候选四处各写一遍。 | `source_fixture.host_resident(name, part)` 唯一定义；`demand()`、初始 GPU 加载、GPU 源与 host 候选均调用它。 |
| P2 | `run_source_profile.py` 与 `run_sources.py` 为两份相同包装；`host_stopped` 消息无人消费；`SourceRuntime` 构造时分配一份随后必被替换的目标；profile 顶层 `unit='Gbps'` 与各条目 `bytes/s` 矛盾；预算报告含 ×4 元数据、page-cache 等推测项；`seconds()` 重复校验 `fit` 已保证的量；探针 `probe` 字符串和 /dev/shm 路径散落多处。 | 单入口 `run_sources.py --profile-only`；删除上述冗余；`TOPOLOGY_PROBES`、`host_backing` 取代散落字面量；控制器对 Step 3 配置键做严格相等比对而非 `.get`。 |

以上修改没有改变 Step 3 的检查项：三类来源、非连续/专家切片、逐位比较、字节总量、profile 类别、owner/backing 跨故障不变、新 PID 不预载不作源、清理与恢复均保留。

## 共享 GPU 正确性模式

Step 3 入口复用 Step 2 审阅时确定的唯一开关 `--correctness-only`：`run_sources.py` 直接继承 `run_topology.main`，默认关闭，`--quick` 自动开启，不新增第二条开关或别名。生效内容：预检与回收按总计划 §4 允许 foreign GPU 进程和 MPS 客户端，不停止其他任务；本测试 worker 身份、独立 CUDA context、kill 范围、逐位与清理检查不放宽；阶段等待上限 120 s、启动 180 s；summary、receipt 与 profile 均标 `correctness_only=true`、`shared_gpu_mode=true`、`timing_valid=false`、`accepted=false`。共享条件下采到的 profile 只作本次正确性测试的选源输入，不作性能 profile。

## 本地验证与边界

| 检查 | 结果与原始证据 |
|---|---|
| failstop unittest | 99/99，通过，含修正后的 Step 3 并发协议与主机库单测；[日志](step3_review_evidence/cpu_tests.log) |
| NCCL packing 与选源回归 | 56/56，通过；[日志](step3_review_evidence/packing.log) |
| 4 进程真实 Gloo | 4/4，通过；[日志](step3_review_evidence/gloo.log) |
| isort、AST 未用 import、py_compile、bash -n、三种 dry-run | 通过；[追溯记录](step3_review_evidence/validation.json) |
| 服务器 GPU / NCCL / profile / 300 秒实际时长 | 未运行，待回传 |

CPU 替身用合成 profile 替代测量，用 CPU 拷贝替代 CUDA 注册和 NCCL；它们证明协议、覆盖、完成边界与字节内容，不证明 GPU 可行性。竞态修复按构造成立，并发替身无法稳定复现原窗口，不把替身通过写成竞态已在服务器验证。
