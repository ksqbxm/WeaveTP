# Step 1 两阶段回归优化

## 已通过的关口与本次范围

用户确认 `s1_shared_20261017_d16b5e8_r2`（实现 `d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c`）共享 GPU 正确性验收通过：SL3060 accepted=true，idle/inflight 各 10/10，两节点 exit=0。此处按用户回传结论登记，未在本地重新解包核验结果包；耗时仍是共享 GPU 诊断数据。

本轮仅修改独立玩具探针，不推进 Step 2/task2 或模型集成。新优化的 GPU 结果待用户执行 N=1 快验回传，不能沿用上一版通过结果。

## 原因与唯一回归流程

旧控制器已在新 worker CUDA 就绪后才发布 transition/2；真正阻塞 EDP1 的是存活 worker 在 world12 校验后直接阻塞等待通知，未继续计算。旧暂停口径从 world12 ready 到 world16 ready，包含新进程启动、torch/CUDA 初始化和 MPS 查询。

现实现只保留两阶段流程：

1. world12 校验完成后，12 个存活 worker 持续校验原张量地址、内容和本地 CUDA 算子。独立通知线程监听控制面，服务循环非阻塞取通知，不调用 world collective/barrier。每步完成后记录 service_progress；20 ms 间隔限制玩具采样负载。
2. 控制器确认各 worker 已开始服务，等待四卡回收完成，允许 SL3061 监管器后台 spawn 四个新 worker。它们完成 torch/CUDA 初始化、张量分配及独立 context 验证，上报 cuda_ready，等待新 world 的 build 命令；不提前加入旧 12-rank world。
3. 收齐两节点 MPS 验证、四个新 PID/就绪报告和每个存活 worker 的 spawn 后计算证据，发布 rejoin_ready，再通知存活 worker 在本地计算边界停止服务。随后统一 abort、协调、以新 PrefixStore 重建并校验 16-rank world。

计算异常、通知错误、准备超时、新 worker 失败仍使本轮失败，沿用登记 PID 的清理路径。初始化期间没有新增 worker barrier；控制器的收报等待不阻塞服务循环。此处“服务”是玩具本地 CUDA 计算，尚未证明模型副本的 TP/EP collective 可持续推进。

## 指标

| 字段 | 定义 |
|---|---|
| spawn_to_ready_s | 四新 worker 最早登记 spawn → 最后 cuda_ready；使用 SL3061 同一单调时钟 |
| rejoin_preparation_s | 控制器 spawn 命令 → 全部就绪/MPS 核对及存活计算证据收齐；EDP1 继续计算 |
| rejoin_join_s | 回归请求 → world16 ready，保留完整回归墙钟 |
| edp1_rejoin_pause_s | 每个 EDP1 worker 最后一次有效服务计算 → world16 校验完成，取最大值；用各进程本地时间差 |
| rebuild_and_validate_rejoin_s | 控制器 build/2 → 全部 world16 校验完成 |
| preparation_service | 每个存活 worker 的首/末计算进度、完成步数、准备期间最大进度间隙；原始每步事件在 worker JSONL |

回归暂停包含本地计算边界、abort、控制协调及重建验证，预期接近重建窗口，不能保证恰好等于 rebuild_and_validate_rejoin_s。共享 GPU 下不设固定秒数性能通过阈值。删除旧 edp1_rejoin_pause_max_s 字段，不保留两套回归停顿口径。缩容暂停指标保持原定义。

## 快速模式与仓库检查

`--repeats N` 是每种故障场景的轮数，默认 10；按 idle_00、inflight_00、idle_01、inflight_01……顺序执行，共 2N 次独立 launch。两节点 N 必须一致，端口使用连续 2N 个端口，非法 N 或端口范围在启动前拒绝。

N<10 时 invocation、config、result、summary 标记 quick_mode=true。N=1 期望每类 1/1、all_cases_passed=true、两节点 exit=0；主验收 accepted=false，防止 2/2 冒充原 20/20。N>=10 时要求所选每类 N 轮全部成功才可 accepted=true；SL3061 始终不作全局 accepted 结论。失败后的 not_run 继续保留，不自动覆盖或重跑。

仓库检查改为 `git status --porcelain --untracked-files=no`：允许未跟踪的 bundle/结果文件，已跟踪文件的暂存或未暂存改动仍拒绝。

## 本地验证

54 项 CPU 测试通过，无跳过。真实 CPU TCPStore + 12 个并发服务线程覆盖独占/共享 × idle/inflight 四种组合，故意延迟新 worker 准备，验证期间计算持续推进、transition/2 与 build/2 未提前发生、就绪后停止与重建有序、暂停起点确实来自服务进度。新增计算/通知失败、准备超时、提前转换拒绝、N=1/3/10/12 矩阵与 summary、临时真实 Git 仓库的未跟踪/暂存/未暂存检查。

isort、Ruff F、compileall 通过；默认、N=1、N=12 dry-run 通过；N=0、负数及越界端口拒绝。证据见 [CPU 日志](step1_two_phase_evidence/cpu_tests.log)。本地测试替代了 CUDA 计算与 GPU 启动，不代表新版本 GPU 性能/正确性已通过。

两阶段优化固定实现：`6bbd25326eb7311688ebb057d35e648471f0e5ae`；54 项 CPU 测试通过，新版 GPU 快验待运行。见 [两阶段回归服务器命令](Step1_两阶段回归服务器命令.md)，包含 SL3061 bundle 分发与两节点 N=1 共享 GPU 命令。
