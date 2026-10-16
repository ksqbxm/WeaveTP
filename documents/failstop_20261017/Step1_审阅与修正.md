# Step 1 审阅与修正

日期：2026-10-17。审阅基线：`e1a5fd0bb1e70be797e340f7127a5561eb151d51`（执行代码为 `a609198`）。本轮按正确性、完备性和简洁性检查 Step 1 的故障通知、MPS、进程所有权、回收计时及最终验收；没有推进其他 Step，也没有改生产入口。

状态：**已修正，31 项 CPU 测试通过；GPU 未执行，real 集成未放行。** 最新固定实现 SHA 和两节点命令以 [服务器命令](Step1_服务器命令.md) 为准。修改前重读 CODEX、总计划、Step 1、AGENTS 和贡献规范；对应文件散列见 [验证记录](step1_review_evidence/validation.json)。

## 发现与处理

| 优先级 | 原问题及影响 | 当前唯一路径 |
|---|---|---|
| P1 | worker 的通知线程与 NCCL 建组共用 TCPStore 连接；PG 的阻塞 get/wait 会持有连接锁，通知无法及时到达。独立 PrefixStore 不能隔离这把锁。 | 通知线程独立建立 TCPStore 客户端连接；保留同一个带外服务和控制命名空间。返回的新 worker 不启动无用通知线程。 |
| P1 | SL3061 controller 在最外层清理完成前发出 `node1_exit`；SL3060 可能先通过并关闭 store，遗漏远端最终清理失败。`events_actual.json` 还可能保留提前写入的 passed。 | 最外层 launcher 持有 store；两节点完成本机 controller/后代清理后交换终态，SL3061 确认收到 SL3060 终态后才关闭服务。最终 `result.json` 与 `events_actual.json` 由 launcher 同源写入；controller 只写 `controller_result.json`。删除 `node1_exit` 和提前生成最终结果的旧路径。 |
| P1 | 子进程创建后，读取启动信息或登记文件失败时尚未进入 children 列表，后续清理找不到它。 | spawn 登记失败即通过本次创建、尚未回收的 Popen 句柄 kill 并 wait；失败继续上抛。launcher 的 spawn 纳入异常处理。 |
| P1 | 登记表清理只发 SIGKILL，不等待退出；直接子进程清理异常还会跳过整个后代清理。 | 发信号后持有同一 pidfd，限时等待其退出可读事件；已退出进程不再要求读取存活环境。直接子进程清理失败仍执行后代检查。身份校验、PID 复用保护及故障目标 UUID 核对继续生效。 |
| P1 | MPS 控制进程仅匹配 `-d`；前台 `-f` 或其他启动形式可能被误认为无服务。管道同一 PID 有多个路径时还可能误报原环境无服务。 | 按控制程序身份识别，再以实际 PID 文件和只读查询证明状态；无法定位即失败。保存全部可达服务路径，正确报告原环境。 |
| P2 | join 指标先按控制器接收/命令时间计算，再覆盖为物理回收/spawn 时间；回收先后核对还绕经时钟偏移来回换算。 | 一次计算指标；保存回收确认的本地时间，直接与本机 spawn 时间核对。只对同一时钟域断言先后，跨机估计保留校准误差，跨机动作顺序由消息门禁保证。 |

没有保留旧结果文件读取、旧终态通知、共享通知连接或只发信号即认定清理完成的兼容分支。保留的异常清理分别负责登记失败、本机直接子进程和登记后代，处理不同的资源所有权边界；不会操作 MPS 或未登记的外部进程。

TCPStore 锁行为已用本地真实 TCPStore 复现：一个线程阻塞 `get` 时，共用连接的 `check` 无法返回，释放 get 后才能返回。PyTorch 2.11 的 [TCPStore 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/distributed/c10d/TCPStore.cpp)也显示 get/check/wait 共用 `activeOpLock_`。NVIDIA 的 [MPS 控制接口](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)列出前台 `-f` 模式。

## 验证及边界

- `python -X utf8 -m unittest discover -s code/current/tools/resharding/failstop -p 'test_*.py' -v`：31 项通过，无跳过。真实 TCPStore 测试包括通知连接隔离、远端清理失败传播、完整 idle/inflight 控制协议和命名空间隔离；pidfd/进程/MPS 的故障注入测试使用 mock。
- 针对旧版本运行三项回归测试，MPS 识别与登记失败回收均被拒绝：共 5 个失败子例、1 个错误；这是预期的旧版本反例，不是当前代码验证失败。见 [旧版回归日志](step1_review_evidence/before_fix_regressions.log)。
- isort、Ruff F 检查、Python 编译、主矩阵和可选 shrink dry-run、`git diff --check` 通过。实际 argv、退出码、源码散列见 [validation.json](step1_review_evidence/validation.json)，完整测试输出见 [cpu_tests.log](step1_review_evidence/cpu_tests.log)。
- 本轮首次新增终态文件比较测试因 Windows 默认编码读取中文 traceback 失败，测试改为显式 UTF-8 后通过。并行 compileall 与 unittest 曾竞争写入 Windows 字节码文件，单独编译重试通过；[首次编译日志](step1_review_evidence/compile_first_attempt.log)保留。Windows TCPStore 的主机名/IPv6 警告后，loopback 联调成功。

本地为 Windows / PyTorch 2.14.0+cpu。Linux pidfd/PDEATHSIG 的真实行为、MPS 权限与隔离、PyTorch 2.11 / NCCL 2.28 的 CUDA abort/rebuild、显存回收及 20/20 仍待服务器执行。CPU 联调不能替代这些验收。历史 [初版证据](step1_evidence/validation.json)保留，不冒充本轮结果。
