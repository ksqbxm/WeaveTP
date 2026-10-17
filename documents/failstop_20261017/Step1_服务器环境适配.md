# Step 1 服务器环境适配

本次依据用户提供的两节点核查结果实现；未在本地重新验证服务器环境。Step 1 仍为玩具探针，未加载真实权重，未放行 real 集成。

## 修改与边界

- pidfd：优先 Python 原生接口；缺失时仅在 x86_64 Linux 调用 syscall 434（open）、424（send_signal），显式设置 ctypes 参数与返回类型，保留 errno。原生接口调用失败不转 syscall，syscall 失败不转 os.kill。登记 PID/start_ticks/launch/role/GPU 的核对与 fd 退出确认保持。`invocation.json`、每轮 `config.json` 和最终 `result.json` 记录 `pidfd_open_impl`、`pidfd_send_signal_impl`。
- 共享 MPS：允许 foreign 客户端，保留查询到的 PID。无法定位真实管道的 daemon 写入 `skipped_daemons`；不可读 PID 文件、未查询 server 分别记录，`query_complete=false`、`state=partial`。不伪造空查询结果。已定位管道的命令失败/错误响应仍失败；独占模式保持完整查询及客户端拒绝。
- worker 证明：首次及回归 worker 均在初始化前使用不存在的独立管道；完成 CUDA 张量分配和同步后，在 worker 内再次确认管道不存在，并查询完整 GPU 进程表。自身 PID 必须恰好出现在指定 UUID 的计算进程条目中，原文存入 `context_<slot>_epoch<epoch>.jsonl`，证明写入 runtime 和 cuda_ready 事件。监管器继续核对本测试 PID/UUID，查询到本测试 MPS 客户端即失败。
- 证明依据：NVIDIA 说明 MPS 的 GPU context 由 server 持有，nvidia-smi/NVML 显示 server 而非客户端 PID。因此“无效管道 + 实际分配后自身计算进程 PID/UUID”用于证明本 worker 独立 context，不能只凭环境变量判定。[NVIDIA MPS 接口说明](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)。该验证依赖服务器 GPU 进程表可见；不可见则失败，不降级。
- `/data`：`--min-free-disk-gb` 默认 5，单位为十进制 GB，允许有限非负数。独立预检与每次 launch 创建子进程前检查；打印并保存总量、已用、可用字节。空间不足不启动本轮 store/controller/worker。此检查是启动门槛，不预留磁盘空间。
- Step 3：仅更新计划，要求加载前核对实际 `/dev/shm` 空间，支持完整库/按需分片库并报告各模式 backing、峰值和注册占用；本次没有实现或加载主机库。

## 验证

50 项 CPU 单测通过，无跳过。新增覆盖原生/混合/syscall 两接口、errno 与架构限制、foreign MPS 客户端及部分 daemon 管道未知、worker context 证明、磁盘阈值和不足时阻止启动。原先共享显存波动、按进程回收、PID 身份、清理、真实 CPU TCPStore 协议测试继续通过。

首次运行因旧 timeout 测试缺少新增配置字段报错；补齐 fixture 后完整重跑通过。执行 isort、Ruff F 检查和 compileall；默认/共享 dry-run 均生成 20 轮，非法磁盘阈值拒绝。日志与源码 SHA256 见 [本次证据](step1_server_environment_evidence/cpu_tests.log)。本地为 Windows/CPU，未执行 Linux 真实 pidfd、服务器 MPS/CUDA/NCCL 或 20 轮 GPU 测试。历史 40 项测试报告对应前一提交，不替代本次证据。

交付命令见 [共享模式服务器命令](Step1_共享GPU服务器命令.md)；分发使用 SL3061 GitHub 拉取、增量 bundle 转交 SL3060。
