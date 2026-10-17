# Step 1 共享 GPU 正确性模式

2026-10-17 按用户明确要求新增 `--allow-foreign-gpu-processes`，默认关闭。**已实现，40 项 CPU 测试通过；服务器 GPU 未运行。** 修改前已阅读 CODEX、总计划、Step 1、AGENTS 及贡献规范。默认独占模式继续要求 GPU 空闲和整卡显存回落，正式验收口径保持不变。

固定实现提交：`d69d5950819626b0cbd09bfa4e32ed4ebbe9396c`。两节点可复制命令见 [共享 GPU 服务器命令](Step1_共享GPU服务器命令.md)。

## 行为与证据

- 预检和每轮 config 保存 `shared_gpu_mode`。两节点模式不一致，在启动 worker 前失败。
- `preflight.json` 的 `foreign` 按 GPU UUID 保存 PID、名称、进程显存 MiB；MPS 空闲 server 也记录。`used_memory_raw` 保存原始值，N/A 记为 null，不能写成 0。
- 后续检查保存 `foreign`、相对预检集合的 `new_foreign`。已有和新外部进程都不会导致共享模式失败；本测试 PID 必须出现在指定 UUID 上，缺失、错误 UUID、重复设备分配或已登记旧 worker 残留均失败。
- MPS 查询、客户端拒绝条件和 worker 独立管道隔离不变。已有 MPS 客户端仍会使预检失败。
- 整卡基线只记录范围，`threshold_mib=null`；不因其他任务显存波动失败。kill 后和最终退出均检查全部已登记本测试 worker PID。进程已经退出且目标 GPU 表中没有本测试 PID，意味着表中本测试进程显存合计为 0；连续三个合格样本确认。仍有 PID 条目时，即使占用为 0 或 N/A 也不能通过。
- `gpu_reclaim_samples.jsonl`、`cleanup_gpu_samples.jsonl` 保留整卡数据及每 GPU 的 `task_usage`（本测试 PID 列表、显存合计）。spawn 前复查继续阻止本测试进程残留，但忽略外部显存增长。最终清理包含旧 PID 和重新启动的新 PID。
- kill 和异常清理只来自登记表；foreign 清单不进入信号目标。现有 `signal_registered` 的启动时间、launch、角色、GPU UUID 核对不变。

最终 `result.json`、`events_actual.json` 和 `summary.json` 标记 `shared_gpu_mode=true`。JSON summary 和 CSV 的 `timing_note` 明确写入：**与其他任务共享 GPU，仅用于正确性和可行性判断，各阶段耗时不作为正式数据**。耗时保留供诊断，不作为通过门槛。共享模式的 `accepted=true` 表示 20/20 正确性与可行性通过，不能充当独占 GPU 正式数据；SL3061 仍不自行给出全局 accepted。

通过仍要求四个登记目标真实 SIGKILL、12 个存活 PID 不变、驻留张量地址与内容不变、本地计算和新 world collective 正确、完整 16→12→16、MPS 检查及两节点清理确认。任一失败或 not_run 都不能通过，不自动重试覆盖失败历史。GPU 结果仍需用户回传确认后才放行 real 集成。

## CPU 验证

`python -B -X utf8 -m unittest discover -s code/current/tools/resharding/failstop -p 'test_*.py' -v`：40 项通过，无跳过。新增九项测试覆盖共享预检、运行时新外部进程、严格 PID/UUID 核对、MPS 拒绝、基线/进程回收、foreign 显存波动、summary、XML 进程显存解析、两节点模式一致性及完整本机监管器流程（其中部分在同一测试中联合覆盖）。真实 TCPStore 控制器联调覆盖默认/共享 × idle/inflight 四种组合。

isort、Ruff F 检查、Python 编译、默认 dry-run、两节点共享 dry-run、共享可选 shrink dry-run、diff 检查均通过。Windows TCPStore 仍有既有的主机名/IPv6 警告，随后 loopback 联调成功，无测试失败。命令、退出码和源码散列见 [validation.json](step1_shared_gpu_evidence/validation.json)，原始输出见 [cpu_tests.log](step1_shared_gpu_evidence/cpu_tests.log)。

这些测试使用 CPU、mock GPU/进程和真实 CPU TCPStore；尚未运行服务器 MPS/CUDA/NCCL/真实 pidfd 或 20 次 GPU launch。
