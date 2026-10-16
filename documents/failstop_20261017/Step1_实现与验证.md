# Step 1 实现与本地验证

日期：2026-10-17。分支：`codex/failstop`。范围：独立 real 玩具探针；不加载 DeepSeek，不修改原实验入口、模型、迁移算法或 AnchorTP。开始实施前完整阅读了根目录 `CODEX.md`、本目录总计划、Step 1、`code/current/AGENTS.md` 和贡献规范；后续修改核对同一组规范。阅读文件的 SHA256 保存在 [validation.json](step1_evidence/validation.json)。

**已实现；CPU 已验证；服务器 GPU 未运行；real 集成未放行。** 固定提交和两节点操作见 [服务器命令](Step1_服务器命令.md)。本地 CPU 包为 PyTorch 2.14.0+cpu，不能证明服务器 PyTorch 2.11 / NCCL 2.28 的能力。

## 交付文件

所有新增执行代码位于 `code/current/tools/resharding/failstop/`：

| 文件 | 职责 |
|---|---|
| `run_probe.py` | 显式选择的独立入口；20 次 launch、总超时、最终清理、逐组汇总；另有只读预检和单独的可选 shrink 探针 |
| `controller.py` | SL3060 控制器、独立 store/本地监管器启动、带外同步、epoch/PID/张量验证、计时 |
| `supervisor.py` | 独立 TCPStore 服务或本节点监管器；MPS 检查、worker 启停、四卡稳定回收 |
| `worker.py` | 保留 CUDA 张量，显式 abort 和新前缀重建，逐位校验、本地计算、CUDA all-gather/all-reduce |
| `safety.py` | 只读 `nvidia-smi`/MPS 查询、PID fd 信号、启动时间/launch/角色/GPU UUID 校验 |
| `common.py` | 坐标/rank 映射、控制 store、时钟记录、稳定回收判据和汇总 |
| `test_probe.py`、`test_protocol.py` | CPU 单元测试和真实 TCPStore 联调；不调用 GPU |

没有增加通用故障层或修改默认生产路径。此入口是 Step 1 专用 real 探针，不作为总计划后续默认 emulate 的服务入口。

## 已实现的协议

SL3060 每轮启动独立 TCPStore 进程、控制器和本机监管器；SL3061 由本机 runner 启动本机监管器。每个监管器通过普通独立子进程管理 worker，不使用 torchrun 的故障联动退出策略。两节点每轮检查相同 commit、配置和真实主机名。

探针初始顺序为 node→EP→ETP；按 GPU index 建立稳定 UUID 表，按 EDP0 四个坐标求出目标。控制器打印映射，CUDA all-gather 校验实际 PG 内的 `(slot, PID)` 顺序。这个玩具配置的目标通常是全局 rank 8/9/12/13，但注入器不使用硬编码 rank 列表。Step 2 的模型/异构副本组不在本步建立。

每个 worker 驻留一个 262144 元素的 int64 张量（2 MiB），内容包含稳定 slot 与启动 epoch，CPU 参考独立构造。每个 epoch 检查原 CUDA 指针、全部内容、`x*3+7` 本地算子、实际 world/rank、all-gather 身份、all-reduce 结果。12 个存活进程跨两个重建阶段保持 PID 和张量；4 个回归进程必须是新 PID。

空闲场景在已验证的计算边界注入；在途场景让四个目标暂不提交下一次 collective，其他 12 个提交异步 all-reduce，确认 Work 尚未完成后注入。该构造明确覆盖未完成通信，不将其描述为随机命中已进行到某个 NCCL kernel 的故障。独立通知线程只操作 CPU TCPStore，不等待 GPU。abort 后丢弃未完成结果，再检验原张量和新 world；不在坏组上 barrier。

主路径调用 PyTorch 2.11 的实验性 `_abort_process_group`，使用三个独立 PrefixStore，关闭 worker 的异步错误自动处理和监控退出。每阶段 120 s，启动协调 600 s，整轮 1200 s；任何超时均失败。进程退出和清理另外有有界等待。失败后保留该轮、将后续轮标记 `not_run`，不会以重跑覆盖失败。

依据：[PyTorch 2.11 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/distributed_c10d.py)、[NVIDIA MPS 只读查询](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)。实际构建能力由每个 worker 的 runtime 文件和服务器执行结果证明。

## MPS、回收与计时

MPS 查询保留原始环境，读取真实服务管道的 PID 文件，匹配系统控制进程，再查询全部 server/client；不能定位、权限不足、非零退出码或错误文本均失败。无服务状态必须由进程清单及管道 PID 文件检查证明，不能从一次查询失败推断。存在任何 MPS 客户端或其他 GPU 任务时拒绝本次独占八卡测试。

worker 单独设置每节点/launch 唯一且不存在的 `CUDA_MPS_PIPE_DIRECTORY`，程序不创建该目录。CUDA 初始化后再次查询真实服务，逐 PID 检查独立 GPU context；回归四个 worker 也做同样检查。没有启动、停止、修改 MPS 服务或 GPU compute mode 的代码。

启动前采集三个 GPU 稳定样本；按 UUID 保存原始 XML、设备显存、进程表、采样起止时间及 1 MiB 分辨率。固定默认基线噪声上限 8 MiB、回收阈值为基线最大值 + 8 MiB、目标采样间隔 100 ms、连续三个合格样本、回收超时 120 s。实际间隔包含 `nvidia-smi` 查询耗时，不能宣称实测采样频率是 10 Hz。阈值预先写入配置，失败后不放宽。

kill 的身份核对包括 Popen 登记、Linux `/proc` 启动 tick、launch/角色环境和 GPU UUID，实际信号经 pidfd 发出。旧进程退出码须为 -9。回收和 world12 重建并行；只有四卡均回落且无残留进程后才允许 spawn，spawn 前再查 MPS、进程和阈值。父进程死亡会触发任务子进程的 PDEATHSIG；兜底清理只处理本轮登记 PID，无法确认则失败。成功轮额外要求全部 worker 正常退出、八卡稳定回归基线、最终 MPS 查询通过。

控制器记录本机单调时钟；每轮开始通过五次 TCPStore ping/pong，取最小 RTT 的中点偏移并保存半 RTT 误差。SL3061 校准前的 `t` 为 null，保留 `t_local`；不会作为控制器时间直接相减。跨机估计假设轮内单调时钟偏移固定，误差随结果保存；同节点耗时直接使用本地时间差。spawn 门禁依赖控制器收到回收证据，另以 SL3061 同一时钟核对实际 spawn 晚于确认。

玩具探针的修复等待为 0 s，在 world12 验证后提出 rejoin；不改变正式负载 45 s 规则。完整 `gpu_reclaim_s` 与落入 join 的 `reclaim_wait_s` 分开保存，不把重叠区间相加。EDP1 暂停定义为两次成功验证计算之间的间隔，包含探针的 MPS/控制/spawn 等待，不能当作模型服务暂停或吞吐结果。

## 本地验证与已知限制

`python -m unittest discover -s code/current/tools/resharding/failstop -p 'test_*.py' -v`：**23 tests 通过，无跳过**。其中真实 TCPStore 联调用 CPU actor 覆盖 idle/inflight 两条完整控制协议、epoch 前缀隔离和缺失消息超时。其他覆盖坐标映射、PID/张量身份反例、监管器注入范围、禁止提前重启、MPS 查询失败、pidfd 信号前身份校验、四卡稳定回收、重叠计时和失败样本统计。

`uv run isort`、Ruff F 类检查、Python 编译，以及主矩阵/可选 shrink 的 dry-run 均通过。实际 argv、退出码、源码 SHA256 与原始输出见 [验证记录](step1_evidence/validation.json) 和 [测试日志](step1_evidence/cpu_tests.log)。

初次真实 TCPStore 联调在 Windows 缺少 `signal.SIGKILL` 的测试夹具处失败；测试显式注入 Linux 常量后通过，没有为服务器生产代码添加平台降级。Windows c10d 输出过 IPv6/主机名连接警告，随后 loopback TCPStore 联调成功，原文保留。首次 isort 的 PowerShell 通配符未展开，改为目录参数后成功；Ruff 首次指出两个新文件的未用 import，已删除并复验。

**本地尚未验证 Linux 的真实 pidfd/PDEATHSIG、服务器 MPS 查询权限与隔离实际效果、CUDA/NCCL abort/rebuild、GPU 显存回收、20/20 GPU 成功或实际 shrink 能力。** 所有这些都由服务器 runner 留存证据；不能用上述 CPU 测试或历史其他分支结果替代。`--optional-shrink` 是独立的一轮能力调用，保留 `supports_shrinking`、SHRINK_ABORT 与实际调用状态；不计入 20/20，也不替代主路径。

## 输出与放行

输出位于 `/data/ubuntu/lxh/weavetp/acceptance/step01_real_probe/<suite>/<node>/<launch>/`。每轮包括 `config.json`、`rank_map.json`（SL3060）、`runtime_*`、`worker_*`、`process_lifecycle.json`、控制进程/launcher 生命周期、登记表、MPS 原始响应、基线和回收采样、NCCL 日志、`events_actual.json`（SL3060）、`result.json`。每节点保留 `progress.json`、`summary.json` 和 `summary.csv`。

主验收以 SL3060 `summary.json` 的 `accepted=true` 为必要条件，还需回传两节点完整包，核查 10 idle + 10 inflight、每轮无残留、四卡回收及 MPS 证据。SL3061 本地完成不会自行宣称全局通过。GPU 回传并经用户确认后才放行 real 集成。Step 2–3 公共部分/emulate 可按总计划另外推进；Step 4 仍要求 standby fix2 GPU 复验通过。
