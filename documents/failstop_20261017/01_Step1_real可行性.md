# Step 1：real fail-stop 独立可行性验证

状态：已实现、CPU 已验证、服务器 GPU 待验收。测试标识：`step01_real_probe`。

## 1. 必读与范围

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 每次代码修改前核对本 Step、总计划及适用 AGENTS.md；Python imports 变动须执行 `uv run isort`。发现实现与已确认设计不符，报告差异再确认。

本步只用玩具张量/模型验证真实 kill 后存活进程继续使用 CUDA、重建 world 和重新加入的可行性；不接入 DeepSeek、模型权重迁移或新服务驱动。完成 GPU 验证后等待用户确认，才集成 real 到 Step 2–3。交付本步后，用户跑服务器期间可继续 Step 2–3 公共部分与 emulate。

## 2. 输入及固定放置

- 基线 `6b0a2b747cc6742ae2b4ec7a41774f62b0419617`，工作分支 `codex/failstop`。两节点各 8 GPU，EDP0=SL3061，EDP1=SL3060。
- SL3060 上独立 TCPStore 服务和控制器，SL3060/SL3061 各一个监管器；不使用会在任一 worker 死亡后杀掉全部兄弟进程的默认 launcher 故障策略。
- SL3061 监管器按配置计算的 EDP0 四故障坐标 kill 指定 PID；PID/启动时间/GPU UUID 在发信号前校验，避免杀错 PID。新四 worker 由该监管器本地启动。
- 控制器、TCPStore、监管器无 GPU 模型状态且不在 kill 集合。SL3060 8 个、SL3061 4 个存活 worker 跨 epoch 保持 PID 和张量。
- 记录 Python、PyTorch 实际构建版本、CUDA runtime/driver、运行时 NCCL、节点/GPU/进程映射和所有通信相关环境变量；目标核查 PyTorch 2.11、NCCL 2.28，不能以本地 CPU 包证明服务器能力。

## 3. 实施任务

### 3.1 MPS 预检与本任务隔离

1. 在两台机器记录进程列表、GPU 进程、MPS 控制进程/server/client 列表及查询返回码。按安装版本选择只读查询；旧接口用 `get_server_list`、对每个 server 用 `get_client_list <PID>`。保存原始管道路径和结果。
2. 查询使用真实服务管道；worker 的环境单独设置。不能只 unset 环境变量，因为客户端可能连接默认管道。
3. 给本次 worker（包含回归新 worker）设置节点/launch 唯一、确认不存在且不创建的 `CUDA_MPS_PIPE_DIRECTORY`，在首次 CUDA 初始化前生效，以绕开已有服务。报告原设置是否会连接 SL3060 MPS、应用了何种隔离及前后结果。
4. worker CUDA 初始化后再查 server/client 列表，确认本次所有 worker PID 均未连接，并记录其独立 GPU context。环境设置只是措施，实际查询才是验收证据。查询权限不足、管道无法定位、隔离后 CUDA 不可用均停止本 case 并报告，不修改服务或 GPU compute mode。
5. 不发 quit、不 start/stop server、不杀 MPS，不调整别人的客户端。既存客户端若占用本次 GPU，预检拒绝该 case 并报告，不驱逐。

NVIDIA 文档说明管道变量控制连接位置，并列出绕开 MPS 的设置；实际效果仍须在服务器构建验证：[环境变量](https://docs.nvidia.com/deploy/topics/topic_5_2_2.html)、[绕开 MPS](https://docs.nvidia.com/deploy/mps/595/when-to-use-mps.html)、[只读控制接口](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html)。

### 3.2 16→12→16 生命周期

1. worker 启动前记录每 GPU 基线显存和进程；各监管器向控制器报告 ready，独立 TCPStore 可用后启动 16 worker。
2. epoch 0 建 16 world，生成可重复且每 rank 不同的驻留张量；保存张量指针、shape/dtype、逐位参考并运行一次计算及 collective。
3. 注入 kill -9 后由 SL3061 监管器记录信号与退出确认；控制器通过独立 TCPStore 发布 fail/membership/epoch。控制消息不使用已损坏的 NCCL world，通知线程不能被 GPU wait 阻塞。
4. 存活进程显式 abort 旧通信器；记录每组/每进程开始结束和未完成 work 的处理。使用新 PrefixStore 命名空间建立 12 rank world，所有新 rank 连续且映射明确。使用带外同步协调重建，不在坏组上 barrier。
5. 验证 12 个存活 PID 未替换、原 CUDA 张量指针和逐位内容保留，继续做本地算子以及新 world collective，结果正确。
6. 并行采样四故障 GPU 回收状态，满足第 3.3 节条件后才允许 SL3061 监管器启动四新进程。玩具探针不必等待正式负载的 45 s，但必须记录等待策略；正式入口沿用总计划 45 s 规则。
7. 发布 rejoin epoch，存活进程退出旧 collective 流、abort 12 world，16 个进程以另一新前缀重建 16 world。原 12 张量仍可用，新 4 有新 PID/新张量；再次校验计算和 collective。
8. 有界退出，收集所有进程结果，只清理本次登记资源。下一次启动不复用旧 store 前缀或错误 PG。

必须覆盖空闲边界与未完成 collective 场景，避免只证明安全停在 barrier 时可重建。实现先用小张量、最少组，再加可控在途 collective；不以全进程重启替代存活者保留状态。

### 3.3 显存回收判据与计时

- 基线来自本轮 worker 启动前，按 GPU UUID 采集多次稳定样本，保存设备 used memory、进程清单、采样分辨率。不可用 kill 前包含玩具张量的显存作“空卡基线”。
- 记录每个故障 PID 已退出，GPU 进程表无本次残留 worker；设备显存回到基线范围。允许量必须在 case 前写入配置，结合采样粒度/稳定基线噪声确定，不能见到回收失败后临时放宽。保留全部采样用于复核。
- 建议初版 100 ms 采样、连续 3 个合格样本确认稳定，超时值写入运行配置。它们是实现默认值，不是已实测延迟。
- 每卡 `gpu_reclaim_s = t_reclaim_confirmed - t_kill_sent`，整体取最后一张卡确认时刻减故障命令时刻；保存基线范围、阈值、观测值和是否有非本任务进程。
- 新进程不能提前启动。出现残留或显存未回落，标记失败并输出进程信息，不 reset GPU、不杀其他进程、不重启 MPS。
- `rejoin_join_s` 从 rejoin 就绪请求到新 16 world 可用，包含尚需等待回收、spawn、注册及 join/rebuild。其结构化子项保存完整 `gpu_reclaim_s` 区间与实际落入 join 的 `reclaim_wait_s`。回收早于 rejoin 请求时 latter=0；full reclaim 不重复加到 join 墙钟。重建阶段指标可重叠，保留区间而非盲目求和。

### 3.4 shrink 能力核查

核查 `torch.distributed.shrink_group`、`SHRINK_ABORT`、后端编译支持及实际调用；仅 Python 属性存在不足以判定可用。单列可选 shrink 探针与结果，不能替代必须通过的 abort + 新 store 前缀重建主路径。若 shrink 不支持则明确说明，不虚构 C++ binding 或将另一版本能力冒充实际构建。

官方 PyTorch 2.11 源码包含 shrink 接口和实验性 `_abort_process_group`，后者建议 `TORCH_NCCL_ASYNC_ERROR_HANDLING=0`；实施时核查并记录实际环境。[PyTorch 2.11 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/distributed_c10d.py)。NCCL API 说明见 [2.28 communicator 文档](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2287/user-guide/docs/api/comms.html)。重建具有环境可行性风险，因此必须先走本步关口。

## 4. 本地验证

窄范围检查坐标映射、rank 压缩/恢复、唯一 store 前缀、监管器状态机、超时退出、MPS 查询失败不当作空列表、显存阈值与稳定样本、回收等待重叠计时。fake store/process/NVML 的测试只证明控制逻辑；文档明确不能证明 CUDA/NCCL 可行性。校验只会向登记的本任务 PID 发信号。

## 5. 专用服务器 GPU 验收

在两台 8 卡机器共执行 **20 次独立 launch**，建议固定 10 次空闲边界 kill + 10 次在途 collective kill。每次都经过完整 16→12→16，并校验：

- 四个目标 PID 确实由 SIGKILL 退出；其余 12 PID 不变；无 watchdog 驱动的全局进程退出。
- MPS 查询完整，所有初始及回归 worker 均未连接 MPS。
- 每次存活张量地址、逐位内容与新 collective/算子正确；world size/rank 映射正确且前缀不同。
- 四卡回收均通过；spawn 时间不早于各自回收完成；记录所有阶段计时及 EDP1 暂停。
- 20/20 成功、无卡死、无残留本任务进程。任何超时、崩溃、跳过、张量损坏算失败；不能仅重跑成功样本覆盖失败历史。

结果按 idle/inflight 分组汇总 count、均值、样本标准差、最大值，并保留 20 次逐阶段数据。失败首先定位最后成功阶段和相关 PID；本步失败不阻挡公共部分/emulate，但不能放行 real 集成。

## 6. 交付与关口

交付 runner、场景配置、runtime 探针、进程生命周期表、GPU 回收原始样本、MPS 原文、rank/epoch 映射、每次阶段 JSON、汇总和测试报告；输出 `/data/.../step01_real_probe/<launch_id>/`。报告明确每进程所在节点、生命周期和新旧 PID。

实施交付必须补齐两节点 `source env.sh`、fetch/checkout 完整 SHA、先启动 SL3060 store/监管器再启动 SL3061 监管器的命令、nohup 日志、打包回传路径、结束后切回 `f7d5559`。脚本未实现前不提供假定能运行的命令。

**用户回传并确认通过后，才开始 real 路径集成；本 Step 交付即继续 Step 2–3 的公共部分与 emulate。Step 4 另受 fix2 GPU 关口限制。**

## 7. 验收登记

2026-10-17：独立 real 玩具探针已实现；23 项 CPU 测试通过，无跳过；isort、Ruff F、编译及 dry-run 通过。实现与证据见 [Step1_实现与验证.md](Step1_实现与验证.md)，固定交付 SHA 与两节点命令见 [Step1_服务器命令.md](Step1_服务器命令.md)。

当前审阅修正版实现 SHA：`ca72cd23d2e6a92cb58e38b6b3ea8b91936ca254`。31 项 CPU 测试通过，无跳过；通知、MPS、进程清理及终态验收问题已修正，见 [审阅报告](Step1_审阅与修正.md)。后续服务器命令文档提交不改变实现；服务器固定检出此 SHA。

GPU：未运行。20 次结果：待回传。Linux PID/MPS/CUDA/NCCL 可行性：待服务器验证。用户 real 集成确认：待确认。
