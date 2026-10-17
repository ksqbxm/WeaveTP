# Step 2 实现与验证

本页保留首次实现的历史记录。本轮审阅修正版及新增测试以[审阅与修正](Step2_审阅与修正.md)为准；服务器请使用[当前固定命令](Step2_服务器命令.md)。

实现提交：`9ca323f3b97dc4830468cf9ebabfb5f3a8d254c6`。快速入口固定版本清单提交：`566cf8e65ffcf547bb83e98ed88ed1f81b6c3f99`。

本轮用户明确要求开始 Step 2，取代此前“只优化 Step 1”的暂停范围。依据总计划登记的 d16b5e8 共享 20/20 已确认关口，接入 real；没有把尚未回传的两阶段优化 GPU 快验写成通过。本次已阅读 CODEX.md、总计划、Step 2、code/current/AGENTS.md 与贡献说明，原工作目录及 AnchorTP 实现未改动。

状态：公共接口、emulate、real 玩具集成和快速入口已实现；CPU 已验证；服务器 GPU 与含启动 300 秒预算均待实际回传。没有 SSH、部署或占用服务器 GPU。

## 实现

- `megatron/core/failstop.py`：公共 FaultSession/FaultBackend、fail/rejoin/current_world/rank_map、launch/round/epoch/id 校验；normal→failed/recovering→degraded→rejoining/expanding→normal；缓存键含稳定设备、有效 membership、实际 world/local rank。旧 work 先完成或在 abort 后丢弃，禁止新 world 使用旧句柄。
- 同文件和 `megatron/rl/parallel_utils.py` 的新入口：显式 rank 列表构造独立 TP/EP/DP 等组，当前 world 所有进程按相同次序建组，同一次构造内部复用相同成员组。原 `build_inference_pg_collection` 函数体未改变。
- 每 EP 枚举 local-rank 分配，按实际权重字节区间交集选择，稳定设备标识破并列。玩具权重用独立明确的 expert 身份与字节区间；保留原 TP4 源坐标。PyTorch 2.11 的 new_group 会排序，记录 requested/actual 并据实际顺序更新映射；实际顺序改变时失效缓存。[PyTorch 2.11 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/distributed_c10d.py)是本次核对依据。
- `failstop_rebind.py`：模型及 named_modules、非 nn.Module 的 token_dispatcher 中的 pg_collection 和已有组对象引用；TP/EP rank/size 缓存；清理本次路由临时缓存。前向 hook 和同一互斥边界防止重绑定/abort 与前向并发；逐参数和 buffer 校验 data_ptr。保留 vocab 范围、每分区形状、attention heads、expert 索引和源分片元数据。旧 TP4 源绑定 TP2 时只能用于迁移，前向被禁止；目标模型另外分配。CUDA graph / 外部 dispatcher handle 不静默复用。
- `NCCLCopyService.group_relative_ranks` 默认 false；仅 fail-stop factory 开启，P2POp 用 group_peer。这样 emulate 降级后的非连续 12-rank 子组，其占位 ring 不会误用全局 rank。factory 拒绝失败设备和包含失败设备的数据组；原入口与默认行为不变。
- `topology_worker.py/topology_probe.py/topology_fixture.py/run_topology.py`：两模式同一玩具 MoE、逐位传输、映射、重绑定和 collective 验证；real 重用 Step 1 监管器 kill/reclaim/替代 PID 策略，每 epoch 使用新 store prefix 实际 abort/rebuild。emulate 预建组，失败设备释放模型并只参加 CPU 控制；回归接收对象由 sentinel 空张量构造。回归者不发送模型数据。健康副本独立执行自己的 TP/EP 前向，同副本空 DP 与非空 DP 保持 EP 次序。
- `quick_check.sh/quick_driver.py/releases.json`：每节点一次 nohup 启动，SL3060 协调唯一 suite，等待对端最多 60 秒。两种模式各一轮，缩/扩均逐位检查，默认共享 GPU；总预算 300 秒，超时失败且清理。固定检出实现 SHA，成功/失败都恢复 f7d555998b45756998465924d733fd70e553d6b4。已跟踪文件必须干净，未跟踪文件不阻断。结果 tar 和最终 receipt 必须一起回传；tar 内 quick_status 明示 PENDING_FINALIZATION，避免打包/对端失败后留下虚假的最终 PASS。

本步玩具目标由健康 GPU 的明确切片填充，验证真实 NCCLCopyService 及严格逐位结果；不声称已实现 Step 3 的主机库和混合选源，也不声称完成 Step 4 异步迁移或 Step 5 DeepSeek 模型验收。真实模型中的重绑定字段已按上述源码路径梳理，当前 GPU runner 使用玩具 MoE。

## 验证

[可追溯记录](step2_cpu_evidence/validation.json)、[单测原始日志](step2_cpu_evidence/cpu_tests.log)、[真实 Gloo 日志](step2_cpu_evidence/check_0.log)。

| 检查 | 结果 |
|---|---|
| failstop unittest | 75/75，通过；含 Step 1 原 54 项 |
| 原 NCCL packing 回归 | 16/16，通过 |
| 4 个真实 CPU/Gloo 进程 | 4/4，通过：组顺序、collective、指针重绑定、非连续子组 P2P |
| 实际 controller/worker 算法的 CPU 并发替身 | emulate 连续两轮、real 替代 worker，一致通过；不代替 NCCL/SIGKILL |
| 快速入口异常路径 | suite 失败、超时、dirty、恢复失败、打包失败、对端失败均 FAIL |
| isort / Ruff F / Python 编译 / bash -n / dry-run / git diff --check | 通过 |
| 服务器 NCCL/GPU / 300 秒实际时长 | 未运行，待回传 |

本地窄范围命令：

```powershell
$env:PYTHONPATH = "$PWD/code/current;$env:TEMP/weavetp-step2-deps"
$env:OMP_NUM_THREADS = '1'
python -B -X utf8 -m unittest discover -s code/current/tools/resharding/failstop -p 'test_*.py' -v
python -B -X utf8 -m pytest --noconftest code/current/tests/unit_tests/resharding/test_nccl_packing.py -q
python -B -X utf8 code/current/tools/resharding/failstop/cpu_topology_gloo.py
```

本地最初缺少 einops，已用 uv 安装到临时目录 `%TEMP%/weavetp-step2-deps`，未修改服务器环境。第一次新单测的 mock 缺少 initialized/subgroup 配置，修正后通过；Ruff 最初指出本次新增的未使用 import/变量，已删除并复验。Windows TCPStore 的 IPv6 警告与缺少 TE/Apex 的提示保留在日志中。首次 Git fetch 因本机 127.0.0.1 代理不可达失败，随后仅在该条命令清空代理环境并用 `git -c http.proxy=` 成功 fetch，确认远端未分叉；未改永久代理设置。

服务器操作见 [Step 2 服务器命令](Step2_服务器命令.md)。Step 4 仍受 standby fix2 GPU 复验关口限制。
