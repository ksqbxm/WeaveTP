# Step 3 实现与验证

实现：`bcc61aeb86ddc1bcce403785493ac38ad59486db`，分支 `codex/failstop`。本轮已在写代码前阅读 CODEX.md、总计划、Step 2/3 文档和 code/current/AGENTS.md；修改 imports 后执行 `python -m uv run isort`。

审阅修正后的当前实现为 `7e61596114b003bf3ecefdd8045a6b10aeeda65b`，问题清单、修正与复验见[审阅与修正](Step3_审阅与修正.md)；本文件下方表格已按修正后的路径更新，历史数字保留。

状态：公共/emulate/real 的 Step 3 独立切片测试路径已实现，CPU 已验证；服务器 CUDA/NCCL、profile、实际 300 秒快验预算均未运行。用户本轮明确要求实现 Step 3 并核查 real 回归，因此实现 real 测试路径；不据此登记 Step 2 real GPU 或 Step 3 real GPU 已通过。Step 4 未启动，fix2 GPU 回传关口不变。

## 两阶段回归核查

审阅起点 `69927c7` 中，Step 2 real 已具备“spawn → CUDA/MPS 验证 → transition/2 → abort/build”的顺序；存活 worker 在 service_until 中持续执行自己的副本算子。因此后台启动/CUDA 初始化已不在 EDP1 重建停顿内。旧 Step 2 缺少该语义的汇总指标和准备期间进度验收。

本次补充：重建前后必须观察到存活副本服务步数增加，回归 worker 所有初始化 CUDA 操作同步完成后才发布 ready；world16 和新组验证后，EDP1 先执行一次正确前向，再运行同步迁移探针。`edp1_rejoin_pause_s` 是各 EDP1 worker 从重建前最后一次正确服务到这次恢复服务的本地单调时钟间隔的最大值。两节点 summary/快验 receipt 和最终单行均输出该字段，缺失或非有限值不得 PASS。suite 汇总取 real launch 最大值。

完整的同步迁移探针校验间隙另记 `edp1_rejoin_validation_gap_s`；`rejoin_validation_wall_s` 包含完整探针，`rejoin_join_s` 到 world16 服务恢复通知为止。Step 3 仍是同步传输正确性探针，迁移期间完整异步服务驱动属于 Step 4，不能把本步的恢复服务采样解释为已完成 Step 4。Step 1 的历史字段定义及历史结果不改写。

## 变更与边界

| 文件/入口 | 交付 |
|---|---|
| `megatron/core/resharding/planner.py` | 本地 src==dst 不进入 L/S/D，保留实际传输任务 |
| `mixed_sources.py` | 全局矩形切片、dtype/形状/参数/global expert 身份；混合 TP4/TP2/host 精确覆盖；host 计 D/M、不计 GPU S；复用 ResidualBandwidthWaveScheduler 的有界分波分配，无在线 replan/新 packing |
| `host_weights.py`、`host_owner.py` | 每节点独立 CPU owner、单份共享 mmap；完整/按需清单；RAM 与实测 /dev/shm 预算，memlock 记录为证据；共享文件页的 CUDA 注册探针先于读权重；直接 readinto backing，无完整 state_dict 临时副本 |
| `mixed_transport.py` | 专用 H2D stream 与现有 NCCL streams 并发；非连续目标用 GPU staging、host 按连续行异步复制；所有流完成后才放行目标；按接收端提交到完成的 batch 墙钟记录实际 payload/control/reuse bytes |
| `source_profile.py`、`run_sources.py --profile-only` | 单卡 H2D、每节点 4/8 卡并发 H2D、240 条有向机内/跨机 P2P；三种 payload、原始样本、拟合方法、硬件身份/状态证据、runtime/源码 SHA、真实 profile 文件 SHA256；独立采集模式不注入故障 |
| `source_fixture.py`、`source_runtime.py` | 独立索引 checkpoint oracle；GPU-only/host-only/同目标互补混合来源，非连续切片和两个 global expert；新 real PID 只读结构建空目标，初始权重 H2D=0，不作源 |
| `run_sources.py`、`quick_check.sh --step 3` | 复用 Step 2 监管/拓扑/清理，默认模式 emulate；快验两模式各一次缩容/扩容，默认共享正确性模式，固定 SHA、自动协调 suite、nohup、打包、恢复原正式提交 |

主机库输入是显式索引的原始分片 checkpoint，入口生成独立的小型整数权重 fixture；本步没有实现 native DeepSeek 整模型加载/推理，该关口仍在 Step 5。按需模式在读取权重文件前固定覆盖清单，只读可能使用的 host-only 和 mixed 分片。完整模式加载全部 fixture 分片；两种模式均报告预计/实际 backing、每 worker 注册量、可用 RAM、/dev/shm、memlock 和 owner VmHWM。服务器正式入口默认 on-demand，可显式选择 full。

共享 backing 使用可注册的 writable MAP_SHARED 映射，应用层只作为源，退出时 owner 按 checkpoint SHA256 逐片检查未修改；不使用私有/COW 映射。原因是 OS 只读 mmap 的 CUDA 注册需要可选设备能力，不能假设 GeForce 支持。这里的“权重只读”是源对象使用合同和完整性校验，不声称 OS 页保护。[CUDA Runtime 官方说明](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-runtime-api/group__CUDART__MEMORY.html)。worker 正常退出只 unregister/detach；owner 在所有 worker 结束后校验并 unlink。异常回收只处理本 launch 登记的路径和 inode。

profile 只包含实际测得的远程条目，缺项直接失败；矩阵保持 decimal Gbps 输出，内部为 bytes/s。对角线明确标记未测量，且本地成本不读取该估计。主机 M 取 4/8 并发实测结果的保守上限，绝不以单卡乘八。共享模式 profile 仅用于正确性测试，timing_valid=false，不作性能结论。

迁移计划的 cache key 包含 membership、epoch、实际 rank map、源覆盖/布局和 profile hash。每次快照使用同一 profile，分别回放 emulate 原始 rank 与 real 压缩 rank，逐项比较稳定源/目标、切片、字节和波次；实际两次独立 launch 的 profile 可以不同，不能伪称独立运行数值相同。

## 本地验证

证据见 [validation.json](step3_cpu_evidence/validation.json)，均在 Windows、Python 3.13.12、Torch 2.14.0+cpu 执行；服务器入口保留已验证的 PyTorch 2.11 / NCCL 2.28 版本检查。

```powershell
python -B -m unittest discover -s code/current/tools/resharding/failstop -p 'test_*.py' -q
$env:PYTHONPATH = (Resolve-Path code/current).Path
python -B -m pytest --noconftest code/current/tests/unit_tests/resharding/test_nccl_packing.py code/current/tests/unit_tests/resharding/test_source_search.py -q
python -B code/current/tools/resharding/failstop/cpu_topology_gloo.py
```

- 99 项 failstop 测试通过，包括实际控制器/worker 算法的并发 CPU 替身、emulate 连续两轮、real 新 PID 语义、实际混合 planner/transport、逐位和字节总量检查。
- 56 项 packing/选源回归通过；本地代价修复涉及的历史断言已更新，远程输入仍要求与冻结历史 planner 的 plan 和 stats 完全一致。
- 4/4 真实 Gloo 进程通过；22 个变动 Python 文件 AST、isort、bash -n 和最终 diff 检查通过。
- 初次完整测试因本机缺少 einops、psutil、scipy 失败，补齐依赖后通过；早期 CPU 替身接线错误已修正。上述问题未被作为 skip/pass 隐藏。

CPU 替身 profile 为合成输入，CUDA 注册和 NCCL 由替身替代；这些结果只证明协议、布局、完成边界和内容逻辑。GPU runner 不使用这些合成 profile。

服务器命令、固定快验清单与 bundle 见 [Step3_服务器命令.md](Step3_服务器命令.md)。服务器回传前，任何 GPU/性能/五分钟实测结论均为待验证。
