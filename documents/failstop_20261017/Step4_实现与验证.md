# Step 4 实现与验证

实现提交：`7b33df6591815bb5db8d98d97a6898bc9ab1c0af`。快速入口固定版本清单提交：`dce3d902edb258d3b028ee8bc333d1448853929f`。分支 `codex/failstop-step4`，从 `codex/failstop` 的 Step 2 交付提交 `69927c7effe6e2ff87133c80c4daaedad30aac89` 建立独立工作树，与 Step 3 并行实现；本分支不包含、不读取 Step 3 提交，合入 `codex/failstop` 由后续单独处理。服务器命令见 [Step 4 服务器命令](Step4_服务器命令.md)。

本轮已阅读 CODEX.md、总计划、Step 4 文档、Step 2 服务器命令和 code/current/AGENTS.md。用户 2026-10-17 明确指示开始实现 Step 4；**standby fix2 的 GPU 复验证据仍未登记**，runner 输出固定 `prerequisites_verified=false`，不把释放已验证写成事实。本轮没有登录服务器、运行 GPU 或改动其他用户任务。

状态：已实现；CPU 已验证；服务器 GPU（emulate/real）与含启动 300 秒快验预算均待实际回传。

## 实现

- `async_state.py`：副本本地状态机 `ReplicaDriver`。每副本自己的 step、EP collective 序号、epoch；`begin_step/commit` 使用票据拒绝过期或重复提交；`fail` 后 EDP0 禁止任何提交；`start_transfer/cutover` 只在切换边界等待迁移句柄、校验 KV 增量与已提交历史一致，再执行校验回调和释放回调，任一失败保留旧布局并进入 `failed`。另有 FCFS 恢复队列（容量不足的请求保持原顺序和提交时间回队首）、`PeakBudget`（源/目标权重、源/目标 KV、接收缓冲、临时算子、余量；预算不足直接报错，不改停服）、同副本 EP 序列审计、CUDA 事件区间重叠和干扰分布统计。
- `async_worker.py`：两模式 CUDA 机制 runner。EDP1 在 EDP0 停服、重建和扩容期间继续在自己的 TP/EP 组推进，每步只等待本副本前向；同副本空 dense DP 分支执行同样的 EP collective。缩容：存活卡保留旧 TP4 分片为源，EDP1 TP4 填充 TP2 目标，EDP0 在途 KV 全部作废后用原 prompt + 已提交历史重建，切换后经 standby 释放路径释放旧 TP4 storage。扩容：空 TP4 目标接收权重与 KV 稳定前缀，EDP0 TP2 持续提交 token，目标就绪后在一致边界补增量、校验、原子切换，再释放 TP2。每阶段记录权重 storage 字节、理论字节、allocated/reserved、设备显存、KV_used 和峰值，并在切换前核对峰值预算。real 模式记录 EDP1 的 abort/rebuild 暂停区间；emulate 不暂停。
- `async_fixture.py`：显式切片传输夹具 `Transfer`，持有专用 stream、源视图和完成状态，暴露 `done()/wait()` 合同供后续 Step 3 适配器替换；逐链路记录字节、提交/完成时刻和 `completion_granularity=batch`。权重和 KV 参考在 CPU 独立生成，测量期间不驻留参考模型。
- `async_topology.py`：real 回归时保留 12 个存活进程的相对 rank 顺序，防止 PyTorch 组内排序改变正在服务的 TP2 分片坐标；worker 显式核对实际 TP2 坐标。
- `async_probe.py`：控制门，复用 Step 1 监管器的 kill/回收/替代 PID 和 Step 2 emulate 释放路径；每阶段校验 PID 身份、实际组映射、同副本 EP 序列/历史一致和空接收目标。
- `run_async.py`：显式 Step 4 套件入口，默认 emulate；`--quick`/`--correctness-only` 共享 GPU，强制 `accepted=false`、`timing_valid=false`；汇总固定写入 `prerequisites_verified=false` 与 `step3_integration=separate acceptance required`。
- `quick_check.sh/quick_driver.py/releases.json`：`--step 4` 由 `releases.json` 取固定实现 SHA，结果标识 `step04_async_driver_and_memory`，握手端口 29879，runner 端口 29880/29881；其余流程与 Step 2 相同（两节点协调同一 suite，300 秒预算，成功失败均切回 `f7d5559`，tar 与 receipt 一起回传）。

## 验证

[追溯记录](step4_cpu_evidence/validation.json)、[单测日志](step4_cpu_evidence/cpu_tests.log)、[packing 日志](step4_cpu_evidence/packing.log)、[dry-run](step4_cpu_evidence/dry_run.log)。

| 检查 | 结果 |
|---|---|
| failstop unittest | 102/102，通过；含 Step 4 新增 16 项（12 项状态机合同、3 项 16 actor 并发协议、1 项 step 4 快验驱动异常路径） |
| 原 NCCL packing 回归 | 16/16，通过 |
| isort / Ruff F / Python 编译 / bash -n / dry-run / git diff --check | 通过；isort 修正了三个新文件的导入顺序后再提交 |
| 服务器 GPU、CUDA 重叠、300 秒实际时长 | 未运行，待回传 |

状态机合同覆盖：故障后禁止提交且 EDP1 独立推进、重复提交拒绝、FCFS 恢复与容量不足回队、快照拒绝外来请求/历史变更、扩容先校验后释放的顺序、校验失败或传输失败均不释放不发布、缺少增量 fence 不切换、预算不足无回退、同副本 EP 序列审计、区间重叠需真实相交、干扰统计需匹配 context。并发协议测试运行真实 controller/supervisor/worker 代码，emulate 连续两轮、real 替代 worker 回归；其中 CUDA 事件为假对象，只检查协议顺序，不代表 GPU 重叠证据。

并发测试期间暴露并修正两处问题：real 回归恢复旧全局 rank 编号会让 PyTorch 组内排序改变存活 TP2 分片的 local rank，已改为保留存活进程相对顺序并核对坐标；重复等待迁移句柄时丢失事件区间，已改为只在首次 wait 转换区间。

本地窄范围命令：

```powershell
$env:PYTHONPATH = "$PWD/code/current;$env:TEMP/weavetp-step2-deps"
$env:OMP_NUM_THREADS = '1'
python -B -X utf8 -m unittest discover -s code/current/tools/resharding/failstop -p 'test_*.py' -v
python -B -X utf8 -m pytest --noconftest code/current/tests/unit_tests/resharding/test_nccl_packing.py -q
```

## 边界

- 本步 runner 使用玩具权重/KV 和显式切片传输，验证副本独立推进、迁移重叠、KV 恢复/增量和单套权重机制；不是 Step 3 混合选源/主机库，也不是 Step 5 DeepSeek 正确性。
- 阶段 A 单请求负载下，恢复容量不足回队逻辑由 CPU 单测覆盖，GPU runner 只执行容量充足分支。
- `accepted` 在本步永远为 false：fix2 GPU 前置与 Step 3 集成分别验收；独占完整矩阵的机制结果写在 `mechanism_matrix_passed`。
- 增量包：`C:/Users/ksqbx/.codex/worktrees/failstop-step4/step4_dce3d90.bundle`，843307 字节，基线 `f7d555998b45756998465924d733fd70e553d6b4`，SHA256 `07047c2ba350c631f4df3c75029eb926a5641a4bc4bb10026e42e36d8b9cecf9`。
