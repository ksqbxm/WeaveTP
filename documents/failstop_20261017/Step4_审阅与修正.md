# Step 4 审阅与修正

实现：`b621125876e8ea8ab07c1f3ba210ddd370a0ae60`。快验清单：`e09d7bce466fb3f7bbd4afc309860898873c3c2a`。审阅对象为 `codex/failstop-step4` 上 Step 4 的全部改动（状态机、worker、夹具、探针、套件入口与快验修正）。已重新阅读 CODEX.md、总计划、Step 4 文档、code/current/AGENTS.md。服务器命令以[当前交付说明](Step4_服务器命令.md)为准。

状态：已实现、CPU 已验证；服务器 GPU 与 300 秒实际预算待回传。fix2 GPU 前置仍未登记。

## 审阅发现与修正

| 级别 | 原问题与影响 | 唯一生效路径及验证 |
|---|---|---|
| P1 | real 模式回归的替代 worker 走 `transition()`，在没有任何进程组时调用 `c10d._abort_process_group()`；PyTorch 2.11 对不存在的默认组直接抛 AssertionError，替代进程会在服务器上立刻崩溃。CPU 协议测试把 abort 打桩为空函数，没有暴露。Step 2 的 topology_worker 替代路径没有这个问题。 | 替代 worker 不再进入 `transition()`，直接在 rejoin 事件上推进 session 并等待 build；`ReplicaDriver.join/follow` 显式表达“空接收者加入降级副本并采用其已提交位置”。协议测试的 abort 桩现在要求调用线程已有进程组；用旧 worker 复跑该测试在 slot 13 替代进程处失败（`abort without a process group`），新 worker 通过。 |
| P1 | real 模式 EDP1 的 abort/rebuild 暂停区间 `end_local` 在 migrate 末尾才写成最后一步的完成时刻，把整个服务期都计入暂停，GPU 验收第 5 项会得到严重高估的暂停时长。 | 暂停在 `transition()` 中以 abort 前最后一次提交完成时刻开始，在新 world 上第一次 `step()` 提交时以其 `submitted` 结束；只有 real 且 healthy 的 rank 记录。 |
| P2 | `cutover` 使用 `wait() is not False` 判定传输失败，这是测试虚构的合同；真实 NCCL 句柄失败是抛异常，返回 None 恒通过。 | 直接调用 `wait()`，异常传播，副本保持旧布局进入 `failed`。测试改为句柄 `wait()` 抛异常。 |
| P2 | 缩容阶段把仅作传输源的旧 TP4 模型重绑到 TP2 组，并记为 `source_rebound`，与“源只用于迁移、前向禁止”的语义相悖且无用。 | 只重绑继续服务的模型（EDP1 各阶段、EDP0 TP2 在扩容期），事件改名 `serving_model_rebound`；TP2 坐标核对保留。 |
| P3 | worker 中 `recovery_queue([request], [], len(request.tokens))` 容量等于请求本身，恒成立，不构成检查；扩容阶段 `snapshot.delta` 在 cutover 前重复校验；victim 用 JSON 往返比较请求，并重复手写 `Request(**{...})` 转换。 | 删除无意义调用（容量不足逻辑由 CPU 单测覆盖）；删除重复校验，cutover 是唯一校验点；新增 `Request.from_dict`，直接比较 dataclass。 |
| P3 | `run_async.py` 原样复制 `run_topology.py` 的 `cases/summarize`；`distribution` 在函数内 import；`Transfer.piece` 冗余变量、`wait()` 返回无意义布尔值。 | 复用 `run_topology.cases/summarize`；import 上移；删除冗余。 |
| P3 | 扩容期重叠断言 `overlap > 0` 是硬失败项，但默认权重 4 MiB 的 TP2→TP4 传输可能在第一步前向开始前就结束，造成与机制无关的误判。 | `--weight-elements` 默认提高到 8388608（32 MiB float32），使传输跨越多个 decode step；预算与临时量按实际尺寸计算，2 GiB admission 预算内。 |

未发现 `async_probe.py` 控制门与 worker 协议错位；emulate/real 的组顺序、TP2 坐标保持、EP 序列审计、先全体校验再释放、单套权重断言逻辑复核无误。

## 验证

| 检查 | 结果与原始证据 |
|---|---|
| failstop unittest | 104/104，通过，含本轮新增 join/follow 用例与 abort 守卫；[日志](step4_cpu_evidence/cpu_tests_review.log) |
| 旧 worker 回归复现 | `test_real_replacements` 在替代进程处失败 `abort without a process group`，新 worker 通过 |
| isort / Ruff F / git diff --check | 通过 |
| 服务器 GPU / 300 秒实际时长 | 未运行，待回传 |

增量包：`C:/Users/ksqbx/.codex/worktrees/failstop-step4/step4_e09d7bc.bundle`，873581 字节，基线 `f7d555998b45756998465924d733fd70e553d6b4`，SHA256 `b75f0215620c193aa2e6416dfc72c427fa55a0eb3c8c8d8ee7941fadd4040232`。
