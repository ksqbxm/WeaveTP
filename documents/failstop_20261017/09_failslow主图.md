# fail-slow 主图：实验入口与分析器（`--scenario failslow`）

状态（2026-10-21）：**已实现，CPU 已验证，服务器 GPU 未运行。** 分支 `codex/failslow`，已 rebase 到 `codex/failstop` 的清单 `568c319`（实现 `c4df30a`），`releases.json` 的 `step6` 与 `failslow` 指向同一实现（§9）。交付登记见 §8。本分支新功能只在 `--scenario failslow` 与 `--method restart` 下生效，weavetp/anchortp 的 fail-stop 行为与 `c4df30a` 相同。fail-slow 的 weavetp 在 `codex/failslow-online` 上改为在线切换（组、模型、权重计划都在开始服务前备好，缩容不重放、KV 一起迁移），见 [10_failslow在线化依据.md](10_failslow在线化依据.md)；下文 §1、§4、§7 已按在线切换更新。

图：横轴 W/M/S，纵轴为窗口内端到端输出吞吐（token/s）除以同参数 nofault 的均值；方法 ignore、restart、anchortp、weavetp。慢速时长 D 默认 600 s（FALCON/GREYHOUND 中计算型 fail-slow 平均约 10 分钟）。

## 1. 两点现有能力（读代码结论）

1. **缩容时不能边服务边缩容。** shrink 一开始 EDP0 就停服：在途请求带已提交 token 回到队首、EDP0 标记 down（`e2e/failstop_adapter.FailstopHooks.before`），TP2 就绪后重新 prefill。shrink 的权重波次期间 EDP1 也不解码（`serving_worker.py` 中 `sender_transfer('shrink')` 不带 decode_fn）。fail-stop 全部方法与 fail-slow 的 anchortp/restart 仍是停服缩容；fail-slow 的 weavetp 改为在线缩容（10 §2）。扩容有 overlap：WeaveTP 在 base 波后跑 `growth['expand']=2` 个前台步，EDP1 在等计划键时继续服务；AnchorTP 两次迁移都是 `overlap_steps=0`。
2. **原驱动不支持按墙钟截止**，只在全部请求完成时下发 `done`。新增 `ControlService.close()`：截止后，对不在 hold、也没有待切换的域直接下发 `done`，不再接纳。worker 的 loop 退出路径不变，在途请求保留为未完成。

emulate 模式已具备 fail-slow 需要的其余条件：进程不 kill；TP2 子组在启动时从现有 world 预建；被撤 rank 在 `emulate_release` 时释放权重和 KV（`StandbyWeights` resize 到 0）；慢卡释放后不发布任何分片，不再是任何迁移的源（在线 weavetp 的慢卡在释放之前是缩容的源）。因此 fail-slow 一律用 emulate 机制（强制 `--mode emulate`）；anchortp 没有新增切换代码，weavetp 的在线切换见 10。

## 2. 时间线（网关时钟，SL3060）

```
serving_start ─warmup─> slow_start → hog 全部 ready = inject_start ─detect─> detect ─(切换类) shrink … shrink_done
                                   └─ inject_start + D：slow_stop → slow_stopped = inject_end → recovered ─(切换类) spawn … expand_done
deadline = inject_start + D + tail → close() → serving_done
```

- `failslow_probe.Timeline` 按前置关系检查事件，乱序或重复都拒绝。shrink_done 只要求在 detect 之后，可以晚于注入结束（restart 加载慢时会出现）。expand 等 shrink_done 和 recovered 都到了才开始。
- 注入结束由网关的独立线程在 inject_start + D 准时发出，不受 shrink 进度影响。
- nofault 不注入：inject_start/detect/inject_end 取名义时刻，没有 slow_start/slow_stop。
- 扩容拖过截止时，等它完成后再 close，超出部分记 `deadline_overrun_s`。
- 所有事件、切换里程碑（freeze…cutover…regroup）、restart 的加载完成时刻都写进 `failslow_result.json`，统一为 serving_start 起算的秒数。

## 3. 注入（无 sudo，参考 Malleus）

- `failslow_hog.py`（仓库内，`code/current/tools/resharding/failstop/`）：通过 `CUDA_VISIBLE_DEVICES` 绑定 GPU，跑 4096² bf16 matmul 死循环。reserved 显存超过 1 GiB 时自行退出；第一个 matmul 完成后写 ready 文件；设 PR_SET_PDEATHSIG，SIGTERM 退出。
- `--slow-level W/M/S` 对应每张故障 GPU 上 1/2/3 个陪跑进程（Malleus level-1/2/3），不做标定。
- `failslow_inject.Injector` 由 SL3061 监管器持有：收到 `slow_start` 时启动，全部 ready 后回复 `slow_started`；收到 `slow_stop` 时 SIGTERM，5 s 后 SIGKILL，再按名清扫，然后回复 `slow_stopped`（含 `leftover`）。清理路径包括 stop()、监管器 finally、SIGTERM/SIGINT/SIGHUP 和 atexit。只按 `/proc/<pid>/cmdline` 里的脚本名 `failslow_hog.py` 加 `--tag <launch>` 匹配，不用 pkill -u，也不用 killall python。
- hog 使用与 worker 相同的不存在 MPS 管道路径，不会连上他人的 MPS。
- `--fault-gpus`（默认 `0,1,4,5`，SL3061 的 nvidia 序号）。三种切换类方法会撤下 fail-stop 拓扑固定的故障坐标 (ep,etp)∈{0,1}²，即 GPU 4·ep+etp = {0,1,4,5}，所以这三种方法下集合必须正好是它，否则在参数校验和监管器里都报错。ignore/nofault 接受任意集合。放开这一点要改 `megatron/core/failstop.py` 的拓扑，不在本次范围。
- `slow_rate_measured` = EDP0 decode 前向时间（timeline 的 `t_decode`）在 [inject_start, detect) 的中位数，除以注入前 30 s 的中位数。所有方法用同一窗口；nofault 约为 1。

## 4. 方法

| 方法 | 检测后 | 注入结束后 | 实现位置 |
|---|---|---|---|
| ignore | 不动 | 不动 | `SWITCHERS['ignore']`（从不触发） |
| weavetp | 在线缩到 TP2（4 个健康 rank）：权重按开始服务前算好的计划迁移（故障卡 p2p 降权 1/(1+k)），KV 一起迁移、不重放，迁移期间两个域都在波内解码；迁移完成后才撤下故障集合（10 §2） | 扩回 TP4：现有扩容路径（KV 前缀 + delta，2 个前台步），目标是再分配的初始 TP4 模型，计划开始服务前算好 | `ServingSwitcher`（`common.online_switch`） |
| anchortp | 同时间点，固定选源、stop-the-world | 同时间点，`overlap_steps=0`、无 delta | 现有 `AnchorTPSwitcher` |
| restart | EDP0 停服，在途请求回队；4 个健康 rank 在同一批进程内从磁盘加载 TP2 后重新 prefill | 再次停服回队，8 个 rank 从磁盘加载 TP4 | 新增 `RestartSwitcher`；也用于 emulate fail-stop（§9） |
| nofault | 不注入、不切换（worker 内部方法为 ignore） | — | `--method nofault` |

restart 的说明：
- 磁盘加载用 `MegatronRuntime.load_local`：直接调 `dist_checkpointing.load(validate_access_integrity=False)`，torch_dist 策略本身就是 no_dist。原因是 `load_checkpoint` 会对整个 world 做 tracker all_reduce 和 barrier，只有 EDP0 加载时会挂住。分片偏移取自模型自己的进程组（MoE 专家用 `pg_collection.ep/expt_tp`）。
- 迁移计划为空，EDP1 只跟随 world transition，继续服务。
- 记录三个时间点：停机开始（worker `fail`/`rejoin_start`）、加载完成（`restart_loaded_s`，取全部加载 rank 的最大值）、恢复服务（EDP0 首个 token）。
- **没有计入进程启动和 NCCL 初始化开销，这对 restart 是偏乐观的。**

## 5. 负载与指标

- ShareGPT，设置同 fail-stop：输出 ≤256、固定输出长度、忽略 EOS、greedy、native 路由。`make_failstop_requests --count 20000` 生成固定请求池，配合 `plan_nofault.json`，时间线全部走命令行，未改生成器。
- 全部请求 t=0 提交，按墙钟截止，不等全部完成。分发沿用全局 FCFS，EDP1 全程 TP4。
- 窗口 = [inject_start, inject_start + D + tail)，各方法等长。吞吐 = 窗口内已提交输出 token 数 / 窗口长度。
- TTFT 从首次接纳算（全部请求 t=0 提交，从到达算只反映排队位置；用户已确认）。TPOT = (末 token − 首 token)/(n−1)，统计 token 间隔与窗口相交的请求，口径同 fail-stop。
- 分析器 `analyze_failstop_e2e.py --out DIR <输出根或 launch 目录>`：发现 `failslow_result.json` 时走 fail-slow 分支（`e2e/failslow_metrics.py`）。
  - 每次运行输出 `<launch>/timeline.csv`（全局/EDP0/EDP1/各副本每秒 token）和 `<launch>/metrics.json`（窗口吞吐、TTFT/TPOT 的 P50/P99、slow_rate、restart 三个时间点和停机时长）。
  - 汇总输出 `fig1.csv`，列为 severity, method, gpu_sharing, n, mean, std, p99_tpot, p99_ttft, slow_rate_measured_mean, restart_downtime_s。mean/std 是对同时间线、同 `gpu_sharing` 的 nofault 均值的归一化吞吐，std 取 ddof=1；p99 取各次运行 P99 的均值；restart_downtime_s 为两次停机之和。
  - 事件顺序检查失败的运行记入 `rejected.json`。
- 结果文件名为 `failslow_result.json`（照旧）。`run_probe.run_case` 自己的 launch 结果写 `launch_result.json`，不再覆盖网关的文件（`c4df30a`）。

## 6. 预算与超时（failslow 自动计算）

- 每次 launch 预算 = warmup + D + tail + 600 s（启动与初始 TP4 加载）+ 2 × 300 s（restart 两次磁盘加载）+ 300 s 余量。常量在 `serving_config.FAILSLOW_*`。正式参数为 120+600+120+600+600+300 = 2340 s。
- `--budget-seconds` 缺省为 launches × 每次预算 + 60；显式给出但不够时拒绝。
- `stage_timeout_s` ≥ D + tail + 两次加载 + 余量。emulate 下被撤 rank 等 `spawn` 约 D 秒，`serve_until` 和 Gloo/NCCL 组也用这个值。`launch_timeout_s` 取剩余预算。
- 单测断言 D=600 时各级超时都够。quick_check 的总预算按同一公式算 3 次 fail-slow launch，再加 fail-stop restart launch 的 600 + 2 × 300 + 300 s，这是上限，实际约 15 分钟。
- fail-stop 下的 restart：`stage_timeout_s` 至少 300 + 300 s（同伴要等一次磁盘加载和 rejoin 延迟）。

## 7. 正确性与验证

- **不做权重的逐参数检查（§10）。** fail-slow 的所有方法和 fail-stop 的 restart 既不对参考 hash 比对，也不做 TP4 自比；fail-stop 结果里标 `reference_audit="skipped"`。correctness 模式下 weavetp/anchortp 仍有扩容 KV 逐位检查和切换时的 logits 比对；fail-slow weavetp 的缩容也逐位检查迁移的 KV。
- **不做 token 逐位比对**，worker 也不记录 token id。
- quick_check `--step 9`（共享 GPU、correctness、W 级、warmup 20 s、D 30 s、tail 20 s、detect 10 s、toy 请求池 2000 条）：依次跑 weavetp、anchortp、restart 三次 fail-slow launch，再跑一次 emulate fail-stop 的 restart（64 条 toy 请求，输出 24–72 token，全部完成）。只检查：每次 launch 跑完且 passed；本节点没有残留 `failslow_hog.py`；SL3060 上 fail-slow 三次的事件顺序正确、截止时确有未完成请求。不需要 `--reference-root`。toy 输入写在 `<run>/inputs_failslow`、`<run>/inputs_failstop`，两节点路径相同（§9）。加 `--method weavetp|restart|ignore|anchortp` 时只跑该方法一次 fail-slow launch，SL3060 打印摘要行（[Step9_failslow_服务器命令.md](Step9_failslow_服务器命令.md) ②）。
- 扩容时加入的 rank 记 decode token 时间的 KeyError 已由 fail-stop 的 `c4df30a` 修复（同一行 setdefault），rebase 后本分支不再单独改这一行。
- CPU 回归（`code/current`）：

```
python -B -X utf8 -m pytest -q -o addopts='' --noconftest -p no:cacheprovider tests/unit_tests/e2e tools/resharding/failstop tools/resharding/correctness/test_router_determinism.py tests/unit_tests/resharding/test_live_storage_lifecycle.py tests/unit_tests/resharding/test_live_defaults.py
```

结果见 §8。新增测试：
- `test_failslow.py`：时间线顺序与变体、detect 越界、组合与故障集合校验、预算与 D=600 超时、入口缺省预算与拒绝、restart 只在 emulate fail-stop 放行且有加载超时余量、`--shared-gpu-ok` 的清单标记、Step 9 两节点各自生成的配置被网关接受（输入放在节点目录下时被拒）、`close()` 不再接纳且 hold/待切换域继续、hog 按名加 tag 匹配（伪 /proc）、按名回收（仅 Linux）。
- `test_serving_protocol.py`：emulate fail-stop 下 restart 的 16 进程模拟（`reference_audit="skipped"`、无迁移、重放、两次加载计时、分析器的停机拆分）；weavetp 的 correctness 结果标 `reference_audit="passed"`，timing 标 `skipped`。
- `test_quick.py`：Step 9 的预算与 launch 名取自检出后的实现。
- `test_failslow_protocol.py`：16 进程模拟 weavetp/anchortp/restart/ignore/nofault（权重检查全部 skipped）、timing 模式。
- `tests/unit_tests/e2e/test_failslow_metrics.py`：窗口吞吐、TTFT/TPOT、slow_rate、restart 停机、fig1 归一化与分组（含共享 GPU 自成一组）、分析器 CLI。

## 8. 交付登记

- 原实现 `be8396746d16a373ada21a7a710545b29123c7cc`（自 `2ff5962`）及其清单 `3d62941`、命令 `f112ef2` 作废。
- rebase 后的 fail-slow 实现：`16e777fd84c9b638dc272c2d0c4f6a52f20b102d`（`be83967` 在 `568c319` 上重放，冲突以 `568c319` 为准）。§9 的五项：`1ccac7804105af6b9d70ee93df78f066a8b32e37`（其清单 `141050f`、命令 `fdcad58`、bundle `failslow_141050f` 作废）。§10：`0fe0a3e3b182d8b200789c9d1a483ee7e66ba6a0`（其清单 `f884ff9`、命令 `3ab6d50`、bundle `failslow_f884ff9` 作废）。实现（A）：`073a50f41e70ee54551f3f2e2613c60203697ad1`（§9 + §10 + §11）。
- 清单（B）：本文件所在提交，`releases.json` 的 `step6` 与 `failslow` 都指向 A。服务器命令在 B 的子提交里，随 bundle 交付：[Step9_failslow_服务器命令.md](Step9_failslow_服务器命令.md)。
- CPU 回归：本机电池供电下全量 297 项测试：294 通过、1 跳过（仅 Linux）、2 失败；subtest 117 通过、2 失败。失败全部是 16 进程 CPU 模拟的 8 s 总线等待上限：两个 restart 模拟（各 1 项）单独重跑通过；`test_expand_with_requests_in_flight` 的 emulate/real 子测试在电池供电下失败（未改动的 `codex/failstop@42f19d3` 同样失败，交流供电时通过）。§11 新增 4 项（`test_server_launch.py`）。日志 [failslow_cpu_evidence/regression_pytest.log](failslow_cpu_evidence/regression_pytest.log)。
- 服务器 GPU 未运行。

## 9. 2026-10-21：rebase 到 568c319、restart 的 fail-stop、`--shared-gpu-ok`、Step 9

| 项 | 内容 | 位置 |
|---|---|---|
| 1 Step 9 输入路径 | toy 输入原写在 `<run>/<node>/inputs`，两节点 `config['serving']` 的路径不同，网关逐键比较时必然拒绝（`node config mismatch: serving`）。改为 `<run>/inputs_failslow` 与 `<run>/inputs_failstop`，两节点同一路径；网关的比较抽成 `serving_probe.require_same_config`，测试用两节点各自生成的配置调用它 | `run_failslow_quick.py`、`serving_probe.py` |
| 2 rebase | `be83967` 重放到 `568c319`；冲突（分析器文档、quick_check、quick_driver、run_failstop_e2e、serving_config）以 `568c319` 为准，fail-slow 的增量保留；KeyError 那一行两边相同；`run_case` 写 `launch_result.json` 的修复保留，`failslow_result.json` 命名照旧 | rebase 后的实现提交 |
| 3 restart 用于 fail-stop | `--method restart --scenario failstop --mode emulate`（`serving_config.require_method` 是方法/场景的唯一规则，real 与 nofault 场景拒绝）：故障后 4 个存活 rank 在同一批进程内用 `MegatronRuntime.load_local` 从磁盘加载 TP2，rejoin 后 8 个 rank 加载 TP4，复用 `RestartSwitcher`；不做参考 hash 比对（`reference_audit="skipped"`，§10）。网关记 `failstop.restart_loaded_s`（shrink/expand 加载完成，`serving_probe.restart_loaded`，fail-slow 共用），不搬 KV 所以 `expand_kv_check_passed=null`；分析器用 fail-slow 同一个 `metrics.restart_downtime` 拆分每次切换的停机→加载完成→恢复，`summary.csv` 加 `restart_downtime_s` | `serving_config.py`、`run_failstop_e2e.py`、`serving_probe.py`、`e2e/metrics.py`、`e2e/failslow_metrics.py` |
| 4 `--shared-gpu-ok` | `run_failstop_e2e --scenario failslow --run-mode timing --shared-gpu-ok`：沿用 `568c319` 的显存检查、前后外来进程快照；`failslow_result.json` 带 `gpu_sharing`，`fig1.csv` 按 `gpu_sharing` 分组且只用同一 `gpu_sharing` 的 nofault 做归一化 | `failslow_probe.py`、`e2e/failslow_metrics.py` |
| 5 Step 9 | 第四次 launch：emulate fail-stop 的 restart（见 §7）；`quick_check.sh` 改为 `--manifest <清单>`：`releases.json` 用 `git show` 从清单读，`quick_driver.py` 从实现提交取出放到 kickoff 目录运行，只有 `quick_driver` 在 quick 锁内检出实现并在结束后切回 `f7d5559`，导入 bundle 时不动主仓库 | `run_failslow_quick.py`、`quick_driver.py`、`quick_check.sh` |

新增实现代码 125 行（不计空行、注释、docstring；物理行 +179/−92），上限 250。

## 10. 2026-10-21（二）：去掉参考 hash 比对与 TP4 自比

- 用户决定：Step 9 不要求 `--reference-root`，fail-stop 的 restart launch 不做参考比对（TP2、TP4 都不比），结果标 `reference_audit="skipped"`。
- TP4 自比一并去掉：它只在 correctness 模式下运行，实际只出现在 Step 9，而 Step 9 不再检查它。删除 worker 的注入前基线与 `tp4_hash/<slot>` 消息、网关的 `tp4_hash` 与 `check.tp4_hash_passed`；`check.passed` 只看事件顺序，分析器也只按事件顺序拒收。
- 规则只有一处：`common.audits_reference(scenario, method)`，只有 fail-stop 的 weavetp/anchortp 迁移目标对参考 hash 比对；SL3061 的 correctness 运行只在这时要求参考 hash（`serving_config`），worker 的 `validate_weights` 其余情况返回 skipped，fail-stop 结果的 `failstop.reference_audit` 为 `passed`（correctness 且有比对）或 `skipped`。
- Step 9 只检查：每次 launch 跑完且 passed、本节点没有残留 `failslow_hog.py`；SL3060 上 fail-slow 三次的事件顺序正确、截止时确有未完成请求（`run_failslow_quick.gate_checks`）。fail-stop restart launch 不再单独查网关文件。

## 11. 2026-10-21（三）：上机暴露的两个缺陷

1. **worker 参数解析。** 自 Step 5 接入（`241fb75`）起，每个真实模型 worker（`model_worker.py`、`serving_worker.py`，任何角色、任何 run_mode）都在 `MegatronRuntime.from_config` 用 `model_reference.add_args` 解析 `config['model_argv']`。那组参数是 model_reference 单独运行时用的，`--mr-output`、`--mr-role` 必填，而没有任何启动器传它们，worker 也不读任何 `mr_*`，所以每个 worker 启动即退出：`model_worker.py: error: the following arguments are required: --mr-output, --mr-role`（SL3060 先报的是 worker 3）。修正：只用 worker 需要的 `benchmark_live_moe_tp.add_live_args`。同一处的下一个问题也一并修正：Megatron 的 `validate_args` 按 `WORLD_SIZE`（缺省 1）检查 TP4 × EP2 × DP2，而 worker 解析时 world 还没建，所以解析期间临时设为拓扑的槽数（16），解析后还原；world 由 worker 自己建立，`initialize_megatron` 沿用它。
2. **控制门导入。** `failslow_probe.py` 在第 17 行 `from tools.resharding.e2e import requests_io`，但服务器上控制门进程的 `sys.path` 只有 failstop 目录，而 `serving_probe` 在导入 `failslow_probe` 之后才插入代码根目录：`ModuleNotFoundError: No module named 'tools'`。修正：每个在模块顶层导入 `tools`/`megatron` 的 failstop 模块都自己插入代码根目录（同 `serving_config`、`model_memory`、`run_failslow_quick` 的写法），不再依赖导入顺序：`failslow_probe`，以及同样只靠别的模块先插入路径的 `anchortp_plan`、`async_fixture`、`async_topology`、`source_profile`、`source_runtime`、`topology_fixture`、`model_runtime`。

新增 `test_server_launch.py`（像服务器一样）：(a) 用监管器真实的 `Supervisor.start_workers` 生成 28 个组合（failstop 10、nofault 8、failslow 10）× 两节点的全部 worker 命令行（含 real 模式的替换进程）和 Step 5 的 model worker 命令行，每个 epoch 内命令行只差 `--slot`，并以启动器设置的环境运行 worker 脚本真实的 `main()`：它自己的 argparse 和 `from_config` 里真实的 Megatron 解析（不建模型；只把 GPU 属性查询替换为 RTX 5090 的值）；(b) 45 个 failstop 模块各在一个新的 `python -I` 进程里导入，`sys.path` 里只有 failstop 目录、不设 PYTHONPATH。
