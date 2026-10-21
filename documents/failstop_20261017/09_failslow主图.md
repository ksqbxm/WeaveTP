# fail-slow 主图：实验入口与分析器（`--scenario failslow`）

状态（2026-10-21）：**已实现，CPU 已验证，服务器 GPU 未运行。** 分支 `codex/failslow`，已 rebase 到 `codex/failstop` 的清单 `568c319`（实现 `c4df30a`），`releases.json` 的 `step6` 与 `failslow` 指向同一实现（§9）。交付登记见 §8。本分支新功能只在 `--scenario failslow` 与 `--method restart` 下生效，weavetp/anchortp 的 fail-stop 行为与 `c4df30a` 相同。

图：横轴 W/M/S，纵轴为窗口内端到端输出吞吐（token/s）除以同参数 nofault 的均值；方法 ignore、restart、anchortp、weavetp。慢速时长 D 默认 600 s（FALCON/GREYHOUND 中计算型 fail-slow 平均约 10 分钟）。

## 1. 两点现有能力（读代码结论）

1. **缩容时不能边服务边缩容。** shrink 一开始 EDP0 就停服：在途请求带已提交 token 回到队首、EDP0 标记 down（`e2e/failstop_adapter.FailstopHooks.before`），TP2 就绪后重新 prefill。shrink 的权重波次期间 EDP1 也不解码（`serving_worker.py` 中 `sender_transfer('shrink')` 不带 decode_fn）。按要求不改 shrink 路径，weavetp/anchortp 的缩容都是停服缩容。扩容有 overlap：WeaveTP 在 base 波后跑 `growth['expand']=2` 个前台步，EDP1 在等计划键时继续服务；AnchorTP 两次迁移都是 `overlap_steps=0`。
2. **原驱动不支持按墙钟截止**，只在全部请求完成时下发 `done`。新增 `ControlService.close()`：截止后，对不在 hold、也没有待切换的域直接下发 `done`，不再接纳。worker 的 loop 退出路径不变，在途请求保留为未完成。

emulate 模式已具备 fail-slow 需要的其余条件：进程不 kill；TP2 子组在启动时从现有 world 预建；被撤 rank 在 `emulate_release` 时释放权重和 KV（`StandbyWeights` resize 到 0）；慢卡释放后不发布任何分片，既不是 shrink 的源，也不是 expand 的源。因此 fail-slow 一律用 emulate 机制（强制 `--mode emulate`），weavetp/anchortp 没有新增切换代码。

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
| weavetp | 撤下故障集合，EDP0 在 4 个健康 rank 上缩到 TP2：混合选源迁移权重，在途请求重新 prefill，停服缩容（§1） | 扩回 TP4：现有扩容路径（KV 前缀 + delta，2 个前台步） | 现有 `ServingSwitcher` |
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
  - 事件顺序或 TP4 hash 检查失败的运行记入 `rejected.json`。
- 结果文件名为 `failslow_result.json`（照旧）。`run_probe.run_case` 自己的 launch 结果写 `launch_result.json`，不再覆盖网关的文件（`c4df30a`）。

## 6. 预算与超时（failslow 自动计算）

- 每次 launch 预算 = warmup + D + tail + 600 s（启动与初始 TP4 加载）+ 2 × 300 s（restart 两次磁盘加载）+ 300 s 余量。常量在 `serving_config.FAILSLOW_*`。正式参数为 120+600+120+600+600+300 = 2340 s。
- `--budget-seconds` 缺省为 launches × 每次预算 + 60；显式给出但不够时拒绝。
- `stage_timeout_s` ≥ D + tail + 两次加载 + 余量。emulate 下被撤 rank 等 `spawn` 约 D 秒，`serve_until` 和 Gloo/NCCL 组也用这个值。`launch_timeout_s` 取剩余预算。
- 单测断言 D=600 时各级超时都够。quick_check 的总预算按同一公式算 3 次 fail-slow launch，再加 fail-stop restart launch 的 600 + 2 × 300 + 300 s，这是上限，实际约 15 分钟。
- fail-stop 下的 restart：`stage_timeout_s` 至少 300 + 300 s（同伴要等一次磁盘加载和 rejoin 延迟）。

## 7. 正确性与验证

- **唯一的正确性硬检查：TP4 自比 hash。** correctness 模式下，EDP0 每个 rank 在初始加载后（注入前）记录逐参数 sha256。weavetp/anchortp 扩回 TP4 后、restart 重新加载 TP4 后再算一次，必须逐位一致，否则 `check.tp4_hash_passed=false`。不依赖服务器上的参考 hash；failslow 下不要求 `--reference-root`。TP2 没有注入前基线，不检查。
- **不做 token 逐位比对**，worker 也不记录 token id。
- quick_check `--step 9`（共享 GPU、correctness、W 级、warmup 20 s、D 30 s、tail 20 s、detect 10 s、toy 请求池 2000 条）：依次跑 weavetp、anchortp、restart 三次 fail-slow launch，再跑一次 emulate fail-stop 的 restart（64 条 toy 请求，输出 24–72 token，全部完成）。每次检查：launch passed；本节点没有残留 `failslow_hog.py`；SL3060 上还检查：fail-slow 的事件顺序、截止时确有未完成请求、4 个 hog 都 ready 后才开始注入、8 个 EDP0 rank 的 TP4 hash 全部一致；fail-stop restart 的 `check.json` 通过、shrink/expand 两次加载都有计时、迁移字节为 0。TP2/TP4 加载对参考 hash 的逐参数检查在 worker 里做，不一致即 launch 失败，所以 Step 9 需要 `--reference-root`。toy 输入写在 `<run>/inputs_failslow`、`<run>/inputs_failstop`，两节点路径相同（§9）。
- 扩容时加入的 rank 记 decode token 时间的 KeyError 已由 fail-stop 的 `c4df30a` 修复（同一行 setdefault），rebase 后本分支不再单独改这一行。
- CPU 回归（`code/current`）：

```
python -B -X utf8 -m pytest -q -o addopts='' --noconftest -p no:cacheprovider tests/unit_tests/e2e tools/resharding/failstop tools/resharding/correctness/test_router_determinism.py tests/unit_tests/resharding/test_live_storage_lifecycle.py tests/unit_tests/resharding/test_live_defaults.py
```

结果见 §8。新增测试：
- `test_failslow.py`：时间线顺序与变体、detect 越界、组合与故障集合校验、预算与 D=600 超时、入口缺省预算与拒绝、restart 只在 emulate fail-stop 放行且有加载超时余量、`--shared-gpu-ok` 的清单标记、Step 9 两节点各自生成的配置被网关接受（输入放在节点目录下时被拒）、`close()` 不再接纳且 hold/待切换域继续、hog 按名加 tag 匹配（伪 /proc）、按名回收（仅 Linux）。
- `test_serving_protocol.py`：emulate fail-stop 下 restart 的 16 进程模拟（TP2/TP4 加载对参考 hash、无迁移、重放、两次加载计时、分析器的停机拆分），TP2 加载不一致时失败。
- `test_quick.py`：Step 9 的预算与 launch 名取自检出后的实现。
- `test_failslow_protocol.py`：16 进程模拟 weavetp/anchortp/restart/ignore/nofault、timing 无 hash、重新加载不一致时检查失败。
- `tests/unit_tests/e2e/test_failslow_metrics.py`：窗口吞吐、TTFT/TPOT、slow_rate、restart 停机、fig1 归一化与分组（含共享 GPU 自成一组）、分析器 CLI。

## 8. 交付登记

- 原实现 `be8396746d16a373ada21a7a710545b29123c7cc`（自 `2ff5962`）及其清单 `3d62941`、命令 `f112ef2` 作废。
- rebase 后的 fail-slow 实现：`16e777fd84c9b638dc272c2d0c4f6a52f20b102d`（`be83967` 在 `568c319` 上重放，冲突以 `568c319` 为准）。实现（A）：`1ccac7804105af6b9d70ee93df78f066a8b32e37`（§9 的五项）。
- 清单（B）：本文件所在提交，`releases.json` 的 `step6` 与 `failslow` 都指向 A。服务器命令在 B 的子提交里，随 bundle 交付：[Step9_failslow_服务器命令.md](Step9_failslow_服务器命令.md)。
- CPU 回归：295 项 = 294 通过 + 1 跳过（hog 按名回收测试需要 Linux /proc），61 个 subtest 通过，0 失败；rebase 后 288 项（fail-stop 258 + fail-slow 30），本次新增 7 项。日志 [failslow_cpu_evidence/regression_pytest.log](failslow_cpu_evidence/regression_pytest.log)。
- 服务器 GPU 未运行。

## 9. 2026-10-21：rebase 到 568c319、restart 的 fail-stop、`--shared-gpu-ok`、Step 9

| 项 | 内容 | 位置 |
|---|---|---|
| 1 Step 9 输入路径 | toy 输入原写在 `<run>/<node>/inputs`，两节点 `config['serving']` 的路径不同，网关逐键比较时必然拒绝（`node config mismatch: serving`）。改为 `<run>/inputs_failslow` 与 `<run>/inputs_failstop`，两节点同一路径；网关的比较抽成 `serving_probe.require_same_config`，测试用两节点各自生成的配置调用它 | `run_failslow_quick.py`、`serving_probe.py` |
| 2 rebase | `be83967` 重放到 `568c319`；冲突（分析器文档、quick_check、quick_driver、run_failstop_e2e、serving_config）以 `568c319` 为准，fail-slow 的增量保留；KeyError 那一行两边相同；`run_case` 写 `launch_result.json` 的修复保留，`failslow_result.json` 命名照旧 | rebase 后的实现提交 |
| 3 restart 用于 fail-stop | `--method restart --scenario failstop --mode emulate`（`serving_config.require_method` 是方法/场景的唯一规则，real 与 nofault 场景拒绝）：故障后 4 个存活 rank 在同一批进程内用 `MegatronRuntime.load_local` 从磁盘加载 TP2，rejoin 后 8 个 rank 加载 TP4，复用 `RestartSwitcher`；correctness 下两次加载都对参考 hash 逐参数比较。网关记 `failstop.restart_loaded_s`（shrink/expand 加载完成，`serving_probe.restart_loaded`，fail-slow 共用），不搬 KV 所以 `expand_kv_check_passed=null`；分析器用 fail-slow 同一个 `metrics.restart_downtime` 拆分每次切换的停机→加载完成→恢复，`summary.csv` 加 `restart_downtime_s` | `serving_config.py`、`run_failstop_e2e.py`、`serving_probe.py`、`e2e/metrics.py`、`e2e/failslow_metrics.py` |
| 4 `--shared-gpu-ok` | `run_failstop_e2e --scenario failslow --run-mode timing --shared-gpu-ok`：沿用 `568c319` 的显存检查、前后外来进程快照；`failslow_result.json` 带 `gpu_sharing`，`fig1.csv` 按 `gpu_sharing` 分组且只用同一 `gpu_sharing` 的 nofault 做归一化 | `failslow_probe.py`、`e2e/failslow_metrics.py` |
| 5 Step 9 | 第四次 launch：emulate fail-stop 的 restart（见 §7）；`quick_check.sh` 改为 `--manifest <清单>`：`releases.json` 用 `git show` 从清单读，`quick_driver.py` 从实现提交取出放到 kickoff 目录运行，只有 `quick_driver` 在 quick 锁内检出实现并在结束后切回 `f7d5559`，导入 bundle 时不动主仓库；Step 9 需要 `--reference-root` | `run_failslow_quick.py`、`quick_driver.py`、`quick_check.sh` |

新增实现代码 125 行（不计空行、注释、docstring；物理行 +179/−92），上限 250。
