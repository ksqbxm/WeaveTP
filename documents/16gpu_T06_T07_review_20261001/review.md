# T06–T07 正确性、完备性与简洁性审阅

日期：2026-10-01。审阅提交：T06 `41da0b6ae781a0516b73136d57378a0bb7e9ac93`、T07 `754feb187d8426cedb978bf073f6ed76de89fc1e`；基准 `ab5042d`。本次重新完整阅读 CODEX.md、总计划、源码 AGENTS.md 和分步 T06/T07，沿 benchmark、真实组构建器、planner、KV 区间处理、观测输出、结果检查、节点进程与清理的调用链审阅。没有修改生产代码、安装、提交、推送、SSH 或使用 GPU。

结论：T06 的核心统计和计时隔离设计基本正确；T07 调整清理顺序解决了过早杀死监督进程的问题，但失败生命周期的证据与验收仍未完全对齐。发现以下三个 P2 问题，其中前两个已用 CPU 反例复现，第三个是脚本路径绑定缺失。当前用户提供的 env.sh 路径是正确的，第三项不是声称服务器已经使用错误路径。

## R1 / P2：清理最终成功后，旧退出回执仍可能导致失败场景验收误报

涉及 [stop_case](../../code/current/tools/resharding/compare_weavetp_16gpu.py)、同文件 `node_action` 和 [test_controller](../../code/current/tests/server_tests/weavetp_16gpu/test_controller.py)。

T07 现在先执行 cleanup，再等待 run RPC。cleanup 会先后终止 wrapper 与其后代；wrapper 先退出时，仍存活的节点监督进程会从 `process.wait()` 返回，并立即查询 worker/显存，随后独占写入 `exit.nodeN.json`。若 GPU 子进程仍在退出，回执中的 `quiescent=false` 是那个时刻的快照。稍后 cleanup 完成并返回 `ok=true`，现有代码并不会让这个旧回执变成清理终态。

问题在于 `test_controller.py:160` 随后把旧 `exit.quiescent` 当作清理后的最终结论，要求所有已有回执为 true。因此即便本任务进程已经消失、显存已恢复、两端 cleanup 都确认成功，失败/中断场景仍可能报 `a host did not return to its baseline`。不是说程序会把失败 launch 标成成功，也不是说本次发现了清理后的真实泄漏。

复现使用真实生产 `node_action` 与 `stop_case`，对 GPU 查询、Popen 和 cleanup 时间安排做 CPU mock：wrapper 退出 → runner 写回执 → 后代退出 → cleanup 确认。实际输出为 `cleanup_confirmed=true, workers_remaining=[], persisted_exit_quiescent=false`，断言失败。该反例没有冒充 Linux/SSH/GPU 实测。

根本修复应明确“进程退出快照”和“清理终态”的关系。最小方案是让失败验收使用已完成 cleanup 的最终进程/显存证据，并保留原退出快照；若设计上要求 exit 本身就是终态，则需要清理完成的明确同步后再写回执。单纯把 15 秒调大无效，因为回执已经写出且 runner 已退出。T07 新增测试只验证替身能写一个 receipt 文件，没有调用真实 node_action 的这段状态采集，因此未覆盖该竞态。

## R2 / P2：采用候选时，T06 检查器放行缺失的默认计划

位置：[test_observations.py](../../code/current/tests/server_tests/weavetp_16gpu/test_observations.py) 52–55 行及 67–69 行。

循环对 default/candidate/adopted 一律允许 None；后续只要求当前采用的计划存在。若扩容实际采用 candidate，将 `default` 改为 None 后，检查器仍通过，缺少默认计划的双向跨机字节、权重/KV 和收发峰值不会被发现。总计划明确要求两个方向都保留默认计划，以便与候选/实际流量解释对照。

CPU 反例先生成检查通过的候选采用结果，再仅将两次扩容的 default 设为 None；再次 verify 仍通过。当前生产 observer 正常路径会生成 default，这项是验收完备性漏洞，不代表已确认正常生产数据漏算流量。

最小修复：先强制 default、adopted 为包含完整字段的对象；只有 candidate 可以按候选状态为 None。保留现有 candidate gate 与采用计划一致性检查，并补候选采用、baseline 回退两条路径的缺失字段反例。不需要改流量算法。

## R3 / P2：T07 检查的仓库与实际执行测试的 CODE_DIR 没有绑定

位置：[run_t07.sh](../16gpu_T07_20261001/run_t07.sh) 8–10、45–66、82–89 行。

脚本从自身路径取得 T07_REPO，检查其 HEAD、Git clean 和源码 manifest；运行 CPU 测试与 dry-run 时却使用独立环境变量 CODE_DIR。没有断言 CODE_DIR 的实际路径等于该仓库的 code/current，也没有在 CODE_DIR 上核验相同 Git 身份。

如果 env.sh 的 CODE_DIR 漂移到另一份也能通过测试的工作副本，脚本可能校验 A 的 commit/hash，执行 B 的测试，最后仍输出 A 的 `CODE_OK` 与 `T07_STATIC_CPU_OK`。NODE1_CODE_DIR 也只被写入 dry-run，不能据此认证对端来源。当前已提供的 env.sh 没有这个漂移；发现来自静态数据流审阅，未在服务器伪造环境来执行。

最小修复：把本机测试入口从已核验 REPO 唯一派生，或以 canonical path 显式断言 CODE_DIR 等于 REPO/code/current。对端实际 Git 身份仍由双机部署/启动预检核对；本机 dry-run 不应被当作远端身份验证。此改动还能去掉 T07_CODE 的独立来源，减少重复状态。

## 未发现问题的部分

- 实际组读取使用两个模型真实 pg_collection，通过 get_world_size/get_process_group_ranks 采集；检查 DP/EDP、成员互相一致、rank/主机/CUDA device/UUID 与机内 TP/ETP/EP，未用公式冒充测量。
- 接收侧计数不扫描 send 镜像；同卡复制与远程流量分开；按目标切片元素数和 element_size 计算字节，KV 前缀/增量使用实际区间而非 1024 容量；双向跨机字节与每卡收发/并列峰值的逻辑一致。
- global_gate_accepted=false 的不同原因分开；全局拒绝后的原候选不可取得时明确 None；cached candidate 与 adaptive 后采用计划分开，未把回退默认计划当成被拒绝候选。
- 新统计扫描/hash/汇总 collective 在全部切换结束后；计时后只保留已有对象引用；未改变迁移算法、原计时起止、阈值或正式配置。
- T07 保留有界 RPC、任务标签、清理未确认和禁止自动重试；清理顺序调整有实际作用，不应全部回退。缺口在终态证据及联合验收。

## 简洁性判断

T06 的分函数结构足够直接，使用标准库即可完成，不需要引入通用观测框架或更多抽象。部分字段在原 record 与 plan_observation 中重复，但承担原始记录与观测一致性核验，不值得仅为少几行大幅重构。

最应简化的是控制进程、节点 runner、cleanup、测试检查器对失败终态的重复判断。应确立一份清理终态证据并复用，而不是继续叠加 sleep、重试或延长等待。T07 脚本的仓库/代码路径也应只有一个已验证来源。

## 验证与证据边界

- 本轮重新执行既有六组测试：28/26/18/11/4/25，共 112 项，全部退出 0，原始命令和输出见 [existing_regressions.json](./existing_regressions.json)。
- 新增两个审阅反例全部失败（预期行为断言失败），命令退出 1，见 [reproduce_findings.py](./reproduce_findings.py) 和 [findings.log](./findings.log)。它们是当前缺陷的证据，未使用 skip/expectedFailure，也未修复生产实现。
- `git diff --exit-code -- code/current` 退出 0；生产源码保持被审阅版本不变。只新增本目录审阅材料，未重写原 T06/T07 历史验收记录。
- 没有运行完整 Linux T07 脚本或真实双机 SSH/信号/GPU。实际网络、设备、显存恢复、16 卡迁移正确性仍需服务器证据。

复现命令（本地仓库根目录，既有 Python）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONUTF8='1'
python -B -X utf8 documents/16gpu_T06_T07_review_20261001/reproduce_findings.py
```

审阅时输出为 `Ran 2 tests`、`FAILED (failures=2)`、退出码 1，原始失败日志保留。当前修复版预期 `Ran 3 tests`、`OK`、退出码 0。

## 按用户指定方案修复后的验收

本次重新完整阅读 CODEX.md、总计划、T06/T07 分步计划和源码规则，仅修复上述三项。清理验收删除旧 exit.quiescent 判定和单机重复进程查询，以两端 cleanup 的 ok 与 remaining_pids 为唯一清理终态；退出快照继续保存，退出码仍用于确认故障注入确实发生。未改变正常成功 launch 的验收，未新增等待、重试或兜底分支。

检查器要求 default/adopted 为完整计划对象，仅 candidate 按既有 gate 规则允许 None。T07_CODE 直接从已核验 REPO/code/current 派生，CODE_DIR 不一致时立即报错；不保留独立代码目录执行路径。

两个原反例已改为修复后通过：第一个验证真实 node_action 保存 quiescent=false 的较早快照，同时最终 cleanup 成功可通过；任一端清理未确认、有剩余 PID 或缺少一端结果都拒绝。第二个覆盖 baseline/candidate 采用路径、四次切换及 default/adopted 的 null、缺键和空对象。原 T06 测试继续覆盖候选被拒绝/收缩不适用时为空。新增第三个测试实际调用 T07 shell，验证路径匹配可到达主机检查，不匹配在主机检查前停止；该测试用 hostname 探针阻止后续服务器流程。

实际执行六组原测试 28/26/18/11/4/25 和三个审阅回归，共 **115/115**；23 个 Python 内存编译、5 份 bash -n、3 段嵌入 Python 编译及 git diff --check 均通过，原始命令/输出见 [fixed_validation.json](./fixed_validation.json)。T07 入口加入这三个回归，source_manifest 更新为 29 个文件；code/current 的 imports 未改，无 isort 触发项，未安装工具。

本地验证使用 Windows 上既有 Python 与 Git Bash，没有执行完整 Linux T07、真实双机 SSH/清理或 GPU 工作。发布后的固定 SHA 需在两机分别检出，并将各自 WORK/env.sh 的 WEAVETP_COMMIT 原赋值更新为该 SHA，再执行 T07；服务器结果仍待用户回传。
