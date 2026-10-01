# T07：集成检查与最终 diff 审查

日期：2026-10-01。**本地静态/CPU 集成及三项审阅问题修复完成，115/115 通过。新版本双机复核尚未执行。** 本阶段未 SSH、安装、复制模型或启动 GPU。初次 112 项集成记录保留如下，后续修复与验证见 [T06–T07 审阅及修复报告](../16gpu_T06_T07_review_20261001/review.md)。

## 本次阅读与范围

已重新完整阅读 CODEX.md、总计划、分步任务 T07、code/current/AGENTS.md 及贡献说明；核对 T02/T03/T04/T05/T06 记录，阅读统计入口、compare、wrapper/launcher、画像校验/测量、T06 观测、benchmark 接入及相关测试。修复失败生命周期前再次完整阅读强制文件。

范围为 T02–T06 的静态/CPU 集成和最终 diff，只修复本任务链引入的问题。遵守当前总计划：launch 60 分钟上限、切换人工监控、画像来源逐日志/UUID 配对暂缓、两机同一 Git commit 且工作树干净。不采用旧记录中的十分钟逐切换硬限制或六文件身份门禁。

## 接收的 T06 服务器进度

来源为本轮用户回传，未由助手远程复查；完整 hash 未提供的部分保留缩写。

| 项目 | 用户回传 |
|---|---|
| 代码 | 两机 `41da0b6ae781a0516b73136d57378a0bb7e9ac93`，`GIT_CLEAN=1` |
| 环境 | 两机 ENV_OK；Python 3.12.13、torch 2.11.0+cu128、CUDA 12.8、NCCL 2.28.9 一致 |
| Python | SL3060 `/home/ubuntu/miniconda3/envs/megatron/bin/python`；SL3061 `/data/ubuntu/lxh/weavetp/envs/megatron/bin/python` |
| CPU | 两机六组 28/26/18/11/3/25，退出 0；共 111 项 |
| checkpoint | 两机 `/data/models/DeepSeek-V2-Lite-megatron-v2`；清单 hash `6fb17c76…de665e`、metadata hash `eebfb80f…bdb02a` 一致 |
| 磁盘 | SL3060 /data 余 55 GB，SL3061 余 120 GB；不是下一次 GPU 启动时的空闲证明 |
| 工作副本 | 两机 `/data/ubuntu/lxh/weavetp/WeaveTP`；env.sh 在仓库外，导出 WORK、REPO、CODE_DIR、NODE1_CODE_DIR、网络、解释器和主机 rank 变量 |

这份用户回传属于 T06 提交，不能替代本次新代码在两机的复核。

## 本次修复与回归

发现 R1 后续缺陷：`stop_case` 在调用 worker cleanup 前强制终止 run RPC。master 的 RPC 就是本地节点监督进程，它被杀后不能保存退出回执；即使随后 worker 清理成功，真实 `test_controller --scenario node1-failure/controller-interrupt` 所需的退出证据仍可能缺失。

新增真实 CPU 子进程回归，模拟监督进程在收到清理通知后写退出回执。旧代码明确失败，原始输出保存在 [initial_r1_failure.log](./initial_r1_failure.log)。修复为：先禁止迟到的 run，再执行两端有 90 秒上限的 cleanup RPC；已确认清理的 run RPC 最多再等待 15 秒写回执；未确认清理或仍未退出的 RPC 强制终止。保留无自动重试、任务标签和未确认状态。新增回归验证异步写回执，不把 Windows CPU 子进程当作真实 Linux/SSH/显存验收。

T04 既有 9×2 实际 shell argv 测试增加方法、调度模式、四次切换、wave/overlap 上限、adaptive、选源门槛、packing 参数及关闭额外开关的断言。没有放松生产门禁、删除断言或将失败改为 skip。

本次只修改 `compare_weavetp_16gpu.py` 的 `stop_case` 和两个相关测试。Python imports 未改变，因此未触发本次 isort 要求；未安装 uv/isort。新增 [run_t07.sh](./run_t07.sh) 仅组织服务器静态/CPU 复核：使用已 source 的环境变量，生成独占 /data 证据目录，关闭字节码并重定向缓存，完成后集中追加 WORK/reports/trial_report.md。它不执行 SSH 或 GPU。

## 实际本地验收

解释器为既有 `C:/Users/ksqbx/miniforge3/python.exe`，Python 3.13.12；Git Bash 为 `C:/Program Files/Git/bin/bash.exe`。所有测试使用 `-B -X utf8`，继承 PYTHONUTF8=1。实际命令、stdout/stderr 和退出码全部记录在 [local_validation.json](./local_validation.json)。

| 检查 | 结果 |
|---|---|
| T02 / T04 / T05 / 服务器检查器 / R1 / T06 | 28 / 26 / 18 / 11 / 4 / 25，全部退出 0，共 112 |
| Python 内存 compile | 22 文件通过，无 pyc 写入 |
| bash -n | 4 个生产入口和 T07 复核脚本，共 5 个通过 |
| 复核脚本中的 Python heredoc | 3 段编译通过；完整 Linux shell 流程未在本机执行 |
| compare shell --dry-run | 9 个轮换 case、18 个节点命令；WARN、batch=8；仅生成命令 |
| 画像 shell --dry-run | node_rank=0/1，两端 INFO 独立环境、同一输出标签；未测量 |
| git diff --check | 通过 |

[compare_dry_run.json](./compare_dry_run.json) 保存实际本地 dry-run 输出：node0 使用本地 Python 调用并生成命令，node1 使用用户给定的 Linux 路径；这不是服务器启动记录。服务器脚本会用 env.sh 中实际 PYTHON 重做 dry-run。两个画像 dry-run 的 Linux 路径来自用户提供的环境信息。

## 最终范围审查与部署身份

累积生产 diff 以初始 `2b96fcc` 为基准，见 [review_diffs.json 的 cumulative_production.patch 字段](./review_diffs.json)；本轮三份代码变更见 [同一 JSON 的 t07_code.patch 字段](./review_diffs.json)。

- `code/current/megatron` 与 `2b96fcc` 的 diff 为空；选源、调度、复制服务、KV 协议没有改变。
- benchmark 的 AST 对比只有 `run_live_benchmark` 改变：实际组读取在第一次 switch_start 前，接收任务扫描/hash/流量汇总和新 collective 在全部切换后；原 `_run_async_waves`、`_validate_cutover` 及其余函数和类不变。
- 三组正式配置、门槛、wave 上限、计时起止与原 BF16-relative 检查不变。INFO 仅在独立画像入口。
- 缺失/无效画像、外部开关污染、标签清理、失败停止、只有 master 结果、缺节点回执、配置漂移、严格续跑等正反例通过；保留服务器实测边界。

[source_manifest.json](./source_manifest.json) 当前固定 29 个本任务链的 Python/shell/fixture 文件及 SHA-256，包括复核脚本和三个审阅回归的测试文件。hash 对应 Git 的 LF 部署字节；本地历史 JSON fixture 的 CRLF 不用于 Linux hash，提交前逐项与 Git 暂存 blob 校验。清单不是全仓库身份的替代；服务器仍必须检出交付消息中的同一完整提交且 Git 工作树干净。新提交完成发布后再在两机执行本目录脚本，不使用旧 T06 提交冒充本次版本。

## T03 门禁独立核对

已查阅本地 T03 回传核验记录：固定 9 份输入 hash、36 次切换和 54/54 项历史参考检查通过，最大误差 0.00047538266060270784 秒，小于 0.000500001 秒；来源为用户历史终端回传。当前汇总脚本 SHA-256 仍为 `e8c6afd4d9230b0d88ef68d90155db49fff8f82fd0ea563c0a310b2c938d6e0f`，与 T03 验收时完全一致。未修改统计逻辑，也未在本轮重新读取服务器历史 JSON；112 项 CPU 回归不作为历史复算证据。

## 执行顺序与未验证项

交付消息提供已推送 main 的完整 SHA。先在 SL3060 执行，成功后在 SL3061 执行同一服务器命令；每段均先 source /data/ubuntu/lxh/weavetp/env.sh，设置 CHECKPOINT，从 GitHub fetch 并检出固定 SHA，将本机 WORK/env.sh 中唯一的 WEAVETP_COMMIT 赋值原位替换为新 SHA，重新 source 并核对，然后调用 `bash "$REPO/documents/16gpu_T07_20261001/run_t07.sh" "$WEAVETP_COMMIT"`。两机都必须更新文件，不能只覆盖当前进程变量。服务器命令未由助手执行，持久化更新由该命令完成。

每机成功应有 `CPU_OK counts=28/26/18/11/4/25 total=112`、`REVIEW_OK tests=3 total_cpu_tests=115`、`T07_STATIC_CPU_OK GIT_CLEAN=1 GPU_STARTED=0` 和 `T07_RESULT ... exit=0`；SL3060 另有 `COMPARE_DRY_RUN_OK cases=9 nodes=2 WARN global_batch=8`。路径固定为 REPO/code/current，CODE_DIR 不一致时直接停止。失败立即停止，保留本次独占 evidence 目录，不进入 T08。

未验证：新提交的 Linux 完整复核、真实双机 SSH/信号/退出回执及超时后清理、GPU 空闲/MPS 基线、NET/IB 实测、16×16 画像、实际 ProcessGroup、真实模型迁移与观测。T08/T09 未启动。现有最终结果检查器仍将跨 case 时间线和原始 NRMSE/cosine 数值缺项标为 blocked；本轮没有把它们改成通过或修改其验收要求。

首次全量暂存 `git diff --cached --check` 因原始 patch 上下文空行和画像 dry-run 自带的行尾空格报错；将这些原始文本无损保存为 JSON 字符串后解决，未修改生产输出或丢弃失败证据。最终暂存检查及 28 个部署 hash 与 Git blob 的核对见 final_review.json。
