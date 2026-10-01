# T06 实际并行组断言与计划观测

**后续更新：T04 过时断言已按用户要求修正，六组 CPU 测试 111/111 全部通过；本次 isort 由用户明确免除。下文保留初版阶段事实（含当时的失败和未提交状态），最终验收以文末追加记录及 `final_cpu_validation.json` 为准。初版独立分支推送示例已被本轮直接提交 main 的授权取代。**

日期：2026-10-01。范围为 T06 编码、CPU 测试及对应的只读结果检查；未执行 T07/T08/T09。工作起点为 `ab5042d`，本次未提交或推送，未连接服务器，未安装任何依赖，未启动 GPU。

## 本次阅读与实现

已完整阅读根目录 CODEX.md、16 卡总计划、分步计划 T06、code/current/AGENTS.md 和贡献说明；检查实际 benchmark、模型 ProcessGroup 来源、`build_inference_pg_collection`、planner 全局门槛恢复逻辑、TransferOp/ReshardPlan、`restrict_plan_sequence`、`collect_transfer_tasks`、adaptive 策略及服务器测试入口。

实现文件：

- `code/current/examples/rl/benchmark_live_moe_tp.py`：37 行新增接线。仅 world_size=16 启用实际组观测；保留切换计划引用，结束后统计。
- `code/current/tools/resharding/weavetp_observations.py`：标准库观测辅助函数，不导入 torch、不执行迁移。
- `code/current/tests/unit_tests/resharding/test_weavetp_observations.py`：25 项 CPU 测试。通过 AST 装载生产 KV 区间限定函数和 adaptive 决策，避免用另写的算法替代被测实现；实际进程组读取接口用 mock 验证，不声称真实 GPU 组已通过。
- `code/current/tests/server_tests/weavetp_16gpu/test_observations.py`：只读现有 result.json，输出独立 acceptance.json。
- `code/current/tests/server_tests/weavetp_16gpu/test_cpu.py`、`test_results.py`：接入 T06 检查；保留正式结果验收其他未完成项。
- 分步任务和服务器 README：记录状态、字段、使用命令及证据限制。

## 观测口径

`parallel_groups` 来自 TP2/TP4 模型实际持有的组，调用 `get_world_size` 和 `get_process_group_ranks`，不是将公式推导值当实测。逐 rank 验证 TP2 的 DP=8/EDP=4、TP4 的 DP=4/EDP=2、EP=2、PP=CP=1，以及成员互相一致、TP/ETP/EP 均机内。保存 hostname、global/local rank、当前 CUDA device、GPU UUID、实际 NCCL_DEBUG。

每次切换的 `plan_observation` 保存：

- `default/candidate/adopted`：全量接收任务表身份 hash、按接收端/源端/类别聚合的 `receiver_rows` 及可复算的流量。两个方向均保留默认计划。默认与候选使用该切换相同的有效序列区间；候选流量不代表它实际执行过。
- `weight/kv_prefix/kv_delta`：分别计数，KV 为 `[0,s)` 和 `[s,c)`，不使用容量 1024 代替有效长度。
- `local_copy_bytes` 单列；远程字节排除 src=dst。机内跨卡计入远程收发量，跨机额外记录 SL3060→SL3061、反向和双向之和。峰值为每卡远程字节总量，保留所有并列 rank；不是瞬时速率或网卡实测。
- `candidate_gate`：保留缓存计划的完整 source_route_stats、预测全局收益、门槛 5%、global_gate_accepted、rejected_global。分类为 accepted、rejected_global、no_effective_reroute、disabled、not_applicable，证据缺失为 unrecorded。收缩不适用。
- 全局否决时 planner 已丢弃前候选任务表，因此 `candidate=null`，原因 `pre_global_gate_task_table_not_retained`。恢复的默认任务表不会被冒充候选。
- 请求/实际计划、实际 `base.execution_mode`、adaptive snapshot 和回退原因独立记录。缓存全局筛选通过后仍可 adaptive 回退；不将两者混为一次门槛判断。

统计仅扫描 recv_ops，send 镜像不重复计数；重复接收 task ID 拒绝。指纹描述接收任务表及切片/元数据，不哈希 tensor 内容。旧结果没有这些字段时，检查器拒绝，不能补造历史观测。

## 计时与范围审查

实际组读取与一次汇总在第一次 switch_start 前。每次 switch_wall_s 结束后仅保存已有计划的引用；所有接收任务扫描、hash、流量计算与新增汇总 collective 位于全部四次切换之后。没有在迁移或 wave 热路径增加同步，没有调整现有计时起止位置。

AST 对比起点版本：benchmark 只有 `run_live_benchmark` 函数发生变化，其余函数和类完全一致，包括 `_run_async_waves`、`_validate_cutover`。planner、live.py、KV 协议、复制服务、调度算法、阈值、launcher 和三组固定参数均未修改。新增引用只延长少量计划元数据的存活期，不复制 GPU tensor。

## 本地验收

实际解释器：`C:\Users\ksqbx\miniforge3\python.exe`，Python 3.13.12。所有测试显式 `-B -X utf8`，并设置 PYTHONUTF8=1。完整命令、stdout/stderr、退出码和 AST/compile 清单见 [local_validation.json](./local_validation.json)。

| 测试 | 结果 | 退出码 |
|---|---:|---:|
| test_weavetp_observations.py | 25/25 | 0 |
| test_summarize_weavetp_formal.py | 28/28 | 0 |
| test_profile_weavetp_16gpu.py | 18/18 | 0 |
| test_weavetp_review_findings.py | 3/3 | 0 |
| test_server_acceptance.py | 11/11 | 0 |
| test_compare_weavetp_16gpu.py | 24/26，两个已有失败 | 1 |

T06 覆盖实际组读取接口、DP/EDP 错误、跨机 TP/ETP/EP、rank/device/UUID 错误、双向计数、机内远程、同卡排除、收发守恒及并列峰值、KV 8/3 有效区间和零长度、分步切片、重复接收拒绝、计划身份、各类 False gate、被否决前候选不可得、通过后 adaptive 回退、实际 FIFO/residual、收缩不适用、结果检查器正反例，以及统计调用位置。

6 个本次新增或修改的 Python 文件使用内存 `compile` 检查通过，无 pyc 写入。`git diff --check` 退出 0。初次测试脚手架因构造 AST 缺 lineno 退出 1，已用 `fix_missing_locations` 修正，最终 25 项全部通过。

T04 两个失败与分步计划已有记录一致：formal shell 测试未提供现在要求显式注入的 PYTHON/MASTER_ADDR/NODE1_SSH；argv 测试仍期待旧地址 `10.60.14.1`，fixture 当前生成 `SL3060`。本次未改 T04 实现或其测试，未把失败改为 skip。T07 全套集成验收仍未通过。

按源码规则实际尝试 `uv run isort`，PowerShell 报 uv 不可识别，退出 1；本机也没有 isort 模块。未安装依赖，导入排序工具检查标为未验证。

## 本地推送命令（尚未执行）

在仓库根目录的 PowerShell 中执行。以下只暂存本次文件，推送到当前个人仓库 origin 的独立分支：

```powershell
git switch -c codex/task6-observations
git add -- code/current/examples/rl/benchmark_live_moe_tp.py code/current/tools/resharding/weavetp_observations.py code/current/tests/unit_tests/resharding/test_weavetp_observations.py code/current/tests/server_tests/weavetp_16gpu/test_observations.py code/current/tests/server_tests/weavetp_16gpu/test_cpu.py code/current/tests/server_tests/weavetp_16gpu/test_results.py code/current/tests/server_tests/weavetp_16gpu/README.md "documents/WeaveTP_16卡正式实验_分步任务.md" documents/16gpu_T06_20261001
git diff --cached --check
git diff --cached --stat
git commit -m "Add actual parallel group and migration plan observations"
git push -u origin codex/task6-observations
git rev-parse HEAD
```

按源码贡献规则，提交前的 `uv run isort` 检查目前仍缺工具；如果使用已有 uv/isort 环境，在 `code/current` 下对上述六个 Python 文件运行排序后，再运行本报告 CPU 测试、检查 diff，并提交。不要为此自动安装或同步大批项目依赖。

## 服务器 CPU 测试命令（尚未执行）

两机从 GitHub 检出同一 commit，工作副本必须在 `/data`。下列 REPO 为示例，需改成实际包含 `code/current` 的 Git 根目录；PY 在每台机器上指向已存在的解释器，SL3061 不假定已部署环境。缺少工作副本或解释器时先停止处理部署，不执行安装。

两机分别执行，COMMIT 使用上面推送后的完整 SHA，不分别取可能已变化的分支头：

```bash
set -euo pipefail
: "${SERVER_USER:?请先设置实际服务器用户名}"
WORK=/data/${SERVER_USER}/lxh/weavetp
REPO="$WORK/WeaveTP"                   # 改为实际 Git 工作副本根目录
PY=/home/${SERVER_USER}/miniconda3/envs/megatron/bin/python  # 每台机器核实
COMMIT=替换成推送后的完整commit

test -d "$REPO/.git"
test -x "$PY"
test -z "$(git -C "$REPO" status --porcelain)"
git -C "$REPO" fetch origin codex/task6-observations
git -C "$REPO" switch --detach "$COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$REPO" status --porcelain)"
git -C "$REPO" rev-parse HEAD

CODE="$REPO/code/current"
RUN="$WORK/acceptance/t06_$(hostname)_$(date +%Y%m%dT%H%M%S)_$$"
mkdir -m 700 "$RUN"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
for key in TMPDIR TMP TEMP XDG_CACHE_HOME CUDA_CACHE_PATH TORCH_HOME TORCH_EXTENSIONS_DIR TRITON_CACHE_DIR HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE PIP_CACHE_DIR; do
  mkdir "$RUN/$key"
  export "$key=$RUN/$key"
done
cd "$RUN"
"$PY" -B -X utf8 "$CODE/tests/unit_tests/resharding/test_weavetp_observations.py" 2>&1 | tee t06_cpu.log
test -z "$(git -C "$REPO" status --porcelain)"
```

预期 `Ran 25 tests`、`OK`、退出 0。两机 HEAD 输出需相同。此命令只验证 CPU 逻辑和 mock 组读取，不加载 checkpoint 或 GPU。

如需要收集整个服务器 CPU 门禁，可在同一环境执行：

```bash
"$PY" -B -X utf8 "$CODE/tests/server_tests/weavetp_16gpu/test_cpu.py" --output-dir "$RUN/all_cpu"
```

当前该命令仍会因 T04 两个旧断言退出 1，应保留失败日志，在 T07 解决；不能宣称集成通过。

## 日后只读验收正式结果（尚未执行）

T07/T08 门禁满足后，使用原计划的 r1 正式 launch 验证实际组与观测，不增加额外性能 launch。既有 GPU 前置资源/批准要求不变。本阶段不提供或执行一个绕过门禁的 torchrun。

已有正式结果后，在上面的 CPU 环境中执行（RESULT 改为真实文件，output-dir 必须是独立新目录）：

```bash
RESULT="$WORK/正式ROOT_OUT/weavetp/r1/result.json"
"$PY" -B -X utf8 "$CODE/tests/server_tests/weavetp_16gpu/test_observations.py" \
  --result "$RESULT" --output-dir "$RUN/weavetp_r1_observations"
```

同样检查 Fixed、Directional 的 r1 及后续正式结果；9 份齐全时 `test_results.py` 会逐份调用同一个 T06 检查器。缺字段、组错误、非 WARN、候选状态与证据不一致、流量无法复算均失败。整个正式结果检查仍保留跨 case 时间线和原始 NRMSE/cosine 缺项，因此不能因 T06 通过就宣布 T10 完成。

## 未验证范围

真实双机 ProcessGroup、GPU UUID、四次迁移产生的流量记录、真实 FIFO/residual/adaptive 分支、服务器读结果入口及资源清理均未执行。当前 CPU 证据不能替代 GPU 实验或证明网络瓶颈。服务器 WORK/reports/trial_report.md 尚未回写；本文件和 local_validation.json 是本地阶段记录。

## 2026-10-01 后续：T04 测试修正与 main 发布验收

已重新完整阅读 CODEX.md、总计划、分步任务、源码规则及贡献说明。用户本轮授权：仅修改测试修正 T04 两个过时断言；不要求 isort；全部相关 CPU 测试通过后直接提交推送 main；提供两机固定 commit、仓库外 env.sh 和 CPU 测试命令，不连接服务器执行。

T04 的生产代码没有新增修改。测试 fixture 显式使用文档保留地址 `192.0.2.10`、`192.0.2.11` 和端口 `29617`；dry-run 自行提供 master 地址、node1 SSH、两端解释器和代码目录、checkpoint 等环境，argv 对照输入而非旧服务器地址。两端 dry-run 使用当前测试解释器，避免 Windows Git Bash 对虚构 POSIX 解释器路径的转换影响断言；未关闭测试或放松生产校验。

最终六组测试均直接以既有 Python `-B -X utf8` 运行；T02 28、T04 26、T05 18、服务器检查器 11、R1 3、T06 25，共 111 项，退出码均为 0。8 个相关 Python 文件编译检查、4 个 launcher 的 `bash -n` 均通过。完整命令/原始输出见 [final_cpu_validation.json](./final_cpu_validation.json)。未安装依赖或启动 GPU。

发布身份：读取远端后确认 R1 commit `ab5042d13fc0226da578aca06745cabd3894dcf2` 已在 origin/main；本次 T04/T06 提交直接继承它，不 squash 或改写 R1。第一次 fetch 因本机已失效的 127.0.0.1 代理失败，使用仅对本条命令生效的 `git -c http.proxy=` 后 fetch 成功，没有修改全局代理配置。最终提交 SHA 和远端核对结果在交付消息中给出。

服务器尚未运行。收到两机回传时核对相同 SHA、空工作树、env.sh 路径在 `/data/ubuntu/lxh/weavetp` 且仓库在其子目录、两机 NODE_RANK=0/1 和各自现有解释器；CPU acceptance.json 应为 passed，六组 exit_codes 均为 0，每组原日志的 Ran 数为 28/26/18/11/3/25。任何失败均保留日志，不继续 GPU。
