# T08 审阅、优化与当前执行命令

2026-10-01。已完成本地审阅和修正；**真实服务器画像尚未执行，T08 GPU 验收仍未完成**。本次没有 SSH、安装、复制模型或启动 GPU，不进入 T09。

## 前置状态与范围

本次重新完整阅读 CODEX.md、总计划、分步任务 T08、code/current/AGENTS.md，并阅读既有 profiler/launcher、compare 的身份/进程/显存检查、服务器资源和 checkpoint 检查器、T08 脚本及测试。用户确认两机在 `09273f55…` 均为 `T07_RESULT exit=0`、`GPU_STARTED=0`，来源仍是用户回传。T03 历史 54/54 验收记录继续有效。

修改仅在本目录及分步任务记录。`code/current` 无改动，服务器使用本次发布提交的干净工作副本；`code/current` 内容与已通过 T07 的 `09273f55bd24e46733253bb62dc80e0625d523c7` 完全一致。画像参数、NCCL 环境和算法保持不变。

## 审阅结论与修正

1. **正确性：中断后只写回执，未保证子进程退出。** 原 run 分支遇到 KeyboardInterrupt 时直接进入 finally，可能留下仍在启动的 shell/timeout 子进程；外层看到 `launcher_exited=false` 只报未确认。旧测试恰好只要求这个 false 值，未要求清理子进程。现由节点直接监督既有 launcher dry-run 生成的带精确 OUT_DIR 标签的 torchrun 命令，所有失败、中断和 60 分钟超时进入同一 `stop_profile` 清理函数。
2. **完备性：启动与取消之间缺少互斥。** 原取消标记与目录轮询无法把“未创建目录”与“将来不会创建目录”统一起来。现启动和取消共用 Linux flock；取消先禁止后续启动，再终止精确标签进程。启动临界区暂存 SIGINT/SIGTERM，Popen 返回并取得子进程句柄后才处理信号，避免 fork/exec 间隙丢失句柄。不再用监督进程旧回执判断清理成功，最终判断来自本任务进程和显存状态。
3. **正确性：阶段间配置可以漂移。** 原 prepare 后的各阶段重新 source env.sh，实际启动时没有与先前已配对的 settings 比较。现两机 prepare 各保存一次 `deployment.json`，配对核验后只使用这份环境；后续不再读取 env.sh。源码仍在每次 GPU 启动前核验固定 commit 和 clean 状态。checkpoint 固定为已部署的 `/data/models/DeepSeek-V2-Lite-megatron-v2`，没有备用路径。
4. **简洁性与可测试性：清除嵌套执行和兼容适配。** 删除 Bash 内嵌 Python、Bash 后台 job/wait/trap 调度、外层 timeout 启动器、启动目录轮询、launcher_exited/launch_started 握手，以及为调用正式 compare.cleanup 临时构造 prepared.nodeN.json 的逻辑。`run_t08.sh` 仅加载环境并进入 `t08.py`；节点和控制器使用同一 Python 文件。测试直接调用实际函数，删除 AST 截取执行的旧测试路径。

修正前两项可复现行为保存在 [review_findings.json](./review_findings.json)：中断后没有 cleanup 调用且启动器未退出，以及准备端口 29500 后实际启动使用变更值 29999。它们是执行旧分支的 CPU 替身证据，不是服务器实测。旧可执行实现已被替换，没有保留兼容开关或另一条启动路径。

## 当前实现

- [run_t08.sh](./run_t08.sh)：唯一 shell 入口，6 行，source 既有 env.sh 后执行同目录的 t08.py。
- [t08.py](./t08.py)：SL3060 单一控制器；通过 SSH 将同一份 Python 源码送到 SL3061。无需在 SL3061 上传辅助文件或手动启动第二份任务。
- [check_commands.py](./check_commands.py)：直接函数回归、真实 CPU 子进程退出检查、控制器场景与实际 launcher dry-run。
- [t08_20261001.sha256](./t08_20261001.sha256)：交付给服务器的两个运行文件的 SHA-256 清单。

资源门禁保留：两机 Git HEAD 必须等于 env.sh 中相同的完整 `WEAVETP_COMMIT`、工作树为空；环境版本一致；两机每次 GPU 启动前 `/data` 剩余至少 20,000,000,000 bytes；8+8 张 GPU 空闲。既有 check_idle 上限为每卡 81 MiB（65 MiB 基线加 16 MiB 容差），只允许 SL3060 上指定 MPS_OWNER 的已有 MPS server。有占用立即停止，不等待空卡、不关闭 MPS。

240 个有向非对角对、64 MiB、每对一次预热、3 轮各 3 次传输、十进制 Gbps 均沿用 T05；对角线 1000 Gbps 为非实测估计。画像独立 INFO、eno1np0、mlx5_0、自动 GID，不继承外部 Socket 或强制 GID 设置。成功要求两端各 8 份原始 NCCL 日志有实际 `via NET/IB` 且无 Socket 数据通路，校验同一 16×16 JSON 和 SHA-256，检查本任务进程及显存恢复。

所有新增服务器文件和缓存均位于 /data。同步仅涉及 checkpoint hash 报告、部署信息及 profile.json，不复制模型。失败保留证据，不自动重试；远程操作有有限时限，失联或残留状态无法确认时明确标记“未确认”。两机按部署和画像两个阶段追加 WORK/reports/trial_report.md。T09 仍须显式使用 NCCL_DEBUG=WARN。

## 本地验证

实际执行 `python -B -X utf8 documents/16gpu_T08_20261001/check_commands.py`，退出 0；[local_validation.json](./local_validation.json) 记录命令、完整输出和源码 hash。

- T08 16 项 unittest 通过，其中协调器测试涵盖 9 个场景：成功、两端各自预检失败、两端各自 run 失败、node1 验收失败、checkpoint 不一致、配置不一致、清理 SSH 不可达。
- 真实 CPU 子进程在中断/超时后退出；取消先到、启动/取消竞争、启动临界区 SIGTERM、缺少旧退出回执时仍能根据最终状态确认、显存未恢复/进程残留/观测不可用、磁盘不足/错误 commit/非完整 SHA/GPU 占用均有回归。
- 真实旧 launcher 仅执行 `--dry-run`，确认 INFO、mlx5_0、2×8、无重启及精确 OUT_DIR；node CLI 参数路径通过检查。
- 画像验收使用原校验器验证合成文件：正常通过，Socket 数据通路和 hash 不匹配均失败；原有 18 项画像测试全部通过。
- shell 语法、两个 Python 文件内存编译、`git diff --check` 通过；未修改 code/current。

本机为 Windows，竞争测试用线程锁代替 Linux flock；真实 Linux flock、SSH、信号传播、GPU/NCCL 和显存恢复仍需服务器验证。没有把 CPU 子进程测试或合成画像写成真实 T08 通过。

## Git 发布与服务器执行

按用户要求，本版提交并推送 main，不再手动上传辅助文件。两机由用户自行准备本次提交对象，再按最终交付消息中的完整 SHA 执行本地 checkout、验证干净工作树、持久化更新 WORK/env.sh 中唯一的 WEAVETP_COMMIT，并验证本目录 SHA-256 清单。服务器命令不执行 git fetch/pull。

发布所需的唯一代码调整：移除写死的 T07 仓库 SHA。prepare/preflight/run 均校验冻结配置中的完整 40 位小写 SHA 与实际干净 HEAD；两机部署配对也要求该 SHA 一致。最终交付消息提供准确发布 SHA，避免把提交自身 SHA 写入其文件造成循环引用。

执行顺序：先 SL3060 完成 checkout 与环境更新（10–30 秒），再 SL3061 执行相同操作（10–30 秒）。两机都出现 T08_CODE_READY 后，只在 SL3060 启动控制器；SL3061 由 SSH 驱动，不另开画像任务。

控制器预计 10–35 分钟：两机 checkpoint 校验约 5–20 分钟，资源/网络/端口预检约 30–90 秒，画像约 3–10 分钟，同步验收约 10–30 秒。时间是估计，不是实测保证。以下启动命令要求两机已 checkout 最终交付 SHA，并更新 WEAVETP_COMMIT：

```bash
bash <<'T08'
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO/documents/16gpu_T08_20261001"
sha256sum -c t08_20261001.sha256
mkdir -p "$WORK/reports"
bash ./run_t08.sh 2>&1 |
  tee "$WORK/reports/t08_console_$(date -u +%Y%m%dT%H%M%SZ).log"
T08
```

启动 GPU 之前必须先看到两机的 `T08_PREFLIGHT_OK node=0/1 >=20GB GPUs=8 idle`；每个节点在实际启动前再次检查磁盘和 GPU。任何门禁失败都停止，不等待或自动重试。

成功须看到实际 `via NET/IB` 日志、两条 `T08_PROFILE_OK node=0/1 NET/IB matrix=16x16 pairs=240 sha256=...`，以及最终 `T08_RESULT exit=0`。两条 sha256 必须相同；输出同时给出 PROFILE 和 PROFILE_SHA256。两机 measurement 目录保留 profile.json、profile.json.sha256、nccl.*.log、nccl.sha256；节点启动日志在任务根目录 launcher.node0/1.log。成功回传前不勾选 T08 完成。

本版发布后停止本地重构，直接进入服务器实跑；只针对实跑暴露的问题修复。不进入 T09。
