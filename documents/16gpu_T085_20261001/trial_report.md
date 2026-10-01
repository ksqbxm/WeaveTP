# T08.5 冒烟入口与本地 CPU 验收

2026-10-01。本任务只准备入口并做本地 CPU/mock 验收；未连接 SL3060/SL3061，未启动 GPU，未执行 T09/T10。真实冒烟尚未运行，不把合成结果或 dry-run 计入实验数据。

## 基线、范围与用户修订

实施前执行 `git -c http.proxy= -c https.proxy= pull --ff-only origin main`，返回 Already up to date；main HEAD 为 `634ac91bb66c050d75da7925de5e1c8a1713bf15`。仅对本次 Git 命令绕过本机失效代理，不修改 Git 配置。工作区仅有用户明确豁免的 `idea.docx`、`~$idea.docx`，不修改、不暂存、不上传。

已重新完整阅读总计划、分步任务、CODEX.md、code/current/AGENTS.md、贡献说明、compare 执行器及 T08 的 run_t08.sh、t08.py。T08 目录、megatron 算法、benchmark 和 summarize_weavetp_formal.py 均不修改。

以本次用户后续指令为准：

- 冒烟不做 NET/IB 或 Socket 检查，不新增替代检查；该阶段由 T08 负责。
- 本轮不修 PS1，compare 的 `rpc`、`rpc_command` 保持原样；没有新增 PS1 保护或 source 包装到这两个函数。
- 冒烟与正式共用 `check_idle`：每卡 81 MiB 上限、相同进程归属规则。结束后仍以相同 GPU UUID、无本任务进程、显存不高于启动基线 +16 MiB 为恢复标准。
- 不改写 benchmark 的 `result.json`，只读取原始字节并记录其 SHA-256。`mode=smoke` 仅写入协调器 request 和完成记录；不向 benchmark 结果插入字段。
- 这是独立 T08.5 冒烟，结果不计入正式九次 launch，不能替代 T08 完成或 T09 数据。

## 文件与行为

| 文件 | 用途 |
|---|---|
| ../../code/current/tools/resharding/compare_weavetp_16gpu.py | `--smoke` 单次 WeaveTP、两次切换；路径隔离、独立结果校验、正式校验拒绝冒烟证据 |
| run_smoke.sh | 保留操作者显式 PROFILE/PROFILE_SHA256，加载既有 env.sh 后进入控制器 |
| smoke.py | 双机预检，复用 compare.run_case，输出中文汇总和失败阶段 |
| test_smoke.py | 冒烟、正式字节回归、原始结果不可改写等 CPU/mock 测试 |
| check_cpu.py | 在指定 /data 目录运行 T02–T08 和 T08.5、语法/编译及保护范围检查 |
| t085_20261001.sha256 | 本目录文件和修改后的 compare 的 SHA-256 清单，不含清单自身 |
| trial_report.md | 使用说明、本地验收与服务器 CPU 命令 |

执行器仅从正式 WeaveTP case 派生一次冒烟，实验环境除 `SWITCHES=2` 外一致；OUT_DIR、RUN_ID 因隔离改变，RUN_ID 以 smoke 开头。正式九次 request/node_commands 不新增字段、不改变序列化顺序。正式路径拒绝 smoke 目录；冒烟路径要求 `$WORK/smoke/` 下新目录，拒绝父目录跳转及实际 Linux 路径中的符号链接。

正式 `validate_result` 拒绝 smoke 路径、同目录 request.json/complete.json 中的 smoke 标记，也防御性拒绝输入 JSON 自带的 smoke 标记。测试包括把冒烟伪装为四次切换，以及复制到非 smoke 目录但保留任一协调器标记的情况。若人为剥离全部来源标记并改造内容，则不在这些可观察来源证据的保证范围内。

独立冒烟校验要求两机退出成功、checkpoint_loaded、恰好扩容再缩容、原 BF16-relative 配置、实际 16 rank 的 TP2/TP4 DP/EDP 组以及两个方向的 default/adopted 计划。BF16-relative 失败会使原 benchmark 抛异常、无法完成原始 JSON；不以 validation_max_diff 代替 NRMSE，也不伪造未保存的 NRMSE/cosine 数值。

汇总表仅记录 wave 数、Transport（逐 wave transport_s 求和）、迁移/切换墙钟、default/adopted 跨机逻辑字节。default 扩容 14.4 GB、缩容 28.8 GB 为十进制参考值，偏离超过 20% 仅 WARNING，不改变退出码。

沿用现有 launch `process.wait(timeout=3600)`（compare 第 318 行）和 run RPC `communicate(..., timeout=3650)`（第 359 行）。没有逐切换超时、自动重试、后台空卡守候或额外 GPU launch。失败/中断交由同一 compare 清理精确 OUT_DIR 标签的任务；无法确认清理时证据中保留“未确认”。

## 冒烟操作（T08 成功之后）

必须从 T08 成功输出逐字复制 PROFILE 和 PROFILE_SHA256。两机校验指定画像实际 hash 与同目录 `profile.json.sha256`；不查找最新目录，不修改 T08 证据或画像，也不检查 NCCL 日志。

在 SL3060 的 tmux 中运行，SL3061 不独立启动第二个控制器：

```bash
# T08 跑完之前不要执行
tmux new -s weavetp_t085
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO/documents/16gpu_T085_20261001"
sha256sum -c t085_20261001.sha256
PROFILE='<T08 成功后打印的完整路径>' \
PROFILE_SHA256='<T08 成功后打印的完整 SHA256>' \
bash ./run_smoke.sh
```

断线后在 SL3060 执行 `tmux attach -t weavetp_t085`。Ctrl+C 通过执行器只清理本任务；不关闭 MPS，不杀他人的进程。失败保留证据后停止，不复用该目录重跑。

启动前必须满足两机干净 HEAD 等于 env.sh 的完整 WEAVETP_COMMIT、代码路径一致、NCCL_DEBUG=WARN、各自 /data 剩余至少 20,000,000,000 bytes、16 卡通过原正式资源门禁。compare 在 prepare/run 时继续使用原来的重复资源和配置检查。

ROOT_OUT 自动生成为 `$WORK/smoke/smoke_<UTC时间>_<pid>/`。SL3060 根目录包含 request.json、两个 preflight 回执和中文 trial_report.md；两端 `weavetp/r1/` 分别保存 request、prepared、退出回执及 node 日志，rank 0 保存原始 result.json 和 mode=smoke 的 complete.json。失败时保留 failure/cleanup 记录；控制台打印失败阶段与证据根目录。旧目录冲突不会向旧目录写入任何新文件。

## 本地验收

本机 Windows，使用现有 `C:/Users/ksqbx/miniforge3/python.exe` 和 Git Bash；输出在 `E:/data/weavetp_t085_cpu_20261001/`。本地 /data 对应当前 E: 盘的 `E:/data`，服务器仍是 Linux `/data`。

```powershell
python -B -X utf8 documents/16gpu_T085_20261001/check_cpu.py --output-dir E:/data/weavetp_t085_cpu_20261001/final
```

最终现有 131 项 + 新增 24 项，共 **155/155 通过，0 失败、0 错误、0 跳过**。逐套件为 T02 28、T04 26、T05 18、服务器检查器 11、R1 4、T06 25、T06/T07 审阅回归 3、T08 16、T08.5 24。16 个 Python 文件 py_compile、7 份 shell 的 bash -n、git diff --check 均通过。报告中的 3 段 Bash 命令和 2 段内嵌 Python 也做本地语法检查，不远程执行。

正式 dry-run 用同一组固定环境，基线代码通过 `git show 634ac91...:code/current/tools/resharding/compare_weavetp_16gpu.py` 在内存执行，直接比较 stdout 原始字节，不重排 JSON、不归一化换行。修改前、修改后各 35,168 bytes，两边 SHA-256 均为 `08243d344b7a25e7ebd728fa85f3f46243752fb39791f68b3324d1511c0a1045`，也与修改前现场保存的输出一致。CPU 验收目录保留 formal_before.json、formal_after.json、smoke_dry_run.json、逐套件输出和 validation.json。

T08 测试通过 unittest 加载，不执行其会重写 local_validation.json 的主入口；临时文件统一重定向到本次 CPU 验收目录。py_compile 显式指定外部 cfile，避免在锁定目录生成字节码。运行前后核对 T08/正式汇总脚本的文件字节及禁止修改目录的 Git diff。

按 AGENTS.md 尝试 `uv run isort`，本机无 uv，命令未能执行；未安装任何依赖。Python 导入排序工具检查标记为未验证，不宣称通过。Linux SSH、真实信号传播、NCCL/GPU、显存恢复尚未实测，留待操作者在 T08 完成后运行冒烟。

## 两机服务器 CPU 验收命令

**T08 跑完之前不要执行。** 以下只更新代码和运行 CPU 测试，不执行 run_smoke.sh。`TARGET_COMMIT` 使用交付消息中的完整提交 SHA。

先在 SL3061 拉取 GitHub，检出交付 SHA，并完成 CPU 验收：

```bash
# T08 跑完之前不要执行 —— SL3061
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
TARGET_COMMIT='<交付消息中的完整 SHA>'
test "$(hostname -s)" = SL3061
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
git -C "$REPO" fetch origin
git -C "$REPO" checkout --detach "$TARGET_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$TARGET_COMMIT"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
"$PYTHON" -B - "$WORK/env.sh" "$TARGET_COMMIT" <<'PY'
import re, sys
from pathlib import Path
path = Path(sys.argv[1])
raw = path.read_text()
updated, count = re.subn(r'(?m)^(\s*(?:export\s+)?WEAVETP_COMMIT=).*$',
                        lambda m: m[1] + "'" + sys.argv[2] + "'", raw)
if count != 1:
    raise SystemExit('STOP: expected exactly one WEAVETP_COMMIT assignment')
path.write_text(updated)
PY
source "$WORK/env.sh"
test "$WEAVETP_COMMIT" = "$TARGET_COMMIT"
cd "$REPO/documents/16gpu_T085_20261001"
sha256sum -c t085_20261001.sha256
"$PYTHON" -B -X utf8 check_cpu.py --output-dir "$WORK/acceptance/t085_cpu_SL3061_$(date -u +%Y%m%dT%H%M%SZ)_$$"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
```

SL3061 成功后，在 SL3060 经局域网取得该提交；本段没有 SL3060 直连 GitHub 的 fetch：

```bash
# T08 跑完之前不要执行 —— SL3060
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
TARGET_COMMIT='<交付消息中的完整 SHA>'
test "$(hostname -s)" = SL3060
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
git -C "$REPO" fetch "ubuntu@10.60.14.2:$REPO" HEAD
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = "$TARGET_COMMIT"
git -C "$REPO" checkout --detach "$TARGET_COMMIT"
test "$(git -C "$REPO" rev-parse HEAD)" = "$TARGET_COMMIT"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
"$PYTHON" -B - "$WORK/env.sh" "$TARGET_COMMIT" <<'PY'
import re, sys
from pathlib import Path
path = Path(sys.argv[1])
raw = path.read_text()
updated, count = re.subn(r'(?m)^(\s*(?:export\s+)?WEAVETP_COMMIT=).*$',
                        lambda m: m[1] + "'" + sys.argv[2] + "'", raw)
if count != 1:
    raise SystemExit('STOP: expected exactly one WEAVETP_COMMIT assignment')
path.write_text(updated)
PY
source "$WORK/env.sh"
test "$WEAVETP_COMMIT" = "$TARGET_COMMIT"
cd "$REPO/documents/16gpu_T085_20261001"
sha256sum -c t085_20261001.sha256
"$PYTHON" -B -X utf8 check_cpu.py --output-dir "$WORK/acceptance/t085_cpu_SL3060_$(date -u +%Y%m%dT%H%M%SZ)_$$"
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
```

两机都必须得到 CPU 验收 `ok=true`；这不代表真实冒烟已经通过。运行产生的服务器日志、JSON、报告和编译缓存均保留在 /data，不提交 GitHub。
