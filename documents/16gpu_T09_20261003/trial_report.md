# T09 W2 正式配置与操作准备（2026-10-03）

**仅本地 CPU 准备；未连接 SL3060/SL3061、未部署、未启动 GPU、未做冒烟。正式九次运行未执行。**

已重新完整阅读 CODEX.md、总计划、分步任务、code/current/AGENTS.md、贡献说明，以及 `outputs/w2_20261002T220122/formal_change_review.md` 与 `formal_review.json`。

基线 main：`3bc12ce74a0ed048d62a9ffd5ca64d1940779705`。首次 git pull 因本地 127.0.0.1:7897 代理不可达失败；用户随后明确“不用拉取”，允许已有探索目录、画像、Office 锁文件保留且不提交。tracked 文件起始干净。T08 成功为用户提供的证据；本地画像 SHA-256 已独立核验，未修改或提交画像。

本次只对 WeaveTP case 设置 `ALLOW_AWARE_SHRINK=1`、`REROUTE_MIN_GAIN_PCT=0.0`，其余环境不变。正式验收要求四次切换的 adopted 几何等于 aware candidate、不同于 default，且各方向全局门槛 5% 通过。baseline/FIFO 仅作为策略与执行标签保留。Fixed/Directional 的生成请求、节点命令及 validate_result 分支不变。

配置来源为执行器 request.json 和两端 exit 回执；首次完成前核验执行器已收集的两端 RPC exit 回执（peer 文件不在 master 本地），complete.json 写成后及续跑时读取其 exits，并核对 request/config/result hash。benchmark 没有输出 gain 字段，故不把该缺失字段冒充运行取证；benchmark 与 megatron/ 完全未改。

## 第 5 项：每机磁盘写入估算

十进制 MB/GB，按每次独立 launch 的冷缓存保守预算；这不是实测磁盘上界。历史九份 result.json 为 0.509–0.576 MB；16 卡新增组/计划流量观测，因此结果预留 16 MB。launcher 以 tee 写 run.log，执行器又捕获到 nodeN.log；rank 0 会打印完整 JSON，所以日志必须重复预算。正式白名单不传 NCCL_DEBUG_FILE，WARN 消息计入这两份日志，不把 T08 INFO 日志重复算到正式 launch。

| 每 launch 项目 | SL3060 MB | SL3061 MB | 依据 |
|---|---:|---:|---|
| run.log（含 NCCL WARN） | 20 | 20 | 四次切换、初始化及 JSON 打印的日志预留 |
| nodeN.log（同一 stdout 的第二份） | 20 | 20 | 执行器捕获 launcher 输出 |
| result.json | 16 | 16 | 仅 rank 0 实际写；peer 仍按同一较大预算留空 |
| request/prepared/exit/complete、预检/失败/清理回执及进度 | 4 | 4 | complete 包含两端环境，非模型内容 |
| CUDA/Torch extensions/Triton/HF 等本 launch 缓存 | 384 | 384 | 已安装环境，无下载模型；仍为冷编译预留，未在服务器实测 |
| tmp/其他临时文件余量 | 56 | 56 | 按每次独立输出路径保留 |
| 合计 / launch | **500** | **500** | **0.500 GB/机** |
| 九次累计 | **4500** | **4500** | **4.500 GB/机** |

计划对象与权重/KV 缓存驻留 RAM/GPU，不由该 benchmark 序列化到磁盘；只读 checkpoint，无 checkpoint 保存或模型复制。计划 gzip 是本地探索产物，不属于服务器正式输出。共享盘可能被别人继续写入，失败目录也占空间；每个 case 前两端重新检查，低于门槛立即退出，无等待或清理他人数据。缓存和异常日志不受硬配额限制，实际写量仍有不确定性。

## 第 6 项：磁盘门槛

| 项目 | 数值 |
|---|---:|
| max(5 GB, 九次每机写量 × 2) 向上取整 | max(5, 4.5 × 2) = **9 GB** |
| t09.py 常量 MIN_FREE_BYTES | **9,000,000,000 bytes** |
| 用户给出的 SL3061 当前约 17 GB 余量 | 约高于门槛 8 GB；运行时重新查询 |
| 是否超过必须停止提交的 15 GB | 否 |

## 第 13 项：耗时依据与排期预算

**历史 result.json 没有 launch 起止时间/elapsed 字段，不能从中恢复加载、组初始化、初始权重同步等耗时。** 可直接复算的是每份 JSON 四次 switch_wall_s 之和；不能将它标成完整 launch 耗时。下表的 16 卡切换总和来自第 15 项 CPU 模型，不是 GPU 实测。

| 方法 | 8 卡四次切换总和均值 s（范围） | 16 卡四次切换预测 s |
|---|---:|---:|
| Fixed（8 卡 WeaveTP 为 W0） | 38.573（37.686–39.696） | 70.400 |
| Directional（8 卡 WeaveTP 为 W0） | 33.656（33.408–33.997） | 64.809 |
| W2（8 卡 WeaveTP 为 W0） | 33.354（32.785–34.260） | 39.225 |

九次合计切换预测 **523.302 s（8.72 分钟）**。

| 完整运行排期 | 预算 | 证据边界 |
|---|---|---|
| 单 launch | 5–20 分钟 | 工程排期假设，给模型加载/初始化留量；历史 JSON 不支持实测校准 |
| 九次完整 launch | 45–180 分钟，加部署/预检 | 9 × 单次预算；不承诺实际耗时 |
| 硬停止 | 单 launch 60 分钟 | 既有执行器 process.wait(timeout=3600)，无自动重试 |

第 13 项完整 launch 的历史实测依据缺失；需要历史 launcher 起止日志才能进一步收紧预算。此次没有为取得缺失数据连接服务器，也未改 benchmark。

## 第 15 项：真实与合成画像并列 CPU 预测

三参数模型保持原校准：`T_switch = a_direction + b × base_waves + M`，a_expand=1.585701538 s、a_shrink=2.345708555 s、b=0.404996370 s/wave；不重新拟合。共享 NIC 每方向 **2.84 GB/s**、全双工；每波跨机两个方向取较大值，再与按完整真实 16×16 矩阵计算的机内/同卡服务时间取最大值。

合成对照使用原 2.85 GB/s 合成矩阵重新规划，服务模型也统一用 2.84 GB/s；真实方案用 T08 原矩阵规划。W2 两方向在真实画像下 gate 均 accepted，四次有效 KV 轨迹按实际 base wave 推进。Fixed/Directional 不选源。预测仅为探索；foreground pressure=0 的静态 FIFO 路径，不模拟在线反馈或全部初始化成本。

| 方法/方向 | 合成跨机 GB | 真实跨机 GB | 合成预测 switch s | 真实预测 switch s | base waves |
|---|---:|---:|---:|---:|---:|
| Fixed / 2->4 | 14.394851328 | 14.394851328 | 12.072918 | 12.243129 | 12 |
| Fixed / 4->2 | 28.789702656 | 28.789702656 | 22.461122 | 22.956686 | 22 |
| Directional / 2->4 | 14.394851328 | 14.394851328 | 9.367836 | 9.448037 | 6 |
| Directional / 4->2 | 28.789702656 | 28.789702656 | 22.461056 | 22.956579 | 22 |
| W2 / 2->4 | 0.315424768 | 0.408943616 | 4.608587 | 4.984612 | 6 |
| W2 / 4->2 | 3.369467904 | 4.959240192 | 13.549426 | 14.627901 | 22 |

## 第 16 项：单 rank wave 负载核对——未满足

**“16 卡单 rank 负载不超过 8 卡”不成立。用户已明确允许保持 W2 与当前 wave 上限，保留该风险后提交。** 全局 FIFO 按 task_id 分波不是按 rank 均分，缩容上限从 2048 翻倍至 4096 后，一些 rank 一波接收量近乎翻倍；不能用 world_size 翻倍推导每卡上界不变。8 卡 W2 缩容候选被原全局 gate 否决、实际为 default；16 卡实际采用 aware 缩容。

以下为四次切换对应方向的逐波、逐 rank 远程发送/接收最大值（发送和接收峰值可来自不同 rank/wave），含跨机部分；同卡复制单列于完整探索 JSON，不计入远程 send/recv。8 卡只有一台机器，跨机分量为零。

| 方向/阶段 | 8 卡发送项/接收项 | 16 卡发送项/接收项 | 8 卡发送/接收 GB | 16 卡发送/接收 GB | 16 卡跨机发送/接收 GB |
|---|---:|---:|---:|---:|---:|
| 2->4 / base | 2664/2741 | 2913/2741 | 4.141032/4.255458 | 4.250477/4.255458 | 0.108003/0.108003 |
| 2->4 / delta | 54/54 | 63/54 | 0.000415/0.000415 | 0.000495/0.000415 | 0.000022/0.000052 |
| 4->2 / base | 1005/2012 | 2006/4020 | 1.628570/3.005350 | 3.130524/6.136560 | 0.329581/0.780403 |
| 4->2 / delta | 108/108 | 129/108 | 0.003041/0.003041 | 0.003638/0.003041 | 0.000000/0.000000 |

完整每 wave 最大任务数、字节数、跨机项数/字节与每 rank 明细保存在本地探索目录 `documents/16gpu_cpu_explore_20261001/outputs/t09_20261003T210744/analysis.json` 的 `rank_wave_loads`；同目录 `wave_report.md` 为逐波表。探索代码/大产物不纳入此 commit；本报告与 `evidence/cpu_analysis_summary.json` 固化摘要及源 hash。

## 缩容峰值显存估算与失败预案

原 benchmark 同时保留 TP2 主模型与 TP4 standby 模型。根据原模型 BF16 参数形状求每卡两套分片；接收缓冲采用保守估法：把最重缩容 base wave 全部远程接收字节都视为新增缓冲。实际 async_execution 对连续切片直接接收到目标参数，非连续切片才分配缓冲，因此该项是接收部分的上界估计，不是 CUDA allocator 总峰值。

| 单卡项 | 十进制 GB |
|---|---:|
| TP2 模型分片 | 8.512592896 |
| TP4 模型分片 | 4.259830784 |
| 缩容峰值接收缓冲预算（rank 10，第四次切换，wave 15，1-based） | 6.136559616 |
| 模型分片 + 接收缓冲 | **18.908983296** |
| 再加两套 1024 容量 KV 后 | **19.121319936**（17.808 GiB） |
| 对比保守按十进制 32 GB | 余量约 12.879 GB |
| 若硬件标示为 32 GiB | 实为 34.360 GB，余量更大；运行时以 nvidia-smi 为准 |

未计入激活/logits、NCCL/CUDA 上下文、发送非连续打包、瞬时 contiguous 临时张量、allocator 预留/碎片及初始化加载峰值；不能保证不 OOM。全局 wave 不均衡仍可能增加 OOM、NCCL 排队或超时风险。

**预案：若 weavetp_r1 OOM 或单 launch 超过 60 分钟，先让现有执行器停止并仅按该 OUT_DIR 标签清理；检查两机任务消失、显存恢复启动前基线 +16 MiB。然后三种方法的缩容上限统一改为 2048，扩容保持原值，在新 ROOT_OUT 批次重新运行全部正式方法。** 不只修改 WeaveTP，不混入当前 4096 批次结果，不自动降上限/自动重跑。该预案是用户指定的后续改版流程，本 commit 仍为 4096，不能仅 export 覆盖 compare 固定配置。

## 启动、断线与人工隔离

必需参数只有 PROFILE、PROFILE_SHA256、ROOT_OUT；env.sh 提供已有 WORK/REPO/解释器/地址。ROOT_OUT 必须是 `$WORK/formal/` 下的批次目录，拒绝路径穿越、symlink、smoke、_failed。两机每 case 前 source env.sh 并检查 HEAD=WEAVETP_COMMIT、clean、画像内容与 T08 的 profile.json.sha256、WARN、9 GB、既有 200 MiB/0%/MPS_OWNER 空闲标准。任一失败立即退出。

先在 SL3060 的终端执行 `tmux new -s t09`。断线后用 `tmux attach -t t09` 回到会话；不启动第二个控制器。以下命令均在 tmux 内运行，stdin 重定向到 /dev/null；先执行第一条，成功后执行第二条。

```bash
source /data/ubuntu/lxh/weavetp/env.sh
export PROFILE=/data/ubuntu/lxh/weavetp/profiles/t08_20261003T123343Z_3644651/measurement/profile.json
export PROFILE_SHA256=ea86ff22f1aab45ab6d306516022354a16c5e529cbcc94f91a6eee209e18449b
export ROOT_OUT="$WORK/formal/t09_w2_20261003"
set -o pipefail
ONLY=weavetp_r1 ROUNDS= bash "$REPO/documents/16gpu_T09_20261003/run_t09.sh" </dev/null
# 仅在上一条成功后；沿用同一 ROOT_OUT，已完成的 weavetp_r1 经核验跳过
ONLY= ROUNDS= bash "$REPO/documents/16gpu_T09_20261003/run_t09.sh" </dev/null
```

`ROUNDS=r1` 只选择原 r1 三个 case；不与 ONLY 同时设置。队列请求/轮号和完成结果与一次跑完等价；先运行 ONLY 必然改变实际日历执行先后，剩余队列按原轮换顺序过滤，不能声称日历顺序相同。完整 dry-run：`bash run_t09.sh --dry-run`，ONLY 同样支持。

每 launch START/END 打印 RUN_ID、耗时、状态；结束打印四次切换墙钟、default/adopted 跨机字节。W2 与 `cpu_reference.json` 四次参考比较，>2 倍或 <1/2 倍只 WARNING、不拒绝。失败或缺失结果输出 summary unavailable，保留原始日志，不编造计时。每轮列出三方法扩/缩均值及 pending。Ctrl+C/SIGTERM 沿用既有执行器 stop_case；仅按 OUT_DIR 标签清理，不停止 MPS 或他人任务。

失败/不完整 case 拒绝自动覆盖或重试。以下是独立人工命令，**run_t09.sh 从不调用它**；必须在交互终端输入非空原因，不要加 </dev/null：

```bash
source /data/ubuntu/lxh/weavetp/env.sh
# 保持上述 PROFILE 和 ROOT_OUT；确认控制器已退出
"$PYTHON" -B -X utf8 "$REPO/documents/16gpu_T09_20261003/isolate_failed.py" weavetp_r1
```

两端检查精确 case 标签进程（含监督进程），取得与控制器互斥的 compare.lock，准备通过后将两端相同 case/rK 移入 `_failed/<case>_r<K>_<UTC>/` 并保存原因。不允许隔离已有 complete 的成功 case。单边失败执行补偿回滚；若断连使回滚无法证实，保留 compare.lock/isolate.lock，报 UNCONFIRMED、禁止续跑，需人工修复连接核对两端后处理。两机无共享文件系统，无法提供跨机器原子 rename；不会把不确定状态称为“两边已隔离”。执行器只枚举固定 case/rK，汇总仅接受显式九文件 manifest 并拒绝 _failed，隔离证据不混入统计。

## 本地验收与变更文件

正式链 CPU/mock **177/177**（T02 28、compare 29、profile 18、server 11、R1 4、observations 25、review 3、T08 17、T08.5 24、T09 18）。探索既有 8+6+6 及新画像审计 5 项通过，合计 **202/202**；历史 54/54 单列。没有 GPU smoke。

命令：`python -B -X utf8 documents/16gpu_T09_20261003/check_cpu.py --output-dir E:/data/weavetp_t09_validation_published`；探索 `test_metadata_adapter.py`、`test_timing_model.py`、`test_w2.py outputs/w2_20261002T220122`、`test_t09_analysis.py`。完整本地原始日志位于 E:/data；提交中保存 validation、历史复核、dry-run 全文与摘要。Python 编译、bash -n、git diff --check、受保护目录 diff 均通过。首次迭代曾发现 fixture 属性/旧冒烟断言不适用，修复后的结果如上；不把早期失败计为通过。

`uv run isort` 已尝试但 uv/isort 均未安装，该工具检查未执行成功；未安装任何依赖。导入顺序人工核对。summarizer 的统计、计时、table2、百分比函数与基线 AST 相同；仅配置验证、_failed 输入排除和 Actual plan 展示变化。T08 目录、benchmark 原始结果、benchmark 源码、megatron/ 无改动。

| 文件 | 原因 |
|---|---|
| compare_weavetp_16gpu.py | WeaveTP 两项 W2 环境、request/complete 取证、两个方向实际几何验证 |
| weavetp_observations.py | 只读几何识别/校验函数，不改采集热路径 |
| summarize_weavetp_formal.py | W2 配置门禁、Actual plan 按几何、排除 _failed；统计逻辑不变 |
| server test_observations.py | 去掉无条件禁止 aware shrink 与基于标签判断 adopted 的旧约束 |
| 两份 unit test 与 T08.5 test/check_cpu | 同步 W2 fixture/argv/基线差分断言；不新增冒烟功能 |
| 本目录 run_t09.sh、t09.py、isolate_failed.py、check_cpu.py、test_t09.py | 操作、预检、续跑、人工隔离与 CPU 验证 |
| 本目录 cpu_reference、evidence、报告、sha256、deploy_t09.sh | 可核验交付及两机既有更新流程 |
| 总计划、分步任务 | 记录用户 W2 修订、未满足项、CPU 准备状态，GPU T09 仍未执行 |

## Fixed / Directional 与基线逐字节对照

同一 dry_environment 下，以 Python `json.dumps(row, indent=2).encode()` 的 request+node_commands 完整项为字节口径；独立 request 与 node_commands hash 也保存在 `evidence/baseline_comparison.json`。六项 old/new 完全相同；三个 WeaveTP 项仅两个环境键变化。

| case | request + node_commands SHA-256（old = new） |
|---|---|
| fixed_r1 | `2cf4c487e591b9924a50906b49041c2ca2fe4f085473ff04eabf84b224deec74` |
| directional_r1 | `d4366f159d705bd4c8c69f35d8e2bc6c4bb051b1ef7c771446bc31eeec1bf700` |
| directional_r2 | `44e090f6a31df485072c49378132e18321bd6dda294192f7d68ae2dfd5b135d5` |
| fixed_r2 | `5abfe72a4d2bda9bae4237a1408b2340d9261cd97fc7f4e2c203817c85a45006` |
| fixed_r3 | `dd83e9a0b6d9e44724fc2ea3bf723a2d8f63a8f4928ea83b3eb4025179443113` |
| directional_r3 | `29468f06667f84bc639ba8927949b21562efd40979b749b27b7e60698a8a45dc` |

完整九 case 与 ONLY/ROUNDS dry-run 分别在 `evidence/dry_run_all.json`、`dry_run_only_weavetp_r1.json`、`dry_run_r1.json`，ROOT_OUT 为 `$WORK/formal/T09_DRY_ONLY`。

## 两机部署命令（尚未执行）

沿用先 SL3061 从 GitHub 更新、再 SL3060 从 SL3061 取得 HEAD 的流程。按交付消息设置完整 TARGET；下列同一段依次在 SL3061、SL3060 执行。不启动 GPU。更新脚本保存在 /data，检出固定 SHA、原位更新唯一 WEAVETP_COMMIT、SHA 校验、CPU 验证。

```bash
set -euo pipefail
source /data/ubuntu/lxh/weavetp/env.sh
TARGET='<交付消息中的完整 SHA>'
test -z "$(git -C "$REPO" status --porcelain --untracked-files=all)"
case "$(hostname -s)" in
  SL3061) git -C "$REPO" fetch origin ;;
  SL3060) git -C "$REPO" fetch "${NODE1_SSH}:$REPO" HEAD
          test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = "$TARGET" ;;
  *) echo "STOP: wrong host" >&2; exit 1 ;;
esac
mkdir -p "$WORK/tmp"
git -C "$REPO" show "$TARGET:documents/16gpu_T09_20261003/deploy_t09.sh" > "$WORK/tmp/deploy_t09_$TARGET.sh"
TARGET="$TARGET" bash "$WORK/tmp/deploy_t09_$TARGET.sh"
```

两机均出现 T09_CODE_READY 后，再只在 SL3060 的 tmux 中使用上面的 ONLY→续跑命令。若 env.sh 的 NCCL_DEBUG 仍为 INFO，应先由操作者明确改为 WARN；脚本不会静默掩盖该配置错误。
