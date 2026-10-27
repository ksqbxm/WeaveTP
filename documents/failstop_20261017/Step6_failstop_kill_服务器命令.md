# 快验（Step 9 fail-slow、Step 6 kill）与 kill 正式运行：服务器命令

交付：实现 A `c3d0dfba74ddde136d05c280c811c5b4eeb1e021`，清单 B `756c43846970d29d0925f724730faa7466ee36b7`（`releases.json["failslow"]` 与 `["step6"]` 都指向 A）。本文件在 B 的子提交 C 里，C 随 bundle `failstop_online_756c438.bundle` 交付（基线 `f7d555998b45756998465924d733fd70e553d6b4`，ref `refs/heads/codex/failstop-online-756c438` 指向 C），bundle 的 SHA256 在同名 `.sha256` 文件里。

本次修正服务器首次走到初始加载时暴露的问题：16 rank 两节点服务的初始 TP4 加载在分片完整性校验处失败（`s9q_20261027T064713_eb030784` 等三次 Step 9 快验）。服务 worker 改用不做集合通信的 `load_local`，详见 [11_failstop在线化依据.md](11_failstop在线化依据.md) §10 与 [10_failslow在线化依据.md](10_failslow在线化依据.md) §8。

作废的交付：`failslow_online_005a4c8`（`Step9_failslow_服务器命令.md` 的 ①②）、`failstop_online_dee4f3b` 及更早的 bundle，两者都还走 `load_checkpoint`；此前用 `073a50f` 跑出的 step6 结果也作废（11 §8）。CPU 回归 307 通过、1 跳过、123 个子测试。**CPU 测试不加载真实检查点，测不到这类问题；以下命令都未在服务器执行过，时长全是估计值。**

## ① 导入 bundle（Windows → SL3061 → SL3060；只 fetch 和校验，不检出主仓库）

Windows：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop-online/failstop_online_756c438.bundle'
Get-FileHash -LiteralPath $Bundle -Algorithm SHA256
Get-Content "$Bundle.sha256"
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

两行 SHA256 应相同。SL3061 上执行：校验，fetch 到本地分支，再转发给 SL3060。

```bash
set -e
cd /data && mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c failstop_online_756c438.bundle.sha256
git -C "$REPO" bundle verify "$PWD/failstop_online_756c438.bundle"
git -C "$REPO" fetch "$PWD/failstop_online_756c438.bundle" refs/heads/codex/failstop-online-756c438:refs/heads/codex/failstop-online-756c438
test "$(git -C "$REPO" rev-parse refs/heads/codex/failstop-online-756c438~1)" = 756c43846970d29d0925f724730faa7466ee36b7
for key in failslow step6; do test "$(git -C "$REPO" show 756c43846970d29d0925f724730faa7466ee36b7:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$key")" = c3d0dfba74ddde136d05c280c811c5b4eeb1e021; done
git -C "$REPO" rev-parse refs/heads/codex/failstop-online-756c438
scp failstop_online_756c438.bundle failstop_online_756c438.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`rev-parse refs/heads/codex/failstop-online-756c438` 打印的是 C（本文件所在提交），请记入回传。SL3060 执行同样的命令，去掉最后一行 `scp`。导入只新增分支 `codex/failstop-online-756c438`，主仓库的检出和工作区不变。

所有快验都直接从清单 B 读出 `quick_check.sh` 运行，不需要先检出。它按 B 的 `releases.json` 找到实现 A，把 A 的 `quick_driver.py` 取到 kickoff 目录运行；`quick_driver` 在 quick 锁内检查已跟踪文件干净、检出 A，结束后（失败退出也一样）切回 `f7d5559`。运行期间主仓库被检出为 A，请避开其他正在使用主仓库工作区的任务。

每次两节点 60 s 内先后启动，SL3060 先。SL3061 的命令只把 `--node SL3060` 改成 `--node SL3061`。同一时间只跑一条。

## ② 只能在服务器上验证的步骤（快验按这个顺序检查）

| # | 步骤 | 由哪次快验覆盖 | 看哪里 |
|---|---|---|---|
| 1 | 初始 TP4 加载：`load_local`，不做集合通信，不做全局分片校验 | 每一次；逐位证明靠 ④ | 每次都看 worker 日志里没有 `CheckpointingException`。④ 是 fail-stop weavetp 的 correctness 运行，每个 `serving_checks_<slot>.json` 的第一个事件为 `load`，`weights` 带 `parameters`，即与参考 load 逐位一致（不一致时 worker 直接报错退出），以此证明 `load_local` 读出的字节与原来的 `load_checkpoint` 相同。Step 9 是 fail-slow，按 10-21 的决定不做参考 hash 比对，`weights` 为 skipped |
| 2 | TP2 预分配：开服前建 TP2 模型并释放存储，计划缩容 | Step 9 weavetp（`online_ready`）、Step 6 kill（`kill_ready`） | `serving_checks_<slot>.json` 的 `online_ready` / `kill_ready` 事件；没有显存不足 |
| 3 | 组预热：`warm_groups` 遍历每个预建通信器，`warm_pairs` 遍历计划用到的 P2P 对 | 同 2 | 控制门进入服务（phase-0 报告齐全）；事件里 `p2p_pairs_warmed > 0`；`nccl_*.log` 中没有 error |
| 4 | kill 回收：victim 被 SIGKILL，存活进程 abort 含死卡的组，GPU 回收 | Step 6 kill | `events_actual.delays.gpu_reclaim_s` 每卡有值，`dead_group_abort_s` 有值，SL3061 没有 `GPU memory/process reclamation did not become stable` |
| 5 | phase-2 重建：在 Gloo 默认 world 上 abort，重建 16 rank NCCL world，扩容迁移 | Step 6 kill | 结论行 PASS（phase 2 的 validate 通过），`abort_s["2"]`、`rebuild_s["2"]`、`edp1_pause_s` 有值 |

按 ③ 的三条 Step 9 依次跑，再跑 ④ 的 Step 6 kill。前一条没通过时先回传，不再往下跑。

## ③ 快验：quick_check --step 9 --method（weavetp → ignore → restart）

每条命令只跑该方法一次 fail-slow W 级 launch：
- 共享 GPU，correctness 模式；
- SL3061 的 GPU 0/1/4/5 每卡 1 个 `failslow_hog.py`；
- warmup 20 s、D 30 s、tail 20 s、检测 10 s；
- toy 请求池 2000 条，到截止时还有请求未完成。

每条约 5 分钟。

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 756c43846970d29d0925f724730faa7466ee36b7:code/current/tools/resharding/failstop/quick_check.sh) --manifest 756c43846970d29d0925f724730faa7466ee36b7 --step 9 --method weavetp --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 756c43846970d29d0925f724730faa7466ee36b7:code/current/tools/resharding/failstop/quick_check.sh) --manifest 756c43846970d29d0925f724730faa7466ee36b7 --step 9 --method ignore --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 756c43846970d29d0925f724730faa7466ee36b7:code/current/tools/resharding/failstop/quick_check.sh) --manifest 756c43846970d29d0925f724730faa7466ee36b7 --step 9 --method restart --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

`nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step09_failslow/kickoff/<NODE>_*/`）的末行应为 `QUICK CHECK step=9 method=<方法> result=PASS ...`。只检查三件事：
- launch 跑完且 passed，本节点没有残留的 `failslow_hog.py`；
- SL3060 上事件顺序正确；
- 截止时确有未完成的请求。

另外按 ② 检查第 1 步（三条都看），以及第 2、3 步（weavetp）。

SL3060 在结论行之前打印一行 `STEP 9 SUMMARY <方法>: {...}`，同样写在 `<run>/SL3060/summary.json` 的 `launches.<方法>.gate.summary`。预期：
- weavetp：`replayed_tokens` 为 0，EDP0 与 EDP1 的缩容 decode 步数都大于 0；
- restart：两个重新加载耗时都有值（重新加载同样走 `load_local`），`replayed_tokens` 大于 0；
- ignore：切换类字段全部为"缺失"。

字段说明见 [Step9_failslow_服务器命令.md](Step9_failslow_服务器命令.md) ②。

## ④ 快验：quick_check --step 6 --layer b3 --mode kill（≤ 300 s）

24 条 toy 请求，correctness 模式，共享 GPU：
- **缩容一次**：真实 SIGKILL 4 个 victim；存活方在预建组上迁移权重；EDP1 在各波之间 decode；EDP0 的在途请求重放；降级后 abort 含死卡的旧组。
- **扩容一次**：abort Gloo 默认 world，重建 16 rank 的 NCCL world，迁移 KV 前缀加增量并逐位检查。

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 756c43846970d29d0925f724730faa7466ee36b7:code/current/tools/resharding/failstop/quick_check.sh) --manifest 756c43846970d29d0925f724730faa7466ee36b7 --step 6 --layer b3 --method weavetp --mode kill --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

`nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step06_batched_serving/kickoff/<NODE>_*/`）里 SL3060 的结论行：

```text
QUICK CHECK step=6 layer=b3 method=weavetp result=PASS ({"mode": "kill", "method": "weavetp", "metrics": {"requests_completed": 24, "check_passed": true, "kv_check_passed": true, "edp1_rejoin_pause_s": <数>, …}, "files": {"result.json": "…", "check.json": "…", "events_actual.json": "…"}}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=false accepted=false logs=…
```

**脚本判定 PASS 的条件**（同原 b3）是控制门 passed：
- 24 条请求全部完成，`check.json` 通过，扩容 KV 逐位检查通过，参考 hash 一致；
- 控制门还核对了 kill_exited 的退出码为 −SIGKILL、替换进程 PID 已变且为空接收端、rank map 一致。

SL3061 的结论行同样是 `result=PASS`，`metrics` 为 null。toy 请求会在 rejoin 前排空，`stats_valid=false` 属预期。

**kill 另加的判据**（② 的第 4、5 步，以及 EDP1 在降级期间 decode）：在 SL3060 上从结论行 `files` 取出 `events_actual.json` 与 `result.json` 的路径，执行下面的命令。它只读文件，不检出：

```bash
EV=<结论行 files 里 events_actual.json 的路径>; RS=<同一目录的 result.json>
"$PYTHON" -X utf8 -c 'import json,sys; d=json.load(open(sys.argv[1]))["delays"]; r=json.load(open(sys.argv[2])); s=[x["overlap_steps"] for x in r["switches"] if x["domain"]=="EDP1" and x["direction"]=="shrink"]; print(json.dumps({"R1_abort_s_2": d["abort_s"]["2"], "R1_rebuild_s_2": d["rebuild_s"]["2"], "R2_gpu_reclaim_s": d["gpu_reclaim_s"], "dead_group_abort_s": d["dead_group_abort_s"], "edp1_pause_s": d["edp1_pause_s"], "EDP1_shrink_overlap_steps": s}, indent=1))' "$EV" "$RS"
```

- **R1**：`R1_abort_s_2`、`R1_rebuild_s_2` 都有值。
- **R2**：`R2_gpu_reclaim_s` 每张 victim 卡都有值，`dead_group_abort_s` 有值。
- **EDP1**：`EDP1_shrink_overlap_steps` 的值 > 0。

失败时先看 `<run>/<node>/error.txt`、`summary.json` 与各 worker 的 `serving_checks_*.json`。

## ⑤ kill 正式矩阵（timing，独占 16 卡；b-max 8；每格 3 次独立 launch）

前提：③④ 两节点都 PASS，② 的五步都满足。请求集与标定沿用 [Step6_AnchorTP接入_服务器命令.md](Step6_AnchorTP接入_服务器命令.md) 的 ②③：
- 生成工具 `tools.resharding.e2e.make_failstop_requests`，标定目标为无故障服务约 108 s。
- 若此前已标定并冻结请求集，可以继续用（请求集与实现无关）。
- 否则先做 ②③，并把其中的 `git -C "$REPO" checkout --detach 560361c…` 换成 `git -C "$REPO" checkout --detach 756c43846970d29d0925f724730faa7466ee36b7`。

此前的正式结果作废，需按下表重跑。

| 格 | `--method` | `--scenario` | `--mode` | `--plan` | launch 数 |
|---|---|---|---|---|---|
| k1 | weavetp | failstop | **kill** | plan_failstop.json | 3 |
| k2 | weavetp | failstop | real | plan_failstop.json | 3 |
| k3 | anchortp | failstop | real | plan_failstop.json | 3 |
| k4 | weavetp（方法无关） | nofault | emulate | plan_nofault.json | 3 |
| 可选 k5–k7 | weavetp / anchortp / restart | failstop | emulate | plan_failstop.json | 各 3 |

- b-max 只取 8。若显存不足，所有格一起降到同一个值，并重新标定请求数。
- 每多一个 b-max 取值，矩阵加倍。

**时间线**（请求集的 plan 参数）：

| 段 | 参数 | 取值 | 说明 |
|---|---|---|---|
| 故障时刻 | `--fail-fraction` | 0.25 | 完成 25% 请求时发起故障（标定 108 s 时约 t≈27 s） |
| rejoin 延迟 | `--rejoin-seconds` | 45（可压到 30） | 从故障到 rejoin 到期；回收与之重叠 |
| 尾段 | `--count` | 标定值 N | 扩容后把剩余请求排空，约 30 s |

**单个 launch 的预计墙钟**（kill，独占模式；均为估计）：

| 阶段 | 估计 | 依据 |
|---|---|---|
| 时钟校准、预检、基线采样、拓扑 | ~10 s | |
| 启动 worker、CUDA/MPS 核实 | ~15 s | |
| Megatron 初始化 + TP4 加载（`load_local`） | ~55 s | Step 5 按实测加载速度估计：6 次加载约 330 s |
| kill 开服前准备（建 TP2 不加载、计划、预热） | ~10 s | |
| 服务：故障前 → 降级 → rejoin 到期 | ~27 + 45 s | |
| rejoin：回收、替换进程启动、重建 world、扩容迁移 | ~30–40 s | 迁移每次约 2–4 s，按 T09 切换模型估计 |
| 尾段排空 | ~30 s | |
| 结束、回收确认、打包 | ~15–30 s | |
| **合计** | **约 4–4.5 分钟** | |

- 目标是单个 launch ≤ 7 分钟，按上面的估计可以做到。
- 若实测超过 7 分钟，瓶颈应在开头的 TP4 加载和 rejoin 时替换进程的启动，这两段不能靠参数压缩。可以把 `--rejoin-seconds` 压到 30，并按 90 s 重新标定 `--count`。
- 实测时长看 kickoff 的 `nohup.log` 时间戳与 `events_actual.json`。

**总时长**：主矩阵 12 次 launch × 约 4.5 分钟 ≈ 55 分钟，加上组间预检约 1 小时；可选的 9 次另需约 45 分钟。

**命令**：每格两节点各一条，`--suite-id` 相同，60 s 内先后启动，SL3060 先。一格出现 `FAILSTOP E2E … result=PASS` 后再启动下一格。
- `run_failstop_e2e.sh` 从当前检出读 `releases.json`，所以先检出清单 B，脚本会再检出 A。
- 运行期间主仓库停在 A，请避开其他使用主仓库工作区的任务（包括切换版本的守护）。

SL3060 的命令如下（SL3061 只改 `--node`）：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 756c43846970d29d0925f724730faa7466ee36b7
REQ=/data/ubuntu/lxh/weavetp/failstop_requests/N<标定的 N>
PROFILE=/data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
OUT=/data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance
export MPS_OWNER=<SL3060 常驻 MPS 的属主>   # 两节点都设，见 Step6_AnchorTP 命令 ③
RUN="bash $REPO/code/current/tools/resharding/failstop/run_failstop_e2e.sh --node SL3060 --output $OUT --run-mode timing --launches 3 --requests $REQ/requests.jsonl --workload $REQ/workload.json --profile-file $PROFILE --seq-length 2048 --b-max 8 --budget-seconds 3600 --min-free-disk-gb 1"
$RUN --suite-id k1 --method weavetp  --scenario failstop --mode kill    --plan "$REQ/plan_failstop.json"
$RUN --suite-id k2 --method weavetp  --scenario failstop --mode real    --plan "$REQ/plan_failstop.json"
$RUN --suite-id k3 --method anchortp --scenario failstop --mode real    --plan "$REQ/plan_failstop.json"
$RUN --suite-id k4 --method weavetp  --scenario nofault  --mode emulate --plan "$REQ/plan_nofault.json"
# 可选：k5 weavetp / k6 anchortp / k7 restart，均为 --scenario failstop --mode emulate
```

（每行是一格。上一格结束后才粘贴下一行。）
- `--budget-seconds 3600` 覆盖一格的 3 次 launch。
- `--min-free-disk-gb 1` 是因为 SL3060 的 `/data` 空间紧张。
- 进度：`tail -n 40 $(ls -td $OUT/kickoff_<NODE>_* | head -1)/nohup.log`。
- 全部结束后执行 `git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4`。

**k1 记录恢复重建停顿的结果字段**（见 11 §4）：
- `events_actual.json` 的 `delays`：
  - `edp1_pause_s`；
  - `abort_s["2"]`、`rebuild_s["2"]`；
  - `rejoin_join_s`；
  - `gpu_reclaim_s`；
  - `reclaim_wait_s`；
  - `dead_group_abort_s`（kill 专有）。
- `result.json`：
  - `rejoin_pauses`；
  - `metrics.edp1_rejoin_pause_s`；
  - `tfs_breakdown.expand_wall_s`。
- kill 的 `abort_s["1"]`、`rebuild_s["1"]`、`tfs_breakdown.world_rebuild_s` 只是总线往返时间。
- 分析沿用 `analyze_failstop_e2e.py`，kill 在 `summary.csv` 里单独成组（`mode=kill`）。

fail-slow 的正式压缩时间线不变，见 [Step9_failslow_服务器命令.md](Step9_failslow_服务器命令.md) ③；它不含 SHA，S 取 ③ restart 快验的 `restart_reload_s.tp2`。

## ⑥ 回传

请回传：
- ③④ 每次快验两节点的 `<suite>_<NODE>.tar.gz` 与 `<suite>_<NODE>.receipt.json`（在 `acceptance/step09_failslow/`、`acceptance/step06_batched_serving/` 下），以及 ④ 判据命令的输出；
- ⑤ 每格的 `matrix_manifest_*.json`、`progress_*.json`，每次 launch 的 `result.json`、`events_actual.json` 与 kickoff 的 `nohup.log`。
