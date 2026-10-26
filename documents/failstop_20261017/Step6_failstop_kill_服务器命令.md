# fail-stop WeaveTP `--mode kill`：快验与正式运行（服务器命令）

交付：实现 A `c4b203e0d059700088f4b23fc84b9ab88605e935`，清单 B `dee4f3b91dff0923764aaa78232fe5cae744e247`（`releases.json["step6"]` 指向 A，`failslow` 仍为 `d8b8cd7ef1d59926e3e294a2e6d905e115cdfdfb`）。本文件在 B 的子提交 C 里，C 随 bundle `failstop_online_dee4f3b.bundle` 交付（基线 `f7d555998b45756998465924d733fd70e553d6b4`，ref `refs/heads/codex/failstop-online-dee4f3b` 指向 C），bundle 的 SHA256 在同名 `.sha256` 文件里。设计与字段见 [11_failstop在线化依据.md](11_failstop在线化依据.md)。CPU 回归 306 通过、1 跳过、123 个子测试；**以下命令都未在服务器执行过，下文的时长全是估计值。**

此前的 step6 实现 `073a50f` 带有两个回放历史问题，用它跑出的 step6 结果作废（11 §8）。首版交付（实现 `6f2fc84`、清单 `c71d737`、bundle `failstop_online_c71d737`）经审阅后作废（11 §9），请改用本 bundle。顺序：① 导入 → ② 快验（kill 的 b3）→ ③ 正式矩阵 → ④ 回传。

## ① 导入 bundle（Windows → SL3061 → SL3060；只 fetch 和校验，不检出主仓库）

Windows：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop-online/failstop_online_dee4f3b.bundle'
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
sha256sum -c failstop_online_dee4f3b.bundle.sha256
git -C "$REPO" bundle verify "$PWD/failstop_online_dee4f3b.bundle"
git -C "$REPO" fetch "$PWD/failstop_online_dee4f3b.bundle" refs/heads/codex/failstop-online-dee4f3b:refs/heads/codex/failstop-online-dee4f3b
test "$(git -C "$REPO" rev-parse refs/heads/codex/failstop-online-dee4f3b~1)" = dee4f3b91dff0923764aaa78232fe5cae744e247
test "$(git -C "$REPO" show dee4f3b91dff0923764aaa78232fe5cae744e247:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["step6"])')" = c4b203e0d059700088f4b23fc84b9ab88605e935
test "$(git -C "$REPO" show dee4f3b91dff0923764aaa78232fe5cae744e247:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["failslow"])')" = d8b8cd7ef1d59926e3e294a2e6d905e115cdfdfb
git -C "$REPO" rev-parse refs/heads/codex/failstop-online-dee4f3b
scp failstop_online_dee4f3b.bundle failstop_online_dee4f3b.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`rev-parse refs/heads/codex/failstop-online-dee4f3b` 打印的是 C（本文件所在提交），请记入回传。SL3060 执行同样的命令，去掉最后一行 `scp`。导入只新增分支 `codex/failstop-online-dee4f3b`，主仓库的检出和工作区不变。

## ② 快验：quick_check --step 6 --layer b3 --mode kill（每节点一条命令，≤ 300 s）

- 24 条 toy 请求，correctness 模式，共享 GPU。
- 一次缩容：真实 SIGKILL 4 个 victim，存活方在预建组上迁移权重，EDP1 在各波之间 decode，EDP0 在途请求重放，降级后 abort 含死卡的旧组。
- 一次扩容：abort Gloo 默认 world，重建 16 rank 的 NCCL world，迁移 KV 前缀加增量，逐位检查。
- `quick_check.sh` 直接从清单 B 读出运行：它按 B 的 `releases.json` 找到实现 A，把 A 的 `quick_driver.py` 取到 kickoff 目录运行。`quick_driver` 在 quick 锁内检查已跟踪文件干净、检出 A，结束后（失败退出也一样）切回 `f7d5559`。运行期间主仓库被检出为 A，请避开其他正在使用主仓库工作区的任务。
- 两节点 60 s 内先后启动，SL3060 先。SL3061 的命令只把 `--node SL3060` 改成 `--node SL3061`。

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show dee4f3b91dff0923764aaa78232fe5cae744e247:code/current/tools/resharding/failstop/quick_check.sh) --manifest dee4f3b91dff0923764aaa78232fe5cae744e247 --step 6 --layer b3 --method weavetp --mode kill --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

`nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step06_batched_serving/kickoff/<NODE>_*/`）中 SL3060 的结论行：

```text
QUICK CHECK step=6 layer=b3 method=weavetp result=PASS ({"mode": "kill", "method": "weavetp", "metrics": {"requests_completed": 24, "check_passed": true, "kv_check_passed": true, "edp1_rejoin_pause_s": <数>, …}, "files": {"result.json": "…", "check.json": "…", "events_actual.json": "…"}}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=false accepted=false logs=…
```

**脚本判定 PASS 的条件**（同原 b3）：控制门 passed，即 24 条请求全部完成、`check.json` 通过、扩容 KV 逐位检查通过、参考 hash 一致；控制门同时核对了 kill_exited 的退出码为 −SIGKILL、替换进程 PID 已变且为空接收端、rank map 一致。SL3061 的结论行同样是 `result=PASS`，`metrics` 为 null。toy 请求会在 rejoin 前排空，`stats_valid=false` 属预期。

**kill 另加的三条判据**：在 SL3060 上从结论行 `files` 取 `events_actual.json` 与 `result.json` 的路径，执行下面的命令。它只读文件，不检出：

```bash
EV=<结论行 files 里 events_actual.json 的路径>; RS=<同一目录的 result.json>
"$PYTHON" -X utf8 -c 'import json,sys; d=json.load(open(sys.argv[1]))["delays"]; r=json.load(open(sys.argv[2])); s=[x["overlap_steps"] for x in r["switches"] if x["domain"]=="EDP1" and x["direction"]=="shrink"]; print(json.dumps({"R1_abort_s_2": d["abort_s"]["2"], "R1_rebuild_s_2": d["rebuild_s"]["2"], "R2_gpu_reclaim_s": d["gpu_reclaim_s"], "dead_group_abort_s": d["dead_group_abort_s"], "edp1_pause_s": d["edp1_pause_s"], "EDP1_shrink_overlap_steps": s}, indent=1))' "$EV" "$RS"
```

- **R1**：phase 2 在 Gloo 默认 world 上 abort 后重建成功。结论行为 PASS，说明 phase 2 的 validate 通过；`R1_abort_s_2`、`R1_rebuild_s_2` 都有值。
- **R2**：`R2_gpu_reclaim_s` 每张 victim 卡都有值；SL3061 的日志里没有 `GPU memory/process reclamation did not become stable`；`dead_group_abort_s` 有值。
- **EDP1 降级期间 decode**：`EDP1_shrink_overlap_steps` 的值 > 0。

失败时先看 `<run>/<node>/error.txt`、`summary.json` 与各 worker 的 `serving_checks_*.json`。若报 `Group … is not registered`，就是 11 §7 R3 所说的模型并行组元数据查询，请回传完整的 traceback。

## ③ 正式矩阵（timing，独占 16 卡；b-max 8；每格 3 次独立 launch）

前提：② 两节点都 PASS，且 R1、R2、EDP1 三条判据都满足。请求集与标定沿用 [Step6_AnchorTP接入_服务器命令.md](Step6_AnchorTP接入_服务器命令.md) 的 ②③。生成工具 `tools.resharding.e2e.make_failstop_requests`，标定目标为无故障服务约 108 s。若此前已按那份文档的 ③ 标定并冻结请求集，可以继续用（请求集与实现无关）；否则先做 ②③，把其中的 `git -C "$REPO" checkout --detach 560361c…` 换成 `git -C "$REPO" checkout --detach dee4f3b91dff0923764aaa78232fe5cae744e247`。此前的正式结果作废，需按下表重跑。

**矩阵**（b-max 8；N=3；顺序执行，不并行）：

| 格 | `--method` | `--scenario` | `--mode` | `--plan` | launch 数 |
|---|---|---|---|---|---|
| k1 | weavetp | failstop | **kill** | plan_failstop.json | 3 |
| k2 | weavetp | failstop | real | plan_failstop.json | 3 |
| k3 | anchortp | failstop | real | plan_failstop.json | 3 |
| k4 | weavetp（方法无关） | nofault | emulate | plan_nofault.json | 3 |
| 可选 k5–k7 | weavetp / anchortp / restart | failstop | emulate | plan_failstop.json | 各 3 |

- b-max 只取 8，即首次上机参数。若 admission 或预检报显存不足，所有格一起降到同一值，并在报告中注明。换 b-max 要重新标定请求数：吞吐变了，108 s 对应的条数也变。
- 每多一个 b-max 取值，矩阵加倍。

**时间线**：参数在请求集的 plan 里，`make_failstop_requests` 的 `--fail-fraction`、`--rejoin-seconds`、`--count`。

| 段 | 参数 | 取值 | 说明 |
|---|---|---|---|
| 故障时刻 | `--fail-fraction` | 0.25 | 完成 25% 请求时发起故障；标定 108 s 时约在 t≈27 s |
| rejoin 延迟 | `--rejoin-seconds` | 45（可压到 30） | 从故障到 rejoin 到期；回收与它重叠 |
| 尾段 | `--count` | 标定值 N | 扩容后排空剩余请求；按 108 s 标定估约 30 s |

**单个 launch 的预计墙钟**（kill，独占模式；均为估计）：

| 阶段 | 估计 | 依据 |
|---|---|---|
| 时钟校准、预检、基线采样、拓扑 | ~10 s | 3 次基线采样，每次 0.1 s；外加总线往返 |
| 启动 worker、CUDA/MPS 核实 | ~15 s | |
| Megatron 初始化 + TP4 检查点加载 | ~55 s | Step 5 按实测加载速度估计：6 次加载约 330 s |
| kill 开服前准备（建 TP2 不加载、计划、预热） | ~10 s | |
| 服务：故障前 → 降级 → rejoin 到期 | ~27 + 45 s | |
| rejoin：回收、替换进程启动（Megatron 初始化，不加载）、重建 world、扩容迁移 | ~30–40 s | 迁移每次约 2–4 s，按 T09 切换模型 |
| 尾段排空 | ~30 s | |
| 结束、回收确认、打包 | ~15–30 s | |
| **合计** | **约 4–4.5 分钟** | |

- 目标是单个 launch ≤ 7 分钟。按上面的估计可以做到，且不必压缩时间线。
- 若实测超过 7 分钟，瓶颈应在两个不能靠参数压缩的阶段：开头的 TP4 检查点加载，以及 rejoin 时替换进程的启动。这时把 `--rejoin-seconds` 压到 30、按 90 s 重新标定 `--count`，可以再省约 30 s。
- 实测时长看 kickoff 的 `nohup.log` 时间戳与 `events_actual.json`。

**总时长**：主矩阵 12 次 launch × 约 4.5 分钟 ≈ 55 分钟；加上组间预检与等待对端，约 1 小时。可选的 9 次 launch 另需约 45 分钟。

**命令**：每格两节点各一条，`--suite-id` 相同，60 s 内先后启动，SL3060 先；一格出现 `FAILSTOP E2E … result=PASS` 后再启动下一格。

- `run_failstop_e2e.sh` 从当前检出读 `releases.json`，所以先检出清单 B；它会再检出 `releases.json["step6"]`，即 A。
- 运行期间主仓库停在 A，请避开其他使用主仓库工作区的任务（包括切换版本的守护）。
- 全部结束后切回 `f7d5559`。

SL3060 的命令如下，SL3061 只改 `--node`：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach dee4f3b91dff0923764aaa78232fe5cae744e247
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

- `--budget-seconds 3600` 覆盖一格的 3 次 launch，按估计每次约 4.5 分钟，余量留给 real 格在 phase 1 的重建。
- `--min-free-disk-gb 1` 是因为 SL3060 的 `/data` 空间紧张。
- 进度：`tail -n 40 $(ls -td $OUT/kickoff_<NODE>_* | head -1)/nohup.log`。
- 全部结束后：`git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4`。

**k1 记录恢复重建停顿的结果字段**（每次 launch 的 `events_actual.json` / `result.json`，见 11 §4）：

- `events_actual.json` 的 `delays`：
  - `edp1_pause_s`：EDP1 在阶段 2 从 abort 到采纳重建组的停顿；
  - `abort_s["2"]`、`rebuild_s["2"]`；
  - `rejoin_join_s`：从 rejoin 请求到 world16 就绪；
  - `gpu_reclaim_s`：每卡从 kill 到回收确认；
  - `reclaim_wait_s`；
  - `dead_group_abort_s`：降级后 abort 死组的耗时，kill 专有。
- `result.json` 的 `rejoin_pauses`、`metrics.edp1_rejoin_pause_s`，以及 `tfs_breakdown.expand_wall_s`（含 worker 的 NCCL 重建）。
- kill 的 `abort_s["1"]`、`rebuild_s["1"]`、`tfs_breakdown.world_rebuild_s` 只是总线往返，不含 abort 或重建。
- 分析沿用 `analyze_failstop_e2e.py`（Step6_AnchorTP 命令 ⑥），kill 在 `summary.csv` 里单独成组（`mode=kill`）。

## ④ 回传

请回传以下内容：
- ② 两节点的 `<suite>_<NODE>.tar.gz` 与 `<suite>_<NODE>.receipt.json`（在 `acceptance/step06_batched_serving/` 下），以及三条 kill 判据命令的输出。
- ③ 每格的 `matrix_manifest_*.json`、`progress_*.json`，每次 launch 的 `result.json`、`events_actual.json`，以及 kickoff 的 `nohup.log`（含实测时长）。
- 分析目录 `analysis_*`。
