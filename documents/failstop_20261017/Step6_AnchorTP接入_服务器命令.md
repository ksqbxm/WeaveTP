# AnchorTP 接入与正式 fail-stop 运行：服务器命令（按顺序）

> 2026-10-21：实现 `560361c` 有扩容在途请求崩溃与结果文件覆盖两处缺陷（[08 §9](08_AnchorTP最小接入.md)）。先按 [Step6_inflight快验_服务器命令.md](Step6_inflight快验_服务器命令.md) 导入 `failstop_inflight_568c319.bundle`；本文件 ②–⑦ 的命令里 `8f8856d…` 换成清单 `568c319c088181723a23fd1d554f0beea87a02f0`、`560361c…` 换成实现 `c4df30a8a2bc2b9435cffde17cb2463be7eea113`，其余不变（③ 读的 `result.json` 的 `serving_wall_s` 现在就是控制门的值）。

交付：实现 `16b1617af90eb4ada0bd32d715dcc45ccd5ca91b` + 启动脚本目录修正 `f2a85e9074e11c98605f72ce25f53c29e2f5fa7b` + 独占预检 MPS_OWNER 例外 `560361c60c6c1ad503b9afda14cd984e011dbd2a`，快验清单 `8f8856d4dbe8f61313180d14b7aa94c8347bb4f6`（`releases.json["step6"]`），bundle `anchortp_8f8856d.bundle`（1368708 字节，SHA256 `1ba26ce896e8c70ac00b96cba4259bd3cee11c061f7dadf80a60929ebd0a12e3`，基线 `f7d555998b45756998465924d733fd70e553d6b4`，含 Step 1–6 全部提交与本次的日志修正、实现、两处修正、钉定等提交）。设计与字段见 [08_AnchorTP最小接入.md](08_AnchorTP最小接入.md)。此前 Step 6 清单 `8a1e576` 的 b3 命令作废；B1/B2 层命令不变（改用本清单 SHA 即可）。

全部步骤：① 导入 → ② 生成请求集 → ③ 标定（无故障试跑，定条数） → ④ b3 快验（两种方法） → ⑤ 正式矩阵 → ⑥ 分析 → ⑦ 打包回传、切回 `f7d5559`。CPU 侧 248 项测试通过；服务器 GPU 全部未运行，以下命令未在服务器执行过。

## ① 导入 bundle

Windows 上传：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/anchortp_8f8856d.bundle'
Get-FileHash -LiteralPath $Bundle -Algorithm SHA256
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

SL3061：校验、导入、转发 SL3060：

```bash
set -e
cd /data && mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c anchortp_8f8856d.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/anchortp_8f8856d.bundle"
git -C "$REPO" fetch "$PWD/anchortp_8f8856d.bundle" refs/heads/codex/anchortp-8f8856d
test "$(git -C "$REPO" rev-parse FETCH_HEAD)" = 8f8856d4dbe8f61313180d14b7aa94c8347bb4f6
git -C "$REPO" checkout --detach 8f8856d4dbe8f61313180d14b7aa94c8347bb4f6
scp anchortp_8f8856d.bundle anchortp_8f8856d.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

SL3060：同上去掉最后一行 `scp`。两节点都检出 `21704bb` 后，`quick_check.sh`/`run_failstop_e2e.sh` 内部再按 `releases.json["step6"]` 检出实现 `560361c`。

## ② 生成请求集（一次，放到两节点同一路径）

在持有 ShareGPT 文件与 HF tokenizer 的节点执行（第 1 行计划里是 SL3060：`/data/ubuntu/lxh/weavetp/e2e_prep/data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json`、`/data/models/DeepSeek-V2-Lite-hf`；若路径不同用 `--sharegpt`/`--hf` 指定）。初始条数 N0=600：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 560361c60c6c1ad503b9afda14cd984e011dbd2a && cd "$REPO/code/current" && "$PYTHON" -B -m tools.resharding.e2e.make_failstop_requests --out /data/ubuntu/lxh/weavetp/failstop_requests/N600 --count 600 --rejoin-seconds 45
```

输出 `requests.jsonl`、`workload.json`、`plan_failstop.json`（25% 完成时 EDP0 四坐标故障、45 s 后 rejoin）、`plan_nofault.json`、`manifest.json`。复制到另一节点同一路径并核对：

```bash
scp -r /data/ubuntu/lxh/weavetp/failstop_requests/N600 ubuntu@<对端>:/data/ubuntu/lxh/weavetp/failstop_requests/
sha256sum /data/ubuntu/lxh/weavetp/failstop_requests/N600/*; ssh ubuntu@<对端> sha256sum /data/ubuntu/lxh/weavetp/failstop_requests/N600/\*
```

前置：冻结的 Step 3 profile 两节点同一路径（B3 已用 `/data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json`，不变）；Step 5 参考 hash 根（SL3061 `…/s5q_20261017T165421_80b77c92_single/SL3061`）只有 correctness 运行（快验）需要。

## ③ 标定：无故障 timing 试跑，定条数使总时长约 108 s

timing 运行启动前检查本节点 8 卡空闲（无外来进程、util 0），不满足直接拒绝；两节点都要空闲。SL3060 的常驻 MPS server 是唯一允许的例外：两节点在每条 timing 命令前先 `export MPS_OWNER=<SL3060 常驻 nvidia-cuda-mps-server 的属主用户名>`（`ps -o user= -p 1641043` 可查），属主匹配、该卡 ≤ 200 MiB 且无 MPS 客户端即放行，记入 `<launch>_preflight_exclusive.json` 的 `mps_owner_exception`。两节点各一条，60 s 内先后启动（`--suite-id` 必须相同）：

```bash
REQ=/data/ubuntu/lxh/weavetp/failstop_requests/N600
PROFILE=/data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
bash "$REPO/code/current/tools/resharding/failstop/run_failstop_e2e.sh" --node SL3060 --suite-id cal01 --output /data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance --method weavetp --scenario nofault --mode emulate --run-mode timing --launches 1 --requests "$REQ/requests.jsonl" --workload "$REQ/workload.json" --plan "$REQ/plan_nofault.json" --profile-file "$PROFILE" --seq-length 2048 --b-max 8 --budget-seconds 1800
```

SL3061 只改 `--node SL3061`。脚本打印日志路径；结束行 `FAILSTOP E2E weavetp nofault emulate timing result=PASS launches=1/1 logs=…`。读 SL3060 的 `…/SL3060/cal01_weavetp_nofault_emulate_00/result.json` 的 `serving_wall_s`（记 W0），`N1 = round(600 × 108 / W0)`，用 `--count N1` 重新生成到 `failstop_requests/N<N1>`（新目录，不覆盖），复制到对端，再以 `--suite-id cal02` 跑一次确认 |W1 − 108| ≤ 10 s；不满足再迭代一次。确认后冻结：记录 `manifest.json` 的 `sha256` 与 N，后续全部命令用这一套文件。标定 launch 不计入正式无故障组。

显存：`--b-max 8`、`--seq-length 2048` 为首次上机参数；若 launch 因 `admission`/预检报显存不足，降 `--b-max`（两方法、所有组保持同一值）并在报告注明。

## ④ b3 快验（两种方法各一次；correctness，共享 GPU 允许，≤ 300 s）

与原 B3 相同的两节点命令，只多 `--method`。先 weavetp，再 anchortp；每次两节点 60 s 内先后启动：

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 8f8856d4dbe8f61313180d14b7aa94c8347bb4f6 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 6 --layer b3 --method weavetp --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

SL3061 只改 `--node SL3061`；第二次把 `--method weavetp` 改为 `--method anchortp`。通过标准不变（控制门 passed、`check.json`、扩容 KV 逐位检查、全部请求完成），anchortp 另要求 EDP0 两次切换 `overlap_steps=0` 且 delta 字节 0。结论行（SL3060）：

```text
QUICK CHECK step=6 layer=b3 method=anchortp result=PASS ({"mode": "emulate", "method": "anchortp", "metrics": {"requests_completed": 24, "check_passed": true, "kv_check_passed": true, "edp0_switches": [{"direction": "shrink", "overlap_steps": 0, "kv_delta_bytes": 0, …}, {"direction": "expand", "overlap_steps": 0, "kv_delta_bytes": 0, …}], …}, "files": {…}}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=false accepted=false logs=…
```

weavetp 的结论行 `overlap_steps` 为 0/2。24 条玩具请求会在 rejoin 前排空，`stats_valid=false` 属预期。结果目录同 B3，另多 `source_plan_0_*.json` 里的 `method`/`planner` 字段。

## ⑤ 正式矩阵（timing，独占 16 卡；N=3）

六组顺序执行，不并行；每组两节点各一条命令（同 `--suite-id`，60 s 内先后启动），一组的 `FAILSTOP E2E … result=PASS` 出现后再启动下一组。`REQ` 指向冻结的 `failstop_requests/N<N1>`。SL3060 示例（SL3061 只改 `--node`）：

```bash
REQ=/data/ubuntu/lxh/weavetp/failstop_requests/N<N1>
PROFILE=/data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
OUT=/data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance
export MPS_OWNER=<SL3060 常驻 MPS 的属主>   # 两节点都设；SL3061 没有常驻 MPS 时也无害
RUN="bash $REPO/code/current/tools/resharding/failstop/run_failstop_e2e.sh --node SL3060 --output $OUT --run-mode timing --launches 3 --requests $REQ/requests.jsonl --workload $REQ/workload.json --profile-file $PROFILE --seq-length 2048 --b-max 8 --budget-seconds 3600"
# 1  WeaveTP  fail-stop emulate
$RUN --suite-id m01 --method weavetp  --scenario failstop --mode emulate --plan "$REQ/plan_failstop.json"
# 2  WeaveTP  fail-stop real
$RUN --suite-id m02 --method weavetp  --scenario failstop --mode real    --plan "$REQ/plan_failstop.json"
# 3  AnchorTP fail-stop emulate
$RUN --suite-id m03 --method anchortp --scenario failstop --mode emulate --plan "$REQ/plan_failstop.json"
# 4  AnchorTP fail-stop real
$RUN --suite-id m04 --method anchortp --scenario failstop --mode real    --plan "$REQ/plan_failstop.json"
# 5  无故障基线（方法无关，记为 weavetp）
$RUN --suite-id m05 --method weavetp  --scenario nofault  --mode emulate --plan "$REQ/plan_nofault.json"
```

每组是 3 次独立 launch（`<suite>_<method>_<scenario>_<mode>_00..02`，端口 29860+k），launch 间自动等对端终态并重做独占预检；某次失败后余下记 `not_run`，重跑换新 `--suite-id`，不删旧结果。`--budget-seconds 3600` 覆盖整组 3 次（每次约 108 s 服务 + 45 s 等待 + 两次模型加载）；real 组若 16 world 重建慢可加到 5400。每组输出 `…/SL3060/matrix_manifest_<method>_<scenario>_<mode>.json`、`progress_<…>.json`、每 launch 的 `result.json`、`check.json`、`events_actual.json`、`timeline_*.jsonl`、`switch_events_*.jsonl`、`serving_checks_*.json`、`source_plan_0_*.json`、`rank_map.json`、`<launch>_preflight_exclusive.json`；timing 不写 `kv/*.pt`。

查看进度（任一节点）：`tail -n 40 $(ls -td /data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance/kickoff_<NODE>_* | head -1)/nohup.log`。

## ⑥ 分析（SL3060，CPU）

```bash
source /data/ubuntu/lxh/weavetp/env.sh && cd "$REPO/code/current" && "$PYTHON" -B tools/resharding/failstop/analyze_failstop_e2e.py --out /data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance/analysis_$(date -u +%Y%m%dT%H%M%SZ) /data/ubuntu/lxh/weavetp/acceptance/step07_e2e_acceptance/SL3060
```

递归找到每个含 `result.json` + `events_actual.json` 的 launch（标定 launch 也会被纳入：分析前把 `cal*` 目录移到 `calibration/` 子目录之外，或在命令里只列正式 launch 目录）。输出 `summary.csv`（方法×场景×模式 的 mean / ddof=1 std / n_valid / n_censored）、`launch_metrics.csv`、`metrics_<launch>.json`、`per_second_<launch>.csv`、`rejected.json`（校验不过的 launch，退出码 1）。缩容 TTP 缺降级参考时用降级窗口后半段自参考（`shrink_reference_source=self…`）；若另有参考值加 `--degraded-reference <tok/s>`。

## ⑦ 打包回传与恢复

两节点各打包 `step07_e2e_acceptance` 下本次的 suite 目录与 kickoff 日志、SL3060 的 `analysis_*`，回传后切回正式实验提交：

```bash
cd /data/ubuntu/lxh/weavetp/acceptance && tar czf step07_<日期>_<NODE>.tar.gz step07_e2e_acceptance
git -C "$REPO" checkout --detach f7d555998b45756998465924d733fd70e553d6b4
```

请一并回传 b3 两方法快验的 tar 与 receipt、标定 launch 的 `result.json`、冻结请求集的 `manifest.json`。
