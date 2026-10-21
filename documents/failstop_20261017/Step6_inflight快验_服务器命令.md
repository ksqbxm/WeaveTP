# 扩容在途请求快验（inflight）：服务器命令

交付：实现 A `c4df30a8a2bc2b9435cffde17cb2463be7eea113`，快验清单 B `568c319c088181723a23fd1d554f0beea87a02f0`（`releases.json["step6"]` = A）；本文件在 B 的子提交 C 里，C 随 bundle `failstop_inflight_568c319.bundle`（基线 `f7d555998b45756998465924d733fd70e553d6b4`，ref `refs/heads/codex/failstop-inflight-568c319` 指向 C）一起交付，bundle 的 SHA256 在同名 `.sha256` 文件里。改动与 CPU 验证（258 项）见 [08_AnchorTP最小接入.md](08_AnchorTP最小接入.md) §9。服务器 GPU 未运行，以下命令未在服务器执行过。

## ① 导入 bundle（Windows → SL3061 → SL3060）

Windows：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failstop/failstop_inflight_568c319.bundle'
Get-FileHash -LiteralPath $Bundle -Algorithm SHA256
Get-Content "$Bundle.sha256"
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

两行 SHA256 应相同。SL3061：校验、导入、转发 SL3060：

```bash
set -e
cd /data && mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c failstop_inflight_568c319.bundle.sha256
test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
git -C "$REPO" bundle verify "$PWD/failstop_inflight_568c319.bundle"
git -C "$REPO" fetch "$PWD/failstop_inflight_568c319.bundle" refs/heads/codex/failstop-inflight-568c319
test "$(git -C "$REPO" rev-parse FETCH_HEAD~1)" = 568c319c088181723a23fd1d554f0beea87a02f0
test "$(git -C "$REPO" show FETCH_HEAD:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["step6"])')" = c4df30a8a2bc2b9435cffde17cb2463be7eea113
git -C "$REPO" rev-parse FETCH_HEAD
git -C "$REPO" checkout --detach 568c319c088181723a23fd1d554f0beea87a02f0
scp failstop_inflight_568c319.bundle failstop_inflight_568c319.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`rev-parse FETCH_HEAD` 打印的是 C（本文件所在提交），请记入回传。SL3060：同上去掉最后一行 `scp`。

## ② inflight 快验（两节点，60 s 内先后启动；共享 GPU 可跑；预算 600 s）

先 emulate，再 real；每次两节点各一条，SL3061 只改 `--node SL3061`。`quick_check.sh` 按 `releases.json["step6"]` 检出 A，结束后自动切回 `f7d5559`，所以每条命令先检出 B。

```bash
source /data/ubuntu/lxh/weavetp/env.sh && git -C "$REPO" checkout --detach 568c319c088181723a23fd1d554f0beea87a02f0 && bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step 6 --layer inflight --method weavetp --mode emulate --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json --reference-root /data/ubuntu/lxh/weavetp/acceptance/step05_model_correctness/s5q_20261017T165421_80b77c92_single/SL3061
```

第二次把 `--mode emulate` 改为 `--mode real`。脚本打印日志路径（`/data/ubuntu/lxh/weavetp/acceptance/step06_batched_serving/kickoff/<NODE>_*/nohup.log`）后在后台运行。SL3060 的结论行：

```text
QUICK CHECK step=6 layer=inflight method=weavetp result=PASS ({"mode": "emulate", "method": "weavetp", "metrics": {"requests_completed": 200, "check_passed": true, "kv_check_passed": true, …}, "files": {…}, "inflight_at_expand": <≥1>, "analysis": {"exit_code": 0, "rejected": [], …}}) backend=block attention=torch_splitkv shared_gpu_mode=true single_node=false accepted=false logs=…
```

通过标准：控制门 passed（200 条请求全部完成、`check.json` 通过、扩容 KV 逐位检查通过），`inflight_at_expand ≥ 1`（扩容 cutover 时 EDP0 上仍在跑、由加入 rank 接着解码的请求数），`analyze_failstop_e2e` 接受门的 `result.json`/`events_actual.json`（退出码 0，`rejected` 为空）。SL3061 的结论行同样是 `result=PASS`，`metrics` 为 null。若失败信息是 `inflight_error`（负载在扩容前排空）或 `quick launch exceeded 600 seconds`，请回传，需要调整条数。

结果在 `/data/ubuntu/lxh/weavetp/acceptance/step06_batched_serving/<suite>/<NODE>/`：`summary.json`、launch 目录 `<suite>_emulate`（或 `_real_00`）里的门文件 `result.json`、`check.json`、`events_actual.json` 与 run_case 的 `launch_result.json`，SL3060 另有 `analysis/`（`summary.csv`、`launch_metrics.csv`、`metrics_*.json`）。请回传两节点两次运行的 `<suite>_<NODE>.tar.gz` 与 `<suite>_<NODE>.receipt.json`（与 suite 目录同级）。
