# 快验：fail-slow 在线切换 Step 9（服务器命令）

交付：实现 A `d8b8cd7ef1d59926e3e294a2e6d905e115cdfdfb`（审阅后的 `a7c64d5` 加上重放 history 的修复：restart 循环的竞态与跨副本重放时 prompt 重复，见 [10_failslow在线化依据.md](10_failslow在线化依据.md) §5），清单 B `005a4c84e40fa2c576e29952f0638f5807f530a3`（`releases.json["failslow"]` 指向 A；`step6` 仍为 `073a50f41e70ee54551f3f2e2613c60203697ad1`，fail-stop 不变）。本文件在 B 的子提交 C 里，C 随 bundle `failslow_online_005a4c8.bundle` 交付（基线 `f7d555998b45756998465924d733fd70e553d6b4`，ref `refs/heads/codex/failslow-online-005a4c8` 指向 C），bundle 的 SHA256 在同名 `.sha256` 文件里。上一版 bundle `failslow_online_9488be4`（A `a7c64d5`、B `9488be4`）作废；已导入的旧分支留着不用即可。设计与删改清单见 10；CPU 回归（交流供电、主机空闲）303 项通过、1 项仅 Linux 的测试跳过、119 subtest 通过，与同时段重跑的 a7c64d5（302 项）相比只多出新单测 ReplayHistoryTests（见 10 §5、§6）。以下命令未在服务器执行过。

## ① 导入 bundle（Windows → SL3061 → SL3060；只 fetch 和校验，不检出主仓库）

Windows：

```powershell
$Bundle = 'C:/Users/ksqbx/.codex/worktrees/failslow-online/failslow_online_005a4c8.bundle'
Get-FileHash -LiteralPath $Bundle -Algorithm SHA256
Get-Content "$Bundle.sha256"
scp $Bundle "$Bundle.sha256" 5090-SL3061:/data/ubuntu/lxh/weavetp/
```

两行 SHA256 应相同。SL3061：校验、fetch 到本地分支、转发 SL3060：

```bash
set -e
cd /data && mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c failslow_online_005a4c8.bundle.sha256
git -C "$REPO" bundle verify "$PWD/failslow_online_005a4c8.bundle"
git -C "$REPO" fetch "$PWD/failslow_online_005a4c8.bundle" refs/heads/codex/failslow-online-005a4c8:refs/heads/codex/failslow-online-005a4c8
test "$(git -C "$REPO" rev-parse refs/heads/codex/failslow-online-005a4c8~1)" = 005a4c84e40fa2c576e29952f0638f5807f530a3
test "$(git -C "$REPO" show 005a4c84e40fa2c576e29952f0638f5807f530a3:code/current/tools/resharding/failstop/releases.json | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["failslow"])')" = d8b8cd7ef1d59926e3e294a2e6d905e115cdfdfb
git -C "$REPO" rev-parse refs/heads/codex/failslow-online-005a4c8
scp failslow_online_005a4c8.bundle failslow_online_005a4c8.bundle.sha256 ubuntu@10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`rev-parse refs/heads/codex/failslow-online-005a4c8` 打印的是 C（本文件所在提交），请记入回传。SL3060：同上，去掉最后一行 `scp`。导入只新增分支 `codex/failslow-online-005a4c8`，主仓库的检出和工作区不变。

快验直接从清单 B 读出 `quick_check.sh` 运行，不需要先检出：它按 B 的 `releases.json["failslow"]` 找到实现 A，把 A 的 `quick_driver.py` 取到 kickoff 目录运行；`quick_driver` 在 quick 锁内检查已跟踪文件干净、检出 A，结束后切回 `f7d5559`。运行期间主仓库被检出为 A，请避开其他正在使用主仓库工作区的任务。

## ② quick_check --step 9 --method（weavetp、restart、ignore 各一次）

每条命令只跑该方法一次 fail-slow W 级 launch：共享 GPU、correctness 模式，SL3061 的 GPU 0/1/4/5 每卡 1 个 `failslow_hog.py`，warmup 20 s、D 30 s、tail 20 s、检测 10 s，toy 请求池 2000 条，到截止时还有请求未完成。每条约 5 分钟。每次两节点 60 s 内先后启动，SL3060 先；SL3061 的命令只把 `--node SL3060` 改成 `--node SL3061`。三条依次跑（同一时间只跑一条）。

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 005a4c84e40fa2c576e29952f0638f5807f530a3:code/current/tools/resharding/failstop/quick_check.sh) --manifest 005a4c84e40fa2c576e29952f0638f5807f530a3 --step 9 --method weavetp --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 005a4c84e40fa2c576e29952f0638f5807f530a3:code/current/tools/resharding/failstop/quick_check.sh) --manifest 005a4c84e40fa2c576e29952f0638f5807f530a3 --step 9 --method restart --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

```bash
source /data/ubuntu/lxh/weavetp/env.sh && bash <(git -C "$REPO" show 005a4c84e40fa2c576e29952f0638f5807f530a3:code/current/tools/resharding/failstop/quick_check.sh) --manifest 005a4c84e40fa2c576e29952f0638f5807f530a3 --step 9 --method ignore --node SL3060 --profile-file /data/ubuntu/lxh/weavetp/profiles/step3_s3q_20261017T200325_62b5818c_real_00.json
```

`nohup.log`（`/data/ubuntu/lxh/weavetp/acceptance/step09_failslow/kickoff/<NODE>_*/`）末行应为 `QUICK CHECK step=9 method=<方法> result=PASS ...`。只检查：launch 跑完且 passed、本节点没有残留的 `failslow_hog.py`；SL3060 上事件顺序正确、截止时确有未完成请求。

SL3060 在结论行之前打印一行 `STEP 9 SUMMARY <方法>: {...}`（同样写在 `<run>/SL3060/summary.json` 的 `launches.<方法>.gate.summary`）。字段全部取自 `failslow_result.json` 已有字段，没有的标"缺失"：

| 字段 | 含义 | 来源 |
|---|---|---|
| `restart_reload_s.tp2` / `.tp4` | restart 从停机到 TP2（缩容）/ TP4（扩回）重新加载完成的耗时 | `restart_loaded_s` − `worker_events_s.fail` / `.rejoin_start` |
| `switch_s.shrink` / `.expand` | EDP0 降级 / 恢复两次切换的耗时 | EDP0 切换里程碑 `source_release_done − switch_trigger` |
| `decode_steps_in_switch.EDP0/…`、`EDP1/…` | 切换期间各域的前台 decode 步数 | 各域 leader 切换记录的 `overlap_steps` |
| `replayed_tokens` | 重新 prefill 的 token 数 | `replayed_tokens` |

预期：weavetp 的 `replayed_tokens` 为 0，EDP0 与 EDP1 的缩容 decode 步数都大于 0；restart 有两个重新加载耗时、`replayed_tokens` 大于 0；ignore 的切换类字段全部为"缺失"。

## ③ 正式实验：压缩时间线（不改代码，只用现有参数）

```text
--scenario failslow --mode emulate --run-mode timing --fault-gpus 0,1,4,5 --warmup-s 60 --slow-seconds <S> --tail-s 60 --detect-seconds 10
```

- S = ⌈(restart TP2 重新加载耗时 + 90) / 30⌉ × 30，TP2 重新加载耗时取 ② restart 快验摘要的 `restart_reload_s.tp2`。**S 留空，快验后填写。**
- 矩阵：weavetp / anchortp / restart / ignore × W / M / S 级（`--method <方法> --slow-level <W|M|S>`），加 nofault 1 次（`--method nofault`，不带 `--slow-level`），共 13 个 launch。
- 每次 launch 的预算与各级超时由 warmup + D + tail + 启动 + 两次重新加载 + 余量自动计算，不需要 `--budget-seconds`。
- 正式队列另行安排，本文件不含其命令。

## 回传

失败时先看 `<run>/<node>/error.txt` 与 `summary.json`。请回传每次快验两节点的 `<suite>_<NODE>.tar.gz` 与 `<suite>_<NODE>.receipt.json`（在 `acceptance/step09_failslow/` 下）。
