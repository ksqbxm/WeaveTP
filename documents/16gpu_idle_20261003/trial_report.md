# 2026-10-03 GPU 空闲判定修订

基线为 main `1934b047d4b89b527340c19d99ab988627597bb2`。修改前重新完整阅读 CODEX.md、总计划、源码 AGENTS.md 和贡献说明，核对 T08、冒烟及正式运行调用路径。用户报告 SL3060 GPU4/6/7 长期稳定的 MPS 残留显存为 159/91/133 MiB、利用率 0%、无可见计算进程；这是用户回传，未远程核验。本次未 SSH、安装或启动 GPU。

## 唯一生产改动

compare_weavetp_16gpu.py 的 gpu_state 添加 utilization.gpu，compute-apps 添加 gpu_uuid 用于占用诊断。check_idle 对全部八张本地卡要求 memory.used ≤200 MiB、utilization.gpu=0；计算进程仅允许 MPS_OWNER 的 nvidia-cuda-mps-server，两节点规则相同。错误给出 node、GPU index/UUID 与显存、利用率或进程规则违例。没有等待、重试或旧阈值兼容路径。

gpu_memory、cleanup、node_action 及所有退出显存检查均未修改；仍要求相同 UUID 集合、无本任务残留进程及每卡结束显存 ≤启动前基线 +16 MiB。没有把结束恢复上限改成 200 MiB。

## T08 文件逐项核对

- run_t08.sh：0 行修改；只加载 env.sh 并运行同目录 t08.py，无固定 commit 或空闲阈值。
- t08.py：0 行修改；185–189 行读取冻结配置 WEAVETP_COMMIT 并与干净 HEAD 比较，两端部署配对要求相同 commit；预检 test_node.py 和运行前第 117 行均调用本次 compare.check_idle。第 91、237 行仍为基线 +16 MiB。
- check_commands.py：0 行修改；`COMMIT = 'a' * 40` 是 CPU 测试夹具，不是部署版本；65/200 数值用于模拟显存恢复，未实现独立空闲判定。
- t08_20261001.sha256：0 行修改；所覆盖的 shell 与 Python 字节未变，原清单仍有效。T08 目录所有文件与 1934b04 完全一致，旧 trial_report 中的 81 MiB 历史描述由本报告和更新的总计划取代。

冒烟运行文件 smoke.py/run_smoke.sh 无修改。test_smoke.py 更新查询夹具和 200/201 边界，不再锁定旧 check_idle 的 AST；恢复、清理、RPC 仍锁定不变。dry-run 基线更新为 1934b04；该版源码超过 Windows 命令行长度限制，测试改为经 stdin 执行原始源码，不修改或归一化输出。T06/T07 旧回归只给查询夹具增加 `, 0`。冒烟 SHA 清单同步更新 compare、测试和阶段报告 hash。

## 验证

新回归先在旧实现上失败，确认 159 MiB 被误拒绝、诊断和查询字段不符；原基线 +16 检查仍通过。修改后通过：159 MiB/0%、200 MiB/0%；拒绝 159 MiB/5%、300 MiB/0%、201 MiB/0%、非 MPS 进程、错误 MPS owner；允许两节点上指定 owner 的 MPS server。启动基线 159 MiB 时，结束 175 MiB 通过、176 MiB 拒绝。

完整验证命令：

```powershell
python -B -X utf8 documents/16gpu_T085_20261001/check_cpu.py --output-dir E:/data/weavetp_idle_20261003/verified
```

158/158 通过：汇总 28、compare 29、画像 18、服务器检查 11、既有审阅 4、观测 25、T06/T07 回归 3、T08 16、冒烟 24。16 个 Python 文件编译、7 个 shell 语法、git diff --check 通过。原始日志位于命令指定目录；validation.json 随本报告保存。第一次全套执行为 157/158：旧 T06/T07 夹具只有三列，补零利用率后全套通过。没有修改 imports，isort 触发条件不适用。

Fixed/Directional 包含在全部九次正式 dry-run 原始输出中，与 1934b04 逐字节相等（35,168 bytes），两侧 SHA-256 均为：

`08243d344b7a25e7ebd728fa85f3f46243752fb39791f68b3324d1511c0a1045`

check_idle 只在节点预检/运行中执行，不进入 dry-run 的 request/node_commands；shell launcher、实验参数和算法没有改动。

## 新提交部署

最终交付消息提供完整新 SHA。用户自行让两台仓库具备提交对象，然后两机按该 SHA checkout，确认干净 HEAD，持久化更新 WORK/env.sh 的 WEAVETP_COMMIT。SL3060 的 MPS_OWNER 应为用户指定的 yiwei；不关闭 MPS 或处理残留显存。服务器命令不含 git fetch/pull，不手动上传文件。

两机更新后，仅在 SL3060 运行仓库 documents/16gpu_T08_20261001/run_t08.sh。不复用旧 deployment.json 或失败任务目录；入口自动创建新 run。两机 GPU 空闲和 /data ≥20 GB 门禁均在 GPU 启动前完成。预计 10–35 分钟，成功必须有两机 T08_PREFLIGHT_OK、实际 via NET/IB、两条 16×16/240 对的 T08_PROFILE_OK 且 SHA 相同，最后 T08_RESULT exit=0。画像文件及其 SHA-256 记录保留在两机 measurement 目录。
