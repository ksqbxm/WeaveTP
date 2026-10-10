# WeaveTP DP 规模敏感性第一步交付

第一步已实现改动 2、3、4：launcher 环境支持、默认关闭的迁移观测、三方法 CPU 汇总。任务规范先于代码写入并单独提交；第二步 TP 参数化、九层随机模型和完整 README 运行示例尚未实施。

## 提交范围

- 基线：7deac20feaf76ee4e6a0c53b2289271a903dd12c。
- 分支：codex/dp-sweep，独立 worktree；第一步范围为上述基线至本报告所在提交。
- 任务文档提交：2940c69；功能、测试和验收器提交：696140129b1cf759ec7ef3c3ca08ecb099d9e2bb。
- main 保持 f7d555998b45756998465924d733fd70e553d6b4，原工作区改动未带入。
- 没有连接实验服务器，没有启动 GPU。Git 分支推送与实验服务器连接分开记录。

## 文件与行数

以下增删统计相对于 7deac20，行号指本交付版本。

| 改动 | 文件 | 主要行号 | 增加/删除 |
|---|---|---|---:|
| 2 | code/current/tools/resharding/run_live_moe_tp_benchmark.sh | 21：override/校验；206：观测开关 | +8 / -1 |
| 2 | code/current/tools/resharding/run_deepseek_v2_lite_live_benchmark.sh | 29：纯环境透传说明；36：本地进程数/EP | +4 / -2 |
| 3 | code/current/examples/rl/benchmark_live_moe_tp.py | 194：参数；1843：reset；2248：读取峰值；2362：事后汇总 | +29 / -3 |
| 3 | code/current/tools/resharding/weavetp_observations.py | 11：实际 host；20：Base/Delta/峰值汇总 | +36 / -0 |
| 4 | code/current/tools/resharding/summarize_dp_sweep.py | 38：指标；68：输入；101：断言；153：统计；250：CLI | +274 / -0 |
| CPU | code/current/tests/unit_tests/resharding/test_dp_sweep.py | 1 | +173 / -0 |
| CPU | code/current/tests/unit_tests/resharding/test_summarize_dp_sweep.py | 1 | +138 / -0 |
| CPU | code/current/tests/unit_tests/resharding/test_dp_sweep_coordinator.py | 1 | +103 / -0 |
| CPU | code/current/tests/unit_tests/resharding/test_live_storage_lifecycle.py | 30、185：为新测试开放参数/依赖注入 | +4 / -1 |
| 验收 | documents/dp_sweep_20261011/check_cpu.py | 1 | +219 / -0 |
| 任务规范 | documents/dp_sweep_20261011/task.md | 1 | +78 / -0 |

证据位于本目录 evidence/；原始比较文件使用 binary 属性保留 Git 中的原始字节，不执行换行转换。

## 行为与统计口径

新观测通过 DP_SWEEP_METRICS=1 或 --live-dp-sweep-metrics 开启。默认参数 Namespace 不增加该项；关闭时没有新增结果字段、观测通信或 CUDA 峰值调用。原顶层 import 顺序也有回归保护。

Base、Delta 接收侧网络字节分开记录；remote_bytes 等于二者之和，也等于 weight_bytes + kv_bytes。同卡字节仅进入 local_copy_bytes。transport_s 只求 Base wave_records 中已跨 rank MAX 的 transport_s 之和，保留原 fallback，不计 Delta transport。allocated/reserved 峰值在原计时边界之外 reset/读取，覆盖切换后的数值校验。

三方法汇总支持任意非空子集，先在 launch 内平均，再跨 launch 等权求均值和样本标准差。wave_overhead_s 是逐切换的 base.wall_s − transport_s，标准差由该差值聚合得到。报告提供相对 default 和 2x 相对 weavetp 的均值比值改善，不平均逐次百分比。

波数检查保留逐 launch、逐方向切换的实际波数和有符号差值；缩容 default/weavetp 允许相差 1 波。执行模式、计划标签或波数不符时输出 ANOMALY；缺少配对为 INCOMPLETE，报告仍会生成且退出码为 1。未选择的方法为 N/A。报告明确波数差异也可能改变性能，不能把全部收益归因于选源。

原始 baseline/fifo 标签及缩容 plan_variant 的历史含义保持不变，可能触发检查；没有重命名标签或修改执行算法使断言通过。cached-plan 观测用于区分全局门槛否决与其他 fallback。

## CPU 结果

- T09 完整检查链：181/181 通过，10 个套件，0 失败、0 跳过。
- 相关既有回归与新功能：321 项测试、130 个子测试通过，0 失败；[完整日志](evidence/regressions.stdout.log)、[JUnit](evidence/regressions.xml)。
- 最终导入顺序保护与协调器定向复查：10 项测试、6 个子测试通过；[日志](evidence/final_targeted.log)。
- Python 编译 8 个文件，两个 launcher 的 bash -n，通过。
- 最终 uv run isort --check-only、工作区和暂存区 git diff --check，通过。
- 选源/执行/KV 核心目录、原 formal 汇总脚本及历史 T09 文件相对基线无差异。

首次最终 isort 检查发现工作目录引起的模块分类差异，已用明确的 megatron/tools first-party 分类修正，并保留 benchmark 原导入顺序。暂存后 diff --check 发现新增文件的末尾空行，已清理。修正后的结果见 [final_static.json](evidence/final_static.json)；首次 isort 非零结果保留在 [isort_check_initial.json](evidence/isort_check_initial.json)。测试中的 Transformer Engine/Apex/absl 缺失提示属于 CPU 环境 warning，不是跳过或失败。

复验命令：

    python -B -X utf8 documents/dp_sweep_20261011/check_cpu.py --output-dir <仓库外的新目录>

原 T09 验收脚本硬编码早期提交的整文件保护条件，因此本验收器调用它的原始完整套件和 worker，另按本次基线检查保护目录；没有修改历史脚本或历史证据。[总体回执](evidence/validation.json)。

## 与基线逐字节比较

[default_parity.json](evidence/default_parity.json) 的 17 组全部一致：

- 12 组 launcher：两个入口，默认/node0/node1，新增参数不传以及显式 SRC_TP=2。
- 4 组真实协调器 CPU 输出：standby release 关闭/开启，各自日志和实际 _write_results 序列化文件。
- 1 组真实 Megatron 参数打印：默认 live 参数 Namespace 与打印内容一致。

[T09 比对回执](evidence/t09_parity.json) 的三组全部一致，直接比较子进程 stdout 原始字节，不归一化换行或重新序列化：

| dry-run | cases | 字节数 | 基线与当前相同的 SHA-256 |
|---|---:|---:|---|
| all | 9 | 34787 | b4a09eb0e27c2a5ee3721c6fe3bccfbec4bfcec524db3622b3daf6f4f2ed6d75 |
| only_weavetp_r1 | 1 | 3869 | b89623b10ab75cd0529b7925858d611d08835fd53781b0e3eb0f1e8179c8f234 |
| r1 | 3 | 11599 | 3ca20a9903cc993306991a4b1b3623c35be5d6333fd2181bf870965b1e1decc4 |

这些证据证明固定 CPU 条件和 dry-run 下的默认兼容，不声称真实 GPU 的非确定性计时数值逐字节相同。

## 第二步边界

SRC_TP/DST_TP/NUM_LAYERS/SKIP_CHECKPOINT 当前只按环境继承；即使传 SRC_TP=1、NUM_LAYERS=9、SKIP_CHECKPOINT=1，也不会产生新的 CLI，DeepSeek 仍按旧规则要求 checkpoint、使用 TP2/TP4 和 27 层。W=10 override 的第一步 launcher 测试预期 GBS=5；GBS=10 随第二步 SRC_TP=1 支持一起交付。

完整 S/E README 运行示例留到第二步，届时显式设置 RELEASE_STANDBY_WEIGHTS=1，并验证 TP1<->TP2 下释放、重建、TP 键索引和第二次 2->1 的真实数据迁移。当前交付后停止等待用户确认。
