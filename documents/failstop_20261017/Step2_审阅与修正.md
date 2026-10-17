# Step 2 审阅、修正与共享 GPU 正确性模式

实现：`fa152b134107f3efdf4335c2a99752d02b61bcf0`。快验清单：`df4d0096243024ca1a1f92447181362e1411f885`。本轮在 `codex/failstop` 审阅并修改；已重新阅读 CODEX.md、总计划、Step 2、code/current/AGENTS.md 和贡献说明。服务器命令以[当前交付说明](Step2_服务器命令.md)为准。

状态：已实现、CPU 已验证；服务器 NCCL/GPU 和 300 秒实际时长待用户回传。本轮没有登录服务器、运行 GPU 或改动其他用户任务。

## 审阅发现与修正

| 级别 | 原问题与影响 | 唯一生效路径及验证 |
|---|---|---|
| P1 | emulate 回归复用 real 的 pre-spawn 检查，四个仍存活的注册 CUDA context 被当作异常进程拒绝；独占时还错误要求模拟失败卡显存回到空卡基线。 | 监管器公共生命周期调用 `rejoin_preflight`；real 保留真实退出/回收要求，emulate 校验全部八个 context。测试运行真实 `Supervisor.run`，共享/独占两种条件均覆盖，不再只测替代监管器的协议模拟。 |
| P1 | TP 组被后端排序后，EP/DP 仍按请求列表的位置配对；不同 EP 使用不同请求顺序时会连错 TP 分片。仅验证 collective 求和、EP 位置或更新 rank_map 无法发现这一错误。 | 从各 TP/ETP 组的实际规范顺序构造 EP/DP/EDP 配对。新增 CPU 单测与 4 进程真实 Gloo 检查：仅 EP0 的 TP 请求顺序反转，EP collective 必须连接同一实际 TP 坐标。 |
| P1 | world 再次 ready 后，未 rebind 的旧模型也能 forward，可能调用已经 abort 的组。 | 资源失效推进 generation；绑定只在构造或成功 rebind 时更新 generation。forward 同时检查 ready、source-only 和当前 generation。旧绑定不能靠全局 ready 恢复。 |
| P1 | 嵌套 forward 被 pre-hook 拒绝时，提前释放一次锁，always-call post-hook 又释放一次，可能在外层 forward 尚未结束时允许重绑定。 | 每次 pre-hook 获取锁并登记，包括将被拒绝的调用；释放统一由 always-call hook 完成。线程竞争用例验证拒绝内层调用不会放开外层边界。 |
| P1 | 原重绑定检查读取过时字段 `enable_cuda_graph` / `dispatcher_manager`，漏掉当前 native 的 `cuda_graph_impl`、`cudagraph_manager` 和 Flex `_comm_manager`；并且可能改完前面的模块才拒绝后面的模块。 | 删除旧字段逻辑，按本仓库当前字段检查，先验证所有对象再替换组。测试证明拒绝后模型原组引用没有被部分修改。 |
| P2 | `next_round()` 只增加 round，旧缓存未清理，resources.key 与 session.cache_key 不一致；不完整/重复 local order 也能进入状态机。 | 同一锁内验证正常可用边界、清理轮次缓存、更新 key；转换状态验证与更新使用同一锁；要求两个 EP 各有恰好两个合法 survivor。 |
| P1 | real 多 launch 只要有一次成功，模式摘要就写 passed；共享多轮测试可能 accepted=true；Step 2 借用虚构 repeats=10/1 控制 Step 1 的 quick 字段。 | 汇总逐一匹配所选 launch，任何失败/缺失均使该模式失败；共享/quick 永不 accepted，独占完整矩阵才可 accepted。删除 Step 2 的假 repeats，公共 launcher 直接读取 quick_mode。预算耗尽也落盘失败结果。 |

这些问题先通过新增回归用例复现再修正。初次用例还暴露测试 fixture 缺少最终 GPU 采样的 `end_local`；补齐 fixture 后通过。没有把测试模拟成功写成 GPU 验收。

## 测试专用开关

`run_topology.py --correctness-only` 是 Step 2 唯一共享 GPU 开关，默认关闭；`--quick` 自动开启。旧 Step 2 `--allow-foreign-gpu-processes` 已移除且不保留别名。Step 1 独立入口及原 homogeneous 正式入口按总计划保持原有合同。

- 输出 `correctness_only=true`、`shared_gpu_mode=true`、`timing_valid=false`、`accepted=false`；两节点必须一致。即使全部检查通过，也仅表示测试正确性通过。
- 复用 Step 1 的注册 PID、pidfd、MPS 隔离、独立 CUDA context 和任务进程回收规则。允许 foreign GPU/MPS 任务；不停止 MPS 或其他用户进程，不要求整卡空闲。
- 权重逐位、指针、拓扑、空目标、故障设备零数据/占位字节、新旧 PID、清理及失败退出检查均保留。耗时不作为正确性或性能判据；超时仍是测试失败。
- 共享条件的单阶段等待上限为 120 秒，启动为 180 秒，均受剩余 launch 预算限制。快速入口仍统一约束含等待/启动/打包的 300 秒总预算，不能以共享为由无限等待。
- 快速入口的 summary 与最终 receipt 都携带上述测试标签，仍要求两个节点的 tar、receipt 和 exit_code 一致。

## 本地验证与边界

| 检查 | 结果与原始证据 |
|---|---|
| failstop unittest | 86/86，通过，含本轮 11 项新回归；[日志](step2_review_evidence/cpu_tests.log) |
| round 边界最终收紧后的定向复验 | 1/1，通过；[日志](step2_review_evidence/round_boundary.log) |
| 原 NCCL packing 回归 | 16/16，通过；[日志](step2_review_evidence/packing.log) |
| 4 进程真实 Gloo | 4/4，通过：TP/EP 坐标配对、实际组顺序、collective、指针重绑定、非连续子组 P2P；[日志](step2_review_evidence/gloo.log) |
| isort、Ruff F、Python 编译、bash 语法、dry-run、diff 检查 | 通过；[追溯记录](step2_review_evidence/validation.json) |
| 服务器 GPU / NCCL / 快验实际时长 | 未运行，待回传 |

CPU 并发协议仍执行 emulate 连续两轮、real 新 worker 回归。Step 1 的原 54 项测试也在 86 项内，验证公共监管器抽取和 launcher 元数据改动未破坏已有 CPU 合同。

本轮未扩大到 Step 3 主机库/混合选源、Step 4 异步驱动或 Step 5 DeepSeek GPU 验收；fix2 GPU 关口不变。重绑定检查依据当前 native 源码字段，实际 native 模型 GPU 正确性仍由后续步骤验收。

增量包：`C:/Users/ksqbx/.codex/worktrees/failstop/step2_df4d009.bundle`，807816 字节，基线 `f7d555998b45756998465924d733fd70e553d6b4`。SHA256：`86ceb7234aeb1b10409f8ccadaf60311f2ae21d13e22a66f6628f967b243a7d4`。包包含固定快验清单及其历史实现；上传 SL3061 后转发 SL3060 的完整命令见服务器说明。
