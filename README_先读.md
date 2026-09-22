# WeaveTP 代码与实验资料

更新日期：2026-09-20。

本包包含阅读指南、代码讲解、实验结果、论文调研、baseline 说明，以及当前实验源码、启动脚本、依赖声明和测试。先按下方顺序阅读，再检查环境并选择对应的实验入口。

## 包含哪些内容

| 分类 | 内容 | 如何使用 |
|---|---|---|
| documents | 阅读指南、代码/设备说明、实验结果、7 份 baseline/复现 README、Excel | 先理解研究目标、结果及复现身份 |
| code / current | 当前 Megatron 工作副本的框架代码、live benchmark、重分片实现、实验入口、依赖/构建声明、单元测试和独立正确性测试 | 后续开发与验证的主体；保留原有目录关系 |
| code / native_preparation | HarMoEny、ExpertFlow 已收集的官方源码及本地准备脚本；FIRM-MoE 的独立算法重实现 | 与本项目的三个机制代理分开，不表示已完成原生性能复现 |
| code / history_reference | 与当前关键源码不同的归档异步执行器 | 仅作版本对照，不是另一套完整可运行工程 |
| package_metadata | 文件 hash 清单、关键版本关系、正式实验记录中的配置、打包检查与环境检查表 | 确认材料身份及运行前条件；不含账号密码 |

不附模型权重、带宽 profile、失败日志、历史服务器连接脚本、编译缓存或已生成的实验输出。性能结果与统计口径见随包 Excel 和 README；本包不是包含所有外部资源的容器镜像。

## 当前实验代码从哪里开始看

| 文件 / 模块名称 | 做什么 |
|---|---|
| benchmark_live_moe_tp.py | 实际模型构造、token 推进、参数/KV 迁移、接管和计时的主体 |
| run_deepseek_v2_lite_directional_wave_compare.sh | 正式三种配置的对比入口：Fixed-2048、Directional、WeaveTP；默认各 3 次启动，轮换执行顺序 |
| run_deepseek_v2_lite_selected_baselines.sh | 三篇论文机制代理和普通 Live-default 的入口；不包含历史 WeaveTP 的单独运行，不能称为自动复现全部五方法 |
| run_deepseek_v2_lite_live_benchmark.sh | 设置 DeepSeek 模型、EP、路由和验证参数，转交通用 benchmark |
| run_live_moe_tp_benchmark.sh | 组织分布式进程启动和实际 benchmark 参数 |
| convert_deepseek_v2_lite_checkpoint.py | checkpoint 转换工具；不包含转换后的权重 |
| planner / async_execution / copy_services | 分片映射、launch/wait/commit、本地及远程复制的实现 |
| live / residual tracker / scheduler | KV 区间处理、剩余任务与分波组织 |
| correctness / reference / harness / run_suite | 独立全局张量参考、测试适配层和 CPU 状态正确性测试 |
| correctness / gpu03b | 小型真实 GPU 通信和迁移验证入口；该入口尚无 GPU 通过记录 |

阅读顺序建议：正式对比入口 → live benchmark → planner → async execution 与 copy service → KV 补齐与接管 → 独立正确性测试。

旧名 MOETP++ / moetp 仍在代码和配置中出现，与论文中的 WeaveTP 指同一项目。阅读时将这些名称对应到同一实现；保留旧接口名称可维持脚本与历史数据的对应关系。

## 三篇论文 baseline 的代码在哪里体现

Flying Serving-proxy、AnchorTP-proxy、Llumnix-proxy 是同一个 live benchmark 的不同方法分支，由 selected-baselines 入口选中，不是三个独立官方 serving 系统。

- Flying Serving-proxy：驻留/共享状态机制参考。
- AnchorTP-proxy：参数/KV 的状态迁移与策略对照。
- Llumnix-proxy：目标权重驻留条件下的 KV-only 对照。
- Live-default：本项目普通 live 基线。

HarMoEny、ExpertFlow、FIRM-MoE 放在独立的 native_preparation 分类；前两者只是已有官方代码及准备入口，后者只有算法级 CPU trace 重实现。官方代码副本不包含大数据集、模型、既有输出和重复下载归档；补齐外部依赖后才能进入对应原生验证。

## 当前版本与历史版本

当前代码采用已有工程映射明确指定的开发工作副本，包含后续第 02/03/03B 步新增的独立审计/正确性测试。

核对了 benchmark、planner、异步执行器、live、NCCL copy service 以及四个主要启动入口，共 9 个关键文件：与本地原工程及 newtext 归档相比，当前异步执行器含局部缺失参数检查修补，其余 8 个文件内容一致。归档异步执行器单独保留为参考，未覆盖当前代码。

这不等于已经找回历史性能运行的精确服务器 Git commit。当前代码不是一个新的“已实测性能版本”，也不能仅凭文件相同就声称历史运行环境完全复现。

## 运行前还需要准备什么

正式实验的平台是 **Linux + 8 张物理 RTX 5090、DeepSeek-V2-Lite、TP2↔TP4、EP=2**。两个布局和 8 个 worker 常驻。现有脚本不会实现完整动态任务调度器，也不是临时加入/退出 GPU 的服务。

保留了 pyproject.toml、setup.py、requirements、uv.lock 及相关构建源文件。它们描述当前源码的依赖，不是对历史实验软件版本的完整锁定；不要不加核对地安装所有可选开发依赖。实际 CUDA、PyTorch、编译扩展与 GPU 架构应匹配。

| 运行条件 | 运行前检查 |
|---|---|
| Python / CUDA 环境 | 原项目声明 Python ≥ 3.12；选择兼容 CUDA 的 PyTorch 和实际需要的扩展 |
| PYTHON | 用于启动 benchmark 的解释器；历史脚本默认值属于旧服务器，不应直接照用 |
| CHECKPOINT | 已转换为 Megatron 格式的 DeepSeek-V2-Lite 权重 |
| PROFILE | 对应当前物理设备与 rank 放置的真实带宽画像；本包没有用模拟文件替代 |
| CUDA_VISIBLE_DEVICES | 明确获准使用的 8 张物理 GPU；不能默认所有设备可用 |
| 结果目录与预算 | 使用新的实验输出位置并确认独立启动次数和运行额度 |

启动脚本中的历史默认环境需通过上述环境变量覆盖；先确认设备授权和运行预算，再启动实验。环境检查表是人工核对清单，不会被启动脚本自动读取。

正式入口与三篇代理入口是不同批次：前者默认三轮；代理入口默认重复次数不是历史五轮入选汇总的自动重建规则。正式配置观察值已从历史 9 份 JSON 单独提取，不能把一次新运行自动追加到旧结果。

## 如何选择并操作

1. **先看实现**：结合《README_代码与设备》中的模块表，按“对比入口 → benchmark → planner → 执行与复制 → KV 接管”的顺序阅读源码。
2. **选择实验**：比较 Fixed-2048、Directional、WeaveTP 时使用正式方向对比入口；研究三篇论文的机制代理时使用 selected-baselines 入口。两类结果分别保存和统计。
3. **准备运行**：完成上表中的解释器、权重、带宽画像、设备和预算检查；不要直接沿用脚本中的历史默认值。
4. **先验证再测量**：先检查小型正确性测试和 GPU 通信，再进行真实模型验证；取得对应通过证据后再运行性能对比。测试入口存在不代表验证已通过。
5. **读取结果**：按《README_实验结果》和 Excel《指标说明》先在单次 launch 内汇总，再跨独立 launches 计算均值与样本标准差。新结果单独记录，不覆盖历史结果。

## 已完成的打包检查

- 当前代码 806 份 Python 文件通过 Python 3.12 语法编译检查。
- 必要 benchmark、规划/执行、checkpoint 转换、构建源与测试入口存在。
- 源码复制文件逐一 hash 核对；说明文档仅调整阅读与操作表述。
- 检查并排除凭据、远程连接脚本、缓存和历史输出；保留原有许可证与版权声明。
- ZIP 逐文件完整性检查。

这些是打包和静态检查，不是模型运行验证；未重新运行 tensor/CPU 测试或 GPU 实验。正确性验证范围见《README_代码与设备》。模型权重和带宽画像未随包提供，运行前必须另行准备。
