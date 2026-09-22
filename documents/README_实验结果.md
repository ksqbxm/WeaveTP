# 实验结果与解读

## 两批结果必须分开

| 批次 | 用途 | 重复单位 | 对比对象 |
|---|---|---|---|
| 2026-09-01 正式三轮 | 同一原型内分析方向配置与源选择的收益 | 每种配置 3 次独立 process launch，每次 4 次切换 | Fixed-2048、Directional、WeaveTP |
| 2026-08-28 至 08-31 历史五轮 | 展示不同状态范围/机制代理的行为 | 每种方法 5 次独立 process launch，每次 4 次切换 | 历史 WeaveTP、Live-default、Flying Serving-proxy、AnchorTP-proxy、Llumnix-proxy |

均使用 DeepSeek-V2-Lite 和 8 张 RTX 5090。历史批次不能与后来调整后的正式三轮混合统计，也不能因系统名字相同就认为配置相同。9 + 25 份入选原始 JSON 已复算，结果与已有汇总一致。

## 正式主实验：优势主要在扩容

| 扩容指标 | Fixed-2048 | Directional | WeaveTP |
|---|---:|---:|---:|
| Migration base wall (s) | 6.398 ± 0.165 | 3.888 ± 0.022 | 3.589 ± 0.057 |
| Transport (s) | 2.176 ± 0.043 | 1.865 ± 0.005 | 1.753 ± 0.022 |
| Benchmark switch wall (s) | 7.156 ± 0.122 | 4.645 ± 0.036 | 4.386 ± 0.081 |
| Foreground TPOT mean (ms) | 333.7 ± 7.8 | 332.8 ± 2.5 | 326.7 ± 4.0 |
| Foreground TPOT p95 摘要 (ms) | 396.8 ± 25.0 | 350.9 ± 15.5 | 341.0 ± 7.6 |

Fixed-2048 在两方向使用 2048 tasks/wave 上限；Directional 扩容使用 4096、收缩使用 2048，并保留默认源；WeaveTP 在方向配置基础上对扩容启用成本筛选的等价源计划。后两者扩容同为 6 波，Fixed-2048 为 12 波。收缩三者均为 22 波，WeaveTP 收缩保留 default sources，candidate routing 关闭。

WeaveTP 扩容 switch wall 相对 Fixed-2048 降低 **38.7%**，相对 Directional 降低 **5.6%**。前者包含波次粒度变化的影响；后者是同波次上限下更接近源计划增量效果的对照，但不宜外推成任意机器上的普遍收益。

| 其余 switch wall (s) | Fixed-2048 | Directional | WeaveTP |
|---|---:|---:|---:|
| 收缩 | 12.130 ± 0.398 | 12.183 ± 0.128 | 12.291 ± 0.333 |
| 循环内每次切换均值 | 9.643 ± 0.256 | 8.414 ± 0.076 | 8.339 ± 0.198 |

收缩未观察到收益，WeaveTP 的均值反而略高。循环内每次切换均值相对 Fixed-2048 降低 13.5%，相对 Directional 仅降低约 0.9%。这些是描述性均值比较，不宣称统计显著。

## 历史三篇论文机制代理与普通基线

下表为每次 launch 内 4 次切换的平均值，再汇总 5 次独立 launch。它不是官方论文系统排名。

| 方法 | Base wall (s) | Transport (s) | Switch wall (s) |
|---|---:|---:|---:|
| WeaveTP（历史） | 8.787 ± 0.135 | 3.092 ± 0.084 | 9.585 ± 0.156 |
| Live-default | 8.769 ± 0.374 | 2.911 ± 0.149 | 9.542 ± 0.424 |
| Flying Serving-proxy | 约 0.000004907 | 0 | 0.790 ± 0.021 |
| AnchorTP-proxy | 19.472 ± 0.345 | 2.250 ± 0.162 | 20.255 ± 0.351 |
| Llumnix-proxy | 0.496 ± 0.066 | 0.045 ± 0.008 | 1.341 ± 0.185 |

WeaveTP 历史 switch wall 比 AnchorTP-proxy 低约 52.7%，但 Transport 反而高约 37.4%；相对 Live-default 的 switch wall 高约 0.5%。因此不能说“所有指标、所有基线都最好”。

Flying Serving-proxy 使用驻留/共享状态语义，Llumnix-proxy 只搬 KV，而 WeaveTP 和 AnchorTP-proxy 的状态范围不同。前两者更短不等于同工作量下更快，WeaveTP 也不能把与 AnchorTP-proxy 的差距写成对官方 AnchorTP 的已证实加速。Flying 代理没有可比的迁移重叠 TPOT 样本，Excel 记为“不适用”，不是 0 ms。

## 指标和统计口径

- **Base wall**：分波主体迁移阶段的墙钟时间，包含该阶段的执行/重叠过程，不只是链路传输。
- **Transport**：按记录累加各 wave 的 transport 时间，再按 launch 汇总；不等于全部重配置开销。
- **Switch wall**：完整 benchmark 切换的计时范围，包含切换后的数值检查，不等于用户可见停顿。
- **TPOT mean / p95**：迁移期间有限前台采样窗口的摘要，不是真实在线服务 SLO。正式数据对各切换的摘要作平均；历史批次按 overlap steps 加权。p95 摘要的平均不是所有 token 合并后的整体 p95。
- 正式方向统计先在每个 launch 内平均两次同方向切换，再对 3 个 launch 求均值和样本标准差；n=3，不是 6，也不是 rank/波次数。
- 循环统计先平均一个 launch 的四次切换，再跨 launch 汇总；不是四次时间之和，也不是整体任务 makespan。
- 三个计时范围重叠，**Transport、Base wall、Switch wall 不能相加**。
- 正式批次有 36 次按历史配置完成的 BF16-relative 数值检查；这不等于逐元素状态验证或逐 token 完全等价，也不能替代真实 GPU 强正确性工作。

旧 40.91→40.29 ms 的阻塞式局部测量、CPU 模拟、首次扩容 Transport 的独立试验，以及代理探索，不作为这里的正式主结果。Excel 保留两批核心数据及可复算的逐 launch 值，不混入不兼容口径。
