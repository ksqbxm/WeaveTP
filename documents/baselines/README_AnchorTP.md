# AnchorTP：状态保留/弹性恢复机制代理

## 原论文与为什么选

[AnchorTP: Resilient LLM Inference with State-Preserving Elastic Tensor Parallelism](https://arxiv.org/abs/2511.11617) 面向 GPU 故障后的弹性 TP 恢复，涉及状态保留、非等宽分片和最小迁移等机制。论文主页注明 DATE 2026。它与 WeaveTP 都关注参数/KV 以及弹性布局，但触发事件和具体协议不同。

## 本项目实际完成了什么

完成的是 **AnchorTP-proxy**：在同一 live benchmark 内进行参数与 KV 状态迁移，采用 residual 分波/带宽感知路径，并允许收缩方向的 aware 配置。它没有复现完整故障恢复服务、独立 daemon、非等宽弹性 TP 或论文全部 CMM 机制。

设备与设置：8 × RTX 5090、DeepSeek-V2-Lite、TP2↔TP4；历史批次 5 次独立 launch，每次 4 次切换。

## 已有结果

| 指标 | AnchorTP-proxy | WeaveTP（同一历史批次） |
|---|---:|---:|
| Base wall | 19.472 ± 0.345 s | 8.787 ± 0.135 s |
| Transport | 2.250 ± 0.162 s | 3.092 ± 0.084 s |
| Switch wall | 20.255 ± 0.351 s | 9.585 ± 0.156 s |

WeaveTP 的 switch wall 低约 52.7%，Transport 却高约 37.4%。该结果提示分波与重叠执行的整体开销不能由纯传输时间替代；并不证明 WeaveTP 对官方 AnchorTP 更快。不能把故障恢复与预先安排的 TP 切换当成同一工作负载。

## 继续复现的步骤

明确是否要比较“故障恢复”还是“计划重配置”，统一可用状态、设备预算和逻辑工作量，再决定是否值得补官方系统或忠实重实现。不要为保持全状态迁移叙事而忽略目标已有的有效数据。

证据：历史五轮的 5 份 AnchorTP 代理记录与当前配置/代理源码。原始数据已复算。
