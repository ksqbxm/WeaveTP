# Llumnix：请求/KV 迁移机制代理

## 原论文与为什么选

[Llumnix: Dynamic Scheduling for Large Language Model Serving](https://www.usenix.org/conference/osdi24/presentation/sun-biao) 是 OSDI 2024 的动态服务调度工作，在实例间移动请求及其状态。选择它是为了对照请求级 KV 迁移与动态 TP 状态准备的边界，而不是因为它原生实现了本项目的 TP2↔TP4 协议。

## 本项目实际完成了什么

完成的是 **Llumnix-proxy**：在目标权重已驻留的前提下，只迁移稳定 KV 前缀和必要增量，不迁移参数。它没有复现 Llumnix 的完整请求调度、负载均衡、隔离与 SLO 管理系统。

设备与设置：8 × RTX 5090、DeepSeek-V2-Lite、TP2↔TP4；历史批次 5 次独立 launch，每次 4 次切换。TP 改变是本项目 harness 的设置，不能写成 Llumnix 原系统的能力。

## 已有结果

Base wall 0.496 ± 0.066 s；Transport 0.045 ± 0.008 s；Switch wall 1.341 ± 0.185 s。数值较小与仅搬 KV 的状态范围有关，不是与 WeaveTP 全状态路径等工作量的速度比较。

其 TPOT 采样窗口较短；均值与 p95 摘要相同不能解释为真实服务尾延迟稳定。switch wall 同样包含本 benchmark 的数值检查。

## 继续复现的步骤

如需原生服务比较，应先运行官方系统的请求迁移工作负载，再统一模型、请求集合、目标权重驻留条件和评价指标。现有结果只用于机制范围说明。

证据：历史五轮的 5 份该方法记录与 KV-only 分支；原始数据已复算。官方代码由论文页面链接的 AlibabaPAI/Llumnix 项目提供。
