# Flying Serving：驻留状态机制代理

## 原论文与为什么选

[Flying Serving: On-the-Fly Parallelism Switching for Large Language Model Serving](https://arxiv.org/abs/2602.22593) 研究在线 DP↔TP 切换，包含驻留权重视图、KV adaptor 和预初始化通信器等机制。它帮助说明“目标已经拥有可用权重/状态”与“需进行状态重分片”的差异。论文主页注明 ICS 2026。

## 本项目实际完成了什么

完成的是 **Flying Serving-proxy**，不是官方 vLLM 系统的完整复现。它在 WeaveTP 的 resident-layout benchmark 中采用驻留/共享状态语义，跳过参数和 KV 的常规 base-copy 工作。入选记录中没有主体迁移波次，delta 长度为 0；代码仍保留通用 delta 处理，不能推导任意请求都无需补齐 KV。其意义接近理想驻留状态参考，不是对 Flying Serving 官方开销的测量。

设备与设置：8 × RTX 5090、DeepSeek-V2-Lite、TP2↔TP4；历史批次 5 次独立 launch，每次 4 次切换。重叠迁移阶段没有可比的前台 TPOT 样本，记“不适用”。

## 已有结果

Switch wall 0.790 ± 0.021 s；Transport 为 0；Base wall 约 4.907 微秒。switch wall 仍包含 benchmark 的切换后数值检查。Transport 为 0 只说明本代理所选状态范围，不意味着官方系统零成本、零停顿。

## 不应如何表述

不能写成“已在 DeepSeek 上完整复现 Flying Serving”，也不能直接拿这个数字与全状态路径作同工作量性能排名。

## 继续复现的步骤

若需要原生系统对比，应使用其原生执行栈，并核对模型支持、权重视图及 KV adaptor 的真实语义、请求状态、部署布局和计时边界。现有代理结果可以保留为机制范围对照。

证据：历史五轮 2026-08-28—08-31 的 5 份该方法记录；现有代理分支源码。原始数据已复算。
