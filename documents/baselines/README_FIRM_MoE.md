# FIRM-MoE：论文机制重实现的小型验证

## 当前身份

依据 [FIRM-MoE: Fine-Grained Expert Decomposition for Resource-Adaptive MoE Inference](https://ojs.aaai.org/index.php/AAAI/article/view/39106) 进行独立机制重实现。收集材料时未取得官方仓库，因此不标为 official reproduction；这不等于断言官方代码永远不存在。

## 已经实现和运行的范围

实现 MoL 交集预测、组件级预取、LRU 缓存和 HEOP 分组坐标搜索，并有确定性路由 trace 下的 CPU 小型输出。它验证的是算法机制和记录流程，不包含官方 GPU runtime、真实模型 checkpoint 或完整端到端推理。

现有输出包括 latency units、缺失次数、命中次数等 trace 指标。latency units 不是秒或毫秒，不能改名为推理时延，也不能与 DeepSeek 三轮实验合并。

## 运行与验证步骤

先对齐论文的缓存准入/淘汰、预取语义、内存预算和路由工作负载，再接入真实模型与 GPU 推理。最终表格仍应标为 reimplementation，除非后续确实采用并核对官方实现。

当前验证状态：**已有 CPU 算法级小型验证；没有原生 GPU 性能复现结果。**
