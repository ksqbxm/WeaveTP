# ExpertFlow：官方复现准备

## 当前身份

已收集 [官方实现](https://github.com/expertflow-dac/expertflow)，固定版本为 05fe01591c0abe7b0ac0669ad917b4b2484d98e5。已有环境与原生小型验证入口；当前材料不能确认完整官方性能复现完成。

其原生验证路线使用 Switch-32、WMT16 路由数据和匹配的 RPP checkpoint，以检查模型构造、offload、预测和生成。现有小型用例设置为 batch size=2、两批测量；这是准备好的用例，不是已确认执行的性能结果。

## 与 WeaveTP 的关系

ExpertFlow 关注专家预测/调度和 offloading 路径，不能直接等同于 TP 重分片机制。早期自有 benchmark 的 ExpertFlow 策略代理不属于官方实现。DeepSeek-V2-Lite 适配应另列实验，不能冒充论文的 Switch 配置。

## 运行与验证步骤

先补原生 Switch 路径的运行证据，核对模型、路由输入、预测器与显存预算，再考虑正式测量及 DeepSeek 适配。可用服务器为 RTX 5090 平台，但没有可据此归属的原生性能结果。

当前验证状态：**官方源码和小型用例已准备，原生性能结果尚未确认。**
