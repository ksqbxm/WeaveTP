# Step 2：公共故障接口、异构副本拓扑与模型进程组替换

状态：待实现。专用 GPU 测试标识：`step02_topology_rebind`。

## 1. 必读与前置

**需要阅读 [CODEX.md](../../CODEX.md) 进行代码编写，需要阅读[总计划](00_总计划.md)。** 修改前核对本 Step 与适用 AGENTS.md；Python imports 变动执行 `uv run isort`。本步与总计划不一致时，以最新用户确认为准并同步文档。

Step 1 交付后即可实施本步公共接口、配置、映射及 emulate，不等 real GPU 结果。real 适配器和模型真实重建集成必须等待 Step 1 GPU 回传及用户确认。两部分独立登记，不能用 emulate 验收证明 real。

## 2. 输入、范围与现有入口

从 `megatron/rl/parallel_utils.py` 的既有 `build_inference_pg_collection` 检查现有组语义，新增 `build_replica_pg_collections(replicas)` 支持显式 rank 列表，保留原入口行为。组创建及模型引用范围以源码核查为准，不做无关重构。

场景配置固定 EDP0=SL3061、EDP1=SL3060；正常两个 8 卡 TP4/ETP4×EP2，降级 EDP0 四卡 TP2/ETP2×EP2、EDP1 不变。故障坐标和 EP 专家归属遵循总计划，rank 不硬编码。每节点一个监管器、TCPStore 在 SL3060，回归由 SL3061 监管器拉起。

## 3. 实施任务

### 3.1 公共接口与状态

- 实现 `fail(ranks)` / `rejoin(ranks)` / `current_world()` / `rank_map()`，调用方共享同一 planner、迁移、KV、服务和统计路径。FAILSTOP_MODE 默认 emulate，不将业务流程复制成两套。
- 显式状态：normal、failed/recovering、degraded、rejoining/expanding、normal；控制事件带 launch、round、epoch 与唯一 id，防止旧消息用于新 world。
- `current_world` 表达有效 membership，`rank_map` 同时提供稳定设备身份、旧/新 world rank 和 replica/EP/ETP local rank；emulate 的进程仍在不代表该 GPU 可参与数据迁移。
- 建立 epoch/membership 变更的 planner 缓存、group 缓存及通信 work 生命周期失效规则；禁止复用旧 rank 的计划。

### 3.2 副本组构造与 local rank

- replicas 为显式 `(rank_list, tp, etp, ep)` 集合；所有参加构造的 world rank 以一致顺序创建组，记录组成员、顺序及后端。当前实际 world 的非副本成员仍遵循所用 PyTorch collective 建组合同。
- 同一副本 dense DP 的前向次序需与 EP collective 对齐；两个 EDP 的执行步数不强行绑定。具体异步执行驱动在 Step 4 完成。
- 在各 EP 内枚举本场景少量 local-rank 分配，按实际可复用权重字节最大选择，稳定设备顺序打破并列。原始源分片元数据不可因新 rank 更名而改写。
- 校验实际组 rank 顺序与预期映射；若 API 排序导致不同，以实际顺序更新显式映射并重新校验，不能错位读写 shard。

### 3.3 emulate 与 real 差异

emulate 可在启动时预建 normal/degraded/expanded 组。fail 后四失败 GPU 释放状态并从所有数据组隔离，`src_module=None`、`dst_module=None`；CPU 控制逻辑可继续工作。检查 NCCLCopyService 的 `ensure_all_ranks_participate` 一字节 ring：不能借占位让失败卡发送或接收数据，通信组仅含有效数据参与者。

real 使用 Step 1 已验收策略 abort/rebuild，不能预建未来 world 复用旧 communicator。用新 rank 编号参与 planner；12→16 时存活模型引用需替换到新进程组。所有旧 pending work 在组销毁前按协议终结并报告，不能拿新组 wait 旧句柄。

### 3.4 模型 pg_collection 重绑定

- 梳理模型及子模块保存的 pg_collection、TP/EP/DP 组、rank/world size、路由相关缓存，建立最小必要替换列表和验证日志。
- **替换组本身不重新分配现有权重。** 保留张量指针与源 TP4 切片元数据，不能重标元数据假装已经变成 TP2；目标 TP2/TP4 的必要迁移分配在 Step 3–4 执行。
- EDP1 张量与权重保持原样，仅 real world 重建后更新必要组引用；EDP0 survivor 旧分片也须保留。
- 回归 worker 以主机库元数据创建 TP4 空结构作为纯接收者，Step 3 主机库接好后验收；本步玩具测试用明确空目标验证协议，不把预填数据当作迁移成功。
- 组替换与模型前向之间设置明确安全边界，禁止前向线程使用已 abort 的组。

## 4. 本地验证

窄范围验证坐标→稳定身份→当前 rank，16→12→16 映射逆变换、local rank 最优复用及并列规则、组创建顺序、None 模块、失效缓存、不接受旧 epoch。对公共接口用 fake backend 测状态机，故障卡不得进入迁移候选/目标或占位 ring。

对原 homogeneous 入口跑相关既有回归，确认未启用 fail-stop 时行为不变。模型引用替换测试检查张量指针而不只检查数值，以发现意外重分配。

## 5. 专用服务器 GPU 验收

使用 16 卡玩具 MoE/张量布局，emulate 先验收；real 在 Step 1 用户放行后以相同逻辑快照补测：

1. 打印三种状态所有 TP/EP/DP 组、实际顺序及映射，验证每组 collective 结果和成员身份。
2. 故障后数据通路完全排除目标 GPU；emulate 失败 PID 可存活但零发送/接收字节，real 四 PID 已退出。
3. 重绑定前后存活源张量指针和逐位内容不变；EDP1 原权重不变。后续新组算子正确，无旧组引用被调用。
4. 同副本空 DP/非空 DP 前向的 EP collective 顺序一致；不以两 EDP 全局 barrier 掩盖次序错误。
5. 回归四个空目标没有被当作源；真实路径新 PID、rank map 与 group membership 一致。

通过标准：对应模式所有检查成功，超时/挂起/错组/失败 GPU 字节非零均失败。分别输出 emulate 和 real 结果；real 未放行时标记 pending，不标记 skipped-pass。

## 6. 交付与后续

交付场景配置、故障后端、PG 构造/重绑定、映射记录、独立 GPU runner 与报告，输出 `/data/.../step02_topology_rebind/<mode>/<launch_id>/`。附逐节点完整服务器命令、SHA、CPU/GPU 原始结果、打包与切回 `f7d5559` 命令，遵循总计划交付合同。

公共/emulate 交付后可继续 Step 3；real 部分必须有 Step 1 放行记录。Step 4 仍等待 fix2 GPU 复验，不因本步通过而自动越过。

## 7. 验收登记

公共/emulate SHA 与 CPU：待填。emulate GPU：待运行。Step 1 real 放行：待确认。real SHA / GPU：待填、待运行。
