# TP Scale-out Resharding Demo 说明

这份说明对应 `tools/resharding/tp_build_src_dst_demo.py`，用于帮助理解这个 demo 如何模拟论文中 FluidScale 的 checkpoint-free elastic training 思路。

## 这个 demo 在验证什么

目标场景：

1. 原来训练只使用 2 张 GPU，模型采用 `TP=2`。
2. 训练过程中又加入 2 张 GPU。
3. 新配置改为 4 张 GPU 上的 `TP=4`。
4. 不保存 checkpoint，不从磁盘重新加载模型。
5. 只根据源模型和目标模型的参数视图重叠关系，直接做 GPU-to-GPU 参数迁移。

也就是说，这个 demo 验证的是：

```text
old active world:  ranks [0, 1],       TP=2
new shadow world:  ranks [0, 1, 2, 3], TP=4
```

迁移完成后，目标 `TP=4` 模型的 forward 输出必须和源 `TP=2` 模型一致。

## 和论文思路的对应关系

论文中的核心思想可以简化成三点：

1. **Abstract Resource View**
   把模型参数看成逻辑上的完整 tensor，而不是固定绑定到某个 rank 的物理内存。

2. **Intersection-based Transfer Planning**
   对每个参数，计算旧配置中每个 source rank 拥有的 tensor 区间，以及新配置中每个 destination rank 需要的 tensor 区间。两者的交集就是实际需要传输的切片。

3. **Streaming / GPU-to-GPU Resharding**
   不把完整参数聚合到单卡，也不写磁盘 checkpoint，而是让每个源 rank 直接把自己拥有的切片发送给需要它的目标 rank。

这个 demo 中的对应实现：

```text
build_centralized_reshard_plan(...)
    -> 根据参数 metadata 生成 source/destination view intersection plan

execute_reshard_plan(...)
    -> 按 plan 执行 send/recv

TracingCopyService
    -> 检查传输 tensor 是否仍在 CUDA 上，并统计 local/remote bytes
```

## 关键参数

最重要的新参数是：

```bash
--fluidscale-scaleout
```

它表示启用 FluidScale-style 的 scale-out 验证模式。

常用参数：

```bash
--src-tp 2
--dst-tp 4
--fluidscale-scaleout
--print-intersection-plan
--refit-backend nccl
```

含义：

```text
--src-tp 2                 旧 active world 的 TP degree
--dst-tp 4                 新 shadow world 的 TP degree
--fluidscale-scaleout      只在旧 rank 上构造 source model，在所有 rank 上构造 destination model
--print-intersection-plan  打印每个参数切片的迁移计划
--refit-backend nccl       使用 NCCL 做 GPU-to-GPU 传输
```

`--fluidscale-scaleout` 会自动启用 zero-copy GPU 检查，相当于要求迁移 payload 保持在 CUDA tensor 上。

## 运行命令

进入 Megatron-LM 仓库：

```bash
cd ~/Megatron-LM
export PYTHONPATH=$PWD:$PYTHONPATH
```

确认至少有 4 张 GPU：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

运行 demo：

```bash
torchrun --standalone --nproc_per_node=4 \
  tools/resharding/tp_build_src_dst_demo.py \
  --src-tp 2 \
  --dst-tp 4 \
  --fluidscale-scaleout \
  --print-intersection-plan \
  --intersection-plan-limit 80 \
  --refit-backend nccl
```

## 输出怎么看

### 1. Source 和 Destination 参数形状

你会看到 `SRC` 只在 rank 0 和 rank 1 上打印：

```text
===== SRC | global_rank=0 tp_rank=0/2 =====
...
===== SRC | global_rank=1 tp_rank=1/2 =====
...
```

这说明旧模型只存在于原来的 2 张 GPU 上。

然后 `DST` 会在 rank 0、1、2、3 上打印：

```text
===== DST | global_rank=0 tp_rank=0/4 =====
...
===== DST | global_rank=3 tp_rank=3/4 =====
...
```

这说明新 shadow model 已经覆盖 4 张 GPU。

### 2. View-overlap transfer matrix

示例输出：

```text
View-overlap transfer matrix for TP scale-out model weights:
  source_ranks=[0, 1] destination_ranks=[0, 1, 2, 3] total_bytes=131584 remote_bytes=96896
  src\dst              0           1           2           3
  0                32896       31104        1792           0
  1                    0        1792       31104       32896
```

这个矩阵表示每个 source rank 给每个 destination rank 发送了多少字节。

例如：

```text
src rank 0 -> dst rank 0: local copy
src rank 0 -> dst rank 1/2: remote GPU transfer
src rank 1 -> dst rank 2/3: remote GPU transfer
```

这正是 TP 从 2 扩到 4 时，旧分片被进一步切分并发送到新 TP rank 的过程。

### 3. Intersection transfer plan

示例：

```text
task=37 dst_rank=1 <- src_rank=0 remote bytes=4096
param=decoder.layers.0.mlp.linear_fc1.weight
src=[32:64, :] dst=[0:32, :]
```

含义：

```text
目标 rank 1 需要 linear_fc1.weight 的 [0:32, :] 这一段；
这段数据在旧 TP=2 布局中属于 source rank 0 的 [32:64, :]；
因此生成一个从 rank 0 到 rank 1 的 remote transfer task。
```

这就是论文中 view intersection 的具体体现。

### 4. Zero-copy GPU 检查

成功时会看到：

```text
Transfer trace for TP scale-out model weights (backend=nccl, transport=gpu-to-gpu, require_cuda=True):
  sends=112 recvs=112 local_bytes=69376 remote_bytes=193792 cpu_tensors_seen=0
  zero-copy GPU path check: passed (no CPU tensors submitted)
```

关键是：

```text
cpu_tensors_seen=0
zero-copy GPU path check: passed
```

这说明传输提交给 copy service 的 payload 没有落到 CPU。

### 5. 正确性检查

最终成功输出：

```text
After scale-out reshard max diff:  0.000000
Scale-out check passed:           True
```

这表示迁移后 `TP=4` 目标模型的 logits 和原 `TP=2` 源模型一致。

## 代码结构速览

主要函数：

```text
build_pg_collection(...)
    构造指定 TP/DP/PP/EP 配置的 ProcessGroupCollection。
    在 scale-out 模式下，source 只覆盖旧 active ranks。

print_view_overlap_matrix(...)
    汇总并打印 source rank 到 destination rank 的 view-overlap 迁移字节矩阵。

verify_fluidscale_scaleout(...)
    新增的主验证函数：
    1. 旧 rank 跑 source forward
    2. 构造 view intersection reshard plan
    3. 执行 GPU-to-GPU 参数迁移
    4. 迁移后跑 destination forward
    5. 比较 logits

TracingCopyService
    包装 nccl/gloo/nvshmem copy service，统计传输，并检查是否出现 CPU tensor。
```

## 注意事项

1. 这个 demo 不是完整训练系统。
   它模拟的是训练中某个 iteration boundary 上的参数迁移阶段。

2. 目前 `--fluidscale-scaleout` 只验证模型权重迁移。
   普通 AdamW 和 Megatron DistributedOptimizer 状态迁移仍保留在原来的 `--optimizer-state` 和 `--distributed-optimizer-state` 路径中。

3. 需要至少 4 张 GPU 来验证 `TP=2 -> TP=4`。

4. 推荐使用 `--refit-backend nccl`。
   `gloo` 会走 CPU-staged 路径，不符合 zero-copy GPU 迁移验证目标。

5. warning 通常不影响这个 demo。
   例如 `Transformer Engine and Apex are not installed`、`OMP_NUM_THREADS`、`CUDA_DEVICE_MAX_CONNECTIONS` 都不是本 demo 成败的关键。

## 一句话总结

这个 demo 用一个小 GPT 模型证明：当训练从 `TP=2` 扩容到 `TP=4` 时，可以不保存 checkpoint、不聚合完整参数，而是根据旧/新 TP 参数视图的几何交集，直接在 GPU 之间迁移所需切片，并得到完全一致的模型输出。

## Atomic Switch 验证

论文中的 Atomic Switch 指的是：Shadow World 准备好以后，所有 rank 在一致的 iteration boundary 上同步，然后把当前使用的 process group / model 引用从 Active World 切到 Shadow World：

```text
P_current <- P_shadow
```

这个 demo 现在在 elastic 路径中显式模拟这一步：

```text
Stable(G0): active_ranks=[...]
Shadow(G1): shadow_ranks=[...]
Consistent cut: waiting at iteration boundary after reshard.
Switch: P_current <- P_shadow; generation=G1
Next iteration uses ranks=[...]
Post-switch forward max diff: 0.000000
Atomic switch check passed: True
```

需要注意，demo 里的 switch 是 Python 层面的引用切换：

```text
current_model/current_ranks <- shadow_model/shadow_ranks
```

真正耗时的部分仍然是前面的 shadow model 构造和参数 reshard。Atomic switch 阶段只验证：迁移完成后，在一致边界上切到新 world，下一次 forward 使用新 TP 配置并保持输出一致。

## 通用弹性模式

如果不想写死是扩容还是缩容，可以使用：

```bash
--fluidscale-elastic
```

脚本会根据 `--src-tp` 和 `--dst-tp` 自动判断方向：

```text
dst_tp > src_tp  -> scale-out
dst_tp < src_tp  -> scale-in
dst_tp = src_tp  -> 报错，因为没有 TP degree 变化
```

扩容示例，不要求代码里写死 2/4，只要求 `torchrun` 启动的新 GPU 数等于目标 TP：

```bash
torchrun --standalone --nproc_per_node=8 \
  tools/resharding/tp_build_src_dst_demo.py \
  --src-tp 4 \
  --dst-tp 8 \
  --fluidscale-elastic \
  --print-intersection-plan \
  --refit-backend nccl
```

缩容示例，不要求代码里写死 4/2，只要求 `torchrun` 仍启动旧 active world 的 rank，让即将退出的 rank 有机会发送自己的分片：

```bash
torchrun --standalone --nproc_per_node=8 \
  tools/resharding/tp_build_src_dst_demo.py \
  --src-tp 8 \
  --dst-tp 4 \
  --fluidscale-elastic \
  --print-intersection-plan \
  --refit-backend nccl
```

注意：缩容验证模拟的是 graceful scale-in。也就是说，旧 rank 在退出前仍然在线，可以把自己持有的 TP 分片发给幸存 rank。如果 GPU 已经硬故障并彻底失联，这些独有分片已经不可读，单靠剩余 GPU 无法恢复完整模型，除非额外引入 checkpoint、冗余副本或容错机制。
