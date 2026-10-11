# DP sweep：服务器命令（仅文档，未执行）

本次交付仅做 CPU 验证。以下命令由用户后续在服务器终端运行；没有通过此文档自动连接服务器。两节点使用同一交付 SHA、相同 Python 环境、同一份对应 W 的画像；E 还需相同真实 checkpoint。

先将 [README 公共配置](../../code/current/tools/resharding/README_dp_sweep.md#公共配置) 的完整 Bash 段保存为两台机器上的 `/data/ubuntu/lxh/weavetp/dp_sweep_20261011/dp_sweep_env.sh`。下列路径沿用 T09 部署示例，运行前替换为实际路径。每个 launch 使用新 r 目录，保留失败目录和原始日志。

## 8+k 示例：S / W10 / weavetp / r1

公共配置已加入 `export INIT_SYNC_MAX_TASKS=2048`；更新已保存的 `dp_sweep_env.sh` 后再 source，S/E 都使用该初始同步批大小，正式迁移波数配置不变。

节点 0 的终端：

```bash
cd /data/ubuntu/lxh/weavetp/WeaveTP/code/current
source /data/ubuntu/lxh/weavetp/dp_sweep_20261011/dp_sweep_env.sh
export NODE_RANK=0 NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

节点 1 的终端：

```bash
cd /data/ubuntu/lxh/weavetp/WeaveTP/code/current
source /data/ubuntu/lxh/weavetp/dp_sweep_20261011/dp_sweep_env.sh
export NODE_RANK=1 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1
```

两边分别运行下列相同段；共享 MASTER_ADDR/MASTER_PORT，等待双方这一 launch 结束后再启动下一组：

```bash
export PYTHON=/data/ubuntu/lxh/weavetp/envs/megatron/bin/python
export EXPERIMENT=S METHOD=weavetp W=10 REPEAT=1
export NNODES=2 WORLD_SIZE_OVERRIDE="$W" MASTER_ADDR=10.60.14.1 MASTER_PORT=29500
export NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=eno1np0 GLOO_SOCKET_IFNAME=eno1np0
export NCCL_IB_HCA=mlx5_0 NCCL_IB_DISABLE=0
unset TORCH_ELASTIC_WORKER_IDENTICAL
dp_sweep_config "$EXPERIMENT" "$METHOD" "$W"
export PROFILE="/data/ubuntu/lxh/weavetp/profiles/dp_sweep/W$W/profile.json"
export OUT_DIR="/data/ubuntu/lxh/weavetp/formal/dp_sweep_20261011/$EXPERIMENT/$METHOD/W$W/r$REPEAT/node$NODE_RANK"
test -f "$PROFILE" && test ! -e "$OUT_DIR" || exit 2
bash tools/resharding/run_deepseek_v2_lite_live_benchmark.sh
```

W=10 时两边的 WORLD_SIZE_OVERRIDE 都是 10，GBS 自动为 10，16×16 画像门禁不触发。它只影响 launcher 的总卡数判断/GBS；torchrun 的实际进程数仍由每个节点的 NPROC_PER_NODE 和 NNODES 决定。不要把两边都设成 8，也不要启用假定相同 worker 数的 `TORCH_ELASTIC_WORKER_IDENTICAL=1`。

| W | NODE_RANK=0 的 NPROC_PER_NODE | NODE_RANK=1 的 NPROC_PER_NODE | 节点 1 CUDA_VISIBLE_DEVICES |
|---:|---:|---:|---|
| 10 | 8 | 2 | 0,1 |
| 12 | 8 | 4 | 0,1,2,3 |
| 14 | 8 | 6 | 0,1,2,3,4,5 |
| 16 | 8 | 8 | 0,1,2,3,4,5,6,7 |

分别令 METHOD 为 megatron_default、weavetp、weavetp_2x，REPEAT 为 1/2/3（或更多），W 取表中值。两边逐 launch 使用相同实验、方法、W、r、端口、画像。W≤8 的 S 使用单节点，NPROC_PER_NODE=W、NNODES=1、NODE_RANK=0，按 W 数量列出 CUDA_VISIBLE_DEVICES；卡数为 2/4/6/8。完整 S 汇总只读 node0 的第二次方向 2->1。

## E：真实 27 层 checkpoint

E 只跑 megatron_default、weavetp，W=4/8/12/16；weavetp_2x 的有效扩容配置与 weavetp 相同，不重复运行。W=12/16 按上表启动两个节点，W=4/8 单节点。在前述共享段中使用以下设置替换实验变量；两边都设置 CHECKPOINT，然后继续执行 dp_sweep_config 及后续 launch 命令：

```bash
export CHECKPOINT=/data/models/DeepSeek-V2-Lite-megatron-v2
export EXPERIMENT=E METHOD=weavetp W=12 REPEAT=1
export NNODES=2 WORLD_SIZE_OVERRIDE="$W"
dp_sweep_config "$EXPERIMENT" "$METHOD" "$W"
# 得到 SRC_TP=2 DST_TP=4 SKIP_CHECKPOINT=0 NUM_LAYERS=27 SWITCHES=1。
# 上表 W12：节点 0 NPROC_PER_NODE=8，节点 1 NPROC_PER_NODE=4。
```

## 汇总（CPU）

将各方法、W、r 的 node0 结果放在同一根目录下，保留 `<method>/W<卡数>/r<轮次>/node0/result.json` 层级。配对 r 命名相同，至少两轮才定义样本标准差。

```bash
cd /data/ubuntu/lxh/weavetp/WeaveTP/code/current
PYTHON=/data/ubuntu/lxh/weavetp/envs/megatron/bin/python
"$PYTHON" -B tools/resharding/summarize_dp_sweep.py \
    --root /data/ubuntu/lxh/weavetp/formal/dp_sweep_20261011/S \
    --direction '2->1' --methods megatron_default,weavetp,weavetp_2x
"$PYTHON" -B tools/resharding/summarize_dp_sweep.py \
    --root /data/ubuntu/lxh/weavetp/formal/dp_sweep_20261011/E \
    --direction '2->4' --methods megatron_default,weavetp
```

汇总的非零退出须结合报告解读。尤其保留原算法缩容的 baseline/FIFO 标签、实际感知选源几何及全局 gate 证据；不为通过候选/执行模式断言而改标签或门槛。主指标、Base-only transport、波数容差、显存统计区间及不能将波数收益全部归因选源的限制见 README。
