# Step 1 固定提交与离线分发

固定实现 SHA：`d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c`。先分发，再按 [共享 GPU 运行命令](Step1_共享GPU服务器命令.md) 启动两节点。SL3060 不访问 GitHub。后续文档提交不改变实现 SHA。

使用以原正式实验提交 `f7d555998b45756998465924d733fd70e553d6b4` 为基线的增量 bundle，避免重复打包仓库历史占用 `/data`。两台原实验仓库须有该基线及其祖先；`git bundle verify` 会检查依赖，失败则停止并回传输出，不盲目 checkout。分发不占用 GPU。

## SL3061：从 GitHub 拉取、打包并发送

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
SHA=d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c
BASE=f7d555998b45756998465924d733fd70e553d6b4
git fetch https://github.com/ksqbxm/WeaveTP.git codex/failstop
git cat-file -e "$SHA^{commit}"
git merge-base --is-ancestor "$BASE" "$SHA"
git checkout --detach "$SHA"
git update-ref refs/heads/codex/step1-d16b5e8 "$SHA"
cd /data/ubuntu/lxh/weavetp
BUNDLE=step1_d16b5e8.bundle
test ! -e "$BUNDLE"
git -C "$REPO" bundle create "$PWD/$BUNDLE" refs/heads/codex/step1-d16b5e8 "^$BASE"
git -C "$REPO" bundle verify "$PWD/$BUNDLE"
sha256sum "$BUNDLE" > "$BUNDLE.sha256"
ls -lh "$BUNDLE" "$BUNDLE.sha256"
scp "$BUNDLE" "$BUNDLE.sha256" 10.60.14.1:/data/ubuntu/lxh/weavetp/
```

`10.60.14.1` 是现有 SL3060 控制地址；scp 使用当前登录用户名。如果两节点 SSH 用户不同，只需在地址前加实际 `用户名@`。若 bundle 文件已存在，先核实对应 SHA 和校验结果，不覆盖旧证据。

## SL3060：校验并导入本地 bundle

```bash
set -e
cd /data
mountpoint -q /data
source /data/ubuntu/lxh/weavetp/env.sh
cd /data/ubuntu/lxh/weavetp
sha256sum -c step1_d16b5e8.bundle.sha256
cd "$REPO"
test -z "$(git status --porcelain --untracked-files=no)"
git bundle verify /data/ubuntu/lxh/weavetp/step1_d16b5e8.bundle
git fetch /data/ubuntu/lxh/weavetp/step1_d16b5e8.bundle refs/heads/codex/step1-d16b5e8
git checkout --detach d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c
test "$(git rev-parse HEAD)" = d16b5e80cbc6f7f4cf7edcf1494b16f73a4bbe6c
```

然后先启动 SL3060，再在 10 分钟内启动 SL3061；两边使用相同 suite id，并同时加 `--allow-foreign-gpu-processes`。完整预检、nohup、状态查询、证据打包及恢复原提交命令见 [共享 GPU 运行命令](Step1_共享GPU服务器命令.md)。50 项 CPU 测试通过，服务器 GPU 验收尚未执行。
