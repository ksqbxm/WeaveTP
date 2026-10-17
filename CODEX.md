# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## 5. Failstop 每步交付长期规则

### 快速正确性测试（长期必需交付）

每个 step 除正式验收入口外，必须同时提供独立的快速正确性测试入口；后续每个 step 均遵守以下规则：

1. **时长**：服务器上含启动在内 5 分钟内完成。超出时减少轮数、层数或数据量，不得删除、跳过或放宽正确性检查项。需要对端时脚本自动等待；等待及超时必须明确记录，超时不能算通过。
2. **覆盖**：每类正确性检查至少执行一次。例如 Step 1 的 idle、inflight 各 1 轮；迁移类 step 缩容、扩容各 1 次，并执行逐位比较。快速测试不以性能或阶段耗时判定正确性，5 分钟要求是入口的运行预算。
3. **运行条件**：默认允许与其他任务共享 GPU，等同开启 `--allow-foreign-gpu-processes`；不放宽本测试进程身份、隔离、正确性及清理检查。仓库干净检查忽略未跟踪文件，仍检查已跟踪文件。结果目录自动带时间戳且不与历史结果冲突，无须手工修改 SUITE；两节点由脚本自动协调同一运行标识，不能各自用本地时间生成互不匹配的 SUITE。
4. **单命令使用**：每台机器只需一条命令，入口形式为 `bash "$REPO/code/current/tools/resharding/failstop/quick_check.sh" --step N --node SL3060`，另一节点使用 `--node SL3061`。两节点测试只需先后各执行一次，脚本自行等待对端。脚本内部完成 `source env.sh`、检出该 step 交付指定的完整 SHA、运行、打印汇总、打包结果，并在结束时（包括失败退出）切回原正式实验提交 `f7d555998b45756998465924d733fd70e553d6b4`。不得用浮动分支代替固定 SHA。
5. **输出与退出码**：结束时打印醒目的单行结论，例如 `QUICK CHECK step=N result=PASS (idle 1/1, inflight 1/1) shared_gpu_mode=true`。失败时打印失败检查项、所在轮次和日志路径；退出码 0/非 0 必须与 PASS/FAIL 一致。结果必须标记 `quick_mode=true`，不得作为正式验收结论；打包或清理等流程失败不能打印成功。
6. **交付顺序与离线分发**：交付说明最前面放快速测试命令，后面放正式验收命令。同时附 Windows 本地生成以 `f7d555998b45756998465924d733fd70e553d6b4` 为基准的增量 Git bundle、上传 SL3061、再转发 SL3060 的实际命令及校验/导入步骤。SL3060 不依赖访问 GitHub；必须使用该次交付的真实固定 SHA，不提供待填占位 SHA。

上述入口形式是长期实现要求，不表示脚本已经存在。每个 step 交付前必须实现并验证对应入口；尚未完成或服务器未验证的项目须如实注明，不能把规则文档当作执行证据。
