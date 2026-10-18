# Claude Code 指令补充 v2（2026-10-18，叠加在 `CLAUDE_CODE_指令_扩缩容端到端最终版_20261017.md` 与 `扩缩容端到端_实施计划修订_20261018.md` 之上）

以下条目覆盖前两份文件中冲突的内容；其余不变。

1. **数据集改用 `最终数据集定稿_v2_4k分段_20261018.md`**：长段 353 条 / 3800–4200；全部分阶段到达；图 1 短段条数不变（8k–80k）、图 2 每短段输入总量 6.5M × 长段数 ÷ 短段数；实测 1 个种子。8k 版只作模拟对照，不上机。
2. **W5 数据集生成器**：直接 `import` 仓库内 `documents/e2e_prep_20261008/tools/longout_profile.py` 的 `extract()` / `rows()` 做长池解析，不重写；源目录名用 `s1k_1.1`（不是 `s1k`）；每源取前 3,000 条可解析样本；硬校验数字按定稿 v2 §3（精确，不设容差；4k 池 ±2%）。先 `ls /data/ubuntu/lxh/weavetp/e2e_prep/data/` 确认五个目录存在，缺哪个再按 `longout_profile.py` 的 `CANDS` 下载。
3. **运行单元**：`--e2e-segments k..m` 之外增加 `--e2e-segment-slice i/n`（段内按条数均分切片，每片独立运行、开始与结束时系统为空）；每个单元结束立刻写 `result.json` + `complete.json`，结果带 `(workload_id, segment, slice)`；模型加载时间单独记，不计入段时长。
4. **汇总器**：按 `(workload, segment, slice)` 拼接 `T_point = Σ T_segment + N_switch × T_switch`，`T_switch` 取 calib 单元按方向与 KV 水位的实测；打印每个点的拼接来源清单；P 段线性外推时同时输出 9k / 18k 两档的单位时间与偏差。
5. **运行模式**：所有出数字的单元用 `--e2e-run-mode timing`；逐位 / 容差校验只在 S1 做。
6. **图 1 不再有连续到达版本**；唯一的连续到达运行是 `repr_4k`（只跑 WeaveTP，时间序列开）。
7. **机时**：任何单元超过 25 min（不含加载）即视为配置错误，先切片再跑，不要排长队。
