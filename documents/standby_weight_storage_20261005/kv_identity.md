# KV request identity（独立默认关闭功能）

基准：`f7d555998b45756998465924d733fd70e553d6b4`。

`KV_REQUEST_IDENTITY=1` 向 live benchmark 传入 `--live-kv-request-identity`。
开启后，以 TP4 DP ordinal 划分本 benchmark 的 synthetic request domains；同一
domain 的 TP2/TP4 副本使用相同 token，KV 名称包含 request identity。权重仍可跨
domain 选源。真实请求调用者应传入真实的 request/batch identity，不能沿用此合成映射。

关闭时不构造 domain 映射、不添加 collective，使用基准 KV 名称及 token 公式，
result.json 不添加 kv_request_domains。未提供/非法的显式请求 identity 不进行猜测或回退；
wrapper 的显式 request_id=None 表示关闭域隔离。

CPU 验证（Windows，PyTorch CPU，GPU 未运行）：

```powershell
python -B -m pytest -q -p no:cacheprovider --confcutdir=tests/unit_tests/resharding tests/unit_tests/resharding/test_live_defaults.py tools/resharding/correctness/test_correctness.py
```

33 项测试与 82 个子测试通过。测试包含真实中央 planner、双向迁移、跨请求错误注入、
缺失来源拒绝，以及从冻结基准 git blob 提取的默认 token AST、结果字典 AST、wrapper
名称/别名/分片属性与实际 token 数值对照。CPU mailbox 不代替 NCCL/GPU 验证。
