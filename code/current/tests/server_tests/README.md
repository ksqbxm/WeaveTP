# 真实服务器验收

这些入口按项目隔离，直接用指定服务器的既有 Python 执行，不通过仓库全局 pytest/conftest 收集，不自动发现机器或申请 GPU。

- [WeaveTP 双机 16 卡](weavetp_16gpu/README.md)：T01–T10 覆盖矩阵、运行顺序、独立证据目录及当前阻断。

可移植的 CPU/mock 回归留在 `tests/unit_tests/resharding/`；服务器入口、只读输入清单和测试替身留在本目录；执行结果不提交到 tests，服务器全部写 `/data`，本地审阅证据写项目根目录的 documents。
