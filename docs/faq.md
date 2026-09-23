# FAQ

## Q: 必须装 numpy 吗？
不必须。numpy 可选，缺失时自动回退纯 Python 实现。

## Q: 为什么用 md5 而非 hash() 做种子？
`hash()` 受 `PYTHONHASHSEED` 影响，跨进程不一致；md5 稳定，保证确定性可复现。

## Q: 如何接入真实模型？
实现一个后端（参考 `backends/mock.py`），或通过 `openai_compat` 后端接语义编码。

## Q: 如何替换仿真数据为真实数据？
把真实触觉数据映射到 34 区域模板（`HeteroMapper`），构造成 `WorldTrajectory`，
再喂给 `Trainer.fit()`。

## Q: 支持哪些 Python 版本？
Python 3.8+，建议 3.10+。
