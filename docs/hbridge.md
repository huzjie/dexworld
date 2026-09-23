# H-Bridge 共享注意力

## 问题

视觉、触觉、动作是三个**异构模态**，特征分布差异大。简单拼接会让模型难以对齐
三者的语义关系。

## H-Bridge 设计

H-Bridge 让每个模态作为 **query**，去 attend 另外两个模态的 **key/value**，
实现跨模态信息交互：

- 视觉 attend 触觉 + 动作
- 触觉 attend 视觉 + 动作
- 动作 attend 视觉 + 触觉

融合结果再进入三专家门控，产出统一的联合表示。

## 代码

```python
from dexworld.encoders import HBridge

bridge = HBridge(dim=64, heads=8)
out = bridge.fuse(vision_emb, tactile_emb, action_emb)
out["joint"]   # 联合表示
out["vision"]  # 桥接后的视觉表示
```
