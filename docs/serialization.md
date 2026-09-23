# 序列化与持久化

```python
from dexworld.utils.serialization import serialize_model, deserialize_model
from dexworld.utils.io import save_json, load_json

state = serialize_model(model)
save_json(state, "checkpoints/model.json")

loaded = load_json("checkpoints/model.json")
deserialize_model(model, loaded)
```

模型状态（skill / memory）可完整序列化到 JSON，支持断点续训。
