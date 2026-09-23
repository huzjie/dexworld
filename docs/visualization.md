# 可视化

dexworld 内置零依赖 ASCII 可视化，无需 matplotlib：

```python
from dexworld.visualization import (
    render_tactile_heatmap, render_vision_ascii, render_timeline,
)

print(render_tactile_heatmap(obs, "left"))   # 触觉热图
print(render_vision_ascii(frame))            # 视觉帧
print(render_timeline(traj))                 # 轨迹时间线
```

示例见 `examples/demo_visualize.py`。
