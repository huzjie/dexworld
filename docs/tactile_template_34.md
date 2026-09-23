# 双手 34 区域触觉模板

## 为什么需要模板

不同机器人、不同传感器（GelSight 视触觉、触觉阵列、力/扭矩传感器）采集的触觉
信号形态各异、维度不一。直接喂给模型会导致**数据异构**问题，模型无法泛化。

ME-Dex 的做法是：把所有触觉数据统一投影到一个**标准模板**上。

## 模板定义

- 每只手 **17 区** = 5 根手指 × 3 个指节（15）+ 掌心近端 / 远端（2）
- 双手合计 **34 区**

```
左手（17）                右手（17）
├─ thumb   远端/中段/近端   ├─ thumb   远端/中段/近端
├─ index   远端/中段/近端   ├─ index   远端/中段/近端
├─ middle  远端/中段/近端   ├─ middle  远端/中段/近端
├─ ring    远端/中段/近端   ├─ ring    远端/中段/近端
├─ pinky   远端/中段/近端   ├─ pinky   远端/中段/近端
├─ palm_distal             ├─ palm_distal
└─ palm_proximal           └─ palm_proximal
```

## 使用

```python
from dexworld.data.tactile_map import TactileMap

tm = TactileMap()
tm.index("left_thumb_distal")   # 0
tm.index("right_palm_proximal") # 33
tm.left_indices()               # 左手区域索引
```

## 异构映射

`HeteroMapper` 负责「传感器读数 -> 区域特征」的投影。每个机器人末端配置一组
传感器，通过 `region_assignment` 把传感器读数映射到对应区域。
