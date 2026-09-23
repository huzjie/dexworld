# -*- coding: utf-8 -*-
"""dexworld — 多模态触觉世界模型（Tactile World Model）训练与推理框架。

灵感来自 2026-09-23 理想汽车 Foundation Model 团队发布的 **ME-Dex-1.0** 世界动作
触觉模型。ME-Dex 的核心思想是：把**触觉**与**视频**共同当作「需要预测的未来世界
状态」，用一个统一的扩散模型**联合去噪**视频、触觉与动作序列，从而让模型不仅能
「看」到未来，还能「摸」到未来——这对具身智能在接触丰富（contact-rich）的灵巧
操作任务上尤为关键。关键技术创新点：

- 双手 34 区域触觉模板：把异构触觉数据统一映射到左右手各 17 区（5 指 x3 指节 +
  掌心 2 区）的标准模板，解决不同传感器/机器人之间的数据异构问题
- 视觉 / 触觉 / 动作三专家架构：三个模态专家各自编码，避免模态间相互干扰
- H-Bridge 共享注意力：一个跨模态的桥接注意力，让视觉、触觉、动作 token 在同一
  语义空间对齐，实现「看见 + 摸到 + 动作」的联合建模
- 联合去噪扩散：单个扩散过程同时对视频、触觉、动作序列去噪，天然保持时间一致性
- 自主式触觉数据引擎：用「模拟器 + 自主策略 + 数据扩充」闭环扩充仿真触觉数据

本仓库提供一个**本地可运行、零重型依赖**的开源实现，把「触觉世界模型」从论文
概念落地为一套完整的训练 + 推理 + 规划框架：

- 数据层：34 区域触觉模板 / 传感器抽象（GelSight、触觉阵列、力扭矩）/ 异构映射 /
  仿真数据引擎 / 数据集与加载器
- 编码层：视觉 / 触觉 / 动作三专家编码器 + H-Bridge 共享注意力 + 联合 tokenizer
- 模型层：扩散去噪器 + 世界模型（未来状态预测 / 动作条件生成 / 触觉反馈预测）
- 训练层：联合去噪损失 + 训练器 + 优化器 / 调度器 + 检查点 / 断点续训
- 推理层：扩散采样 / 世界模型 rollout / MPC 规划 / 触觉反馈预测
- 评测层：预测精度 / 触觉重建误差 / 规划成功率 + 基准与报告
- 后端层：零依赖 Mock（可训练、确定性）+ OpenAI 兼容后端
- 服务层：零依赖 HTTP 服务 / 可观测 / CLI / 端到端编排 Pipeline
- 部署层：Docker / docker-compose / Kubernetes / Helm + GitHub Actions CI

最小可跑示例（零依赖，Mock 后端演示真实「预测误差随训练下降」曲线）：

    from dexworld import DexWorldPipeline, TrainConfig
    cfg = TrainConfig(model={"backend": "mock"}, train={"epochs": 3})
    result = DexWorldPipeline(cfg).run()
    print(result.summary())

模块总览：
- version / errors / constants / types / config       基础
- utils                                               工具（含零依赖 YAML 解析）
- data                                                数据（34 区模板 / 传感器 / 仿真引擎 / 数据集）
- encoders / model                                    编码器 / H-Bridge / 扩散 / 世界模型
- train / inference / eval                            训练 / 推理 / 评测
- backends / serving / cli / pipeline                 后端 / 服务 / CLI / 编排
"""

from ._version import __version__, __author__
from .config import DexWorldConfig, TrainConfig
from .pipeline import DexWorldPipeline, PipelineResult
from .model.world_model import DexWorldModel

__all__ = [
    "__version__", "__author__",
    "DexWorldConfig", "TrainConfig",
    "DexWorldPipeline", "PipelineResult", "DexWorldModel",
]
