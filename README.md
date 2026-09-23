# dexworld · 多模态触觉世界模型训练与推理框架

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Zero Deps](https://img.shields.io/badge/Dependencies-0-orange.svg)](#零依赖)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-brightgreen.svg)](.github/workflows/ci.yml)

**把触觉和视频共同当作「需要预测的未来」，一个模型同时「看见」并「摸到」未来。**

受 2026-09-23 理想汽车 Foundation Model 团队开源的 **ME-Dex-1.0 世界动作触觉模型** 启发。

</div>

---

## 这是什么

`dexworld` 是一个**本地可运行、零重型依赖**的多模态触觉世界模型（Tactile World
Model）训练与推理框架。它把「触觉世界模型」从论文概念落地为一套完整工程：

- **双手 34 区域触觉模板**：统一异构触觉数据（GelSight / 触觉阵列 / 力扭矩）
- **视觉 / 触觉 / 动作三专家架构**：三模态各自编码，互不干扰
- **H-Bridge 共享注意力**：跨模态语义对齐
- **联合去噪扩散**：单扩散过程同时去噪视频 + 触觉 + 动作，保持时间一致
- **自主式触觉数据引擎**：仿真 + 自主策略闭环扩充触觉数据

传统世界模型只看「视觉未来」，但在抓取、装配这类接触丰富任务里，视觉常被遮挡，
**触觉才是关键**。dexworld 让模型能预测「手会摸到什么」。

## 快速开始

```bash
git clone https://github.com/huzjie/dexworld.git
cd dexworld

# 健康检查（零依赖）
python -m dexworld.cli doctor

# 完整训练（零依赖 Mock 后端，演示误差随训练下降）
python -m dexworld.cli train --config config.example.yaml
```

最小代码：

```python
from dexworld import DexWorldPipeline
from dexworld.config import DexWorldConfig

cfg = DexWorldConfig()
cfg.model.backend = "mock"
cfg.train.epochs = 4
result = DexWorldPipeline(cfg).run()
print(result.summary())
```

## 功能特性

- ✅ 34 区域触觉模板 + 异构传感器映射
- ✅ 三专家编码 + H-Bridge 共享注意力
- ✅ 联合去噪扩散世界模型（可训练、确定性）
- ✅ 自主式触觉仿真数据引擎 + 数据增强 + 经验回放
- ✅ MPC / Beam / CEM 规划器（用世界模型预测触觉接触质量选最优动作）
- ✅ 课程学习 + 集成预测 + 消融实验 + 交叉验证
- ✅ 完整评测（预测精度 / 触觉重建误差 / benchmark）
- ✅ ASCII 可视化（触觉热图 / 视觉帧 / 轨迹时间线）
- ✅ 零依赖 Mock 后端（无网络也能跑通闭环）
- ✅ OpenAI 兼容后端（可接 MiMo / GLM 等）
- ✅ CLI + HTTP 服务 + Docker / K8s / Helm / CI

## 架构

```
Pipeline（编排） → serving/cli → inference(rollout/MPC/预测) + eval
                    ↑                    ↑
              model(三专家+H-Bridge+联合扩散+世界模型)
                    ↑                    ↑
              encoders(三专家+tokenizer)  train(损失/训练器/优化器)
                    ↑
              data(34区模板/传感器/异构映射/仿真引擎)
                    ↑
              backends(Mock可训练/OpenAI兼容)
                    ↑
              core(types/config/constants/errors/utils)
```

详见 [docs/architecture.md](docs/architecture.md)。

## 目录结构

```
dexworld/
├── dexworld/
│   ├── core         # types / config / constants / errors
│   ├── utils        # logging / seed / registry / tensor / yamlish / profiler
│   ├── data         # 34 区模板 / 传感器 / 异构映射 / 仿真引擎 / 增强 / 回放 / 课程
│   ├── encoders     # 三专家编码 + H-Bridge + tokenizer + 位置编码
│   ├── model        # 三专家混合 / 扩散 / 世界模型 / 残差
│   ├── train        # 损失 / 训练器 / 优化器 / 调度 / 分布式 / 检查点
│   ├── inference    # rollout / MPC 规划 / 触觉反馈预测 / 集成
│   ├── planner      # 候选生成 / CEM 规划
│   ├── eval         # 指标 / benchmark / 消融 / 交叉验证 / 报告
│   ├── visualization# ASCII 热图 / 视觉帧 / 时间线
│   ├── backends     # Mock / OpenAI 兼容 / LRU 缓存
│   ├── serving      # HTTP 服务 / 可观测
│   ├── pipeline.py  # 端到端编排
│   └── cli.py       # 命令行
├── examples/        # 演示脚本（14 个）
├── tests/           # 单元测试（40 个，unittest）
├── docs/            # 中文文档 + 模型卡（20+ 篇）
├── deploy/          # Docker / K8s / Helm
├── .github/         # CI
└── config.example.yaml
```

## 命令行

```bash
python -m dexworld.cli doctor     # 健康检查
python -m dexworld.cli train      # 训练
python -m dexworld.cli eval       # 评测
python -m dexworld.cli predict    # 单步预测
python -m dexworld.cli serve      # 启动 HTTP 服务
```

## 配置

```yaml
model:
  backend: mock        # mock / openai_compat
  hidden_dim: 64
  denoise_steps: 20
train:
  epochs: 5
  lr: 0.0003
  seed: 42
data:
  sim_episodes: 64
  seq_len: 8
```

完整配置见 [config.example.yaml](config.example.yaml)。

## 文档

| 文档 | 说明 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 架构设计 |
| [docs/quickstart.md](docs/quickstart.md) | 快速开始 |
| [docs/tactile_template_34.md](docs/tactile_template_34.md) | 34 区域触觉模板 |
| [docs/data_engine.md](docs/data_engine.md) | 自主式数据引擎 |
| [docs/hbridge.md](docs/hbridge.md) | H-Bridge 共享注意力 |
| [docs/joint_denoising.md](docs/joint_denoising.md) | 联合去噪扩散 |
| [docs/deployment.md](docs/deployment.md) | 部署指南 |
| [docs/faq.md](docs/faq.md) | FAQ |

## 基准（零依赖 Mock）

训练前后对比（示例）：

| 指标 | 训练前 | 训练后 |
|---|---|---|
| 视觉预测误差 | 较高 | 下降 |
| 触觉重建误差 | 较高 | 下降 |
| 模型能力 skill | 0.00 | 递增 |

`examples/benchmark.py` 可复现完整对比。

## 技术栈

- Python 3.8+（仅标准库，`numpy` 可选）
- 扩散模型 / 注意力 / 专家混合均零依赖实现
- Docker / Kubernetes / Helm 部署

## 许可

[Apache-2.0](LICENSE) © 智能交付知识工程组
