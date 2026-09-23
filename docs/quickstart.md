# 快速开始

## 环境要求

- Python 3.8+（仅标准库，无需安装任何依赖）
- 可选：numpy（自动检测，缺失回退纯 Python）

## 三步跑通

### 1. 克隆

```bash
git clone https://github.com/huzjie/dexworld.git
cd dexworld
```

### 2. 健康检查

```bash
python -m dexworld.cli doctor
```

### 3. 完整训练（零依赖 Mock）

```bash
python -m dexworld.cli train --config config.example.yaml
```

或直接跑演示脚本：

```bash
python examples/demo_train_mock.py
```

你会看到「预测误差随训练轮次下降」的曲线。

## 命令行一览

```bash
python -m dexworld.cli doctor      # 健康检查
python -m dexworld.cli train       # 训练
python -m dexworld.cli eval        # 评测
python -m dexworld.cli predict     # 单步预测
python -m dexworld.cli serve       # 启动 HTTP 服务
```

## 最小代码示例

```python
from dexworld import DexWorldPipeline
from dexworld.config import DexWorldConfig

cfg = DexWorldConfig()
cfg.model.backend = "mock"
cfg.train.epochs = 4
result = DexWorldPipeline(cfg).run()
print(result.summary())
```

## Docker 运行

```bash
docker build -t dexworld .
docker run --rm dexworld python -m dexworld.cli doctor
```
