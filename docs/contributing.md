# 贡献指南

## 环境准备

```bash
git clone https://github.com/huzjie/dexworld.git
cd dexworld
python -m dexworld.cli doctor
```

## 运行测试

```bash
python -m unittest discover -s tests -t .
```

## 代码风格

- Python 3 语法，`from __future__ import annotations`
- 类型标注用 `dataclass`
- 零重型依赖，新增功能优先用标准库
