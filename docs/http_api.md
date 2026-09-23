# HTTP API

## 启动

```bash
python -m dexworld.cli serve --port 8765
```

## 端点

| 端点 | 方法 | 说明 |
|---|---|---|
| /health | GET | 健康检查（状态 + skill） |
| /metrics | GET | 指标快照 |
| /predict | POST | 预测 |

## 示例

```bash
curl http://127.0.0.1:8765/health
curl -X POST http://127.0.0.1:8765/predict
```
