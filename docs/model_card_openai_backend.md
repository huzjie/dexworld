# 模型卡：OpenAI 兼容后端

| 项 | 值 |
|---|---|
| 类型 | 可选高层语义编码 |
| 接口 | /chat/completions |
| 用途 | 接 MiMo / GLM 等做语义编码 |
| 需配置 | api_base / api_key / model_name |

通过 `config.yaml` 的 `model.backend: openai_compat` 启用。
