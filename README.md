# DeepSeek API Adapter for OpenAI Responses API

让 Codex 等使用 OpenAI Responses API 的工具调用 DeepSeek 模型。

## 为什么需要这个？

Codex 默认走 OpenAI 的 **Responses API**（`/v1/responses`），不支持直接使用兼容 OpenAI 格式的第三方 API。本适配器在本地起一个代理服务，把 Responses API 的请求翻译成 DeepSeek 可识别的 Chat Completions API 格式。

```
Codex → OpenAI Responses API (/v1/responses)
           ↓
     deepseek_adapter.py (翻译层)
           ↓
     DeepSeek Chat API (/v1/chat/completions)
```

## 模型映射

根据 [DeepSeek 官方 API 文档](https://api-docs.deepseek.com/)，支持的模型：

| 模型名 | 用途 | 说明 |
|--------|------|------|
| `deepseek-v4-flash` | 默认对话 | **推荐使用**，支持 thinking/non-thinking |
| `deepseek-v4-pro` | 高性能推理 | 更强的推理能力 |
| `deepseek-chat` | 对话 (旧名) | ⚠️ 2026/07/24 后废弃，映射到 v4-flash non-thinking |
| `deepseek-reasoner` | 推理 (旧名) | ⚠️ 2026/07/24 后废弃，映射到 v4-flash thinking 模式 |

### 思考模式 (Thinking/Reasoning)

适配器支持将 OpenAI Responses API 的 `reasoning_effort` 参数映射到 DeepSeek 的 `thinking` + `reasoning_effort`：

| OpenAI | → | DeepSeek |
|--------|---|----------|
| `reasoning_effort: "low"` | → | `thinking: enabled, reasoning_effort: "low"` |
| `reasoning_effort: "medium"` | → | `thinking: enabled, reasoning_effort: "high"` |
| `reasoning_effort: "high"` | → | `thinking: enabled, reasoning_effort: "max"` |

## 快速开始

```bash
# 1. 安装依赖
pip install fastapi uvicorn httpx

# 2. 设置 DeepSeek API Key
export DEEPSEEK_API_KEY="sk-your-deepseek-api-key"

# 3. 启动服务
python deepseek_adapter.py

# 4. Codex 中设置
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY="sk-not-needed-by-adapter"

# Codex 会通过 /v1/responses 请求，适配器自动翻译到 DeepSeek
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8080` | 监听端口 |
| `MODEL_MAP` | — | 自定义模型映射，如 `"gpt-4o→deepseek-v4-flash,claude-sonnet-4→deepseek-v4-pro"` |
| `LOG_LEVEL` | `info` | 日志级别 |

## 技术说明

适配器自动处理以下翻译：

| OpenAI Responses API | → | DeepSeek Chat API |
|---|---|---|
| `input` (string) | → | `messages: [{role: "user", content: input}]` |
| `input` (message array) | → | `messages` (逐条映射) |
| `instructions` | → | `messages[0]` 插入 system message |
| `model` | → | `model` (按映射表转换) |
| `max_output_tokens` | → | `max_tokens` |
| `tools` / `tool_choice` | → | `tools` / `tool_choice` (透传) |
| `reasoning` / `reasoning_effort` | → | `thinking` + `reasoning_effort` |
| `temperature`, `top_p`, `stop` | → | 同名参数 (透传) |
| Response `output[].content[].text` | ← | `choices[].message.content` |
| 流式 SSE chunks | ← | streaming chunks 逐块翻译 |

## 兼容性

- **Codex**: 直接可用，设置 `OPENAI_BASE_URL` 即可
- **其他 Responses API 工具**: 通用兼容
- **Chat Completions API**: `/v1/chat/completions` 端点也做了透明代理，支持 `deepseek-chat` 等旧名映射
