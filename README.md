# DeepSeek API Adapter for OpenAI Responses API

让 Codex 等使用 OpenAI Responses API 的工具调用 DeepSeek 模型。

## 为什么需要这个？

Codex 默认走 OpenAI 的 **Responses API**，只跟 OpenAI 官方通信。本适配器在本地起一个代理服务，把 Responses API 请求翻译成 DeepSeek 可识别的 Chat Completions API 格式，让 Codex 可以直接用 DeepSeek 模型。

```
Codex → /responses 或 /v1/responses
           ↓
     deepseek_adapter.py (翻译层)
           ↓
     DeepSeek Chat API (/v1/chat/completions)
```

## 快速开始

```bash
# 1. 安装依赖
pip install fastapi uvicorn httpx

# 2. 设置 DeepSeek API Key 并启动
export DEEPSEEK_API_KEY="sk-your-deepseek-api-key"
python deepseek_adapter.py

# 3. Codex 中配置自定义 provider
export OPENAI_BASE_URL=http://localhost:8080
export OPENAI_API_KEY=***-ignored-by-adapter
codex "hello world"
```

## Codex 配置（避免 OAuth 登录）

Codex 默认走 OAuth 登录流程。推荐用自定义 provider 模式，绕开登录：

### 方法一：`~/.codex/config.toml`

```toml
[model_providers.deepseek]
base_url = "http://localhost:8080"
api_key = "your-api-key"
wire_api = "responses"
requires_oauth = false

[models.deepseek-v4-flash]
provider = "deepseek"
```

然后用：
```bash
codex --provider deepseek --model deepseek-v4-flash "hello world"
```

### 方法二：环境变量（最简单）

```bash
export OPENAI_BASE_URL=http://localhost:8080
export OPENAI_API_KEY=***
codex "hello world"
```

适配器内置了登录端点（`/login`、`/api/login`、`/auth` 等），全部返回 `requires_oauth: false`，Codex 检查到后不会再弹登录框。

## 模型映射

根据 [DeepSeek 官方 API 文档](https://api-docs.deepseek.com/)：

| 模型名 | 用途 | 说明 |
|--------|------|------|
| `deepseek-v4-flash` | 默认对话 | **推荐**，支持 thinking/non-thinking |
| `deepseek-v4-pro` | 高性能推理 | 更强的推理能力 |
| `deepseek-chat` | 旧名 | ⚠️ 2026/07/24 后废弃 |
| `deepseek-reasoner` | 旧名 | ⚠️ 2026/07/24 后废弃 |

内置模型名映射（自动将常用模型名转成 DeepSeek 模型）：

| 请求名 | → | 实际调用 |
|--------|---|----------|
| `gpt-4o` | → | `deepseek-v4-pro` |
| `gpt-4o-mini` / `gpt-3.5-turbo` | → | `deepseek-v4-flash` |
| `claude-sonnet-4` | → | `deepseek-v4-pro` |

可通过 `MODEL_MAP` 环境变量自定义覆盖。

### 思考模式映射

| OpenAI | → | DeepSeek |
|--------|---|----------|
| `reasoning_effort: "low"` | → | `thinking: enabled, reasoning_effort: "low"` |
| `reasoning_effort: "medium"` | → | `thinking: enabled, reasoning_effort: "high"` |
| `reasoning_effort: "high"` | → | `thinking: enabled, reasoning_effort: "max"` |

## 端点列表

| 路径 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/models` | GET | 模型列表 |
| `/v1/responses` | POST | Responses API (OpenAI 标准) |
| **`/responses`** | POST | **Codex 实际调用的路径** |
| `/v1/chat/completions` | POST | Chat API 透传代理 |
| `/login` | GET/POST | 登录认证（返回 requires_oauth: false） |
| `/api/login` | GET/POST | 同上 |
| `/v1/login` | GET/POST | 同上 |
| `/auth` | GET/POST | 同上 |
| `/api/auth` | GET/POST | 同上 |
| `/v1/auth` | GET/POST | 同上 |
| `/*` | 任意 | 兜底（返回有效响应） |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API Key（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek API 地址 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8080` | 监听端口 |
| `MODEL_MAP` | — | 自定义模型映射 |
| `LOG_LEVEL` | `info` | 日志级别 |

## 技术说明

| OpenAI Responses API | → | DeepSeek Chat API |
|---|---|---|
| `input` (string) | → | `messages: [{role: "user", content: input}]` |
| `input` (message array) | → | 按角色逐条映射 |
| `instructions` | → | 插入 system message |
| `model` | → | 按映射表转换 |
| `max_output_tokens` | → | `max_tokens` |
| `tools` / `tool_choice` | → | 透传 |
| `reasoning` / `reasoning_effort` | → | `thinking` + `reasoning_effort` |
| `temperature`, `top_p`, `stop` | → | 透传 |
