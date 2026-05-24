#!/usr/bin/env python3
"""
DeepSeek API Adapter for OpenAI Responses API

Translates OpenAI Responses API requests to DeepSeek Chat Completions API,
so Codex (which uses the Responses API) can work with DeepSeek models.

Usage:
    export DEEPSEEK_API_KEY="sk-xxx"
    python deepseek_adapter.py
"""

import os
import json
import uuid
import time
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# ── Config ──────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info")

# Model name mapping: OpenAI model → DeepSeek model
# Maps common model names from various clients to their DeepSeek equivalents.
# Built-in defaults can be overridden by the MODEL_MAP env var.
DEFAULT_MODEL_MAP: Dict[str, str] = {
    # OpenAI models
    "gpt-4o": "deepseek-v4-pro",
    "gpt-4o-mini": "deepseek-v4-flash",
    "gpt-4-turbo": "deepseek-v4-flash",
    "gpt-3.5-turbo": "deepseek-v4-flash",
    "o1": "deepseek-v4-pro",
    "o3-mini": "deepseek-v4-flash",
    # Anthropic models
    "claude-sonnet-4": "deepseek-v4-pro",
    "claude-haiku-3": "deepseek-v4-flash",
    # Generic model types used by tools like Codex
    "chat": "deepseek-v4-flash",
    "reasoning": "deepseek-v4-pro",
}

MODEL_MAP_RAW = os.environ.get("MODEL_MAP", "")
MODEL_MAP: Dict[str, str] = dict(DEFAULT_MODEL_MAP)  # start with defaults
for pair in MODEL_MAP_RAW.split(",") if MODEL_MAP_RAW else []:
    if "→" in pair:
        k, v = pair.split("→", 1)
        MODEL_MAP[k.strip()] = v.strip()
    elif ":" in pair:
        k, v = pair.split(":", 1)
        MODEL_MAP[k.strip()] = v.strip()

# Default model: use the latest stable DeepSeek model.
# deepseek-chat and deepseek-reasoner are deprecated after 2026/07/24.
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

# Reasoning effort mapping: OpenAI → DeepSeek
# DeepSeek reasoning_effort accepts: off | low | high | max
# OpenAI reasoning_effort accepts: low | medium | high
REASONING_EFFORT_MAP = {
    "low": "low",
    "medium": "high",
    "high": "max",
}

# Models that support thinking/reasoning
THINKING_CAPABLE_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro", "deepseek-reasoner"}

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("deepseek-adapter")


# ── HTTP Client ─────────────────────────────────────────────────────────
_client: Optional[httpx.AsyncClient] = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    return _client


async def close_client():
    global _client
    if _client:
        await _client.aclose()
        _client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_client()


app = FastAPI(title="DeepSeek Adapter", version="1.0.0", lifespan=lifespan)


# ── Helpers ─────────────────────────────────────────────────────────────

def resolve_model(requested_model: str) -> str:
    """Map OpenAI model name to DeepSeek model name."""
    if requested_model in MODEL_MAP:
        return MODEL_MAP[requested_model]
    return DEFAULT_DEEPSEEK_MODEL


def convert_responses_input_to_messages(input_data: Any) -> List[Dict[str, Any]]:
    """
    Convert OpenAI Responses API 'input' to Chat Completions 'messages'.
    
    The Responses API 'input' can be:
      - A plain string
      - A list of message objects like:
        [{"role": "user", "content": "hello"}, ...]
      - A list of input items with type fields
    """
    messages = []

    if isinstance(input_data, str):
        # Plain text prompt
        messages.append({"role": "user", "content": input_data})
        return messages

    if isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, dict):
                role = item.get("role", "user")
                # Map OpenAI-specific roles to DeepSeek-compatible roles.
                # DeepSeek supports: system, user, assistant, tool, latest_reminder
                # OpenAI also uses: developer (→ system), function (→ tool)
                if role == "developer":
                    role = "system"
                elif role == "function":
                    role = "tool"
                content = item.get("content", "")
                
                # Handle content that could be a string or list of content parts
                if isinstance(content, list):
                    # Build text content from parts
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                            elif part.get("type") == "input_image" or part.get("type") == "image":
                                # DeepSeek doesn't support image input via Chat API,
                                # but we pass it through for compatibility
                                text_parts.append(f"[image: {part.get('image_url', part.get('source', {}))}]")
                        else:
                            text_parts.append(str(part))
                    content = "\n".join(text_parts)
                
                messages.append({"role": role, "content": str(content)})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages

    # Fallback
    messages.append({"role": "user", "content": str(input_data)})
    return messages


def convert_tools(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """
    Convert OpenAI Responses API tools to DeepSeek Chat Completions format.
    
    OpenAI Responses API uses a flat tool schema:
      {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    
    DeepSeek requires nested 'function' field (OpenAI Chat Completions format):
      {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    
    DeepSeek only supports 'function' type tools. Hosted/built-in tools
    (web_search, file_search, code_interpreter, etc.) are silently dropped.
    Compound tools with nested 'tools' arrays are expanded into individual function tools.
    """
    if not tools:
        return tools
    
    # DeepSeek only supports 'function' type tools
    DEEPSEEK_SUPPORTED_TYPES = {"function"}
    
    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        
        tool_type = tool.get("type", "function")
        
        # Skip unsupported tool types (web_search, file_search, etc.)
        if tool_type not in DEEPSEEK_SUPPORTED_TYPES:
            log.info(f"  ⚠ dropping unsupported tool type: {tool_type}")
            continue
        
        # Compound tool with nested 'tools' array → expand sub-tools
        if "tools" in tool and isinstance(tool["tools"], list):
            converted.extend(_convert_single_tool_list(tool["tools"]))
            continue
        
        # Already has nested 'function' field → pass through
        if "function" in tool:
            converted.append(tool)
            continue
        
        # Flat schema: wrap non-type fields into 'function'
        func_def = {k: v for k, v in tool.items() if k != "type"}
        converted.append({"type": "function", "function": func_def})
    
    return converted


def _convert_single_tool_list(tools: List[Dict]) -> List[Dict]:
    """Convert a list of tool definitions (used for nested compound tools)."""
    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "function")
        if tool_type != "function":
            continue
        if "function" in tool:
            result.append(tool)
        else:
            func_def = {k: v for k, v in tool.items() if k != "type"}
            result.append({"type": "function", "function": func_def})
    return result


def deepseek_response_to_openai(
    ds_response: Dict[str, Any],
    request_model: str,
    request_id: str,
) -> Dict[str, Any]:
    """
    Convert DeepSeek Chat Completions response to OpenAI Responses API format.
    """
    choices = ds_response.get("choices", [])
    usage = ds_response.get("usage", {})

    output_items = []

    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content", "")
        tool_calls = message.get("tool_calls")

        # Main text content
        if content:
            output_items.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            })

        # Tool calls
        if tool_calls:
            for tc in tool_calls:
                output_items.append({
                    "type": "function_call",
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                    "status": "completed",
                })

        # Handle refusal
        refusal = message.get("refusal")
        if refusal:
            output_items.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": refusal}],
            })

    response: Dict[str, Any] = {
        "id": request_id,
        "object": "response",
        "created": int(time.time()),
        "model": request_model,
        "status": "completed",
        "output": output_items,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens_details": {"cached_tokens": usage.get("prompt_cache_hit_tokens", 0)},
            "output_tokens_details": {"reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0) if isinstance(usage.get("completion_tokens_details"), dict) else 0},
        },
    }

    return response


def build_error_response(status: int, message: str, request_id: str) -> Dict[str, Any]:
    return {
        "id": request_id,
        "object": "response",
        "status": "failed",
        "error": {
            "code": f"deepseek_adapter_{status}",
            "message": message,
        },
    }


# ── Streaming Helpers ───────────────────────────────────────────────────

def convert_stream_chunk(chunk: Dict[str, Any], request_id: str) -> Optional[str]:
    """
    Convert a DeepSeek streaming chunk to an OpenAI Responses API
    server-sent event (SSE) data string.
    """
    choices = chunk.get("choices", [])
    if not choices:
        return None

    delta = choices[0].get("delta", {})
    finish_reason = choices[0].get("finish_reason")

    if finish_reason is not None:
        # Final chunk
        event = {
            "type": "response.output_item.done",
            "item": {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": ""}],
                "status": "completed",
            },
        }
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    text = delta.get("content", "")
    if text:
        event = {
            "type": "response.output_text.delta",
            "delta": text,
            "item_id": f"msg_{uuid.uuid4().hex[:8]}",
            "output_index": 0,
            "content_index": 0,
        }
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return None


# ── Routes ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "adapter": "deepseek-openai-responses"}


@app.get("/v1/models")
async def list_models():
    """List available models (mocked for compatibility)."""
    return {
        "object": "list",
        "data": [
            {"id": "deepseek-v4-flash", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
            {"id": "deepseek-v4-pro", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
            {"id": "deepseek-chat", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
        ]
    }


@app.api_route("/v1/responses", methods=["POST"])
async def create_response(request: Request):
    """
    Main endpoint: Translate OpenAI Responses API → DeepSeek Chat API.
    """
    if not DEEPSEEK_API_KEY:
        return JSONResponse(
            status_code=500,
            content=build_error_response(
                500,
                "DEEPSEEK_API_KEY not set. Please export DEEPSEEK_API_KEY=sk-xxx",
                f"req_{uuid.uuid4().hex[:12]}",
            ),
        )

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content=build_error_response(400, f"Invalid JSON: {e}", f"req_{uuid.uuid4().hex[:12]}"),
        )

    request_id = body.get("request_id", f"req_{uuid.uuid4().hex[:12]}")
    requested_model = body.get("model", DEFAULT_DEEPSEEK_MODEL)
    deepseek_model = resolve_model(requested_model)
    is_stream = body.get("stream", False)

    log.info(f"→ {requested_model} → {deepseek_model} | stream={is_stream} | id={request_id}")
    log.info(f"  req body keys: {list(body.keys())}")
    if body.get("reasoning"):
        log.info(f"  reasoning: {json.dumps(body['reasoning'], ensure_ascii=False)[:200]}")
    if body.get("reasoning_effort"):
        log.info(f"  reasoning_effort: {body['reasoning_effort']}")

    # ── Build DeepSeek request ──
    dm = convert_responses_input_to_messages(body.get("input", ""))

    # Handle instructions → system message
    instructions = body.get("instructions")
    if instructions and isinstance(instructions, str):
        dm.insert(0, {"role": "system", "content": instructions})

    # ── Map reasoning_effort (Responses API) → thinking + reasoning_effort (DeepSeek) ──
    # DeepSeek's thinking parameter controls whether thinking output is enabled.
    # reasoning_effort controls the depth of reasoning.
    #
    # OpenAI Responses API 'reasoning' is an object: {"effort": "low"|"medium"|"high", "summary": "auto"|...}
    # We also support top-level 'reasoning_effort' as a plain string for compatibility.
    reasoning_raw = body.get("reasoning")
    if isinstance(reasoning_raw, dict):
        reasoning_effort = reasoning_raw.get("effort")
    elif isinstance(reasoning_raw, str):
        reasoning_effort = reasoning_raw
    else:
        reasoning_effort = body.get("reasoning_effort")
    
    ds_thinking = None
    ds_reasoning_effort = None
    
    if reasoning_effort and deepseek_model in THINKING_CAPABLE_MODELS:
        # Convert from OpenAI's scale (low/medium/high) to DeepSeek's scale (off/low/high/max)
        mapped = REASONING_EFFORT_MAP.get(reasoning_effort)
        if mapped == "max":
            ds_thinking = {"type": "enabled"}
            ds_reasoning_effort = "max"
        elif mapped == "high":
            ds_thinking = {"type": "enabled"}
            ds_reasoning_effort = "high"
        elif mapped == "low":
            ds_thinking = {"type": "enabled"}
            ds_reasoning_effort = reasoning_effort
    elif reasoning_effort == "off" or reasoning_effort is False:
        ds_reasoning_effort = "off"
    
    # Also allow explicit 'thinking' parameter passthrough
    if body.get("thinking"):
        ds_thinking = body["thinking"]

    # ── Build DeepSeek request ──
    # See https://api-docs.deepseek.com/ for full parameter reference
    ds_request: Dict[str, Any] = {
        "model": deepseek_model,
        "messages": dm,
        "stream": is_stream,
        "temperature": body.get("temperature", 1.0),
        "top_p": body.get("top_p", 1.0),
        "max_tokens": body.get("max_output_tokens", body.get("max_tokens", 8192)),
        "stop": body.get("stop", body.get("stop_sequences")),
    }
    
    # Add thinking/reasoning if configured
    if ds_thinking:
        ds_request["thinking"] = ds_thinking
    if ds_reasoning_effort:
        ds_request["reasoning_effort"] = ds_reasoning_effort

    # Remove None values
    ds_request = {k: v for k, v in ds_request.items() if v is not None}

    # DeepSeek does not allow temperature/top_p with reasoning_effort
    if ds_request.get("reasoning_effort") or ds_request.get("thinking"):
        ds_request.pop("temperature", None)
        ds_request.pop("top_p", None)

    # Tools
    raw_tools = body.get("tools")
    if raw_tools and isinstance(raw_tools, list):
        ds_request["tools"] = convert_tools(raw_tools)

        # Tool choice — only set when tools are present
        tool_choice = body.get("tool_choice", "auto")
        if tool_choice:
            ds_request["tool_choice"] = tool_choice

    # Metadata / user
    if body.get("user"):
        ds_request["user"] = body["user"]

    log.info(f"  → DeepSeek request: {json.dumps({k: v for k, v in ds_request.items() if k != 'messages'}, ensure_ascii=False)}")

    # ── Call DeepSeek API ──
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    client = get_client()
    ds_url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"

    if is_stream:
        return await _handle_streaming(client, ds_url, headers, ds_request, request_id, requested_model)
    else:
        return await _handle_non_streaming(client, ds_url, headers, ds_request, request_id, requested_model)


async def _handle_non_streaming(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    ds_request: Dict[str, Any],
    request_id: str,
    requested_model: str,
) -> JSONResponse:
    """Non-streaming: call DeepSeek, translate response."""
    try:
        resp = await client.post(url, json=ds_request, headers=headers)
    except httpx.TimeoutException:
        log.error(f"DeepSeek API timeout for {request_id}")
        return JSONResponse(
            status_code=504,
            content=build_error_response(504, "DeepSeek API timeout", request_id),
        )
    except Exception as e:
        log.error(f"DeepSeek API error: {e}")
        return JSONResponse(
            status_code=502,
            content=build_error_response(502, f"DeepSeek API error: {e}", request_id),
        )

    if resp.status_code != 200:
        log.error(f"DeepSeek API returned {resp.status_code}: {resp.text[:500]}")
        try:
            err_body = resp.json()
            error_msg = err_body.get("error", {}).get("message", resp.text[:200])
        except Exception:
            error_msg = resp.text[:200]
        return JSONResponse(
            status_code=resp.status_code,
            content=build_error_response(resp.status_code, f"DeepSeek: {error_msg}", request_id),
        )

    try:
        ds_response = resp.json()
    except Exception as e:
        return JSONResponse(
            status_code=502,
            content=build_error_response(502, f"Invalid DeepSeek response: {e}", request_id),
        )

    oai_response = deepseek_response_to_openai(ds_response, requested_model, request_id)
    log.info(f"✓ {request_id} | tokens: {oai_response['usage']['total_tokens']}")
    return JSONResponse(content=oai_response)


async def _handle_streaming(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    ds_request: Dict[str, Any],
    request_id: str,
    requested_model: str,
) -> StreamingResponse:
    """Streaming: translate each DeepSeek chunk to Responses API SSE format."""

    async def event_stream():
        # Start event
        yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': request_id, 'model': requested_model, 'status': 'in_progress'}}, ensure_ascii=False)}\n\n"

        try:
            async with client.stream("POST", url, json=ds_request, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    error_str = error_text.decode()[:1000]
                    log.error(f"DeepSeek stream error {resp.status_code}: {error_str}")
                    yield f"data: {json.dumps({'type': 'error', 'code': resp.status_code, 'message': error_str}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                buffer = ""
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_data = line[6:].strip()
                    if chunk_data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(chunk_data)
                        sse_event = convert_stream_chunk(chunk, request_id)
                        if sse_event:
                            yield sse_event
                    except json.JSONDecodeError:
                        continue

                # Done event
                yield f"data: {json.dumps({'type': 'response.completed', 'response': {'id': request_id, 'model': requested_model, 'status': 'completed'}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Also proxy the Chat Completions endpoint for compatibility ───────────
@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """
    Passthrough proxy for Chat Completions API (unchanged format).
    Allows other tools that use /v1/chat/completions to also work.
    """
    if not DEEPSEEK_API_KEY:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "DEEPSEEK_API_KEY not set"}},
        )

    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": {"message": str(e)}})

    # Remap model if needed
    if body.get("model") in MODEL_MAP:
        body["model"] = MODEL_MAP[body["model"]]

    client = get_client()
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    is_stream = body.get("stream", False)
    url = f"{DEEPSEEK_BASE_URL}/v1/chat/completions"

    if is_stream:
        async def proxy_stream():
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        return StreamingResponse(
            proxy_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        resp = await client.post(url, json=body, headers=headers)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


# ── Entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        log.warning("⚠ DEEPSEEK_API_KEY is not set!")
        log.warning("  export DEEPSEEK_API_KEY=sk-your-key-here")
        log.warning("  The server will start but return errors on API calls.")
    else:
        log.info(f"✓ DeepSeek API key configured (prefix: {DEEPSEEK_API_KEY[:8]}...)")

    log.info(f"  Listening on http://{HOST}:{PORT}")
    log.info(f"  DeepSeek endpoint: {DEEPSEEK_BASE_URL}")
    log.info(f"  Model map: {MODEL_MAP}")
    log.info("")
    log.info("  Codex usage:")
    log.info(f'    export OPENAI_BASE_URL=http://localhost:{PORT}')
    log.info(f'    export OPENAI_API_KEY=sk-not-needed')
    log.info("  (The adapter forwards requests to DeepSeek using your DEEPSEEK_API_KEY)")

    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower())
