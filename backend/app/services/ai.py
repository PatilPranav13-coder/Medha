from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx

from app.core.config import settings


class AIConfigurationError(RuntimeError):
    pass


class AIProviderError(RuntimeError):
    pass


def model_catalog() -> dict[str, Any]:
    return {
        "provider": settings.ai_provider,
        "configured": bool(settings.ai_base_url and settings.configured_models and (settings.ai_provider != "openai-compatible" or settings.ai_api_key)),
        "models": [{"id": model, "name": model} for model in settings.configured_models],
    }


def require_model(model_name: str | None = None) -> str:
    catalog = settings.configured_models
    if not settings.ai_base_url or not catalog:
        raise AIConfigurationError("Configure MEDHA_AI_BASE_URL and MEDHA_MODEL_NAME in the environment to enable AI responses.")
    if settings.ai_provider == "openai-compatible" and not settings.ai_api_key:
        raise AIConfigurationError("Configure MEDHA_AI_API_KEY for the OpenAI-compatible provider.")
    if settings.ai_provider not in {"ollama", "openai-compatible"}:
        raise AIConfigurationError("MEDHA_AI_PROVIDER must be ollama or openai-compatible.")
    selected = model_name or catalog[0]
    if selected not in catalog:
        raise AIConfigurationError("Select a model listed by Medha's configured model registry.")
    return selected


def _endpoint(suffix: str) -> str:
    base = settings.ai_base_url.rstrip("/")
    if base.endswith(suffix):
        return base
    if settings.ai_provider == "ollama":
        return f"{base}{suffix}"
    if base.endswith("/v1"):
        return f"{base}{suffix.replace('/v1', '', 1)}"
    return f"{base}/v1{suffix.replace('/v1', '', 1)}"


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if settings.ai_provider == "openai-compatible":
        headers["Authorization"] = f"Bearer {settings.ai_api_key}"
    return headers


async def stream_chat(
    messages: list[dict[str, str]],
    model_name: str,
    on_usage: Callable[[int | None, int | None], None] | None = None,
) -> AsyncIterator[str]:
    model = require_model(model_name)
    if settings.ai_provider == "ollama":
        url = _endpoint("/api/chat")
        body = {"model": model, "messages": messages, "stream": True, "options": {"temperature": 0.25}}
    else:
        url = _endpoint("/v1/chat/completions")
        body = {"model": model, "messages": messages, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.25}

    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, headers=_headers(), json=body) as response:
                if response.is_error:
                    await response.aread()
                    detail = response.text[:500]
                    raise AIProviderError(f"The configured model provider returned HTTP {response.status_code}: {detail}")
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if settings.ai_provider == "ollama":
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = (payload.get("message") or {}).get("content", "")
                        if token:
                            yield token
                        if payload.get("error"):
                            raise AIProviderError(str(payload["error"]))
                        if on_usage and ("prompt_eval_count" in payload or "eval_count" in payload):
                            on_usage(payload.get("prompt_eval_count"), payload.get("eval_count"))
                        if payload.get("done"):
                            break
                    else:
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choice = (payload.get("choices") or [{}])[0]
                        usage = payload.get("usage") or {}
                        if on_usage and usage:
                            on_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
                        delta = choice.get("delta") or {}
                        token = delta.get("content", "")
                        if isinstance(token, str) and token:
                            yield token
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise AIProviderError("Medha could not reach the configured model provider. Check its URL and service status.") from exc
    except httpx.HTTPError as exc:
        raise AIProviderError("The model provider request failed. Check the provider configuration and logs.") from exc


async def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts or not settings.embedding_model or not settings.ai_base_url:
        return []
    if settings.ai_provider == "ollama":
        url = _endpoint("/api/embed")
        body: dict[str, Any] = {"model": settings.embedding_model, "input": texts}
    elif settings.ai_provider == "openai-compatible":
        url = _endpoint("/v1/embeddings")
        body = {"model": settings.embedding_model, "input": texts}
    else:
        return []
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(url, headers=_headers(), json=body)
            response.raise_for_status()
            payload = response.json()
        if settings.ai_provider == "ollama":
            vectors = payload.get("embeddings", [])
        else:
            vectors = [item.get("embedding") for item in payload.get("data", [])]
        result = [vector for vector in vectors if isinstance(vector, list) and len(vector) == settings.embedding_dimension]
        return result
    except (httpx.HTTPError, ValueError, TypeError):
        return []
