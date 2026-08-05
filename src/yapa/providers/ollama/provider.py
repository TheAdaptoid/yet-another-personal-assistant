"""Ollama inference provider using the official Ollama SDK."""

import re

from ollama import AsyncClient

from yapa.models import (
    ContentDelta,
    EmbeddingResult,
    EmbedModel,
    LanguageModel,
    ModelDataUnion,
    ModelType,
    ReasoningDelta,
    ReasoningEffort,
    StreamEndEvent,
    TokenUsage,
    ToolCallDeltaEvent,
)
from yapa.services.config import Config, ProviderConfig

from .._retry import retry_async
from ..base import InferenceProvider
from ..exceptions import ModelsFetchError

_RETRYABLE = (ConnectionError, TimeoutError)


class OllamaIP(InferenceProvider):
    """Inference provider for Ollama (official SDK)."""

    DEFAULT_HOST = "http://127.0.0.1:11434"

    def __init__(self, config: Config):
        """Initialize the Ollama provider with the configured host."""
        super().__init__("ollama", "Ollama")
        pc = config.provider_configs.get("ollama", ProviderConfig())
        self._host = pc.base_url or self.DEFAULT_HOST
        self._max_retries = config.provider_max_retries
        self._client = AsyncClient(host=self._host)

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelDataUnion]:
        resp = await self._client.list()
        models = []
        for item in _get(resp, "models") or []:
            name = _get(item, "name") or _get(item, "model") or ""
            if name:
                caps = _get(item, "capabilities") or []
                models.append(
                    self._format_model(name, supports_reasoning="thinking" in caps)
                )
        if model_type:
            target = model_type.value
            return [m for m in models if getattr(m.type, "value", m.type) == target]
        return models

    def _format_model(self, model_id: str, supports_reasoning: bool = False):
        if "embed" in model_id.lower():
            return EmbedModel(id=model_id, provider_id=self.id)
        return LanguageModel(
            id=model_id,
            provider_id=self.id,
            supports_reasoning=supports_reasoning,
        )

    async def _get_model_impl(self, model_id: str) -> ModelDataUnion:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                try:
                    info = await self._client.show(model=model_id)
                    params = _get(info, "parameters") or {}
                    caps = _get(info, "capabilities") or []
                    # NOTE: the ollama SDK does not expose context_length in
                    # its parsed list() or show() responses. The raw API has
                    # it in models[].details.context_length and
                    # model_info.*.context_length, but the SDK drops it. We
                    # read num_ctx from the raw parameters string instead;
                    # for listing, context_length stays None until first use.
                    return LanguageModel(
                        id=model_id,
                        provider_id=self.id,
                        context_length=params.get("num_ctx"),
                        supports_tools=bool(_get(info, "template")),
                        supports_reasoning="thinking" in caps,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise ModelsFetchError(
                        f"Ollama show failed for '{model_id}': {exc}"
                    ) from exc
        raise ModelsFetchError(
            f"Model '{model_id}' not found in Ollama (no fabrication)."
        )

    def _map_reasoning(self, reasoning):
        if reasoning is None or reasoning == ReasoningEffort.OFF:
            return {"think": False}
        return {"think": True}

    def _new_messages(self, messages) -> list[dict]:
        from yapa.models import ImagePart, TextPart

        out = []
        for m in messages:
            d = {"role": m.role}
            if m.role == "user" and isinstance(m.content, list):
                text_parts = [p.text for p in m.content if isinstance(p, TextPart)]
                images = [
                    _strip_data_uri(p.image_url.url)
                    for p in m.content
                    if isinstance(p, ImagePart)
                ]
                d["content"] = " ".join(text_parts) or ""
                if images:
                    d["images"] = images
            elif m.role == "user":
                d["content"] = m.content or ""
            elif m.role == "assistant":
                d["content"] = m.content
                if getattr(m, "reasoning_content", None):
                    d["thinking"] = m.reasoning_content
            elif m.role == "system":
                d["content"] = m.content or ""
            elif m.role == "tool":
                d["content"] = m.content or ""
            out.append(d)
        return out

    async def _stream_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        async def _call():
            return await self._client.chat(
                model=model_id,
                messages=self._new_messages(messages),
                stream=True,
                **self._map_reasoning(reasoning),
            )

        stream = await retry_async(
            _call,
            max_attempts=self._max_retries + 1,
            retryable=lambda e: isinstance(e, _RETRYABLE),
        )
        usage = None
        finish_reason = None
        async for chunk in stream:
            chunk = chunk if isinstance(chunk, dict) else chunk.model_dump()
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("done", False):
                finish_reason = finish_reason or "stop"
                usage = _usage_from_ollama(chunk)
                continue
            msg = chunk.get("message", {})
            content = msg.get("content")
            if content:
                yield ContentDelta(content=content)
            thinking = msg.get("thinking")
            if thinking and thinking.strip():
                yield ReasoningDelta(content=thinking)
            for tc in msg.get("tool_calls", []):
                yield ToolCallDeltaEvent(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    name=tc.get("function", {}).get("name"),
                    arguments=tc.get("arguments"),
                )
        yield StreamEndEvent(
            finish_reason=finish_reason, usage=usage, model_id=model_id
        )

    async def _static_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        async def _call():
            return await self._client.chat(
                model=model_id,
                messages=self._new_messages(messages),
                stream=False,
                **self._map_reasoning(reasoning),
            )

        resp = await retry_async(
            _call,
            max_attempts=self._max_retries + 1,
            retryable=lambda e: isinstance(e, _RETRYABLE),
        )
        msg = (
            resp.get("message", {})
            if isinstance(resp, dict)
            else (getattr(resp, "message", {}) or {})
        )
        from yapa.models import AssistantMessage

        thinking = msg.get("thinking")
        return AssistantMessage(
            content=msg.get("content"),
            reasoning_content=thinking if thinking and thinking.strip() else None,
        )

    async def _embed_impl(self, model_id, input) -> EmbeddingResult:
        async def _call():
            return await self._client.embed(model=model_id, input=input)

        resp = await retry_async(
            _call,
            max_attempts=self._max_retries + 1,
            retryable=lambda e: isinstance(e, _RETRYABLE),
        )
        emb = (
            resp.get("embeddings", [])
            if isinstance(resp, dict)
            else getattr(resp, "embeddings", [])
        )
        usage = None
        count = (
            resp.get("prompt_eval_count")
            if isinstance(resp, dict)
            else getattr(resp, "prompt_eval_count", None)
        )
        if count is not None:
            usage = TokenUsage(
                prompt_tokens=count, completion_tokens=0, total_tokens=count
            )
        return EmbeddingResult(vectors=emb, model_id=model_id, usage=usage)


def _strip_data_uri(url: str) -> str:
    match = re.match(r"data:image/[a-zA-Z0-9.+-]+;base64,(.*)", url)
    return match.group(1) if match else url


def _get(obj, key: str):
    """Return ``obj[key]`` (dict) or ``getattr(obj, key)`` (pydantic), else None."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _usage_from_ollama(chunk: dict) -> TokenUsage | None:
    p = chunk.get("prompt_eval_count")
    c = chunk.get("eval_count")
    if p is None and c is None:
        return None
    p = p or 0
    c = c or 0
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)
