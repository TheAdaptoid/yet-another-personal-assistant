"""Shared implementation for OpenAI-family providers (OpenAI, OpenRouter, LM Studio)."""

import json
from abc import ABC
from typing import Any, AsyncGenerator, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from yapa.models import (
    AssistantMessage,
    ContentDelta,
    EmbeddingResult,
    ImagePart,
    InferenceParams,
    LanguageModel,
    Message,
    ModelData,
    ModelDataUnion,
    ModelType,
    ReasoningDelta,
    ReasoningEffort,
    StreamEndEvent,
    StreamEvent,
    TextPart,
    TokenUsage,
    ToolCall,
    ToolCallDeltaEvent,
    ToolMessage,
)
from yapa.tools import Tool

from .._classify import classify_model_type
from ..base import InferenceProvider
from ..exceptions import ModelInvocationError
from ._noauth import build_openai_client


class OpenAICompatibleProvider(InferenceProvider, ABC):
    """
    Base for OpenAI-family APIs using the official AsyncOpenAI SDK.

    api_key is ``str | None``; a None/empty key produces requests with no
    Authorization header (REQ-PROV-16). The client carries configured timeout
    and max retries (REQ-PROV-04/25).
    """

    _SUPPORTS_STREAM_USAGE: bool = True

    _OPENAI_CHAT_PARAMS = frozenset(
        {
            "temperature",
            "max_tokens",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "stop",
            "seed",
        }
    )

    def __init__(
        self,
        identifier: str,
        name: str,
        api_key: str | None,
        base_url: str | None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        """Initialize the provider and build the AsyncOpenAI client."""
        super().__init__(identifier, name)
        self._api_key = api_key
        self._client = build_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ── Classification ──

    def _format_model(
        self,
        model_id: str,
        native_type: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        context_length: int | None = None,
        max_output: int | None = None,
        supports_tools: bool | None = None,
        supports_vision: bool | None = None,
        supports_reasoning: bool | None = None,
        pricing=None,
    ) -> ModelData:
        from yapa.models import EmbedModel
        from yapa.models import ModelData as _MD

        mtype = classify_model_type(model_id, native_type)
        if mtype is ModelType.EMBED:
            return EmbedModel(
                id=model_id,
                provider_id=self.id,
                name=name,
                description=description,
                pricing=pricing,
            )
        if mtype is ModelType.LLM:
            return LanguageModel(
                id=model_id,
                provider_id=self.id,
                name=name,
                description=description,
                context_length=context_length,
                max_output=max_output,
                supports_tools=bool(supports_tools),
                supports_vision=bool(supports_vision),
                supports_reasoning=bool(supports_reasoning),
                pricing=pricing,
            )
        return _MD(
            id=model_id,
            provider_id=self.id,
            type=mtype,
            name=name,
            description=description,
        )

    # ── Message formatting ──

    def _format_message(
        self, message
    ) -> (
        ChatCompletionUserMessageParam
        | ChatCompletionSystemMessageParam
        | ChatCompletionAssistantMessageParam
        | ChatCompletionToolMessageParam
    ):
        if message.role == "user":
            if message.content is None:
                raise ValueError("User message content cannot be None.")
            return ChatCompletionUserMessageParam(
                role=message.role, content=self._format_user_content(message.content)
            )
        elif message.role == "system":
            if message.content is None:
                raise ValueError("System message content cannot be None.")
            return ChatCompletionSystemMessageParam(
                role=message.role, content=message.content
            )
        elif isinstance(message, AssistantMessage):
            msg: dict = dict(role=message.role, content=message.content)
            if message.reasoning_content:
                msg["reasoning_content"] = message.reasoning_content
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in message.tool_calls
                ]
            return cast(ChatCompletionAssistantMessageParam, msg)
        elif isinstance(message, ToolMessage):
            return ChatCompletionToolMessageParam(
                role=message.role,
                tool_call_id=message.tool_call_id,
                content=message.content or "",
            )
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    def _format_user_content(self, content):
        if isinstance(content, str):
            return content
        parts: list[dict] = []
        for part in content:
            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                item: dict = {
                    "type": "image_url",
                    "image_url": {"url": part.image_url.url},
                }
                if part.image_url.detail:
                    item["image_url"]["detail"] = part.image_url.detail
                parts.append(item)
            else:
                raise ValueError(f"Unsupported content part: {part!r}")
        return parts

    def _format_tools(self, tools):
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def _extract_reasoning(self, obj):
        """
        Extract reasoning content.

        Prefer reasoning_content, fall back to reasoning (e.g. OpenRouter).
        """
        text = getattr(obj, "reasoning_content", None) or getattr(
            obj, "reasoning", None
        )
        if text is not None and text.strip() == "":
            return None
        return text

    # ── Request building ──

    def _map_reasoning(self, reasoning):
        """Map a resolved ReasoningEffort to OpenAI/OpenRouter request params."""
        if reasoning is None or reasoning == ReasoningEffort.OFF:
            return None
        return {"reasoning": {"effort": reasoning.value}}

    def _build_request_kwargs(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None,
        params: InferenceParams | None,
        stream: bool,
        reasoning: ReasoningEffort | None,
    ) -> dict[str, Any]:
        params = params or InferenceParams()
        formatted_messages = [self._format_message(m) for m in messages]
        body = {
            k: v
            for k, v in params.model_dump(exclude_none=True).items()
            if k in self._OPENAI_CHAT_PARAMS
        }
        kwargs: dict[str, Any] = dict(
            model=model_id, messages=formatted_messages, stream=stream
        )
        kwargs.update(body)
        if stream and self._SUPPORTS_STREAM_USAGE:
            kwargs["stream_options"] = {"include_usage": True}
        reasoning_param = self._map_reasoning(reasoning)
        if reasoning_param is not None:
            kwargs.update(reasoning_param)
        formatted_tools = self._format_tools(tools)
        if formatted_tools is not None:
            kwargs["tools"] = formatted_tools
        return kwargs

    # ── Model listing ──

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelDataUnion]:
        response = await self._client.models.list()
        formatted = [self._format_model(m.id) for m in response.data]
        if model_type:
            target = model_type.value
            return [m for m in formatted if getattr(m.type, "value", m.type) == target]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelDataUnion:
        model = await self._client.models.retrieve(model_id)
        return self._format_model(model.id)

    # ── Streaming ──

    async def _stream_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        from openai import AsyncStream
        from openai.types.chat import ChatCompletionChunk

        kwargs = self._build_request_kwargs(
            model_id=model_id,
            messages=messages,
            tools=tools,
            params=params,
            stream=True,
            reasoning=reasoning,
        )
        response_stream: AsyncStream[
            ChatCompletionChunk
        ] = await self._client.chat.completions.create(**kwargs)

        finish_reason: str | None = None
        usage: TokenUsage | None = None
        async for chunk in response_stream:
            if not chunk.choices:
                if chunk.usage is not None:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                yield ContentDelta(content=delta.content)
            reasoning_text = self._extract_reasoning(delta)
            if reasoning_text:
                yield ReasoningDelta(content=reasoning_text)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield ToolCallDeltaEvent(
                        index=tc.index,
                        id=tc.id,
                        name=tc.function.name if tc.function else None,
                        arguments=tc.function.arguments if tc.function else None,
                    )
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if chunk.usage is not None:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
        yield StreamEndEvent(
            finish_reason=finish_reason, usage=usage, model_id=model_id
        )

    # ── Static chat ──

    async def _static_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AssistantMessage:
        kwargs = self._build_request_kwargs(
            model_id=model_id,
            messages=messages,
            tools=tools,
            params=params,
            stream=False,
            reasoning=reasoning,
        )
        response = await self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content = msg.content
        reasoning_content = self._extract_reasoning(msg)
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_args = getattr(tc.function, "arguments", None) or ""
                if raw_args.strip() == "":
                    parsed: dict = {}
                else:
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        raise ModelInvocationError(
                            f"Malformed tool-call arguments from provider"
                            f" '{self.id}': {e}"
                        ) from e
                tool_calls.append(
                    ToolCall(id=tc.id, tool_name=tc.function.name, arguments=parsed)
                )
        usage: TokenUsage | None = None
        if response.usage is not None:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )
        return AssistantMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            usage=usage,
        )

    # ── Embed ──

    async def _embed_impl(
        self, model_id: str, input: str | list[str]
    ) -> EmbeddingResult:
        response = await self._client.embeddings.create(model=model_id, input=input)
        vectors = [
            [float(v) for v in d.embedding]
            for d in sorted(response.data, key=lambda d: d.index)
        ]
        usage: TokenUsage | None = None
        if response.usage is not None:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens or 0,
            )
        return EmbeddingResult(vectors=vectors, model_id=model_id, usage=usage)
