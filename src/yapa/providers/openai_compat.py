"""Shared implementation for OpenAI-compatible inference providers."""

import json
from abc import ABC
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from yapa.config import DEFAULT_PROVIDER_TIMEOUT
from yapa.models import (
    AssistantMessage,
    InferenceParams,
    Message,
    ModelData,
    ModelType,
    StreamDelta,
    ToolCall,
    ToolCallDelta,
    ToolMessage,
)
from yapa.tools import Tool

from .base import InferenceProvider


class OpenAICompatibleProvider(InferenceProvider, ABC):
    """
    Base provider for any OpenAI-compatible API.

    Supports OpenAI, LM Studio, Ollama, and any other service that
    exposes an OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        identifier: str,
        name: str,
        api_key: str,
        base_url: str | None,
        timeout: int = DEFAULT_PROVIDER_TIMEOUT,
    ) -> None:
        """
        Initialize the OpenAI-compatible provider.

        Args:
            identifier: Unique provider identifier (e.g. 'openai').
            name: Human-readable provider name.
            api_key: API key for the provider.
            base_url: Base URL for the API endpoint.
            timeout: Timeout in seconds for API calls.
        """
        super().__init__(identifier, name)
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # ── Model fetching ──

    def _format_model(self, model_id: str) -> ModelData:
        model_type_keywords = ["embed", "audio", "image"]
        if any(kw in model_id for kw in model_type_keywords):
            inferred_type = ModelType.OTHER
        else:
            inferred_type = ModelType.LLM
        return ModelData(id=model_id, provider_id=self.id, type=inferred_type)

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]:
        response = await self._client.models.list()
        formatted = [self._format_model(m.id) for m in response.data]
        if model_type:
            return [m for m in formatted if m.type == model_type]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        model = await self._client.models.retrieve(model_id)
        return self._format_model(model.id)

    # ── Message formatting ──

    def _format_message(self, message: Message) -> ChatCompletionMessageParam:
        if message.role == "user":
            if message.content is None:
                raise ValueError("User message content cannot be None.")
            return ChatCompletionUserMessageParam(
                role=message.role, content=message.content
            )
        elif message.role == "system":
            if message.content is None:
                raise ValueError("System message content cannot be None.")
            return ChatCompletionSystemMessageParam(
                role=message.role, content=message.content
            )
        elif isinstance(message, AssistantMessage):
            msg = ChatCompletionAssistantMessageParam(
                role=message.role, content=message.content
            )
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
            return msg
        elif isinstance(message, ToolMessage):
            return ChatCompletionToolMessageParam(
                role=message.role,
                tool_call_id=message.tool_call_id,
                content=message.content or "",
            )
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    def _format_tools(self, tools: list[Tool] | None) -> list[dict] | None:
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

    def _extract_reasoning_content(self, obj: object) -> str | None:
        text: str | None = getattr(obj, "reasoning", None) or getattr(
            obj, "reasoning_content", None
        )
        if text is not None and text.strip() == "":
            return None
        return text

    # ── Inference ──

    def _common_pre_invoke(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        params = params or InferenceParams()
        formatted_messages = [self._format_message(m) for m in messages]
        kwargs: dict[str, Any] = dict(
            model=model_id,
            messages=formatted_messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            stream=stream,
        )
        formatted_tools = self._format_tools(tools)
        if formatted_tools is not None:
            kwargs["tools"] = formatted_tools
        return kwargs

    async def _stream_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        kwargs = self._common_pre_invoke(
            model_id=model_id,
            messages=messages,
            tools=tools,
            params=params,
            stream=True,
        )

        response_stream: AsyncStream[ChatCompletionChunk] = (
            await self._client.chat.completions.create(**kwargs)
        )

        async for chunk in response_stream:
            delta = chunk.choices[0].delta
            content = delta.content
            reasoning_content = self._extract_reasoning_content(delta)

            tool_call_deltas: list[ToolCallDelta] = []
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    tool_call_deltas.append(
                        ToolCallDelta(
                            index=tc.index,
                            id=tc.id,
                            name=tc.function.name if tc.function else None,
                            arguments=tc.function.arguments if tc.function else None,
                        )
                    )

            yield StreamDelta(
                content=content,
                reasoning_content=reasoning_content,
                tool_calls=tool_call_deltas,
            )

    async def _static_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AssistantMessage:
        kwargs = self._common_pre_invoke(
            model_id=model_id,
            messages=messages,
            tools=tools,
            params=params,
            stream=False,
        )

        response: ChatCompletion = await self._client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        content = msg.content
        reasoning_content = self._extract_reasoning_content(msg)

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, ChatCompletionMessageCustomToolCall):
                    raise ValueError(
                        "Tool call in static response is not a function call."
                    )
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        tool_name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return AssistantMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
