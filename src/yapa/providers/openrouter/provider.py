"""OpenRouter inference provider implementation."""

import json
from typing import Any, AsyncGenerator

from openrouter import OpenRouter
from openrouter.components import (
    ChatFunctionToolFunction,
    ChatFunctionToolFunctionFunction,
    ChatResult,
    Model,
    ModelsListResponse,
)

from yapa.config import UNSET, Config
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

from ..base import InferenceProvider
from ..exceptions import ModelsFetchError


class OpenRouterProvider(InferenceProvider):
    """Inference provider for OpenRouter."""

    def __init__(self, config: Config):
        """
        Initialize the OpenRouter provider.

        Args:
            config: Application config containing the OpenRouter API key.
        """
        if config.openrouter_api_key in (None, UNSET):
            raise ValueError("OpenRouter API key is not set.")
        super().__init__(identifier="openrouter", name="OpenRouter")
        self._config = config
        self._timeout = config.provider_timeout

    # ── Model fetching ──

    def _format_model(self, model_info: Model) -> ModelData:
        model_id = model_info.id
        if "text" in model_info.architecture.output_modalities:
            inferred_type = ModelType.LLM
        else:
            inferred_type = ModelType.OTHER
        return ModelData(id=model_id, provider_id=self.id, type=inferred_type)

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]:
        filter_type = None
        if model_type == ModelType.LLM:
            filter_type = "text"

        async with OpenRouter(
            api_key=self._config.openrouter_api_key,
            url_params={"output_modalities": filter_type} if filter_type else None,
        ) as client:
            response: ModelsListResponse = await client.models.list_async()
            formatted = [self._format_model(m) for m in response.data]
            return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        raise ModelsFetchError(f"Model with ID '{model_id}' not found in OpenRouter.")

    # ── Message formatting ──

    def _format_message(self, message: Message) -> dict[str, Any]:
        if message.role == "user":
            if message.content is None:
                raise ValueError("User message content cannot be None.")
            return {"role": "user", "content": message.content}
        elif message.role == "system":
            if message.content is None:
                raise ValueError("System message content cannot be None.")
            return {"role": "system", "content": message.content}
        elif isinstance(message, AssistantMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": message.content}
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
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id,
                "content": message.content or "",
            }
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    def _format_tools(
        self, tools: list[Tool] | None
    ) -> list[ChatFunctionToolFunction] | None:
        if not tools:
            return None
        return [
            ChatFunctionToolFunction(
                type="function",
                function=ChatFunctionToolFunctionFunction(
                    name=t.name,
                    description=t.description,
                    parameters=t.parameters,
                ),
            )
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
            timeout_ms=self._timeout * 1000,
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

        async with OpenRouter(api_key=self._config.openrouter_api_key) as client:
            stream = await client.chat.send_async(**kwargs)

            async for chunk in stream:
                for choice in chunk.choices:
                    delta = choice.delta
                    content: str | None = (
                        delta.content if isinstance(delta.content, str) else None
                    )
                    reasoning: str | None = self._extract_reasoning_content(delta)

                    tool_call_deltas: list[ToolCallDelta] = []
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            tool_call_deltas.append(
                                ToolCallDelta(
                                    index=tc.index,
                                    id=tc.id,
                                    name=tc.function.name if tc.function else None,
                                    arguments=(
                                        tc.function.arguments
                                        if tc.function
                                        else None
                                    ),
                                )
                            )

                    yield StreamDelta(
                        content=content,
                        reasoning_content=reasoning,
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

        async with OpenRouter(api_key=self._config.openrouter_api_key) as client:
            response: ChatResult = await client.chat.send_async(**kwargs)

        msg = response.choices[0].message
        content: str | None = (
            msg.content if isinstance(msg.content, str) else None
        )
        reasoning: str | None = self._extract_reasoning_content(msg)

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
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
            reasoning_content=reasoning,
            tool_calls=tool_calls,
        )
