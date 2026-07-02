"""Protocols for OpenRouter."""

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

from yapa.config import Config
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

from ..exceptions import ModelsFetchError
from ..protocols import LLMInferenceProtocol, ModelFetchProtocol


class OpenRouterFetchProtocol(ModelFetchProtocol):
    """Protocol for fetching models from OpenRouter."""

    def __init__(self, config: Config, provider_id: str):
        """Initialize the OpenRouter model fetch protocol."""
        self._provider_id = provider_id
        self._config = config

    def _format_model(self, model_info: Model) -> ModelData:
        """
        Format raw model data from OpenRouter into ModelData.

        Args:
            model_info (Model): The raw model information from OpenRouter.

        Returns:
            ModelData: The formatted model data.
        """
        model_id = model_info.id

        # Infer model type based on output modalities
        if "text" in model_info.architecture.output_modalities:
            inferred_type = ModelType.LLM
        else:
            inferred_type = ModelType.OTHER

        return ModelData(id=model_id, provider_id=self._provider_id, type=inferred_type)

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]:
        """Fetch available models from OpenRouter."""

        # Determine filter type for API request
        filter_type = None
        if model_type == ModelType.LLM:
            filter_type = "text"

        async with OpenRouter(
            api_key=self._config.openrouter_api_key,
            url_params={"output_modalities": filter_type} if filter_type else None,
        ) as client:
            response: ModelsListResponse = client.models.list()
            unformatted_models: list[Model] = response.data
            formatted_models = [
                self._format_model(model_info) for model_info in unformatted_models
            ]
            return formatted_models

    async def get_model(self, model_id: str) -> ModelData:
        """
        Fetch detailed information about a specific model from OpenRouter.

        The OpenRouter API does not support fetching detailed model information by ID,
        so this method retrieves the list of models and searches for the specified ID.
        """
        models = await self.list_models()
        for model in models:
            if model.id == model_id:
                return model
        raise ModelsFetchError(f"Model with ID '{model_id}' not found in OpenRouter.")


class OpenRouterLLMInferenceProtocol(LLMInferenceProtocol):
    """Implements the inference protocol using the native OpenRouter SDK."""

    def __init__(self, config: Config):
        """Initialize the OpenRouter inference protocol."""
        self._config = config

    def _format_message(self, message: Message) -> dict[str, Any]:
        """
        Convert a Message to an OpenRouter-compatible message dict.

        Args:
            message: The message to convert.

        Returns:
            A dict with keys ``role``, ``content``, and other optional fields
            depending on the message type.
        """
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
        """
        Format tools using the native OpenRouter Pydantic models.

        Args:
            tools: The tools to format.

        Returns:
            List of ChatFunctionToolFunction, or None if no tools provided.
        """
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

    def _common_pre_invoke(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Pre-invocation logic for both static and streaming invocations.

        Args:
            model_id: The model to invoke.
            messages: The messages to send.
            tools: Optional tools.
            params: Optional inference parameters.
            stream: Whether to stream.

        Returns:
            Kwargs dict for send_async.
        """
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

    async def stream_invoke(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """
        Invoke a language model and stream the response.

        Args:
            model_id: The model identifier to invoke.
            messages: The conversation history.
            tools: Optional tools for the model to use.
            params: Optional inference parameters.

        Yields:
            StreamDelta chunks.
        """
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
                    reasoning: str | None = (
                        delta.reasoning if isinstance(delta.reasoning, str) else None
                    )
                    if reasoning is not None and reasoning.strip() == "":
                        reasoning = None

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
                        done=False,
                    )

    async def static_invoke(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AssistantMessage:
        """
        Invoke a language model and return the complete response.

        Args:
            model_id: The model identifier to invoke.
            messages: The conversation history.
            tools: Optional tools for the model to use.
            params: Optional inference parameters.

        Returns:
            AssistantMessage: The complete response.
        """
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
        reasoning: str | None = (
            msg.reasoning if isinstance(msg.reasoning, str) else None
        )
        if reasoning is not None and reasoning.strip() == "":
            reasoning = None

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
