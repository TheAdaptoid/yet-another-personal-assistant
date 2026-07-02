"""Protocol implementations for OpenAI-compatible providers."""

import json
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

from ..protocols import LLMInferenceProtocol, ModelFetchProtocol


class OpenAIModelFetchProtocol(ModelFetchProtocol):
    """Implements the model fetching protocol for OpenAI."""

    def __init__(self, client: AsyncOpenAI, provider_id: str):
        """Initialize the OpenAI model fetch protocol."""

        self.client = client
        self.provider_id = provider_id

    def _format_model(self, model_id: str) -> ModelData:
        """
        Format raw model data into ModelData.

        OpensAI's API doesn't allow fetching detailed model info,
        so we infer the type from the ID.

        Args:
            model_id (str): The raw model identifier.

        Returns:
            ModelData: The formatted model data with inferred type.
        """

        model_type_keywords = ["embed", "audio", "image"]
        if any(kw in model_id for kw in model_type_keywords):
            inferred_type = ModelType.OTHER
        else:
            inferred_type = ModelType.LLM

        return ModelData(id=model_id, provider_id=self.provider_id, type=inferred_type)

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]:
        """
        Retrieve a list of available models for this provider.

        Args:
            model_type (ModelType | None): Optional filter for the type of models
                to list.

        Returns:
            list[ModelData]: A list of available models.
        """
        unformatted_models = await self.client.models.list()
        formatted_models = [self._format_model(m.id) for m in unformatted_models.data]

        if model_type:
            filtered_models = [m for m in formatted_models if m.type == model_type]
            return filtered_models
        else:
            return formatted_models

    async def get_model(self, model_id: str) -> ModelData:
        """
        Retrieve detailed information about a specific model.

        Args:
            model_id (str): The unique identifier of the model to retrieve.

        Returns:
            ModelData: Detailed information about the specified model.
        """
        model = await self.client.models.retrieve(model_id)
        formatted_model = self._format_model(model.id)
        return formatted_model


class OpenAILLMInferenceProtocol(LLMInferenceProtocol):
    """Implements the inference protocol for OpenAI."""

    def __init__(self, client: AsyncOpenAI):
        """Initialize the OpenAI inference protocol."""
        self.client = client

    def _format_message(self, message: Message) -> ChatCompletionMessageParam:
        """
        Convert a Message to the appropriate OpenAI ChatCompletionMessageParam.

        Args:
            message (Message): The message to convert.

        Returns:
            ChatCompletionMessageParam: A dictionary with keys "role", "content", and
            other optional fields depending on the message type.
        """
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
        """
        Format a list of Tool objects for the OpenAI API.

        Args:
            tools (list[Tool] | None): The tools to format.

        Returns:
            list[dict] | None: The formatted tools, or None if no tools provided.
        """
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
        """
        Extract reasoning content from a given object.

        Args:
            obj (object): The object from which to extract reasoning content.

        Returns:
            str | None: The extracted reasoning content, or None if not present.
        """
        text: str | None = getattr(obj, "reasoning", None) or getattr(
            obj, "reasoning_content", None
        )
        if text is not None and text.strip() == "":
            return None
        return text

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
            model_id (str): The unique identifier of the model to invoke.
            messages (list[Message]): The list of messages to send to the model.
            tools (list[Tool] | None): Optional list of tools to use.
            params (InferenceParams | None): Optional inference parameters.
            stream (bool): Whether to stream the response.
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
            timeout=120,  # seconds
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
            model_id (str): The unique identifier of the model to invoke.
            messages (list[Message]): The conversation history to provide as input.
            tools (list[Tool] | None): A list of tools available for the model to use.
            params (InferenceParams | None): Parameters for model inference.

        Yields:
            StreamDelta: A delta representing a chunk of the model's response.
        """
        kwargs = self._common_pre_invoke(
            model_id=model_id,
            messages=messages,
            tools=tools,
            params=params,
            stream=True,
        )

        response_stream: AsyncStream[
            ChatCompletionChunk
        ] = await self.client.chat.completions.create(**kwargs)

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
            model_id (str): The unique identifier of the model to invoke.
            messages (list[Message]): The conversation history to provide as input.
            tools (list[Tool] | None): A list of tools available for the model to use.
            params (InferenceParams | None): Parameters for model inference.

        Returns:
            AssistantMessage: The complete response from the model.
        """
        kwargs = self._common_pre_invoke(
            model_id=model_id,
            messages=messages,
            tools=tools,
            params=params,
            stream=False,
        )

        response: ChatCompletion = await self.client.chat.completions.create(**kwargs)

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
