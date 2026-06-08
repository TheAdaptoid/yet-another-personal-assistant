"""Protocol implementations for OpenAI-compatible providers."""

from typing import AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from yapa.models import InferenceParams, Message, ModelData, ModelType, StreamDelta

from ..base import InferenceProtocol, ModelFetchProtocol


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

        OpenAI's API doesn't support fetching detailed info for a single model,
        so we list all models and find the matching one.

        Args:
            model_id (str): The unique identifier of the model to retrieve.

        Returns:
            ModelData: Detailed information about the specified model.
        """
        model = await self.client.models.retrieve(model_id)
        formatted_model = self._format_model(model.id)
        return formatted_model


class OpenAIInferenceProtocol(InferenceProtocol):
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
            return ChatCompletionUserMessageParam(
                role=message.role, content=message.content
            )
        elif message.role == "system":
            return ChatCompletionSystemMessageParam(
                role=message.role, content=message.content
            )
        elif message.role == "assistant":
            return ChatCompletionAssistantMessageParam(
                role=message.role, content=message.content
            )
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    async def invoke_llm(
        self,
        model_id: str,
        messages: list[Message],
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """
        Invoke a language model and stream the response.

        Args:
            model_id (str): The unique identifier of the model to invoke.
            messages (list[Message]): The conversation history to provide as input.
            params (InferenceParams | None): Parameters for model inference.

        Yields:
            StreamDelta: A delta representing a chunk of the model's response.
        """
        params = params or InferenceParams()
        formatted_messages = [self._format_message(m) for m in messages]

        response_stream = await self.client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            stream=True,
            timeout=120,  # seconds
        )

        async for chunk in response_stream:
            content: str | None = chunk.choices[0].delta.content
            reasoning_content: str | None = getattr(
                chunk.choices[0].delta,
                "reasoning",
                getattr(chunk.choices[0].delta, "reasoning_content", None),
            )
            delta = StreamDelta(
                content=content, reasoning_content=reasoning_content, done=False
            )
            yield delta
