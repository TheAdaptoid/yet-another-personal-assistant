"""Defines protocols for decoupling inference provider logic."""

from typing import AsyncGenerator, Protocol

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    Message,
    ModelData,
    ModelType,
    StreamDelta,
)
from yapa.tools import Tool


class ModelFetchProtocol(Protocol):
    """Defines the protocol for fetching models from a provider."""

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]:
        """
        Retrieve a list of available models for this provider.

        Args:
            model_type (ModelType | None): Optional filter for the type of models
                to list.

        Returns:
            list[ModelData]: A list of available models.
        """
        ...

    async def get_model(self, model_id: str) -> ModelData:
        """
        Retrieve detailed information about a specific model.

        Args:
            model_id (str): The unique identifier of the model to retrieve.

        Returns:
            ModelData: Detailed information about the specified model.
        """
        ...


class LLMInferenceProtocol(Protocol):
    """Defines protocols for invoking an LLM."""

    async def static_invoke(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AssistantMessage:
        """
        Invoke the model with the given list of messages.

        This method returns a single `AssistantMessage` representing an aggregated
        version of the model's response.

        Args:
            model_id (str): The unique identifier of the model to invoke.
            messages (list[Message]): The list of messages to send to the model.
            tools (list[Tool] | None): Optional list of tools to use.
            params (InferenceParams | None): Optional inference parameters.

        Returns:
            AssistantMessage: The model's response.
        """
        ...

    def stream_invoke(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """
        Invoke the model and stream the response as deltas.

        Each yielded StreamDelta represents an incremental update. The final
        delta MUST have `done=True` to signal the end of the stream.

        Args:
            model_id: The model identifier to invoke.
            messages: The conversation history.
            tools: Optional tools the model may use.
            params: Optional inference parameters.

        Yields:
            StreamDelta chunks, with a final `StreamDelta(done=True)`.
        """
        ...
