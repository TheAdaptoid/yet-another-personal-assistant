"""Inference provider base class and utilities."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from yapa.logging import get_logger
from yapa.models import (
    AssistantMessage,
    InferenceParams,
    Message,
    ModelData,
    ModelType,
    StreamDelta,
)
from yapa.tools import Tool

from .exceptions import ModelInvocationError, ModelsFetchError


class InferenceProvider(ABC):
    """Abstract base class for inference providers."""

    def __init__(self, identifier: str, name: str) -> None:
        """Initialize the provider with an identifier and display name."""
        self._id = identifier
        self._name = name
        self._logger = get_logger(f"inference_provider.{identifier}")

    @property
    def id(self) -> str:
        """Return the unique identifier for this provider."""
        return self._id

    @property
    def name(self) -> str:
        """Return the human-readable name of this provider."""
        return self._name

    # ── Public methods (logging + error wrapping) ──

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]:
        """
        Retrieve available models, optionally filtered by type.

        Args:
            model_type: Optional filter to only include models of a specific type.

        Returns:
            A list of available models.

        Raises:
            ModelsFetchError: If fetching models from the provider fails.
        """
        self._logger.info("Fetching models...")
        try:
            return await self._list_models_impl(model_type)
        except ModelsFetchError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to fetch models: {e}")
            raise ModelsFetchError(
                f"Failed to fetch models from provider '{self.id}': {e}"
            ) from e

    async def get_model(self, model_id: str) -> ModelData:
        """
        Retrieve detailed information about a specific model.

        Args:
            model_id: The unique identifier of the model to retrieve.

        Returns:
            Detailed information about the specified model.

        Raises:
            ModelsFetchError: If fetching the model from the provider fails.
        """
        self._logger.info(f"Fetching model '{model_id}'...")
        try:
            return await self._get_model_impl(model_id)
        except ModelsFetchError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to fetch model '{model_id}': {e}")
            raise ModelsFetchError(
                f"Failed to fetch model '{model_id}' from provider '{self.id}': {e}"
            ) from e

    async def stream_chat(
        self,
        model: ModelData,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """
        Invoke the model and stream the response.

        Args:
            model: The model to invoke.
            messages: The conversation history.
            tools: Optional tools the model may use.
            params: Optional inference parameters.

        Yields:
            StreamDelta chunks from the model response.

        Raises:
            ModelInvocationError: If model invocation fails.
        """
        self._pre_invoke_check(model)
        try:
            async for delta in self._stream_chat_impl(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
            ):
                yield delta
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(
                f"Streaming model invocation failed for '{model.id}': {e}"
            )
            raise ModelInvocationError(
                f"Streaming model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e

    async def static_chat(
        self,
        model: ModelData,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AssistantMessage:
        """
        Invoke the model and return the complete response.

        Args:
            model: The model to invoke.
            messages: The conversation history.
            tools: Optional tools the model may use.
            params: Optional inference parameters.

        Returns:
            The complete assistant response.

        Raises:
            ModelInvocationError: If model invocation fails.
        """
        self._pre_invoke_check(model)
        try:
            return await self._static_chat_impl(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
            )
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(f"Model invocation failed for '{model.id}': {e}")
            raise ModelInvocationError(
                f"Model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e

    def _pre_invoke_check(self, model: ModelData) -> None:
        if model.type != ModelType.LLM:
            raise ModelInvocationError(f"Model '{model.id}' is not an LLM.")

    # ── Private implementation methods ──

    @abstractmethod
    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]: ...

    @abstractmethod
    async def _get_model_impl(self, model_id: str) -> ModelData: ...

    @abstractmethod
    def _stream_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]: ...

    @abstractmethod
    async def _static_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AssistantMessage: ...
