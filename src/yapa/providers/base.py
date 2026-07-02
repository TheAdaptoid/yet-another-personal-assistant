"""Inference provider base class and utilities."""

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
from .protocols import (
    LLMInferenceProtocol,
    ModelFetchProtocol,
)


class InferenceProvider:
    """Base class for inference providers."""

    def __init__(
        self,
        identifier: str,
        name: str,
        model_fetcher: ModelFetchProtocol,
        llm_invoker: LLMInferenceProtocol,
    ) -> None:
        """
        Initialize a new inference provider.

        Args:
            identifier (str): The unique identifier for this provider.
            name (str): The human-readable name of this provider.
            model_fetcher (ModelFetchProtocol): The protocol for fetching models.
            llm_invoker (LLMInferenceProtocol): The protocol for invoking models.
        """
        self._identifier = identifier
        self._name = name
        self._model_fetcher: ModelFetchProtocol = model_fetcher
        self._llm_invoker: LLMInferenceProtocol = llm_invoker
        self._logger = get_logger(f"inference_provider.{self._identifier}")

    @property
    def id(self) -> str:
        """Returns the unique identifier for this provider."""
        return self._identifier

    @property
    def name(self) -> str:
        """Returns the human-readable name of this provider."""
        return self._name

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]:
        """
        Retrieve a list of available models for this provider.

        Args:
            model_type (ModelType | None): Optional filter for the type of models
                to list.

        Returns:
            list[ModelData]: A list of available models.

        Raises:
            ModelsFetchError: If fetching models from the provider fails.
        """
        self._logger.info("Fetching models...")
        try:
            return await self._model_fetcher.list_models(model_type=model_type)
        except Exception as e:
            self._logger.error(f"Failed to fetch models: {e}")
            raise ModelsFetchError(
                f"Failed to fetch models from provider '{self.id}': {e}"
            ) from e

    async def get_model(self, model_id: str) -> ModelData:
        """
        Retrieve detailed information about a specific model.

        Args:
            model_id (str): The unique identifier of the model to retrieve.

        Returns:
            ModelData: Detailed information about the specified model.

        Raises:
            ModelsFetchError: If fetching the model from the provider fails.
        """
        self._logger.info(f"Fetching model '{model_id}'...")
        try:
            return await self._model_fetcher.get_model(model_id=model_id)
        except Exception as e:
            self._logger.error(f"Failed to fetch model '{model_id}': {e}")
            raise ModelsFetchError(
                f"Failed to fetch model '{model_id}' from provider '{self.id}': {e}"
            ) from e

    def _pre_invoke(self, model: ModelData) -> None:
        """
        Perform any necessary pre-invocation checks or setup.

        Args:
            model (ModelData): The model to be invoked.

        Raises:
            ModelInvocationError: If the model is not suitable for invocation.
        """
        if model.type != ModelType.LLM:
            raise ModelInvocationError(f"Model '{model.id}' is not an LLM.")
        self._logger.info(f"Invoking model '{model.id}'.")

    async def invoke_llm_stream(
        self,
        model: ModelData,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """
        Invoke the specified model with the given messages and stream the response.

        Args:
            model (ModelData): The model to invoke.
            messages (list[Message]): A list of messages to send to the model.
            tools (list[Tool] | None): Optional list of tools to use.
            params (InferenceParams | None): Optional inference parameters.

        Yields:
            StreamDelta: The next chunk of the response from the model.

        Raises:
            ModelInvocationError: If model invocation fails.
        """
        self._pre_invoke(model)
        try:
            async for delta in self._llm_invoker.stream_invoke(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
            ):
                yield delta
        except Exception as e:
            self._logger.error(
                f"Streaming model invocation failed for '{model.id}': {e}"
            )
            raise ModelInvocationError(
                f"Streaming model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e
        else:
            yield StreamDelta(content=None, reasoning_content=None, done=True)

    async def invoke_llm_static(
        self,
        model: ModelData,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
    ) -> AssistantMessage:
        """
        Invoke the specified model with the given messages and return the response.

        Args:
            model (ModelData): The model to invoke.
            messages (list[Message]): A list of messages to send to the model.
            tools (list[Tool] | None): Optional list of tools to use.
            params (InferenceParams | None): Optional inference parameters.

        Returns:
            AssistantMessage: The complete response from the model.

        Raises:
            ModelInvocationError: If model invocation fails.
        """
        self._pre_invoke(model)
        try:
            response = await self._llm_invoker.static_invoke(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
            )
            return response
        except Exception as e:
            self._logger.error(f"Model invocation failed for '{model.id}': {e}")
            raise ModelInvocationError(
                f"Model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e
