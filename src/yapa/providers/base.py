"""Inference provider base class and utilities."""

from abc import ABC
from typing import AsyncGenerator

from openai import AsyncOpenAI

from yapa.logging import get_logger
from yapa.models import InferenceParams, Message, ModelData, StreamDelta

from .exceptions import ModelInvocationError, ModelsFetchError


class InferenceProvider(ABC):
    """Base class for inference providers."""

    def __init__(self, identifier: str, name: str, client: AsyncOpenAI):
        """
        Initialize a new inference provider.

        Args:
            identifier (str): The unique identifier for this provider.
            name (str): The human-readable name of this provider.
            client (AsyncOpenAI): The OpenAI client to use for making requests.
        """
        self._identifier = identifier
        self._client = client
        self._logger = get_logger(f"inference_provider.{self._identifier}")

    @property
    def id(self) -> str:
        """Returns the unique identifier for this provider."""
        return self._identifier

    def _filter_supported_models(self, models: list[ModelData]) -> list[ModelData]:
        """
        Filter the given list of models to those supported by this provider.

        This is a placeholder implementation that returns all models. Subclasses
        can override this method to implement provider-specific filtering logic.

        Args:
            models (list[ModelData]): The list of models to filter.

        Returns:
            list[ModelData]: The filtered list of models supported by this provider.
        """
        return models

    async def get_models(self) -> list[ModelData]:
        """
        Retrieve a list of available models for this provider.

        Returns:
            list[ModelData]: A list of available models.

        Raises:
            ModelsFetchError: If fetching models from the provider fails.
        """
        self._logger.info(f"Fetching models for provider '{self.id}'")
        try:
            models = await self._client.models.list()
            all_models = [
                ModelData(id=model.id, provider_id=self.id) for model in models.data
            ]
            return self._filter_supported_models(all_models)
        except Exception as e:
            self._logger.error(f"Failed to fetch models for provider '{self.id}': {e}")
            raise ModelsFetchError(
                f"Failed to fetch models for provider '{self.id}': {e}"
            ) from e

    async def invoke_model(
        self,
        model: str,
        messages: list[Message],
        params: InferenceParams | None = None,
    ) -> AsyncGenerator[StreamDelta, None]:
        """
        Invoke the specified model with the given messages and stream the response.

        Args:
            model (str): The model to invoke.
            messages (list[Message]): A list of messages to send to the model.
            params (InferenceParams | None): Optional inference parameters.

        Yields:
            StreamDelta: The next chunk of the response from the model.

        Raises:
            ModelInvocationError: If model invocation fails.
        """
        self._logger.info(f"Invoking model '{model}' from provider '{self.id}'.")
        try:
            async for chunk in await self._client.chat.completions.create(
                model=model,
                messages=[message.to_openai_format() for message in messages],
                temperature=params.temperature if params else None,
                max_tokens=params.max_tokens if params else None,
                top_p=params.top_p if params else None,
                stream=True,
            ):
                content: str | None = chunk.choices[0].delta.content
                reasoning_content: str | None = getattr(
                    chunk.choices[0].delta,
                    "reasoning_content",
                    getattr(chunk.choices[0].delta, "reasoning", None),
                )

                yield StreamDelta(
                    content=content, reasoning_content=reasoning_content, done=False
                )

            yield StreamDelta(content=None, reasoning_content=None, done=True)
        except Exception as e:
            self._logger.error(f"Streaming model invocation failed for '{model}': {e}")
            raise ModelInvocationError(
                f"Streaming model invocation failed for '{model}': {e}"
            ) from e
