"""Inference provider base class and utilities."""

import logging
from abc import abstractmethod

from openai import AsyncOpenAI

from yapa.core.inference.exceptions import InferenceError
from yapa.shared.models import AssistantMessage, InferenceParams, Message, ModelData


class InferenceProvider:
    """Base class for inference providers."""

    def __init__(
        self, logger: logging.Logger, identifier: str, name: str, client: AsyncOpenAI
    ):
        """
        Initialize a new inference provider.

        Args:
            logger (logging.Logger): The logger to use for this provider.
            identifier (str): The unique identifier for this provider.
            name (str): The human-readable name of this provider.
            client (AsyncOpenAI): The OpenAI client to use for making requests.
        """
        self._logger = logger
        self._identifier = identifier
        self._name = name
        self._client = client

    @property
    def id(self) -> str:
        """Returns the unique identifier for this provider."""
        return self._identifier

    @property
    def name(self) -> str:
        """Returns the human-readable name of this provider."""
        return self._name

    @abstractmethod
    async def get_models(self) -> list[ModelData]:
        """
        Retrieve a list of available models for this provider.

        Returns:
            list[ModelData]: A list of available models.

        Raises:
            ModelsFetchError: If fetching models from the provider fails.
        """
        pass

    async def invoke_model(
        self,
        model: ModelData,
        messages: list[Message],
        params: InferenceParams | None = None,
    ) -> AssistantMessage:
        """
        Invoke the specified model with the given messages.

        Args:
            model (ModelData): The model to invoke.
            messages (list[Message]): A list of messages to send to the model.
            params (InferenceParams | None): Optional inference parameters.

        Returns:
            AssistantMessage: The response from the model.

        Raises:
            InferenceError: If the model invocation fails.
        """
        self._logger.info(f"Invoking model '{model.id}' with {len(messages)} messages.")
        try:
            response = await self._client.chat.completions.create(
                model=model.id,
                messages=[message.to_openai_format() for message in messages],
                temperature=params.temperature if params else None,
                max_tokens=params.max_tokens if params else None,
                top_p=params.top_p if params else None,
            )
        except Exception as e:
            self._logger.error(f"Model invocation failed for '{model.id}': {e}")
            raise InferenceError(
                f"Failed to invoke model '{model.id}'", model_id=model.id, cause=e
            )

        assistant_message = AssistantMessage.from_openai_format(
            response.choices[0].message, model_id=model.id
        )
        self._logger.info(f"Received response from model '{model.id}'.")
        return assistant_message
