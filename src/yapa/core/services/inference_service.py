"""Service for managing inference providers and model invocations."""

import logging

from yapa.core.inference.exceptions import ModelNotFoundError, ModelsFetchError
from yapa.core.inference.provider import InferenceProvider
from yapa.core.inference.providers import LMStudioIP, OpenRouterIP
from yapa.shared.models import AssistantMessage, InferenceParams, Message, ModelData


class InferenceService:
    """Service for managing inference providers and model invocations."""

    def __init__(self, logger: logging.Logger, providers: list[InferenceProvider]):
        """
        Initialize the inference service.

        Args:
            logger (logging.Logger): Logger instance for recording operations.
            providers (list[InferenceProvider]): List of inference providers to manage.
        """
        self._logger = logger
        self._providers = {provider.id: provider for provider in providers}
        self._logger.debug(
            msg=(
                "InferenceService initialized with providers: "
                f"{list(self._providers.keys())}"
            )
        )

    async def get_models(self) -> list[ModelData]:
        """
        Retrieve available models from all providers.

        Returns:
            list[ModelData]: A list of all available models.
        """
        models = []
        for provider in self._providers.values():
            try:
                models.extend(await provider.get_models())
            except ModelsFetchError:
                self._logger.warning(
                    "Skipping provider '%s' due to fetch error", provider.name
                )
        return models

    def _find_model_provider(self, model: ModelData) -> InferenceProvider:
        """
        Find the provider that offers the specified model.

        Args:
            model (ModelData): The model to find.

        Returns:
            InferenceProvider: The provider that offers the specified model.

        Raises:
            ModelNotFoundError: If no provider offers the specified model.
        """
        if model.provider_id in self._providers:
            return self._providers[model.provider_id]
        raise ModelNotFoundError(model.id)

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
        """
        ip = self._find_model_provider(model)
        return await ip.invoke_model(model, messages, params)

    @classmethod
    def create_default(cls, logger: logging.Logger) -> "InferenceService":
        """
        Create an InferenceService instance with default providers.

        Args:
            logger (logging.Logger): Logger instance for recording operations.

        Returns:
            InferenceService: An instance of InferenceService with default providers.
        """
        providers: list[InferenceProvider] = [
            LMStudioIP(logger),
            OpenRouterIP(logger),
        ]
        return cls(logger, providers)
