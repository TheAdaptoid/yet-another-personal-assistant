"""Service for managing inference providers and models."""

from yapa.logging import get_logger
from yapa.models import ModelData, ModelType
from yapa.providers import (
    DEFAULT_PROVIDER_CLASSES,
    InferenceProvider,
    ModelsFetchError,
    ProviderRegistry,
)

logger = get_logger(__name__)


class ProviderService:
    """Thin service wrapping ProviderRegistry with model-fetching utilities."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        """
        Initialize the provider service.

        Args:
            registry: Provider registry. Defaults to a new ProviderRegistry
                with DEFAULT_PROVIDER_CLASSES.
        """
        self._registry = registry or ProviderRegistry(DEFAULT_PROVIDER_CLASSES)

    def get_provider(self, provider_id: str) -> InferenceProvider:
        """
        Get a provider by its identifier.

        Args:
            provider_id: The provider identifier.

        Returns:
            The provider instance.

        Raises:
            ProviderNotAvailableError: If the provider is not available.
        """
        return self._registry.get(provider_id)

    def get_provider_by_model(self, model: ModelData) -> InferenceProvider:
        """
        Get the provider that serves a given model.

        Args:
            model: The model to find a provider for.

        Returns:
            The provider instance.

        Raises:
            ProviderNotAvailableError: If no provider serves this model.
        """
        return self._registry.get(model.provider_id)

    async def list_models(
        self,
        provider_id: str | None = None,
        model_type: ModelType | None = None,
    ) -> dict[str, list[ModelData]]:
        """
        Fetch available models from one or all providers.

        Args:
            provider_id: If set, fetch only from this provider.
            model_type: Optional filter to only include models of a
                specific type.

        Returns:
            Dict mapping provider IDs to lists of models.
        """
        if provider_id:
            provider = self.get_provider(provider_id)
            try:
                models = await provider.list_models(model_type)
                return {provider_id: models}
            except ModelsFetchError as e:
                logger.error(f"Failed to fetch models for '{provider_id}': {e}")
                return {provider_id: []}

        result: dict[str, list[ModelData]] = {}
        for provider in self._registry.available:
            try:
                models = await provider.list_models(model_type)
                result[provider.id] = models
            except ModelsFetchError as e:
                logger.error(f"Failed to fetch models for '{provider.id}': {e}")
                result[provider.id] = []
        return result

    async def get_model(self, model_full_id: str) -> ModelData:
        """
        Fetch details for a specific model.

        Args:
            model_full_id: Full model ID in ``provider_id:model_id`` format.

        Returns:
            ModelData for the requested model.

        Raises:
            ValueError: If the full ID is malformed.
            ProviderNotAvailableError: If the provider is not available.
            ValueError: If the model fetch fails.
        """
        try:
            provider_id, model_id = model_full_id.split(":", 1)
        except ValueError:
            raise ValueError(
                f"Invalid model full ID '{model_full_id}': "
                "expected 'provider_id:model_id'"
            )

        provider = self.get_provider(provider_id)
        try:
            return await provider.get_model(model_id=model_id)
        except ModelsFetchError as e:
            logger.error(f"Failed to fetch model '{model_full_id}': {e}")
            raise ValueError(f"Failed to fetch model '{model_full_id}': {e}") from e
