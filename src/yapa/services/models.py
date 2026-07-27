"""Model service — thin wrapper around ProviderRegistry."""

from yapa.logging import get_logger
from yapa.models import ModelData, ModelType
from yapa.providers import (
    DEFAULT_PROVIDER_CLASSES,
    InferenceProvider,
    ModelsFetchError,
    ProviderRegistry,
)

logger = get_logger(__name__)


class ModelService:
    """Thin wrapper around ProviderRegistry for model fetching."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        """Initialize the model service."""
        self._registry = registry or ProviderRegistry(DEFAULT_PROVIDER_CLASSES)

    def get_provider(self, provider_id: str) -> InferenceProvider:
        """Get a provider by ID."""
        return self._registry.get(provider_id)

    def get_provider_by_model(self, model: ModelData) -> InferenceProvider:
        """Get the provider that serves the given model."""
        return self._registry.get(model.provider_id)

    async def list_models(
        self,
        provider_id: str | None = None,
        model_type: ModelType | None = None,
    ) -> list[ModelData]:
        """Fetch models from one or all providers, returning a flat list."""
        if provider_id:
            provider = self.get_provider(provider_id)
            try:
                return await provider.list_models(model_type)
            except ModelsFetchError as e:
                logger.error(f"Failed to fetch models for '{provider_id}': {e}")
                return []

        results: list[ModelData] = []
        for provider in self._registry.available:
            try:
                models = await provider.list_models(model_type)
                results.extend(models)
            except ModelsFetchError as e:
                logger.error(f"Failed to fetch models for '{provider.id}': {e}")
        return results

    async def get_model(self, model_full_id: str) -> ModelData:
        """Fetch details for a specific model by full ID (provider_id:model_id)."""
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
