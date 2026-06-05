"""Service for managing inference providers and models."""

from yapa.logging import get_logger
from yapa.models import ModelData
from yapa.providers import InferenceProvider, LMStudioIP, ModelsFetchError, OpenRouterIP

logger = get_logger(__name__)

DEFAULT_PROVIDERS: list[type[InferenceProvider]] = [LMStudioIP, OpenRouterIP]


class ProviderService:
    """Service for managing inference providers and models."""

    def __init__(self, providers: list[type[InferenceProvider]] | None = None) -> None:
        """Initialize the provider service."""
        self._provider_cache: dict[str, InferenceProvider] = {}

        for provider_cls in providers or DEFAULT_PROVIDERS:
            try:
                provider_instance = provider_cls()  # ty: ignore
                self._provider_cache[provider_instance.id] = provider_instance
            except ValueError as e:
                logger.warning(
                    f"Failed to initialize provider {provider_cls.__name__}: {e}"
                )
                continue

    def list_providers(self) -> list[InferenceProvider]:
        """
        List all available inference providers.

        Returns:
            list[InferenceProvider]: A list of all available inference provider
                instances.
        """
        return list(self._provider_cache.values())

    def get_provider(self, provider_id: str) -> InferenceProvider:
        """
        Get an inference provider by its identifier.

        Args:
            provider_id (str): The unique identifier of the provider to retrieve.

        Returns:
            InferenceProvider: An instance of the requested inference provider.

        Raises:
            ValueError: If no provider with the given ID is found.
        """
        provider = self._provider_cache.get(provider_id)
        if not provider:
            raise ValueError(f"Provider '{provider_id}' not found")
        return provider

    def get_provider_by_model(self, model: ModelData) -> InferenceProvider:
        """
        Get an inference provider that supports a given model.

        Args:
            model (ModelData): The model to find a provider for.

        Returns:
            InferenceProvider: An instance of the provider that supports the
                given model.

        Raises:
            ValueError: If no provider supports the given model.
        """
        provider = self._provider_cache.get(model.provider_id)
        if not provider:
            raise ValueError(f"No provider found that supports model '{model.id}'")
        return provider

    async def get_provider_by_model_id(self, model_id: str) -> InferenceProvider:
        """
        Find the provider whose model list contains the given model ID.

        Args:
            model_id: The model identifier to look up.

        Returns:
            The InferenceProvider that serves this model.

        Raises:
            ValueError: If no provider serves the given model ID.
        """
        all_models = await self.get_models()
        for pid, models in all_models.items():
            for m in models:
                if m.id == model_id:
                    return self.get_provider(pid)
        raise ValueError(f"No provider found that supports model '{model_id}'")

    async def get_models(
        self, provider_id: str | None = None
    ) -> dict[str, list[ModelData]]:
        """
        Get a list of available models for a specific provider or all providers.

        Args:
            provider_id (str | None): The ID of the provider to get models for, or
                None to get models for all providers.

        Returns:
            dict[str, list[ModelData]]: A dictionary mapping provider IDs to lists of
                available models.
        """
        if provider_id:
            provider = self.get_provider(provider_id)
            try:
                models = await provider.get_models()
                return {provider_id: models}
            except ModelsFetchError as e:
                logger.error(
                    f"Failed to fetch models for provider '{provider_id}': {e}"
                )
                return {provider_id: []}
        else:
            all_models = {}
            for pid, provider in self._provider_cache.items():
                try:
                    models = await provider.get_models()
                    all_models[pid] = models
                except ModelsFetchError as e:
                    logger.error(f"Failed to fetch models for provider '{pid}': {e}")
                    all_models[pid] = []
            return all_models

    @staticmethod
    def group_models_by_vendor(models: list[ModelData]) -> dict[str, list[ModelData]]:
        """
        Group models by vendor prefix.

        Models with a vendor prefix (e.g. 'openai/gpt-4') are grouped
        under that prefix. Models without a '/' are grouped as 'other'.

        Args:
            models: List of models to group.

        Returns:
            dict mapping vendor group names to lists of models, sorted by
            group name.
        """
        grouped: dict[str, list[ModelData]] = {}
        for model in models:
            if "/" in model.id:
                vendor = model.id.split("/")[0]
            else:
                vendor = "other"
            grouped.setdefault(vendor, []).append(model)
        return dict(sorted(grouped.items()))
