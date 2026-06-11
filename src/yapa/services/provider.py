"""Service for managing inference providers and models."""

from yapa.logging import get_logger
from yapa.models import ModelData
from yapa.providers import (
    DEFAULT_PROVIDERS,
    InferenceProvider,
    ModelsFetchError,
)

logger = get_logger(__name__)


class ProviderService:
    """Service for managing inference providers and models."""

    def __init__(self, providers: list[type[InferenceProvider]] | None = None) -> None:
        """Initialize the provider service."""
        self._provider_cache: dict[str, InferenceProvider] = {}
        self.refresh_providers(providers=providers)

    def refresh_providers(
        self, providers: list[type[InferenceProvider]] | None = None
    ) -> None:
        """
        Refresh the provider cache with a new list of providers.

        Args:
            providers: Optional list of provider classes to refresh with. If None,
                will use DEFAULT_PROVIDERS.
        """
        self._provider_cache.clear()
        for provider_cls in providers if providers is not None else DEFAULT_PROVIDERS:
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

    def get_provider_by_model_full_id(self, model_full_id: str) -> InferenceProvider:
        """
        Get an inference provider that supports a given model.

        This method expects the model_id to be in the format
        "provider_id:model_specific_id". This format can be obtained from the `full_id`
        property of the `ModelData` object.

        Args:
            model_full_id (str): The full identifier of the model to look up.

        Returns:
            The InferenceProvider that serves this model.

        Raises:
            ValueError: If no provider serves the given model ID.
        """
        provider_id, model_id = model_full_id.split(":", 1)
        provider = self._provider_cache.get(provider_id)
        if not provider:
            raise ValueError(f"No provider found that supports model '{model_id}'")
        return provider

    async def _list_models_all_providers(self) -> dict[str, list[ModelData]]:
        """
        Get a list of available models for all providers.

        Returns:
            dict[str, list[ModelData]]: A dictionary mapping provider IDs to lists of
                available models.
        """
        all_models = {}
        for pid, provider in self._provider_cache.items():
            try:
                models = await provider.list_models()
                all_models[pid] = models
            except ModelsFetchError as e:
                logger.error(f"Failed to fetch models for provider '{pid}': {e}")
                all_models[pid] = []
        return all_models

    async def _list_models_single_provider(
        self, provider_id: str
    ) -> dict[str, list[ModelData]]:
        """
        Get a list of available models for a specific provider.

        Args:
            provider_id (str): The ID of the provider to get models for.

        Returns:
            dict[str, list[ModelData]]: A dictionary with a single key (the provider ID)
                mapping to a list of available models for that provider.
        """
        provider = self.get_provider(provider_id)
        try:
            models = await provider.list_models()
            return {provider_id: models}
        except ModelsFetchError as e:
            logger.error(f"Failed to fetch models for provider '{provider_id}': {e}")
            return {provider_id: []}

    async def list_models(
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
            return await self._list_models_single_provider(provider_id)
        else:
            return await self._list_models_all_providers()

    async def get_model(self, model_full_id: str) -> ModelData:
        """
        Get detailed information about a specific model by its full ID.

        Args:
            model_full_id (str): The full identifier of the model to retrieve, in the
                format "provider_id:model_specific_id".

        Returns:
            ModelData: Detailed information about the specified model.

        Raises:
            ValueError: If no provider supports the given model ID or if fetching the
                model fails.
        """
        provider_id, model_id = model_full_id.split(":", 1)
        provider = self.get_provider(provider_id)
        try:
            return await provider.get_model(model_id=model_id)
        except ModelsFetchError as e:
            logger.error(f"Failed to fetch model '{model_full_id}': {e}")
            raise ValueError(f"Failed to fetch model '{model_full_id}': {e}") from e
