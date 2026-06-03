"""Provider manager for YAPA."""

from .base import InferenceProvider
from .exceptions import ModelsFetchError
from .lmstudio import LMStudioIP
from .openrouter import OpenRouterIP


class ProviderManager:
    """A manager class for handling inference providers."""

    @property
    def providers(self) -> list[InferenceProvider]:
        """
        Get a list of all available inference providers.

        Returns:
            list[InferenceProvider]: A list of all available inference provider
                instances.
        """
        return [LMStudioIP(), OpenRouterIP()]

    def get_provider(self, provider_id: str) -> InferenceProvider:
        """
        Get an inference provider by its identifier.

        Args:
            provider_id (str): The unique identifier of the provider to retrieve.

        Returns:
            InferenceProvider: An instance of the requested inference provider.

        Raises:
            ValueError: If no provider with the given identifier is found.
        """
        provider_map = {provider.id: provider for provider in self.providers}
        if provider_id not in provider_map:
            raise ValueError(f"Unknown provider ID: {provider_id}")
        return provider_map[provider_id]

    async def get_provider_by_model(self, model_id: str) -> InferenceProvider:
        """
        Get an inference provider that supports a given model ID.

        Args:
            model_id (str): The unique identifier of the model to find a provider for.

        Returns:
            InferenceProvider: An instance of the provider that supports the given
                model ID.

        Raises:
            ValueError: If no provider supports the given model ID.
        """
        for provider in self.providers:
            try:
                models = await provider.get_models()
                if model_id in [model.id for model in models]:
                    return provider
            except ModelsFetchError:
                continue  # If fetching models fails for a provider, skip it
        raise ValueError(f"No provider found for model ID: {model_id}")
