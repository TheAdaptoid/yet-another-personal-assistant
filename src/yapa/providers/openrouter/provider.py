"""OpenRouter inference provider implementation."""

from yapa.config import UNSET, Config

from ..openai_compat import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """Inference provider for OpenRouter."""

    def __init__(self, config: Config):
        """
        Initialize the OpenRouter provider.

        Args:
            config: Application config containing the OpenRouter API key.
        """
        if config.openrouter_api_key in (None, UNSET):
            raise ValueError("OpenRouter API key is not set.")
        super().__init__(
            identifier="openrouter",
            name="OpenRouter",
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
