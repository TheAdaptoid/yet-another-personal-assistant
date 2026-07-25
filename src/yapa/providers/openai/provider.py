"""OpenAI inference provider implementation."""

from yapa.config import UNSET, Config

from ..openai_compat import OpenAICompatibleProvider


class OpenAIIP(OpenAICompatibleProvider):
    """Inference provider for OpenAI."""

    def __init__(self, config: Config):
        """
        Initialize the OpenAI provider.

        Args:
            config: Application config containing the OpenAI API key and base URL.
        """
        if config.openai_api_key in (None, UNSET):
            raise ValueError("OpenAI API key is not set.")
        super().__init__(
            identifier="openai",
            name="OpenAI",
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            timeout=config.provider_timeout,
        )
