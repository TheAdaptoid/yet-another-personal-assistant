"""LM Studio inference provider implementation."""

from yapa.config import Config

from ..openai_compat import OpenAICompatibleProvider


class LMStudioIP(OpenAICompatibleProvider):
    """Inference provider for LM Studio."""

    def __init__(self, config: Config):
        """
        Initialize the LM Studio provider.

        Args:
            config: Application config containing LM Studio credentials.
        """
        super().__init__(
            identifier="lmstudio",
            name="LM Studio",
            api_key=config.lmstudio_api_key,
            base_url=config.lmstudio_base_url,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
