"""Ollama inference provider implementation."""

from yapa.config import Config

from ..openai_compat import OpenAICompatibleProvider


class OllamaIP(OpenAICompatibleProvider):
    """Inference provider for Ollama."""

    def __init__(self, config: Config):
        """
        Initialize the Ollama provider.

        Args:
            config: Application config containing Ollama credentials.
        """
        super().__init__(
            identifier="ollama",
            name="Ollama",
            api_key=config.ollama_api_key,
            base_url=config.ollama_base_url,
            timeout=config.provider_timeout,
        )
