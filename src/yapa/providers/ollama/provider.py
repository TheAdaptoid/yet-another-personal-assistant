"""Ollama inference provider implementation."""

from yapa.services.config import Config, ProviderConfig

from ..openai_compat import OpenAICompatibleProvider


class OllamaIP(OpenAICompatibleProvider):
    """Inference provider for Ollama."""

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, config: Config):
        pc = config.provider_configs.get("ollama", ProviderConfig())
        super().__init__(
            identifier="ollama",
            name="Ollama",
            api_key=pc.api_key or "",
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
