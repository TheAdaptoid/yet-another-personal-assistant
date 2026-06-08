"""OpenRouter inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import UNSET, Config, get_config

from ..base import InferenceProvider
from ..protocols import OpenAIInferenceProtocol, OpenRouterFetchProtocol


class OpenRouterIP(InferenceProvider):
    """Inference provider for OpenRouter."""

    def __init__(self, config: Config | None = None):
        """
        Initialize a new OpenRouter inference provider.

        Args:
            config: Optional config override.
        """
        cfg = config or get_config()

        if cfg.openrouter_api_key in (None, UNSET):
            raise ValueError("OpenRouter API key is not set.")

        model_fetcher = OpenRouterFetchProtocol(config=cfg, provider_id="openrouter")
        model_invoker = OpenAIInferenceProtocol(
            client=AsyncOpenAI(
                api_key=cfg.openrouter_api_key, base_url=cfg.openrouter_base_url
            )
        )
        super().__init__(
            identifier="openrouter",
            name="OpenRouter",
            model_fetcher=model_fetcher,
            model_invoker=model_invoker,
        )
