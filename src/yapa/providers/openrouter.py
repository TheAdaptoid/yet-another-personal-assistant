"""OpenRouter inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import UNSET, get_config

from .base import InferenceProvider


class OpenRouterIP(InferenceProvider):
    """Inference provider for OpenRouter."""

    def __init__(self):
        """
        Initialize a new OpenRouter inference provider.

        Args:
            logger (logging.Logger): The logger to use for this provider.
        """
        config = get_config()

        if config.openrouter_api_key in (None, UNSET):
            raise ValueError("OpenRouter API key is not set.")

        client = AsyncOpenAI(
            api_key=config.openrouter_api_key, base_url=config.openrouter_base_url
        )
        super().__init__(identifier="openrouter", name="OpenRouter", client=client)
