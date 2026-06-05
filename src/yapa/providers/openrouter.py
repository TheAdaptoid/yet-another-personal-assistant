"""OpenRouter inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import UNSET, Config, get_config
from yapa.models import ModelData

from .base import InferenceProvider

SUPPORTED_MODEL_PREFIXES = [
    "deepseek",
    "google",
    "openai",
    "anthropic",
    "moonshot",
]


class OpenRouterIP(InferenceProvider):
    """Inference provider for OpenRouter."""

    def __init__(self, config: Config | None = None):
        """
        Initialize a new OpenRouter inference provider.

        Args:
            config: Optional config override. Falls back to get_config().
        """
        cfg = config or get_config()

        if cfg.openrouter_api_key in (None, UNSET):
            raise ValueError("OpenRouter API key is not set.")

        client = AsyncOpenAI(
            api_key=cfg.openrouter_api_key, base_url=cfg.openrouter_base_url
        )
        super().__init__(identifier="openrouter", name="OpenRouter", client=client)

    def _filter_supported_models(self, models: list[ModelData]) -> list[ModelData]:
        """
        Filter the given list of models to those supported by this provider.

        By default, OpenRouter supports ~350 models from various providers. For
        simplicity, this implementation filters to a curated subset of popular models.

        Args:
            models: The list of models to filter.

        Returns:
            The filtered list of models supported by OpenRouter.
        """
        supported = []
        for model in models:
            if any(model.id.startswith(prefix) for prefix in SUPPORTED_MODEL_PREFIXES):
                supported.append(model)
        return supported
