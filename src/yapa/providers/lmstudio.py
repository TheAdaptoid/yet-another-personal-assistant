"""LM Studio inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import Config, get_config

from .base import InferenceProvider


class LMStudioIP(InferenceProvider):
    """Inference provider for LM Studio."""

    def __init__(self, config: Config | None = None):
        """
        Initialize a new LM Studio inference provider.

        Args:
            config: Optional config override. Falls back to get_config().
        """
        cfg = config or get_config()
        client = AsyncOpenAI(
            api_key=cfg.lmstudio_api_key, base_url=cfg.lmstudio_base_url
        )
        super().__init__(identifier="lmstudio", name="LM Studio", client=client)
