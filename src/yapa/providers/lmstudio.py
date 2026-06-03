"""LM Studio inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import get_config

from .base import InferenceProvider


class LMStudioIP(InferenceProvider):
    """Inference provider for LM Studio."""

    def __init__(
        self,
    ):
        """
        Initialize a new LM Studio inference provider.

        Args:
            logger (logging.Logger): The logger to use for this provider.
        """
        config = get_config()
        client = AsyncOpenAI(
            api_key=config.lmstudio_api_key, base_url=config.lmstudio_base_url
        )
        super().__init__(identifier="lmstudio", name="LM Studio", client=client)
