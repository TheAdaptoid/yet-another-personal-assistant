"""OpenAI inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import UNSET, Config, get_config

from ..base import InferenceProvider
from .protocols import OpenAILLMInferenceProtocol, OpenAIModelFetchProtocol


class OpenAIIP(InferenceProvider):
    """Inference provider for OpenAI."""

    def __init__(self, config: Config | None = None):
        """
        Initialize a new OpenAI inference provider.

        Args:
            config: Optional config override.
        """
        cfg = config or get_config()

        if cfg.openai_api_key in (None, UNSET):
            raise ValueError("OpenAI API key is not set.")

        client = AsyncOpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)

        super().__init__(
            identifier="openai",
            name="OpenAI",
            model_fetcher=OpenAIModelFetchProtocol(client=client, provider_id="openai"),
            llm_invoker=OpenAILLMInferenceProtocol(client=client),
        )
