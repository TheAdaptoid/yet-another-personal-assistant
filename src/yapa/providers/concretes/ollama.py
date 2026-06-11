"""Ollama inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import Config, get_config

from ..base import InferenceProvider
from ..protocols import OpenAIInferenceProtocol, OpenAIModelFetchProtocol


class OllamaIP(InferenceProvider):
    """Inference provider for Ollama."""

    def __init__(self, config: Config | None = None):
        """
        Initialize a new Ollama inference provider.

        Args:
            config: Optional config override. Falls back to get_config().
        """
        cfg = config or get_config()

        openai_client = AsyncOpenAI(
            api_key=cfg.ollama_api_key, base_url=cfg.ollama_base_url
        )
        model_fetcher = OpenAIModelFetchProtocol(
            client=openai_client, provider_id="ollama"
        )
        model_invoker = OpenAIInferenceProtocol(client=openai_client)
        super().__init__(
            identifier="ollama",
            name="Ollama",
            model_fetcher=model_fetcher,
            model_invoker=model_invoker,
        )
