"""Ollama inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import Config, get_config

from ..base import InferenceProvider
from ..openai.protocols import OpenAILLMInferenceProtocol, OpenAIModelFetchProtocol


class OllamaIP(InferenceProvider):
    """Inference provider for Ollama."""

    def __init__(self, config: Config | None = None):
        """
        Initialize a new Ollama inference provider.

        Args:
            config: Optional config override. Falls back to get_config().
        """
        cfg = config or get_config()
        client = AsyncOpenAI(api_key=cfg.ollama_api_key, base_url=cfg.ollama_base_url)
        super().__init__(
            identifier="ollama",
            name="Ollama",
            model_fetcher=OpenAIModelFetchProtocol(client=client, provider_id="ollama"),
            llm_invoker=OpenAILLMInferenceProtocol(client=client),
        )
