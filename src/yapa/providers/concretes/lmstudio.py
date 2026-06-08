"""LM Studio inference provider implementation."""

from openai import AsyncOpenAI

from yapa.config import Config, get_config

from ..base import InferenceProvider
from ..protocols import OpenAIInferenceProtocol, OpenAIModelFetchProtocol


class LMStudioIP(InferenceProvider):
    """Inference provider for LM Studio."""

    def __init__(self, config: Config | None = None):
        """
        Initialize a new LM Studio inference provider.

        Args:
            config: Optional config override.
        """
        cfg = config or get_config()
        openai_client = AsyncOpenAI(
            api_key=cfg.lmstudio_api_key, base_url=cfg.lmstudio_base_url
        )
        model_fetcher = OpenAIModelFetchProtocol(
            client=openai_client, provider_id="lmstudio"
        )
        model_invoker = OpenAIInferenceProtocol(client=openai_client)
        super().__init__(
            identifier="lmstudio",
            name="LM Studio",
            model_fetcher=model_fetcher,
            model_invoker=model_invoker,
        )
