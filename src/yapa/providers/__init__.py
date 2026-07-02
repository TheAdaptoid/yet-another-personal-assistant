"""Inference provider implementations."""

from .base import InferenceProvider
from .exceptions import InferenceProviderError, ModelInvocationError, ModelsFetchError
from .protocols import LLMInferenceProtocol, ModelFetchProtocol


def _get_default_providers() -> list[type[InferenceProvider]]:
    """Return a list of default inference providers."""
    from .lmstudio.provider import LMStudioIP
    from .ollama.provider import OllamaIP
    from .openai.provider import OpenAIIP
    from .openrouter.provider import OpenRouterIP

    return [
        LMStudioIP,
        OllamaIP,
        OpenAIIP,
        OpenRouterIP,
    ]


DEFAULT_PROVIDERS: list[type[InferenceProvider]] = _get_default_providers()

__all__ = [
    "InferenceProvider",
    "InferenceProviderError",
    "ModelInvocationError",
    "ModelsFetchError",
    "LLMInferenceProtocol",
    "ModelFetchProtocol",
    "DEFAULT_PROVIDERS",
]
