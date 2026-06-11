"""Inference provider implementations."""

from .base import InferenceProvider
from .concretes import LMStudioIP, OllamaIP, OpenRouterIP
from .exceptions import InferenceProviderError, ModelInvocationError, ModelsFetchError

DEFAULT_PROVIDERS: list[type[InferenceProvider]] = [
    LMStudioIP,
    OllamaIP,
    OpenRouterIP,
]

__all__ = [
    "InferenceProvider",
    "InferenceProviderError",
    "ModelInvocationError",
    "ModelsFetchError",
    "LMStudioIP",
    "OllamaIP",
    "OpenRouterIP",
    "DEFAULT_PROVIDERS",
]
