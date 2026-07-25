"""Inference provider implementations."""

from .base import InferenceProvider
from .exceptions import InferenceProviderError, ModelInvocationError, ModelsFetchError
from .lmstudio import LMStudioIP
from .ollama import OllamaIP
from .openai import OpenAIIP
from .openrouter import OpenRouterProvider
from .registry import ProviderNotAvailableError, ProviderRegistry

DEFAULT_PROVIDER_CLASSES: list[type[InferenceProvider]] = [
    OpenAIIP,
    LMStudioIP,
    OllamaIP,
    OpenRouterProvider,
]

__all__ = [
    "DEFAULT_PROVIDER_CLASSES",
    "InferenceProvider",
    "InferenceProviderError",
    "LMStudioIP",
    "ModelInvocationError",
    "ModelsFetchError",
    "OllamaIP",
    "OpenAIIP",
    "OpenRouterProvider",
    "ProviderNotAvailableError",
    "ProviderRegistry",
]
