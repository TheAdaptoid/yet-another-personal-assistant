"""Inference provider protocols and related utilities."""

from .openai import OpenAIInferenceProtocol, OpenAIModelFetchProtocol
from .openrouter import OpenRouterFetchProtocol

__all__ = [
    "OpenAIInferenceProtocol",
    "OpenAIModelFetchProtocol",
    "OpenRouterFetchProtocol",
]
