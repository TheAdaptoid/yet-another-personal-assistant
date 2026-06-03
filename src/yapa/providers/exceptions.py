"""Custom exceptions for the providers package."""


class InferenceProviderError(Exception):
    """Base exception for inference provider errors."""


class ModelsFetchError(InferenceProviderError):
    """Raised when fetching models from a provider fails."""


class ModelInvocationError(InferenceProviderError):
    """Raised when model invocation fails."""
