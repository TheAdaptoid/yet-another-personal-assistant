"""Custom exceptions for the providers package."""


class InferenceProviderError(Exception):
    """Base exception for inference provider errors."""


class ModelsFetchError(InferenceProviderError):
    """Raised when fetching models from a provider fails."""


class ModelTypeError(InferenceProviderError):
    """Raised when a model type is invalid for the requested operation."""


class ModelInvocationError(InferenceProviderError):
    """Raised when model invocation fails."""
