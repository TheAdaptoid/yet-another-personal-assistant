"""Exceptions for inference operations."""


class InferenceError(Exception):
    """Raised when a model invocation fails."""

    def __init__(
        self,
        message: str,
        model_id: str | None = None,
        cause: Exception | None = None,
    ):
        """
        Initialize an InferenceError.

        Args:
            message: The error message.
            model_id: The ID of the model that failed.
            cause: The underlying cause of this error.
        """
        self.model_id = model_id
        self.cause = cause
        super().__init__(message)
        if cause:
            self.__cause__ = cause


class ModelNotFoundError(Exception):
    """Raised when a model is not found in any available provider."""

    def __init__(self, model_id: str):
        """
        Initialize a ModelNotFoundError.

        Args:
            model_id: The ID of the model that was not found.
        """
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found in any provider")


class ModelsFetchError(Exception):
    """Raised when fetching available models from a provider fails."""

    def __init__(self, provider: str, cause: Exception | None = None):
        """
        Initialize a ModelsFetchError.

        Args:
            provider: The name of the provider that failed.
            cause: The underlying cause of this error.
        """
        self.provider = provider
        self.cause = cause
        super().__init__(f"Failed to fetch models from {provider}")
        if cause:
            self.__cause__ = cause