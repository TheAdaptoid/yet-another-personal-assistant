"""Provider registry — attempts to initialize all known providers."""

from yapa.config import Config, get_config

from .base import InferenceProvider


class ProviderNotAvailableError(Exception):
    """Requested provider is not configured or failed to initialize."""


class ProviderRegistry:
    """
    Registry that surfaces available and failed providers.

    Attempts to initialize all registered provider classes. Providers that
    fail initialization (e.g. missing API keys) are tracked separately
    rather than failing the entire registry.
    """

    def __init__(
        self,
        provider_classes: list[type[InferenceProvider]],
        config: Config | None = None,
    ) -> None:
        """
        Initialize the registry.

        Attempts to instantiate each provider class. Providers that fail
        (e.g. missing API keys) are recorded in ``failures`` rather than
        raising.

        Args:
            provider_classes: Provider classes to register.
            config: Application config. Falls back to ``get_config()``.
        """
        self._available: dict[str, InferenceProvider] = {}
        self._failures: dict[str, str] = {}

        cfg = config or get_config()
        for cls in provider_classes:
            try:
                instance = cls(config=cfg)  # type: ignore
                self._available[instance.id] = instance
            except ValueError as e:
                self._failures[cls.__name__] = str(e)

    @property
    def available(self) -> list[InferenceProvider]:
        """Providers that initialized successfully."""
        return list(self._available.values())

    @property
    def failures(self) -> dict[str, str]:
        """Providers that failed to initialize, keyed by class name."""
        return dict(self._failures)

    def is_available(self, provider_id: str) -> bool:
        """Return True if the given provider ID was initialized successfully."""
        return provider_id in self._available

    def get(self, provider_id: str) -> InferenceProvider:
        """
        Get a provider by ID.

        Args:
            provider_id: The provider identifier.

        Returns:
            The provider instance.

        Raises:
            ProviderNotAvailableError: If the provider is not available
                (unregistered, unconfigured, or failed to initialize).
        """
        if provider_id in self._failures:
            raise ProviderNotAvailableError(
                f"Provider '{provider_id}' is not configured: "
                f"{self._failures[provider_id]}"
            )
        try:
            return self._available[provider_id]
        except KeyError:
            raise ProviderNotAvailableError(
                f"Provider '{provider_id}' not found."
            )
