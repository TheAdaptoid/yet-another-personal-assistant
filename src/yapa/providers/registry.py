"""Provider registry — attempts to initialize all known providers."""

import logging

from yapa.services.config import Config, JsonConfigStore

from .base import InferenceProvider

logger = logging.getLogger(__name__)


class ProviderNotAvailableError(Exception):
    """Requested provider is not configured or failed to initialize."""


class ProviderRegistry:
    """
    Registry that surfaces available and failed providers.

    Providers that fail initialization are keyed by provider id when one could
    be determined, else by class name (see REQ-PROV-06). The id comes from a
    per-provider class constant ``PROVIDER_ID``. A provider whose constructor
    raises is recorded in ``failures`` and logged at error level.
    """

    def __init__(
        self,
        provider_classes: list[type[InferenceProvider]],
        config: Config | None = None,
    ) -> None:
        """
        Initialize the registry, attempting each provider class.

        Providers that fail (e.g. missing API keys) are recorded in
        ``failures`` and logged rather than failing the whole registry. When a
        provider id can be determined (via the ``PROVIDER_ID`` class constant),
        the failure is keyed by that id; otherwise it falls back to the class
        name.

        Args:
            provider_classes: Provider classes to register.
            config: Application config. Falls back to ``JsonConfigStore``.
        """
        self._available: dict[str, InferenceProvider] = {}
        self._failures: dict[str, str] = {}

        cfg = config or JsonConfigStore().load()
        for cls in provider_classes:
            self._register(cls, cfg)

    def _register(self, cls: type[InferenceProvider], cfg: Config) -> None:
        key = cls.__name__
        try:
            instance = cls(config=cfg)  # type: ignore
            self._available[instance.id] = instance
        except Exception as e:
            id_key = getattr(cls, "PROVIDER_ID", None)
            if id_key is not None:
                key = id_key
            self._failures[key] = str(e)
            logger.error("Provider %s failed to initialize: %s", key, e)

    @property
    def available(self) -> list[InferenceProvider]:
        """Providers that initialized successfully."""
        return list(self._available.values())

    @property
    def failures(self) -> dict[str, str]:
        """Providers that failed to initialize, keyed by id or class name."""
        return dict(self._failures)

    def is_available(self, provider_id: str) -> bool:
        """Return True if the given provider ID was initialized successfully."""
        return provider_id in self._available

    def get(self, provider_id: str) -> InferenceProvider:
        """
        Return an available provider or raise with a diagnostic message.

        Args:
            provider_id: The provider identifier.

        Returns:
            The provider instance.

        Raises:
            ProviderNotAvailableError: If the provider is unregistered or
                failed to initialize, with the stored failure reason when
                applicable.
        """
        if provider_id in self._available:
            return self._available[provider_id]
        reason = self._failures.get(provider_id)
        if reason is not None:
            raise ProviderNotAvailableError(
                f"Provider '{provider_id}' failed to initialize: {reason}"
            )
        raise ProviderNotAvailableError(f"Provider '{provider_id}' is unknown.")
