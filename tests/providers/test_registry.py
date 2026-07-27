"""Tests for ProviderRegistry."""

from unittest.mock import patch

import pytest

from yapa.models import AssistantMessage, ModelData, ModelType, StreamDelta
from yapa.providers.base import InferenceProvider
from yapa.providers.registry import ProviderNotAvailableError, ProviderRegistry


class _MockProv(InferenceProvider):
    """Provider that initializes successfully."""

    def __init__(self, config=None):
        super().__init__("mock", "Mock Provider")

    async def _list_models_impl(self, model_type=None):
        return []

    async def _get_model_impl(self, model_id):
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None):
        yield StreamDelta(content="test")

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None):
        return AssistantMessage(content="test", role="assistant")


class _MockProvB(InferenceProvider):
    """Another provider that initializes successfully."""

    def __init__(self, config=None):
        super().__init__("prov_b", "Provider B")

    async def _list_models_impl(self, model_type=None):
        return []

    async def _get_model_impl(self, model_id):
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None):
        yield StreamDelta(content="test")

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None):
        return AssistantMessage(content="test", role="assistant")


class _FailingProv(InferenceProvider):
    """Provider that fails to initialize."""

    def __init__(self, config=None):
        raise ValueError("Missing API key")

    async def _list_models_impl(self, model_type=None):
        raise RuntimeError("should not be called")

    async def _get_model_impl(self, model_id):
        raise RuntimeError("should not be called")

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None):
        raise RuntimeError("should not be called")
        yield  # pragma: no cover

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None):
        raise RuntimeError("should not be called")


class TestInit:
    """Tests for ProviderRegistry.__init__()."""

    def test_loads_persisted_config_when_not_provided(self) -> None:
        with patch("yapa.providers.registry.JsonConfigStore") as mock_store_cls:
            mock_store = mock_store_cls.return_value
            mock_store.load.return_value = None

            ProviderRegistry([])

            mock_store.load.assert_called_once()

    def test_all_succeed(self) -> None:
        registry = ProviderRegistry([_MockProv, _MockProvB])
        assert len(registry.available) == 2
        assert len(registry.failures) == 0

    def test_some_fail(self) -> None:
        registry = ProviderRegistry([_MockProv, _FailingProv])
        assert len(registry.available) == 1
        assert registry.available[0].id == "mock"
        assert len(registry.failures) == 1
        assert "Missing API key" in registry.failures["_FailingProv"]

    def test_captures_non_value_error_init_failures(self) -> None:
        class _RuntimeFailingProv(InferenceProvider):
            def __init__(self, config=None):
                raise RuntimeError("boom")

            async def _list_models_impl(self, model_type=None):
                raise RuntimeError("should not be called")

            async def _get_model_impl(self, model_id):
                raise RuntimeError("should not be called")

            async def _stream_chat_impl(
                self,
                model_id,
                messages,
                tools=None,
                params=None,
            ):
                raise RuntimeError("should not be called")

            async def _static_chat_impl(
                self,
                model_id,
                messages,
                tools=None,
                params=None,
            ):
                raise RuntimeError("should not be called")

        registry = ProviderRegistry([_RuntimeFailingProv])
        assert len(registry.available) == 0
        assert registry.failures["_RuntimeFailingProv"] == "boom"

    def test_all_fail(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        assert len(registry.available) == 0
        assert len(registry.failures) == 1

    def test_empty(self) -> None:
        registry = ProviderRegistry([])
        assert len(registry.available) == 0
        assert len(registry.failures) == 0


class TestAvailable:
    """Tests for ProviderRegistry.available."""

    def test_returns_available_providers(self) -> None:
        registry = ProviderRegistry([_MockProv, _MockProvB])
        result = registry.available
        assert len(result) == 2
        ids = {p.id for p in result}
        assert ids == {"mock", "prov_b"}

    def test_excludes_failed_providers(self) -> None:
        registry = ProviderRegistry([_MockProv, _FailingProv])
        result = registry.available
        assert len(result) == 1
        assert result[0].id == "mock"


class TestFailures:
    """Tests for ProviderRegistry.failures."""

    def test_returns_failure_diagnostics(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        result = registry.failures
        assert "_FailingProv" in result
        assert "Missing API key" in result["_FailingProv"]

    def test_returns_copy(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        result = registry.failures
        result["extra"] = "should not affect"
        assert "extra" not in registry.failures

    def test_empty_when_all_succeed(self) -> None:
        registry = ProviderRegistry([_MockProv])
        assert registry.failures == {}


class TestIsAvailable:
    """Tests for ProviderRegistry.is_available()."""

    def test_returns_true_for_available(self) -> None:
        registry = ProviderRegistry([_MockProv])
        assert registry.is_available("mock")

    def test_returns_false_for_unknown(self) -> None:
        registry = ProviderRegistry([])
        assert not registry.is_available("unknown")

    def test_returns_false_for_failed_provider_id(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        assert not registry.is_available("unknown")


class TestGet:
    """Tests for ProviderRegistry.get()."""

    def test_returns_provider(self) -> None:
        registry = ProviderRegistry([_MockProv])
        provider = registry.get("mock")
        assert provider.id == "mock"

    def test_raises_for_unknown(self) -> None:
        registry = ProviderRegistry([])
        with pytest.raises(ProviderNotAvailableError, match="not found"):
            registry.get("unknown")

    def test_raises_for_unregistered_id(self) -> None:
        registry = ProviderRegistry([_MockProv])
        with pytest.raises(ProviderNotAvailableError, match="not found"):
            registry.get("nonexistent")

    def test_raises_when_provider_failed_to_init(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        with pytest.raises(ProviderNotAvailableError):
            registry.get("unknown")
