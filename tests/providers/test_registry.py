"""Tests for ProviderRegistry."""

from unittest.mock import patch

import pytest

from yapa.models import (
    AssistantMessage,
    EmbeddingResult,
    ModelData,
    ModelDataUnion,
    ModelType,
)
from yapa.providers.base import InferenceProvider
from yapa.providers.openai.provider import OpenAIIP
from yapa.providers.registry import ProviderNotAvailableError, ProviderRegistry


class _MockProv(InferenceProvider):
    """Provider that initializes successfully."""

    def __init__(self, config=None):
        super().__init__("mock", "Mock Provider")

    async def _list_models_impl(self, model_type=None) -> list[ModelDataUnion]:
        return []

    async def _get_model_impl(self, model_id) -> ModelDataUnion:
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        if False:
            yield

    async def _static_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        return AssistantMessage(content="test", role="assistant")

    async def _embed_impl(self, model_id, input):
        return EmbeddingResult(vectors=[[1.0]], model_id=model_id)


class _MockProvB(InferenceProvider):
    """Another provider that initializes successfully."""

    def __init__(self, config=None):
        super().__init__("prov_b", "Provider B")

    async def _list_models_impl(self, model_type=None) -> list[ModelDataUnion]:
        return []

    async def _get_model_impl(self, model_id) -> ModelDataUnion:
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        if False:
            yield

    async def _static_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        return AssistantMessage(content="test", role="assistant")

    async def _embed_impl(self, model_id, input):
        return EmbeddingResult(vectors=[[1.0]], model_id=model_id)


class _FailingProv(InferenceProvider):
    """Provider that fails to initialize before an id exists."""

    def __init__(self, config=None):
        raise ValueError("Missing API key")

    async def _list_models_impl(self, model_type=None):
        raise RuntimeError("should not be called")

    async def _get_model_impl(self, model_id):
        raise RuntimeError("should not be called")

    async def _stream_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        raise RuntimeError("should not be called")
        yield  # pragma: no cover

    async def _static_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        raise RuntimeError("should not be called")

    async def _embed_impl(self, model_id, input):
        raise RuntimeError("should not be called")


class _LateFailProv(InferenceProvider):
    """Fails after super().__init__, so an id exists."""

    PROVIDER_ID = "latefail"

    def __init__(self, config=None):
        super().__init__("latefail", "Late Fail")
        raise ValueError("boom after id")

    async def _list_models_impl(self, model_type=None):
        raise RuntimeError("should not be called")

    async def _get_model_impl(self, model_id):
        raise RuntimeError("should not be called")

    async def _stream_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        if False:
            yield

    async def _static_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        raise RuntimeError("should not be called")

    async def _embed_impl(self, model_id, input):
        raise RuntimeError("should not be called")


class _RuntimeFailingProv(InferenceProvider):
    """Provider that fails to initialize with a non-ValueError."""

    def __init__(self, config=None):
        raise RuntimeError("boom")

    async def _list_models_impl(self, model_type=None):
        raise RuntimeError("should not be called")

    async def _get_model_impl(self, model_id):
        raise RuntimeError("should not be called")

    async def _stream_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        raise RuntimeError("should not be called")
        yield  # pragma: no cover

    async def _static_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        raise RuntimeError("should not be called")

    async def _embed_impl(self, model_id, input):
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
        with pytest.raises(ProviderNotAvailableError, match="unknown"):
            registry.get("unknown")

    def test_raises_for_unregistered_id(self) -> None:
        registry = ProviderRegistry([_MockProv])
        with pytest.raises(ProviderNotAvailableError, match="unknown"):
            registry.get("nonexistent")

    def test_raises_when_provider_failed_to_init(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        with pytest.raises(ProviderNotAvailableError):
            registry.get("unknown")


class TestFailureKeying:
    """Failures keyed by id when determinable, else class name."""

    def test_failure_before_id_keyed_by_class_name(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        assert registry.failures == {"_FailingProv": "Missing API key"}

    def test_failure_after_id_keyed_by_provider_id(self) -> None:
        registry = ProviderRegistry([_LateFailProv])
        assert "latefail" in registry.failures
        assert "boom after id" in registry.failures["latefail"]

    def test_real_provider_failure_keyed_by_provider_id(self) -> None:
        class _FailingOpenAI(OpenAIIP):
            def __init__(self, config=None):
                raise ValueError("openai boom")

        registry = ProviderRegistry([_FailingOpenAI])
        assert registry.failures["openai"] == "openai boom"
        with pytest.raises(ProviderNotAvailableError, match="openai boom"):
            registry.get("openai")


class TestFailureLogging:
    """Init failures are logged at error level."""

    def test_logs_failure_at_error(self) -> None:
        with patch("yapa.providers.registry.logger") as mock_logger:
            ProviderRegistry([_FailingProv])
            mock_logger.error.assert_called()
            msg = str(mock_logger.error.call_args)
            assert "Missing API key" in msg

    def test_no_error_log_when_success(self) -> None:
        with patch("yapa.providers.registry.logger") as mock_logger:
            ProviderRegistry([_MockProv])
            for call in mock_logger.error.call_args_list:
                raise AssertionError(f"unexpected error log: {call}")


class TestGetIncludeFailureReason:
    """get() includes the stored failure reason for failed providers."""

    def test_get_raises_with_failure_reason(self) -> None:
        registry = ProviderRegistry([_LateFailProv])
        with pytest.raises(ProviderNotAvailableError, match="boom after id"):
            registry.get("latefail")

    def test_get_raises_unknown_for_unregistered(self) -> None:
        registry = ProviderRegistry([_MockProv])
        with pytest.raises(ProviderNotAvailableError, match="unknown"):
            registry.get("nope")
