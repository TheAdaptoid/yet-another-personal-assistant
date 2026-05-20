"""Tests for InferenceService."""

import logging
from unittest.mock import AsyncMock, create_autospec

import pytest

from yapa.core.inference.exceptions import (
    InferenceError,
    ModelNotFoundError,
    ModelsFetchError,
)
from yapa.core.inference.provider import InferenceProvider
from yapa.core.services.inference_service import InferenceService
from yapa.shared.models import AssistantMessage, InferenceParams, ModelData


@pytest.fixture
def dummy_logger():
    """Return a logger with a NullHandler for test isolation."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def mock_provider_a():
    """Return an autospec'd provider with identifier 'provider-a'."""
    provider = create_autospec(InferenceProvider, instance=True)
    provider.id = "provider-a"
    provider.name = "Provider A"
    provider.get_models = AsyncMock()
    provider.invoke_model = AsyncMock()
    return provider


@pytest.fixture
def mock_provider_b():
    """Return an autospec'd provider with identifier 'provider-b'."""
    provider = create_autospec(InferenceProvider, instance=True)
    provider.id = "provider-b"
    provider.name = "Provider B"
    provider.get_models = AsyncMock()
    provider.invoke_model = AsyncMock()
    return provider


@pytest.fixture
def service(dummy_logger, mock_provider_a, mock_provider_b):
    """Return an InferenceService wired with two mock providers."""
    return InferenceService(dummy_logger, [mock_provider_a, mock_provider_b])


class TestGetModels:
    """InferenceService.get_models."""

    @pytest.mark.asyncio
    async def test_empty_with_no_providers(self, dummy_logger):
        """Returns an empty list when there are no providers."""
        svc = InferenceService(dummy_logger, [])
        models = await svc.get_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_returns_models_from_all_providers(
        self, service, mock_provider_a, mock_provider_b
    ):
        """Returns combined models from all providers."""
        model_a = ModelData(
            id="a", name="A", provider_id="provider-a", provider_name="A"
        )
        model_b = ModelData(
            id="b", name="B", provider_id="provider-b", provider_name="B"
        )
        mock_provider_a.get_models.return_value = [model_a]
        mock_provider_b.get_models.return_value = [model_b]

        models = await service.get_models()

        assert models == [model_a, model_b]

    @pytest.mark.asyncio
    async def test_skips_failed_provider(
        self, service, mock_provider_a, mock_provider_b
    ):
        """Skips a provider that raises ModelsFetchError and continues."""
        mock_provider_a.get_models.side_effect = ModelsFetchError("Provider A")
        model_b = ModelData(
            id="b", name="B", provider_id="provider-b", provider_name="B"
        )
        mock_provider_b.get_models.return_value = [model_b]

        models = await service.get_models()

        assert models == [model_b]

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_providers_fail(
        self, service, mock_provider_a, mock_provider_b
    ):
        """Returns an empty list when every provider raises ModelsFetchError."""
        mock_provider_a.get_models.side_effect = ModelsFetchError("Provider A")
        mock_provider_b.get_models.side_effect = ModelsFetchError("Provider B")

        models = await service.get_models()

        assert models == []


class TestFindModelProvider:
    """InferenceService._find_model_provider."""

    def test_returns_matching_provider(self, service, mock_provider_a):
        """Returns the provider when provider_id matches a registered provider."""
        model = ModelData(
            id="m", name="M", provider_id="provider-a", provider_name="A"
        )
        result = service._find_model_provider(model)
        assert result is mock_provider_a

    def test_raises_on_unknown_provider(self, service):
        """Raises ModelNotFoundError when no provider matches."""
        model = ModelData(
            id="m", name="M", provider_id="unknown", provider_name="Unknown"
        )
        with pytest.raises(ModelNotFoundError) as exc_info:
            service._find_model_provider(model)
        assert exc_info.value.model_id == "m"


class TestInvokeModel:
    """InferenceService.invoke_model."""

    @pytest.mark.asyncio
    async def test_delegates_to_correct_provider(
        self, service, mock_provider_b
    ):
        """Invokes the model on the matching provider."""
        model = ModelData(
            id="m", name="M", provider_id="provider-b", provider_name="B"
        )
        expected = AssistantMessage(content="hi")
        mock_provider_b.invoke_model.return_value = expected

        result = await service.invoke_model(model, [])

        assert result is expected
        mock_provider_b.invoke_model.assert_awaited_once_with(model, [], None)

    @pytest.mark.asyncio
    async def test_forwards_inference_params(
        self, service, mock_provider_a
    ):
        """Forwards InferenceParams when provided."""
        model = ModelData(
            id="m", name="M", provider_id="provider-a", provider_name="A"
        )
        params = InferenceParams(temperature=0.5)

        await service.invoke_model(model, [], params=params)

        mock_provider_a.invoke_model.assert_awaited_once_with(
            model, [], params
        )

    @pytest.mark.asyncio
    async def test_propagates_inference_error(self, service, mock_provider_a):
        """Propagates InferenceError from the underlying provider."""
        model = ModelData(
            id="m", name="M", provider_id="provider-a", provider_name="A"
        )
        mock_provider_a.invoke_model.side_effect = InferenceError("fail")

        with pytest.raises(InferenceError):
            await service.invoke_model(model, [])


class TestCreateDefault:
    """InferenceService.create_default — factory for default setup."""

    def test_returns_inference_service(self, dummy_logger, monkeypatch):
        """create_default returns an InferenceService instance."""
        import yapa.shared.config as config_module

        config_module._config = None
        monkeypatch.setenv("LMSTUDIO_API_KEY", "sk-test")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        svc = InferenceService.create_default(dummy_logger)
        assert isinstance(svc, InferenceService)
