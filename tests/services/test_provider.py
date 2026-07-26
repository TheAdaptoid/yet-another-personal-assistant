"""Tests for ProviderService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import ModelData, ModelType
from yapa.providers import DEFAULT_PROVIDER_CLASSES, InferenceProvider, ModelsFetchError
from yapa.services import ProviderService


@pytest.fixture(autouse=True)
def _mock_provider_logger():
    with patch("yapa.services.provider.logger") as mock:
        yield mock


@pytest.fixture
def registry():
    return MagicMock()


class TestInit:
    """Tests for ProviderService.__init__()."""

    def test_accepts_registry(self, registry):
        svc = ProviderService(registry=registry)
        assert svc._registry is registry

    def test_creates_default_registry(self):
        with patch("yapa.services.provider.ProviderRegistry") as mock_reg_cls:
            ProviderService()
            mock_reg_cls.assert_called_once_with(DEFAULT_PROVIDER_CLASSES)


class TestGetProvider:
    """Tests for ProviderService.get_provider()."""

    def test_delegates_to_registry(self, registry):
        svc = ProviderService(registry=registry)
        svc.get_provider("openai")
        registry.get.assert_called_once_with("openai")


class TestGetProviderByModel:
    """Tests for ProviderService.get_provider_by_model()."""

    def test_delegates_to_registry(self, registry):
        svc = ProviderService(registry=registry)
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        svc.get_provider_by_model(model)
        registry.get.assert_called_once_with("openai")


class TestListModels:
    """Tests for ProviderService.list_models()."""

    @pytest.fixture
    def svc(self, registry):
        provider_a = MagicMock(spec=InferenceProvider)
        provider_a.id = "prov_a"
        provider_a.list_models = AsyncMock(
            return_value=[
                ModelData(id="gpt-4", provider_id="prov_a", type=ModelType.LLM),
            ]
        )
        provider_b = MagicMock(spec=InferenceProvider)
        provider_b.id = "prov_b"
        provider_b.list_models = AsyncMock(
            return_value=[
                ModelData(id="claude", provider_id="prov_b", type=ModelType.LLM),
            ]
        )
        registry.available = [provider_a, provider_b]
        registry.get.return_value = provider_a
        return ProviderService(registry=registry)

    async def test_returns_all_providers(self, svc):
        result = await svc.list_models()

        assert result == {
            "prov_a": [
                ModelData(id="gpt-4", provider_id="prov_a", type=ModelType.LLM),
            ],
            "prov_b": [
                ModelData(id="claude", provider_id="prov_b", type=ModelType.LLM),
            ],
        }

    async def test_returns_single_provider(self, svc):
        result = await svc.list_models(provider_id="prov_a")

        assert result == {
            "prov_a": [
                ModelData(id="gpt-4", provider_id="prov_a", type=ModelType.LLM),
            ],
        }

    async def test_continues_on_provider_error(self, _mock_provider_logger):
        registry = MagicMock()

        provider_a = MagicMock(spec=InferenceProvider)
        provider_a.id = "prov_a"
        provider_a.list_models = AsyncMock(
            side_effect=ModelsFetchError("API down")
        )

        provider_b = MagicMock(spec=InferenceProvider)
        provider_b.id = "prov_b"
        provider_b.list_models = AsyncMock(
            return_value=[
                ModelData(id="claude", provider_id="prov_b", type=ModelType.LLM),
            ]
        )

        registry.available = [provider_a, provider_b]
        svc = ProviderService(registry=registry)

        result = await svc.list_models()

        assert result == {
            "prov_a": [],
            "prov_b": [
                ModelData(id="claude", provider_id="prov_b", type=ModelType.LLM),
            ],
        }
        _mock_provider_logger.error.assert_called()

    async def test_single_provider_returns_empty_on_error(
        self, _mock_provider_logger, registry
    ):
        provider_a = MagicMock(spec=InferenceProvider)
        provider_a.id = "prov_a"
        provider_a.list_models = AsyncMock(
            side_effect=ModelsFetchError("API down")
        )
        registry.get.return_value = provider_a

        svc = ProviderService(registry=registry)
        result = await svc.list_models(provider_id="prov_a")

        assert result == {"prov_a": []}
        _mock_provider_logger.error.assert_called()


class TestGetModel:
    """Tests for ProviderService.get_model()."""

    @pytest.fixture
    def svc(self, registry):
        provider = MagicMock(spec=InferenceProvider)
        provider.id = "prov_a"
        provider.get_model = AsyncMock(
            return_value=ModelData(
                id="gpt-4", provider_id="prov_a", type=ModelType.LLM
            )
        )
        registry.get.return_value = provider
        return ProviderService(registry=registry)

    async def test_returns_model_data(self, svc):
        result = await svc.get_model("prov_a:gpt-4")

        assert result == ModelData(
            id="gpt-4", provider_id="prov_a", type=ModelType.LLM
        )

    async def test_passes_model_id_to_provider(self, svc, registry):
        await svc.get_model("prov_a:gpt-4")
        provider = registry.get.return_value
        provider.get_model.assert_called_once_with(model_id="gpt-4")

    async def test_wraps_models_fetch_error(self, _mock_provider_logger, registry):
        provider = MagicMock(spec=InferenceProvider)
        provider.id = "prov_a"
        provider.get_model = AsyncMock(
            side_effect=ModelsFetchError("API error")
        )
        registry.get.return_value = provider

        svc = ProviderService(registry=registry)
        with pytest.raises(ValueError, match="Failed to fetch model"):
            await svc.get_model("prov_a:gpt-4")

        _mock_provider_logger.error.assert_called_once()

    async def test_raises_on_malformed_full_id(self, svc):
        with pytest.raises(ValueError, match="expected 'provider_id:model_id'"):
            await svc.get_model("no-colon")
