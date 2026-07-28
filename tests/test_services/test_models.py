"""Tests for ModelService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import ModelData, ModelType
from yapa.providers.base import InferenceProvider
from yapa.providers.exceptions import ModelsFetchError
from yapa.providers.registry import ProviderNotAvailableError
from yapa.services.models import ModelService


@pytest.fixture(autouse=True)
def _mock_logger():
    with patch("yapa.services.models.get_logger") as mock:
        yield mock


class TestInit:
    def test_creates_default_registry(self):
        with patch("yapa.services.models.ProviderRegistry") as mock_reg_cls:
            ModelService()
            mock_reg_cls.assert_called_once()

    def test_accepts_custom_registry(self):
        registry = MagicMock()
        svc = ModelService(registry=registry)
        assert svc._registry is registry


class TestGetProvider:
    def test_delegates_to_registry(self):
        registry = MagicMock()
        svc = ModelService(registry=registry)
        svc.get_provider("openai")
        registry.get.assert_called_once_with("openai")

    def test_raises_models_fetch_error_on_unknown(self):
        registry = MagicMock()
        registry.get.side_effect = ProviderNotAvailableError("not found")
        svc = ModelService(registry=registry)
        with pytest.raises(ModelsFetchError, match="not found"):
            svc.get_provider("unknown")


class TestGetProviderByModel:
    def test_delegates_to_registry(self):
        registry = MagicMock()
        svc = ModelService(registry=registry)
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        svc.get_provider_by_model(model)
        registry.get.assert_called_once_with("openai")


class TestListModels:
    @pytest.fixture
    def svc(self):
        registry = MagicMock()
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
        return ModelService(registry=registry)

    async def test_returns_flat_list(self, svc):
        result = await svc.list_models()
        assert len(result) == 2
        assert all(isinstance(m, ModelData) for m in result)

    async def test_filters_by_provider(self, svc):
        result = await svc.list_models(provider_id="prov_a")
        assert len(result) == 1
        assert result[0].id == "gpt-4"
        assert result[0].provider_id == "prov_a"

    async def test_filters_by_model_type(self, svc):
        result = await svc.list_models(model_type=ModelType.LLM)
        assert len(result) == 2

    async def test_continues_on_provider_error(self, _mock_logger):
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
        svc = ModelService(registry=registry)
        result = await svc.list_models()
        assert len(result) == 1
        assert result[0].id == "claude"

    async def test_propagates_provider_id_stamp(self, svc):
        result = await svc.list_models()
        for m in result:
            assert m.provider_id in ("prov_a", "prov_b")

    async def test_returns_empty_on_unknown_specific_provider(self, _mock_logger):
        registry = MagicMock()
        registry.get.side_effect = ProviderNotAvailableError("not found")
        svc = ModelService(registry=registry)
        result = await svc.list_models(provider_id="unknown")
        assert result == []


class TestGetModel:
    @pytest.fixture
    def svc(self):
        registry = MagicMock()
        provider = MagicMock(spec=InferenceProvider)
        provider.id = "prov_a"
        provider.get_model = AsyncMock(
            return_value=ModelData(
                id="gpt-4", provider_id="prov_a", type=ModelType.LLM
            )
        )
        registry.get.return_value = provider
        return ModelService(registry=registry)

    async def test_returns_model_data(self, svc):
        result = await svc.get_model("prov_a:gpt-4")
        assert result.id == "gpt-4"
        assert result.provider_id == "prov_a"

    async def test_raises_on_malformed_id(self, svc):
        with pytest.raises(ValueError, match="expected 'provider_id:model_id'"):
            await svc.get_model("no-colon")

    async def test_raises_value_error_on_unknown_provider(self):
        registry = MagicMock()
        registry.get.side_effect = ProviderNotAvailableError("not found")
        svc = ModelService(registry=registry)
        with pytest.raises(ValueError, match="unknown"):
            await svc.get_model("unknown:gpt-4")
