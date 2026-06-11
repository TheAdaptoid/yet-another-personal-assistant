"""Tests for ProviderService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import ModelData, ModelType
from yapa.providers.base import InferenceProvider
from yapa.providers.exceptions import ModelsFetchError
from yapa.services import ProviderService


@pytest.fixture(autouse=True)
def _mock_provider_logger():
    with patch("yapa.services.provider.logger") as mock:
        yield mock


class _MockProv(InferenceProvider):
    """A mock provider that can be instantiated without real connections."""

    def __init__(self, identifier="mock", name="Mock Provider"):
        super().__init__(
            identifier=identifier,
            name=name,
            model_fetcher=MagicMock(),
            model_invoker=MagicMock(),
        )


class _FailingProv(InferenceProvider):
    """A mock provider that raises ValueError on construction."""

    def __init__(self):
        raise ValueError("Missing API key")


class TestInit:
    """Tests for ProviderService.__init__()."""

    def test_constructs_with_custom_providers(self):
        class ProvA(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_a", name="Provider A")

        class ProvB(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_b", name="Provider B")

        svc = ProviderService(providers=[ProvA, ProvB])

        assert len(svc._provider_cache) == 2
        assert svc._provider_cache["prov_a"].name == "Provider A"
        assert svc._provider_cache["prov_b"].name == "Provider B"

    def test_skips_failing_providers(self, _mock_provider_logger):
        class ProvA(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_a", name="Provider A")

        svc = ProviderService(providers=[_FailingProv, ProvA])

        assert "prov_a" in svc._provider_cache
        _mock_provider_logger.warning.assert_called_once()

    def test_empty_providers_list(self):
        svc = ProviderService(providers=[])
        assert svc._provider_cache == {}


class TestRefreshProviders:
    """Tests for ProviderService.refresh_providers()."""

    def test_replaces_existing_cache(self):
        class ProvA(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_a", name="Provider A")

        svc = ProviderService(providers=[ProvA])
        assert list(svc._provider_cache.keys()) == ["prov_a"]

        class ProvB(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_b", name="Provider B")

        svc.refresh_providers(providers=[ProvB])
        assert list(svc._provider_cache.keys()) == ["prov_b"]


class TestListProviders:
    """Tests for ProviderService.list_providers()."""

    def test_returns_all_cached_providers(self):
        class ProvA(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_a", name="Provider A")

        svc = ProviderService(providers=[ProvA])
        result = svc.list_providers()
        assert len(result) == 1
        assert result[0].id == "prov_a"

    def test_empty_when_no_providers(self):
        svc = ProviderService(providers=[])
        assert svc.list_providers() == []


class TestGetProvider:
    """Tests for ProviderService.get_provider()."""

    def test_returns_provider(self):
        class ProvA(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_a", name="Provider A")

        svc = ProviderService(providers=[ProvA])
        result = svc.get_provider("prov_a")
        assert result.id == "prov_a"

    def test_raises_on_missing(self):
        svc = ProviderService(providers=[])
        with pytest.raises(ValueError, match="not found"):
            svc.get_provider("nonexistent")


class TestGetProviderByModel:
    """Tests for ProviderService.get_provider_by_model()."""

    def test_returns_provider_for_model(self):
        class ProvA(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_a", name="Provider A")

        svc = ProviderService(providers=[ProvA])
        model = ModelData(id="gpt-4", provider_id="prov_a", type=ModelType.LLM)
        result = svc.get_provider_by_model(model)
        assert result.id == "prov_a"

    def test_raises_on_unknown_provider(self):
        svc = ProviderService(providers=[])
        model = ModelData(id="gpt-4", provider_id="unknown", type=ModelType.LLM)
        with pytest.raises(ValueError, match="supports model"):
            svc.get_provider_by_model(model)


class TestGetProviderByModelFullId:
    """Tests for ProviderService.get_provider_by_model_full_id()."""

    def test_returns_provider(self):
        class ProvA(_MockProv):
            def __init__(self):
                super().__init__(identifier="prov_a", name="Provider A")

        svc = ProviderService(providers=[ProvA])
        result = svc.get_provider_by_model_full_id("prov_a:gpt-4")
        assert result.id == "prov_a"

    def test_raises_on_unknown_prefix(self):
        svc = ProviderService(providers=[])
        with pytest.raises(ValueError, match="supports model"):
            svc.get_provider_by_model_full_id("unknown:gpt-4")


class TestListModels:
    """Tests for ProviderService.list_models()."""

    @pytest.fixture
    def svc(self):
        svc = ProviderService(providers=[])
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
        svc._provider_cache = {"prov_a": provider_a, "prov_b": provider_b}
        return svc

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

    async def test_raises_for_unknown_provider(self, svc):
        with pytest.raises(ValueError, match="not found"):
            await svc.list_models(provider_id="nonexistent")

    async def test_continues_on_provider_error(self, _mock_provider_logger):
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

        svc = ProviderService(providers=[])
        svc._provider_cache = {"prov_a": provider_a, "prov_b": provider_b}

        result = await svc.list_models()

        assert result == {
            "prov_a": [],
            "prov_b": [
                ModelData(id="claude", provider_id="prov_b", type=ModelType.LLM),
            ],
        }
        _mock_provider_logger.error.assert_called()

class TestGetModel:
    """Tests for ProviderService.get_model()."""

    @pytest.fixture
    def svc(self):
        svc = ProviderService(providers=[])
        provider = MagicMock(spec=InferenceProvider)
        provider.id = "prov_a"
        provider.get_model = AsyncMock(
            return_value=ModelData(
                id="gpt-4", provider_id="prov_a", type=ModelType.LLM
            )
        )
        svc._provider_cache = {"prov_a": provider}
        return svc

    async def test_returns_model_data(self, svc):
        result = await svc.get_model("prov_a:gpt-4")

        assert result == ModelData(
            id="gpt-4", provider_id="prov_a", type=ModelType.LLM
        )

    async def test_passes_model_id_to_provider(self, svc):
        await svc.get_model("prov_a:gpt-4")
        svc._provider_cache["prov_a"].get_model.assert_called_once_with(
            model_id="gpt-4"
        )

    async def test_raises_on_unknown_provider(self, svc):
        with pytest.raises(ValueError, match="not found"):
            await svc.get_model("unknown:gpt-4")

    async def test_wraps_models_fetch_error(self, _mock_provider_logger):
        svc = ProviderService(providers=[])
        provider = MagicMock(spec=InferenceProvider)
        provider.id = "prov_a"
        provider.get_model = AsyncMock(
            side_effect=ModelsFetchError("API error")
        )
        svc._provider_cache = {"prov_a": provider}

        with pytest.raises(ValueError, match="Failed to fetch model"):
            await svc.get_model("prov_a:gpt-4")

        _mock_provider_logger.error.assert_called_once()

    async def test_raises_on_malformed_full_id(self, svc):
        with pytest.raises(ValueError):
            await svc.get_model("no-colon")

    async def test_empty_cache(self):
        svc = ProviderService(providers=[])
        with pytest.raises(ValueError, match="not found"):
            await svc.get_model("prov_a:gpt-4")


    async def test_single_provider_returns_empty_on_error(self, _mock_provider_logger):
        provider_a = MagicMock(spec=InferenceProvider)
        provider_a.id = "prov_a"
        provider_a.list_models = AsyncMock(
            side_effect=ModelsFetchError("API down")
        )

        svc = ProviderService(providers=[])
        svc._provider_cache = {"prov_a": provider_a}

        result = await svc.list_models(provider_id="prov_a")

        assert result == {"prov_a": []}
        _mock_provider_logger.error.assert_called()
