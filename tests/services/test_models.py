"""Tests for provider model listing and grouping."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from yapa.models import ModelData
from yapa.services import ProviderService


class TestGroupByVendor:
    """Tests for ProviderService.group_models_by_vendor()."""

    def test_vendor_prefixed(self):
        models = [ModelData(id="openai/gpt-4", provider_id="p")]
        groups = ProviderService.group_models_by_vendor(models)
        assert "openai" in groups
        assert groups["openai"] == models

    def test_nested_prefix(self):
        models = [ModelData(id="huggingface/TheBloke/Mistral-7B", provider_id="p")]
        groups = ProviderService.group_models_by_vendor(models)
        assert "huggingface" in groups

    def test_no_prefix(self):
        models = [ModelData(id="codellama-7b", provider_id="p")]
        groups = ProviderService.group_models_by_vendor(models)
        assert "other" in groups

    def test_multiple_groups(self):
        models = [
            ModelData(id="openai/gpt-4", provider_id="p"),
            ModelData(id="anthropic/claude", provider_id="p"),
            ModelData(id="codellama-7b", provider_id="p"),
        ]
        groups = ProviderService.group_models_by_vendor(models)
        assert sorted(groups) == ["anthropic", "openai", "other"]

    def test_sorted_keys(self):
        models = [
            ModelData(id="z/gpt-4", provider_id="p"),
            ModelData(id="a/claude", provider_id="p"),
        ]
        groups = ProviderService.group_models_by_vendor(models)
        assert list(groups.keys()) == ["a", "z"]


class TestGetModels:
    """Tests for ProviderService.get_models()."""

    @pytest.fixture
    def mock_provider_service(self):
        svc = MagicMock(spec=ProviderService)
        svc.get_models = AsyncMock()
        return svc

    async def test_returns_models_from_all_providers(self, mock_provider_service):
        mock_provider_service.get_models.return_value = {
            "prov_a": [ModelData(id="gpt-4", provider_id="prov_a")],
            "prov_b": [ModelData(id="claude", provider_id="prov_b")],
        }

        result = await mock_provider_service.get_models()

        assert result == {
            "prov_a": [ModelData(id="gpt-4", provider_id="prov_a")],
            "prov_b": [ModelData(id="claude", provider_id="prov_b")],
        }

    async def test_returns_models_for_provider(self, mock_provider_service):
        mock_provider_service.get_models.return_value = {
            "prov_a": [ModelData(id="gpt-4", provider_id="prov_a")],
        }

        result = await mock_provider_service.get_models("prov_a")

        assert result.get("prov_a") == [ModelData(id="gpt-4", provider_id="prov_a")]

    async def test_returns_empty_for_unknown_provider(self, mock_provider_service):
        mock_provider_service.get_models.return_value = {}

        result = await mock_provider_service.get_models("nonexistent")

        assert result == {}
