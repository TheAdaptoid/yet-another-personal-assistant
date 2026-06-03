from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from yapa.models import ModelData
from yapa.providers.exceptions import ModelsFetchError
from yapa.providers.manager import ProviderManager


class TestProvidersProperty:
    def test_returns_lmstudio_and_openrouter(self) -> None:
        with (
            patch("yapa.providers.manager.LMStudioIP") as mock_lmstudio,
            patch("yapa.providers.manager.OpenRouterIP") as mock_openrouter,
        ):
            mock_lmstudio.return_value = MagicMock(id="lmstudio")
            mock_openrouter.return_value = MagicMock(id="openrouter")
            manager = ProviderManager()
            providers = manager.providers
            assert len(providers) == 2
            assert providers[0].id == "lmstudio"
            assert providers[1].id == "openrouter"


class TestGetProvider:
    def test_returns_matching_provider(self) -> None:
        mock_a = MagicMock(id="prov_a")
        mock_b = MagicMock(id="prov_b")
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a, mock_b]
            result = manager.get_provider("prov_a")
            assert result is mock_a

    def test_raises_value_error_for_unknown_id(self) -> None:
        mock_a = MagicMock(id="prov_a")
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a]
            with pytest.raises(ValueError, match="Unknown provider ID: unknown"):
                manager.get_provider("unknown")

    def test_matches_exact_id_only(self) -> None:
        mock_a = MagicMock(id="a")
        mock_aa = MagicMock(id="aa")
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a, mock_aa]
            assert manager.get_provider("a") is mock_a


class TestGetProviderByModel:
    async def test_returns_provider_with_matching_model(self) -> None:
        mock_a = MagicMock(id="prov_a")
        mock_a.get_models = AsyncMock(
            return_value=[ModelData(id="gpt-4", provider_id="prov_a")]
        )
        mock_b = MagicMock(id="prov_b")
        mock_b.get_models = AsyncMock(
            return_value=[ModelData(id="claude-3", provider_id="prov_b")]
        )
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a, mock_b]
            result = await manager.get_provider_by_model("gpt-4")
            assert result is mock_a

    async def test_returns_first_matching_provider(self) -> None:
        mock_a = MagicMock(id="prov_a")
        mock_a.get_models = AsyncMock(
            return_value=[ModelData(id="gpt-4", provider_id="prov_a")]
        )
        mock_b = MagicMock(id="prov_b")
        mock_b.get_models = AsyncMock(
            return_value=[ModelData(id="gpt-4", provider_id="prov_b")]
        )
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a, mock_b]
            result = await manager.get_provider_by_model("gpt-4")
            assert result is mock_a

    async def test_raises_value_error_when_no_match(self) -> None:
        mock_a = MagicMock(id="prov_a")
        mock_a.get_models = AsyncMock(
            return_value=[ModelData(id="gpt-4", provider_id="prov_a")]
        )
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a]
            with pytest.raises(
                ValueError, match="No provider found for model ID: unknown-model"
            ):
                await manager.get_provider_by_model("unknown-model")

    async def test_queries_all_providers_until_match(self) -> None:
        mock_a = MagicMock(id="prov_a")
        mock_a.get_models = AsyncMock(
            return_value=[ModelData(id="model-1", provider_id="prov_a")]
        )
        mock_b = MagicMock(id="prov_b")
        mock_b.get_models = AsyncMock(
            return_value=[ModelData(id="model-2", provider_id="prov_b")]
        )
        mock_c = MagicMock(id="prov_c")
        mock_c.get_models = AsyncMock(
            return_value=[ModelData(id="model-3", provider_id="prov_c")]
        )
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a, mock_b, mock_c]
            result = await manager.get_provider_by_model("model-3")
            assert result is mock_c
            mock_a.get_models.assert_awaited_once()
            mock_b.get_models.assert_awaited_once()
            mock_c.get_models.assert_awaited_once()

    async def test_skips_provider_on_models_fetch_error(self) -> None:
        mock_a = MagicMock(id="prov_a")
        mock_a.get_models = AsyncMock(side_effect=ModelsFetchError("boom"))
        mock_b = MagicMock(id="prov_b")
        mock_b.get_models = AsyncMock(
            return_value=[ModelData(id="gpt-4", provider_id="prov_b")]
        )
        manager = ProviderManager()
        with patch.object(
            ProviderManager, "providers", new_callable=PropertyMock
        ) as prop:
            prop.return_value = [mock_a, mock_b]
            result = await manager.get_provider_by_model("gpt-4")
            assert result is mock_b
            mock_a.get_models.assert_awaited_once()
