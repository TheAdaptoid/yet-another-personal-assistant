"""Tests for OpenRouter fetch protocol."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.config import Config
from yapa.models import ModelData, ModelType
from yapa.providers.exceptions import ModelsFetchError


class TestOpenRouterFetchProtocol:
    """Tests for OpenRouterFetchProtocol."""

    @pytest.fixture
    def config(self):
        return Config(openrouter_api_key="sk-or-v1-test")

    @pytest.fixture
    def protocol(self, config):
        from yapa.providers.protocols.openrouter import OpenRouterFetchProtocol

        return OpenRouterFetchProtocol(config=config, provider_id="openrouter")

    def _make_model(self, id: str, output_modalities: list[str]):
        return SimpleNamespace(
            id=id,
            architecture=SimpleNamespace(output_modalities=output_modalities),
        )

    async def test_list_models_returns_formatted_data(self, protocol):
        models = [
            self._make_model("gpt-4", ["text"]),
            self._make_model("claude-opus", ["text"]),
        ]
        response = SimpleNamespace(data=models)

        mock_client = MagicMock()
        mock_client.models.list.return_value = response

        with patch(
            "yapa.providers.protocols.openrouter.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            result = await protocol.list_models()

        assert result == [
            ModelData(id="gpt-4", provider_id="openrouter", type=ModelType.LLM),
            ModelData(
                id="claude-opus", provider_id="openrouter", type=ModelType.LLM
            ),
        ]

    async def test_list_models_filters_llm(self, protocol):
        models = [
            self._make_model("gpt-4", ["text"]),
            self._make_model("dall-e-3", ["image"]),
        ]
        response = SimpleNamespace(data=models)

        mock_client = MagicMock()
        mock_client.models.list.return_value = response

        with patch(
            "yapa.providers.protocols.openrouter.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ) as mock_or:
            await protocol.list_models(model_type=ModelType.LLM)

        mock_or.assert_called_once()
        _kwargs = mock_or.call_args.kwargs
        assert _kwargs["url_params"] == {"output_modalities": "text"}

    async def test_list_models_no_filter_no_url_params(self, protocol):
        models = [self._make_model("gpt-4", ["text"])]
        response = SimpleNamespace(data=models)

        mock_client = MagicMock()
        mock_client.models.list.return_value = response

        with patch(
            "yapa.providers.protocols.openrouter.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ) as mock_or:
            await protocol.list_models()

        mock_or.assert_called_once()
        _kwargs = mock_or.call_args.kwargs
        assert _kwargs["url_params"] is None

    async def test_get_model_found(self, protocol):
        models = [self._make_model("gpt-4", ["text"])]
        response = SimpleNamespace(data=models)

        mock_client = MagicMock()
        mock_client.models.list.return_value = response

        with patch(
            "yapa.providers.protocols.openrouter.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            result = await protocol.get_model(model_id="gpt-4")

        assert result == ModelData(
            id="gpt-4", provider_id="openrouter", type=ModelType.LLM
        )

    async def test_get_model_not_found_raises(self, protocol):
        models = [self._make_model("claude-opus", ["text"])]
        response = SimpleNamespace(data=models)

        mock_client = MagicMock()
        mock_client.models.list.return_value = response

        with patch(
            "yapa.providers.protocols.openrouter.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            with pytest.raises(ModelsFetchError, match="gpt-4"):
                await protocol.get_model(model_id="gpt-4")

    def test_format_model_llm(self, protocol):
        model_info = self._make_model("gpt-4", ["text"])
        result = protocol._format_model(model_info)

        assert result == ModelData(
            id="gpt-4", provider_id="openrouter", type=ModelType.LLM
        )

    def test_format_model_other(self, protocol):
        model_info = self._make_model("dall-e-3", ["image"])
        result = protocol._format_model(model_info)

        assert result == ModelData(
            id="dall-e-3", provider_id="openrouter", type=ModelType.OTHER
        )
