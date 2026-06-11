"""Tests for OpenRouterIP concrete provider."""

from unittest.mock import MagicMock, patch

import pytest

from yapa.config import UNSET, Config
from yapa.providers.concretes.openrouter import OpenRouterIP


class TestOpenRouterIP:
    """Tests for OpenRouterIP."""

    def test_raises_on_default_config(self) -> None:
        config = Config()
        with pytest.raises(ValueError, match="API key is not set"):
            OpenRouterIP(config=config)

    def test_raises_on_unset_key(self) -> None:
        config = Config(openrouter_api_key=UNSET)
        with pytest.raises(ValueError, match="API key is not set"):
            OpenRouterIP(config=config)

    def test_initialization_with_valid_key(self) -> None:
        config = Config(
            openrouter_api_key="sk-or-v1-test",
            openrouter_base_url="https://custom.example.com/v1",
        )
        with (
            patch("yapa.providers.concretes.openrouter.AsyncOpenAI") as mock_openai,
            patch("yapa.providers.concretes.openrouter.OpenRouterFetchProtocol"),
        ):
            mock_openai.return_value = MagicMock()
            provider = OpenRouterIP(config=config)

        assert provider.id == "openrouter"
        assert provider.name == "OpenRouter"
        mock_openai.assert_called_once_with(
            api_key="sk-or-v1-test",
            base_url="https://custom.example.com/v1",
        )
