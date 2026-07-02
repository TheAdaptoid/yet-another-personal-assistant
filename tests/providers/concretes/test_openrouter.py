"""Tests for OpenRouterIP concrete provider."""

from unittest.mock import MagicMock, patch

import pytest

from yapa.config import UNSET, Config
from yapa.providers.openrouter.provider import OpenRouterIP


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
        config = Config(openrouter_api_key="sk-or-v1-test")
        with (
            patch(
                "yapa.providers.openrouter.provider.OpenRouterLLMInferenceProtocol"
            ) as mock_proto,
            patch("yapa.providers.openrouter.provider.OpenRouterFetchProtocol"),
        ):
            mock_proto.return_value = MagicMock()
            provider = OpenRouterIP(config=config)

        assert provider.id == "openrouter"
        assert provider.name == "OpenRouter"
