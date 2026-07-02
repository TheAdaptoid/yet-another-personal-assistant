"""Tests for OpenAIIP concrete provider."""

from unittest.mock import MagicMock, patch

import pytest

from yapa.config import UNSET, Config
from yapa.providers.openai.protocols import (
    OpenAILLMInferenceProtocol,
    OpenAIModelFetchProtocol,
)
from yapa.providers.openai.provider import OpenAIIP


class TestOpenAIIP:
    """Tests for OpenAIIP."""

    def test_raises_on_default_config(self) -> None:
        config = Config()
        with pytest.raises(ValueError, match="API key is not set"):
            OpenAIIP(config=config)

    def test_raises_on_unset_key(self) -> None:
        config = Config(openai_api_key=UNSET)
        with pytest.raises(ValueError, match="API key is not set"):
            OpenAIIP(config=config)

    def test_initialization_with_valid_key(self) -> None:
        config = Config(openai_api_key="sk-test")
        with patch("yapa.providers.openai.provider.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider = OpenAIIP(config=config)

        assert provider.id == "openai"
        assert provider.name == "OpenAI"
        assert isinstance(provider._model_fetcher, OpenAIModelFetchProtocol)
        assert isinstance(provider._llm_invoker, OpenAILLMInferenceProtocol)
        mock_openai.assert_called_once_with(
            api_key="sk-test",
            base_url=config.openai_base_url,
        )

    def test_default_config_calls_get_config(self) -> None:
        with (
            patch("yapa.providers.openai.provider.get_config") as mock_get_config,
            patch("yapa.providers.openai.provider.AsyncOpenAI"),
        ):
            mock_get_config.return_value = Config(openai_api_key="sk-test")
            provider = OpenAIIP()

        assert provider.id == "openai"
        assert provider.name == "OpenAI"
