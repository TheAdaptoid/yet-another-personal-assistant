"""Tests for concrete provider constructors."""

from unittest.mock import patch

import pytest

from yapa.config import UNSET, Config
from yapa.providers.lmstudio import LMStudioIP
from yapa.providers.ollama import OllamaIP
from yapa.providers.openai import OpenAIIP
from yapa.providers.openrouter import OpenRouterProvider


class TestOpenAIIPInit:
    """Tests for OpenAIIP constructor."""

    def test_raises_on_unset_key(self) -> None:
        config = Config(openai_api_key=UNSET)
        with pytest.raises(ValueError, match="API key is not set"):
            OpenAIIP(config=config)

    def test_raises_on_none_key(self) -> None:
        config = Config(openai_api_key=UNSET)
        with pytest.raises(ValueError, match="API key is not set"):
            OpenAIIP(config=config)

    def test_initializes_with_valid_key(self, mock_openai_client) -> None:
        config = Config(openai_api_key="sk-test")
        with patch(
            "yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OpenAIIP(config=config)
        assert provider.id == "openai"
        assert provider.name == "OpenAI"


class TestLMStudioIPInit:
    """Tests for LMStudioIP constructor."""

    def test_initializes_with_config(self, mock_openai_client) -> None:
        config = Config(lmstudio_api_key="test-key")
        with patch(
            "yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = LMStudioIP(config=config)
        assert provider.id == "lmstudio"
        assert provider.name == "LM Studio"


class TestOllamaIPInit:
    """Tests for OllamaIP constructor."""

    def test_initializes_with_config(self, mock_openai_client) -> None:
        config = Config(ollama_api_key="test-key")
        with patch(
            "yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OllamaIP(config=config)
        assert provider.id == "ollama"
        assert provider.name == "Ollama"


class TestOpenRouterProviderInit:
    """Tests for OpenRouterProvider constructor."""

    def test_raises_on_unset_key(self) -> None:
        config = Config(openrouter_api_key=UNSET)
        with pytest.raises(ValueError, match="API key is not set"):
            OpenRouterProvider(config=config)

    def test_raises_on_none_key(self) -> None:
        config = Config(openrouter_api_key=UNSET)
        with pytest.raises(ValueError, match="API key is not set"):
            OpenRouterProvider(config=config)

    def test_initializes_with_valid_key(self, mock_openai_client) -> None:
        config = Config(openrouter_api_key="sk-or-test")
        with patch(
            "yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client
        ):
            provider = OpenRouterProvider(config=config)
        assert provider.id == "openrouter"
        assert provider.name == "OpenRouter"
