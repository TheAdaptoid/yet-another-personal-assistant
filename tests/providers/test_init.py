"""Tests for concrete provider constructors."""

from unittest.mock import patch

import pytest

from yapa.providers.lmstudio import LMStudioIP
from yapa.providers.ollama import OllamaIP
from yapa.providers.openai import OpenAIIP
from yapa.providers.openrouter import OpenRouterProvider
from yapa.services.config import Config, ProviderConfig

_ASYNC_OPENAI = "yapa.providers.openai._noauth.AsyncOpenAI"


class TestOpenAIIPInit:
    """Tests for OpenAIIP constructor."""

    def test_raises_on_unset_key(self) -> None:
        config = Config(provider_configs={})
        with pytest.raises(ValueError, match="API key is not set"):
            OpenAIIP(config=config)

    def test_raises_on_none_key(self) -> None:
        config = Config(provider_configs={"openai": ProviderConfig(api_key=None)})
        with pytest.raises(ValueError, match="API key is not set"):
            OpenAIIP(config=config)

    def test_initializes_with_valid_key(self, mock_openai_client) -> None:
        config = Config(provider_configs={"openai": ProviderConfig(api_key="sk-test")})
        with patch(_ASYNC_OPENAI, return_value=mock_openai_client) as mock_client:
            provider = OpenAIIP(config=config)
        assert provider.id == "openai"
        assert provider.name == "OpenAI"
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["timeout"] == 120
        assert mock_client.call_args.kwargs["max_retries"] == 2

    def test_custom_timeout(self, mock_openai_client) -> None:
        config = Config(
            provider_configs={"openai": ProviderConfig(api_key="sk-test")},
            provider_timeout=300,
        )
        with patch(_ASYNC_OPENAI, return_value=mock_openai_client) as mock_client:
            provider = OpenAIIP(config=config)
        assert provider.id == "openai"
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["timeout"] == 300

    def test_custom_max_retries(self, mock_openai_client) -> None:
        config = Config(
            provider_configs={"openai": ProviderConfig(api_key="sk-test")},
            provider_max_retries=5,
        )
        with patch(_ASYNC_OPENAI, return_value=mock_openai_client) as mock_client:
            provider = OpenAIIP(config=config)
        assert provider.id == "openai"
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["max_retries"] == 5


class TestLMStudioIPInit:
    """Tests for LMStudioIP constructor."""

    def test_initializes_with_config(self, mock_openai_client) -> None:
        config = Config(
            provider_configs={"lmstudio": ProviderConfig(api_key="test-key")}
        )
        with patch(_ASYNC_OPENAI, return_value=mock_openai_client) as mock_client:
            provider = LMStudioIP(config=config)
        assert provider.id == "lmstudio"
        assert provider.name == "LM Studio"
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["timeout"] == 120
        assert mock_client.call_args.kwargs["max_retries"] == 2


class TestOllamaIPInit:
    """Tests for OllamaIP constructor."""

    _ASYNC_CLIENT = "yapa.providers.ollama.provider.AsyncClient"

    def test_initializes_with_config(self, mock_openai_client) -> None:
        config = Config(provider_configs={"ollama": ProviderConfig(api_key="test-key")})
        with patch(self._ASYNC_CLIENT, return_value=mock_openai_client) as mock_client:
            provider = OllamaIP(config=config)
        assert provider.id == "ollama"
        assert provider.name == "Ollama"
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["host"] == "http://127.0.0.1:11434"

    def test_initializes_with_custom_host(self, mock_openai_client) -> None:
        config = Config(
            provider_configs={
                "ollama": ProviderConfig(
                    api_key="test-key", base_url="http://localhost:8080"
                )
            }
        )
        with patch(self._ASYNC_CLIENT, return_value=mock_openai_client) as mock_client:
            provider = OllamaIP(config=config)
        assert provider.id == "ollama"
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["host"] == "http://localhost:8080"


class TestOpenRouterProviderInit:
    """Tests for OpenRouterProvider constructor."""

    def test_raises_on_unset_key(self) -> None:
        config = Config(provider_configs={})
        with pytest.raises(ValueError, match="API key is not set"):
            OpenRouterProvider(config=config)

    def test_raises_on_none_key(self) -> None:
        config = Config(provider_configs={"openrouter": ProviderConfig(api_key=None)})
        with pytest.raises(ValueError, match="API key is not set"):
            OpenRouterProvider(config=config)

    def test_initializes_with_valid_key(self, mock_openai_client) -> None:
        config = Config(
            provider_configs={"openrouter": ProviderConfig(api_key="sk-or-test")}
        )
        with patch(_ASYNC_OPENAI, return_value=mock_openai_client) as mock_client:
            provider = OpenRouterProvider(config=config)
        assert provider.id == "openrouter"
        assert provider.name == "OpenRouter"
        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["timeout"] == 120
        assert mock_client.call_args.kwargs["max_retries"] == 2
