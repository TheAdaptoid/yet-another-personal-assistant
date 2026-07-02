"""Tests for OllamaIP concrete provider."""

from unittest.mock import MagicMock, patch

from yapa.config import Config
from yapa.providers.ollama.provider import OllamaIP
from yapa.providers.openai.protocols import (
    OpenAILLMInferenceProtocol,
    OpenAIModelFetchProtocol,
)


class TestOllamaIP:
    """Tests for OllamaIP."""

    def test_initialization_with_config(self) -> None:
        config = Config(
            ollama_api_key="test-key",
            ollama_base_url="http://custom:11434/v1",
        )
        with patch("yapa.providers.ollama.provider.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider = OllamaIP(config=config)

        assert provider.id == "ollama"
        assert provider.name == "Ollama"
        assert isinstance(provider._model_fetcher, OpenAIModelFetchProtocol)
        assert isinstance(provider._llm_invoker, OpenAILLMInferenceProtocol)
        mock_openai.assert_called_once_with(
            api_key="test-key", base_url="http://custom:11434/v1"
        )

    def test_default_config_calls_get_config(self) -> None:
        with (
            patch("yapa.providers.ollama.provider.get_config") as mock_get_config,
            patch("yapa.providers.ollama.provider.AsyncOpenAI"),
        ):
            mock_get_config.return_value = Config()
            provider = OllamaIP()

        assert provider.id == "ollama"
        assert provider.name == "Ollama"
