"""Tests for LMStudioIP concrete provider."""

from unittest.mock import MagicMock, patch

from yapa.config import Config
from yapa.providers.concretes.lmstudio import LMStudioIP
from yapa.providers.protocols import OpenAIInferenceProtocol, OpenAIModelFetchProtocol


class TestLMStudioIP:
    """Tests for LMStudioIP."""

    def test_initialization_with_config(self) -> None:
        config = Config(
            lmstudio_api_key="test-key",
            lmstudio_base_url="http://custom:1234/v1",
        )
        with patch("yapa.providers.concretes.lmstudio.AsyncOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            provider = LMStudioIP(config=config)

        assert provider.id == "lmstudio"
        assert provider.name == "LM Studio"
        assert isinstance(provider._model_fetcher, OpenAIModelFetchProtocol)
        assert isinstance(provider._model_invoker, OpenAIInferenceProtocol)
        mock_openai.assert_called_once_with(
            api_key="test-key", base_url="http://custom:1234/v1"
        )

    def test_default_config_calls_get_config(self) -> None:
        with (
            patch("yapa.providers.concretes.lmstudio.get_config") as mock_get_config,
            patch("yapa.providers.concretes.lmstudio.AsyncOpenAI"),
        ):
            mock_get_config.return_value = Config()
            provider = LMStudioIP()

        assert provider.id == "lmstudio"
        assert provider.name == "LM Studio"
