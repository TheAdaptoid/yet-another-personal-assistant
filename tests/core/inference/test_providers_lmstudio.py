"""Tests for LM Studio inference provider."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.core.inference.exceptions import ModelsFetchError
from yapa.core.inference.providers.lmstudio import (
    LMStudioIP,
    convert_model_id_to_name,
)
from yapa.shared.models import ModelData


class TestConvertModelIdToName:
    """convert_model_id_to_name — model ID to human-readable name."""

    def test_standard_path(self):
        """A model ID with a prefix is converted to a readable name."""
        assert convert_model_id_to_name("qwen/qwen3-4b-2507") == "Qwen3 4b 2507"

    def test_no_slash(self):
        """A model ID without a slash is handled correctly."""
        assert convert_model_id_to_name("phi-4") == "Phi 4"

    def test_multiple_dashes(self):
        """Multiple dashes are replaced with spaces."""
        assert convert_model_id_to_name("deepseek/deepseek-v3") == "Deepseek v3"


@pytest.fixture
def dummy_logger():
    """Return a logger with a NullHandler for test isolation."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def mock_lmstudio_env():
    """Patch get_config and AsyncOpenAI so LMStudioIP uses a mock client."""
    config_patcher = patch(
        "yapa.core.inference.providers.lmstudio.get_config",
        return_value=MagicMock(
            lmstudio_api_key="test-key",
            lmstudio_base_url="http://test:1234/v1",
        ),
    )
    config_patcher.start()

    client_patcher = patch(
        "yapa.core.inference.providers.lmstudio.AsyncOpenAI",
    )
    mock_client_cls = client_patcher.start()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    yield mock_client

    config_patcher.stop()
    client_patcher.stop()


class TestLMStudioIP:
    """LMStudioIP — inference provider for LM Studio."""

    @pytest.mark.asyncio
    async def test_get_models_returns_model_data_list(
        self, dummy_logger, mock_lmstudio_env
    ):
        """get_models returns ModelData objects from the API response."""
        mock_client = mock_lmstudio_env
        model_a = MagicMock(id="model-alpha")
        model_b = MagicMock(id="model-beta")
        response = MagicMock()
        response.data = [model_a, model_b]
        mock_client.models.list = AsyncMock(return_value=response)

        provider = LMStudioIP(dummy_logger)
        models = await provider.get_models()

        assert len(models) == 2
        assert all(isinstance(m, ModelData) for m in models)
        assert models[0].id == "model-alpha"
        assert models[0].provider_id == "lmstudio"
        assert models[1].id == "model-beta"
        assert models[1].provider_id == "lmstudio"

    @pytest.mark.asyncio
    async def test_get_models_raises_on_api_error(
        self, dummy_logger, mock_lmstudio_env
    ):
        """get_models raises ModelsFetchError when the API call fails."""
        mock_client = mock_lmstudio_env
        mock_client.models.list = AsyncMock(
            side_effect=ConnectionError("connection refused")
        )

        provider = LMStudioIP(dummy_logger)

        with pytest.raises(ModelsFetchError) as exc_info:
            await provider.get_models()

        assert exc_info.value.provider == "LM Studio"
        assert isinstance(exc_info.value.cause, ConnectionError)
