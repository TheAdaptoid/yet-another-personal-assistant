"""Tests for OpenRouter inference provider."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.core.inference.exceptions import ModelsFetchError
from yapa.core.inference.providers.openrouter import OpenRouterIP
from yapa.shared.models import ModelData


@pytest.fixture
def dummy_logger():
    """Return a logger with a NullHandler for test isolation."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def mock_openrouter_env():
    """Patch get_config and AsyncOpenAI so OpenRouterIP uses a mock client."""
    config_patcher = patch(
        "yapa.core.inference.providers.openrouter.get_config",
        return_value=MagicMock(
            openrouter_api_key="sk-test",
            openrouter_base_url="https://openrouter.ai/api/v1",
        ),
    )
    config_patcher.start()

    client_patcher = patch(
        "yapa.core.inference.providers.openrouter.AsyncOpenAI",
    )
    mock_client_cls = client_patcher.start()
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    yield mock_client

    config_patcher.stop()
    client_patcher.stop()


class TestOpenRouterIP:
    """OpenRouterIP — inference provider for OpenRouter."""

    @pytest.mark.asyncio
    async def test_get_models_returns_model_data_list(
        self, dummy_logger, mock_openrouter_env
    ):
        """get_models returns ModelData objects from the API response."""
        mock_client = mock_openrouter_env
        model_a = MagicMock(id="openai/gpt-4o")
        model_b = MagicMock(id="anthropic/claude-3")
        response = MagicMock()
        response.data = [model_a, model_b]
        mock_client.models.list = AsyncMock(return_value=response)

        provider = OpenRouterIP(dummy_logger)
        models = await provider.get_models()

        assert len(models) == 2
        assert all(isinstance(m, ModelData) for m in models)
        assert models[0].id == "openai/gpt-4o"
        assert models[0].provider_id == "openrouter"
        assert models[1].id == "anthropic/claude-3"
        assert models[1].provider_id == "openrouter"

    @pytest.mark.asyncio
    async def test_get_models_raises_on_api_error(
        self, dummy_logger, mock_openrouter_env
    ):
        """get_models raises ModelsFetchError when the API call fails."""
        mock_client = mock_openrouter_env
        mock_client.models.list = AsyncMock(
            side_effect=ConnectionError("timeout")
        )

        provider = OpenRouterIP(dummy_logger)

        with pytest.raises(ModelsFetchError) as exc_info:
            await provider.get_models()

        assert exc_info.value.provider == "OpenRouter"
        assert isinstance(exc_info.value.cause, ConnectionError)
