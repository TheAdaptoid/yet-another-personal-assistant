"""Tests for InferenceProvider base class."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from yapa.core.inference.exceptions import InferenceError
from yapa.core.inference.provider import InferenceProvider
from yapa.shared.models import AssistantMessage, InferenceParams, Message, ModelData


class _ConcreteProvider(InferenceProvider):
    """Minimal concrete subclass for testing the base class."""

    async def get_models(self) -> list[ModelData]:
        return []


@pytest.fixture
def dummy_logger():
    """Return a logger with a NullHandler for test isolation."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@pytest.fixture
def mock_client():
    """Return a MagicMock(spec=AsyncOpenAI) with a chat.completions chain."""
    client = MagicMock(spec=AsyncOpenAI)
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    return client


@pytest.fixture
def provider(dummy_logger, mock_client):
    """Return a concrete InferenceProvider wired with a mock client."""
    return _ConcreteProvider(
        logger=dummy_logger,
        identifier="test-provider",
        name="Test Provider",
        client=mock_client,
    )


class TestInferenceProvider:
    """InferenceProvider — base class for inference providers."""

    def test_id_property(self, provider):
        """Id returns the identifier passed at construction."""
        assert provider.id == "test-provider"

    def test_name_property(self, provider):
        """Name returns the name passed at construction."""
        assert provider.name == "Test Provider"

    @pytest.mark.asyncio
    async def test_get_models_empty_by_default(self, provider):
        """The concrete subclass returns an empty list."""
        models = await provider.get_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_invoke_model_returns_assistant_message(self, provider, mock_client):
        """invoke_model calls the OpenAI client and returns an AssistantMessage."""
        msg = MagicMock(spec=ChatCompletionMessage)
        msg.content = "Hello, world!"
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        mock_client.chat.completions.create = AsyncMock(return_value=response)

        model = ModelData(
            id="test-model",
            name="Test Model",
            provider_id="test-provider",
            provider_name="Test Provider",
        )
        messages: list[Message] = []
        result = await provider.invoke_model(model, messages)

        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello, world!"
        assert result.model == "test-model"
        mock_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_model_forwards_params(self, provider, mock_client):
        """invoke_model passes inference params to the client."""
        msg = MagicMock(spec=ChatCompletionMessage)
        msg.content = "Hi"
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        mock_client.chat.completions.create = AsyncMock(return_value=response)

        model = ModelData(
            id="test-model",
            name="Test Model",
            provider_id="test-provider",
            provider_name="Test Provider",
        )
        params = InferenceParams(temperature=0.7, max_tokens=100)
        result = await provider.invoke_model(model, [], params=params)

        assert result.content == "Hi"
        call = mock_client.chat.completions.create.await_args
        assert call is not None
        assert call.kwargs.get("temperature") == 0.7
        assert call.kwargs.get("max_tokens") == 100

    @pytest.mark.asyncio
    async def test_invoke_model_raises_inference_error(self, provider, mock_client):
        """invoke_model raises InferenceError when the API call fails."""
        mock_client.chat.completions.create = AsyncMock(
            side_effect=ConnectionError("API down")
        )

        model = ModelData(
            id="test-model",
            name="Test Model",
            provider_id="test-provider",
            provider_name="Test Provider",
        )

        with pytest.raises(InferenceError) as exc_info:
            await provider.invoke_model(model, [])

        assert exc_info.value.model_id == "test-model"
        assert isinstance(exc_info.value.cause, ConnectionError)
