"""Tests for OpenRouter provider — native API model metadata."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from yapa.config import Config
from yapa.models import ModelType
from yapa.providers.openrouter import OpenRouterProvider


@pytest.fixture
def openrouter_provider(mock_openai_client):
    with patch(
        "yapa.providers.openai_compat.AsyncOpenAI",
        return_value=mock_openai_client,
    ):
        provider = OpenRouterProvider(
            config=Config(openrouter_api_key="sk-or-test")
        )
    provider._client = mock_openai_client
    return provider


@pytest.fixture
def native_or_response():
    return {
        "data": [
            {
                "id": "openai/gpt-4o",
                "name": "GPT-4o",
                "context_length": 128000,
                "max_completion_tokens": 16384,
                "pricing": {
                    "prompt": "0.0000025",
                    "completion": "0.00001",
                    "image": "0.000007225",
                    "request": "0",
                },
                "architecture": {
                    "modality": "text+vision",
                    "tokenizer": "cl100k_base",
                    "instruct_type": "chat",
                },
                "supported_parameters": [
                    "frequency_penalty",
                    "max_tokens",
                    "tools",
                    "temperature",
                ],
            },
            {
                "id": "openai/gpt-4o-mini",
                "name": "GPT-4o mini",
                "context_length": 128000,
                "max_completion_tokens": 16384,
                "pricing": {
                    "prompt": "0.00000015",
                    "completion": "0.0000006",
                    "image": "0.000007225",
                    "request": "0",
                },
                "architecture": {
                    "modality": "text+vision",
                    "tokenizer": "cl100k_base",
                    "instruct_type": "chat",
                },
                "supported_parameters": [
                    "tools",
                    "temperature",
                ],
            },
            {
                "id": "openai/text-embedding-3",
                "name": "Text Embedding 3",
                "context_length": None,
                "architecture": {
                    "modality": "embeddings",
                },
                "supported_parameters": [],
            },
        ]
    }


@pytest.fixture
def mock_or_httpx_get(native_or_response):
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.json.return_value = native_or_response
    mock_resp.raise_for_status.return_value = None
    with patch.object(
        httpx.AsyncClient, "get", new=AsyncMock(return_value=mock_resp)
    ) as mock:
        yield mock


class TestOpenRouterModelList:
    """OpenRouter native /api/v1/models endpoint parsing."""

    async def test_populates_context_length(
        self, openrouter_provider, mock_or_httpx_get
    ):
        models = await openrouter_provider._list_models_impl()
        gpt4o = models[0]
        assert gpt4o.id == "openai/gpt-4o"
        assert gpt4o.context_length == 128000
        assert gpt4o.max_output == 16384

    async def test_populates_pricing(
        self, openrouter_provider, mock_or_httpx_get
    ):
        models = await openrouter_provider._list_models_impl()
        gpt4o = models[0]
        assert gpt4o.pricing is not None
        assert gpt4o.pricing["input"] == 2.5
        assert gpt4o.pricing["output"] == 10.0

    async def test_detects_vision_and_tools(
        self, openrouter_provider, mock_or_httpx_get
    ):
        models = await openrouter_provider._list_models_impl()
        gpt4o = models[0]
        assert gpt4o.supports_vision is True
        assert gpt4o.supports_tools is True

    async def test_embed_model_gets_other_type(
        self, openrouter_provider, mock_or_httpx_get
    ):
        models = await openrouter_provider._list_models_impl()
        embed = models[2]
        assert embed.type == ModelType.OTHER
        assert embed.pricing is None

    async def test_filters_by_model_type(
        self, openrouter_provider, mock_or_httpx_get
    ):
        llms = await openrouter_provider._list_models_impl(
            model_type=ModelType.LLM
        )
        assert len(llms) == 2

    async def test_get_model_impl_returns_from_list(
        self, openrouter_provider, mock_or_httpx_get
    ):
        model = await openrouter_provider._get_model_impl("openai/gpt-4o")
        assert model.id == "openai/gpt-4o"
        assert model.context_length == 128000

    async def test_get_model_impl_falls_back(
        self, openrouter_provider, mock_or_httpx_get
    ):
        model = await openrouter_provider._get_model_impl("unknown-model")
        assert model.id == "unknown-model"
        assert model.context_length is None
