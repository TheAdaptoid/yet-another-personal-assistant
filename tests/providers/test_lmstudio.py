"""Tests for LM Studio provider — native API model metadata."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from yapa.models import ModelType
from yapa.providers.lmstudio import LMStudioIP
from yapa.services.config import Config, ProviderConfig


@pytest.fixture
def lmstudio_provider(mock_openai_client):
    with patch(
        "yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client
    ):
        provider = LMStudioIP(
            config=Config(
                provider_configs={"lmstudio": ProviderConfig(api_key="test-key")}
            )
        )
    provider._client = mock_openai_client
    return provider


@pytest.fixture
def native_response():
    return {
        "models": [
            {
                "key": "qwen/qwen3-4b@q4_k_m",
                "type": "llm",
                "loaded_instances": [
                    {"config": {"context_length": 32768, "gpu_offload_layers": 0}}
                ],
                "capabilities": {
                    "vision": False,
                    "trained_for_tool_use": True,
                },
                "max_context_length": 32768,
            },
            {
                "key": "llama-3.2-8b",
                "type": "llm",
                "loaded_instances": [],
                "capabilities": {
                    "vision": False,
                    "trained_for_tool_use": False,
                },
                "max_context_length": None,
            },
            {
                "key": "text-embedding-3",
                "type": "embedding",
                "loaded_instances": [{"config": {"context_length": 8192}}],
                "capabilities": {},
                "max_context_length": 8192,
            },
        ]
    }


@pytest.fixture
def mock_httpx_get(native_response):
    mock_resp = AsyncMock(spec=httpx.Response)
    mock_resp.json.return_value = native_response
    mock_resp.raise_for_status.return_value = None
    with patch.object(
        httpx.AsyncClient, "get", new=AsyncMock(return_value=mock_resp)
    ) as mock:
        yield mock


class TestNativeModelList:
    """LM Studio native /api/v1/models endpoint parsing."""

    async def test_populates_context_length_and_caps(
        self, lmstudio_provider, mock_httpx_get
    ):
        models = await lmstudio_provider._list_models_impl()
        assert len(models) == 3
        qwen = models[0]
        assert qwen.id == "qwen/qwen3-4b@q4_k_m"
        assert qwen.context_length == 32768
        assert qwen.supports_tools is True
        assert qwen.supports_vision is False
        assert qwen.type == ModelType.LLM

    async def test_graceful_when_no_loaded_instance(
        self, lmstudio_provider, mock_httpx_get
    ):
        models = await lmstudio_provider._list_models_impl()
        llama = models[1]
        assert llama.id == "llama-3.2-8b"
        assert llama.context_length is None
        assert llama.supports_tools is False

    async def test_detects_embed_model(
        self, lmstudio_provider, mock_httpx_get
    ):
        models = await lmstudio_provider._list_models_impl()
        embed = models[2]
        assert embed.id == "text-embedding-3"
        assert embed.type == ModelType.OTHER
        assert embed.context_length == 8192

    async def test_filters_by_model_type(
        self, lmstudio_provider, mock_httpx_get
    ):
        llms = await lmstudio_provider._list_models_impl(
            model_type=ModelType.LLM
        )
        assert len(llms) == 2
        assert all(m.type == ModelType.LLM for m in llms)

    async def test_get_model_impl_returns_from_list(
        self, lmstudio_provider, mock_httpx_get
    ):
        model = await lmstudio_provider._get_model_impl(
            "qwen/qwen3-4b@q4_k_m"
        )
        assert model.id == "qwen/qwen3-4b@q4_k_m"
        assert model.context_length == 32768

    async def test_get_model_impl_falls_back(
        self, lmstudio_provider, mock_httpx_get
    ):
        model = await lmstudio_provider._get_model_impl("unknown-model")
        assert model.id == "unknown-model"
        assert model.context_length is None
