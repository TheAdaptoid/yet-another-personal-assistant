from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from yapa.models import InferenceParams, ModelData, StreamDelta
from yapa.providers.exceptions import ModelInvocationError, ModelsFetchError


async def _stream(*deltas: dict) -> AsyncMock:
    for delta_kwargs in deltas:
        delta = SimpleNamespace(**delta_kwargs)
        choice = SimpleNamespace(delta=delta)
        chunk = SimpleNamespace(choices=[choice])
        yield chunk


class TestInit:
    def test_stores_attributes(self, mock_client: MagicMock) -> None:
        from yapa.providers.base import InferenceProvider

        provider = InferenceProvider("my_prov", "My Provider", mock_client)
        assert provider._identifier == "my_prov"
        assert provider.id == "my_prov"
        assert provider._client is mock_client


class TestGetModels:
    async def test_returns_model_data_list(
        self, provider, mock_client: MagicMock
    ) -> None:
        model_a = SimpleNamespace(id="gpt-4")
        model_b = SimpleNamespace(id="gpt-3.5")
        mock_list = AsyncMock(return_value=SimpleNamespace(data=[model_a, model_b]))
        mock_client.models.list = mock_list

        result = await provider.get_models()

        assert result == [
            ModelData(id="gpt-4", provider_id="test_prov"),
            ModelData(id="gpt-3.5", provider_id="test_prov"),
        ]
        mock_list.assert_awaited_once_with()

    async def test_raises_models_fetch_error_on_api_error(
        self, provider, mock_client: MagicMock
    ) -> None:
        api_error = Exception("API failure")
        mock_client.models.list = AsyncMock(side_effect=api_error)

        with pytest.raises(ModelsFetchError) as exc_info:
            await provider.get_models()

        assert "API failure" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_error

    async def test_raises_models_fetch_error_on_unexpected_error(
        self, provider, mock_client: MagicMock
    ) -> None:
        value_error = ValueError("weird")
        mock_client.models.list = AsyncMock(side_effect=value_error)

        with pytest.raises(ModelsFetchError) as exc_info:
            await provider.get_models()

        assert "weird" in str(exc_info.value)
        assert exc_info.value.__cause__ is value_error


class TestInvokeModel:
    async def test_streams_content_chunks(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        stream = _stream(
            {"content": "Hello", "reasoning_content": None},
            {"content": " world", "reasoning_content": None},
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for chunk in provider.invoke_model("gpt-4", sample_messages):
            results.append(chunk)

        assert results == [
            StreamDelta(content="Hello", reasoning_content=None, done=False),
            StreamDelta(content=" world", reasoning_content=None, done=False),
            StreamDelta(content=None, reasoning_content=None, done=True),
        ]

    async def test_streams_reasoning_content(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        stream = _stream(
            {"content": None, "reasoning_content": "thinking..."},
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for chunk in provider.invoke_model("gpt-4", sample_messages):
            results.append(chunk)

        assert results[0] == StreamDelta(
            content=None, reasoning_content="thinking...", done=False
        )

    async def test_streams_reasoning_fallback(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        stream = _stream(
            {"content": "answer", "reasoning": "thinking..."},
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for chunk in provider.invoke_model("gpt-4", sample_messages):
            results.append(chunk)

        assert results[0] == StreamDelta(
            content="answer", reasoning_content="thinking...", done=False
        )

    async def test_yields_done_delta(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        stream = _stream({"content": "Hi", "reasoning_content": None})
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for chunk in provider.invoke_model("gpt-4", sample_messages):
            results.append(chunk)

        assert results[-1] == StreamDelta(
            content=None, reasoning_content=None, done=True
        )

    async def test_passes_inference_params(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        stream = _stream({"content": "ok", "reasoning_content": None})
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        async for _ in provider.invoke_model("gpt-4", sample_messages, params=params):
            pass

        mock_create.assert_awaited_once_with(
            model="gpt-4",
            messages=[sample_messages[0].to_openai_format()],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            stream=True,
        )

    async def test_no_params_passes_none_values(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        stream = _stream({"content": "ok", "reasoning_content": None})
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create

        async for _ in provider.invoke_model("gpt-4", sample_messages, params=None):
            pass

        mock_create.assert_awaited_once_with(
            model="gpt-4",
            messages=[sample_messages[0].to_openai_format()],
            temperature=None,
            max_tokens=None,
            top_p=None,
            stream=True,
        )

    async def test_uses_default_params_when_none(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        stream = _stream({"content": "ok", "reasoning_content": None})
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create

        async for _ in provider.invoke_model("gpt-4", sample_messages):
            pass

        mock_create.assert_awaited_once_with(
            model="gpt-4",
            messages=[sample_messages[0].to_openai_format()],
            temperature=None,
            max_tokens=None,
            top_p=None,
            stream=True,
        )

    async def test_raises_model_invocation_error_on_api_error(
        self, provider, mock_client: MagicMock, sample_messages
    ) -> None:
        api_error = Exception("API down")
        mock_client.chat.completions.create = AsyncMock(side_effect=api_error)

        with pytest.raises(ModelInvocationError) as exc_info:
            async for _ in provider.invoke_model("gpt-4", sample_messages):
                pass

        assert "API down" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_error
