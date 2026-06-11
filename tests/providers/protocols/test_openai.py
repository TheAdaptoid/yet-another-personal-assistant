"""Tests for OpenAI protocol implementations."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from yapa.models import InferenceParams, ModelData, ModelType, StreamDelta
from yapa.providers.protocols.openai import (
    OpenAIInferenceProtocol,
    OpenAIModelFetchProtocol,
)


def _chunk(content=None, reasoning_content=None, **extra):
    """Build a mock OpenAI streaming chunk."""
    attrs = {"content": content, "reasoning_content": reasoning_content}
    attrs.update(extra)
    delta = SimpleNamespace(**attrs)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


async def _stream(*chunks):
    for c in chunks:
        yield c


class _Model:
    def __init__(self, id):
        self.id = id


class TestOpenAIModelFetchProtocol:
    """Tests for OpenAIModelFetchProtocol."""

    @pytest.fixture
    def protocol(self):
        client = MagicMock()
        return OpenAIModelFetchProtocol(client=client, provider_id="test_prov")

    @pytest.fixture
    def mock_client(self, protocol):
        return protocol.client

    async def test_list_models_returns_formatted_data(self, protocol, mock_client):
        mock_client.models.list = AsyncMock(
            return_value=SimpleNamespace(data=[_Model("gpt-4"), _Model("gpt-3.5")])
        )

        result = await protocol.list_models()

        assert result == [
            ModelData(id="gpt-4", provider_id="test_prov", type=ModelType.LLM),
            ModelData(id="gpt-3.5", provider_id="test_prov", type=ModelType.LLM),
        ]
        mock_client.models.list.assert_awaited_once()

    async def test_list_models_filters_by_llm_type(self, protocol, mock_client):
        mock_client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[_Model("gpt-4"), _Model("text-embedding-3")]
            )
        )

        result = await protocol.list_models(model_type=ModelType.LLM)

        assert result == [
            ModelData(id="gpt-4", provider_id="test_prov", type=ModelType.LLM),
        ]

    async def test_list_models_filters_by_other_type(self, protocol, mock_client):
        mock_client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[_Model("gpt-4"), _Model("text-embedding-3")]
            )
        )

        result = await protocol.list_models(model_type=ModelType.OTHER)

        assert result == [
            ModelData(
                id="text-embedding-3", provider_id="test_prov", type=ModelType.OTHER
            ),
        ]

    async def test_get_model_returns_formatted_data(self, protocol, mock_client):
        mock_client.models.retrieve = AsyncMock(return_value=_Model("gpt-4"))

        result = await protocol.get_model(model_id="gpt-4")

        assert result == ModelData(
            id="gpt-4", provider_id="test_prov", type=ModelType.LLM
        )
        mock_client.models.retrieve.assert_awaited_once_with("gpt-4")

    def test_format_model_llm(self, protocol):
        data = _Model("gpt-4")
        result = protocol._format_model(data.id)

        assert result.type == ModelType.LLM

    def test_format_model_embed(self, protocol):
        data = _Model("text-embedding-3")
        result = protocol._format_model(data.id)

        assert result.type == ModelType.OTHER

    def test_format_model_audio(self, protocol):
        data = _Model("my-audio-model")
        result = protocol._format_model(data.id)

        assert result.type == ModelType.OTHER

    def test_format_model_image(self, protocol):
        data = _Model("my-image-model")
        result = protocol._format_model(data.id)

        assert result.type == ModelType.OTHER


class TestOpenAIInferenceProtocol:
    """Tests for OpenAIInferenceProtocol."""

    @pytest.fixture
    def protocol(self):
        client = MagicMock()
        return OpenAIInferenceProtocol(client=client)

    @pytest.fixture
    def mock_client(self, protocol):
        return protocol.client

    async def test_invoke_llm_streams_content(self, protocol, mock_client):
        stream = _stream(
            _chunk(content="Hello", reasoning_content=None),
            _chunk(content=" world", reasoning_content=None),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.invoke_llm(
            model_id="gpt-4", messages=[_msg("user", "hi")]
        ):
            results.append(delta)

        assert results == [
            StreamDelta(content="Hello", reasoning_content=None, done=False),
            StreamDelta(content=" world", reasoning_content=None, done=False),
        ]

    async def test_invoke_llm_uses_reasoning_content(
        self, protocol, mock_client
    ) -> None:
        stream = _stream(
            _chunk(content=None, reasoning_content="thinking..."),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.invoke_llm(
            model_id="gpt-4", messages=[_msg("user", "hi")]
        ):
            results.append(delta)

        assert results[0] == StreamDelta(
            content=None, reasoning_content="thinking...", done=False
        )

    async def test_invoke_llm_falls_back_to_reasoning(
        self, protocol, mock_client
    ) -> None:
        stream = _stream(
            _chunk(content="answer", reasoning_content=None),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.invoke_llm(
            model_id="gpt-4", messages=[_msg("user", "hi")]
        ):
            results.append(delta)

        assert results[0] == StreamDelta(
            content="answer", reasoning_content=None, done=False
        )

    async def test_invoke_llm_uses_reasoning_fallback(
        self, protocol, mock_client
    ) -> None:
        stream = _stream(
            _chunk(content="answer", reasoning_content=None, reasoning="thinking..."),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.invoke_llm(
            model_id="gpt-4", messages=[_msg("user", "hi")]
        ):
            results.append(delta)

        assert results[0] == StreamDelta(
            content="answer", reasoning_content="thinking...", done=False
        )

    async def test_passes_params_to_create(self, protocol, mock_client):
        stream = _stream(_chunk(content="ok", reasoning_content=None))
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        async for _ in protocol.invoke_llm(
            model_id="gpt-4",
            messages=[_msg("user", "hi")],
            params=params,
        ):
            pass

        mock_create.assert_awaited_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            stream=True,
            timeout=120,
        )

    async def test_uses_default_params_when_none(self, protocol, mock_client):
        stream = _stream(_chunk(content="ok", reasoning_content=None))
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create

        async for _ in protocol.invoke_llm(
            model_id="gpt-4",
            messages=[_msg("user", "hi")],
        ):
            pass

        mock_create.assert_awaited_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
            temperature=None,
            max_tokens=None,
            top_p=None,
            stream=True,
            timeout=120,
        )

    def test_format_user_message(self, protocol):
        msg = _msg("user", "hello")
        result = protocol._format_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_format_system_message(self, protocol):
        msg = _msg("system", "be helpful")
        result = protocol._format_message(msg)
        assert result == {"role": "system", "content": "be helpful"}

    def test_format_assistant_message(self, protocol):
        msg = _msg("assistant", "hi there")
        result = protocol._format_message(msg)
        assert result == {"role": "assistant", "content": "hi there"}

    def test_format_unknown_role_raises(self, protocol):
        msg = _msg("unknown", "foo")
        with pytest.raises(ValueError, match="Unsupported message role"):
            protocol._format_message(msg)


def _msg(role, content):
    return SimpleNamespace(role=role, content=content)
