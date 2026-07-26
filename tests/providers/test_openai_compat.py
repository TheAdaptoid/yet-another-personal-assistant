"""Tests for OpenAICompatibleProvider."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelType,
    StreamDelta,
    ToolCall,
    ToolCallDelta,
    ToolMessage,
    TokenUsage,
    UserMessage,
)
from yapa.providers.openai import OpenAIIP
from yapa.providers.openai_compat import OpenAICompatibleProvider


def _chunk(content=None, reasoning_content=None, tool_calls=None, **extra):
    attrs = {
        "content": content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }
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


class _TC:
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = SimpleNamespace(name=name, arguments=arguments)


@pytest.fixture
def compat_provider(mock_openai_client):
    with patch(
        "yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client
    ):
        provider = OpenAICompatibleProvider(
            identifier="test",
            name="Test",
            api_key="test-key",
            base_url="http://test",
            timeout=120,
        )
        provider._client = mock_openai_client
        return provider


class TestFormatModel:
    """Tests for OpenAICompatibleProvider._format_model()."""

    def test_llm_model(self, compat_provider) -> None:
        result = compat_provider._format_model("gpt-4")
        assert result.type == ModelType.LLM

    def test_embed_model(self, compat_provider) -> None:
        result = compat_provider._format_model("text-embedding-3")
        assert result.type == ModelType.OTHER

    def test_audio_model(self, compat_provider) -> None:
        result = compat_provider._format_model("my-audio-model")
        assert result.type == ModelType.OTHER

    def test_image_model(self, compat_provider) -> None:
        result = compat_provider._format_model("my-image-model")
        assert result.type == ModelType.OTHER


class TestListModelsImpl:
    """Tests for OpenAICompatibleProvider._list_models_impl()."""

    async def test_delegates_to_client(self, compat_provider) -> None:
        compat_provider._client.models.list = AsyncMock(
            return_value=SimpleNamespace(data=[_Model("gpt-4"), _Model("gpt-3.5")])
        )
        result = await compat_provider._list_models_impl()
        assert len(result) == 2
        assert result[0].id == "gpt-4"
        assert result[1].id == "gpt-3.5"

    async def test_filters_by_llm_type(self, compat_provider) -> None:
        compat_provider._client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[_Model("gpt-4"), _Model("text-embedding-3")]
            )
        )
        result = await compat_provider._list_models_impl(model_type=ModelType.LLM)
        assert len(result) == 1
        assert result[0].id == "gpt-4"

    async def test_filters_by_other_type(self, compat_provider) -> None:
        compat_provider._client.models.list = AsyncMock(
            return_value=SimpleNamespace(
                data=[_Model("gpt-4"), _Model("text-embedding-3")]
            )
        )
        result = await compat_provider._list_models_impl(model_type=ModelType.OTHER)
        assert len(result) == 1
        assert result[0].id == "text-embedding-3"


class TestGetModelImpl:
    """Tests for OpenAICompatibleProvider._get_model_impl()."""

    async def test_delegates_to_client(self, compat_provider) -> None:
        compat_provider._client.models.retrieve = AsyncMock(
            return_value=_Model("gpt-4")
        )
        result = await compat_provider._get_model_impl("gpt-4")
        assert result.id == "gpt-4"
        compat_provider._client.models.retrieve.assert_awaited_once_with("gpt-4")


class TestFormatMessage:
    """Tests for OpenAICompatibleProvider._format_message()."""

    def test_user_message(self, compat_provider) -> None:
        msg = UserMessage(content="hello")
        result = compat_provider._format_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_user_message_no_content_raises(self, compat_provider) -> None:
        with pytest.raises(ValueError, match="cannot be None"):
            compat_provider._format_message(UserMessage(content=None))

    def test_system_message(self, compat_provider) -> None:
        from yapa.models import SystemMessage

        msg = SystemMessage(content="be helpful")
        result = compat_provider._format_message(msg)
        assert result == {"role": "system", "content": "be helpful"}

    def test_assistant_message(self, compat_provider) -> None:
        msg = AssistantMessage(content="hi there")
        result = compat_provider._format_message(msg)
        assert result == {"role": "assistant", "content": "hi there"}

    def test_assistant_message_with_tool_calls(self, compat_provider) -> None:
        tc = ToolCall(id="call_1", tool_name="get_weather", arguments={"loc": "SF"})
        msg = AssistantMessage(content=None, tool_calls=[tc])
        result = compat_provider._format_message(msg)
        assert result == {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"loc": "SF"}),
                    },
                }
            ],
        }

    def test_tool_message(self, compat_provider) -> None:
        msg = ToolMessage(content="sunny", tool_call_id="call_1", tool_name="weather")
        result = compat_provider._format_message(msg)
        assert result == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny",
        }

    def test_unknown_role_raises(self, compat_provider) -> None:
        msg = SimpleNamespace(role="unknown", content="foo")
        with pytest.raises(ValueError, match="Unsupported message role"):
            compat_provider._format_message(msg)


class TestFormatTools:
    """Tests for OpenAICompatibleProvider._format_tools()."""

    def test_none(self, compat_provider) -> None:
        assert compat_provider._format_tools(None) is None

    def test_empty(self, compat_provider) -> None:
        assert compat_provider._format_tools([]) is None

    def test_formats_tools(self, compat_provider) -> None:

        from yapa.tools.base import Tool

        tool = MagicMock(spec=Tool)
        tool.name = "get_weather"
        tool.description = "Get weather"
        tool.parameters = {"type": "object", "properties": {}}
        result = compat_provider._format_tools([tool])
        assert result == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


class TestExtractReasoningContent:
    """Tests for OpenAICompatibleProvider._extract_reasoning_content()."""

    def test_reasoning_content(self, compat_provider) -> None:
        obj = SimpleNamespace(reasoning_content="thinking...", reasoning=None)
        assert compat_provider._extract_reasoning_content(obj) == "thinking..."

    def test_reasoning_fallback(self, compat_provider) -> None:
        obj = SimpleNamespace(reasoning="thinking...", reasoning_content=None)
        assert compat_provider._extract_reasoning_content(obj) == "thinking..."

    def test_both_prioritizes_reasoning(self, compat_provider) -> None:
        obj = SimpleNamespace(
            reasoning="old", reasoning_content="new"
        )
        assert compat_provider._extract_reasoning_content(obj) == "old"

    def test_empty_string_returns_none(self, compat_provider) -> None:
        obj = SimpleNamespace(reasoning="", reasoning_content=None)
        assert compat_provider._extract_reasoning_content(obj) is None

    def test_all_none(self, compat_provider) -> None:
        obj = SimpleNamespace(reasoning=None, reasoning_content=None)
        assert compat_provider._extract_reasoning_content(obj) is None


class TestCommonPreInvoke:
    """Tests for OpenAICompatibleProvider._common_pre_invoke()."""

    def test_builds_kwargs(self, compat_provider) -> None:
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)
        result = compat_provider._common_pre_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            params=params,
            stream=True,
        )
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert result["top_p"] == 0.9
        assert result["stream"] is True
        assert result["messages"] == [{"role": "user", "content": "hi"}]

    def test_default_params(self, compat_provider) -> None:
        result = compat_provider._common_pre_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            stream=False,
        )
        assert result["temperature"] is None
        assert result["max_tokens"] is None
        assert result["top_p"] is None
        assert result["stream"] is False

    def test_includes_tools(self, compat_provider) -> None:

        from yapa.tools.base import Tool

        tool = MagicMock(spec=Tool)
        tool.name = "get_weather"
        tool.description = "Get weather"
        tool.parameters = {"type": "object", "properties": {}}
        result = compat_provider._common_pre_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            tools=[tool],
            stream=True,
        )
        assert "tools" in result
        assert len(result["tools"]) == 1

    def test_omits_tools_when_none(self, compat_provider) -> None:
        result = compat_provider._common_pre_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            tools=None,
            stream=True,
        )
        assert "tools" not in result


class TestStreamChatImpl:
    """Tests for OpenAICompatibleProvider._stream_chat_impl()."""

    async def test_yields_content_chunks(self, compat_provider) -> None:
        stream = _stream(
            _chunk(content="Hello", reasoning_content=None),
            _chunk(content=" world", reasoning_content=None),
        )
        mock_create = AsyncMock(return_value=stream)
        compat_provider._client.chat.completions.create = mock_create

        results: list[StreamDelta] = []
        async for delta in compat_provider._stream_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
        ):
            results.append(delta)

        assert results[0].content == "Hello"
        assert results[1].content == " world"

    async def test_yields_reasoning_content(self, compat_provider) -> None:
        stream = _stream(
            _chunk(content=None, reasoning_content="thinking..."),
        )
        mock_create = AsyncMock(return_value=stream)
        compat_provider._client.chat.completions.create = mock_create

        results: list[StreamDelta] = []
        async for delta in compat_provider._stream_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
        ):
            results.append(delta)

        assert results[0].reasoning_content == "thinking..."

    async def test_yields_tool_call_deltas(self, compat_provider) -> None:
        tc1 = _TC(
            index=0, id="call_1", name="get_weather", arguments='{"loc":"SF"}'
        )
        stream = _stream(
            _chunk(content=None, reasoning_content=None, tool_calls=[tc1]),
        )
        mock_create = AsyncMock(return_value=stream)
        compat_provider._client.chat.completions.create = mock_create

        results: list[StreamDelta] = []
        async for delta in compat_provider._stream_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="weather?")],
        ):
            results.append(delta)

        assert len(results[0].tool_calls) == 1
        assert results[0].tool_calls[0] == ToolCallDelta(
            index=0, id="call_1", name="get_weather", arguments='{"loc":"SF"}'
        )

    async def test_passes_params_to_client(self, compat_provider) -> None:
        stream = _stream(_chunk(content="ok", reasoning_content=None))
        mock_create = AsyncMock(return_value=stream)
        compat_provider._client.chat.completions.create = mock_create
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        async for _ in compat_provider._stream_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            params=params,
        ):
            pass

        mock_create.assert_awaited_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["model"] == "gpt-4"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is True


class TestOpenAIModelMetadata:
    """Tests for OpenAIIP._format_model() metadata lookup."""

    @pytest.fixture
    def openai_provider(self, mock_openai_client):
        from yapa.config import Config

        with patch(
            "yapa.providers.openai_compat.AsyncOpenAI",
            return_value=mock_openai_client,
        ):
            provider = OpenAIIP(config=Config(openai_api_key="sk-test"))
        provider._client = mock_openai_client
        return provider

    def test_known_model_gets_metadata(self, openai_provider):
        model = openai_provider._format_model("gpt-4o")
        assert model.context_length == 128000
        assert model.max_output == 16384
        assert model.supports_tools is True
        assert model.supports_vision is True

    def test_unknown_model_defaults(self, openai_provider):
        model = openai_provider._format_model("unknown-model")
        assert model.context_length is None
        assert model.max_output is None
        assert model.supports_tools is False
        assert model.supports_vision is False

    def test_embed_model_gets_other_type(self, openai_provider):
        model = openai_provider._format_model("text-embedding-3")
        assert model.type == ModelType.OTHER
        assert model.supports_tools is False
        assert model.supports_vision is False

    def test_all_known_models_have_metadata(self, openai_provider):
        from yapa.providers.openai.provider import _MODEL_METADATA

        for model_id in _MODEL_METADATA:
            model = openai_provider._format_model(model_id)
            assert model.context_length is not None
            assert model.max_output is not None


class TestStaticChatImpl:
    """Tests for OpenAICompatibleProvider._static_chat_impl()."""

    def _make_response(self, content="Hello", reasoning_content=None, tool_calls=None):
        message = SimpleNamespace(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    async def test_returns_assistant_message(self, compat_provider) -> None:
        response = self._make_response(content="Hello world")
        mock_create = AsyncMock(return_value=response)
        compat_provider._client.chat.completions.create = mock_create

        result = await compat_provider._static_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
        )

        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello world"
        assert result.role == "assistant"

    async def test_returns_reasoning_content(self, compat_provider) -> None:
        response = self._make_response(
            content="answer", reasoning_content="thinking..."
        )
        mock_create = AsyncMock(return_value=response)
        compat_provider._client.chat.completions.create = mock_create

        result = await compat_provider._static_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
        )

        assert result.reasoning_content == "thinking..."

    async def test_returns_tool_calls(self, compat_provider) -> None:
        tc1 = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"loc":"SF"}',
            ),
        )
        response = self._make_response(content=None, tool_calls=[tc1])
        mock_create = AsyncMock(return_value=response)
        compat_provider._client.chat.completions.create = mock_create

        result = await compat_provider._static_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="weather?")],
        )

        assert result.tool_calls == [
            ToolCall(id="call_1", tool_name="get_weather", arguments={"loc": "SF"})
        ]
        assert result.content is None

    async def test_passes_params_to_client(self, compat_provider) -> None:
        response = self._make_response(content="ok")
        mock_create = AsyncMock(return_value=response)
        compat_provider._client.chat.completions.create = mock_create
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        await compat_provider._static_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            params=params,
        )

        mock_create.assert_awaited_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is False

    async def test_handles_none_content(self, compat_provider) -> None:
        response = self._make_response(content=None)
        mock_create = AsyncMock(return_value=response)
        compat_provider._client.chat.completions.create = mock_create

        result = await compat_provider._static_chat_impl(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
        )

        assert result.content is None
