"""Tests for OpenAI protocol implementations."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
    ToolCall,
    ToolCallDelta,
    ToolMessage,
    UserMessage,
)


def _chunk(content=None, reasoning_content=None, tool_calls=None, **extra):
    """Build a mock OpenAI streaming chunk."""
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
    """Mock ChoiceDeltaToolCall for testing."""

    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = SimpleNamespace(
            name=name,
            arguments=arguments,
        )


class TestOpenAIModelFetchProtocol:
    """Tests for OpenAIModelFetchProtocol."""

    @pytest.fixture
    def protocol(self):
        from yapa.providers.openai.protocols import OpenAIModelFetchProtocol

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


class TestOpenAILLMInferenceProtocol:
    """Tests for OpenAILLMInferenceProtocol."""

    @pytest.fixture
    def protocol(self):
        from yapa.providers.openai.protocols import OpenAILLMInferenceProtocol

        client = MagicMock()
        return OpenAILLMInferenceProtocol(client=client)

    @pytest.fixture
    def mock_client(self, protocol):
        return protocol.client

    # ── _format_message tests ────────────────────────────────────────

    def test_format_user_message(self, protocol):
        msg = UserMessage(content="hello")
        result = protocol._format_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_format_user_message_no_content_raises(self, protocol):
        with pytest.raises(ValueError, match="cannot be None"):
            protocol._format_message(UserMessage(content=None))

    def test_format_system_message(self, protocol):
        msg = SimpleNamespace(role="system", content="be helpful")
        result = protocol._format_message(msg)
        assert result == {"role": "system", "content": "be helpful"}

    def test_format_assistant_message(self, protocol):
        msg = AssistantMessage(content="hi there")
        result = protocol._format_message(msg)
        assert result == {"role": "assistant", "content": "hi there"}

    def test_format_assistant_message_with_tool_calls(self, protocol):
        tc = ToolCall(id="call_1", tool_name="get_weather", arguments={"loc": "SF"})
        msg = AssistantMessage(content=None, tool_calls=[tc])
        result = protocol._format_message(msg)
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

    def test_format_tool_message(self, protocol):
        msg = ToolMessage(content="sunny", tool_call_id="call_1", tool_name="weather")
        result = protocol._format_message(msg)
        assert result == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "sunny",
        }

    def test_format_unknown_role_raises(self, protocol):
        msg = SimpleNamespace(role="unknown", content="foo")
        with pytest.raises(ValueError, match="Unsupported message role"):
            protocol._format_message(msg)

    # ── _format_tools tests ──────────────────────────────────────────

    def test_format_tools_none(self, protocol):
        assert protocol._format_tools(None) is None

    def test_format_tools_empty(self, protocol):
        assert protocol._format_tools([]) is None

    def test_format_tools(self, protocol):
        from unittest.mock import MagicMock

        from yapa.tools.base import Tool

        tool = MagicMock(spec=Tool)
        tool.name = "get_weather"
        tool.description = "Get weather"
        tool.parameters = {"type": "object", "properties": {}}
        result = protocol._format_tools([tool])
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

    # ── stream_invoke tests ──────────────────────────────────────────

    async def test_stream_invoke_content(self, protocol, mock_client):
        stream = _stream(
            _chunk(content="Hello", reasoning_content=None),
            _chunk(content=" world", reasoning_content=None),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.stream_invoke(
            model_id="gpt-4", messages=[UserMessage(content="hi")]
        ):
            results.append(delta)

        assert results == [
            StreamDelta(content="Hello", reasoning_content=None, done=False),
            StreamDelta(content=" world", reasoning_content=None, done=False),
        ]

    async def test_stream_invoke_reasoning_content(self, protocol, mock_client):
        stream = _stream(
            _chunk(content=None, reasoning_content="thinking..."),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.stream_invoke(
            model_id="gpt-4", messages=[UserMessage(content="hi")]
        ):
            results.append(delta)

        assert results[0].reasoning_content == "thinking..."

    async def test_stream_invoke_reasoning_fallback(self, protocol, mock_client):
        stream = _stream(
            _chunk(content="answer", reasoning=None, reasoning_content="thinking..."),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.stream_invoke(
            model_id="gpt-4", messages=[UserMessage(content="hi")]
        ):
            results.append(delta)

        assert results[0].reasoning_content == "thinking..."

    async def test_stream_invoke_tool_calls(self, protocol, mock_client):
        tc1 = _TC(index=0, id="call_1", name="get_weather", arguments='{"loc":"SF"}')
        stream = _stream(
            _chunk(content=None, reasoning_content=None, tool_calls=[tc1]),
        )
        mock_client.chat.completions.create = AsyncMock(return_value=stream)

        results: list[StreamDelta] = []
        async for delta in protocol.stream_invoke(
            model_id="gpt-4", messages=[UserMessage(content="weather?")]
        ):
            results.append(delta)

        assert results[0].tool_calls == [
            ToolCallDelta(
                index=0,
                id="call_1",
                name="get_weather",
                arguments='{"loc":"SF"}',
            ),
        ]

    async def test_stream_invoke_passes_params(self, protocol, mock_client):
        stream = _stream(_chunk(content="ok", reasoning_content=None))
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        async for _ in protocol.stream_invoke(
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

    async def test_stream_invoke_uses_default_params(self, protocol, mock_client):
        stream = _stream(_chunk(content="ok", reasoning_content=None))
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create

        async for _ in protocol.stream_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
        ):
            pass

        mock_create.assert_awaited_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["temperature"] is None
        assert kwargs["max_tokens"] is None
        assert kwargs["top_p"] is None
        assert kwargs["stream"] is True

    async def test_stream_invoke_passes_tools(self, protocol, mock_client):
        from unittest.mock import MagicMock

        from yapa.tools.base import Tool

        stream = _stream(_chunk(content="ok", reasoning_content=None))
        mock_create = AsyncMock(return_value=stream)
        mock_client.chat.completions.create = mock_create
        tool = MagicMock(spec=Tool)
        tool.name = "get_weather"
        tool.description = "Get weather"
        tool.parameters = {"type": "object", "properties": {}}

        async for _ in protocol.stream_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            tools=[tool],
        ):
            pass

        mock_create.assert_awaited_once()
        kwargs = mock_create.call_args.kwargs
        assert "tools" in kwargs
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    # ── static_invoke tests ──────────────────────────────────────────

    def _make_response(self, content="Hello", reasoning_content=None, tool_calls=None):
        """Build a mock OpenAI non-streaming response."""
        message = SimpleNamespace(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    async def test_static_invoke_content(self, protocol, mock_client):
        response = self._make_response(content="Hello world")
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await protocol.static_invoke(
            model_id="gpt-4", messages=[UserMessage(content="hi")]
        )

        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello world"
        assert result.role == "assistant"

    async def test_static_invoke_reasoning_content(self, protocol, mock_client):
        response = self._make_response(
            content="answer", reasoning_content="thinking..."
        )
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await protocol.static_invoke(
            model_id="gpt-4", messages=[UserMessage(content="hi")]
        )

        assert result.reasoning_content == "thinking..."

    async def test_static_invoke_reasoning_fallback(self, protocol, mock_client):
        message = SimpleNamespace(
            content="answer",
            reasoning="thinking...",
            tool_calls=None,
        )
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await protocol.static_invoke(
            model_id="gpt-4", messages=[UserMessage(content="hi")]
        )

        assert result.reasoning_content == "thinking..."

    async def test_static_invoke_tool_calls(self, protocol, mock_client):
        tc1 = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"loc":"SF"}',
            ),
        )
        response = self._make_response(content=None, tool_calls=[tc1])
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await protocol.static_invoke(
            model_id="gpt-4", messages=[UserMessage(content="weather?")]
        )

        assert result.tool_calls == [
            ToolCall(id="call_1", tool_name="get_weather", arguments={"loc": "SF"}),
        ]
        assert result.content is None

    async def test_static_invoke_passes_params(self, protocol, mock_client):
        response = self._make_response(content="ok")
        mock_create = AsyncMock(return_value=response)
        mock_client.chat.completions.create = mock_create
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        await protocol.static_invoke(
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

    async def test_static_invoke_uses_default_params(self, protocol, mock_client):
        response = self._make_response(content="ok")
        mock_create = AsyncMock(return_value=response)
        mock_client.chat.completions.create = mock_create

        await protocol.static_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
        )

        mock_create.assert_awaited_once()
        kwargs = mock_create.call_args.kwargs
        assert kwargs["temperature"] is None
        assert kwargs["max_tokens"] is None
        assert kwargs["top_p"] is None
        assert kwargs["stream"] is False

    async def test_static_invoke_passes_tools(self, protocol, mock_client):
        from unittest.mock import MagicMock

        from yapa.tools.base import Tool

        response = self._make_response(content="ok")
        mock_create = AsyncMock(return_value=response)
        mock_client.chat.completions.create = mock_create
        tool = MagicMock(spec=Tool)
        tool.name = "get_weather"
        tool.description = "Get weather"
        tool.parameters = {"type": "object", "properties": {}}

        await protocol.static_invoke(
            model_id="gpt-4",
            messages=[UserMessage(content="hi")],
            tools=[tool],
        )

        mock_create.assert_awaited_once()
        kwargs = mock_create.call_args.kwargs
        assert "tools" in kwargs
        assert kwargs["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def test_static_invoke_none_content(self, protocol, mock_client):
        response = self._make_response(content=None)
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await protocol.static_invoke(
            model_id="gpt-4", messages=[UserMessage(content="hi")]
        )

        assert result.content is None
