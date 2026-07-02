"""Tests for OpenRouterLLMInferenceProtocol."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.config import Config
from yapa.models import (
    AssistantMessage,
    InferenceParams,
    StreamDelta,
    ToolCall,
    ToolCallDelta,
    ToolMessage,
    UserMessage,
)


def _chunk(content=None, reasoning=None, tool_calls=None):
    """Build a mock OpenRouter streaming chunk."""
    delta = SimpleNamespace(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


async def _stream(*chunks):
    for c in chunks:
        yield c


class _TC:
    """Mock ChatStreamToolCall for testing."""

    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = SimpleNamespace(
            name=name,
            arguments=arguments,
        )


class TestOpenRouterLLMInferenceProtocol:
    """Tests for OpenRouterLLMInferenceProtocol."""

    @pytest.fixture
    def config(self):
        return Config(openrouter_api_key="sk-or-v1-test")

    @pytest.fixture
    def protocol(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        return OpenRouterLLMInferenceProtocol(config=config)

    # ── _format_message tests ──────────────────────────────────────────

    def test_format_user_message(self, protocol):
        msg = UserMessage(content="hello")
        result = protocol._format_message(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_format_user_message_no_content_raises(self, protocol):
        with pytest.raises(ValueError, match="cannot be None"):
            protocol._format_message(UserMessage(content=None))

    def test_format_system_message(self, protocol):
        msg_type = type(
            "SystemMessage", (), {"role": "system", "content": "be helpful"}
        )
        msg = msg_type()
        msg.__class__.__name__ = "SystemMessage"
        result = protocol._format_message(msg)
        assert result == {"role": "system", "content": "be helpful"}

    def test_format_system_message_no_content_raises(self, protocol):
        msg_type = type("SystemMessage", (), {"role": "system", "content": None})
        msg = msg_type()
        msg.__class__.__name__ = "SystemMessage"
        with pytest.raises(ValueError, match="cannot be None"):
            protocol._format_message(msg)

    def test_format_assistant_message(self, protocol):
        msg = AssistantMessage(content="hi")
        result = protocol._format_message(msg)
        assert result == {"role": "assistant", "content": "hi"}

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

    def test_format_tool_message_empty_content(self, protocol):
        msg = ToolMessage(
            content=None, tool_call_id="call_1", tool_name="weather"
        )
        result = protocol._format_message(msg)
        assert result == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "",
        }

    def test_format_unknown_role_raises(self, protocol):
        msg = SimpleNamespace(role="custom_role", content="foo")
        with pytest.raises(ValueError, match="custom_role"):
            protocol._format_message(msg)

    # ── _format_tools tests ────────────────────────────────────────────

    def test_format_tools_none(self, protocol):
        assert protocol._format_tools(None) is None

    def test_format_tools_empty(self, protocol):
        assert protocol._format_tools([]) is None

    def test_format_tools(self, protocol):
        from unittest.mock import MagicMock

        from yapa.tools.base import Tool

        tool = MagicMock(spec=Tool)
        tool.name = "get_weather"
        tool.description = "Get weather for a location"
        tool.parameters = {
            "type": "object",
            "properties": {"loc": {"type": "string"}},
        }
        result = protocol._format_tools([tool])
        assert result is not None
        assert len(result) == 1
        ft = result[0]
        assert ft.type == "function"
        assert ft.function.name == "get_weather"
        assert ft.function.description == "Get weather for a location"
        assert ft.function.parameters == {
            "type": "object",
            "properties": {"loc": {"type": "string"}},
        }

    # ── stream_invoke tests ────────────────────────────────────────────

    async def test_stream_invoke_content(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()
        stream = _stream(
            _chunk(content="Hello", reasoning=None),
            _chunk(content=" world", reasoning=None),
        )
        mock_client.chat.send_async = AsyncMock(return_value=stream)

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            results: list[StreamDelta] = []
            async for delta in protocol.stream_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="hi")],
            ):
                results.append(delta)

        assert results == [
            StreamDelta(
                content="Hello", reasoning_content=None, tool_calls=[], done=False
            ),
            StreamDelta(
                content=" world", reasoning_content=None, tool_calls=[], done=False
            ),
        ]

    async def test_stream_invoke_reasoning(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()
        stream = _stream(
            _chunk(content=None, reasoning="thinking step by step..."),
            _chunk(content="Answer", reasoning=None),
        )
        mock_client.chat.send_async = AsyncMock(return_value=stream)

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            results: list[StreamDelta] = []
            async for delta in protocol.stream_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="solve this")],
            ):
                results.append(delta)

        assert results[0].reasoning_content == "thinking step by step..."
        assert results[1].content == "Answer"

    async def test_stream_invoke_tool_calls(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()

        tc1 = _TC(index=0, id="call_1", name="get_weather", arguments='{"loc":"SF"}')
        tc2 = _TC(index=1, id="call_2", name="get_time", arguments='{"tz":"PST"}')
        stream = _stream(
            _chunk(content=None, tool_calls=[tc1]),
            _chunk(content=None, tool_calls=[tc2]),
        )
        mock_client.chat.send_async = AsyncMock(return_value=stream)

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            results: list[StreamDelta] = []
            async for delta in protocol.stream_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="weather?")],
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
        assert results[1].tool_calls == [
            ToolCallDelta(
                index=1,
                id="call_2",
                name="get_time",
                arguments='{"tz":"PST"}',
            ),
        ]

    async def test_stream_invoke_passes_params(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()
        stream = _stream(_chunk(content="ok"))
        mock_send = AsyncMock(return_value=stream)
        mock_client.chat.send_async = mock_send

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            async for _ in protocol.stream_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="hi")],
                params=InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9),
            ):
                pass

        mock_send.assert_awaited_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is True

    # ── static_invoke tests ────────────────────────────────────────────

    def _make_response(self, content="Hello", reasoning=None, tool_calls=None):
        """Build a mock OpenRouter non-streaming response."""
        message = SimpleNamespace(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])

    async def test_static_invoke_content(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()
        response = self._make_response(content="Hello world")
        mock_client.chat.send_async = AsyncMock(return_value=response)

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            result = await protocol.static_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="hi")],
            )

        assert isinstance(result, AssistantMessage)
        assert result.content == "Hello world"
        assert result.role == "assistant"
        assert result.tool_calls == []

    async def test_static_invoke_reasoning(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()
        response = self._make_response(
            content="Answer", reasoning="thinking step by step..."
        )
        mock_client.chat.send_async = AsyncMock(return_value=response)

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            result = await protocol.static_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="solve this")],
            )

        assert result.reasoning_content == "thinking step by step..."

    async def test_static_invoke_tool_calls(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()

        tc1 = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(
                name="get_weather",
                arguments='{"loc":"SF"}',
            ),
        )
        tc2 = SimpleNamespace(
            id="call_2",
            function=SimpleNamespace(
                name="get_time",
                arguments='{"tz":"PST"}',
            ),
        )
        response = self._make_response(
            content=None,
            tool_calls=[tc1, tc2],
        )
        mock_client.chat.send_async = AsyncMock(return_value=response)

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            result = await protocol.static_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="weather?")],
            )

        assert result.tool_calls == [
            ToolCall(id="call_1", tool_name="get_weather", arguments={"loc": "SF"}),
            ToolCall(id="call_2", tool_name="get_time", arguments={"tz": "PST"}),
        ]
        assert result.content is None

    async def test_static_invoke_passes_params(self, config):
        from yapa.providers.openrouter.protocols import OpenRouterLLMInferenceProtocol

        protocol = OpenRouterLLMInferenceProtocol(config=config)
        mock_client = MagicMock()
        response = self._make_response(content="ok")
        mock_send = AsyncMock(return_value=response)
        mock_client.chat.send_async = mock_send

        with patch(
            "yapa.providers.openrouter.protocols.OpenRouter",
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_client)),
        ):
            await protocol.static_invoke(
                model_id="openai/gpt-4o",
                messages=[UserMessage(content="hi")],
                params=InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9),
            )

        mock_send.assert_awaited_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["model"] == "openai/gpt-4o"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
        assert kwargs["top_p"] == 0.9
        assert kwargs["stream"] is False
