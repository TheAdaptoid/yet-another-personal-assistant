"""Tests for ChatService agentic loop."""

from unittest.mock import MagicMock

import pytest

from yapa.models import (
    ModelData,
    ModelType,
    StreamDelta,
    TokenUsage,
    ToolCallDelta,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse
from yapa.services.chat import ChatService
from yapa.services.models import ModelService
from yapa.services.session import SessionService
from yapa.services.store import JsonSessionStore
from yapa.tools.base import Tool
from yapa.tools.registry import ToolRegistry


class ToolThatReturns(Tool):
    """Tool that returns a fixed result."""

    def __init__(self, name="echo", result="tool_result", needs_approval=False):
        super().__init__(
            name=name,
            description="Echo tool",
            parameters={
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
            needs_approval=needs_approval,
        )
        self._result = result

    async def execute(self, input: str = "", **kwargs):
        return self._result


class ToolThatRaises(Tool):
    def __init__(self):
        super().__init__(
            name="failing",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            needs_approval=False,
        )

    async def execute(self, **kwargs):
        msg = "internal failure"
        raise RuntimeError(msg)


@pytest.fixture
def models(tmp_path):
    svc = MagicMock(spec=ModelService)
    provider = MagicMock()
    svc.get_provider_by_model.return_value = provider
    return svc


@pytest.fixture
def sessions(tmp_path):
    store = JsonSessionStore(storage_dir=tmp_path)
    return SessionService(store=store)


@pytest.fixture
def registry():
    return ToolRegistry([ToolThatReturns()])


@pytest.fixture
def chat(models, sessions, registry):
    return ChatService(sessions=sessions, models=models, tools=registry)


model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)


class TestToolLoop:
    async def test_text_only_no_loop(self, chat, sessions, models):
        """When model returns text, no tool loop."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(
                content="Hello",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert isinstance(events[-1], AgentDoneEvent)
        assert events[-1].content == "Hello"
        # Only one provider call
        assert provider.stream_chat.call_count == 1

    async def test_single_tool_call_then_text(self, chat, sessions, models):
        """Model calls tool, tool executes, then model responds with text."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0,
                            id="call_1",
                            name="echo",
                            arguments='{"input": "hi"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert isinstance(events[-1], AgentDoneEvent)
        assert events[-1].content == "Done"
        assert provider.stream_chat.call_count == 2

    async def test_tool_call_event_yielded(self, chat, sessions, models):
        """ToolCallEvent is yielded for each tool call."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0,
                            id="call_1",
                            name="echo",
                            arguments='{"input": "hi"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].tool_name == "echo"

    async def test_tool_result_event_yielded(self, chat, sessions, models):
        """ToolResultEvent is yielded after tool execution."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0,
                            id="call_1",
                            name="echo",
                            arguments='{"input": "hi"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].tool_name == "echo"

    async def test_dangerous_tool_requests_approval(self, chat, sessions, models):
        """Tool with needs_approval=True requests approval and awaits callback."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        # Replace registry with a dangerous tool
        dangerous = ToolThatReturns(name="dangerous", needs_approval=True)
        chat._tools = ToolRegistry([dangerous])

        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0, id="call_1", name="dangerous", arguments="{}"
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        approved = False

        async def get_approval(req: ToolApprovalRequest) -> ToolApprovalResponse:
            nonlocal approved
            approved = True
            return ToolApprovalResponse(call_id=req.call_id, approved=True)

        events = []
        async for e in chat.stream(
            session_id=session.id, prompt="Hi", model=model, get_approval=get_approval
        ):
            events.append(e)

        assert approved
        assert isinstance(events[-1], AgentDoneEvent)

    async def test_denied_tool_feeds_back_to_model(self, chat, sessions, models):
        """Denied tool call feeds rejection as ToolMessage and continues loop."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        dangerous = ToolThatReturns(name="dangerous", needs_approval=True)
        chat._tools = ToolRegistry([dangerous])

        call_count = 0
        prompts_seen = []

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            prompts_seen.append(len(messages))
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0, id="call_1", name="dangerous", arguments="{}"
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Ok", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        async def get_approval(req):
            return ToolApprovalResponse(
                call_id=req.call_id, approved=False, reason="not now"
            )

        events = []
        async for e in chat.stream(
            session_id=session.id, prompt="Hi", model=model, get_approval=get_approval
        ):
            events.append(e)

        assert provider.stream_chat.call_count == 2
        # Second call should have more messages (denied ToolMessage)
        assert prompts_seen[1] > prompts_seen[0]

    async def test_unknown_tool(self, chat, sessions, models):
        """Unknown tool name is fed back as error ToolMessage."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0, id="call_1", name="nonexistent", arguments="{}"
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Ok", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert provider.stream_chat.call_count == 2

    async def test_tool_execution_error(self, chat, sessions, models):
        """Tool that raises is fed back as error ToolMessage."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        chat._tools = ToolRegistry([ToolThatRaises()])

        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0, id="call_1", name="failing", arguments="{}"
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Ok", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert provider.stream_chat.call_count == 2

    async def test_max_iterations(self, chat, sessions, models):
        """Loop terminates after MAX_ITERATIONS if model keeps calling tools."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(
                tool_calls=[
                    ToolCallDelta(
                        index=0, id="call_1", name="echo", arguments='{"input": "x"}'
                    )
                ],
                finish_reason="tool_calls",
            )

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert isinstance(events[-1], AgentErrorEvent)
        assert "max iterations" in events[-1].message.lower()
        assert provider.stream_chat.call_count == ChatService.MAX_ITERATIONS

    async def test_persists_messages_only_once(self, chat, sessions, models):
        """Messages are persisted only when the loop terminates, not per iteration."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[
                        ToolCallDelta(
                            index=0,
                            id="call_1",
                            name="echo",
                            arguments='{"input": "x"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        async for _ in chat.stream(session_id=session.id, prompt="Hi", model=model):
            pass

        loaded = sessions.get(str(session.id))
        # Only one user + one assistant message (from final iteration)
        user_msgs = [m for m in loaded.messages if isinstance(m, UserMessage)]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Hi"
