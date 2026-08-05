"""Tests for ChatService."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    LanguageModel,
    ReasoningEffort,
    TokenUsage,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    ReasoningEvent,
    TextEvent,
    ToolCallEvent,
)
from yapa.models.stream import (
    ContentDelta,
    ReasoningDelta,
    StreamEndEvent,
)
from yapa.services.chat import ChatService
from yapa.services.models import ModelService
from yapa.services.session import SessionService
from yapa.services.store import JsonSessionStore
from yapa.tools.registry import ToolRegistry


@pytest.fixture
def models(tmp_path):
    svc = MagicMock(spec=ModelService)
    provider = MagicMock()
    svc.get_provider_by_model.return_value = provider
    svc.get_provider = MagicMock(return_value=provider)
    return svc


@pytest.fixture
def sessions(tmp_path):
    store = JsonSessionStore(storage_dir=tmp_path)
    svc = SessionService(store=store)
    return svc


@pytest.fixture
def chat(models, sessions):
    return ChatService(sessions=sessions, models=models, tools=ToolRegistry())


class TestStream:
    async def test_agent_start_then_text_then_done(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ContentDelta(content="Hello")
            yield ContentDelta(content=" world")
            yield StreamEndEvent(
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        assert len(events) == 4
        assert isinstance(events[0], AgentStartEvent)
        assert events[0].model_id == "openai:gpt-4"
        assert isinstance(events[1], TextEvent)
        assert events[1].content == "Hello"
        assert isinstance(events[2], TextEvent)
        assert events[2].content == " world"
        assert isinstance(events[3], AgentDoneEvent)
        assert events[3].content == "Hello world"
        assert events[3].finish_reason == "stop"
        assert events[3].usage is not None
        assert events[3].usage.total_tokens == 8

    async def test_persists_user_and_assistant_messages(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ContentDelta(content="Hello")
            yield ContentDelta(content=" world")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            pass

        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 2
        assert isinstance(loaded.messages[0], UserMessage)
        assert loaded.messages[0].content == "Hi"
        assert isinstance(loaded.messages[1], AssistantMessage)
        assert loaded.messages[1].content == "Hello world"
        assert loaded.messages[1].model == "openai:gpt-4"

    async def test_includes_reasoning_content(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ReasoningDelta(content="thinking...")
            yield ContentDelta(content="Answer")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        assert isinstance(events[1], ReasoningEvent)
        assert events[1].content == "thinking..."
        assert isinstance(events[2], TextEvent)
        assert events[2].content == "Answer"
        assert isinstance(events[3], AgentDoneEvent)
        assert events[3].content == "Answer"

    async def test_includes_reasoning_on_assistant_message(
        self, chat, sessions, models
    ):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ReasoningDelta(content="thinking...")
            yield ContentDelta(content="Answer")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            pass

        loaded = sessions.get(str(session.id))
        assistant = loaded.messages[1]
        assert isinstance(assistant, AssistantMessage)
        assert assistant.reasoning_content == "thinking..."

    async def test_forwards_reasoning_to_provider(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        captured = {}

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            captured["reasoning"] = reasoning
            yield ContentDelta(content="OK")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai", supports_reasoning=True)

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
            reasoning=ReasoningEffort.HIGH,
        ):
            pass

        assert captured["reasoning"] == ReasoningEffort.HIGH

    async def test_reasoning_rejected_for_unsupported_model(
        self, chat, sessions, models
    ):
        session = sessions.create()
        model = LanguageModel(
            id="plain", provider_id="ollama", supports_reasoning=False
        )

        with pytest.raises(ValueError, match="does not support reasoning"):
            async for _ in chat.stream(
                session_id=session.id,
                prompt="Hi",
                model=model,
                reasoning=ReasoningEffort.HIGH,
            ):
                pass

    async def test_reasoning_off_works_for_unsupported_model(
        self, chat, sessions, models
    ):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ContentDelta(content="OK")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(
            id="plain", provider_id="ollama", supports_reasoning=False
        )

        events = [
            ev
            async for ev in chat.stream(
                session_id=session.id,
                prompt="Hi",
                model=model,
                reasoning=ReasoningEffort.OFF,
            )
        ]
        assert any(isinstance(e, TextEvent) for e in events)

    async def test_uses_session_system_prompt(self, chat, sessions, models):
        session = sessions.create()
        sessions.update_system_prompt(str(session.id), "Be concise.")
        provider = models.get_provider_by_model.return_value
        captured_kwargs = {}

        async def _capture(model, messages, tools=None, params=None, reasoning=None):
            captured_kwargs["messages"] = messages
            yield ContentDelta(content="OK")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _capture
        model = LanguageModel(id="gpt-4", provider_id="openai")

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            pass

        messages = captured_kwargs["messages"]
        # System prompt should be prepended to conversation history
        assert any(
            hasattr(m, "role") and m.role == "system" and m.content == "Be concise."
            for m in messages
        )

    async def test_uses_session_inference_params(self, chat, sessions, models):
        session = sessions.create()
        params = InferenceParams(temperature=0.3, max_tokens=100)
        sessions.update_inference_params(str(session.id), params)
        provider = models.get_provider_by_model.return_value
        captured_kwargs = {}

        async def _capture(model, messages, tools=None, params=None, reasoning=None):
            captured_kwargs["params"] = params
            yield ContentDelta(content="OK")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _capture
        model = LanguageModel(id="gpt-4", provider_id="openai")

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            pass

        assert captured_kwargs["params"] == params

    async def test_raises_error_when_no_model(self, chat, sessions):
        session = sessions.create()
        with pytest.raises(ValueError, match="No model specified"):
            async for _ in chat.stream(
                session_id=session.id,
                prompt="Hi",
            ):
                pass

    async def test_uses_session_model_as_fallback(self, chat, sessions, models):
        session = sessions.create()
        # Manually set model on session (simulating previous stream)
        session.model = LanguageModel(id="gpt-4", provider_id="openai")
        sessions._store.save(session, overwrite=True)
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ContentDelta(content="OK")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
        ):
            pass

        models.get_provider_by_model.assert_called_with(
            LanguageModel(id="gpt-4", provider_id="openai")
        )

    async def test_saves_model_to_session_after_stream(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ContentDelta(content="Hi")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hello",
            model=model,
        ):
            pass

        loaded = sessions.get(str(session.id))
        assert loaded.model is not None
        assert loaded.model.id == "gpt-4"
        assert loaded.model.provider_id == "openai"

    async def test_agent_error_on_model_failure(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        provider.stream_chat.side_effect = Exception("API failure")
        model = LanguageModel(id="gpt-4", provider_id="openai")

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], AgentStartEvent)
        assert isinstance(events[1], AgentErrorEvent)
        assert "API failure" in events[1].message

        # Messages should NOT be persisted on error
        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 0

    async def test_stream_is_stateless(self, chat, sessions, models):
        """Two consecutive stream calls should be independent."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield ContentDelta(content="Resp")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        async for _ in chat.stream(session_id=session.id, prompt="First", model=model):
            pass

        async for _ in chat.stream(session_id=session.id, prompt="Second", model=model):
            pass

        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 4  # 2 user + 2 assistant

    async def test_agent_error_on_empty_response(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            yield StreamEndEvent()

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], AgentStartEvent)
        assert isinstance(events[1], AgentErrorEvent)
        assert "empty response" in events[1].message

        # No messages should be persisted
        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 0

    async def test_raises_on_missing_session(self, chat, models):
        with pytest.raises(ValueError, match="not found"):
            async for _ in chat.stream(
                session_id=uuid4(),
                prompt="Hi",
                model=LanguageModel(id="gpt-4", provider_id="openai"),
            ):
                pass

    async def test_normalizes_malformed_tool_call_arguments(
        self, chat, sessions, models
    ):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None, reasoning=None):
            from yapa.models.stream import ToolCallDeltaEvent

            yield ToolCallDeltaEvent(
                index=0, id="call_1", name="calc", arguments='{"a":'
            )
            yield StreamEndEvent(finish_reason="tool_calls")
            # Second turn: no tool calls -> done
            yield ContentDelta(content="Done")
            yield StreamEndEvent(finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = LanguageModel(id="gpt-4", provider_id="openai")

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        # Find the ToolCallEvents and assert their arguments normalized to {}
        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert tool_call_events, "expected at least one ToolCallEvent"
        assert all(tc.arguments == {} for tc in tool_call_events)
        assert tool_call_events[0].call_id == "call_1"
