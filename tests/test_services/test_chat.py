"""Tests for ChatService."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
    TokenUsage,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    ReasoningEvent,
    TextEvent,
)
from yapa.services.chat import ChatService
from yapa.services.models import ModelService
from yapa.services.session import SessionService
from yapa.services.store import JsonSessionStore


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
    return ChatService(sessions=sessions, models=models)


class TestStream:
    async def test_agent_start_then_text_then_done(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Hello")
            yield StreamDelta(
                content=" world",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

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

    async def test_persists_user_and_assistant_messages(
        self, chat, sessions, models
    ):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Hello")
            yield StreamDelta(
                content=" world",
                finish_reason="stop",
            )

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

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

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(reasoning_content="thinking...")
            yield StreamDelta(content="Answer", finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

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

    async def test_uses_session_system_prompt(self, chat, sessions, models):
        session = sessions.create()
        sessions.update_system_prompt(str(session.id), "Be concise.")
        provider = models.get_provider_by_model.return_value
        captured_kwargs = {}

        async def _capture(model, messages, tools=None, params=None):
            captured_kwargs["messages"] = messages
            yield StreamDelta(content="OK", finish_reason="stop")

        provider.stream_chat.side_effect = _capture
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

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

        async def _capture(model, messages, tools=None, params=None):
            captured_kwargs["params"] = params
            yield StreamDelta(content="OK", finish_reason="stop")

        provider.stream_chat.side_effect = _capture
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

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
        session.model = ModelData(
            id="gpt-4", provider_id="openai", type=ModelType.LLM
        )
        sessions._store.save(session, overwrite=True)
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="OK", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
        ):
            pass

        models.get_provider_by_model.assert_called_with(
            ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        )

    async def test_saves_model_to_session_after_stream(
        self, chat, sessions, models
    ):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Hi", finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

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
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

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

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Resp", finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        async for _ in chat.stream(
            session_id=session.id, prompt="First", model=model
        ):
            pass

        async for _ in chat.stream(
            session_id=session.id, prompt="Second", model=model
        ):
            pass

        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 4  # 2 user + 2 assistant

    async def test_raises_on_missing_session(self, chat, models):
        with pytest.raises(ValueError, match="not found"):
            async for _ in chat.stream(
                session_id=uuid4(),
                prompt="Hi",
                model=ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM),
            ):
                pass
