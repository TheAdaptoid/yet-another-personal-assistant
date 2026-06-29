"""Tests for ConversationService."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from yapa.models import (
    AssistantMessage,
    ModelData,
    ModelType,
    Session,
    StreamDelta,
    SystemMessage,
    UserMessage,
)
from yapa.providers.exceptions import ModelInvocationError
from yapa.services import ConversationError, ConversationService


class TestStart:
    """Tests for ConversationService.start()."""

    async def test_creates_new_session(self, mock_provider_service, store, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        info = await service.start()

        assert len(info.messages) == 0
        assert service.session_id is not None
        assert service.model == ModelData(
            id="test-default-model",
            provider_id="test",
            type=ModelType.LLM,
        )

    async def test_resumes_session(self, mock_provider_service, store, config):
        session = Session()
        session.messages = [
            UserMessage(content="hello"),
            AssistantMessage(content="hi", model="m"),
        ]
        store.save(session)

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        model = ModelData(id="test-model", provider_id="test", type=ModelType.LLM)
        service.model = model
        info = await service.start(session_id=session.id)

        assert len(info.messages) == 2
        assert info.id == session.id
        assert service.model == model

    async def test_uses_default_model_when_not_set(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        await service.start()

        assert service.model == ModelData(
            id="test-default-model",
            provider_id="test",
            type=ModelType.LLM,
        )

    async def test_uses_explicit_model_param(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        model = ModelData(id="explicit-model", provider_id="test", type=ModelType.LLM)
        info = await service.start(model=model)

        assert service.model == model
        assert len(info.messages) == 0


class TestStreamResponse:
    """Tests for ConversationService.stream_response()."""

    async def test_streams_response(self, mock_provider_service, store, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        results: list[StreamDelta] = []
        async for delta in service.stream_response("hello"):
            results.append(delta)

        assert len(results) == 2
        assert results[0].content == "Hi!"
        assert results[-1].done is True

    async def test_saves_user_and_assistant_messages(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        async for _ in service.stream_response("hello"):
            pass

        assert len(service.messages) == 2
        assert isinstance(service.messages[0], UserMessage)
        assert service.messages[0].content == "hello"
        assert isinstance(service.messages[1], AssistantMessage)
        assert service.messages[1].content == "Hi!"

        session = store.load(str(service.session_id))
        assert len(session.messages) == 2

    async def test_raises_error_before_start(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        with pytest.raises(ConversationError, match="Call start"):
            async for _ in service.stream_response("hello"):
                pass  # pragma: no cover

    async def test_wraps_model_invocation_error(
        self, mock_provider_service, store, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _fail(model, messages, params=None):
            raise ModelInvocationError("API error")
            yield  # pragma: no cover

        provider.invoke_llm_stream = _fail

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        with pytest.raises(ConversationError, match="Model invocation failed"):
            async for _ in service.stream_response("hello"):
                pass

    async def test_raises_error_on_empty_response(
        self, mock_provider_service, store, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _empty(model, messages, params=None):
            yield StreamDelta(content=None, reasoning_content=None, done=True)

        provider.invoke_llm_stream = _empty

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        with pytest.raises(ConversationError, match="empty response"):
            async for _ in service.stream_response("hello"):
                pass

        assert len(service.messages) == 0

    async def test_buffers_multiple_content_chunks(
        self, mock_provider_service, store, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _chunky(model, messages, params=None):
            yield StreamDelta(content="Hello", reasoning_content=None, done=False)
            yield StreamDelta(content=" world", reasoning_content=None, done=False)
            yield StreamDelta(content=None, reasoning_content=None, done=True)

        provider.invoke_llm_stream = _chunky

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        async for _ in service.stream_response("hello"):
            pass

        assert len(service.messages) == 2
        assert service.messages[1].content == "Hello world"

    async def test_accepts_per_call_model_override(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        override = ModelData(
            id="override-model", provider_id="other", type=ModelType.LLM
        )
        async for _ in service.stream_response("hello", model=override):
            pass

        assert service.model == override


class TestResolveModel:
    """Tests for ConversationService.resolve_model()."""

    async def test_returns_model_data(self, mock_provider_service, store, config):
        expected = ModelData(
            id="prov:gpt-4", provider_id="prov", type=ModelType.LLM
        )
        mock_provider_service.get_model = AsyncMock(return_value=expected)

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        result = await service.resolve_model("prov:gpt-4")

        assert result == expected
        mock_provider_service.get_model.assert_called_once_with("prov:gpt-4")

    async def test_raises_on_bad_full_id(self, mock_provider_service, store, config):
        mock_provider_service.get_model.side_effect = ValueError("not found")
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        with pytest.raises(ValueError, match="not found"):
            await service.resolve_model("bad:id")


class TestSwitchSession:
    """Tests for ConversationService.switch_session()."""

    async def test_switches_to_existing_session(
        self, mock_provider_service, store, config
    ):
        session = Session()
        session.messages = [
            UserMessage(content="hello"),
            AssistantMessage(content="hi", model="m"),
        ]
        store.save(session)

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        info = service.switch_session(session.id)
        assert info.id == session.id
        assert len(info.messages) == 2
        assert len(service.messages) == 2

    def test_raises_on_missing_session(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            service.switch_session(uuid4())


class TestClose:
    """Tests for ConversationService.close()."""

    async def test_close_clears_state(self, mock_provider_service, store, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        await service.start()
        assert service.session_id is not None
        assert service.model is not None

        await service.close()

        assert service.session_id is None
        assert service.model is None
        assert service.messages == []

    async def test_close_idempotent(self, mock_provider_service, store, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        await service.start()
        assert service.session_id is not None
        assert service.model is not None

        await service.close()

        assert service.session_id is None
        assert service.model is None
        assert service.messages == []

    async def test_close_idempotent(
        self, mock_provider_service, repo, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        await service.close()
        await service.close()
        await service.close()
        assert service.session_id is None

    async def test_context_manager(self, mock_provider_service, store, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        async with service as svc:
            await svc.start()
            assert svc.session_id is not None
        assert service.session_id is None
        assert service.model is None

    async def test_context_manager_on_exception(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        with pytest.raises(RuntimeError):
            async with service as svc:
                await svc.start()
                raise RuntimeError("test error")
        assert service.session_id is None


class TestMessages:
    """Tests for ConversationService.messages property."""

    def test_returns_copy(self, mock_provider_service, store, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        msgs = service.messages
        msgs.append(UserMessage(content="extra"))
        assert len(service.messages) == 0


class TestModel:
    """Tests for ConversationService.model property and setter."""

    def test_getter_returns_none_initially(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        assert service.model is None

    def test_setter_and_getter(self, mock_provider_service, store, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        model = ModelData(id="custom/model", provider_id="custom", type=ModelType.LLM)
        service.model = model
        assert service.model == model


class TestGenerateTitle:
    """Tests for ConversationService.generate_title()."""

    async def test_generates_title(self, mock_provider_service, store, config):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _response(model, messages, params=None):
            return AssistantMessage(content="My Title", role="assistant")

        provider.invoke_llm = _response

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        title = await service.generate_title("Hello world")
        assert title == "My Title"

    async def test_strips_quotes(self, mock_provider_service, store, config):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _response(model, messages, params=None):
            return AssistantMessage(content='"My Title"', role="assistant")

        provider.invoke_llm = _response

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        title = await service.generate_title("Hello world")
        assert title == "My Title"

    async def test_truncates_long_title(self, mock_provider_service, store, config):
        long_title = "x" * 100
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _response(model, messages, params=None):
            return AssistantMessage(content=long_title, role="assistant")

        provider.invoke_llm = _response

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        title = await service.generate_title("Hello world")
        assert title == "x" * 60

    async def test_returns_none_on_exception(
        self, mock_provider_service, store, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _fail(model, messages, params=None):
            raise ModelInvocationError("API error")

        provider.invoke_llm = _fail

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        title = await service.generate_title("Hello world")
        assert title is None

    async def test_returns_none_when_model_not_set(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        title = await service.generate_title("Hello world")
        assert title is None

    async def test_returns_none_on_empty_response(
        self, mock_provider_service, store, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _empty(model, messages, params=None):
            return AssistantMessage(content="", role="assistant")

        provider.invoke_llm = _empty

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        title = await service.generate_title("Hello world")
        assert title is None

    async def test_passes_system_prompt_with_user_message(
        self, mock_provider_service, store, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value
        captured_messages = None

        async def _capture(model, messages, params=None):
            nonlocal captured_messages
            captured_messages = messages
            return AssistantMessage(content="Title", role="assistant")

        provider.invoke_llm = _capture

        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        await service.generate_title("Hello world")
        assert captured_messages is not None
        assert len(captured_messages) == 2
        assert isinstance(captured_messages[0], SystemMessage)
        assert captured_messages[1].content == "Hello world"
        assert captured_messages[1].role == "user"


class TestAutoTitle:
    """Tests for ConversationService.auto_title()."""

    async def test_auto_titles_first_user_message(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()
        service._messages = [UserMessage(content="Hello world")]

        async def _response(model, messages, params=None):
            return AssistantMessage(content="My Title", role="assistant")

        provider = mock_provider_service.get_provider_by_model.return_value
        provider.invoke_llm = _response

        with patch.object(service._store, "save") as mock_save:
            title = await service.auto_title()

        assert title == "My Title"
        mock_save.assert_called_once()
        saved_session = mock_save.call_args[0][0]
        assert saved_session.title == "My Title"

    async def test_auto_title_no_user_message(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()

        title = await service.auto_title()
        assert title is None

    async def test_auto_title_no_session(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        title = await service.auto_title()
        assert title is None

    async def test_auto_title_skips_non_user_roles(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()
        service._messages = [
            SystemMessage(content="system note"),
            UserMessage(content="Hello world"),
        ]

        async def _response(model, messages, params=None):
            return AssistantMessage(content="My Title", role="assistant")

        provider = mock_provider_service.get_provider_by_model.return_value
        provider.invoke_llm = _response

        with patch.object(service._store, "save") as mock_save:
            title = await service.auto_title()

        assert title == "My Title"
        mock_save.assert_called_once()

    async def test_auto_title_returns_none_on_empty(
        self, mock_provider_service, store, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, store=store
        )
        service.model = ModelData(
            id="test-model", provider_id="test", type=ModelType.LLM
        )
        await service.start()
        service._messages = [UserMessage(content="Hello world")]

        provider = mock_provider_service.get_provider_by_model.return_value

        async def _empty(model, messages, params=None):
            return AssistantMessage(content="", role="assistant")

        provider.invoke_llm = _empty

        with patch.object(service._store, "save") as mock_save:
            title = await service.auto_title()

        assert title is None
        mock_save.assert_not_called()
