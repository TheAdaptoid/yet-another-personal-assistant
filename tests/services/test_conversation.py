"""Tests for ConversationService."""

from unittest.mock import AsyncMock

import pytest

from yapa.models import AssistantMessage, ModelData, StreamDelta, UserMessage
from yapa.providers.exceptions import ModelInvocationError
from yapa.services import ConversationError, ConversationService


class TestStart:
    """Tests for ConversationService.start()."""

    def test_creates_new_session(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        info = service.start()

        assert info.message_count == 0
        assert service.session_id is not None
        assert service.model == ModelData(id="test-default-model", provider_id="test")

    def test_resumes_session(self, mock_provider_service, repo, config):
        existing = repo.create()
        repo.add_message(existing.id, UserMessage(content="hello"))
        repo.add_message(
            existing.id, AssistantMessage(content="hi", model="m")
        )

        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        model = ModelData(id="test-model", provider_id="test")
        service.model = model
        info = service.start(session_id=existing.id)

        assert info.message_count == 2
        assert info.id == existing.id
        assert service.model == model

    def test_uses_default_model_when_not_set(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        service.start()

        assert service.model == ModelData(id="test-default-model", provider_id="test")


class TestStreamResponse:
    """Tests for ConversationService.stream_response()."""

    async def test_streams_response(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        service.model = ModelData(id="test-model", provider_id="test")
        service.start()

        results: list[StreamDelta] = []
        async for delta in service.stream_response("hello"):
            results.append(delta)

        assert len(results) == 2
        assert results[0].content == "Hi!"
        assert results[-1].done is True

    async def test_saves_user_and_assistant_messages(
        self, mock_provider_service, repo, config
    ):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        service.model = ModelData(id="test-model", provider_id="test")
        service.start()

        async for _ in service.stream_response("hello"):
            pass

        assert len(service.messages) == 2
        assert isinstance(service.messages[0], UserMessage)
        assert service.messages[0].content == "hello"
        assert isinstance(service.messages[1], AssistantMessage)
        assert service.messages[1].content == "Hi!"

    async def test_raises_error_before_start(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        service.model = ModelData(id="test-model", provider_id="test")
        with pytest.raises(ConversationError, match="Call start"):
            async for _ in service.stream_response("hello"):
                pass  # pragma: no cover

    async def test_wraps_model_invocation_error(
        self, mock_provider_service, repo, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _fail(model, messages):
            raise ModelInvocationError("API error")
            yield  # pragma: no cover — makes it an async generator

        provider.invoke_model = _fail
        provider.close = AsyncMock()

        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        service.model = ModelData(id="test-model", provider_id="test")
        service.start()

        with pytest.raises(ConversationError, match="Model invocation failed"):
            async for _ in service.stream_response("hello"):
                pass

    async def test_raises_error_on_empty_response(
        self, mock_provider_service, repo, config
    ):
        provider = mock_provider_service.get_provider_by_model.return_value

        async def _empty(model, messages):
            yield StreamDelta(content=None, done=True)

        provider.invoke_model = _empty
        provider.close = AsyncMock()

        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        service.model = ModelData(id="test-model", provider_id="test")
        service.start()

        with pytest.raises(ConversationError, match="empty response"):
            async for _ in service.stream_response("hello"):
                pass

        assert len(service.messages) == 0


class TestSwitchSession:
    """Tests for ConversationService.switch_session()."""

    def test_switches_to_existing_session(self, mock_provider_service, repo, config):
        existing = repo.create()
        repo.add_message(existing.id, UserMessage(content="hello"))
        repo.add_message(
            existing.id, AssistantMessage(content="hi", model="m")
        )

        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        service.model = ModelData(id="test-model", provider_id="test")
        service.start()

        info = service.switch_session(existing.id)
        assert info.id == existing.id
        assert info.message_count == 2
        assert len(service.messages) == 2

    def test_raises_on_missing_session(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        with pytest.raises(ValueError, match="not found"):
            service.switch_session("bad-id")


class TestClose:
    """Tests for ConversationService.close()."""

    async def test_close_does_not_raise(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        await service.close()


class TestMessages:
    """Tests for ConversationService.messages property."""

    def test_returns_copy(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        msgs = service.messages
        msgs.append(UserMessage(content="extra"))
        assert len(service.messages) == 0


class TestModel:
    """Tests for ConversationService.model property and setter."""

    def test_getter_returns_none_initially(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        assert service.model is None

    def test_setter_and_getter(self, mock_provider_service, repo, config):
        service = ConversationService(
            provider_service=mock_provider_service, config=config, session_repo=repo
        )
        model = ModelData(id="custom/model", provider_id="custom")
        service.model = model
        assert service.model == model
