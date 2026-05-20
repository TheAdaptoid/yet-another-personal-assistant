"""Tests for the chat service."""

import logging
from unittest.mock import AsyncMock, create_autospec

import pytest

from yapa.core.inference.exceptions import InferenceError, ModelNotFoundError
from yapa.core.services import ChatService, InferenceService, SessionService
from yapa.shared.models import (
    AssistantMessage,
    ChatResponse,
    ModelData,
    Session,
    UserMessage,
)

# --- Fixtures ------------------------------------------------------------------


@pytest.fixture
def dummy_logger():
    """Return a mock logger with the methods used in the service."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.NullHandler()
        logger.addHandler(handler)
    return logger


@pytest.fixture
def mock_session_service():
    """Mock SessionService with AsyncMock for every public method."""
    service = create_autospec(SessionService, instance=True)
    service.get_session = AsyncMock()
    service.append_message = AsyncMock()
    return service


@pytest.fixture
def mock_inference_service():
    """Mock InferenceService with AsyncMock for public methods."""
    service = create_autospec(InferenceService, instance=True)
    service.invoke_model = AsyncMock()
    service.get_models = AsyncMock()
    return service


@pytest.fixture
def sample_model() -> ModelData:
    """A minimal ModelData instance for tests."""
    return ModelData(
        id="test-model",
        name="Test Model",
        provider_id="test-provider",
        provider_name="Test Provider",
    )


@pytest.fixture
def sample_session() -> Session:
    """Pre-built Session for reuse in happy-path tests."""
    return Session(title="test")


@pytest.fixture
def chat_service(dummy_logger, mock_session_service, mock_inference_service):
    """ChatService backed by mocked dependencies."""
    return ChatService(
        logger=dummy_logger,
        session_service=mock_session_service,
        inference_service=mock_inference_service,
    )


# --- process_message (normal flow) ---------------------------------------------


class TestProcessMessageNormal:
    """Happy-path: user sends message, inference succeeds, response is saved."""

    @pytest.mark.asyncio
    async def test_success(
        self, chat_service, mock_session_service, mock_inference_service,
        sample_session, sample_model,
    ):
        """A full round-trip returns a successful ChatResponse."""
        user_msg = UserMessage(content="hello")
        assistant_msg = AssistantMessage(content="hi there", model=sample_model.id)

        mock_session_service.append_message = AsyncMock(
            side_effect=[sample_session.add_message(user_msg), sample_session]
        )
        mock_inference_service.invoke_model.return_value = assistant_msg

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
        )

        assert result.response == "hi there"
        assert result.done is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_session_not_found(
        self, chat_service, mock_session_service, sample_model,
    ):
        """If the session does not exist, an error ChatResponse is returned."""
        mock_session_service.append_message.return_value = None

        result = await chat_service.process_message(
            session_id="missing",
            model=sample_model,
            message="hello",
        )

        assert result.error is not None
        assert "not found" in result.error
        assert result.done is True

    @pytest.mark.asyncio
    async def test_session_disappears_after_inference(
        self, chat_service, mock_session_service, mock_inference_service,
        sample_session, sample_model,
    ):
        """If the session vanishes between inference and save, error is set."""
        user_msg = UserMessage(content="hello")
        assistant_msg = AssistantMessage(content="hi", model=sample_model.id)

        mock_session_service.append_message = AsyncMock(
            side_effect=[sample_session.add_message(user_msg), None]
        )
        mock_inference_service.invoke_model.return_value = assistant_msg

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
        )

        assert result.response == "hi"
        assert result.error is not None
        assert "could not be saved" in result.error
        assert result.done is True


# --- process_message (inference errors) ----------------------------------------


class TestProcessMessageInferenceErrors:
    """Inference failures are mapped to ChatResponse.error."""

    @pytest.mark.asyncio
    async def test_model_not_found(
        self, chat_service, mock_session_service, mock_inference_service,
        sample_session, sample_model,
    ):
        """ModelNotFoundError yields an error ChatResponse."""
        user_msg = UserMessage(content="hello")
        mock_session_service.append_message.return_value = (
            sample_session.add_message(user_msg)
        )
        mock_inference_service.invoke_model.side_effect = ModelNotFoundError(
            sample_model.id
        )

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
        )

        assert result.error is not None
        assert "not available" in result.error
        assert result.done is True

    @pytest.mark.asyncio
    async def test_inference_error(
        self, chat_service, mock_session_service, mock_inference_service,
        sample_session, sample_model,
    ):
        """InferenceError yields an error ChatResponse."""
        user_msg = UserMessage(content="hello")
        mock_session_service.append_message.return_value = (
            sample_session.add_message(user_msg)
        )
        mock_inference_service.invoke_model.side_effect = InferenceError(
            "API timeout", model_id=sample_model.id
        )

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
        )

        assert result.error is not None
        assert "failed" in result.error
        assert result.done is True

    @pytest.mark.asyncio
    async def test_unexpected_error(
        self, chat_service, mock_session_service, mock_inference_service,
        sample_session, sample_model,
    ):
        """An unexpected exception yields a generic error ChatResponse."""
        user_msg = UserMessage(content="hello")
        mock_session_service.append_message.return_value = (
            sample_session.add_message(user_msg)
        )
        mock_inference_service.invoke_model.side_effect = RuntimeError("boom")

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
        )

        assert result.error is not None
        assert "unexpected" in result.error.lower()
        assert result.done is True


# --- process_message (retry flow) ----------------------------------------------


class TestProcessMessageRetry:
    """Retry re-uses the existing session and skips appending the user message."""

    @pytest.mark.asyncio
    async def test_retry_success(
        self, chat_service, mock_session_service, mock_inference_service,
        sample_session, sample_model,
    ):
        """On retry, the user message is not re-appended and inference re-runs."""
        user_msg = UserMessage(content="hello")
        session_with_msg = sample_session.add_message(user_msg)
        assistant_msg = AssistantMessage(content="retry ok", model=sample_model.id)

        mock_session_service.get_session.return_value = session_with_msg
        mock_inference_service.invoke_model.return_value = assistant_msg
        mock_session_service.append_message.return_value = session_with_msg

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
            retry=True,
        )

        assert result.response == "retry ok"
        assert result.error is None
        # append_message should have been called only for the assistant response
        mock_session_service.append_message.assert_awaited_once_with(
            sample_session.id, assistant_msg
        )

    @pytest.mark.asyncio
    async def test_retry_session_not_found(
        self, chat_service, mock_session_service, sample_model,
    ):
        """Retry on a missing session returns an error."""
        mock_session_service.get_session.return_value = None

        result = await chat_service.process_message(
            session_id="missing",
            model=sample_model,
            message="hello",
            retry=True,
        )

        assert result.error is not None
        assert "not found for retry" in result.error
        assert result.done is True

    @pytest.mark.asyncio
    async def test_retry_last_message_not_user(
        self, chat_service, mock_session_service, sample_session, sample_model,
    ):
        """Retry fails if the last message is not a UserMessage."""
        session_with_assistant = sample_session.add_message(
            AssistantMessage(content="previous", model=sample_model.id)
        )
        mock_session_service.get_session.return_value = session_with_assistant

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
            retry=True,
        )

        assert result.error is not None
        assert "not a user message" in result.error
        assert result.done is True
        mock_session_service.append_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_empty_session(
        self, chat_service, mock_session_service, sample_session, sample_model,
    ):
        """Retry on an empty session fails."""
        mock_session_service.get_session.return_value = sample_session

        result = await chat_service.process_message(
            session_id=sample_session.id,
            model=sample_model,
            message="hello",
            retry=True,
        )

        assert result.error is not None
        assert "not a user message" in result.error
        assert result.done is True
