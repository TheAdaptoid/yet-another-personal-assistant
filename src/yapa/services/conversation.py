"""Conversation service — UI-agnostic chat orchestration."""

from typing import AsyncGenerator

from yapa.config import Config, get_config
from yapa.database.repositories import SessionRepository
from yapa.models import (
    AssistantMessage,
    Message,
    ModelData,
    SessionSummary,
    StreamDelta,
    UserMessage,
)
from yapa.providers.exceptions import ModelInvocationError

from .exceptions import ConversationError
from .provider import ProviderService


class ConversationService:
    """Orchestrates a conversation session, agnostic of UI framework."""

    def __init__(
        self,
        provider_service: ProviderService | None = None,
        config: Config | None = None,
        session_repo: SessionRepository | None = None,
    ) -> None:
        """
        Initialize a new conversation service.

        Args:
            provider_service: Provider service instance. Defaults to fresh
                ProviderService.
            config: Application config. Defaults to get_config().
            session_repo: Session repository. Defaults to a fresh
                SessionRepository using the global database engine.
        """
        self._ps = provider_service or ProviderService()
        self._cfg = config or get_config()
        self._session_repo = session_repo or SessionRepository()
        self._session_id: str | None = None
        self._model: ModelData | None = None
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        """All messages in the current session (read-only copy)."""
        return list(self._messages)

    @property
    def session_id(self) -> str | None:
        """Current session ID, or None if no session has been started."""
        return self._session_id

    @property
    def model(self) -> ModelData | None:
        """Current model identifier, or None if not yet resolved."""
        return self._model

    @model.setter
    def model(self, model: ModelData) -> None:
        """Set the model to use for this conversation."""
        self._model = model

    def _save_message(self, message: Message) -> None:
        """Persist a message to the current session."""
        if not self._session_id:
            raise ConversationError("No active session to save message to")
        self._messages.append(message)
        self._session_repo.add_message(self._session_id, message)

    def start(
        self,
        session_id: str | None = None,
        model: ModelData | None = None,
    ) -> SessionSummary:
        """
        Resolve provider and load or create a session.

        Args:
            session_id: Existing session ID to resume, if provided.
            model: Optional model to use for this session. If not provided,
                the default model from config is used.

        Returns:
            SessionSummary describing the active session.
        """
        if session_id:
            session = self._session_repo.get(session_id)
            table_messages = self._session_repo.get_messages(session_id)
            self._messages = [m.to_pydantic() for m in table_messages]
        else:
            session = self._session_repo.create()
            self._messages = []

        self._session_id = session.id

        if self._model is None:
            if model:
                self._model = model
            else:
                self._model = ModelData(
                    id=self._cfg.default_model_id,
                    provider_id=self._cfg.default_provider_id,
                )

        return session.to_summary()

    def switch_session(self, session_id: str) -> SessionSummary:
        """
        Switch to a different session without resetting the model.

        Args:
            session_id: ID of the session to switch to.

        Returns:
            SessionSummary for the newly active session.

        Raises:
            ValueError: If session_id is not found.
        """
        session = self._session_repo.get(session_id)
        table_messages = self._session_repo.get_messages(session_id)
        self._messages = [m.to_pydantic() for m in table_messages]
        self._session_id = session.id
        return session.to_summary()

    async def close(self) -> None:
        """Clean up resources."""
        return None

    async def stream_response(
        self, prompt: str, model: ModelData | None = None
    ) -> AsyncGenerator[StreamDelta, None]:
        """
        Send a user message and stream the assistant response.

        Both the user and assistant messages are persisted atomically only
        after the model has responded successfully. If the model returns an
        empty response, both messages are discarded and ConversationError is
        raised.

        Args:
            prompt: The user's message content to send.
            model: Optional model to use for this message. If not provided,
                the conversation's current model is used.

        Yields:
            StreamDelta for each chunk of the assistant response. A final
            delta with done=True is yielded after persistence.

        Raises:
            ConversationError: If the model invocation fails or returns an
                empty response.
        """
        if model:
            self._model = model
        if (not self._session_id) or (not self._model):
            raise ConversationError("Call start() before sending messages")

        user_msg = UserMessage(content=prompt)
        provider = self._ps.get_provider_by_model(self._model)
        buffer = ""

        try:
            async for delta in provider.invoke_model(
                model=self._model.id,
                messages=[*self._messages, user_msg],
            ):
                if delta.content:
                    buffer += delta.content
                if not delta.done:
                    yield delta
        except ModelInvocationError as e:
            raise ConversationError("Model invocation failed") from e

        if not buffer:
            raise ConversationError("Model returned empty response")

        assistant_msg = AssistantMessage(content=buffer, model=self._model.id)
        self._save_message(user_msg)
        self._save_message(assistant_msg)

        yield StreamDelta(content=None, reasoning_content=None, done=True)
