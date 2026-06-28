"""Conversation service — UI-agnostic chat orchestration."""

from collections.abc import AsyncGenerator
from typing import Self
from uuid import UUID

from yapa.config import Config, get_config
from yapa.models import (
    AssistantMessage,
    Message,
    ModelData,
    Session,
    StreamDelta,
    SystemMessage,
    UserMessage,
)
from yapa.providers.exceptions import ModelInvocationError
from yapa.storage import GenericStore

from .exceptions import ConversationError
from .provider import ProviderService

TITLE_SYSTEM_PROMPT = (
    "Generate a concise title (max 5 words) for this conversation "
    "based on the user's first message. Respond with ONLY the title, "
    "no punctuation or quotes."
)


class ConversationService:
    """Orchestrates a conversation session, agnostic of UI framework."""

    def __init__(
        self,
        provider_service: ProviderService | None = None,
        config: Config | None = None,
        store: GenericStore[Session] | None = None,
    ) -> None:
        """
        Initialize a new conversation service.

        Args:
            provider_service: Provider service instance. Defaults to fresh
                ProviderService.
            config: Application config. Defaults to get_config().
            store: Session store. Defaults to a GenericStore[Session] at
                {storage_dir}/sessions.
        """
        self._ps = provider_service or ProviderService()
        self._cfg = config or get_config()
        self._store = store or GenericStore[Session](
            storage_dir=self._cfg.storage_dir / "sessions",
            entity_type=Session,
        )
        self._session_id: UUID | None = None
        self._model: ModelData | None = None
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        """All messages in the current session (read-only copy)."""
        return list(self._messages)

    @property
    def session_id(self) -> UUID | None:
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

    async def resolve_model(self, model_full_id: str) -> ModelData:
        """
        Resolve a model full ID string to a ModelData with the correct provider.

        Args:
            model_full_id: The full model identifier to resolve.

        Returns:
            ModelData with the correct provider_id.

        Raises:
            ValueError: If no provider serves the given model ID.
        """
        return await self._ps.get_model(model_full_id)

    def _save_message(self, message: Message) -> None:
        """Persist a message to the current session."""
        if not self._session_id:
            raise ConversationError("No active session to save message to")
        self._messages.append(message)
        session = self._store.load(str(self._session_id))
        session.messages.append(message)
        self._store.save(session, overwrite=True)

    async def start(
        self,
        session_id: UUID | None = None,
        model: ModelData | None = None,
    ) -> Session:
        """
        Resolve provider and load or create a session.

        Args:
            session_id: Existing session ID to resume, if provided.
            model: Optional model to use for this session. If not provided,
                the default model from config is used.

        Returns:
            The active Session.
        """
        if session_id:
            session = self._store.load(str(session_id))
            self._messages = list(session.messages)
        else:
            session = Session()
            self._store.save(session)
            self._messages = []

        self._session_id = session.id

        if self._model is None:
            self._model = model or await self._ps.get_model(self._cfg.default_model)

        return session

    def switch_session(self, session_id: UUID) -> Session:
        """
        Switch to a different session without resetting the model.

        Args:
            session_id: ID of the session to switch to.

        Returns:
            Session for the newly active session.

        Raises:
            ValueError: If session_id is not found.
        """
        session = self._store.load(str(session_id))
        self._messages = list(session.messages)
        self._session_id = session.id
        return session

    async def close(self) -> None:
        """Clear in-memory state and mark the service as closed."""
        self._messages.clear()
        self._session_id = None
        self._model = None

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager, closing the service."""
        await self.close()

    async def generate_title(self, user_prompt: str) -> str | None:
        """
        Generate a conversation title from a user message using the LLM.

        Args:
            user_prompt: The user's message to base the title on.

        Returns:
            The generated title string, or None if generation failed.
        """
        if not self._model:
            return None
        provider = self._ps.get_provider_by_model(self._model)
        system_msg = SystemMessage(content=TITLE_SYSTEM_PROMPT)
        user_msg = UserMessage(content=user_prompt)
        buffer = ""
        try:
            async for delta in provider.invoke_llm(
                model=self._model,
                messages=[system_msg, user_msg],
            ):
                if delta.content:
                    buffer += delta.content
        except ModelInvocationError:
            return None
        title = buffer.strip().strip('"').strip("'").strip()
        return title[:60] if title else None

    async def auto_title(self) -> str | None:
        """Auto-title the current session based on the first user message."""
        if not self._session_id:
            return None
        for msg in self._messages:
            if msg.role == "user":
                title = await self.generate_title(msg.content)
                if title:
                    session = self._store.load(str(self._session_id))
                    session.title = title
                    self._store.save(session, overwrite=True)
                return title
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
            async for delta in provider.invoke_llm(
                model=self._model,
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
