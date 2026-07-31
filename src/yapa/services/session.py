"""Session management service — CRUD and message appending."""

from typing import List

from yapa.models import InferenceParams, Message, ModelData, Session
from yapa.services.store import SessionStore


class SessionService:
    """CRUD for sessions + message appending."""

    def __init__(self, store: SessionStore) -> None:
        """Initialize the session service."""
        self._store = store

    def create(self) -> Session:
        """Create and persist a new session."""
        session = Session()
        self._store.save(session)
        return session

    def count(self) -> int:
        """Return the total number of sessions."""
        return self._store.count()

    def list(self, *, newest_first: bool = True) -> List[Session]:
        """List all sessions, newest first by default."""
        sessions = self._store.list()
        sessions.sort(key=lambda s: s.updated_at, reverse=newest_first)
        return sessions

    def get(self, session_id: str) -> Session:
        """Retrieve a session by ID."""
        try:
            return self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e

    def rename(self, session_id: str, title: str) -> Session:
        """Rename a session."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.title = title
        self._store.save(session, overwrite=True)
        return session

    def update_system_prompt(self, session_id: str, prompt: str | None) -> Session:
        """Set or clear the session's system prompt."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.system_prompt = prompt
        self._store.save(session, overwrite=True)
        return session

    def update_inference_params(
        self, session_id: str, params: InferenceParams | None
    ) -> Session:
        """Set or clear the session's inference parameters."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.inference_params = params
        self._store.save(session, overwrite=True)
        return session

    def add_messages(
        self,
        session_id: str,
        messages: List[Message],
        model: ModelData | None = None,
    ) -> Session:
        """Append messages to a session, optionally update model, and persist."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.messages.extend(messages)
        if model is not None:
            session.model = model
        session.touch()
        self._store.save(session, overwrite=True)
        return session

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        if not self._store.exists(session_id):
            raise ValueError(f"Session '{session_id}' not found")
        self._store.delete(session_id)
