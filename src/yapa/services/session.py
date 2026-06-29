"""Session management service — UI-agnostic session CRUD and listing."""

from datetime import datetime
from pathlib import Path

from yapa.config import get_config
from yapa.models import Session
from yapa.storage import GenericStore


class SessionService:
    """Service for session CRUD and listing, agnostic of UI framework."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """
        Initialize a new session service.

        Args:
            storage_dir: The directory to store the sessions in. If None, the
                default storage directory is used.
        """
        self._store = GenericStore[Session](
            storage_dir=storage_dir or get_config().storage_dir / "sessions",
            entity_type=Session,
        )

    def create(self, title: str | None = None) -> Session:
        """
        Create a new session.

        Args:
            title: Optional title. Defaults to "New Session".

        Returns:
            The new session.
        """
        session = Session()
        if title:
            session.title = title
        self._store.save(session)
        return session

    def list_sessions(
        self,
        *,
        min_date: datetime | None = None,
        max_date: datetime | None = None,
        newest_first: bool = True,
    ) -> list[Session]:
        """
        List all sessions ordered by most recently updated.

        Newest sessions first.

        Returns:
            A list of all sessions
        """
        sessions = self._store.list()

        if min_date:
            sessions = [s for s in sessions if s.updated_at >= min_date]
        if max_date:
            sessions = [s for s in sessions if s.updated_at <= max_date]

        sessions.sort(key=lambda s: s.updated_at, reverse=newest_first)
        return sessions

    def get_session(self, session_id: str) -> Session:
        """
        Retrieve a session.

        Args:
            session_id: The session ID.

        Returns:
            Session for the session.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        try:
            return self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e

    def rename(self, session_id: str, title: str) -> Session:
        """
        Rename a session.

        Args:
            session_id: The session ID.
            title: New title.

        Returns:
            Updated Session.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.title = title
        self._store.save(session, overwrite=True)
        return session

    def delete(self, session_id: str) -> None:
        """
        Delete a session and its messages.

        Args:
            session_id: The session ID.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        try:
            self._store.delete(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
