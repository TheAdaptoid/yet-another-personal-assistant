"""Session management service — UI-agnostic session CRUD and listing."""

from yapa.database import SessionRepository, SessionTable
from yapa.models import Message, SessionSummary


class SessionService:
    """Service for session CRUD and listing, agnostic of UI framework."""

    def __init__(self, session_repo: SessionRepository | None = None) -> None:
        """
        Initialize a new session service.

        Args:
            session_repo: Session repository. Defaults to a fresh
                SessionRepository using the global database engine.
        """
        self._repo = session_repo or SessionRepository()

    def list_all(self) -> list[SessionSummary]:
        """
        List all sessions ordered by most recently updated.

        Returns:
            list of SessionSummary for every session.
        """
        sessions = self._repo.list_all()
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return [self._to_summary(s) for s in sessions]

    def get(self, session_id: str) -> SessionSummary:
        """
        Get a single session's summary.

        Args:
            session_id: The session ID.

        Returns:
            SessionSummary for the session.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        return self._to_summary(self._repo.get(session_id))

    def get_messages(self, session_id: str) -> list[Message]:
        """
        Get all messages for a session as Pydantic models.

        Args:
            session_id: The session ID.

        Returns:
            list of Message (UserMessage, AssistantMessage, SystemMessage).

        Raises:
            ValueError: If no session with the given ID is found.
        """
        table_messages = self._repo.get_messages(session_id)
        return [m.to_pydantic() for m in table_messages]

    def create(self, title: str | None = None) -> SessionSummary:
        """
        Create a new session.

        Args:
            title: Optional title. Defaults to "New Session".

        Returns:
            SessionSummary for the new session.
        """
        return self._to_summary(self._repo.create(title=title))

    def rename(self, session_id: str, title: str) -> SessionSummary:
        """
        Rename a session.

        Args:
            session_id: The session ID.
            title: New title.

        Returns:
            Updated SessionSummary.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        return self._to_summary(self._repo.rename(session_id, title))

    def delete(self, session_id: str) -> None:
        """
        Delete a session and its messages.

        Args:
            session_id: The session ID.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        self._repo.delete(session_id)

    def purge(self, min_messages: int = 2) -> list[str]:
        """
        Delete all sessions with fewer than *min_messages* messages.

        Args:
            min_messages: Minimum number of messages for a session to
                survive. Defaults to 2.

        Returns:
            list of session IDs that were deleted.
        """
        return self._repo.purge(min_messages=min_messages)

    @staticmethod
    def _to_summary(table: SessionTable) -> SessionSummary:
        """Convert a SessionTable to a SessionSummary."""
        return SessionSummary(
            id=table.id,
            title=table.title,
            created_at=table.created_at,
            updated_at=table.updated_at,
            message_count=len(table.messages) if table.messages else 0,
        )
