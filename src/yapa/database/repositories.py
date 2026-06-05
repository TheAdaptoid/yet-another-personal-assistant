"""Repository for session and message CRUD operations."""

from sqlalchemy import Engine, func
from sqlmodel import Session, asc, desc, select

from yapa.database.engine import get_session
from yapa.database.models import MessageTable, SessionTable
from yapa.models.message import Message


class SessionRepository:
    """CRUD operations for sessions and messages."""

    def __init__(self, engine: Engine | None = None) -> None:
        """
        Initialize a new session repository.

        Args:
            engine: Optional SQLAlchemy engine. Falls back to get_session().
        """
        self._engine = engine

    def _session(self) -> Session:
        """Return a new DB session for this repository's engine."""
        if self._engine is not None:
            return Session(self._engine)
        return get_session()

    def create(self, title: str | None = None) -> SessionTable:
        """
        Create a new conversation session.

        Args:
            title (str | None): Optional title for the session.

        Returns:
            SessionTable: The created session object.
        """
        with self._session() as db:
            session = SessionTable.create(title=title)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    def get(self, session_id: str) -> SessionTable:
        """
        Get a session by ID.

        Args:
            session_id (str): The ID of the session to retrieve.

        Returns:
            SessionTable: The session with the specified ID.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        with self._session() as db:
            session = db.get(SessionTable, session_id)
            if session is None:
                msg = f"Session '{session_id}' not found"
                raise ValueError(msg)
            return session

    def list_all(self) -> list[SessionTable]:
        """
        List all sessions.

        Returns:
            list[SessionTable]: A list of all sessions ordered by creation time
                (newest first).
        """
        with self._session() as db:
            stmt = select(SessionTable).order_by(desc(SessionTable.created_at))
            return list(db.exec(stmt).all())

    def rename(self, session_id: str, title: str) -> SessionTable:
        """Rename a session. Raises ValueError if not found."""
        session = self.get(session_id)
        with self._session() as db:
            session.title = title
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    def delete(self, session_id: str) -> None:
        """
        Delete a session and its messages.

        Args:
            session_id (str): The ID of the session to delete.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        session = self.get(session_id)
        with self._session() as db:
            db.delete(session)
            db.commit()

    def purge(self, min_messages: int = 2) -> list[str]:
        """
        Delete all sessions with fewer than *min_messages* messages.

        Uses a single grouped query to identify candidates, then ORM-deletes
        them (respecting cascade_delete on the relationship).

        Args:
            min_messages: Minimum number of messages for a session to survive.

        Returns:
            list of deleted session IDs.
        """
        with self._session() as db:
            subq = (
                select(SessionTable.id)
                .outerjoin(
                    MessageTable,
                    MessageTable.session_id == SessionTable.id,  # type: ignore
                )
                .group_by(SessionTable.id)
                .having(func.count(MessageTable.id) < min_messages)  # type: ignore
            ).subquery()
            stmt = select(SessionTable).where(
                SessionTable.id.in_(select(subq.c.id)),  # type: ignore
            )
            sessions = list(db.exec(stmt).all())
            ids = [s.id for s in sessions]
            for s in sessions:
                db.delete(s)
            db.commit()
            return ids

    def add_message(
        self,
        session_id: str,
        message: Message,
    ) -> MessageTable:
        """
        Add a message to a session. Raises ValueError if session not found.

        Args:
            session_id (str): The ID of the session to which the message will be added.
            message (Message): The message to add to the session.

        Returns:
            MessageTable: The created message table entry.
        """
        session = self.get(session_id)
        with self._session() as db:
            table_msg = MessageTable.from_pydantic(message, session_id=session.id)
            db.add(table_msg)
            db.commit()
            db.refresh(table_msg)
            return table_msg

    def get_messages(self, session_id: str) -> list[MessageTable]:
        """
        Get all messages for a session ordered by creation time (oldest first).

        Args:
            session_id (str): The ID of the session whose messages to retrieve.

        Returns:
            list[MessageTable]: A list of messages for the session.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        session = self.get(session_id)
        with self._session() as db:
            stmt = (
                select(MessageTable)
                .where(MessageTable.session_id == session.id)
                .order_by(asc(MessageTable.created_at))
            )
            return list(db.exec(stmt).all())
