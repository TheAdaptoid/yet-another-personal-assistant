"""Repository for session and message CRUD operations."""

from sqlmodel import asc, desc, select

from yapa.database.engine import get_session
from yapa.database.models import MessageTable, SessionTable
from yapa.models.message import Message


class SessionRepository:
    """CRUD operations for sessions and messages."""

    @staticmethod
    def create(title: str | None = None) -> SessionTable:
        """
        Create a new conversation session.

        Args:
            title (str | None): Optional title for the session.

        Returns:
            SessionTable: The created session object.
        """
        with get_session() as db:
            session = SessionTable.create(title=title)
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    @staticmethod
    def get(session_id: str) -> SessionTable:
        """
        Get a session by ID.

        Args:
            session_id (str): The ID of the session to retrieve.

        Returns:
            SessionTable: The session with the specified ID.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        with get_session() as db:
            session = db.get(SessionTable, session_id)
            if session is None:
                msg = f"Session '{session_id}' not found"
                raise ValueError(msg)
            return session

    @staticmethod
    def list_all() -> list[SessionTable]:
        """
        List all sessions.

        Returns:
            list[SessionTable]: A list of all sessions ordered by creation time
                (newest first).
        """
        with get_session() as db:
            stmt = select(SessionTable).order_by(desc(SessionTable.created_at))
            return list(db.exec(stmt).all())

    @staticmethod
    def rename(session_id: str, title: str) -> SessionTable:
        """Rename a session. Raises ValueError if not found."""
        session = SessionRepository.get(session_id)
        with get_session() as db:
            session.title = title
            db.add(session)
            db.commit()
            db.refresh(session)
            return session

    @staticmethod
    def delete(session_id: str) -> None:
        """
        Delete a session and its messages.

        Args:
            session_id (str): The ID of the session to delete.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        session = SessionRepository.get(session_id)
        with get_session() as db:
            db.delete(session)
            db.commit()

    @staticmethod
    def add_message(
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
        session = SessionRepository.get(session_id)
        with get_session() as db:
            table_msg = MessageTable.from_pydantic(message, session_id=session.id)
            db.add(table_msg)
            db.commit()
            db.refresh(table_msg)
            return table_msg

    @staticmethod
    def get_messages(session_id: str) -> list[MessageTable]:
        """
        Get all messages for a session ordered by creation time (oldest first).

        Args:
            session_id (str): The ID of the session whose messages to retrieve.

        Returns:
            list[MessageTable]: A list of messages for the session.

        Raises:
            ValueError: If no session with the given ID is found.
        """
        session = SessionRepository.get(session_id)
        with get_session() as db:
            stmt = (
                select(MessageTable)
                .where(MessageTable.session_id == session.id)
                .order_by(asc(MessageTable.created_at))
            )
            return list(db.exec(stmt).all())
