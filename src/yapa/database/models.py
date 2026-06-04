"""SQLModel table definitions for YAPA database."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Field, Relationship, SQLModel

from yapa.models.message import Message


class BaseTable(SQLModel):
    """
    Base table with common fields.

    Attributes:
        id (str): Unique identifier for the record, generated as a UUID4 hex string.
        created_at (datetime): Timestamp of when the record was created, in UTC.
        updated_at (datetime): Timestamp of when the record was last updated, in UTC.
    """

    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )


class SessionTable(BaseTable, table=True):
    """
    A conversation session.

    Attributes:
        title (str): Title of the session.
        messages (list[MessageTable]): List of messages in the session.
    """

    __tablename__ = "sessions"

    title: str

    messages: list["MessageTable"] = Relationship(
        back_populates="session",
        cascade_delete=True,
    )

    @classmethod
    def create(cls, title: str | None = None) -> "SessionTable":
        """
        Create a new session with an optional title.

        Args:
            title (str | None): Optional title for the session.
                Defaults to "New Session".

        Returns:
            SessionTable: The created session instance.
        """
        return cls(title=title or "New Session")


class MessageTable(BaseTable, table=True):
    """
    A message within a conversation session.

    Attributes:
        role (str): The role of the message sender ("user", "assistant", "system").
        content (str): The content of the message.
        model (str | None): The model that generated the message. This field is specific
            to messages with role "assistant".
        session_id (str): Foreign key linking to the parent session.
        session (SessionTable | None): Relationship to the parent session.
    """

    __tablename__ = "messages"

    role: str
    content: str
    model: str | None = Field(default=None)

    session_id: str = Field(foreign_key="sessions.id")
    session: SessionTable | None = Relationship(back_populates="messages")

    @classmethod
    def from_pydantic(
        cls,
        message: Message,
        session_id: str,
    ) -> "MessageTable":
        """
        Create a MessageTable from a Pydantic message model.

        Args:
            message: The Pydantic message to convert.
            session_id: The session to associate the message with.

        Returns:
            MessageTable: The created message table instance.
        """
        return cls(
            session_id=session_id,
            role=message.role,
            content=message.content,
            model=getattr(message, "model", None),
        )
