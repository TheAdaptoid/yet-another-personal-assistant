"""Data model for a chat session."""

from time import time
from uuid import uuid4

from pydantic import BaseModel, Field

from yapa.shared.models.message import Message

DEFAULT_SESSION_TITLE = "New Session"


class SessionData(BaseModel):
    """
    Details of a chat session, excluding messages.

    Attributes:
        id (str): Unique identifier for the session, generated as a UUID4 hex string.
        title (str): Human-readable title for the session.
        created_at (int): Timestamp of when the session was created (Unix epoch).
        updated_at (int): Timestamp of the last update to the session (Unix epoch).
    """

    id: str = Field(
        ...,
        description=(
            "Unique identifier for the session, generated as a UUID4 hex string."
        ),
    )
    title: str = Field(..., description="Human-readable title for the session")
    created_at: int = Field(
        ...,
        description="Timestamp of when the session was created (Unix epoch)",
    )
    updated_at: int = Field(
        ...,
        description="Timestamp of the last update to the session (Unix epoch)",
    )

    class Config:
        """Make SessionData immutable."""

        frozen = True


class Session(SessionData):
    """
    Details of a chat session, including a series of messages.

    Attributes:
        id (str): Unique identifier for the session, generated as a UUID4 hex string.
        title (str): Human-readable title for the session.
        created_at (int): Timestamp of when the session was created (Unix epoch).
        updated_at (int): Timestamp of the last update to the session (Unix epoch).
        messages (list[Message]): Ordered list of messages in this session. Messages are
            sorted by their timestamp in ascending order (oldest first).
    """

    messages: list[Message] = Field(
        ...,
        description=(
            "Ordered list of messages in this session. Messages are sorted by their "
            "timestamp in ascending order (oldest first)."
        ),
    )

    class Config:
        """Make Session immutable."""

        frozen = True

    @property
    def data(self) -> SessionData:
        """
        Get the session details excluding messages.

        Returns:
            SessionData: A SessionData instance containing the session's metadata.
        """
        return SessionData(
            id=self.id,
            title=self.title,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def create(cls, title: str | None = None) -> "Session":
        """
        Create a new Session with a unique ID and timestamps.

        Args:
            title (str | None): Optional title for the session. If None or empty,
                defaults to "New Session".

        Returns:
            Session: A new Session instance with the specified title and default values.
        """
        if not title or title.strip() == "":
            title = DEFAULT_SESSION_TITLE
        return cls(
            id=uuid4().hex,
            title=title,
            created_at=int(time()),
            updated_at=int(time()),
            messages=[],
        )

    def update_title(self, new_title: str) -> "Session":
        """
        Update the session's title and refresh the updated_at timestamp.

        Args:
            new_title (str): The new title for the session.

        Returns:
            Session: A new Session instance with updated title and timestamp.
        """
        return self.model_copy(update={"title": new_title, "updated_at": int(time())})

    def add_message(self, message: Message) -> "Session":
        """
        Add a new message to the session and update the updated_at timestamp.

        Args:
            message (Message): The message to add to the session.

        Returns:
            Session: A new Session instance with added message and updated timestamp.
        """
        return self.model_copy(
            update={"messages": self.messages + [message], "updated_at": int(time())}
        )
