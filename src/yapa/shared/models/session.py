"""Data model for a chat session."""

from time import time
from uuid import uuid4

from pydantic import BaseModel, Field

from yapa.shared.models.message import Message

DEFAULT_SESSION_TITLE = "New Session"


class Session(BaseModel):
    """
    Represents a chat session containing a series of messages.

    Attributes:
        id (str): Unique identifier for the session, generated as a UUID4 hex string.
        title (str): Human-readable title for the session, defaults to "New Session".
        messages (list[Message]): Ordered list of messages in this session.
        created_at (int): Timestamp of when the session was created (Unix epoch).
        updated_at (int): Timestamp of the last update to the session (Unix epoch).
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    title: str = Field(default=DEFAULT_SESSION_TITLE)
    messages: list[Message] = Field(default_factory=list)
    created_at: int = Field(default_factory=lambda: int(time()))
    updated_at: int = Field(default_factory=lambda: int(time()))

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
        return cls(title=title)

    def add_message(self, message: Message) -> None:
        """
        Add a message to the session and update the timestamp.

        Args:
            message (Message): The message to append to the session.
        """
        self.messages.append(message)
        self.updated_at = int(time())
