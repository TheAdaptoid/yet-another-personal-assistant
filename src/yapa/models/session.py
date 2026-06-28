"""Session related models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .base import TrackedEntity
from .message import Message

DEFAULT_TITLE = "Untitled Session"


class SessionSummary(BaseModel):
    """
    Read-only summary of a session for list display.

    Attributes:
        id (str): The unique identifier of the session.
        title (str): The title of the session.
        created_at (datetime): The timestamp when the session was created.
        updated_at (datetime): The timestamp when the session was last updated.
        message_count (int): The number of messages in the session.
    """

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    model_config = ConfigDict(
        frozen=True,
    )


class Session(TrackedEntity):
    """
    A session is a collection of messages between a user and an AI.

    Attributes:
        title (str): The title of the session.
        messages (list[Message]): The list of messages in the session.
    """

    title: str = Field(default=DEFAULT_TITLE)
    messages: list[Message] = Field(default_factory=list)
