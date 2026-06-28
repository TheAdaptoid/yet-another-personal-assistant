"""Session related models."""

from pydantic import Field

from .base import TrackedEntity
from .message import Message

DEFAULT_TITLE = "Untitled Session"


class Session(TrackedEntity):
    """
    A session is a collection of messages between a user and an AI.

    Attributes:
        title (str): The title of the session.
        messages (list[Message]): The list of messages in the session.
    """

    title: str = Field(default=DEFAULT_TITLE)
    messages: list[Message] = Field(default_factory=list)
