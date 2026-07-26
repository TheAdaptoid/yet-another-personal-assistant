"""Session related models."""

from pydantic import Field

from .base import TrackedEntity
from .inference import ModelData
from .message import Message

DEFAULT_SESSION_TITLE = "Untitled Session"


class Session(TrackedEntity):
    """
    A session is a collection of messages between a user and an AI.

    Attributes:
        title (str): The title of the session.
        model (ModelData | None): The model used for this session.
        messages (list[Message]): The list of messages in the session.
    """

    title: str = Field(default=DEFAULT_SESSION_TITLE)
    model: ModelData | None = Field(default=None)
    messages: list[Message] = Field(default_factory=list)
