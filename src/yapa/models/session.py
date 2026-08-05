"""Session related models."""

from pydantic import Field

from .base import TrackedEntity
from .inference import InferenceParams, LanguageModel
from .message import Message

DEFAULT_SESSION_TITLE = "Untitled Session"


class Session(TrackedEntity):
    """A session is a collection of messages between a user and an AI."""

    title: str = Field(default=DEFAULT_SESSION_TITLE)
    model: LanguageModel | None = Field(default=None)
    system_prompt: str | None = Field(default=None)
    inference_params: InferenceParams | None = Field(default=None)
    messages: list[Message] = Field(default_factory=list)
