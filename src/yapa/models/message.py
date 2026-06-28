"""Data models for messages in the chat application."""

from typing import Annotated, Literal

from pydantic import Field

from .base import TrackedEntity


class BaseMessage(TrackedEntity):
    """
    Base class for all message types.

    Attributes:
        role (Literal["user", "assistant", "system"]): The role of the message sender.
        content (str): The content of the message.
    """

    role: Literal["user", "assistant", "system"]
    content: str


class UserMessage(BaseMessage):
    """
    Represents a message sent by the user.

    Attributes:
        role (Literal["user"]): The role of the message sender, set to "user".
    """

    role: Literal["user"] = "user"


class SystemMessage(BaseMessage):
    """
    Represents a message sent by the system or application.

    Attributes:
        role (Literal["system"]): The role of the message sender, set to "system".
    """

    role: Literal["system"] = "system"


class AssistantMessage(BaseMessage):
    """
    Represents a message sent by the AI assistant.

    Attributes:
        role (Literal["assistant"]): The role of the message sender, set to "assistant".
        model (str | None): The model identifier that generated this response.
    """

    role: Literal["assistant"] = "assistant"
    model: str | None = Field(default=None)


Message = Annotated[
    UserMessage | SystemMessage | AssistantMessage,
    Field(discriminator="role"),
]
