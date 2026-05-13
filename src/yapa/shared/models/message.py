"""Data models for messages in the chat application."""

from time import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseMessage(BaseModel):
    """
    Base class for all message types.

    Attributes:
        id (str): Unique identifier for the message, generated as a UUID4 hex string.
        role (Literal["user", "assistant", "system"]): The role of the message sender.
        content (str): The content of the message.
        timestamp (int): The timestamp of when the message was created, represented as
            an integer (Unix time).
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: int = Field(default_factory=lambda: int(time()))


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
        name (str | None): An optional name for the system entity.
    """

    role: Literal["system"] = "system"
    name: str | None = None


class AssistantMessage(BaseMessage):
    """
    Represents a message sent by the AI assistant.

    Attributes:
        role (Literal["assistant"]): The role of the message sender, set to "assistant".
        model (str | None): The model identifier that generated this response.
    """

    role: Literal["assistant"] = "assistant"
    model: str | None = None


Message = Annotated[
    UserMessage | SystemMessage | AssistantMessage,
    Field(discriminator="role"),
]


def create_user_message(content: str) -> UserMessage:
    """
    Create a UserMessage instance.

    Args:
        content (str): The content of the user message.

    Returns:
        UserMessage: An instance of UserMessage with the provided content.
    """

    return UserMessage(content=content)


def create_system_message(content: str, name: str | None = None) -> SystemMessage:
    """
    Create a SystemMessage instance.

    Args:
        content (str): The content of the system message.
        name (str | None): An optional name for the system entity.

    Returns:
        SystemMessage: An instance of SystemMessage with the provided
        content and optional name.
    """

    return SystemMessage(content=content, name=name)


def create_assistant_message(
    content: str, model: str | None = None
) -> AssistantMessage:
    """
    Create an AssistantMessage instance.

    Args:
        content (str): The content of the assistant message.
        model (str | None): An optional model identifier.

    Returns:
        AssistantMessage: An instance of AssistantMessage with the
        provided content and optional model.
    """

    return AssistantMessage(content=content, model=model)
