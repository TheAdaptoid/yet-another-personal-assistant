"""Data models for messages in the chat application."""

from abc import ABC
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseMessage(ABC, BaseModel):
    """
    Base class for all message types.

    Attributes:
        id (str): Unique identifier for the message, generated as a UUID4 hex string.
        role (Literal["user", "assistant", "system"]): The role of the message sender.
        content (str): The content of the message.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        json_schema_extra={
            "sa_column_kwargs": {"onupdate": lambda: datetime.now(timezone.utc)},
        },
    )
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
