"""Data models for messages in the chat application."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
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
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    role: Literal["user", "assistant", "system"]
    content: str

    @abstractmethod
    def to_openai_format(self) -> ChatCompletionMessageParam:
        """
        Convert the message to a format compatible with OpenAI's API.

        Returns:
            ChatCompletionMessageParam: A dictionary with keys "role", "content", and
            other optional fields depending on the message type.
        """
        pass


class UserMessage(BaseMessage):
    """
    Represents a message sent by the user.

    Attributes:
        role (Literal["user"]): The role of the message sender, set to "user".
    """

    role: Literal["user"] = "user"

    def to_openai_format(self) -> ChatCompletionUserMessageParam:
        """Convert the UserMessage to OpenAI's ChatCompletionUserMessageParam."""
        return ChatCompletionUserMessageParam(role=self.role, content=self.content)


class SystemMessage(BaseMessage):
    """
    Represents a message sent by the system or application.

    Attributes:
        role (Literal["system"]): The role of the message sender, set to "system".
    """

    role: Literal["system"] = "system"

    def to_openai_format(self) -> ChatCompletionSystemMessageParam:
        """Convert the SystemMessage to OpenAI's ChatCompletionSystemMessageParam."""
        return ChatCompletionSystemMessageParam(role=self.role, content=self.content)


class AssistantMessage(BaseMessage):
    """
    Represents a message sent by the AI assistant.

    Attributes:
        role (Literal["assistant"]): The role of the message sender, set to "assistant".
        model (str | None): The model identifier that generated this response.
    """

    role: Literal["assistant"] = "assistant"
    model: str | None = Field(default=None)

    def to_openai_format(self) -> ChatCompletionAssistantMessageParam:
        """Convert the AssistantMessage to OpenAI's ChatCompletionAssistantMessageParam."""  # noqa: E501
        return ChatCompletionAssistantMessageParam(
            role=self.role,
            content=self.content,
        )


Message = Annotated[
    UserMessage | SystemMessage | AssistantMessage,
    Field(discriminator="role"),
]
