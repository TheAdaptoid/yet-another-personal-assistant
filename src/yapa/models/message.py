"""Data models for messages in the chat application."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .base import TrackedEntity
from .inference import TokenUsage
from .tool import ToolCall


class TextPart(BaseModel):
    """A text content part."""

    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    """A URL (http(s) or data URL) plus an optional detail hint."""

    url: str
    detail: str | None = Field(default=None)


class ImagePart(BaseModel):
    """An image content part."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class BaseMessage(TrackedEntity):
    """
    Base class for all message types.

    Attributes:
        role (Literal["user", "assistant", "system", "tool"]):
            The role of the message sender.
        content (str | None): The content of the message.
    """

    role: Literal["user", "assistant", "system", "tool"]
    content: str | None = Field(default=None)


class UserMessage(BaseMessage):
    """
    Represents a message sent by the user.

    Attributes:
        role (Literal["user"]): The role of the message sender, set to "user".
    """

    role: Literal["user"] = "user"
    content: str | list[ContentPart]


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
        reasoning_content (str | None): Optional content that explains the reasoning
            behind the assistant's response.
        model (str | None): The model identifier that generated this response.
        tool_calls (list[ToolCall]): A list of tool calls made by the assistant
            during the generation of this message.
    """

    role: Literal["assistant"] = "assistant"
    reasoning_content: str | None = Field(default=None)
    model: str | None = Field(default=None)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage | None = Field(
        default=None, description="Token usage for this response"
    )


class ToolMessage(BaseMessage):
    """
    Represents a message sent by a tool or external service.

    Attributes:
        role (Literal["tool"]): The role of the message sender, set to "tool".
        metadata (ToolMessageMetadata): Metadata associated with the tool message.
    """

    role: Literal["tool"] = "tool"
    tool_call_id: str
    tool_name: str


Message = Annotated[
    UserMessage | SystemMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]
