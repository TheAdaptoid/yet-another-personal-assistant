"""Streaming event union for the provider boundary."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .inference import TokenUsage


class ContentDelta(BaseModel):
    """A content delta during streaming."""

    type: Literal["content"] = "content"
    content: str


class ReasoningDelta(BaseModel):
    """A reasoning-content delta during streaming."""

    type: Literal["reasoning"] = "reasoning"
    content: str


class ToolCallDeltaEvent(BaseModel):
    """An incremental tool-call delta with raw JSON argument fragments."""

    type: Literal["tool_call"] = "tool_call"
    index: int
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    arguments: str | None = Field(default=None)


class StreamEndEvent(BaseModel):
    """The single final event of a stream."""

    type: Literal["stream_end"] = "stream_end"
    finish_reason: str | None = Field(default=None)
    usage: TokenUsage | None = Field(default=None)
    model_id: str | None = Field(default=None)


StreamEvent = Annotated[
    ContentDelta | ReasoningDelta | ToolCallDeltaEvent | StreamEndEvent,
    Field(discriminator="type"),
]
