"""Phase 1 event types for the agent-service event system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from yapa.models.inference import TokenUsage
from yapa.tools.base import JsonValue


class EventType(str, Enum):
    """Enumeration of event types for the event system."""

    TEXT_CHUNK = "text_chunk"
    REASONING_CHUNK = "reasoning_chunk"
    AGENT_START = "agent_start"
    AGENT_DONE = "agent_done"
    AGENT_ERROR = "agent_error"
    TOOL_CALL = "tool_call"
    TOOL_APPROVAL_REQUEST = "tool_approval_request"
    TOOL_RESULT = "tool_result"


class EventSource(str, Enum):
    """Enumeration of event sources for the event system."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class Event(BaseModel):
    """Base class for all agent events."""

    type: EventType
    source: EventSource = EventSource.SYSTEM
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TextEvent(Event):
    """A chunk of streaming text content from the agent."""

    type: EventType = EventType.TEXT_CHUNK
    source: EventSource = EventSource.AGENT
    content: str


class ReasoningEvent(Event):
    """A chunk of streaming reasoning/thinking content from the agent."""

    type: EventType = EventType.REASONING_CHUNK
    source: EventSource = EventSource.AGENT
    content: str


class AgentStartEvent(Event):
    """Emitted when the agent begins processing a message."""

    type: EventType = EventType.AGENT_START
    source: EventSource = EventSource.AGENT
    model_id: str


class AgentDoneEvent(Event):
    """Emitted after the model response is complete."""

    type: EventType = EventType.AGENT_DONE
    source: EventSource = EventSource.AGENT
    content: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class AgentErrorEvent(Event):
    """Emitted when the agent encounters an unrecoverable error."""

    type: EventType = EventType.AGENT_ERROR
    source: EventSource = EventSource.AGENT
    message: str


class ToolCallEvent(Event):
    """Emitted when the model requests a tool call."""

    type: EventType = EventType.TOOL_CALL
    source: EventSource = EventSource.AGENT
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


class ToolApprovalRequestEvent(Event):
    """Emitted before executing a tool that requires approval."""

    type: EventType = EventType.TOOL_APPROVAL_REQUEST
    source: EventSource = EventSource.SYSTEM
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


class ToolResultEvent(Event):
    """Emitted after a tool has been executed."""

    type: EventType = EventType.TOOL_RESULT
    source: EventSource = EventSource.SYSTEM
    tool_name: str
    call_id: str
    result: JsonValue
