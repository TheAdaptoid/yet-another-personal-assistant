"""Agent event types for the agent-service event system."""

import time
from enum import Enum

from pydantic import BaseModel, Field

from .tool import ToolApprovalRequest, ToolApprovalResponse


class EventType(str, Enum):
    """Enumeration of event types for the event system."""

    TEXT_CHUNK = "text_chunk"
    REASONING_CHUNK = "reasoning_chunk"

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_APPROVAL_REQUEST = "tool_approval_request"
    TOOL_APPROVAL_RESPONSE = "tool_approval_response"

    AGENT_START = "agent_start"
    AGENT_ERROR = "agent_error"
    AGENT_DONE = "agent_done"


class EventSource(str, Enum):
    """Enumeration of event sources for the event system."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class Event(BaseModel):
    """
    Base class for all agent events.

    Attributes:
        type: Discriminator for the concrete event kind.
        source: Originator of the event.
        timestamp: Epoch seconds when the event was created.
    """

    type: EventType
    source: EventSource = EventSource.SYSTEM
    timestamp: float = Field(default_factory=time.time)


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
    """
    Emitted when the agent begins processing a message.

    Sent once per ``process_message`` invocation, before the first model
    call.
    """

    type: EventType = EventType.AGENT_START
    source: EventSource = EventSource.AGENT


class AgentDoneEvent(Event):
    """
    Emitted after all messages have been persisted.

    Marks clean termination of the agent loop for this turn.
    """

    type: EventType = EventType.AGENT_DONE
    source: EventSource = EventSource.AGENT


class AgentErrorEvent(Event):
    """Emitted when the agent encounters an unrecoverable error."""

    type: EventType = EventType.AGENT_ERROR
    source: EventSource = EventSource.AGENT
    message: str


class ToolResultEvent(Event):
    """Notification that a single tool call has completed."""

    type: EventType = EventType.TOOL_RESULT
    source: EventSource = EventSource.SYSTEM
    call_id: str
    tool_name: str
    result: str
    execution_time: float
    success: bool


class ToolApprovalRequestEvent(Event):
    """
    Request for the user to approve one or more pending tool calls.

    One ``ToolApprovalRequestEvent`` carries the full batch of tool calls
    requested by the model in a single turn. The UI must present all of
    them and collect a response for each before the agent can continue.
    """

    type: EventType = EventType.TOOL_APPROVAL_REQUEST
    source: EventSource = EventSource.AGENT
    requests: list[ToolApprovalRequest]


class ToolApprovalResponseEvent(Event):
    """
    User decision(s) on a previously issued approval request.

    Contains one ``ToolApprovalResponse`` per tool call in the batch.
    """

    type: EventType = EventType.TOOL_APPROVAL_RESPONSE
    source: EventSource = EventSource.USER
    responses: list[ToolApprovalResponse]
