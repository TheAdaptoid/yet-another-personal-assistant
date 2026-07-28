"""Data models for tool calls and tool approval."""

from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    """
    Represents a tool call made by the assistant.

    Attributes:
        tool_name (str): The name of the tool being called.
        arguments (dict[str, Any]): A dictionary of arguments passed to the tool.
    """

    id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolApprovalRequest(BaseModel):
    """
    Represents a tool approval request made by the assistant.

    Attributes:
        call_id (str): The unique identifier of the tool call being requested.
            Matches the id field of the corresponding ToolCall.
        name (str): The name of the tool for which approval is requested.
        arguments (dict[str, Any]): A dictionary of arguments for the tool call.
    """

    call_id: str
    name: str
    arguments: dict[str, Any]


class ToolApprovalResponse(BaseModel):
    """
    Represents a user's decision on a tool approval request.

    Attributes:
        call_id (str): The unique identifier of the tool call this response
            refers to. Matches the call_id of the corresponding
            ToolApprovalRequest.
        approved (bool): Whether the tool call is approved or not.
        reason (str | None): Optional reason for the rejection. Ignored when
            approved is True.
    """

    call_id: str
    approved: bool
    reason: str | None = None
