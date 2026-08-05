"""Tests for WebSocket chat endpoint."""

from uuid import uuid4

import pytest

from yapa.models import LanguageModel, Session
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    TextEvent,
    ToolApprovalRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from yapa.models.tool import ToolApprovalRequest


def test_chat_ws_streams_events(client, mock_chat_service, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=LanguageModel(id="gpt-4o", provider_id="openai")
    )

    async def _stream(*, session_id, prompt, model, **kwargs):
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield TextEvent(content="Hello")
        yield AgentDoneEvent(content="Hello", finish_reason="stop")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Hi"})

        msg1 = ws.receive_json()
        assert msg1["type"] == "agent_start"
        assert msg1["model_id"] == "openai:gpt-4o"

        msg2 = ws.receive_json()
        assert msg2["type"] == "text_chunk"
        assert msg2["content"] == "Hello"

        msg3 = ws.receive_json()
        assert msg3["type"] == "agent_done"
        assert msg3["content"] == "Hello"


def test_chat_ws_uses_requested_model(
    client,
    mock_chat_service,
    mock_session_service,
    mock_model_service,
):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session()
    requested_model = LanguageModel(
        id="gpt-4o",
        provider_id="openai",
    )
    mock_model_service.get_model.return_value = requested_model

    seen_model = {}

    async def _stream(*, session_id, prompt, model, **kwargs):
        seen_model["value"] = model
        yield AgentStartEvent(model_id=model.full_id)
        yield AgentDoneEvent(content="Hello", finish_reason="stop")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Hi", "model": "openai:gpt-4o"})
        ws.receive_json()
        ws.receive_json()

    assert seen_model["value"] == requested_model
    mock_model_service.get_model.assert_called_once_with("openai:gpt-4o")


def test_chat_ws_closes_when_no_model_can_be_resolved(
    client,
    mock_chat_service,
    mock_session_service,
):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session()

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Hi"})
        with pytest.raises(Exception):
            ws.receive_json()


def test_chat_ws_multiple_prompts(client, mock_chat_service, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=LanguageModel(id="gpt-4o", provider_id="openai")
    )

    calls = 0

    async def _stream(*, session_id, prompt, model, **kwargs):
        nonlocal calls
        calls += 1
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield TextEvent(content=f"Response {calls}")
        yield AgentDoneEvent(content=f"Response {calls}", finish_reason="stop")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "First"})
        assert ws.receive_json()["type"] == "agent_start"
        assert ws.receive_json()["content"] == "Response 1"
        assert ws.receive_json()["type"] == "agent_done"

        ws.send_json({"prompt": "Second"})
        assert ws.receive_json()["type"] == "agent_start"
        assert ws.receive_json()["content"] == "Response 2"
        assert ws.receive_json()["type"] == "agent_done"


def test_chat_ws_error_event(client, mock_chat_service, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=LanguageModel(id="gpt-4o", provider_id="openai")
    )

    async def _stream(*, session_id, prompt, model, **kwargs):
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield AgentErrorEvent(message="Something went wrong")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Hi"})
        ws.receive_json()  # agent_start
        error = ws.receive_json()
        assert error["type"] == "agent_error"
        assert "Something went wrong" in error["message"]


def test_chat_ws_missing_prompt(client, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=LanguageModel(id="gpt-4o", provider_id="openai")
    )

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"model": "openai:gpt-4o"})
        with pytest.raises(Exception):
            ws.receive_json()


def test_chat_ws_invalid_json(client, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=LanguageModel(id="gpt-4o", provider_id="openai")
    )

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_text("not json")
        with pytest.raises(Exception):
            ws.receive_json()


def test_chat_ws_tool_approval_flow(client, mock_chat_service, mock_session_service):
    """WebSocket sends ToolApprovalRequestEvent and client responds with approval."""
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=LanguageModel(id="gpt-4o", provider_id="openai")
    )

    async def _stream(*, session_id, prompt, model, reasoning=None, get_approval=None):
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield ToolCallEvent(
            tool_name="write_file",
            arguments={"path": "/tmp/test.txt"},
            call_id="call_1",
        )
        yield ToolApprovalRequestEvent(
            tool_name="write_file",
            arguments={"path": "/tmp/test.txt"},
            call_id="call_1",
        )
        # Simulate waiting for approval
        response = await get_approval(
            ToolApprovalRequest(
                call_id="call_1", name="write_file", arguments={"path": "/tmp/test.txt"}
            )
        )
        if response.approved:
            yield ToolResultEvent(tool_name="write_file", call_id="call_1", result="ok")
        yield AgentDoneEvent(content="File written", finish_reason="stop")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Write a file"})

        # Receive events until we get an approval request
        while True:
            msg = ws.receive_json()
            if msg["type"] == "tool_approval_request":
                assert msg["tool_name"] == "write_file"
                # Send approval response
                ws.send_json(
                    {"type": "tool_approval", "call_id": "call_1", "approved": True}
                )
                break

        # Receive remaining events
        msg = ws.receive_json()
        assert msg["type"] in ("tool_result", "agent_done")
        if msg["type"] == "tool_result":
            msg = ws.receive_json()
            assert msg["type"] == "agent_done"


def test_chat_ws_tool_denial_flow(client, mock_chat_service, mock_session_service):
    """Client can deny a tool call."""
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=LanguageModel(id="gpt-4o", provider_id="openai")
    )

    async def _stream(*, session_id, prompt, model, reasoning=None, get_approval=None):
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield ToolCallEvent(tool_name="write_file", arguments={}, call_id="call_1")
        yield ToolApprovalRequestEvent(
            tool_name="write_file", arguments={}, call_id="call_1"
        )
        response = await get_approval(
            ToolApprovalRequest(call_id="call_1", name="write_file", arguments={})
        )
        assert not response.approved
        yield AgentDoneEvent(content="Denied", finish_reason="stop")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Write a file"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "tool_approval_request":
                ws.send_json(
                    {
                        "type": "tool_approval",
                        "call_id": "call_1",
                        "approved": False,
                        "reason": "unsafe",
                    }
                )
                break
        msg = ws.receive_json()
        assert msg["type"] == "agent_done"
