"""Tests for WebSocket chat endpoint."""

from uuid import uuid4

import pytest

from yapa.models import ModelData, ModelType, Session
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    TextEvent,
)


def test_chat_ws_streams_events(client, mock_chat_service, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
    )

    async def _stream(*, session_id, prompt, model):
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
    requested_model = ModelData(
        id="gpt-4o",
        provider_id="openai",
        type=ModelType.LLM,
    )
    mock_model_service.get_model.return_value = requested_model

    seen_model = {}

    async def _stream(*, session_id, prompt, model):
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
        model=ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
    )

    calls = 0

    async def _stream(*, session_id, prompt, model):
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
        model=ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
    )

    async def _stream(*, session_id, prompt, model):
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
        model=ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
    )

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"model": "openai:gpt-4o"})
        with pytest.raises(Exception):
            ws.receive_json()


def test_chat_ws_invalid_json(client, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
    )

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_text("not json")
        with pytest.raises(Exception):
            ws.receive_json()


def test_chat_ws_invalid_session(client, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.side_effect = ValueError("Session not found")

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        with pytest.raises(Exception):
            ws.receive_json()
