"""Tests for the chat WebSocket routes."""

from unittest.mock import AsyncMock, create_autospec

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yapa.core.routers.chat import get_chat_service, router
from yapa.core.services.chat_service import ChatService
from yapa.shared.models import ChatResponse

# --- Fixtures ------------------------------------------------------------------

WS_PATH = "/chat/ws/test-session"

PAYLOAD = {
    "model": {
        "id": "test-model",
        "name": "Test",
        "provider_id": "test",
        "provider_name": "Test",
    },
    "message": "Hi",
}


@pytest.fixture
def mock_chat_service():
    """Mock ChatService with AsyncMock for public methods."""
    service = create_autospec(ChatService, instance=True)
    service.process_message = AsyncMock()
    return service


@pytest.fixture
def app(mock_chat_service):
    """Minimal FastAPI app with the chat router and a mocked dependency."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    """TestClient backed by the mocked app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# --- WebSocket /chat/ws/{session_id} -------------------------------------------


class TestChatWebSocket:
    """WebSocket chat endpoint — ChatService.process_message."""

    def test_success(self, client, mock_chat_service):
        """A valid message returns a successful ChatResponse."""
        mock_chat_service.process_message.return_value = ChatResponse(
            response="Hello, world!",
            done=True,
        )

        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(PAYLOAD)
            data = ws.receive_json()

        assert data["response"] == "Hello, world!"
        assert data["done"] is True
        assert data["error"] is None

    def test_error_keeps_alive(self, client, mock_chat_service):
        """An error response keeps the connection alive for further messages."""
        mock_chat_service.process_message.side_effect = [
            ChatResponse(response="", done=True, error="session gone"),
            ChatResponse(response="Recovered", done=True),
        ]

        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(PAYLOAD)
            data1 = ws.receive_json()
            assert data1["error"] == "session gone"

            ws.send_json(PAYLOAD)
            data2 = ws.receive_json()
            assert data2["response"] == "Recovered"

    def test_retry_flag_forwarded(self, client, mock_chat_service):
        """The retry field is forwarded to ChatService.process_message."""
        mock_chat_service.process_message.return_value = ChatResponse(
            response="Retry OK",
            done=True,
        )

        payload = {**PAYLOAD, "retry": True}
        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(payload)
            data = ws.receive_json()

        assert data["response"] == "Retry OK"
        _, kwargs = mock_chat_service.process_message.call_args
        assert kwargs.get("retry") is True

    def test_multiple_messages(self, client, mock_chat_service):
        """Multiple messages can be sent over a single connection."""
        mock_chat_service.process_message.side_effect = [
            ChatResponse(response="First reply", done=True),
            ChatResponse(response="Second reply", done=True),
        ]

        with client.websocket_connect(WS_PATH) as ws:
            ws.send_json(PAYLOAD)
            data1 = ws.receive_json()

            ws.send_json(PAYLOAD)
            data2 = ws.receive_json()

        assert data1["response"] == "First reply"
        assert data2["response"] == "Second reply"
