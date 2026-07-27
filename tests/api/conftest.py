from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from yapa.api.app import create_app
from yapa.services import ChatService, ModelService, SessionService


@pytest.fixture
def mock_session_service():
    return MagicMock(spec=SessionService)


@pytest.fixture
def mock_model_service():
    return MagicMock(spec=ModelService)


@pytest.fixture
def mock_chat_service():
    mock = MagicMock(spec=ChatService)
    mock.stream = AsyncMock()
    return mock


@pytest.fixture
def app(mock_session_service, mock_model_service, mock_chat_service):
    app = create_app()
    app.state.session_service = mock_session_service
    app.state.model_service = mock_model_service
    app.state.chat_service = mock_chat_service
    return app


@pytest.fixture
def client(app):
    return TestClient(app)
