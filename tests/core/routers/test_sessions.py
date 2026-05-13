"""Tests for session API routes."""

from unittest.mock import AsyncMock, create_autospec

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yapa.core.repositories import (
    SessionDeleteError,
    SessionLoadError,
    SessionSaveError,
)
from yapa.core.routers.sessions import get_session_service, router
from yapa.core.services.session_service import SessionService
from yapa.shared.models import Session


# --- Fixtures ------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Mock SessionService with AsyncMock for every public method."""
    service = create_autospec(SessionService, instance=True)
    service.create_session = AsyncMock()
    service.list_sessions = AsyncMock()
    service.get_session = AsyncMock()
    service.rename_session = AsyncMock()
    service.delete_session = AsyncMock()
    return service


@pytest.fixture
def app(mock_service):
    """Minimal FastAPI app with the sessions router and a mocked dependency."""
    app = FastAPI()
    app.include_router(router, prefix="/sessions")
    app.dependency_overrides[get_session_service] = lambda: mock_service
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    """TestClient backed by the mocked app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def sample_session() -> Session:
    """Pre-built Session for reuse in happy-path tests."""
    return Session(title="test")


# --- POST /sessions/ -----------------------------------------------------------


class TestCreateSession:
    """POST /sessions/ — SessionService.create_session."""

    def test_201_with_body(self, client, mock_service):
        """A valid title is passed to the service and the session is returned."""
        session = Session(title="My Chat")
        mock_service.create_session.return_value = session

        response = client.post("/sessions/", json={"title": "My Chat"})

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == session.id
        assert data["title"] == "My Chat"

    def test_201_without_body(self, client, mock_service):
        """Omitting the body still creates a session (default title)."""
        session = Session()
        mock_service.create_session.return_value = session

        response = client.post("/sessions/")

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == session.id

    def test_500_on_save_error(self, client, mock_service):
        """SessionSaveError propagates to a 500 response."""
        mock_service.create_session.side_effect = SessionSaveError("disk full")

        response = client.post("/sessions/", json={"title": "boom"})

        assert response.status_code == 500


# --- GET /sessions/ ------------------------------------------------------------


class TestListSessions:
    """GET /sessions/ — SessionService.list_sessions."""

    def test_200_returns_list(self, client, mock_service):
        """Returns a JSON array of sessions."""
        s1 = Session(title="first")
        s2 = Session(title="second")
        mock_service.list_sessions.return_value = [s1, s2]

        response = client.get("/sessions/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == s1.id
        assert data[1]["id"] == s2.id

    def test_200_empty(self, client, mock_service):
        """No sessions returns an empty array."""
        mock_service.list_sessions.return_value = []

        response = client.get("/sessions/")

        assert response.status_code == 200
        assert response.json() == []

    def test_500_on_load_error(self, client, mock_service):
        """SessionLoadError propagates to a 500 response."""
        mock_service.list_sessions.side_effect = SessionLoadError("storage error")

        response = client.get("/sessions/")

        assert response.status_code == 500


# --- GET /sessions/{session_id} ------------------------------------------------


class TestGetSession:
    """GET /sessions/{session_id} — SessionService.get_session."""

    def test_200_found(self, client, mock_service, sample_session):
        """Found session is returned as JSON."""
        mock_service.get_session.return_value = sample_session

        response = client.get(f"/sessions/{sample_session.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_session.id
        assert data["title"] == "test"

    def test_404_not_found(self, client, mock_service):
        """Missing session returns 404."""
        mock_service.get_session.return_value = None

        response = client.get("/sessions/missing-id")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_500_on_load_error(self, client, mock_service):
        """SessionLoadError propagates to a 500 response."""
        mock_service.get_session.side_effect = SessionLoadError("corrupt file")

        response = client.get("/sessions/any")

        assert response.status_code == 500


# --- PATCH /sessions/{session_id} ----------------------------------------------


class TestRenameSession:
    """PATCH /sessions/{session_id} — SessionService.rename_session."""

    def test_200_success(self, client, mock_service, sample_session):
        """Successful rename returns the session fetched by get_session."""
        mock_service.rename_session.return_value = True
        mock_service.get_session.return_value = sample_session

        response = client.patch(
            f"/sessions/{sample_session.id}",
            json={"title": "renamed"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_session.id

    def test_404_not_found(self, client, mock_service):
        """rename_session returning False yields a 404."""
        mock_service.rename_session.return_value = False

        response = client.patch("/sessions/missing", json={"title": "new name"})

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_500_on_load_error(self, client, mock_service):
        """SessionLoadError from rename_session propagates to 500."""
        mock_service.rename_session.side_effect = SessionLoadError("corrupt")

        response = client.patch("/sessions/any", json={"title": "new name"})

        assert response.status_code == 500

    def test_500_on_save_error(self, client, mock_service):
        """SessionSaveError from rename_session propagates to 500."""
        mock_service.rename_session.side_effect = SessionSaveError("disk full")

        response = client.patch("/sessions/any", json={"title": "new name"})

        assert response.status_code == 500

    def test_422_missing_title(self, client, mock_service):
        """Omitting the required 'title' field returns 422."""
        response = client.patch("/sessions/any", json={})

        assert response.status_code == 422
        # The service should not be called when the request body is invalid
        mock_service.rename_session.assert_not_called()


# --- DELETE /sessions/{session_id} ---------------------------------------------


class TestDeleteSession:
    """DELETE /sessions/{session_id} — SessionService.delete_session."""

    def test_204_success(self, client, mock_service):
        """Successful deletion returns 204 No Content."""
        mock_service.delete_session.return_value = True

        response = client.delete("/sessions/some-id")

        assert response.status_code == 204
        assert response.content == b""

    def test_404_not_found(self, client, mock_service):
        """delete_session returning False yields a 404."""
        mock_service.delete_session.return_value = False

        response = client.delete("/sessions/missing")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    def test_500_on_delete_error(self, client, mock_service):
        """SessionDeleteError propagates to a 500 response."""
        mock_service.delete_session.side_effect = SessionDeleteError(
            "permission denied"
        )

        response = client.delete("/sessions/any")

        assert response.status_code == 500
