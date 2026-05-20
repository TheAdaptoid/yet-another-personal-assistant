"""Tests for the models router."""

from unittest.mock import AsyncMock, create_autospec

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from yapa.core.routers.models import get_inference_service, router
from yapa.core.services.inference_service import InferenceService
from yapa.shared.models import ModelData


@pytest.fixture
def mock_service():
    """Return an autospec'd InferenceService with get_models mocked."""
    service = create_autospec(InferenceService, instance=True)
    service.get_models = AsyncMock()
    return service


@pytest.fixture
def app(mock_service):
    """Return a FastAPI app with the models router and mock dependency."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_inference_service] = lambda: mock_service
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app):
    """Return a TestClient for the test app."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestGetModels:
    """GET /models/."""

    def test_200_returns_empty_list(self, client, mock_service):
        """Returns 200 with an empty list when no models are available."""
        mock_service.get_models.return_value = []

        response = client.get("/models/")

        assert response.status_code == 200
        assert response.json() == []

    def test_200_returns_model_list(self, client, mock_service):
        """Returns 200 with a list of models when models are available."""
        models = [
            ModelData(
                id="model-1",
                name="Model 1",
                provider_id="p1",
                provider_name="Provider 1",
            ),
            ModelData(
                id="model-2",
                name="Model 2",
                provider_id="p2",
                provider_name="Provider 2",
            ),
        ]
        mock_service.get_models.return_value = models

        response = client.get("/models/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "model-1"
        assert data[1]["id"] == "model-2"
