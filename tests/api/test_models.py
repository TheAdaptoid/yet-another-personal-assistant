"""Tests for model listing and lookup endpoints."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from yapa.models import ModelData, ModelType
from yapa.providers.registry import ProviderNotAvailableError


@pytest.fixture
def sample_models():
    return [
        ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM),
        ModelData(id="text-embedding-ada-002", provider_id="openai",
                  type=ModelType.OTHER),
        ModelData(id="llama-3", provider_id="ollama", type=ModelType.LLM),
        ModelData(id="text-embedding-v3", provider_id="openai", type=ModelType.OTHER),
        ModelData(id="mistral-7b", provider_id="mistral", type=ModelType.LLM),
    ]

def test_filter_valid_llm(client, sample_models):
    client.app.state.model_service.list_models = AsyncMock(
        return_value=[m for m in sample_models if m.type == ModelType.LLM]
    )
    response = client.get("/api/v1/models?model_type=llm")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    for m in data:
        assert m["type"] == "llm"

def test_filter_valid_other(client, sample_models):
    client.app.state.model_service.list_models = AsyncMock(
        return_value=[m for m in sample_models if m.type == ModelType.OTHER]
    )
    response = client.get("/api/v1/models?model_type=other")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    for m in data:
        assert m["type"] == "other"

def test_filter_invalid_model_type(client):
    response = client.get("/api/v1/models?model_type=invalid")
    assert response.status_code == 400
    assert "Invalid model type" in response.json()["detail"]

def test_no_models_found(client):
    client.app.state.model_service.list_models = AsyncMock(return_value=[])
    response = client.get("/api/v1/models?model_type=llm")
    assert response.status_code == 200
    assert response.json() == []

def test_combined_provider_and_type_filtering(client, sample_models):
    expected = [m for m in sample_models
                if m.provider_id == "openai" and m.type == ModelType.LLM]
    client.app.state.model_service.list_models = AsyncMock(return_value=expected)
    response = client.get("/api/v1/models?provider_id=openai&model_type=llm")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == "gpt-4o"
    assert data[0]["type"] == "llm"


def test_list_models(client, sample_models):
    client.app.state.model_service.list_models = AsyncMock(return_value=sample_models)
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5


def test_list_models_by_provider(client, sample_models):
    client.app.state.model_service.list_models = AsyncMock(
        return_value=[sample_models[0]]
    )
    response = client.get("/api/v1/models?provider_id=openai")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["provider_id"] == "openai"


def test_get_model_by_full_id(client, sample_models):
    client.app.state.model_service.get_model = AsyncMock(return_value=sample_models[0])
    response = client.get("/api/v1/models/openai:gpt-4o")
    assert response.status_code == 200
    assert response.json()["id"] == "gpt-4o"


def test_get_model_not_found(client):
    client.app.state.model_service.get_model = AsyncMock(
        side_effect=ValueError("Failed to fetch model 'openai:ghost'")
    )
    response = client.get("/api/v1/models/openai:ghost")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_get_model_invalid_format(client):
    client.app.state.model_service.get_model = AsyncMock(
        side_effect=ValueError("Invalid format")
    )
    response = client.get("/api/v1/models/bad-format")
    assert response.status_code == 404


def test_get_model_provider_not_available(client):
    client.app.state.model_service.get_model = AsyncMock(
        side_effect=ProviderNotAvailableError("Provider 'unknown' not found")
    )
    response = client.get("/api/v1/models/unknown:gpt-4")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"]


def test_unhandled_exception_returns_500(client, app):
    quiet_client = TestClient(app, raise_server_exceptions=False)
    quiet_client.app.state.model_service.list_models = AsyncMock(
        side_effect=RuntimeError("unexpected")
    )
    response = quiet_client.get("/api/v1/models")
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
