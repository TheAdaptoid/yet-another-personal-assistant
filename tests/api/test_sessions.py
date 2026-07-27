"""Tests for session CRUD routes."""

from uuid import uuid4

from yapa.models import InferenceParams, Session


def _make_session(**overrides) -> Session:
    defaults = {
        "title": "Test Session",
        "model": None,
        "system_prompt": None,
        "inference_params": None,
        "messages": [],
    }
    defaults.update(overrides)
    return Session(**defaults)


def test_list_sessions_empty(client, mock_session_service):
    mock_session_service.list.return_value = []
    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    assert response.json() == []


def test_list_sessions_with_pagination(client, mock_session_service):
    sessions = [_make_session(title=f"Session {i}") for i in range(5)]
    mock_session_service.list.return_value = sessions

    response_page_1 = client.get("/api/v1/sessions?page=1&per_page=2")
    assert response_page_1.status_code == 200
    data_page_1 = response_page_1.json()
    assert len(data_page_1) == 2
    assert data_page_1[0]["title"] == "Session 0"
    assert data_page_1[1]["title"] == "Session 1"

    response_page_3 = client.get("/api/v1/sessions?page=3&per_page=2")
    assert response_page_3.status_code == 200
    data_page_3 = response_page_3.json()
    assert len(data_page_3) == 1
    assert data_page_3[0]["title"] == "Session 4"


def test_list_sessions_defaults_to_per_page_20(client, mock_session_service):
    sessions = [_make_session(title=f"Session {i}") for i in range(25)]
    mock_session_service.list.return_value = sessions

    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    assert len(response.json()) == 20


def test_list_sessions_per_page_max_100(client, mock_session_service):
    sessions = [_make_session(title=f"Session {i}") for i in range(150)]
    mock_session_service.list.return_value = sessions

    response = client.get("/api/v1/sessions?per_page=200")
    assert response.status_code == 200
    assert len(response.json()) == 100


def test_create_session(client, mock_session_service):
    session = _make_session(title="New Session")
    mock_session_service.create.return_value = session

    response = client.post("/api/v1/sessions")
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "New Session"
    assert response.headers["Location"] == f"/api/v1/sessions/{session.id}"


def test_get_session(client, mock_session_service):
    session = _make_session(title="Specific Session")
    session_id = session.id
    mock_session_service.get.return_value = session

    response = client.get(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Specific Session"
    assert data["id"] == str(session_id)


def test_get_session_not_found(client, mock_session_service):
    mock_session_service.get.side_effect = ValueError("Session not found")

    response = client.get(f"/api/v1/sessions/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_patch_session_title(client, mock_session_service):
    session = _make_session(title="Renamed")
    session_id = session.id
    mock_session_service.rename.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"title": "Renamed"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Renamed"
    assert data["id"] == str(session_id)
    mock_session_service.rename.assert_called_once_with(
        str(session_id), "Renamed"
    )


def test_patch_session_not_found(client, mock_session_service):
    session_id = uuid4()
    mock_session_service.rename.side_effect = ValueError("Session not found")

    response = client.patch(
        f"/api/v1/sessions/{session_id}",
        json={"title": "Renamed"},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_delete_session(client, mock_session_service):
    session_id = uuid4()

    response = client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 204
    mock_session_service.delete.assert_called_once_with(str(session_id))


def test_delete_session_not_found(client, mock_session_service):
    session_id = uuid4()
    mock_session_service.delete.side_effect = ValueError("Session not found")

    response = client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"


def test_patch_system_prompt(client, mock_session_service):
    session = _make_session(title="Test", system_prompt="You are a helpful assistant.")
    session_id = session.id
    mock_session_service.update_system_prompt.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session_id}/system-prompt",
        json={"system_prompt": "You are a helpful assistant."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["system_prompt"] == "You are a helpful assistant."
    mock_session_service.update_system_prompt.assert_called_once_with(
        str(session_id), "You are a helpful assistant."
    )


def test_patch_system_prompt_clear(client, mock_session_service):
    session = _make_session(title="Test", system_prompt=None)
    session_id = session.id
    mock_session_service.update_system_prompt.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session_id}/system-prompt",
        json={"system_prompt": None},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["system_prompt"] is None
    mock_session_service.update_system_prompt.assert_called_once_with(
        str(session_id), None
    )


def test_patch_inference_params(client, mock_session_service):
    session = _make_session(
        title="Test",
        inference_params=InferenceParams(temperature=0.7),
    )
    session_id = session.id
    mock_session_service.update_inference_params.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session_id}/inference-params",
        json={"temperature": 0.7},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["inference_params"]["temperature"] == 0.7
    mock_session_service.update_inference_params.assert_called_once()
    args, _ = mock_session_service.update_inference_params.call_args
    assert args[0] == str(session_id)
    assert args[1].temperature == 0.7


def test_patch_inference_params_clear(client, mock_session_service):
    session = _make_session(title="Test", inference_params=None)
    session_id = session.id
    mock_session_service.update_inference_params.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session_id}/inference-params",
        json={},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["inference_params"] is None
    mock_session_service.update_inference_params.assert_called_once()
    args, _ = mock_session_service.update_inference_params.call_args
    assert args[0] == str(session_id)
    assert args[1] is not None
    assert args[1].temperature is None
    assert args[1].max_tokens is None
    assert args[1].top_p is None
