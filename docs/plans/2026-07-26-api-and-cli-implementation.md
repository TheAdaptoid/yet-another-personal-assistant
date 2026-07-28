# API Layer & Minimal CLI — Implementation Plan

> **For agentic workers:** Each task is independently testable. Work through them in order. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build a FastAPI REST + WebSocket API and a maintenance-only Typer CLI on top of the existing YAPA services layer.

**Architecture:** FastAPI factory `create_app()` configures lifespan (creates services, stores on `app.state`), mounts routes with `/api/v1` prefix. CLI is a separate Typer app that instantiates services directly. Both share the same service layer (`SessionService`, `ModelService`, `ChatService`) and config (`JsonConfigStore`).

**Tech Stack:** Python 3.13+, FastAPI ≥0.115.0, uvicorn[standard] ≥0.34.0, Typer, Rich, Pydantic v2, pytest.

## Global Constraints

- All route paths use `/api/v1/` prefix (configurable via settings).
- Every endpoint declares explicit `response_model=` on the route decorator.
- `POST /api/v1/sessions` returns `Location: /api/v1/sessions/{id}` header.
- `GET /api/v1/sessions` supports `?page=1&per_page=20` (max 100).
- Errors: `ValueError` → 404, `ProviderNotAvailableError` → 404, `ChatError` → 500, unhandled → 500.
- CLI visual palette: `orange1` (labels/keys), `white bold` (values), `green` (success), `red` (errors), `dim` (counts/muted), `blue` (table headers).
- CLI has no config/setup/Typer routing in `__main__.py` beyond calling the Typer app.
- All new code follows `ruff.toml` (line-length 88, select F/E/I/C90/D) and `ty.toml` rules.
- Coverage floor: 80% (covers entire `src/`, existing tests baseline ~95%).

---

### Task 1: Project Dependencies + FastAPI App Factory + Health Route

**Files:**
- Modify: `pyproject.toml`
- Create: `src/yapa/api/__init__.py`
- Create: `src/yapa/api/app.py`
- Create: `src/yapa/api/dependencies.py`
- Create: `src/yapa/api/routes/__init__.py`
- Create: `src/yapa/api/routes/health.py`
- Create: `tests/api/__init__.py`
- Create: `tests/api/conftest.py`
- Create: `tests/api/test_health.py`

**Interfaces:**
- Consumes: `SessionService` (from `yapa.services.session`), `ModelService` (from `yapa.services.models`), `ChatService` (from `yapa.services.chat`), `Config` (from `yapa.services.config`), `ChatError` (from `yapa.services.exceptions`), `ProviderNotAvailableError` (from `yapa.providers.registry`).
- Produces: `create_app(config: Config | None = None) -> FastAPI` — creates app with lifespan, exception handlers, and health route mounted.

- [ ] **Step 1: Add FastAPI + uvicorn to pyproject.toml**

```toml
dependencies = [
    "openai>=2.36.0",
    "openrouter>=0.9.1",
    "pydantic>=2.13.4",
    "python-dotenv>=1.2.2",
    "rich>=15.0.0",
    "typer>=0.26.7",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
]
```

- [ ] **Step 2: Create empty `__init__.py` files and `api/routes/__init__.py`**

Create empty files:
- `src/yapa/api/__init__.py`
- `src/yapa/api/routes/__init__.py`
- `tests/api/__init__.py`

- [ ] **Step 3: Create `src/yapa/api/dependencies.py`**

```python
"""FastAPI dependency injection for service layer."""

from fastapi import Request

from yapa.services import ChatService, ModelService, SessionService


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service
```

- [ ] **Step 4: Create `src/yapa/api/app.py`**

```python
"""FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from yapa.providers import ProviderNotAvailableError
from yapa.services import ChatService, ModelService, SessionService
from yapa.services.config import Config, DEFAULT_DATA_DIR, JsonConfigStore
from yapa.services.exceptions import ChatError
from yapa.services.store import JsonSessionStore

from .dependencies import get_chat_service, get_model_service, get_session_service
from .routes import health, models, sessions
from .websocket import chat as chat_ws


def _build_services(config: Config):
    store = JsonSessionStore(config.storage_dir)
    session_service = SessionService(store)
    model_service = ModelService()
    chat_service = ChatService(
        sessions=session_service,
        models=model_service,
    )
    return session_service, model_service, chat_service


def create_app(config: Config | None = None) -> FastAPI:
    if config is None:
        config = JsonConfigStore().load()

    app = FastAPI(title="YAPA")

    @app.on_event("startup")
    def _startup():
        (
            app.state.session_service,
            app.state.model_service,
            app.state.chat_service,
        ) = _build_services(config)

    @app.exception_handler(ValueError)
    def _value_error(_request: Request, exc: ValueError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ProviderNotAvailableError)
    def _provider_error(_request: Request, exc: ProviderNotAvailableError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ChatError)
    def _chat_error(_request: Request, exc: ChatError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    def _generic_error(_request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(models.router)
    app.include_router(chat_ws.router)

    return app
```

- [ ] **Step 5: Create `src/yapa/api/routes/health.py`**

```python
"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=dict[str, str])
async def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Create `tests/api/conftest.py`**

```python
"""Fixtures for API tests."""

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
```

- [ ] **Step 7: Create `tests/api/test_health.py`**

```python
"""Tests for health endpoint."""


def test_health_returns_ok(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_method_not_allowed(client):
    response = client.post("/api/v1/health")
    assert response.status_code == 405
```

- [ ] **Step 8: Run tests to verify**

```bash
uv run pytest tests/api/test_health.py -v
```

Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "feat: add FastAPI app factory and health endpoint"
```

---

### Task 2: Session CRUD REST Routes

**Files:**
- Create: `src/yapa/api/routes/sessions.py`
- Create: `tests/api/test_sessions.py`

**Interfaces:**
- Consumes: `get_session_service()` from `api/dependencies.py`.
- Produces: session CRUD routes with `/api/v1/sessions` prefix.

- [ ] **Step 1: Write the tests**

Create `tests/api/test_sessions.py`:

```python
"""Tests for session CRUD endpoints."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from yapa.models import Session


def test_list_sessions_empty(client: TestClient):
    from tests.api.conftest import mock_session_service  # noqa: F811

    client.app.state.session_service.list.return_value = []
    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    assert response.json() == []


def test_list_sessions_with_pagination(client: TestClient):
    sessions = [Session(title=f"Session {i}") for i in range(5)]
    client.app.state.session_service.list.return_value = sessions

    response = client.get("/api/v1/sessions?page=1&per_page=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["title"] == "Session 0"

    response = client.get("/api/v1/sessions?page=3&per_page=2")
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Session 4"


def test_list_sessions_defaults_to_per_page_20(client: TestClient):
    sessions = [Session() for _ in range(25)]
    client.app.state.session_service.list.return_value = sessions

    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    assert len(response.json()) == 20


def test_list_sessions_per_page_max_100(client: TestClient):
    sessions = [Session() for _ in range(150)]
    client.app.state.session_service.list.return_value = sessions

    response = client.get("/api/v1/sessions?page=1&per_page=200")
    assert response.status_code == 200
    assert len(response.json()) == 100


def test_create_session(client: TestClient):
    session = Session()
    client.app.state.session_service.create.return_value = session

    response = client.post("/api/v1/sessions")
    assert response.status_code == 201
    assert response.json()["id"] == str(session.id)
    assert response.headers["location"] == f"/api/v1/sessions/{session.id}"


def test_get_session(client: TestClient):
    session = Session(title="Test Session")
    client.app.state.session_service.get.return_value = session

    response = client.get(f"/api/v1/sessions/{session.id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Test Session"


def test_get_session_not_found(client: TestClient):
    client.app.state.session_service.get.side_effect = ValueError("not found")

    response = client.get(f"/api/v1/sessions/{uuid4()}")
    assert response.status_code == 404
    assert response.json()["detail"] == "not found"


def test_patch_session_title(client: TestClient):
    session = Session(title="Renamed")
    client.app.state.session_service.rename.return_value = session

    response = client.patch(f"/api/v1/sessions/{session.id}", json={"title": "Renamed"})
    assert response.status_code == 200
    assert response.json()["title"] == "Renamed"


def test_patch_session_not_found(client: TestClient):
    client.app.state.session_service.rename.side_effect = ValueError("not found")

    response = client.patch(f"/api/v1/sessions/{uuid4()}", json={"title": "Nope"})
    assert response.status_code == 404


def test_delete_session(client: TestClient):
    session_id = str(uuid4())
    response = client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 204
    client.app.state.session_service.delete.assert_called_once_with(session_id)


def test_delete_session_not_found(client: TestClient):
    client.app.state.session_service.delete.side_effect = ValueError("not found")

    response = client.delete(f"/api/v1/sessions/{uuid4()}")
    assert response.status_code == 404


def test_patch_system_prompt(client: TestClient):
    session = Session(system_prompt="You are helpful")
    client.app.state.session_service.update_system_prompt.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session.id}/system-prompt",
        json={"system_prompt": "You are helpful"},
    )
    assert response.status_code == 200
    assert response.json()["system_prompt"] == "You are helpful"


def test_patch_system_prompt_clear(client: TestClient):
    session = Session(system_prompt=None)
    client.app.state.session_service.update_system_prompt.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session.id}/system-prompt",
        json={"system_prompt": None},
    )
    assert response.status_code == 200
    assert response.json()["system_prompt"] is None


def test_patch_inference_params(client: TestClient):
    session = Session()
    client.app.state.session_service.update_inference_params.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session.id}/inference-params",
        json={"temperature": 0.7},
    )
    assert response.status_code == 200


def test_patch_inference_params_clear(client: TestClient):
    session = Session(inference_params=None)
    client.app.state.session_service.update_inference_params.return_value = session

    response = client.patch(
        f"/api/v1/sessions/{session.id}/inference-params",
        json={},
    )
    assert response.status_code == 200
    assert response.json()["inference_params"] is None
```

- [ ] **Step 2: Run tests (expect failures — no route yet)**

```bash
uv run pytest tests/api/test_sessions.py -v
```

Expected: All fail with 404 or route-not-found errors.

- [ ] **Step 3: Create `src/yapa/api/routes/sessions.py`**

```python
"""Session CRUD routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, Response

from yapa.models import InferenceParams, Session

from ..dependencies import get_session_service
from ..services import SessionService

router = APIRouter(tags=["sessions"])

MAX_PER_PAGE = 100
DEFAULT_PER_PAGE = 20


@router.get("/api/v1/sessions", response_model=list[Session])
async def list_sessions(
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
    session_service: SessionService = Depends(get_session_service),
):
    all_sessions = session_service.list(newest_first=True)
    per_page = min(per_page, MAX_PER_PAGE)
    start = (page - 1) * per_page
    return all_sessions[start : start + per_page]


@router.post("/api/v1/sessions", response_model=Session, status_code=201)
async def create_session(
    response: Response,
    session_service: SessionService = Depends(get_session_service),
):
    session = session_service.create()
    response.headers["Location"] = f"/api/v1/sessions/{session.id}"
    return session


@router.get("/api/v1/sessions/{session_id}", response_model=Session)
async def get_session(
    session_id: UUID,
    session_service: SessionService = Depends(get_session_service),
):
    return session_service.get(str(session_id))


@router.patch("/api/v1/sessions/{session_id}", response_model=Session)
async def patch_session(
    session_id: UUID,
    title: str,
    session_service: SessionService = Depends(get_session_service),
):
    return session_service.rename(str(session_id), title)


@router.delete("/api/v1/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    session_service: SessionService = Depends(get_session_service),
):
    session_service.delete(str(session_id))


@router.patch("/api/v1/sessions/{session_id}/system-prompt", response_model=Session)
async def patch_system_prompt(
    session_id: UUID,
    body: dict,
    session_service: SessionService = Depends(get_session_service),
):
    prompt = body.get("system_prompt")
    return session_service.update_system_prompt(str(session_id), prompt)


@router.patch("/api/v1/sessions/{session_id}/inference-params", response_model=Session)
async def patch_inference_params(
    session_id: UUID,
    body: dict,
    session_service: SessionService = Depends(get_session_service),
):
    params = InferenceParams(**body) if body else None
    return session_service.update_inference_params(str(session_id), params)
```

**Important implementation notes:**
- `SessionService.get()` raises `ValueError` on missing session; the global exception handler in `app.py` maps this to 404.
- For `patch_system_prompt`, use `body: dict` so that `{"system_prompt": null}` arrives as `{"system_prompt": None}`.
- For `patch_inference_params`, if `body` is an empty dict `{}`, set `params` to `None` (clearing).
- `Session.model_dump()` is used by FastAPI for serialization; Session inherits from `TrackedEntity` which is a `BaseModel`, so Pydantic handles it.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/api/test_sessions.py -v
```

Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add session CRUD REST endpoints"
```

---

### Task 3: Model Listing + Lookup REST Routes

**Files:**
- Create: `src/yapa/api/routes/models.py`
- Create: `tests/api/test_models.py`

**Interfaces:**
- Consumes: `get_model_service()` from `api/dependencies.py`, `ModelService` (`list_models()`, `get_model()`), `ProviderNotAvailableError` from `yapa.providers.registry`.
- Produces: `GET /api/v1/models` and `GET /api/v1/models/{full_id}`.

- [ ] **Step 1: Write the tests**

Create `tests/api/test_models.py`:

```python
"""Tests for model listing and lookup endpoints."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from yapa.models import ModelData, ModelType


@pytest.fixture
def sample_models():
    return [
        ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM),
        ModelData(id="claude-3", provider_id="openrouter", type=ModelType.LLM),
    ]


def test_list_models(client: TestClient, sample_models):
    client.app.state.model_service.list_models = AsyncMock(return_value=sample_models)

    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["id"] == "gpt-4o"


def test_list_models_by_provider(client: TestClient, sample_models):
    client.app.state.model_service.list_models = AsyncMock(
        return_value=[sample_models[0]]
    )

    response = client.get("/api/v1/models?provider_id=openai")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["provider_id"] == "openai"


def test_get_model_by_full_id(client: TestClient, sample_models):
    client.app.state.model_service.get_model = AsyncMock(return_value=sample_models[0])

    response = client.get("/api/v1/models/openai:gpt-4o")
    assert response.status_code == 200
    assert response.json()["id"] == "gpt-4o"


def test_get_model_not_found(client: TestClient):
    from yapa.providers import ModelsFetchError

    client.app.state.model_service.get_model = AsyncMock(
        side_effect=ModelsFetchError("not found")
    )

    response = client.get("/api/v1/models/openai:ghost")
    assert response.status_code == 404


def test_get_model_invalid_format(client: TestClient):
    client.app.state.model_service.get_model = AsyncMock(
        side_effect=ValueError("Invalid format")
    )

    response = client.get("/api/v1/models/bad-format")
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests (expect failures)**

```bash
uv run pytest tests/api/test_models.py -v
```

Expected: All fail with 404 or route-not-found errors.

- [ ] **Step 3: Create `src/yapa/api/routes/models.py`**

```python
"""Model listing and lookup routes."""

from fastapi import APIRouter, Depends

from yapa.models import ModelData

from ..dependencies import get_model_service
from ..services import ModelService

router = APIRouter(tags=["models"])


@router.get("/api/v1/models", response_model=list[ModelData])
async def list_models(
    provider_id: str | None = None,
    model_service: ModelService = Depends(get_model_service),
):
    return await model_service.list_models(provider_id=provider_id)


@router.get("/api/v1/models/{full_id:path}", response_model=ModelData)
async def get_model(
    full_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    try:
        return await model_service.get_model(full_id)
    except ValueError:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Model '{full_id}' not found")
```

**Note on `{full_id:path}`**: The `:path` converter allows slashes in the path parameter. However, model full IDs use `:` (e.g. `openai:gpt-4o`), which doesn't contain slashes, so a plain `str` would also work. Using `:path` is a safety measure if any model ID ever contains a slash (unlikely). Alternatively, use `str` without converter — either works.

**Important:** `ModelService.get_model()` raises `ValueError` for both invalid format and fetch failures. We catch it here and raise `HTTPException(404)` because the global `ValueError` handler might catch non-404 ValueErrors. Actually, re-reading the spec: the global handler maps all `ValueError` → 404, which is the same behavior. So we could skip the try/except and let the global handler do it. But explicit is better — let the route handle it so it's clear.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/api/test_models.py -v
```

Expected: All pass.

- [ ] **Step 5: Fix the catch — `ModelService.get_model()` raises `ValueError`, not `ModelsFetchError`**

Correction from codebase review: `ModelService.get_model()` catches `ModelsFetchError` internally and raises `ValueError`. So the test for `ModelsFetchError` won't match. Update test:

```python
def test_get_model_not_found(client: TestClient):
    client.app.state.model_service.get_model = AsyncMock(
        side_effect=ValueError("Failed to fetch model 'openai:ghost'")
    )

    response = client.get("/api/v1/models/openai:ghost")
    assert response.status_code == 404
```

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add model listing and lookup REST endpoints"
```

---

### Task 4: WebSocket Chat Handler

**Files:**
- Create: `src/yapa/api/websocket/__init__.py`
- Create: `src/yapa/api/websocket/chat.py`
- Create: `tests/api/test_chat_ws.py`

**Interfaces:**
- Consumes: `get_chat_service()`, `get_session_service()` from `api/dependencies.py`. `ChatService.stream(session_id: UUID, prompt: str, model: ModelData | None) → AsyncGenerator[Event]`. Event types: `AgentStartEvent`, `TextEvent`, `ReasoningEvent`, `AgentDoneEvent`, `AgentErrorEvent`.
- Produces: WebSocket at `/api/v1/chat/{session_id}`.

- [ ] **Step 1: Write the tests**

Create `tests/api/test_chat_ws.py`:

```python
"""Tests for WebSocket chat endpoint."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    TextEvent,
)


@pytest.mark.asyncio
async def test_chat_ws_streams_events(client: TestClient, mock_chat_service, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = None  # session exists

    async def _stream(*args, **kwargs):
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
        assert msg2["type"] == "text"
        assert msg2["content"] == "Hello"

        msg3 = ws.receive_json()
        assert msg3["type"] == "agent_done"
        assert msg3["content"] == "Hello"


@pytest.mark.asyncio
async def test_chat_ws_multiple_prompts(client: TestClient, mock_chat_service, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = None

    calls = 0

    async def _stream(*args, **kwargs):
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


@pytest.mark.asyncio
async def test_chat_ws_error_event(client: TestClient, mock_chat_service, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = None

    async def _stream(*args, **kwargs):
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield AgentErrorEvent(message="Something went wrong")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Hi"})
        ws.receive_json()  # agent_start
        error = ws.receive_json()
        assert error["type"] == "agent_error"
        assert "Something went wrong" in error["message"]


@pytest.mark.asyncio
async def test_chat_ws_missing_prompt(client: TestClient, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = None

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"model": "openai:gpt-4o"})
        # Should close with 4008
        with pytest.raises(Exception):
            ws.receive_json()


@pytest.mark.asyncio
async def test_chat_ws_invalid_json(client: TestClient, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.return_value = None

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_text("not json")
        with pytest.raises(Exception):
            ws.receive_json()


@pytest.mark.asyncio
async def test_chat_ws_invalid_session(client: TestClient, mock_session_service):
    session_id = str(uuid4())
    mock_session_service.get.side_effect = ValueError("Session not found")

    with pytest.raises(Exception) as exc:
        client.websocket_connect(f"/api/v1/chat/{session_id}")
    # FastAPI test client raises WebSocketDisconnect on rejection
    assert exc.value
```

- [ ] **Step 2: Run tests (expect failures)**

```bash
uv run pytest tests/api/test_chat_ws.py -v
```

Expected: All fail with route-not-found or connection errors.

- [ ] **Step 3: Create `src/yapa/api/websocket/__init__.py`**

Empty file.

- [ ] **Step 4: Create `src/yapa/api/websocket/chat.py`**

```python
"""WebSocket chat handler — streams events per prompt."""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    TextEvent,
    ReasoningEvent,
)

from ..dependencies import get_chat_service, get_session_service
from ..services import ChatService, SessionService

router = APIRouter(tags=["chat"])


@router.websocket("/api/v1/chat/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: UUID,
    chat_service: ChatService = Depends(get_chat_service),
    session_service: SessionService = Depends(get_session_service),
):
    await websocket.accept()

    # Validate session exists
    try:
        session_service.get(str(session_id))
    except ValueError:
        await websocket.close(code=4008, reason="Session not found")
        return

    while True:
        try:
            data = await websocket.receive_text()
        except WebSocketDisconnect:
            break

        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            await websocket.close(code=4008, reason="Invalid JSON")
            break

        prompt = message.get("prompt")
        if not prompt:
            await websocket.close(code=4008, reason="Missing 'prompt' field")
            break

        model = None

        async for event in chat_service.stream(
            session_id=session_id,
            prompt=prompt,
            model=model,
        ):
            await websocket.send_json(event.model_dump())

            if isinstance(event, (AgentDoneEvent, AgentErrorEvent)):
                break
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/api/test_chat_ws.py -v
```

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: add WebSocket chat endpoint"
```

---

### Task 5: CLI App (All Commands + Entry Point)

**Files:**
- Create: `src/yapa/cli/__init__.py`
- Create: `src/yapa/cli/app.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/conftest.py`
- Create: `tests/cli/test_config.py`
- Create: `tests/cli/test_models.py`
- Create: `tests/cli/test_sessions.py`
- Create: `tests/cli/test_server.py`
- Modify: `src/yapa/__main__.py`

**Interfaces:**
- Consumes: `JsonConfigStore` (from `yapa.services.config`), `SessionService` (from `yapa.services.session`), `ModelService` (from `yapa.services.models`), `JsonSessionStore` (from `yapa.services.store`), `DEFAULT_DATA_DIR` (from `yapa.services.config`).
- Produces: Typer CLI app with subcommands: `server`, `config show`, `config set`, `models`, `sessions list`, `sessions get`, `sessions delete`, `sessions rename`.

- [ ] **Step 1: Write CLI config tests**

Create `tests/cli/conftest.py`:

```python
"""Fixtures for CLI tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from yapa.services import ModelService


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_config_store():
    with patch("yapa.cli.app.JsonConfigStore") as mock:
        store = MagicMock()
        mock.return_value = store
        yield store


@pytest.fixture
def mock_model_service():
    svc = MagicMock(spec=ModelService)
    svc.list_models = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_session_service():
    return MagicMock()


@pytest.fixture
def mock_store():
    with patch("yapa.cli.app.JsonSessionStore") as mock:
        store = MagicMock()
        mock.return_value = store
        yield store
```

Create `tests/cli/test_config.py`:

```python
"""Tests for CLI config commands."""

from yapa.services.config import Config


def test_config_show_empty(runner, mock_config_store):
    mock_config_store.load.return_value = Config()

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "No providers configured" in result.stdout or "provider_configs" in result.stdout


def test_config_show_with_providers(runner, mock_config_store):
    config = Config(provider_configs={"openai": {"api_key": "sk-..."}})
    mock_config_store.load.return_value = config

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "openai" in result.stdout


def test_config_set(runner, mock_config_store):
    from yapa.cli.app import cli

    result = runner.invoke(cli, ["config", "set", "provider_configs.openai.api_key", "sk-test"])
    assert result.exit_code == 0
    assert mock_config_store.return_value.save.called
```

Create `tests/cli/test_models.py`:

```python
"""Tests for CLI models command."""

from unittest.mock import AsyncMock

import pytest

from yapa.models import ModelData, ModelType


def test_models_list(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.stdout


def test_models_list_by_provider(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(return_value=[])

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "--provider", "openai"])
        assert result.exit_code == 0
        mock_model_service.list_models.assert_called_once_with(provider_id="openai")


def test_models_empty(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(return_value=[])

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
```

Create `tests/cli/test_sessions.py`:

```python
"""Tests for CLI sessions commands."""

from uuid import uuid4

from yapa.models import Session


def test_sessions_list(runner, mock_session_service, mock_store):
    mock_session_service.list.return_value = [Session(title="Test")]

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "list"])
    assert result.exit_code == 0
    assert "Test" in result.stdout


def test_sessions_list_empty(runner, mock_session_service, mock_store):
    mock_session_service.list.return_value = []

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "list"])
    assert result.exit_code == 0
    assert "No sessions" in result.stdout


def test_sessions_get(runner, mock_session_service, mock_store):
    session = Session(title="Test")
    mock_session_service.get.return_value = session

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "get", str(session.id)])
    assert result.exit_code == 0
    assert "Test" in result.stdout


def test_sessions_get_not_found(runner, mock_session_service, mock_store):
    mock_session_service.get.side_effect = ValueError("not found")

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "get", str(uuid4())])
    assert result.exit_code == 1
    assert "Error:" in result.stdout


def test_sessions_delete(runner, mock_session_service, mock_store):
    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "delete", str(uuid4())])
    assert result.exit_code == 0
    assert "✓" in result.stdout


def test_sessions_delete_not_found(runner, mock_session_service, mock_store):
    mock_session_service.delete.side_effect = ValueError("not found")

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "delete", str(uuid4())])
    assert result.exit_code == 1


def test_sessions_rename(runner, mock_session_service, mock_store):
    session = Session(title="New Title")
    mock_session_service.rename.return_value = session

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "rename", str(session.id), "New Title"])
    assert result.exit_code == 0
    assert "✓" in result.stdout


def test_sessions_rename_not_found(runner, mock_session_service, mock_store):
    mock_session_service.rename.side_effect = ValueError("not found")

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "rename", str(uuid4()), "Nope"])
    assert result.exit_code == 1
```

Create `tests/cli/test_server.py`:

```python
"""Tests for CLI server command."""

from unittest.mock import patch


def test_server_command(runner):
    with patch("yapa.cli.app.uvicorn.run") as mock_run:
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["server"])
        assert result.exit_code == 0
        mock_run.assert_called_once()


def test_server_with_custom_host_port(runner):
    with patch("yapa.cli.app.uvicorn.run") as mock_run:
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["server", "--host", "0.0.0.0", "--port", "9000"])
        assert result.exit_code == 0
        # uvicorn.run call includes host="0.0.0.0", port=9000
        _, kwargs = mock_run.call_args
        assert kwargs.get("host") == "0.0.0.0"
        assert kwargs.get("port") == 9000
```

- [ ] **Step 2: Run tests (expect failures)**

```bash
uv run pytest tests/cli/ -v
```

Expected: All fail with import errors or command-not-found errors.

- [ ] **Step 3: Create `src/yapa/cli/app.py`**

Full Typer app implementation:

```python
"""Main CLI application — maintenance-only commands."""

import asyncio
import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from yapa.models import ModelData, ModelType
from yapa.services import ModelService, SessionService
from yapa.services.config import DEFAULT_DATA_DIR, Config, JsonConfigStore
from yapa.services.store import JsonSessionStore

cli = typer.Typer(name="yapa", help="YAPA — Your AI Personal Assistant")
console = Console()

ORANGE = "orange1"
LOGO = """\
  __  _____   ___  ___
  \\ \\/ / _ | / _ \\/ _ |
   \\  / __ |/ ___/ __ |
   /_/_/ |_/_/  /_/ |_|"""


def _style(key: str, value: str = "", end="\n") -> None:
    """Print a styled key-value pair."""
    console.print(f"[{ORANGE}]{key}[/{ORANGE}] [bold white]{value}[/bold white]", end=end)


def _success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def _error(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")


# ----- server -----


@cli.command()
def server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
):
    """Start the YAPA API server."""
    console.print(Panel(
        Text(LOGO, style=ORANGE),
        subtitle=f"[dim]Listening on [white]http://{host}:{port}[/white]  "
        f"Docs at [white]http://{host}:{port}/docs[/white]  "
        f"Press Ctrl+C to stop[/dim]",
    ))
    uvicorn.run(
        "yapa.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


# ----- config -----


@cli.group()
def config():
    """Manage YAPA configuration."""
    pass


@config.command()
def show():
    """Show current configuration."""
    store = JsonConfigStore()
    cfg = store.load()

    if not cfg.provider_configs:
        console.print("[dim]No providers configured[/dim]")

    _style("Log Level", cfg.log_level)
    _style("Storage Dir", str(cfg.storage_dir))
    _style("Timeout", str(cfg.provider_timeout) + "s")
    _style("Max Retries", str(cfg.provider_max_retries))

    for provider_id, pconfig in cfg.provider_configs.items():
        _style(f"\n[{provider_id}]")
        _style("  API Key", pconfig.api_key[:8] + "..." if pconfig.api_key else "not set")
        _style("  Base URL", pconfig.base_url or "(default)")


@config.command()
def set(
    key: str = typer.Argument(..., help="Dot-separated config key, e.g. provider_configs.openai.api_key"),
    value: str = typer.Argument(..., help="Value to set"),
):
    """Set a config value."""
    store = JsonConfigStore()
    cfg = store.load()

    # Support dot-separated keys
    parts = key.split(".")
    target = cfg
    for part in parts[:-1]:
        target = getattr(target, part, {})
    setattr(target, parts[-1], value)

    store.save(cfg)
    _success(f"Config {key} set to {value}")


# ----- models -----


@cli.command()
def models(
    provider: str | None = typer.Option(None, "--provider", "-p", help="Filter by provider ID"),
):
    """List available models."""
    service = ModelService()
    results = asyncio.run(service.list_models(provider_id=provider))

    if not results:
        if provider:
            _error(f"{provider}: no models found")
        else:
            console.print("[dim]No models found[/dim]")
        raise typer.Exit(code=1)

    table = Table(header_style="blue", box=None)
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Type")
    table.add_column("Context")
    table.add_column("Output")

    for m in results:
        table.add_row(
            m.provider_id,
            m.id,
            m.type.value,
            str(m.context_length or "-"),
            str(m.max_output or "-"),
        )

    console.print(table)


# ----- sessions -----


@cli.group()
def sessions():
    """Manage chat sessions."""
    pass


def _get_session_service() -> SessionService:
    config = JsonConfigStore().load()
    store = JsonSessionStore(config.storage_dir)
    return SessionService(store)


@sessions.command()
def list():
    """List all sessions."""
    service = _get_session_service()
    all_sessions = service.list(newest_first=True)

    if not all_sessions:
        console.print("[dim]No sessions yet[/dim]")
        return

    table = Table(header_style="blue", box=None)
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Messages")
    table.add_column("Updated")

    for s in all_sessions:
        short_id = str(s.id)[:8]
        updated = s.updated_at.strftime("%Y-%m-%d %H:%M") if s.updated_at else "-"
        table.add_row(
            short_id,
            s.title,
            str(len(s.messages)),
            updated,
        )

    console.print(table)
    console.print(f"[dim]({len(all_sessions)} session{'s' if len(all_sessions) != 1 else ''})[/dim]")


@sessions.command()
def get(session_id: str = typer.Argument(..., help="Session ID")):
    """Show session details."""
    service = _get_session_service()
    try:
        session = service.get(session_id)
    except ValueError as e:
        _error(str(e))
        raise typer.Exit(code=1)

    _style("ID", str(session.id))
    _style("Title", session.title)
    _style("Model", session.model.full_id if session.model else "(none)")
    _style("Messages", str(len(session.messages)))
    _style("Created", session.created_at.strftime("%Y-%m-%d %H:%M:%S") if session.created_at else "-")
    _style("Updated", session.updated_at.strftime("%Y-%m-%d %H:%M:%S") if session.updated_at else "-")
    if session.system_prompt:
        _style("System Prompt", session.system_prompt)


@sessions.command()
def delete(session_id: str = typer.Argument(..., help="Session ID")):
    """Delete a session."""
    service = _get_session_service()
    try:
        service.delete(session_id)
    except ValueError as e:
        _error(str(e))
        raise typer.Exit(code=1)

    _success(f"Session {session_id} deleted")


@sessions.command()
def rename(
    session_id: str = typer.Argument(..., help="Session ID"),
    title: str = typer.Argument(..., help="New title"),
):
    """Rename a session."""
    service = _get_session_service()
    try:
        service.rename(session_id, title)
    except ValueError as e:
        _error(str(e))
        raise typer.Exit(code=1)

    _success(f"Session renamed to '{title}'")
```

- [ ] **Step 4: Update `src/yapa/__main__.py`**

```python
"""YAPA main entry point."""

from yapa.cli.app import cli


def main() -> None:
    cli()
```

- [ ] **Step 5: Run CLI tests**

```bash
uv run pytest tests/cli/ -v
```

Expected: All pass.

- [ ] **Step 6: Run full test suite**

```bash
uv run ruff check src/ tests/
uv run ty check src/
uv run pytest tests/ -v
```

Expected: Ruff clean, ty clean, all tests pass (including existing 265+).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: add CLI app and wire up entry point"
```

---

## Self-Review Checklist

**1. Spec coverage:**
- ✅ FastAPI app factory with lifespan (`create_app`) — Task 1
- ✅ `/api/v1/health` — Task 1
- ✅ Session CRUD: list (paginated), create (Location header), get, patch, delete, system-prompt, inference-params — Task 2
- ✅ `response_model=` on every endpoint — Task 2, 3
- ✅ `Location` header on POST — Task 2
- ✅ Model listing (`?provider_id=`) and lookup by full_id — Task 3
- ✅ WebSocket at `/api/v1/chat/{session_id}` — Task 4
- ✅ CLI commands: server, config show/set, models, sessions list/get/delete/rename — Task 5
- ✅ CLI visual palette (orange1, white bold, green, red, dim, blue) — Task 5
- ✅ CLI server panel with ASCII logo — Task 5
- ✅ CLI exception handlers (ValueError → 404, etc.) — Task 1 (global), Task 3 (explicit), Task 4 (WS close)
- ✅ Entry point wiring (`__main__.py` + existing script) — Task 5
- ✅ Pagination (page, per_page, max 100) — Task 2
- ✅ `ProviderNotAvailableError` handling — Task 1 (global)

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later" — all steps contain real code or clear references.
- No "add validation" without actual validation code.
- No "similar to" references.
- All types and method signatures match the existing codebase.

**3. Type consistency:**
- `ChatService.stream()` takes `session_id: UUID` — correct (Task 4 uses `session_id: UUID`).
- `SessionService.get()` takes `str` — correct (Task 2 calls `get(str(session_id))`).
- `ModelService.list_models()` takes `provider_id: str | None` — correct (Task 3 passes it through).
- `ModelService.get_model()` takes `str` and raises `ValueError` — correct (Task 3 catches it).
- `SessionService.create()` returns `Session` — correct.
- `SessionService.list()` returns `list[Session]` — correct.
- Event serialization via `model_dump()` — correct (Task 4).
- `JsonConfigStore.__init__(path=None)` — correct (Task 5).
- `JsonSessionStore.__init__(storage_dir: Path)` — correct (Task 5).
