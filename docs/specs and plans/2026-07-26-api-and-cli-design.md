# API Layer & Minimal CLI — Design

Date: 2026-07-26

## Overview

Add a FastAPI-based REST + WebSocket API and a minimal maintenance CLI to YAPA. The API exposes the existing service layer (SessionService, ModelService, ChatService) over HTTP/WS. The CLI is a thin Typer utility for config management, model listing, session inspection, and server startup — it imports services directly and does not talk to the API.

### Guiding Principles

- Transport code goes in `api/` and `cli/`, not in `services/`.
- CLI is a maintenance utility only — no chat interaction.
- The API is the primary way to consume ChatService (streaming).
- Services are created during FastAPI lifespan and injected via `app.state` + `Depends()`.

## Architecture

```
__main__.py  ──→  cli/app.py (Typer)
                     ├── server  →  uvicorn → api/app.py (FastAPI factory)
                     ├── config  →  JsonConfigStore
                     ├── models  →  ModelService
                     └── sessions → SessionService

api/
  app.py              # create_app() factory, lifespan → app.state
  dependencies.py     # get_session_service, get_model_service, get_chat_service
  routes/
    health.py         # GET /health
    sessions.py       # Session CRUD (list, create, get, rename, delete, system-prompt, inference-params)
    models.py         # Model listing (all, by provider)
  websocket/
    chat.py           # WS /chat/{session_id}

cli/
  app.py              # Typer app, subcommand definitions
```

### Entry Point

`src/yapa/__main__.py` provides `main()`, referenced by the `yapa` console script in `pyproject.toml`. It calls the Typer app.

## REST API

All routes are prefixed with `/api/v1`. The prefix is configurable via app settings.

### Health

| Method | Path | Status | Response |
|--------|------|--------|----------|
| GET | `/api/v1/health` | 200 | `{"status": "ok"}` |

### Sessions

All session routes return the `Session` model serialized as JSON.

| Method | Path | Status | Body / Notes |
|--------|------|--------|-------------|
| GET | `/api/v1/sessions` | 200 | Returns `list[Session]`, newest first. Supports pagination: `?page=1&per_page=20` |
| POST | `/api/v1/sessions` | 201 | Returns new `Session`. Includes `Location` header |
| GET | `/api/v1/sessions/{id}` | 200 / 404 | — |
| PATCH | `/api/v1/sessions/{id}` | 200 / 404 | `{"title": "..."}` |
| DELETE | `/api/v1/sessions/{id}` | 204 / 404 | — |
| PATCH | `/api/v1/sessions/{id}/system-prompt` | 200 / 404 | `{"system_prompt": "..."}` or `{"system_prompt": null}` to clear |
| PATCH | `/api/v1/sessions/{id}/inference-params` | 200 / 404 | Partial `InferenceParams` or `{}` to clear |

Every endpoint declares an explicit `response_model` on the route decorator (e.g. `@router.get("/api/v1/sessions", response_model=list[Session])`). This ensures OpenAPI docs reflect the exact response shape.

`POST /api/v1/sessions` includes a `Location: /api/v1/sessions/{id}` header pointing to the newly created resource.

`ModelData` is a frozen Pydantic model. The API returns it as plain JSON (no special handling needed). On PATCH requests, unknown fields are ignored (extra="forbid" on ModelData is enforced by the service layer, not the API).

### Models

| Method | Path | Query | Response |
|--------|------|-------|----------|
| GET | `/api/v1/models` | `?provider_id=str` (optional) | `200 list[ModelData]` |
| GET | `/api/v1/models/{full_id}` | — | `200 ModelData` / 404 |

When `provider_id` is omitted on `/models`, returns models from all available providers. Calls `ModelService.list_models()`.

`GET /models/{full_id}` resolves a fully-qualified model ID (e.g. `openai:gpt-4o`). Calls `ModelService.get_model(full_id)` which splits on `:` to identify the provider. Returns 404 if the provider is unavailable or the model is not found.

### Pagination

`GET /api/v1/sessions` supports offset-based pagination via query parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `page` | 1 | Page number (1-indexed) |
| `per_page` | 20 | Items per page (max 100) |

The response body returns only the items for the requested page. Clients infer total pages from the response array length:
- If the response has fewer items than `per_page`, the current page is the last.

No `meta` envelope — the items are returned directly as `list[Session]`. Third-party UIs that need pagination metadata can derive it from the `Content-Range` or a future `X-Total-Count` header if needed.

### Error Responses

All errors return JSON with a `detail` key (FastAPI default). The app registers exception handlers for:

- `ValueError` (raised by SessionService on missing sessions) → 404
- `ProviderNotAvailableError` → 404
- `ChatError` → 500
- Unhandled → 500

## WebSocket Protocol

### Connection

```
WS /api/v1/chat/{session_id}
```

### Client → Server Messages

Sent as single JSON messages:

```json
{"prompt": "Hello", "model": "openai:gpt-4o"}
```

`model` is optional. If omitted, the session's stored model is used. If neither is set, the server closes with an error.

### Server → Client Messages

Each event is a single JSON message corresponding to the Event model hierarchy:

```json
{"type": "agent_start",     "model_id": "openai:gpt-4o"}
{"type": "text",            "content": "Hello"}
{"type": "reasoning",       "content": "thinking..."}
{"type": "agent_done",      "content": "Hello!", "finish_reason": "stop", "usage": {...}}
{"type": "agent_error",     "message": "Something went wrong"}
```

Events are produced by calling `Event.model_dump()`. The dispatcher reads `type` from the event and routes accordingly.

### Flow

1. Client connects to `WS /api/v1/chat/{session_id}`.
2. Server validates the session exists, sends `agent_start`.
3. Client sends a `prompt` message.
4. Server calls `chat_service.stream(session_id, prompt, model)` and yields events.
5. After `agent_done` or `agent_error`, the server waits for the next prompt.
6. Client closes the connection when done.

The server gracefully handles invalid JSON, missing prompts, and unknown session IDs (closes with 4008 close code + descriptive message).

## CLI

### Commands

```
yapa server [--host HOST] [--port PORT] [--reload]
yapa config show
yapa config set <key> <value>
yapa models [--provider TEXT]
yapa sessions list
yapa sessions get <id>
yapa sessions delete <id>
yapa sessions rename <id> <title>
```

### Implementation

Each command constructs the necessary service directly:

- **server**: Calls `uvicorn.run("yapa.api.app:create_app", ..., factory=True)`.
- **config**: Instantiates `JsonConfigStore` directly. `show` prints the config as YAML or JSON. `set` modifies the in-memory config and calls `save()`.
- **models**: Instantiates `ModelService()` (uses default `ProviderRegistry`). Calls `await list_models(provider_id)`. Uses Rich for table display.
- **sessions**: Instantiates `SessionService(JsonSessionStore(...))`. Calls CRUD methods. Uses Rich table for list.

### Startup

```python
def main() -> None:
    cli()
```

`cli` is a `typer.Typer()` instance. The `server` command is a subcommand of this app.

### Visual Design

All CLI output uses Rich with a consistent semantic palette:

| Element | Style | Example |
|---------|-------|---------|
| Labels / keys | `orange` (Rich `bold orange1`) | `Model:`, `Provider:` |
| Values / IDs | `white bold` | `openai:gpt-4o` |
| Success | `green` | `✓ Session renamed` |
| Errors | `red` (no bold) | `Error: session not found` |
| Counts / muted | `dim` | `(3 sessions)` |
| Table headers | `blue` | |
| Section dividers | `dim` rules | `──────` |

**Model listing** (`yapa models`): Flat Rich Table with columns: `Provider`, `Model`, `Type`, `Context`, `Output`. Grouping by vendor is dropped in favor of sortability.

```
Provider       Model              Type    Context  Output
─────────────────────────────────────────────────────────
openrouter     claude-3-opus      LLM     200K     8K
openrouter     claude-3-sonnet    LLM     200K     8K
openai         gpt-4o             LLM     128K     16K
```

**Session listing** (`yapa sessions list`): Rich Table with columns: session ID (shortened), title, message count, last updated.

**Server banner** (`yapa server`): Rich Panel with the ASCII logo in `orange1`, server URL, and docs link:

```
┌──────────────────────────────────────────────────────┐
│    __  _____   ___  ___                              │
│    \ \/ / _ | / _ \/ _ |                             │
│     \  / __ |/ ___/ __ |                             │
│     /_/_/ |_/_/  /_/ |_|                             │
│                                                      │
│   Listening on http://127.0.0.1:8000                 │
│   Docs at      http://127.0.0.1:8000/docs            │
│                                                      │
│   Press Ctrl+C to stop                               │
└──────────────────────────────────────────────────────┘
```

**Config display** (`yapa config show`): Styled key-value pairs with orange labels and white values, one per line.

**Empty states**:
- No sessions: `No sessions yet` in dim text
- No models from a provider: `<provider>: no models found` in red
- Config with no providers configured: shows `No providers configured` in dim

**Error formatting**: Errors always start with `Error:` in red (no bold), followed by the message in white. Never raw tracebacks.

**Success formatting**: Destructive/state-changing commands show a green check prefix: `✓ Session renamed`, `✓ Session deleted`.

## Dependencies

Add to `pyproject.toml`:

```
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
```

These are core dependencies (always installed). The `standard` extras include `websockets` for WebSocket support.

## File Manifest

| File | Purpose |
|------|---------|
| `src/yapa/__main__.py` | Entry point, calls `cli.app.cli` |
| `src/yapa/cli/app.py` | Typer app with all subcommands |
| `src/yapa/api/app.py` | `create_app()` factory, lifespan, exception handlers |
| `src/yapa/api/dependencies.py` | FastAPI `Depends()` callables |
| `src/yapa/api/routes/__init__.py` | Empty |
| `src/yapa/api/routes/health.py` | `GET /health` handler |
| `src/yapa/api/routes/sessions.py` | Session CRUD handlers |
| `src/yapa/api/routes/models.py` | Model listing + single model lookup handlers |
| `src/yapa/api/websocket/__init__.py` | Empty |
| `src/yapa/api/websocket/chat.py` | `WS /api/v1/chat/{session_id}` handler |
| `tests/api/__init__.py` | Empty |
| `tests/api/conftest.py` | Fixtures: mocked services, test client |
| `tests/api/test_health.py` | Health endpoint tests |
| `tests/api/test_sessions.py` | Session CRUD endpoint tests |
| `tests/api/test_models.py` | Model listing + single model lookup endpoint tests |
| `tests/api/test_chat_ws.py` | WebSocket chat tests |
| `tests/cli/__init__.py` | Empty |
| `tests/cli/conftest.py` | Fixtures: mocked services, CliRunner |
| `tests/cli/test_config.py` | Config command tests |
| `tests/cli/test_models.py` | Models command tests |
| `tests/cli/test_sessions.py` | Sessions command tests |
| `tests/cli/test_server.py` | Server command test (smoke) |

## Testing Strategy

- **REST tests**: Use `TestClient(app)` where `app` is created by `create_app()`. Mock `SessionService`, `ModelService` at the `app.state` level by overriding dependencies.
- **WebSocket tests**: Use `TestClient` with `.websocket_connect()`. Mock `ChatService.stream()` to yield predefined events.
- **CLI tests**: Use `CliRunner`. Mock `JsonConfigStore`, `SessionService`, `ModelService` via `unittest.mock.patch`.
- Tests go in `tests/api/` and `tests/cli/`, following existing patterns (autouse logger patches, conftest fixtures).

## Out of Scope

- Authentication / API keys (future).
- CORS configuration (add when a web UI consumer exists).
- OpenAPI doc customization (FastAPI auto-generates docs; customize later).
- Rate limiting, request validation beyond Pydantic.
- Graceful shutdown / connection draining for WebSocket.
- CLI chat interaction (CLI is maintenance-only).
