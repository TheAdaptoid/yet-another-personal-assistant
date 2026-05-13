# YAPA — Developer Guide

## Architecture (actual vs docs)

`docs/architecture.md` describes an aspirational future state (agent loop, tasks, chat WebSocket). The **current** codebase is narrower: a FastAPI session CRUD service with JSON file storage and an in-memory variant for tests.

```
src/yapa/
├── core/       # FastAPI service (repositories + services + routers + app factory)
├── tui/        # Empty — not yet implemented
└── shared/     # Models, config, logging
```

## Key Commands

```bash
uv run pytest tests/ -v                    # full test suite
uv run pytest tests/core/repositories/ -v  # single package
uv run pytest tests/core/routers/test_sessions.py::TestGetSession -v  # single class
uv run ruff check src/                     # lint (excludes tests/)
uv run ty check src/                       # type check (excludes tests/)
uv run python -m yapa.core                 # start the API server
```

Order matters: `ruff check src/ && ty check src/ && uv run pytest tests/ -v` is the full pre‑commit gate.

## Testing quirks

- `pytest.ini` sets `asyncio_mode = auto` — no `@pytest.mark.asyncio` needed for *tests* (you still need it when the test function is async).
- Coverage is always on (`--cov=src`). Look for `Coverage.py warning` in output, not just test counts.
- Repo/service mocks use `create_autospec(SessionRepository, instance=True)` then assign `AsyncMock()` to each async method. Never use `MagicMock` without a spec.
- Router tests override `get_session_service` via `app.dependency_overrides` and use `TestClient(app, raise_server_exceptions=False)`.
- Fixture pattern: `dummy_logger` (NullHandler), `test_config` (uses `tmp_path` for file tests), mock repo, service/router fixtures.

## Repository error contract

Repository methods never return sentinel values. They raise:

| Exception | Raised by |
|---|---|
| `SessionNotFoundError` | `load()`, `delete()` when session is missing |
| `SessionSaveError` | `save()` on I/O failure |
| `SessionLoadError` | `load()`, `load_all()` on I/O or parse failure |
| `SessionDeleteError` | `delete()` on I/O failure |

The service layer catches `SessionNotFoundError` → returns `None`/`False`. All other exceptions propagate to the router → HTTP 500.

## Conventions

- **Package root**: `src/yapa/`. Import as `from yapa.shared import Config`, `from yapa.core.repositories import SessionRepository`.
- **Config file**: `~/.yapa/config.json` (see `src/yapa/shared/config.py`).
- **Logging**: `from yapa.shared import get_logger; logger = get_logger("core", console=True)` writes to `~/.yapa/logs/{date}/core.log`.
- **Docstrings**: Required (ruff D100–D107). Use line comments sparingly.
- **Line length**: 88 (enforced by ruff).
- **Python**: 3.13+.
- **No generated code, migrations, codegen, or build artifacts.** Just the lockfile (`uv.lock`).

## Don't

- Don't edit `ruff.toml` or `ty.toml` to bypass lint/type rules — fix the code instead.
- Don't use JSON for config storage without writing a test for it first.
