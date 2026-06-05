# YAPA — Developer Guide

## Architecture (current state)

The current codebase is a Typer CLI app with SQLite session persistence
(sqlmodel) and provider abstractions for model discovery and streaming
inference.

```
src/yapa/
├── __main__.py      # Entry point (`python -m yapa`)
├── config.py        # Config loading/caching + JSON persistence
├── logging.py       # File + optional console logging
├── cli/
│   ├── __init__.py
│   ├── app.py       # Typer routing
│   ├── chat.py      # run_conversation() — session-aware interactive loop
│   ├── models.py    # list_models(), grouped display
│   └── sessions.py  # list/rename/delete session commands
├── database/
│   ├── __init__.py
│   ├── engine.py    # sync SQLite engine, get_session(), init_db()
│   ├── models.py    # SessionTable, MessageTable, BaseTable
│   └── repositories.py  # SessionRepository CRUD
├── models/
│   ├── inference.py # InferenceParams, ModelData, StreamDelta
│   └── message.py   # User/System/Assistant message models
└── providers/
    ├── base.py      # InferenceProvider base class
    ├── exceptions.py
    ├── lmstudio.py
    ├── manager.py   # ProviderManager
    └── openrouter.py
```

There is currently no `src/yapa/core/`, `src/yapa/shared/`, or `src/yapa/tui/`.

## Key Commands

```bash
uv run python -m yapa                              # run CLI
uv run python -m yapa models                       # list models (grouped)
uv run python -m yapa chat                         # interactive chat loop
uv run python -m yapa chat --model <id>            # chat with a specific model
uv run python -m yapa chat --session <id>          # resume a session
uv run python -m yapa sessions list                # list sessions
uv run python -m yapa sessions rename <id> <title> # rename a session
uv run python -m yapa sessions delete <id>         # delete a session
uv run pytest tests/ -v                            # full test suite
uv run pytest tests/cli/ -v                        # CLI tests
uv run pytest tests/database/ -v                   # database tests
uv run pytest tests/providers/ -v                  # provider tests
uv run ruff check src/ tests/                      # lint
uv run ty check src/                               # type check
```

Recommended local gate:
`uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`

## Testing Notes

- `pytest.ini` sets `asyncio_mode = auto`.
- Coverage is always on (`--cov=src`).
- All three test suites use in-memory SQLite via an autouse `patch_get_engine`
  fixture in each directory's `conftest.py`. The engine is disposed after each
  test to avoid `ResourceWarning`.
- Provider tests use lightweight mocks (`AsyncMock`, `MagicMock`,
  `PropertyMock`) and `SimpleNamespace` test payloads.
- `tests/providers/conftest.py` patches `yapa.providers.base.get_logger`
  automatically to avoid writing real log files during tests.
- `tests/cli/conftest.py` provides a `seeded_session` fixture (one user message
  and one assistant message) for session command tests.
- Chat tests (`tests/cli/test_chat.py`) inject mock `Console`, `ProviderManager`,
  and `Config` into `run_conversation()` via its keyword-only parameters to
  avoid interactive I/O and real provider calls.
- CLI command routing is tested with `typer.testing.CliRunner`.

## Provider Error Contract

Provider-layer custom exceptions live in `src/yapa/providers/exceptions.py`:

| Exception | Raised by |
|---|---|
| `ModelsFetchError` | `InferenceProvider.get_models()` on provider model-list failures |
| `ModelInvocationError` | `InferenceProvider.invoke_model()` on streaming invocation failures |

`ProviderManager.get_provider_by_model()` catches `ModelsFetchError` per provider
and continues to the next provider.

## Database Schema

- **`SessionTable`**: `id` (str UUID4 PK), `title` (str, default `"New Session"`),
  `created_at`, `updated_at`. Has a one-to-many `messages` relationship with
  cascade delete.
- **`MessageTable`**: `id` (str UUID4 PK), `role` (str), `content` (str),
  `model` (str | None, set only on assistant messages), `session_id` (str FK).
  No `timestamp` or `name` fields.
- Both inherit from `BaseTable` which defines the common `id`, `created_at`,
  `updated_at` columns.
- `MessageTable` has `from_pydantic()` / `to_pydantic()` converters for working
  with the Pydantic message types in `yapa.models.message`.

## Session Commands

- `chat` auto-creates a session when `--session` is not provided.
- Resuming requires an explicit `--model` (no auto-detection from history).
- Full session IDs are shown in CLI output.
- `SessionRepository.get()` and `delete()` raise `ValueError` on missing session.

## Model Display

Models are grouped by vendor prefix (the part before `/` in the model ID):

```
Models for provider 'openrouter' (8 total):

  anthropic (2):
    claude-3-opus
    claude-3-sonnet

  openai (3):
    gpt-3.5-turbo
    gpt-4-turbo
    gpt-4o

  other (1):
    some-unprefixed-model
```

Models without a `/` are grouped under `"other"`. This approach generalizes to
any provider without data loss.

## Conventions

- **Package root**: `src/yapa/`.
- **Import style**:
  - `from yapa.config import Config, get_config`
  - `from yapa.logging import get_logger`
  - `from yapa.database import SessionRepository, SessionTable, MessageTable`
  - `from yapa.providers import ProviderManager`
  - `from yapa.models import UserMessage, StreamDelta, InferenceParams`
- **Config file**: `~/.yapa/config.json`.
- **Database file**: `~/.yapa/yapa.db` (derived from `Config.database_path`).
- **Logs directory**: `~/.yapa/logs/{YYYY-MM-DD}/`.
- **Docstrings**: required (ruff docstring rules enabled).
- **Line length**: 88.
- **Python**: 3.13+.
- **No generated artifacts**: do not commit build/codegen output.

## Don't

- Don't bypass lint/type rules by weakening `ruff.toml` or `ty.toml`.
- Don't add config/model behavior without tests when it changes runtime behavior.
