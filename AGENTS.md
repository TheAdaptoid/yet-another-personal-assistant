# YAPA — Developer Guide

## Commands

```bash
uv sync                                         # install dev deps
uvx yapa --help                                 # run without installing
uv run python -m yapa                           # run from checkout (wires to CLI)
uv run pytest tests/ -v                         # full suite (≥80% coverage)
uv run pytest tests/api/ -v                     # API tests only
uv run pytest tests/cli/ -v                     # CLI tests only
uv run pytest tests/providers/ -v               # provider tests only
uv run pytest tests/models/ -v                  # model tests only
uv run pytest tests/storage/ -v                 # storage tests only
uv run pytest tests/services/ -v           # services tests only
uv run pytest tests/ -k "test_name"             # single test
uv run ruff check src/ tests/                   # lint (F, E, I, C90, D)
uv run ruff format src/ tests/                  # auto-format
uv run ty check src/                            # type check (excludes tests/)
```

Gate: `uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`

## Architecture

```
src/yapa/
  __main__.py   # Entry point → delegates to yapa.cli.app:cli()
  logging.py    # File + console logging to ~/.yapa/logs/{YYYY-MM-DD}/{name}.log
  api/          # FastAPI app (factory pattern), REST routes, WebSocket chat
  cli/          # Typer commands: server, config, models, sessions
  models/       # Pydantic v2: Session, Message (discriminated union), Event, InferenceParams
  providers/    # InferenceProvider ABC → OpenAICompatibleProvider ABC
  services/     # Business logic with protocol-based DI (ConfigStore, SessionStore)
  storage/      # GenericJSONStore — JSON file persistence, one file per entity
  tools/        # Tool ABC + ToolRegistry (not integrated into ChatService yet)
```

Key facts:
- `services/__init__.py` uses lazy `__getattr__` imports to avoid circular deps
- `ChatService` is stateless. Each `stream()` call is independent. Do not reuse.
- `Session` stores `model: ModelData`, `system_prompt`, `inference_params`, and `messages`
- Config is NOT a singleton. Inject via constructor. `JsonConfigStore` caches in memory.
- Provider API keys go in `~/.yapa/config.json` only. Set with `yapa config set provider_configs.{id}.api_key <key>`.

## Providers

```
InferenceProvider (ABC, template method pattern)
  └── OpenAICompatibleProvider (ABC, shared OpenAI client logic)
       ├── OpenAIIP             (hardcoded _MODEL_METADATA dict)
       ├── OpenRouterProvider   (custom httpx model listing for richer metadata)
       ├── LMStudioIP           (custom httpx to /api/v1/models for native data)
       └── OllamaIP             (no overrides, pure OpenAI-compatible)
```

- All registered in `providers/__init__.py:DEFAULT_PROVIDER_CLASSES` — add new providers there
- `ProviderRegistry` instantiates all classes, tracks failures per-provider without failing the whole registry
- Provider constructors accept `config: Config` and read `config.provider_configs[provider_id]`
- Default base URLs are in each provider file (e.g. `OpenRouterProvider.DEFAULT_BASE_URL`)
- Public methods wrap private `_impl` methods with logging + error conversion to `ModelsFetchError`/`ModelInvocationError`/`ModelTypeError`

## Config

`~/.yapa/config.json`. Env var overrides (only these, no provider keys):

| Env var | Config key |
|---|---|
| `YAPA_LOG_LEVEL` | `log_level` |
| `YAPA_STORAGE_DIR` | `storage_dir` |
| `YAPA_PROVIDER_TIMEOUT` | `provider_timeout` |
| `YAPA_PROVIDER_MAX_RETRIES` | `provider_max_retries` |
| `YAPA_API_PREFIX` | `api_prefix` |

Provider API keys must be set via CLI or written directly into the JSON
file. They are NOT read from environment variables.

## Events (Phase 1 Contract)

```
AgentStartEvent → (TextEvent | ReasoningEvent)* → AgentDoneEvent | AgentErrorEvent
```

All event types in `yapa.models.event`. Sent as JSON over WebSocket and
internal `AsyncGenerator[Event]` in `ChatService.stream()`.

## API module

- `api/app.py:create_app(config=None)` → FastAPI application factory
- Routes registered under `config.api_prefix` (default `/api/v1`)
- Exception mapping: `ValueError` → 404, `ProviderNotAvailableError` → 404, `ChatError` → 500
- Services injected via `app.state` (no `Depends` for the lifespan)
- WebSocket at `/chat/{session_id}` — accepts JSON `{"prompt": ..., "model": ...}`, streams events
- Tests: `TestClient` with mocked services in `app.state` (`tests/api/conftest.py`)

## CLI module

- `cli/app.py` — single `typer.Typer()` app with sub-typers for `config` and `sessions`
- Commands: `server`, `config show`, `config set`, `models`, `sessions list|get|delete|rename`
- Tests: `CliRunner` with patched `yapa.cli.app.*` module-level imports (`tests/cli/conftest.py`)

## Testing quirks

- `tests/providers/conftest.py` — autouse patch `yapa.providers.base.get_logger`
- `tests/services/conftest.py` — autouse patch `yapa.services.models.get_logger`
- All provider tests use lightweight mocks (`AsyncMock`, `MagicMock`, `SimpleNamespace`)
- Storage tests use `tmp_path` for isolated filesystem
- ruff ignores D100-D104, D107 in test files; docstrings required elsewhere
- ty skips `tests/` entirely

## Commits

- Start commit messages with one of these prefixes: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`

## Documentation

- All specs and design docs should be in `docs/specs/`
- All implementation plans should be in `docs/plans/`
- All user-facing documentation should be in `docs/references/`