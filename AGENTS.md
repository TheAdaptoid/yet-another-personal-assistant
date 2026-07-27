# YAPA — Developer Guide

## Architecture

```
src/yapa/
├── __main__.py      # Entry point (currently bare — no CLI wired up)
├── logging.py       # File ± console logging to ~/.yapa/logs/{date}/{name}.log
├── models/          # Pydantic v2 models (Session, Message, Event, InferenceParams, etc.)
├── providers/       # InferenceProvider ABC + implementations (OpenAI, OpenRouter, LM Studio, Ollama)
├── services/        # UI-agnostic business logic with protocol-based DI
│   ├── config.py    # Config, ProviderConfig, ConfigStore protocol, JsonConfigStore
│   ├── store.py     # SessionStore protocol, JsonSessionStore (wraps GenericStore)
│   ├── session.py   # SessionService — session CRUD + message appending
│   ├── models.py    # ModelService — wraps ProviderRegistry for model discovery
│   ├── chat.py      # ChatService — stateless, one stream() call per invocation
│   └── exceptions.py# ChatError
├── storage/         # GenericStore — JSON file persistence
└── tools/           # Tool abstractions (pre-existing, not yet integrated into ChatService)
```

**Key design decisions:**
- Services depend on protocol abstractions (`ConfigStore`, `SessionStore`), not concrete stores.
- `ChatService` is fully stateless — no `start()`, no `switch_session()`, no `close()`. Each `stream()` call is independent.
- `Session` stores `model`, `system_prompt`, and `inference_params` — not on the service call.
- Storage is JSON file-based via `GenericStore` (not SQLite). Sessions saved per-file in `~/.yapa/storage/`.
- Config is NOT a singleton — injected via constructor. `JsonConfigStore` reads `~/.yapa/config.json` + env var overrides.
- Providers define their own `DEFAULT_BASE_URL` when `ProviderConfig.base_url` is `None`.

## Key Commands

```bash
uv sync                                       # install dev dependencies
uv run python -m yapa                         # run (entry point is currently empty — no CLI)
uv run pytest tests/ -v                       # full test suite (enforces ≥80% coverage)
uv run pytest tests/test_services/ -v          # services tests only
uv run pytest tests/providers/ -v              # provider tests only
uv run pytest tests/models/ -v                # model tests only
uv run pytest tests/storage/ -v               # storage tests only
uv run pytest tests/ -k "test_name"           # single test filter
uv run ruff check src/ tests/                 # lint (select: F, E, I, C90, D)
uv run ty check src/                          # type check
```

Recommended local gate:
`uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`

## Config

Config file: `~/.yapa/config.json`  
Storage dir: `~/.yapa/storage/`  
Logs dir: `~/.yapa/logs/{YYYY-MM-DD}/`

Env var overrides (defined in `services/config.py:ENV_OVERRIDES`):
`YAPA_LOG_LEVEL`, `YAPA_STORAGE_DIR`, `YAPA_PROVIDER_TIMEOUT`, `YAPA_PROVIDER_MAX_RETRIES`

Provider configs stored under `provider_configs` dict keyed by provider ID (e.g. `openai`, `openrouter`).
Provider-specific API keys read from env: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.

## Testing

- `pytest.ini`: `asyncio_mode = auto`, `--cov=src --cov-fail-under=80`.
- Services tests (`tests/test_services/conftest.py`): autouse patch `yapa.services.models.get_logger`.
- Provider tests (`tests/providers/conftest.py`): autouse patch `yapa.providers.base.get_logger`, provides `sample_config` fixture with `ProviderConfig` for all 4 providers, `mock_openai_client`, `sample_llm_model`, `sample_other_model`, `sample_messages`.
- All provider tests use lightweight mocks (`AsyncMock`, `MagicMock`).
- Storage tests rely on `GenericStore` reading/writing JSON files (use `tmp_path`).

## Provider Error Contract

Defined in `src/yapa/providers/exceptions.py`:

| Exception | Raised by |
|---|---|
| `ModelsFetchError` | `InferenceProvider.list_models()` / `get_model()` on provider failures |
| `ModelInvocationError` | `InferenceProvider.stream_chat()` / `static_chat()` on streaming failures |
| `ModelTypeError` | `_pre_invoke_check()` if model type is not LLM |

`ModelService.list_models()` catches `ModelsFetchError` per provider and continues.

## Events (Phase 1 Contract)

`AgentStartEvent → (TextEvent | ReasoningEvent)* → AgentDoneEvent | AgentErrorEvent`

Event types in `yapa.models.event`:
- `AgentStartEvent` — emitted at start, includes `model_id`
- `TextEvent` — streaming content chunk
- `ReasoningEvent` — streaming reasoning/thinking chunk
- `AgentDoneEvent` — final event with `content`, `finish_reason`, optional `usage`
- `AgentErrorEvent` — emitted on unrecoverable error, includes `message`

## Conventions

- **Package root**: `src/yapa/`.
- **Import style**: absolute from package root — `from yapa.services import ChatService, SessionService, Config`.
- **Python**: 3.13+.
- **Line length**: 88 (ruff + ty enforce).
- **Docstrings**: required (ruff D rules, ignored in tests).
- **No generated artifacts**: do not commit build/codegen output.
- **No weakening config**: do not bypass lint/type rules in `ruff.toml` or `ty.toml`.
- **Tests required**: new config/model behavior must include tests.

## Don't

- Don't circumvent `ChatService` to call providers directly for chat operations — always go through the service layer.
- Don't add `cli/` module or Typer routing without also wiring up `__main__.py`.
- Don't add SQLite/sqlmodel dependencies — Phase 2+, not current scope.
- Don't add config/model behavior without tests when it changes runtime behavior.
- Don't bypass lint/type rules by weakening `ruff.toml` or `ty.toml`.
