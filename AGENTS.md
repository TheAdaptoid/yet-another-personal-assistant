# YAPA — Developer Guide

## Architecture (current state)

The current codebase is a Typer CLI app with provider abstractions for model
discovery and streaming inference.

```
src/yapa/
├── __main__.py      # Entry point (`python -m yapa`)
├── config.py        # Config loading/caching + JSON persistence
├── logging.py       # File + optional console logging
├── cli/
│   ├── __init__.py
│   └── cli.py       # Typer commands: `models`, `invoke`
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
uv run python -m yapa models                       # list models
uv run python -m yapa invoke                       # interactive chat loop
uv run pytest tests/ -v                            # full test suite
uv run pytest tests/providers/ -v                  # provider tests
uv run pytest tests/providers/test_manager.py -v   # single test file
uv run ruff check src/ tests/                      # lint
uv run ty check src/                               # type check
```

Recommended local gate:
`uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`

## Testing Notes

- `pytest.ini` sets `asyncio_mode = auto`.
- Coverage is always on (`--cov=src`).
- Provider tests use lightweight mocks (`AsyncMock`, `MagicMock`,
  `PropertyMock`) and `SimpleNamespace` test payloads.
- `tests/providers/conftest.py` patches `yapa.providers.base.get_logger`
  automatically to avoid writing real log files during tests.

## Provider Error Contract

Provider-layer custom exceptions live in `src/yapa/providers/exceptions.py`:

| Exception | Raised by |
|---|---|
| `ModelsFetchError` | `InferenceProvider.get_models()` on provider model-list failures |
| `ModelInvocationError` | `InferenceProvider.invoke_model()` on streaming invocation failures |

`ProviderManager.get_provider_by_model()` catches `ModelsFetchError` per provider
and continues to the next provider.

## Conventions

- **Package root**: `src/yapa/`.
- **Import style**:
  - `from yapa.config import Config, get_config`
  - `from yapa.logging import get_logger`
  - `from yapa.providers import ProviderManager`
  - `from yapa.models import UserMessage, StreamDelta, InferenceParams`
- **Config file**: `~/.yapa/config.json`.
- **Logs directory**: `~/.yapa/logs/{YYYY-MM-DD}/`.
- **Docstrings**: required (ruff docstring rules enabled).
- **Line length**: 88.
- **Python**: 3.13+.
- **No generated artifacts**: do not commit build/codegen output.

## Don't

- Don't bypass lint/type rules by weakening `ruff.toml` or `ty.toml`.
- Don't add config/model behavior without tests when it changes runtime behavior.
