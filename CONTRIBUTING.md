# Contributing to YAPA

Thanks for your interest in contributing. This guide covers the basics.

## Prerequisites

- Python 3.13 or later
- The [uv](https://docs.astral.sh/uv/) package manager

## Getting started

```bash
git clone https://github.com/TheAdaptoid/yet-another-personal-assistant
cd yet-another-personal-assistant
uv sync
```

Configure at least one provider API key

```bash
yapa config set provider_configs.openai.api_key <OPENAI_API_KEY>
```

Run the test suite to verify your setup:

```bash
uv run pytest tests/ -v
```

## Code quality

All contributions must pass the full quality gate before merging:

```bash
uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v
```

| Tool | Purpose | Config |
|------|---------|--------|
| [ruff](https://docs.astral.sh/ruff/) | Linting and formatting | `ruff.toml` |
| [ty](https://docs.astral.sh/ty/) | Type checking | `ty.toml` |
| [pytest](https://docs.pytest.org/) | Testing | `pytest.ini` |

### Linting

```bash
uv run ruff check src/ tests/     # check
uv run ruff format src/ tests/    # auto-format
```

### Type checking

```bash
uv run ty check src/
```

### Testing

```bash
uv run pytest tests/ -v                # full suite
uv run pytest tests/services/ -v         # service tests only
uv run pytest tests/providers/ -v      # provider tests only
uv run pytest tests/models/ -v         # model tests only
uv run pytest tests/storage/ -v        # storage tests only
uv run pytest tests/api/ -v            # API tests only
uv run pytest tests/cli/ -v            # CLI tests only
```

Coverage is enforced at 80 percent (`--cov-fail-under=80` in
`pytest.ini`).

## Code conventions

- **Docstrings**: Required on all public classes, methods, and functions.
  This is enforced by ruff D-rules.
- **Line length**: 88 characters.
- **Imports**: Sorted by ruff (isort rules).
- **Async**: Services use `async` and `await`. Tests use
  `pytest-asyncio` with `asyncio_mode = auto`.
- **No comments**: Do not add inline comments unless they are truly
  necessary.
- **No generated artifacts**: Do not commit build or codegen output.

See [AGENTS.md](AGENTS.md) for the full architecture, conventions, and
reference import patterns.

## Project structure

```
src/yapa/
  __main__.py  # Entry point. Wires to the Typer CLI.
  logging.py   # File and console logging.
  api/         # FastAPI application with REST routes and WebSocket
               # chat. Uses application factory pattern.
  cli/         # Typer CLI commands (server, config, models, sessions).
  models/      # Pydantic v2 models (Session, Message, Event,
               # InferenceParams).
  providers/   # Provider ABC and implementations (OpenAI, OpenRouter,
               # LM Studio, Ollama).
  services/    # Business logic with protocol-based dependency
               # injection.
  storage/     # GenericJSONStore for JSON file persistence.
  tools/       # Tool abstractions (not yet integrated).
```

## Adding a new provider

1. Create a provider directory under `src/yapa/providers/`. Add a
   `provider.py` that implements the `InferenceProvider` ABC.
2. Add the provider class to `src/yapa/providers/__init__.py` in the
   `DEFAULT_PROVIDER_CLASSES` list.
3. Add a `ProviderConfig` entry to the `sample_config` fixture in
   `tests/providers/conftest.py`.
4. Add tests for init, model listing, and streaming in
   `tests/providers/`. See existing provider tests for patterns.
5. Use `AsyncMock` and `MagicMock` for the OpenAI client. If your
   provider uses a native HTTP API for model listing, patch
   `httpx.AsyncClient` instead.

## Adding tests for the API

Tests for API routes live in `tests/api/`. Each test file mirrors one
route module.

Key patterns:

- The `conftest.py` provides `mock_session_service`,
  `mock_model_service`, and `mock_chat_service` as `MagicMock` objects.
  These are injected into `app.state`.
- Use `TestClient` from `starlette.testclient` for HTTP tests.
- Use `client.websocket_connect` for WebSocket tests.
- Mock `ChatService.stream` to yield event objects.

## Adding tests for the CLI

Tests for CLI commands live in `tests/cli/`. Each test file mirrors one
command group.

Key patterns:

- The `conftest.py` provides a `CliRunner` instance.
- Patch module-level imports in `yapa.cli.app` to control service
  behavior.
- Use `runner.invoke()` to run CLI commands and check `exit_code` and
  `output`.

## Adding a new feature

1. Add or update source code in the appropriate module.
2. Add tests alongside the code. New runtime behavior requires tests.
3. Add docstrings to all public APIs.
4. Run the full quality gate before submitting.

See [AGENTS.md](AGENTS.md) for architecture details, design decisions,
and conventions.

## Submitting changes

1. Pull the latest from `master`.
2. Create a feature branch from `master`.
3. Make your changes following the conventions above.
4. Run the quality gate and make sure it passes.
5. Open a pull request targeting `development`. Changes are batched into `development` and merged into `master` on release.

The CI runs lint and test workflows on every push and pull request to
`master`. The release workflow builds and drafts a GitHub
release when you push a tag matching `v*`.