# Contributing to YAPA

Thanks for your interest in contributing! This guide covers the basics.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

## Getting started

```bash
git clone https://github.com/TheAdaptoid/yet-another-personal-assistant
cd yet-another-personal-assistant
uv sync
```

Configure at least one provider API key (or use a `.env` file):

```bash
export OPENROUTER_API_KEY=your_key_here
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
| [ruff](https://docs.astral.sh/ruff/) | Linting + formatting | `ruff.toml` |
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
uv run pytest tests/ -v                # full suite (≥80% coverage required)
uv run pytest tests/test_services/ -v  # service tests only
uv run pytest tests/providers/ -v      # provider tests only
uv run pytest tests/models/ -v         # model tests only
uv run pytest tests/storage/ -v        # storage tests only
```

Coverage is enforced at 80% (`--cov-fail-under=80` in `pytest.ini`).

## Code conventions

- **Docstrings**: Required on all public classes, methods, and functions (enforced by ruff D-rules).
- **Line length**: 88 characters.
- **Imports**: Sorted by ruff (isort rules).
- **Async**: Services use `async/await`. Tests use `pytest-asyncio` with `asyncio_mode = auto`.
- **No comments**: Avoid inline comments unless truly necessary.
- **No generated artifacts**: Do not commit build or codegen output.

See [AGENTS.md](AGENTS.md) for the full architecture, conventions, and reference import patterns.

## Project structure

```
src/yapa/
  __main__.py  # Entry point (currently bare — no CLI wired up)
  logging.py   # File ± console logging
  models/      # Pydantic v2 data models (Session, Message, Event, InferenceParams)
  providers/   # Provider ABC + implementations (OpenAI, OpenRouter, LM Studio, Ollama)
  services/    # UI-agnostic business logic with protocol-based DI
  storage/     # GenericStore — JSON file persistence
  tools/       # Tool abstractions (pre-existing, not yet integrated)
```

## Adding a new provider

1. Create a provider directory under `src/yapa/providers/` with a `provider.py` implementing the `InferenceProvider` ABC.
2. Add the provider class to `src/yapa/providers/__init__.py`'s `DEFAULT_PROVIDER_CLASSES` list.
3. Add `ProviderConfig` entries in `tests/providers/conftest.py`'s `sample_config` fixture.
4. Add tests covering init, model listing, and streaming in `tests/providers/`.
5. Use `AsyncMock`/`MagicMock` for the OpenAI client — see existing provider tests for patterns.

## Adding a new feature

1. Add or update source code in the appropriate module.
2. Add tests alongside the code. New runtime behavior requires tests.
3. Ensure docstrings are present on all public APIs.
4. Run the full quality gate before submitting.

See [AGENTS.md](AGENTS.md) for architecture details, design decisions, and conventions.

## Submitting changes

1. Pull the latest from `master`.
2. Create a feature branch from `master`.
3. Make your changes following the conventions above.
4. Run the quality gate and ensure it passes.
5. Open a pull request targeting `development`. PRs are batched into `development` before being merged to `master`.
