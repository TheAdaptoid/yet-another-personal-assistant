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

Run the CLI to verify your setup:

```bash
uv run python -m yapa --help
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
uv run pytest tests/ -v           # full suite (≥80% coverage required)
uv run pytest tests/cli/ -v       # CLI tests only
uv run pytest tests/services/ -v  # service tests only
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
  cli/         # Typer commands (app, chat, models, sessions)
  database/    # SQLite models, engine, repositories (sqlmodel)
  models/      # Message, inference, and session data models
  providers/   # Provider abstraction + implementations
  services/    # UI-agnostic business logic
  config.py    # Config loading and persistence
  logging.py   # File and console logging helpers
```

## Adding a new provider

1. Create a protocol class in `src/yapa/providers/protocols/` implementing `ModelFetchProtocol` and/or `InferenceProtocol`.
2. Create a concrete provider class in `src/yapa/providers/concretes/` that composes the protocol(s).
3. Register the provider in `src/yapa/providers/__init__.py`'s `DEFAULT_PROVIDERS` list.
4. Add tests in `tests/providers/concretes/` and `tests/providers/protocols/`.

## Adding a new feature

1. Add or update source code in the appropriate module.
2. Add tests alongside the code. New runtime behavior requires tests.
3. Ensure docstrings are present on all public APIs.
4. Run the full quality gate before submitting.

## Submitting changes

1. Pull the latest from `master`.
2. Create a feature branch from `master`.
3. Make your changes following the conventions above.
4. Run the quality gate and ensure it passes.
5. Open a pull request targeting `development`. PRs are batched into `development` before being merged to `master`.
