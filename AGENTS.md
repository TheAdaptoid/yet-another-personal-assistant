# YAPA — Developer Guide

## Project Structure

```
src/yapa/
├── core/      # yapa-core (FastAPI service)
├── tui/       # Terminal UI (Textual)
└── shared/    # Models, config, logging, tools
```

## Key Commands

```bash
# Setup (already done, for reference)
uv sync

# Run tests
pytest tests/ -v

# Lint & Type check (excludes tests/)
ruff check src/
ty check src/

# Run core (not yet implemented)
python -m yapa.core

# Run TUI (not yet implemented)
python -m yapa.tui
```

## Important Conventions

- **Package location**: Code lives in `src/yapa/`, not root. Imports are `from yapa.shared import ...`
- **Config file**: `~/.yapa/config.json` — see `src/yapa/shared/config.py`
- **Log location**: `~/.yapa/logs/{YYYY-MM-DD}/{component}.log`
- **Logging**: Use `from yapa.shared import get_logger; logger = get_logger("core", console=True)`
- **Models**: Located in `src/yapa/shared/models/` — use Pydantic with discriminated unions
- **Test naming**: `tests/*/test_{module}.py`
- **Docstrings**: Required (ruff rule D100-D107). Write them. Use line comments sparingly — code should be self-explanatory where possible.
- **Line length**: 88 chars (ruff.toml)
- **Python**: 3.13+

## State

- Phase 1 in progress: shared models, config, logging done
- 61 tests passing
- No runtime entry points yet (core/main.py, tui/main.py)

## Don't

- Don't edit `ruff.toml` to bypass lint rules — fix the code instead
- Don't use JSON for config storage without writing a test for it first