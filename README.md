# Yet Another Personal Assistant (YAPA)

YAPA is a terminal-first assistant with a Typer CLI, SQLite-backed session
persistence, and pluggable inference providers.

## What it does today

- Lists available models from configured providers (grouped by vendor prefix)
- Runs an interactive chat loop with streaming responses
- Supports provider selection by model ID via a provider manager
- Persists conversations as sessions in a local SQLite database (`~/.yapa/yapa.db`)
- Manages sessions (list, rename, delete) via CLI commands

Current providers:

- `openrouter`
- `lmstudio`

## Development setup

1. Clone the repository.
2. Install dependencies:

   ```bash
   uv sync
   ```

3. Configure environment variables (example):

   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key
   LMSTUDIO_API_KEY=your_lmstudio_api_key_or_placeholder
   YAPA_DEFAULT_MODEL=openrouter/free
   YAPA_LOG_LEVEL=INFO
   ```

Configuration can also be persisted to `~/.yapa/config.json`.

## Usage

Run the CLI:

```bash
uv run python -m yapa
```

List models (grouped by vendor):

```bash
uv run python -m yapa models
uv run python -m yapa models openrouter
```

Start interactive chat (auto-creates a session):

```bash
uv run python -m yapa chat
uv run python -m yapa chat --model openrouter/free
```

Resume a previous session:

```bash
uv run python -m yapa chat --session <session-id> --model openrouter/free
```

Manage sessions:

```bash
uv run python -m yapa sessions list
uv run python -m yapa sessions rename <session-id> "New Title"
uv run python -m yapa sessions delete <session-id>
```

Type `exit` or `quit` to leave the chat loop.

## Quality checks

```bash
uv run ruff check src/ tests/
uv run ty check src/
uv run pytest tests/ -v
```

Recommended local gate:

```bash
uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v
```

## Project layout

```text
src/yapa/
  cli/         # Typer commands (app, chat, models, sessions)
  database/    # SQLite models, engine, repositories (sqlmodel)
  models/      # Message and inference data models
  providers/   # Provider abstraction + implementations
  config.py    # Config loading and persistence
  logging.py   # File and console logging helpers
```
