# Yet Another Personal Assistant (YAPA)

YAPA is a terminal-first assistant with a Typer CLI and pluggable inference
providers.

## What it does today

- Lists available models from configured providers
- Runs an interactive chat loop with streaming responses
- Supports provider selection by model ID via a provider manager

Current providers:

- `openrouter`
- `lmstudio`

## Development setup

1. Clone the repository.
2. Install dependencies:

   ```bash
   uv sync
   uv sync --dev
   ```

3. Configure environment variables (example):

   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   LMSTUDIO_API_KEY=your_lmstudio_api_key_or_placeholder
   LMSTUDIO_BASE_URL=http://localhost:1234/v1
   YAPA_DEFAULT_MODEL=openrouter/free
   YAPA_DATA_DIR=~/.yapa
   YAPA_LOG_LEVEL=INFO
   ```

Configuration can also be persisted to `~/.yapa/config.json`.

## Usage

Run the CLI:

```bash
uv run python -m yapa
```

List models:

```bash
uv run python -m yapa models
uv run python -m yapa models openrouter
```

Start interactive chat:

```bash
uv run python -m yapa invoke
uv run python -m yapa invoke openrouter/free
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
  cli/         # Typer commands
  models/      # Message and inference data models
  providers/   # Provider abstraction + implementations
  config.py    # Config loading and persistence
  logging.py   # File and console logging helpers
```
