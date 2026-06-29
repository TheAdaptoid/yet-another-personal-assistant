# Yet Another Personal Assistant (YAPA)

YAPA is a terminal-first assistant with a Typer CLI, SQLite-backed session
persistence, and pluggable inference providers.

## What it does today

- Lists available models from configured providers (grouped by vendor prefix, filtered to LLMs only)
- Sets a default model via `--set` and scoped provider lookup via `--provider`
- Runs an interactive chat loop with streaming responses
- Auto-titles new sessions based on the first user message using the default LLM
- Supports slash commands for in-chat model/session switching and help
- Persists conversations as sessions in a local SQLite database (`~/.yapa/yapa.db`)
- Manages sessions (list, rename with manual or auto-generated title, delete, purge) via CLI commands
- Resumes the most recent session via `--continue`

Current providers:

- `openrouter`
- `lmstudio`
- `ollama`

## Installation

Install globally with `uv` (recommended):

```bash
uv tool install git+https://github.com/TheAdaptoid/yet-another-personal-assistant
```

After installation the `yapa` command is available from any directory:

```bash
yapa --help
```

To run without installing (temporary, uses a cached ephemeral environment):

```bash
uvx yapa --help
```

To upgrade an existing installation:

```bash
uv tool upgrade yapa
```

## Development setup

1. Clone the repository.
2. Install dependencies:

   ```bash
   uv sync
   ```

3. Configure environment variables (example, or add to a `.env` file):

   ```bash
    OPENROUTER_API_KEY=your_openrouter_api_key
    LMSTUDIO_API_KEY=your_lmstudio_api_key_or_placeholder
    OLLAMA_API_KEY=your_ollama_api_key_or_placeholder
    YAPA_DEFAULT_MODEL=openrouter:openrouter/free
    YAPA_LOG_LEVEL=INFO
   ```

Configuration can also be persisted to `~/.yapa/config.json`. Environment variables
override file values.

## Usage

From an installed copy:

```bash
yapa --help
```

From a development checkout:

```bash
uv run python -m yapa
```

List models (grouped by vendor):

```bash
yapa models
yapa models --provider openrouter
```

Set the default model (validates the model exists):

```bash
yapa models --set openrouter/free
yapa models --provider openrouter --set openai/gpt-4o
```

Start interactive chat (auto-creates a session):

```bash
yapa chat
yapa chat --model openrouter/free
```

Continue the most recent session:

```bash
yapa chat --continue
```

Resume a previous session:

```bash
yapa chat --session <session-id> --model openrouter/free
```

Manage sessions:

```bash
yapa sessions list
yapa sessions rename <session-id> "New Title"
yapa sessions rename <session-id> --auto    # generate title via LLM
yapa sessions delete <session-id>
yapa sessions delete --purge                # delete empty sessions
```

> **Note:** All examples below use the installed `yapa` command. Replace with
> `uv run python -m yapa` when running from a development checkout.

Within a chat session the following slash commands are available:

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/exit` | Exit the chat session |
| `/model <model-id>` | Switch to a different model |
| `/session <session-id>` | Switch to a different session |
| `/sessions` | List all sessions |

## Quality checks

```bash
uv run ruff check src/ tests/
uv run ty check src/
uv run pytest tests/ -v  # enforces ≥80% coverage
```

Recommended local gate:

```bash
uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v
```

## Project layout

```text
src/yapa/
  cli/         # Typer commands (app, chat, models, sessions)
  models/      # Message, inference, and session data models
  providers/   # Provider abstraction + implementations (OpenRouter, LM Studio)
  services/    # UI-agnostic business logic (conversation, session, provider services)
  storage/     # JSON-based object persistence
  config.py    # Config loading and persistence
  logging.py   # File and console logging helpers
```
