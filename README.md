# Yet Another Personal Assistant (YAPA)

YAPA is a Bring-Your-Own-Frontend AI backend. It has a Typer CLI for
maintenance tasks and a FastAPI server with REST and WebSocket endpoints.
A WebUI is in development. It uses pluggable inference providers for
model access.

## What it does today

- Lists models from configured providers. Optionally filter by provider or model type.
- Starts a local API server with REST and WebSocket endpoints.
- Runs chat sessions over a WebSocket connection. Messages stream as
  events in real time.
- Persists conversations as sessions in JSON files at
  `~/.yapa/storage/{uuid}.json`.
- Manages sessions from the CLI (list, get, rename, delete).
- Shows current configuration from the CLI.

Current providers:

- `openai`
- `openrouter`
- `lmstudio`
- `ollama`

## Installation

Install globally with `uv`:

```bash
uv tool install git+https://github.com/TheAdaptoid/yet-another-personal-assistant
```

After installation the `yapa` command is available from any directory:

```bash
yapa --help
```

To run without installing:

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

3. Configure at least one provider:

   ```bash
   yapa config set provider_configs.openai.api_key <OPENAI_API_KEY>
   yapa config set provider_configs.openrouter.api_key <OPENROUTER_API_KEY>
   ```

   You can also edit `~/.yapa/config.json` directly. Other settings use
   defaults. Change them with the CLI or env vars.

## Usage

From an installed copy:

```bash
yapa --help
```

From a development checkout:

```bash
uv run python -m yapa
```

List models:

```bash
yapa models
yapa models --provider openrouter
yapa models --model-type llm
```

Show current configuration:

```bash
yapa config show
yapa config set log_level DEBUG
yapa config set provider_configs.openai.api_key sk-...
```

Manage sessions:

```bash
yapa sessions list
yapa sessions get <session-id>
yapa sessions rename <session-id> "New Title"
yapa sessions delete <session-id>
```

Start the API server:

```bash
yapa server
yapa server --host 0.0.0.0 --port 9000
```

The server provides:

- `GET /api/v1/health` — health check
- `GET /api/v1/models` — list models (optional `provider_id`,
  `model_type` query params)
- `GET /api/v1/models/{full_id}` — get a single model
- `GET /api/v1/sessions` — list sessions (pagination via `page`,
  `per_page`)
- `POST /api/v1/sessions` — create a session
- `GET /api/v1/sessions/{id}` — get a session
- `PATCH /api/v1/sessions/{id}` — rename a session
- `DELETE /api/v1/sessions/{id}` — delete a session
- `PATCH /api/v1/sessions/{id}/system-prompt` — set or clear the system
  prompt with `{"system_prompt": "Be concise."}` or
  `{"system_prompt": null}`
- `PATCH /api/v1/sessions/{id}/inference-params` — set or clear inference
  params with `{"inference_params": {"temperature": 0.7}}` or
  `{"inference_params": null}`
- `WS /api/v1/chat/{session_id}` — streaming chat over WebSocket

For the two PATCH endpoints above, the wrapper field is required. Omitting it
(including sending `{}`) returns `422`. Use JSON `null` as the explicit clear
operation.

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
  __main__.py  # Entry point — wires to the Typer CLI
  logging.py   # File and console logging to ~/.yapa/logs/
  api/         # FastAPI application (REST routes + WebSocket chat)
  cli/         # Typer CLI commands (server, config, models, sessions)
  models/      # Pydantic v2 data models (Session, Message, Event,
               # InferenceParams)
  providers/   # Provider ABC + implementations (OpenAI, OpenRouter,
               # LM Studio, Ollama)
  services/    # Business logic with protocol-based dependency injection
  storage/     # GenericStore — JSON file persistence
  tools/       # Tool abstractions (not yet integrated into ChatService)
```
