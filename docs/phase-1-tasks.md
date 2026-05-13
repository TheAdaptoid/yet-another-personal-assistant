# Phase 1 Tasks

**Goal:** TUI-Core communication with a basic chat loop using WebSockets and no tool calls.

---

## Task List

### 1. Project Setup

- [x] Create `pyproject.toml` with dependencies:
  - `fastapi`, `uvicorn[standard]`
  - `textual`
  - `openai`, `httpx`
  - `pydantic`, `orjson`
  - `pytest`, `pytest-asyncio` (dev)
- [x] Create basic package structure (`core/`, `tui/`, `shared/`)
- [x] Verify imports work: `python -c "from yapa.core import main; from yapa.tui import app; from yapa.shared import models"`

### 2. Shared Models

- [ ] Create `shared/models/__init__.py` — export all models
- [ ] Create `shared/models/message.py` — `Message` pydantic model
- [ ] Create `shared/models/session.py` — `Session` pydantic model
- [ ] Create `shared/models/task.py` — `Task`, `SubTask`, `TaskPriority`, `TaskStatus` models

### 3. Configuration & Logging

- [ ] Create `shared/config.py` — config management (OpenRouter API key, model, data dir)
- [ ] Create `shared/logging.py` — logging setup to `~/.yapa/logs/`
- [ ] Verify: logs appear in `~/.yapa/logs/` when core runs

### 4. LLM Client

- [ ] Create `shared/llm.py` — OpenAI client wrapper for OpenRouter
- [ ] Support: `chat.completions.create()` with streaming
- [ ] Add environment variable config: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`
- [ ] Test: call API with a simple prompt

### 5. Session Management

- [ ] Create `core/storage.py` — JSON file read/write for sessions
- [ ] `save_session(session: Session)` — write to `~/.yapa/sessions/{id}.json`
- [ ] `load_session(session_id: str) -> Session | None`
- [ ] `list_sessions() -> list[SessionSummary]`

### 6. API: Health Endpoint

- [ ] Create `core/main.py` — FastAPI app
- [ ] `GET /health` — returns `{status, version, active_sessions}`
- [ ] Add CORS for local development

### 7. API: Session CRUD

- [ ] `GET /sessions` — list all sessions
- [ ] `POST /sessions` — create new session, return ID
- [ ] `GET /sessions/{id}` — get session with messages
- [ ] `DELETE /sessions/{id}` — delete session

### 8. WebSocket Chat

- [ ] `WS /chat/{session_id}` — chat endpoint
- [ ] On connect: load session, send current messages
- [ ] On message: append user message, call LLM, stream response, append assistant message
- [ ] On disconnect: save session
- [ ] No tool calls in this phase — pure chat loop

### 9. Agent Loop (Basic)

- [ ] Create `core/agent.py` — simple loop:
  1. Take user message
  2. Build prompt from session history
  3. Call LLM with messages
  4. Stream response back to client
  5. Save final response to session

### 10. TUI Setup

- [ ] Create `tui/main.py` — Textual entry point
- [ ] Create `tui/app.py` — main `TUI` app class
- [ ] Set up basic layout: header + main area

### 11. TUI Chat Panel

- [ ] Create `tui/widgets/chat_panel.py`
- [ ] Display message history (scrollable)
- [ ] Input field at bottom
- [ ] Send on Enter

### 12. TUI WebSocket Connection

- [ ] Connect to `ws://localhost:8000/chat/{session_id}`
- [ ] Send user messages
- [ ] Receive and display deltas in real-time
- [ ] Handle disconnect/reconnect

### 13. TUI Session Management

- [ ] On TUI start: create or reuse session
- [ ] Store session ID in `~/.yapa/last_session`
- [ ] On TUI launch: load last session if exists

### 14. Integration Test

- [ ] Run core: `python -m yapa.core`
- [ ] Run TUI: `python -m yapa.tui`
- [ ] Send message: "Hello, who are you?"
- [ ] Verify response streams back
- [ ] Close TUI, reopen, verify chat history persists

### 15. Documentation

- [ ] Update `README.md` with setup and running instructions
- [ ] Document required environment variables

---

## Dependencies Summary

```
# Runtime
fastapi
uvicorn[standard]
textual
openai
httpx
pydantic
orjson

# Dev
pytest
pytest-asyncio
```

---

## Not Included in Phase 1

- Task creation / management (UI + API)
- Tool calls (file read/write, curl)
- Task sidebar in TUI
- LM Studio integration
- Automatic failover models
- Background task dispatching
- Session pause/resume
- Health endpoint beyond basic status