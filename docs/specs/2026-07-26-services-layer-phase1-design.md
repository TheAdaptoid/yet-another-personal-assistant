# Services Layer Redesign — Phase 1

Date: 2026-07-26

## Overview

Redesign the services layer as a stable, interface-agnostic application core.
Services are modeled around domain concepts, not CLI workflows. Transport
(CLI, REST API, WebSockets) exists solely as a consumer of this layer.

### Guiding Principles

- Design for interfaces, not implementations.
- Keep transport separate from business logic.
- Keep services cohesive and narrowly focused.
- Favor composition over monolithic service classes.
- Build around domain concepts rather than UI workflows.
- Make streaming a first-class concept.
- Treat every service as reusable by any frontend.

## Architecture

```
          Clients
 ┌─────────────────────────┐
 │ CLI   Web UI   3rd-party│
 └───────────┬─────────────┘
             │
     Routing / Transport
 ┌─────────────────────────┐
 │ FastAPI   WebSockets    │
 └───────────┬─────────────┘
             │
       Services Layer
 ┌─────────────────────────┐
 │ ChatService             │
 │ SessionService          │
 │ ModelService            │
 │ ConfigStore (protocol)  │
 └───────────┬─────────────┘
             │
    Storage / Providers
 ┌─────────────────────────┐
 │ JsonSessionStore        │
 │ JsonConfigStore         │
 │ LLM Providers           │
 └─────────────────────────┘
```

## Service Definitions

### ChatService

Stateless orchestrator for a single model invocation. Does not own session
lifecycle, active session state, or message caching.

```python
class ChatService:
    def __init__(
        self,
        *,
        sessions: SessionService,
        models: ModelService,
    ) -> None: ...

    async def stream(
        self,
        session_id: UUID,
        prompt: str,
        model: ModelData | None = None,
    ) -> AsyncGenerator[Event, None]: ...
```

**Responsibilities:**
- Accept a session ID, prompt, and optional model.
- Retrieve the session via SessionService (history, system prompt, inference params).
- Construct model input (session system prompt + history + new prompt).
- Invoke the provider with session inference params and yield Event objects.
- Persist user and assistant messages via SessionService after completion.

**Stateless guarantees:**
- No `self._session_id`, `self._messages`, or `self._model` state.
- Each `stream()` call is independent.
- No `start()`, `switch_session()`, or `auto_title()` methods.
- No `close()` method — cancellation is via `asyncio.CancelledError`.

**System prompt and inference params are per-session:** `system_prompt` and
`inference_params` (temperature, max_tokens, top_p) live on the `Session`
model. Consumers set them once via `SessionService` methods and each
`stream()` call reads them from the session.

**Model fallback chain (when `model` is None):**
1. Check `session.model` on the persisted Session.
2. Raise `ValueError("No model specified")`.

**Event streaming contract:**

```
AgentStartEvent
  → (TextEvent | ReasoningEvent)*
  → AgentDoneEvent
```

Uses existing Event types from `yapa.models.event`. Provider emits
`StreamDelta` internally — ChatService maps to Event types.

### SessionService

CRUD for sessions + message appending. Does not invoke models or manage
active sessions.

```python
class SessionService:
    def __init__(self, store: SessionStore) -> None: ...

    def create(self) -> Session: ...
    def list(self, *, newest_first: bool = True) -> list[Session]: ...
    def get(self, session_id: str) -> Session: ...
    def rename(self, session_id: str, title: str) -> Session: ...
    def update_system_prompt(self, session_id: str, prompt: str | None) -> Session: ...
    def update_inference_params(self, session_id: str, params: InferenceParams | None) -> Session: ...
    def add_messages(self, session_id: str, messages: list[Message]) -> Session: ...
    def delete(self, session_id: str) -> None: ...
```

### ModelService

Thin wrapper around ProviderRegistry. Returns flat model lists — consumers
group by provider_id (stamped on each ModelData).

```python
class ModelService:
    def __init__(self, registry: ProviderRegistry | None = None) -> None: ...

    def get_provider(self, provider_id: str) -> InferenceProvider: ...
    def get_provider_by_model(self, model: ModelData) -> InferenceProvider: ...
    async def list_models(
        self,
        provider_id: str | None = None,
        model_type: ModelType | None = None,
    ) -> list[ModelData]: ...
    async def get_model(self, model_full_id: str) -> ModelData: ...
```

### ConfigStore Protocol

```python
class ConfigStore(Protocol):
    def load(self) -> Config: ...
    def save(self, config: Config) -> None: ...
```

### SessionStore Protocol

```python
class SessionStore(Protocol):
    def load(self, id: str) -> Session: ...
    def save(self, session: Session, *, overwrite: bool = False) -> None: ...
    def list(self) -> list[Session]: ...
    def delete(self, id: str) -> None: ...
```

## Event Model

Phase 1 uses a trimmed set of event types. Tool-related events (`TOOL_CALL`,
`TOOL_RESULT`, `TOOL_APPROVAL_REQUEST`, `TOOL_APPROVAL_RESPONSE`) are removed
and will return in Phase 2.

```python
class EventType(str, Enum):
    TEXT_CHUNK = "text_chunk"
    REASONING_CHUNK = "reasoning_chunk"
    AGENT_START = "agent_start"
    AGENT_DONE = "agent_done"
    AGENT_ERROR = "agent_error"


class Event(BaseModel):
    type: EventType
    source: EventSource = EventSource.SYSTEM
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TextEvent(Event):
    type: EventType = EventType.TEXT_CHUNK
    source: EventSource = EventSource.AGENT
    content: str


class ReasoningEvent(Event):
    type: EventType = EventType.REASONING_CHUNK
    source: EventSource = EventSource.AGENT
    content: str


class AgentStartEvent(Event):
    type: EventType = EventType.AGENT_START
    source: EventSource = EventSource.AGENT
    model_id: str


class AgentDoneEvent(Event):
    type: EventType = EventType.AGENT_DONE
    source: EventSource = EventSource.AGENT
    content: str          # full accumulated content
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class AgentErrorEvent(Event):
    type: EventType = EventType.AGENT_ERROR
    source: EventSource = EventSource.AGENT
    message: str
```

`AgentDoneEvent.content` carries the **full accumulated response**, not the
last token chunk. This allows ChatService to avoid buffering the complete
response internally — it can pass `AgentDoneEvent.content` directly to
`SessionService.add_messages()`.

## Data Model Changes

### Config

- Removed per-provider fields (`openrouter_api_key`, `lmstudio_base_url`, etc.).
- Replaced with `provider_configs: dict[str, ProviderConfig]`.
- Removed `default_model`.
- Removed `database_path` (no SQLite in Phase 1).
- Removed `UNSET` sentinel — uses `None` for unset API keys.

```python
class ProviderConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None

class Config(BaseModel):
    provider_configs: dict[str, ProviderConfig] = Field(default_factory=dict)
    storage_dir: Path = DEFAULT_STORAGE_DIR
    log_level: str = "INFO"
    provider_timeout: int = 120
    provider_max_retries: int = 2
```

### Session

- Added `model: ModelData | None` field — stores the last model used in
  this session. Set after each successful stream.
- Added `system_prompt: str | None` field — per-session system prompt.
  Read by ChatService on each `stream()` call. Set via
  `SessionService.update_system_prompt()`.
- Added `inference_params: InferenceParams | None` field — per-session
  inference parameters (temperature, max_tokens, top_p). Read by
  ChatService on each `stream()` call. Set via
  `SessionService.update_inference_params()`.

### JsonConfigStore

Replaces module-level `get_config()` / `save_config()` pattern. Owns its
own path and cache. No global singleton.

```python
class JsonConfigStore:
    def __init__(self, path: Path | None = None) -> None: ...

    def load(self) -> Config:
        """Read config file + apply env overrides."""
        load_dotenv()
        config_data: dict[str, Any] = {}
        if self._path.exists():
            with open(self._path) as f:
                config_data = json.load(f) or {}
        for key, value in ENV_OVERRIDES.items():
            if (env_val := os.environ.get(value)) is not None and env_val != "":
                config_data[key] = env_val
        self._cache = Config(**config_data)
        return self._cache

    def save(self, config: Config) -> None:
        self._cache = config
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            f.write(config.model_dump_json(indent=2))
```

### Config Lifecycle

No global singleton. The app entry point creates a `JsonConfigStore` once
and injects it into anything that needs config:

```python
# CLI entry point
def main() -> None:
    config_store = JsonConfigStore()
    session_store = JsonSessionStore(config_store.load().storage_dir)
    session_service = SessionService(store=session_store)
    chat_service = ChatService(sessions=session_service, models=ModelService())
    # ... pass chat_service into CLI handler
```

In a FastAPI transport, the store is created at startup and injected via
`Depends()`. Services that need config receive it as a constructor
parameter.

### ProviderConfig Default URLs

`ProviderConfig` has no default base URL. Each provider class defines its
own `DEFAULT_BASE_URL` constant and falls back to it when
`config.provider_configs[id].base_url` is `None`.

```python
class OpenAIIP:
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: Config) -> None:
        pc = config.provider_configs.get("openai", ProviderConfig())
        self.base_url = pc.base_url or self.DEFAULT_BASE_URL
        self.api_key = pc.api_key
```

Config stays provider-agnostic — no per-provider field imports.

### JsonSessionStore

Wraps existing `GenericStore[Session]` behind the `SessionStore` protocol.
No behavioral changes — services never import `GenericStore` directly.

## Files

### Create

| File | Contents |
|---|---|
| `src/yapa/services/__init__.py` | Re-exports |
| `src/yapa/services/exceptions.py` | Service exceptions |
| `src/yapa/services/chat.py` | ChatService |
| `src/yapa/services/session.py` | SessionService |
| `src/yapa/services/models.py` | ModelService |
| `src/yapa/services/config.py` | Config, ProviderConfig, ConfigStore, JsonConfigStore |
| `src/yapa/services/store.py` | SessionStore protocol, JsonSessionStore |

### Remove

| File | Reason |
|---|---|
| `src/yapa/services/conversation.py` | Replaced by chat.py + session.py |
| `src/yapa/services/provider.py` | Replaced by models.py |
| `src/yapa/config.py` | Replaced by services/config.py |

### Update

| File | Change |
|---|---|
| `src/yapa/models/event.py` | Trim to Phase 1 events (remove tool events, add `model_id`/`content`/`finish_reason`/`usage` fields) |
| `src/yapa/models/session.py` | Add `model`, `system_prompt`, `inference_params` fields |
| `src/yapa/models/__init__.py` | Update exports to match trimmed event.py |
| `src/yapa/providers/*` | Read config via `config.provider_configs[id]` instead of `config.{id}_api_key` |

## Service Dependencies

```
ChatService
  ├── SessionService → SessionStore ← JsonSessionStore
  ├── ModelService    → ProviderRegistry
  └── (none — no config dep)

SessionService
  └── SessionStore ← JsonSessionStore

ModelService
  └── ProviderRegistry
```

ChatService does not depend on ConfigStore. Model resolution and session
retrieval handle model/session identification without global config.

## Extension Points (Phase 2)

```python
class ChatService:
    def __init__(
        self,
        *,
        sessions: SessionService,
        models: ModelService,
        tools: ToolService | None = None,  # Phase 2: optional
    ) -> None: ...
```

Tool loop is a private method called from within `stream()`. New services
are `None` by default — existing consumers don't change.

## End-to-End Flow

```python
# Create session
session = session_service.create()

# Configure session
session_service.update_system_prompt(str(session.id), "You are a helpful assistant.")
session_service.update_inference_params(
    str(session.id),
    InferenceParams(temperature=0.7, max_tokens=4096),
)

# Chat
async for event in chat.stream(session_id=session.id, prompt="Hello"):
    match event:
        case AgentStartEvent():
            show_header(event.model_id)
        case TextEvent():
            buffer += event.content
        case ReasoningEvent():
            buffer += event.content
        case AgentDoneEvent():
            display(buffer)

# Next message — independent call, same session config applies
async for event in chat.stream(session_id=session.id, prompt="Tell me more"):
    ...

# List sessions
sessions = session_service.list()
```

## Future Phases

- Phase 2: ToolService, ApprovalService, tool loop in ChatService.
- Phase 3: MemoryService, FileService, MCP integration, extract
  Orchestrator if ChatService grows large enough to warrant it.
