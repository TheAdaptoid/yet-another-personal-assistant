# Services Layer Redesign — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stateful, config-singleton services layer with stateless, protocol-driven services backed by JSON file storage.

**Architecture:** Config access moves from a module-level `get_config()` singleton to explicit dependency injection via a `ConfigStore` protocol. `ChatService` is fully stateless — each `stream()` call is independent, reading session config from the `Session` model. `SessionService` owns CRUD behind a `SessionStore` protocol. `ModelService` wraps `ProviderRegistry` with flat model lists.

**Tech Stack:** Python 3.13+, Pydantic v2, pytest (asyncio_mode=auto), ruff, ty (type checker)

## Global Constraints

- Python 3.13+ only.
- Line length: 88.
- Package root: `src/yapa/`.
- Import style: absolute imports only (`from yapa.config import Config`).
- Docstrings required (ruff docstring rules enabled).
- No generated artifacts (no build/codegen output committed).
- Coverage floor: 80% (`--cov-fail-under=80`).
- Config is NOT a singleton — injected via constructor.
- No SQLite in Phase 1 — JSON file-based storage via `GenericStore`.
- Every provider defines its own `DEFAULT_BASE_URL` when `ProviderConfig.base_url` is `None`.
- `ChatService` is stateless: no `start()`, `switch_session()`, `close()`.

---
### Task 1: Config Model + ConfigStore Protocol + JsonConfigStore Implementation

**Files:**
- Create: `src/yapa/services/config.py`
- Test: `tests/test_services/test_config.py`
- Create: `tests/test_services/__init__.py`

**Interfaces:**
- Produces: `Config`, `ProviderConfig`, `ConfigStore` protocol, `JsonConfigStore`, `DEFAULT_STORAGE_DIR`, `DEFAULT_CONFIG_PATH`

- [ ] **Step 1: Create test directory and conftest**

Create `tests/test_services/__init__.py` (empty file).
Create `tests/test_services/conftest.py` with an autouse logger fixture:

```python
"""Test fixtures for services-layer tests."""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_logger():
    with patch("yapa.services.config.get_logger") as mock:
        yield mock
```

- [ ] **Step 2: Write the failing tests for Config defaults**

```python
"""Tests for config models and JsonConfigStore."""

import json
from pathlib import Path

import pytest

from yapa.services.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigStore,
    JsonConfigStore,
    ProviderConfig,
)


class TestProviderConfig:
    def test_defaults(self):
        pc = ProviderConfig()
        assert pc.api_key is None
        assert pc.base_url is None

    def test_custom_values(self):
        pc = ProviderConfig(api_key="sk-abc", base_url="https://example.com/v1")
        assert pc.api_key == "sk-abc"
        assert pc.base_url == "https://example.com/v1"


class TestConfigDefaults:
    def test_provider_configs_defaults_to_empty(self):
        cfg = Config()
        assert cfg.provider_configs == {}

    def test_storage_dir_default(self):
        cfg = Config()
        assert cfg.storage_dir == Path.home() / ".yapa" / "storage"

    def test_log_level_default(self):
        cfg = Config()
        assert cfg.log_level == "INFO"

    def test_provider_timeout_default(self):
        cfg = Config()
        assert cfg.provider_timeout == 120

    def test_provider_max_retries_default(self):
        cfg = Config()
        assert cfg.provider_max_retries == 2


class TestJsonConfigStore:
    def test_load_returns_config_with_defaults_when_no_file(self, tmp_path):
        store = JsonConfigStore(path=tmp_path / "config.json")
        cfg = store.load()
        assert cfg.log_level == "INFO"
        assert cfg.storage_dir == Path.home() / ".yapa" / "storage"

    def test_load_reads_from_file(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"log_level": "DEBUG", "storage_dir": str(tmp_path)})
        )
        store = JsonConfigStore(path=config_path)
        cfg = store.load()
        assert cfg.log_level == "DEBUG"
        assert cfg.storage_dir == tmp_path

    def test_env_override_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YAPA_LOG_LEVEL", "ERROR")
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"log_level": "DEBUG"}))
        store = JsonConfigStore(path=config_path)
        cfg = store.load()
        assert cfg.log_level == "ERROR"

    def test_save_writes_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        store = JsonConfigStore(path=config_path)
        cfg = Config(log_level="WARNING", provider_timeout=60)
        store.save(cfg)
        data = json.loads(config_path.read_text())
        assert data["log_level"] == "WARNING"
        assert data["provider_timeout"] == 60

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "config.json"
        store = JsonConfigStore(path=nested)
        store.save(Config())
        assert nested.exists()

    def test_roundtrip(self, tmp_path):
        config_path = tmp_path / "config.json"
        store = JsonConfigStore(path=config_path)
        original = Config(log_level="WARNING", provider_configs={
            "openai": ProviderConfig(api_key="sk-abc"),
        })
        store.save(original)
        loaded = store.load()
        assert loaded.log_level == "WARNING"
        assert loaded.provider_configs["openai"].api_key == "sk-abc"

    def test_empty_json_uses_defaults(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{}")
        store = JsonConfigStore(path=config_path)
        cfg = store.load()
        assert cfg.log_level == "INFO"

    def test_caches_after_load(self, tmp_path):
        config_path = tmp_path / "config.json"
        store = JsonConfigStore(path=config_path)
        cfg1 = store.load()
        # Modify file behind the scenes
        config_path.write_text(json.dumps({"log_level": "DEBUG"}))
        cfg2 = store.load()
        # Should read from cache, not disk — still has defaults
        assert cfg2.log_level == "INFO"
```

Run: `uv run pytest tests/test_services/test_config.py -v`
Expected: FAIL (Config, ProviderConfig, JsonConfigStore not defined)

- [ ] **Step 3: Implement Config, ProviderConfig, ConfigStore, and JsonConfigStore**

```python
"""Config models, ConfigStore protocol, and JsonConfigStore implementation."""

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from yapa.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DATA_DIR = Path.home() / ".yapa"
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "config.json"
DEFAULT_STORAGE_DIR = DEFAULT_DATA_DIR / "storage"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_PROVIDER_TIMEOUT = 120

ENV_OVERRIDES: dict[str, str] = {
    "log_level": "YAPA_LOG_LEVEL",
    "storage_dir": "YAPA_STORAGE_DIR",
    "provider_timeout": "YAPA_PROVIDER_TIMEOUT",
    "provider_max_retries": "YAPA_PROVIDER_MAX_RETRIES",
}


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""

    api_key: str | None = None
    base_url: str | None = None


class Config(BaseModel):
    """Application configuration."""

    provider_configs: dict[str, ProviderConfig] = Field(default_factory=dict)
    storage_dir: Path = DEFAULT_STORAGE_DIR
    log_level: str = DEFAULT_LOG_LEVEL
    provider_timeout: int = DEFAULT_PROVIDER_TIMEOUT
    provider_max_retries: int = 2


@runtime_checkable
class ConfigStore(Protocol):
    """Protocol for config persistence."""

    def load(self) -> Config: ...
    def save(self, config: Config) -> None: ...


class JsonConfigStore:
    """JSON-file-backed config store with env variable overrides."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_CONFIG_PATH
        self._cache: Config | None = None

    def load(self) -> Config:
        """Read config file, apply env overrides, cache and return."""
        load_dotenv()
        config_data: dict[str, Any] = {}
        if self._path.exists():
            with open(self._path) as f:
                config_data = json.load(f) or {}

        for key, env_var in ENV_OVERRIDES.items():
            value = os.environ.get(env_var)
            if value is not None and value != "":
                if key == "storage_dir":
                    config_data[key] = Path(value)
                elif key in ("provider_timeout", "provider_max_retries"):
                    try:
                        config_data[key] = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer for {key}: {value}")
                else:
                    config_data[key] = value

        self._cache = Config(**config_data)
        return self._cache

    def save(self, config: Config) -> None:
        """Persist config to JSON file."""
        self._cache = config
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            f.write(config.model_dump_json(indent=2))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_services/test_config.py -v`
Expected: PASS (green)

- [ ] **Step 5: Commit**

```bash
git add src/yapa/services/config.py tests/test_services/
git commit -m "feat: add Config model, ConfigStore protocol, and JsonConfigStore"
```

---
### Task 2: Add system_prompt and inference_params to Session model

**Files:**
- Modify: `src/yapa/models/session.py`
- Test: `tests/models/test_session.py`

**Interfaces:**
- Produces: `Session` with `system_prompt: str | None = None` and `inference_params: InferenceParams | None = None`

- [ ] **Step 1: Write the failing tests for new Session fields**

Append to `tests/models/test_session.py`:

```python
from yapa.models import InferenceParams


class TestSessionNewFields:
    def test_system_prompt_defaults_to_none(self):
        session = Session()
        assert session.system_prompt is None

    def test_system_prompt_round_trip(self):
        session = Session(system_prompt="You are helpful.")
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert restored.system_prompt == "You are helpful."

    def test_inference_params_defaults_to_none(self):
        session = Session()
        assert session.inference_params is None

    def test_inference_params_round_trip(self):
        params = InferenceParams(temperature=0.7, max_tokens=4096)
        session = Session(inference_params=params)
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert restored.inference_params is not None
        assert restored.inference_params.temperature == 0.7
        assert restored.inference_params.max_tokens == 4096
```

Run: `uv run pytest tests/models/test_session.py::TestSessionNewFields -v`
Expected: FAIL (Session has no system_prompt/inference_params fields)

- [ ] **Step 2: Add system_prompt and inference_params fields**

Edit `src/yapa/models/session.py`:

```python
"""Session related models."""

from pydantic import Field

from .base import TrackedEntity
from .inference import InferenceParams, ModelData
from .message import Message

DEFAULT_SESSION_TITLE = "Untitled Session"


class Session(TrackedEntity):
    """A session is a collection of messages between a user and an AI."""

    title: str = Field(default=DEFAULT_SESSION_TITLE)
    model: ModelData | None = Field(default=None)
    system_prompt: str | None = Field(default=None)
    inference_params: InferenceParams | None = Field(default=None)
    messages: list[Message] = Field(default_factory=list)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_session.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/models/session.py tests/models/test_session.py
git commit -m "feat: add system_prompt and inference_params to Session model"
```

---
### Task 3: Trim Event Model to Phase 1 Event Types

**Files:**
- Modify: `src/yapa/models/event.py`
- Modify: `src/yapa/models/__init__.py`
- Create: `tests/models/test_event.py`

**Interfaces:**
- Produces: `EventType`, `Event`, `TextEvent`, `ReasoningEvent`, `AgentStartEvent`, `AgentDoneEvent`, `AgentErrorEvent`
- Removes: `ToolResultEvent`, `ToolApprovalRequestEvent`, `ToolApprovalResponseEvent`, `TOOL_CALL`, `TOOL_RESULT`, `TOOL_APPROVAL_REQUEST`, `TOOL_APPROVAL_RESPONSE`
- Changes: `AgentStartEvent` gains `model_id: str`, `AgentDoneEvent` gains `content: str`, `finish_reason: str | None`, `usage: TokenUsage | None`. `timestamp` changes from `float` to `datetime`.

- [ ] **Step 1: Write the failing tests for the trimmed event model**

```python
"""Tests for Phase 1 event model."""

from datetime import datetime, timezone

import pytest

from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    Event,
    EventSource,
    EventType,
    ReasoningEvent,
    TextEvent,
)
from yapa.models.inference import TokenUsage


class TestEventType:
    def test_text_chunk(self):
        assert EventType.TEXT_CHUNK == "text_chunk"

    def test_reasoning_chunk(self):
        assert EventType.REASONING_CHUNK == "reasoning_chunk"

    def test_agent_start(self):
        assert EventType.AGENT_START == "agent_start"

    def test_agent_done(self):
        assert EventType.AGENT_DONE == "agent_done"

    def test_agent_error(self):
        assert EventType.AGENT_ERROR == "agent_error"

    def test_tool_events_removed(self):
        with pytest.raises(AttributeError):
            EventType.TOOL_CALL
        with pytest.raises(AttributeError):
            EventType.TOOL_RESULT
        with pytest.raises(AttributeError):
            EventType.TOOL_APPROVAL_REQUEST
        with pytest.raises(AttributeError):
            EventType.TOOL_APPROVAL_RESPONSE


class TestEventBase:
    def test_type_is_required(self):
        event = Event(type=EventType.AGENT_START)
        assert event.type == EventType.AGENT_START

    def test_source_defaults_to_system(self):
        event = Event(type=EventType.AGENT_START)
        assert event.source == EventSource.SYSTEM

    def test_timestamp_is_datetime_with_tz(self):
        event = Event(type=EventType.AGENT_START)
        assert isinstance(event.timestamp, datetime)
        assert event.timestamp.tzinfo is not None


class TestTextEvent:
    def test_creates_with_content(self):
        event = TextEvent(content="Hello")
        assert event.content == "Hello"
        assert event.type == EventType.TEXT_CHUNK
        assert event.source == EventSource.AGENT

    def test_content_empty_string(self):
        event = TextEvent(content="")
        assert event.content == ""


class TestReasoningEvent:
    def test_creates_with_content(self):
        event = ReasoningEvent(content="thinking...")
        assert event.content == "thinking..."
        assert event.type == EventType.REASONING_CHUNK
        assert event.source == EventSource.AGENT


class TestAgentStartEvent:
    def test_creates_with_model_id(self):
        event = AgentStartEvent(model_id="openai:gpt-4")
        assert event.model_id == "openai:gpt-4"
        assert event.type == EventType.AGENT_START
        assert event.source == EventSource.AGENT


class TestAgentDoneEvent:
    def test_creates_with_content(self):
        event = AgentDoneEvent(content="Full response")
        assert event.content == "Full response"
        assert event.type == EventType.AGENT_DONE
        assert event.source == EventSource.AGENT
        assert event.finish_reason is None
        assert event.usage is None

    def test_with_finish_reason(self):
        event = AgentDoneEvent(content="Done", finish_reason="stop")
        assert event.finish_reason == "stop"

    def test_with_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        event = AgentDoneEvent(content="Done", usage=usage)
        assert event.usage == usage


class TestAgentErrorEvent:
    def test_creates_with_message(self):
        event = AgentErrorEvent(message="Something went wrong")
        assert event.message == "Something went wrong"
        assert event.type == EventType.AGENT_ERROR
        assert event.source == EventSource.AGENT
```

Run: `uv run pytest tests/models/test_event.py -v`
Expected: FAIL (tool events still present, missing model_id/content fields)

- [ ] **Step 2: Implement the trimmed event model**

Replace `src/yapa/models/event.py` entirely:

```python
"""Phase 1 event types for the agent-service event system."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from yapa.models.inference import TokenUsage


class EventType(str, Enum):
    """Enumeration of event types for the event system."""

    TEXT_CHUNK = "text_chunk"
    REASONING_CHUNK = "reasoning_chunk"
    AGENT_START = "agent_start"
    AGENT_DONE = "agent_done"
    AGENT_ERROR = "agent_error"


class EventSource(str, Enum):
    """Enumeration of event sources for the event system."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class Event(BaseModel):
    """Base class for all agent events."""

    type: EventType
    source: EventSource = EventSource.SYSTEM
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TextEvent(Event):
    """A chunk of streaming text content from the agent."""

    type: EventType = EventType.TEXT_CHUNK
    source: EventSource = EventSource.AGENT
    content: str


class ReasoningEvent(Event):
    """A chunk of streaming reasoning/thinking content from the agent."""

    type: EventType = EventType.REASONING_CHUNK
    source: EventSource = EventSource.AGENT
    content: str


class AgentStartEvent(Event):
    """Emitted when the agent begins processing a message."""

    type: EventType = EventType.AGENT_START
    source: EventSource = EventSource.AGENT
    model_id: str


class AgentDoneEvent(Event):
    """Emitted after the model response is complete."""

    type: EventType = EventType.AGENT_DONE
    source: EventSource = EventSource.AGENT
    content: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class AgentErrorEvent(Event):
    """Emitted when the agent encounters an unrecoverable error."""

    type: EventType = EventType.AGENT_ERROR
    source: EventSource = EventSource.AGENT
    message: str
```

Update `src/yapa/models/__init__.py`:

```python
"""Shared data models for the Yapa application."""

from .base import TrackedEntity
from .event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    Event,
    EventType,
    ReasoningEvent,
    TextEvent,
)
from .inference import (
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
    TokenUsage,
    ToolCallDelta,
)
from .message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from .session import Session
from .tool import (
    ToolApprovalRequest,
    ToolApprovalResponse,
    ToolCall,
)

__all__ = [
    # Messages
    "AssistantMessage",
    "Message",
    "SystemMessage",
    "ToolMessage",
    "UserMessage",
    # Inference
    "InferenceParams",
    "ModelData",
    "ModelType",
    "StreamDelta",
    "TokenUsage",
    "ToolCallDelta",
    # Session
    "Session",
    "TrackedEntity",
    # Tool models
    "ToolCall",
    "ToolApprovalRequest",
    "ToolApprovalResponse",
    # Event types
    "Event",
    "EventType",
    "TextEvent",
    "ReasoningEvent",
    "AgentStartEvent",
    "AgentDoneEvent",
    "AgentErrorEvent",
]
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_event.py -v`
Expected: PASS

- [ ] **Step 4: Run existing model tests to ensure nothing else broke**

Run: `uv run pytest tests/models/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/event.py src/yapa/models/__init__.py tests/models/test_event.py
git commit -m "feat: trim event model to Phase 1 types"
```

---
### Task 4: SessionStore Protocol + JsonSessionStore

**Files:**
- Create: `src/yapa/services/store.py`
- Test: `tests/test_services/test_store.py`

**Interfaces:**
- Consumes: `Session` (from Task 2)
- Produces: `SessionStore` protocol, `JsonSessionStore`

- [ ] **Step 1: Write failing tests for SessionStore**

```python
"""Tests for SessionStore protocol and JsonSessionStore."""

from uuid import uuid4

import pytest

from yapa.models import AssistantMessage, InferenceParams, Session, UserMessage
from yapa.services.store import JsonSessionStore, SessionStore


class TestSessionStoreProtocol:
    def test_json_session_store_conforms(self):
        assert isinstance(JsonSessionStore, object)
        # Protocol check: JsonSessionStore should implement SessionStore
        store = JsonSessionStore("/tmp")
        assert isinstance(store, SessionStore)


class TestJsonSessionStore:
    def test_save_and_load(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="test")
        store.save(session)
        loaded = store.load(str(session.id))
        assert loaded.id == session.id
        assert loaded.title == "test"

    def test_load_missing_raises(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            store.load("nonexistent")

    def test_save_with_overwrite(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="original")
        store.save(session)

        session.title = "updated"
        # Should not raise — overwrite=True
        store.save(session, overwrite=True)
        loaded = store.load(str(session.id))
        assert loaded.title == "updated"

    def test_save_without_overwrite_raises(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="original")
        store.save(session)

        session.title = "updated"
        with pytest.raises(FileExistsError):
            store.save(session, overwrite=False)

    def test_list_empty(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        assert store.list() == []

    def test_list_returns_all(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        s1 = Session(title="one")
        s2 = Session(title="two")
        store.save(s1)
        store.save(s2)
        sessions = store.list()
        ids = {str(s.id) for s in sessions}
        assert ids == {str(s1.id), str(s2.id)}

    def test_delete(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="delete-me")
        store.save(session)
        store.delete(str(session.id))
        with pytest.raises(FileNotFoundError):
            store.load(str(session.id))

    def test_delete_missing_raises(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.delete("nonexistent")

    def test_preserves_messages(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        session = Session(title="chat")
        session.messages = [
            UserMessage(content="hi"),
            AssistantMessage(content="hello", model="m"),
        ]
        store.save(session)
        loaded = store.load(str(session.id))
        assert len(loaded.messages) == 2
        assert loaded.messages[0].content == "hi"

    def test_preserves_inference_params(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        params = InferenceParams(temperature=0.5, max_tokens=2048)
        session = Session(title="configured", inference_params=params)
        store.save(session)
        loaded = store.load(str(session.id))
        assert loaded.inference_params is not None
        assert loaded.inference_params.temperature == 0.5

    def test_creates_storage_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "sessions"
        store = JsonSessionStore(storage_dir=nested)
        session = Session()
        store.save(session)
        assert nested.exists()
        assert (nested / f"{session.id}.json").exists()
```

Run: `uv run pytest tests/test_services/test_store.py -v`
Expected: FAIL (JsonSessionStore not defined)

- [ ] **Step 2: Implement SessionStore protocol + JsonSessionStore**

```python
"""Session persistence — SessionStore protocol and JsonSessionStore."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from yapa.models import Session
from yapa.storage import GenericStore


@runtime_checkable
class SessionStore(Protocol):
    """Protocol for session persistence."""

    def load(self, id: str) -> Session: ...
    def save(self, session: Session, *, overwrite: bool = False) -> None: ...
    def list(self) -> list[Session]: ...
    def delete(self, id: str) -> None: ...


class JsonSessionStore:
    """JSON-file-backed session store wrapping GenericStore."""

    def __init__(self, storage_dir: Path) -> None:
        self._store = GenericStore[Session](
            storage_dir=storage_dir,
            entity_type=Session,
        )

    def load(self, id: str) -> Session:
        return self._store.load(id)

    def save(self, session: Session, *, overwrite: bool = False) -> None:
        self._store.save(session, overwrite=overwrite)

    def list(self) -> list[Session]:
        return self._store.list()

    def delete(self, id: str) -> None:
        self._store.delete(id)
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_services/test_store.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/services/store.py tests/test_services/test_store.py
git commit -m "feat: add SessionStore protocol and JsonSessionStore"
```

---
### Task 5: Update Service Exceptions

**Files:**
- Modify: `src/yapa/services/exceptions.py`
- Test: `tests/test_services/test_exceptions.py`

- [ ] **Step 1: Write failing tests for ChatError**

Replace `tests/test_services/test_exceptions.py`:

```python
"""Tests for service-layer exception classes."""

import pytest

from yapa.services.exceptions import ChatError


class TestChatError:
    def test_is_exception(self):
        assert issubclass(ChatError, Exception)

    def test_can_be_raised(self):
        with pytest.raises(ChatError):
            raise ChatError("test error")

    def test_message_preserved(self):
        try:
            raise ChatError("something broke")
        except ChatError as e:
            assert str(e) == "something broke"
```

Run: `uv run pytest tests/test_services/test_exceptions.py -v`
Expected: FAIL (ChatError not defined)

- [ ] **Step 2: Replace ConversationError with ChatError**

Replace `src/yapa/services/exceptions.py`:

```python
"""Service-layer exceptions."""


class ChatError(Exception):
    """Raised when a chat operation fails."""
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_services/test_exceptions.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/services/exceptions.py tests/test_services/test_exceptions.py
git commit -m "feat: replace ConversationError with ChatError"
```

---
### Task 6: Implement SessionService

**Files:**
- Create: `src/yapa/services/session.py` (replaces existing)
- Test: `tests/test_services/test_session.py` (replaces existing)

**Interfaces:**
- Consumes: `SessionStore` protocol (from Task 4), `Session` (from Task 2)
- Produces: `SessionService` with: `create()`, `list()`, `get()`, `rename()`, `update_system_prompt()`, `update_inference_params()`, `add_messages()`, `delete()`

- [ ] **Step 1: Write failing tests for SessionService**

Replace `tests/test_services/test_session.py` with the full new test suite:

```python
"""Tests for SessionService."""

from uuid import uuid4

import pytest

from yapa.models import AssistantMessage, InferenceParams, UserMessage
from yapa.services.session import SessionService
from yapa.services.store import JsonSessionStore


class TestCreate:
    def test_creates_new_session(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        assert session.title == "Untitled Session"
        assert session.id is not None
        assert session.model is None
        assert session.system_prompt is None
        assert session.inference_params is None

    def test_persists_to_disk(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        loaded = service.get(str(session.id))
        assert loaded.id == session.id
        assert loaded.title == "Untitled Session"


class TestList:
    def test_empty_when_no_sessions(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        assert service.list() == []

    def test_returns_all_sessions(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        s1 = service.create()
        s2 = service.create()
        sessions = service.list()
        assert len(sessions) == 2

    def test_ordered_newest_first_by_default(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        s1 = service.create()
        s2 = service.create()
        sessions = service.list()
        assert sessions[0].id == s2.id
        assert sessions[1].id == s1.id


class TestGet:
    def test_returns_session(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        created = service.create()
        loaded = service.get(str(created.id))
        assert loaded.id == created.id

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.get("nonexistent")


class TestRename:
    def test_updates_title(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        updated = service.rename(str(session.id), "My Chat")
        assert updated.title == "My Chat"

    def test_persists_rename(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.rename(str(session.id), "Persisted")
        loaded = service.get(str(session.id))
        assert loaded.title == "Persisted"

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.rename("nonexistent", "new title")


class TestUpdateSystemPrompt:
    def test_sets_system_prompt(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        updated = service.update_system_prompt(str(session.id), "Be helpful.")
        assert updated.system_prompt == "Be helpful."

    def test_clears_system_prompt(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.update_system_prompt(str(session.id), "Be helpful.")
        service.update_system_prompt(str(session.id), None)
        loaded = service.get(str(session.id))
        assert loaded.system_prompt is None

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.update_system_prompt("nonexistent", "prompt")


class TestUpdateInferenceParams:
    def test_sets_params(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        params = InferenceParams(temperature=0.7, max_tokens=4096)
        updated = service.update_inference_params(str(session.id), params)
        assert updated.inference_params is not None
        assert updated.inference_params.temperature == 0.7

    def test_clears_params(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.update_inference_params(
            str(session.id), InferenceParams(temperature=0.5)
        )
        service.update_inference_params(str(session.id), None)
        loaded = service.get(str(session.id))
        assert loaded.inference_params is None

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.update_inference_params("nonexistent", None)


class TestAddMessages:
    def test_adds_single_message(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        msg = UserMessage(content="hello")
        updated = service.add_messages(str(session.id), [msg])
        assert len(updated.messages) == 1
        assert updated.messages[0].content == "hello"

    def test_adds_multiple_atomically(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        msgs = [
            UserMessage(content="q1"),
            AssistantMessage(content="a1", model="m"),
        ]
        updated = service.add_messages(str(session.id), msgs)
        assert len(updated.messages) == 2

    def test_persists_messages(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.add_messages(
            str(session.id),
            [UserMessage(content="persist-me")],
        )
        loaded = service.get(str(session.id))
        assert len(loaded.messages) == 1

    def test_updates_model(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        updated = service.add_messages(
            str(session.id),
            [UserMessage(content="hi")],
            model=model,
        )
        assert updated.model is not None
        assert updated.model.id == "gpt-4"
        assert updated.model.provider_id == "openai"
        loaded = service.get(str(session.id))
        assert loaded.model is not None
        assert loaded.model.id == "gpt-4"

    def test_updates_model_without_messages(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        model = ModelData(id="claude", provider_id="anthropic", type=ModelType.LLM)
        updated = service.add_messages(
            str(session.id), [], model=model
        )
        assert updated.model is not None
        assert updated.model.id == "claude"

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.add_messages("nonexistent", [UserMessage(content="hi")])


class TestDelete:
    def test_removes_session(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        session = service.create()
        service.delete(str(session.id))
        assert service.list() == []

    def test_raises_on_missing(self, tmp_path):
        store = JsonSessionStore(storage_dir=tmp_path)
        service = SessionService(store=store)
        with pytest.raises(ValueError, match="not found"):
            service.delete("nonexistent")
```

Run: `uv run pytest tests/test_services/test_session.py -v`
Expected: FAIL (SessionService not importable)

- [ ] **Step 2: Implement new SessionService**

Replace `src/yapa/services/session.py`:

```python
"""Session management service — CRUD and message appending."""

from yapa.models import InferenceParams, Message, ModelData, Session
from yapa.services.store import SessionStore


class SessionService:
    """CRUD for sessions + message appending."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def create(self) -> Session:
        """Create and persist a new session."""
        session = Session()
        self._store.save(session)
        return session

    def list(self, *, newest_first: bool = True) -> list[Session]:
        """List all sessions, newest first by default."""
        sessions = self._store.list()
        sessions.sort(key=lambda s: s.updated_at, reverse=newest_first)
        return sessions

    def get(self, session_id: str) -> Session:
        """Retrieve a session by ID."""
        try:
            return self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e

    def rename(self, session_id: str, title: str) -> Session:
        """Rename a session."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.title = title
        self._store.save(session, overwrite=True)
        return session

    def update_system_prompt(self, session_id: str, prompt: str | None) -> Session:
        """Set or clear the session's system prompt."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.system_prompt = prompt
        self._store.save(session, overwrite=True)
        return session

    def update_inference_params(
        self, session_id: str, params: InferenceParams | None
    ) -> Session:
        """Set or clear the session's inference parameters."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.inference_params = params
        self._store.save(session, overwrite=True)
        return session

    def add_messages(
        self,
        session_id: str,
        messages: list[Message],
        model: ModelData | None = None,
    ) -> Session:
        """Append messages to a session, optionally update model, and persist."""
        try:
            session = self._store.load(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
        session.messages.extend(messages)
        if model is not None:
            session.model = model
        session.touch()
        self._store.save(session, overwrite=True)
        return session

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        try:
            self._store.delete(session_id)
        except FileNotFoundError as e:
            raise ValueError(str(e)) from e
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_services/test_session.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/services/session.py tests/test_services/test_session.py
git commit -m "feat: implement new SessionService with SessionStore dependency injection"
```

---
### Task 7: Update ProviderRegistry to Use New Config

**Files:**
- Modify: `src/yapa/providers/registry.py`
- Existing test: `tests/providers/test_registry.py`

**Interfaces:**
- Consumes: `Config` from `yapa.services.config` (instead of `yapa.config`)
- Produces: Updated `ProviderRegistry` that takes new `Config`

- [ ] **Step 1: Write failing tests for updated registry**

Replace `tests/providers/test_registry.py`:

```python
"""Tests for ProviderRegistry."""

from unittest.mock import MagicMock, patch

import pytest

from yapa.providers.registry import ProviderNotAvailableError, ProviderRegistry
from yapa.services.config import Config


class TestProviderRegistry:
    def test_empty_provider_list(self):
        registry = ProviderRegistry(provider_classes=[])
        assert registry.available == []
        assert registry.failures == {}

    def test_available_providers(self):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.id = "mock_provider"
        mock_cls.return_value = mock_instance

        registry = ProviderRegistry(
            provider_classes=[mock_cls],
            config=Config(),
        )
        assert len(registry.available) == 1
        assert registry.available[0].id == "mock_provider"

    def test_tracks_failures(self):
        failing_cls = MagicMock()
        failing_cls.side_effect = ValueError("Missing API key")
        failing_cls.__name__ = "FailingProvider"

        registry = ProviderRegistry(
            provider_classes=[failing_cls],
            config=Config(),
        )
        assert registry.available == []
        assert "FailingProvider" in registry.failures

    def test_get_available_provider(self):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.id = "test_provider"
        mock_cls.return_value = mock_instance

        registry = ProviderRegistry(
            provider_classes=[mock_cls],
            config=Config(),
        )
        provider = registry.get("test_provider")
        assert provider.id == "test_provider"

    def test_get_unavailable_provider_raises(self):
        registry = ProviderRegistry(provider_classes=[], config=Config())
        with pytest.raises(ProviderNotAvailableError, match="not found"):
            registry.get("nonexistent")

    def test_is_available(self):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.id = "test_provider"
        mock_cls.return_value = mock_instance

        registry = ProviderRegistry(
            provider_classes=[mock_cls],
            config=Config(),
        )
        assert registry.is_available("test_provider") is True
        assert registry.is_available("nonexistent") is False
```

Run: `uv run pytest tests/providers/test_registry.py -v`
Expected: FAIL (imports from yapa.config)

- [ ] **Step 2: Update ProviderRegistry import**

Edit `src/yapa/providers/registry.py` — change the import line:

```python
from yapa.services.config import Config
```

Remove the old import:
```python
# OLD
from yapa.config import Config, get_config
```

Also update the `__init__` method — it currently falls back to `get_config()`. Change that fallback:

```python
cfg = config or Config()
```

Wait, we need a default Config. Let me check the current code again...

The current code:
```python
def __init__(
    self,
    provider_classes: list[type[InferenceProvider]],
    config: Config | None = None,
) -> None:
    ...
    cfg = config or get_config()
```

Change to:
```python
cfg = config or Config()
```

But `Config()` now has no `default_model`, so we need to also handle the case where config is None. Since `Config()` is the production default (no per-provider settings), this is fine — providers that require an API key will fail initialization and be tracked as failures.

Full replacement for `src/yapa/providers/registry.py`:

```python
"""Provider registry — attempts to initialize all known providers."""

from yapa.services.config import Config

from .base import InferenceProvider


class ProviderNotAvailableError(Exception):
    """Requested provider is not configured or failed to initialize."""


class ProviderRegistry:
    """
    Registry that surfaces available and failed providers.

    Attempts to initialize all registered provider classes. Providers that
    fail initialization (e.g. missing API keys) are tracked separately
    rather than failing the entire registry.
    """

    def __init__(
        self,
        provider_classes: list[type[InferenceProvider]],
        config: Config | None = None,
    ) -> None:
        self._available: dict[str, InferenceProvider] = {}
        self._failures: dict[str, str] = {}

        cfg = config or Config()
        for cls in provider_classes:
            try:
                instance = cls(config=cfg)
                self._available[instance.id] = instance
            except ValueError as e:
                self._failures[cls.__name__] = str(e)

    @property
    def available(self) -> list[InferenceProvider]:
        return list(self._available.values())

    @property
    def failures(self) -> dict[str, str]:
        return dict(self._failures)

    def is_available(self, provider_id: str) -> bool:
        return provider_id in self._available

    def get(self, provider_id: str) -> InferenceProvider:
        try:
            return self._available[provider_id]
        except KeyError:
            raise ProviderNotAvailableError(
                f"Provider '{provider_id}' not found."
            )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_registry.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/providers/registry.py tests/providers/test_registry.py
git commit -m "refactor: update ProviderRegistry to use new Config from services.config"
```

---
### Task 8: Update Provider Implementations for New Config

**Files:**
- Modify: `src/yapa/providers/openai_compat.py`
- Modify: `src/yapa/providers/openai/provider.py`
- Modify: `src/yapa/providers/openrouter/provider.py`
- Modify: `src/yapa/providers/lmstudio/provider.py`
- Modify: `src/yapa/providers/ollama/provider.py`
- Test: `tests/providers/concretes/test_openai.py`, `tests/providers/concretes/test_openrouter.py`, `tests/providers/concretes/test_lmstudio.py`, `tests/providers/concretes/test_ollama.py`

**Interfaces:**
- Consumes: `Config` from `yapa.services.config` (instead of `yapa.config`)
- Consumes: `ProviderConfig` from `yapa.services.config`

- [ ] **Step 1: Update OpenAICompatibleProvider import**

Edit `src/yapa/providers/openai_compat.py`:

Change:
```python
from yapa.config import DEFAULT_PROVIDER_TIMEOUT
```

To:
```python
from yapa.services.config import DEFAULT_PROVIDER_TIMEOUT
```

That's the only change needed in this file.

- [ ] **Step 2: Update OpenAIIP**

Replace `src/yapa/providers/openai/provider.py`:

```python
"""OpenAI inference provider implementation."""

from typing import cast

from yapa.models import ModelData
from yapa.services.config import Config, ProviderConfig

from ..openai_compat import OpenAICompatibleProvider

_MODEL_METADATA: dict[str, dict[str, object]] = {
    "gpt-5.6-sol": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.6": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.6-terra": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.6-luna": {
        "context_length": 1_050_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.5": {
        "context_length": 1_000_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.4": {
        "context_length": 400_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.4-mini": {
        "context_length": 400_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
    "gpt-5.4-nano": {
        "context_length": 400_000,
        "max_output": 131072,
        "supports_tools": True,
        "supports_vision": True,
    },
}


class OpenAIIP(OpenAICompatibleProvider):
    """Inference provider for OpenAI."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: Config):
        pc = config.provider_configs.get("openai", ProviderConfig())
        if pc.api_key is None:
            raise ValueError("OpenAI API key is not set.")
        super().__init__(
            identifier="openai",
            name="OpenAI",
            api_key=pc.api_key,
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )

    def _format_model(self, model_id: str) -> ModelData:
        model = super()._format_model(model_id)
        meta = _MODEL_METADATA.get(model_id)
        if meta is not None:
            return ModelData(
                id=model.id,
                provider_id=model.provider_id,
                type=model.type,
                context_length=cast(int | None, meta["context_length"]),
                max_output=cast(int | None, meta["max_output"]),
                supports_tools=cast(bool, meta["supports_tools"]),
                supports_vision=cast(bool, meta["supports_vision"]),
            )
        return model
```

- [ ] **Step 3: Update OpenRouterProvider**

Replace `src/yapa/providers/openrouter/provider.py`:

```python
"""OpenRouter inference provider implementation."""

import httpx

from yapa.models import ModelData, ModelType
from yapa.services.config import Config, ProviderConfig

from ..openai_compat import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    """Inference provider for OpenRouter."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: Config):
        pc = config.provider_configs.get("openrouter", ProviderConfig())
        if pc.api_key is None:
            raise ValueError("OpenRouter API key is not set.")
        super().__init__(
            identifier="openrouter",
            name="OpenRouter",
            api_key=pc.api_key,
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )

    def _format_model_from_openrouter(self, raw: dict) -> ModelData:
        model_id = raw["id"]
        model = self._format_model(model_id)
        context_length = raw.get("context_length")
        max_output = raw.get("max_completion_tokens")
        arch = raw.get("architecture", {})
        modality = arch.get("modality", "")
        supported = raw.get("supported_parameters", [])
        pricing: dict[str, float] | None = None
        if "pricing" in raw:
            p = raw["pricing"]
            try:
                prompt = float(p.get("prompt", 0)) * 1_000_000
                completion = float(p.get("completion", 0)) * 1_000_000
                pricing = {"input": prompt, "output": completion}
            except (ValueError, TypeError):
                pricing = None
        return ModelData(
            id=model.id,
            provider_id=model.provider_id,
            type=model.type,
            context_length=context_length,
            max_output=max_output,
            supports_tools="tools" in supported,
            supports_vision="vision" in modality or "image" in modality,
            pricing=pricing,
        )

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]:
        headers = {"Authorization": f"Bearer {self._client.api_key}"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{str(self._client.base_url).rstrip('/')}/models",
                headers=headers,
            )
            resp.raise_for_status()
            raw_models = resp.json().get("data", [])
        formatted = [self._format_model_from_openrouter(m) for m in raw_models]
        if model_type:
            return [m for m in formatted if m.type == model_type]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        return self._format_model(model_id)
```

- [ ] **Step 4: Update LMStudioIP**

Replace `src/yapa/providers/lmstudio/provider.py`:

```python
"""LM Studio inference provider implementation."""

import httpx

from yapa.models import ModelData, ModelType
from yapa.services.config import Config, ProviderConfig

from ..openai_compat import OpenAICompatibleProvider


class LMStudioIP(OpenAICompatibleProvider):
    """Inference provider for LM Studio."""

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(self, config: Config):
        pc = config.provider_configs.get("lmstudio", ProviderConfig())
        super().__init__(
            identifier="lmstudio",
            name="LM Studio",
            api_key=pc.api_key or "",
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )

    def _native_base_url(self) -> str:
        base = str(self._client.base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[: -3]
        return base + "/api/v1"

    def _format_model_from_native(self, raw: dict) -> ModelData:
        model_id = raw.get("key", "")
        model = self._format_model(model_id)
        caps = raw.get("capabilities", {})
        instances = raw.get("loaded_instances", [])
        config = instances[0].get("config", {}) if instances else {}
        return ModelData(
            id=model.id,
            provider_id=model.provider_id,
            type=model.type,
            context_length=config.get("context_length"),
            supports_tools=caps.get("trained_for_tool_use", False),
            supports_vision=caps.get("vision", False),
        )

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]:
        headers = {}
        if self._client.api_key:
            headers["Authorization"] = f"Bearer {self._client.api_key}"
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self._native_base_url()}/models", headers=headers
            )
            resp.raise_for_status()
            raw_models = resp.json().get("data", [])
        formatted = [self._format_model_from_native(m) for m in raw_models]
        if model_type:
            return [m for m in formatted if m.type == model_type]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        return self._format_model(model_id)
```

- [ ] **Step 5: Update OllamaIP**

Replace `src/yapa/providers/ollama/provider.py`:

```python
"""Ollama inference provider implementation."""

from yapa.services.config import Config, ProviderConfig

from ..openai_compat import OpenAICompatibleProvider


class OllamaIP(OpenAICompatibleProvider):
    """Inference provider for Ollama."""

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, config: Config):
        pc = config.provider_configs.get("ollama", ProviderConfig())
        super().__init__(
            identifier="ollama",
            name="Ollama",
            api_key=pc.api_key or "",
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
```

- [ ] **Step 6: Run tests to verify provider changes**

Run: `uv run pytest tests/providers/ -v`
Expected: PASS (existing provider tests should still pass with the new config)

If tests fail, check that `tests/providers/conftest.py` patches config correctly. The existing tests may import from `yapa.config` and need updating to `yapa.services.config`.

- [ ] **Step 7: Commit**

```bash
git add src/yapa/providers/openai_compat.py src/yapa/providers/openai/provider.py src/yapa/providers/openrouter/provider.py src/yapa/providers/lmstudio/provider.py src/yapa/providers/ollama/provider.py
git commit -m "refactor: update providers to use new Config with provider_configs dict and DEFAULT_BASE_URL"
```

---
### Task 9: Implement ModelService

**Files:**
- Create: `src/yapa/services/models.py`
- Test: `tests/test_services/test_models.py`

**Interfaces:**
- Consumes: `ProviderRegistry` (from Task 7), `InferenceProvider`, `DEFAULT_PROVIDER_CLASSES`
- Produces: `ModelService` with: `get_provider()`, `get_provider_by_model()`, `list_models()`, `get_model()`

- [ ] **Step 1: Write failing tests for ModelService**

```python
"""Tests for ModelService."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import ModelData, ModelType
from yapa.providers import DEFAULT_PROVIDER_CLASSES, InferenceProvider, ModelsFetchError
from yapa.services.models import ModelService


@pytest.fixture(autouse=True)
def _mock_logger():
    with patch("yapa.services.models.get_logger") as mock:
        yield mock


class TestInit:
    def test_creates_default_registry(self):
        with patch("yapa.services.models.ProviderRegistry") as mock_reg_cls:
            ModelService()
            mock_reg_cls.assert_called_once()

    def test_accepts_custom_registry(self):
        registry = MagicMock()
        svc = ModelService(registry=registry)
        assert svc._registry is registry


class TestGetProvider:
    def test_delegates_to_registry(self):
        registry = MagicMock()
        svc = ModelService(registry=registry)
        svc.get_provider("openai")
        registry.get.assert_called_once_with("openai")


class TestGetProviderByModel:
    def test_delegates_to_registry(self):
        registry = MagicMock()
        svc = ModelService(registry=registry)
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        svc.get_provider_by_model(model)
        registry.get.assert_called_once_with("openai")


class TestListModels:
    @pytest.fixture
    def svc(self):
        registry = MagicMock()
        provider_a = MagicMock(spec=InferenceProvider)
        provider_a.id = "prov_a"
        provider_a.list_models = AsyncMock(
            return_value=[
                ModelData(id="gpt-4", provider_id="prov_a", type=ModelType.LLM),
            ]
        )
        provider_b = MagicMock(spec=InferenceProvider)
        provider_b.id = "prov_b"
        provider_b.list_models = AsyncMock(
            return_value=[
                ModelData(id="claude", provider_id="prov_b", type=ModelType.LLM),
            ]
        )
        registry.available = [provider_a, provider_b]
        registry.get.return_value = provider_a
        return ModelService(registry=registry)

    async def test_returns_flat_list(self, svc):
        result = await svc.list_models()
        assert len(result) == 2
        assert all(isinstance(m, ModelData) for m in result)

    async def test_filters_by_provider(self, svc):
        result = await svc.list_models(provider_id="prov_a")
        assert len(result) == 1
        assert result[0].id == "gpt-4"
        assert result[0].provider_id == "prov_a"

    async def test_filters_by_model_type(self, svc):
        result = await svc.list_models(model_type=ModelType.LLM)
        assert len(result) == 2

    async def test_continues_on_provider_error(self, _mock_logger):
        registry = MagicMock()
        provider_a = MagicMock(spec=InferenceProvider)
        provider_a.id = "prov_a"
        provider_a.list_models = AsyncMock(
            side_effect=ModelsFetchError("API down")
        )
        provider_b = MagicMock(spec=InferenceProvider)
        provider_b.id = "prov_b"
        provider_b.list_models = AsyncMock(
            return_value=[
                ModelData(id="claude", provider_id="prov_b", type=ModelType.LLM),
            ]
        )
        registry.available = [provider_a, provider_b]
        svc = ModelService(registry=registry)
        result = await svc.list_models()
        assert len(result) == 1
        assert result[0].id == "claude"

    async def test_propagates_provider_id_stamp(self, svc):
        result = await svc.list_models()
        for m in result:
            assert m.provider_id in ("prov_a", "prov_b")


class TestGetModel:
    @pytest.fixture
    def svc(self):
        registry = MagicMock()
        provider = MagicMock(spec=InferenceProvider)
        provider.id = "prov_a"
        provider.get_model = AsyncMock(
            return_value=ModelData(
                id="gpt-4", provider_id="prov_a", type=ModelType.LLM
            )
        )
        registry.get.return_value = provider
        return ModelService(registry=registry)

    async def test_returns_model_data(self, svc):
        result = await svc.get_model("prov_a:gpt-4")
        assert result.id == "gpt-4"
        assert result.provider_id == "prov_a"

    async def test_raises_on_malformed_id(self, svc):
        with pytest.raises(ValueError, match="expected 'provider_id:model_id'"):
            await svc.get_model("no-colon")
```

Run: `uv run pytest tests/test_services/test_models.py -v`
Expected: FAIL (ModelService not defined)

- [ ] **Step 2: Implement ModelService**

```python
"""Model service — thin wrapper around ProviderRegistry."""

from yapa.logging import get_logger
from yapa.models import ModelData, ModelType
from yapa.providers import (
    DEFAULT_PROVIDER_CLASSES,
    InferenceProvider,
    ModelsFetchError,
    ProviderRegistry,
)

logger = get_logger(__name__)


class ModelService:
    """Thin wrapper around ProviderRegistry for model fetching."""

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self._registry = registry or ProviderRegistry(DEFAULT_PROVIDER_CLASSES)

    def get_provider(self, provider_id: str) -> InferenceProvider:
        return self._registry.get(provider_id)

    def get_provider_by_model(self, model: ModelData) -> InferenceProvider:
        return self._registry.get(model.provider_id)

    async def list_models(
        self,
        provider_id: str | None = None,
        model_type: ModelType | None = None,
    ) -> list[ModelData]:
        """Fetch models from one or all providers, returning a flat list."""
        if provider_id:
            provider = self.get_provider(provider_id)
            try:
                return await provider.list_models(model_type)
            except ModelsFetchError as e:
                logger.error(f"Failed to fetch models for '{provider_id}': {e}")
                return []

        results: list[ModelData] = []
        for provider in self._registry.available:
            try:
                models = await provider.list_models(model_type)
                results.extend(models)
            except ModelsFetchError as e:
                logger.error(f"Failed to fetch models for '{provider.id}': {e}")
        return results

    async def get_model(self, model_full_id: str) -> ModelData:
        """Fetch details for a specific model by full ID (provider_id:model_id)."""
        try:
            provider_id, model_id = model_full_id.split(":", 1)
        except ValueError:
            raise ValueError(
                f"Invalid model full ID '{model_full_id}': "
                "expected 'provider_id:model_id'"
            )
        provider = self.get_provider(provider_id)
        try:
            return await provider.get_model(model_id=model_id)
        except ModelsFetchError as e:
            logger.error(f"Failed to fetch model '{model_full_id}': {e}")
            raise ValueError(f"Failed to fetch model '{model_full_id}': {e}") from e
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_services/test_models.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/services/models.py tests/test_services/test_models.py
git commit -m "feat: implement ModelService wrapping ProviderRegistry"
```

---
### Task 10: Implement ChatService

**Files:**
- Create: `src/yapa/services/chat.py`
- Test: `tests/test_services/test_chat.py`

**Interfaces:**
- Consumes: `SessionService` (from Task 6), `ModelService` (from Task 9), `Event` types (from Task 3)
- Produces: `ChatService` with: `stream()` → `AsyncGenerator[Event, None]`

- [ ] **Step 1: Write failing tests for ChatService**

```python
"""Tests for ChatService."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    Session,
    StreamDelta,
    TokenUsage,
    UserMessage,
)
from yapa.services.chat import ChatService
from yapa.services.exceptions import ChatError
from yapa.services.session import SessionService
from yapa.services.models import ModelService
from yapa.services.store import JsonSessionStore
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    ReasoningEvent,
    TextEvent,
)


@pytest.fixture
def models(tmp_path):
    svc = MagicMock(spec=ModelService)
    provider = MagicMock()
    provider.stream_chat = AsyncMock()
    svc.get_provider_by_model.return_value = provider
    svc.get_provider = MagicMock(return_value=provider)
    return svc


@pytest.fixture
def sessions(tmp_path):
    store = JsonSessionStore(storage_dir=tmp_path)
    svc = SessionService(store=store)
    return svc


@pytest.fixture
def chat(models, sessions):
    return ChatService(sessions=sessions, models=models)


class TestStream:
    async def test_agent_start_then_text_then_done(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Hello")
            yield StreamDelta(
                content=" world",
                finish_reason="stop",
                usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            )

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        assert len(events) == 4
        assert isinstance(events[0], AgentStartEvent)
        assert events[0].model_id == "openai:gpt-4"
        assert isinstance(events[1], TextEvent)
        assert events[1].content == "Hello"
        assert isinstance(events[2], TextEvent)
        assert events[2].content == " world"
        assert isinstance(events[3], AgentDoneEvent)
        assert events[3].content == "Hello world"
        assert events[3].finish_reason == "stop"
        assert events[3].usage is not None
        assert events[3].usage.total_tokens == 8

    async def test_persists_user_and_assistant_messages(
        self, chat, sessions, models
    ):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Hello")
            yield StreamDelta(
                content=" world",
                finish_reason="stop",
            )

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            pass

        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 2
        assert isinstance(loaded.messages[0], UserMessage)
        assert loaded.messages[0].content == "Hi"
        assert isinstance(loaded.messages[1], AssistantMessage)
        assert loaded.messages[1].content == "Hello world"
        assert loaded.messages[1].model == "openai:gpt-4"

    async def test_includes_reasoning_content(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(reasoning_content="thinking...")
            yield StreamDelta(content="Answer", finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        assert isinstance(events[1], ReasoningEvent)
        assert events[1].content == "thinking..."
        assert isinstance(events[2], TextEvent)
        assert events[2].content == "Answer"
        assert isinstance(events[3], AgentDoneEvent)
        assert events[3].content == "Answer"

    async def test_uses_session_system_prompt(self, chat, sessions, models):
        session = sessions.create()
        sessions.update_system_prompt(str(session.id), "Be concise.")
        provider = models.get_provider_by_model.return_value
        captured_kwargs = {}

        async def _capture(model, messages, tools=None, params=None):
            captured_kwargs["messages"] = messages
            yield StreamDelta(content="OK", finish_reason="stop")

        provider.stream_chat.side_effect = _capture
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            pass

        messages = captured_kwargs["messages"]
        # System prompt should be prepended to conversation history
        assert any(
            hasattr(m, "role") and m.role == "system" and m.content == "Be concise."
            for m in messages
        )

    async def test_uses_session_inference_params(self, chat, sessions, models):
        session = sessions.create()
        params = InferenceParams(temperature=0.3, max_tokens=100)
        sessions.update_inference_params(str(session.id), params)
        provider = models.get_provider_by_model.return_value
        captured_kwargs = {}

        async def _capture(model, messages, tools=None, params=None):
            captured_kwargs["params"] = params
            yield StreamDelta(content="OK", finish_reason="stop")

        provider.stream_chat.side_effect = _capture
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            pass

        assert captured_kwargs["params"] == params

    async def test_raises_error_when_no_model(self, chat, sessions):
        session = sessions.create()
        with pytest.raises(ValueError, match="No model specified"):
            async for _ in chat.stream(
                session_id=session.id,
                prompt="Hi",
            ):
                pass

    async def test_uses_session_model_as_fallback(self, chat, sessions, models):
        session = sessions.create()
        # Manually set model on session (simulating previous stream)
        session.model = ModelData(
            id="gpt-4", provider_id="openai", type=ModelType.LLM
        )
        sessions._store.save(session, overwrite=True)
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="OK", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hi",
        ):
            pass

        models.get_provider_by_model.assert_called_with(
            ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        )

    async def test_saves_model_to_session_after_stream(
        self, chat, sessions, models
    ):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Hi", finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        async for _ in chat.stream(
            session_id=session.id,
            prompt="Hello",
            model=model,
        ):
            pass

        loaded = sessions.get(str(session.id))
        assert loaded.model is not None
        assert loaded.model.id == "gpt-4"
        assert loaded.model.provider_id == "openai"

    async def test_agent_error_on_model_failure(self, chat, sessions, models):
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        provider.stream_chat.side_effect = Exception("API failure")
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        events = []
        async for event in chat.stream(
            session_id=session.id,
            prompt="Hi",
            model=model,
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], AgentStartEvent)
        assert isinstance(events[1], AgentErrorEvent)
        assert "API failure" in events[1].message

        # Messages should NOT be persisted on error
        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 0

    async def test_stream_is_stateless(self, chat, sessions, models):
        """Two consecutive stream calls should be independent."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Resp", finish_reason="stop")

        provider.stream_chat.side_effect = _stream
        model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)

        async for _ in chat.stream(
            session_id=session.id, prompt="First", model=model
        ):
            pass

        async for _ in chat.stream(
            session_id=session.id, prompt="Second", model=model
        ):
            pass

        loaded = sessions.get(str(session.id))
        assert len(loaded.messages) == 4  # 2 user + 2 assistant

    async def test_raises_on_missing_session(self, chat, models):
        with pytest.raises(ValueError, match="not found"):
            async for _ in chat.stream(
                session_id=uuid4(),
                prompt="Hi",
                model=ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM),
            ):
                pass
```

Run: `uv run pytest tests/test_services/test_chat.py -v`
Expected: FAIL (ChatService not defined)

- [ ] **Step 2: Implement ChatService**

```python
"""Stateless chat orchestrator — single model invocation per stream() call."""

from collections.abc import AsyncGenerator
from uuid import UUID

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    Message,
    ModelData,
    StreamDelta,
    SystemMessage,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    Event,
    ReasoningEvent,
    TextEvent,
)
from yapa.services.exceptions import ChatError
from yapa.services.models import ModelService
from yapa.services.session import SessionService


class ChatService:
    """Stateless orchestrator for a single model invocation."""

    def __init__(
        self,
        *,
        sessions: SessionService,
        models: ModelService,
    ) -> None:
        self._sessions = sessions
        self._models = models

    async def stream(
        self,
        session_id: UUID,
        prompt: str,
        model: ModelData | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Stream a model response for the given prompt."""
        session = self._sessions.get(str(session_id))

        if model is None:
            if session.model is not None:
                model = session.model
            else:
                raise ValueError("No model specified")

        yield AgentStartEvent(model_id=model.full_id)

        provider = self._models.get_provider_by_model(model)

        messages: list[Message] = []
        if session.system_prompt is not None:
            messages.append(SystemMessage(content=session.system_prompt))
        messages.extend(session.messages)
        messages.append(UserMessage(content=prompt))

        params = session.inference_params or InferenceParams()

        content_buffer = ""
        finish_reason: str | None = None

        try:
            async for delta in provider.stream_chat(
                model=model,
                messages=messages,
                params=params,
            ):
                if delta.reasoning_content:
                    yield ReasoningEvent(content=delta.reasoning_content)
                if delta.content:
                    content_buffer += delta.content
                    yield TextEvent(content=delta.content)
                if delta.finish_reason:
                    finish_reason = delta.finish_reason
                usage = delta.usage
        except Exception as e:
            yield AgentErrorEvent(message=str(e))
            return

        if not content_buffer:
            yield AgentErrorEvent(message="Model returned empty response")
            return

        assistant_msg = AssistantMessage(
            content=content_buffer,
            model=model.full_id,
        )

        self._sessions.add_messages(
            str(session_id),
            [UserMessage(content=prompt), assistant_msg],
            model=model,
        )

        yield AgentDoneEvent(
            content=content_buffer,
            finish_reason=finish_reason,
            usage=usage,
        )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run pytest tests/test_services/test_chat.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/services/chat.py tests/test_services/test_chat.py
git commit -m "feat: implement stateless ChatService with Event streaming"
```

---
### Task 11: Update logging.py to Import from New Config Location

**Files:**
- Modify: `src/yapa/logging.py`

**Interfaces:**
- Consumes: `Config` from `yapa.services.config` (instead of `yapa.config`)

- [ ] **Step 1: Update the import in logging.py**

Edit `src/yapa/logging.py` — change:

```python
from yapa.config import get_config
```

To:

```python
from yapa.services.config import Config, DEFAULT_DATA_DIR
```

Then update the `get_logger()` function to use `DEFAULT_DATA_DIR` instead of `get_config()`:

```python
def get_logger(
    name: str,
    console: bool = False,
    level: str | None = None,
) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]

    log_level = level or "INFO"

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()

    log_dir = DEFAULT_DATA_DIR / "logs" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger
```

- [ ] **Step 2: Verify tests still pass**

Run: `uv run pytest tests/ -v --ignore=tests/test_config.py`
Expected: PASS (test_config.py tests the old config module which will be removed)

- [ ] **Step 3: Commit**

```bash
git add src/yapa/logging.py
git commit -m "refactor: update logging.py to use new Config and DEFAULT_DATA_DIR"
```

---
### Task 12: Remove Old Files and Update Exports

**Files:**
- Remove: `src/yapa/services/conversation.py`
- Remove: `src/yapa/services/provider.py`
- Remove: `src/yapa/config.py`
- Remove: `tests/test_config.py`
- Remove: `tests/services/test_conversation.py`
- Remove: `tests/services/test_provider.py`
- Modify: `src/yapa/services/__init__.py`
- Modify: `src/yapa/providers/__init__.py` (add `openai_compat` to exports if missing)

- [ ] **Step 1: Delete old files**

```bash
rm src/yapa/services/conversation.py
rm src/yapa/services/provider.py
rm src/yapa/config.py
rm tests/test_config.py
rm tests/services/test_conversation.py
rm tests/services/test_provider.py
```

- [ ] **Step 2: Update services/__init__.py**

Replace `src/yapa/services/__init__.py`:

```python
"""Service layer — UI-agnostic business logic."""

from .chat import ChatService
from .config import Config, ConfigStore, JsonConfigStore, ProviderConfig
from .exceptions import ChatError
from .models import ModelService
from .session import SessionService
from .store import JsonSessionStore, SessionStore

__all__ = [
    "ChatError",
    "ChatService",
    "Config",
    "ConfigStore",
    "JsonConfigStore",
    "JsonSessionStore",
    "ModelService",
    "ProviderConfig",
    "SessionService",
    "SessionStore",
]
```

- [ ] **Step 3: Update providers/__init__.py if needed**

Check if `OpenAICompatibleProvider` needs to be exported. It's currently not in `__all__`, and consumers import it directly. Leave as-is.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v --cov=src --cov-fail-under=80`
Expected: PASS with coverage >= 80%

If any tests fail, fix them. Common issues:
- Tests still importing from `yapa.config` → update to `yapa.services.config`
- Tests importing `ConversationService` or `ProviderService` → update to `ChatService` or `ModelService`
- Tests importing `ConversationError` → update to `ChatError`

- [ ] **Step 5: Commit**

```bash
git add src/yapa/services/__init__.py \
       src/yapa/services/conversation.py \
       src/yapa/services/provider.py \
       src/yapa/config.py \
       tests/test_config.py \
       tests/services/test_conversation.py \
       tests/services/test_provider.py
git commit -m "refactor: remove old config.py, ConversationService, ProviderService; update exports"
```

---
### Task 13: Full Lint and Type Check

**Files:** All modified files

- [ ] **Step 1: Run ruff linter**

Run: `uv run ruff check src/ tests/`
Expected: No errors

If there are errors, fix them. Common issues:
- Unused imports after removing old code
- Missing docstrings (ruff docstring rules enabled)
- Line length violations

- [ ] **Step 2: Run type checker**

Run: `uv run ty check src/`
Expected: No type errors

If errors occur, fix them by adding correct type annotations.

- [ ] **Step 3: Final full test suite**

Run: `uv run pytest tests/ -v --cov=src --cov-fail-under=80`
Expected: PASS with coverage >= 80%

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: fix lint and type errors after services layer refactor"
```
