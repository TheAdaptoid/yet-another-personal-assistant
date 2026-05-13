# YAPA Architecture

## Project Overview

**YAPA** (Yet Another Personal Assistant) — A local-first personal AI assistant with a TUI interface.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        TUI (textual)                        │
│   ┌─────────────────────┐  ┌─────────────────────────────┐  │
│   │   Chat Panel        │  │   Task Sidebar              │  │
│   │                     │  │                             │  │
│   │                     │  │   ▸ Task 1   PENDING        │  │
│   │                     │  │   ⏳ Task 2   IN_PROGRESS   │  │
│   │                     │  │   ✓ Task 3   COMPLETED     │  │
│   │                     │  │                             │  │
│   └─────────────────────┘  └─────────────────────────────┘  │
│                                                             │
└────────────────────────┬────────────────────────────────────┘
                         │ WebSocket + HTTP
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    yapa-core (FastAPI)                      │
│                                                             │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │ /health  │  │ /tasks   │  │/sessions │  │ /chat WS │  │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                 Agent Loop                            │  │
│   │   LLM + Tools + Session + Task Manager               │  │
│   └──────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     ~/.yapa/                                │
│                                                             │
│   ├── config.json          # Global config                 │
│   ├── sessions/            # Session data (JSON)           │
│   │   └── {session_id}.json                              │
│   ├── tasks/               # Task data (JSON)             │
│   │   └── {task_id}.json                                 │
│   └── logs/                # Log files                     │
└─────────────────────────────────────────────────────────────┘
```

## Components

### yapa-core (Daemon)

**Purpose:** Background service that runs the agent loop, manages tasks, sessions, and tools.

**Entry point:** `python -m yapa.core`

**Responsibilities:**
- HTTP/WebSocket server
- Agent loop execution
- Task management (create, dispatch, pause, resume, complete)
- Session management
- Tool registry and execution

### TUI (Terminal UI)

**Entry point:** `python -m yapa.tui`

**Layout:**
- Left panel: Chat history (scrollable)
- Right panel: Task list with status badges
- Bottom: Input field + Send button

**Features:**
- WebSocket connection to core
- Real-time message streaming
- Task status display
- Session switching

### shared (Common Library)

**Location:** `yapa/shared/`

**Contents:**
- `models/` — Pydantic models for Task, Session, SubTask, Message
- `tools/` — Tool implementations (read_file, write_file, curl, create_task, get_pending_tasks, etc.)
- `config/` — Configuration management
- `logging/` — Logging utilities

---

## API Contracts

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Core health status |

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "active_sessions": 2,
  "active_tasks": 1
}
```

### Sessions

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/sessions` | List all sessions |
| POST | `/sessions` | Create new session |
| GET | `/sessions/{id}` | Get session details + messages |
| DELETE | `/sessions/{id}` | Delete session |

**GET /sessions response:**
```json
{
  "sessions": [
    {"id": "abc123", "created_at": "2025-05-13T10:00:00Z", "updated_at": "2025-05-13T10:30:00Z"}
  ]
}
```

### Tasks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tasks` | List all tasks |
| POST | `/tasks` | Create task |
| GET | `/tasks/{id}` | Get task + subtasks |
| PATCH | `/tasks/{id}` | Update task status (pause/resume/complete) |
| POST | `/tasks/{id}/interrupt` | Stop running task |
| GET | `/tasks/pending` | Get pending tasks (for agent tool) |

**Task model:**
```json
{
  "id": "task_123",
  "name": "Research topic",
  "description": "Find information about X",
  "priority": 1,
  "status": "pending",
  "subtasks": [
    {"id": "st_1", "name": "Search", "description": "...", "status": "pending"}
  ],
  "created_at": "2025-05-13T10:00:00Z",
  "updated_at": "2025-05-13T10:00:00Z"
}
```

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| WS | `/chat/{session_id}` | Real-time chat |

**WebSocket messages:**

*Client → Server:*
```json
{"type": "message", "content": "Write a summary of..."}
{"type": "interrupt"}
```

*Server → Client:*
```json
{"type": "delta", "content": "I will..."}
{"type": "delta", "content": "I found..."}
{"type": "complete", "result": "Final summary..."}
{"type": "error", "message": "..."}
```

---

## Data Models

```python
class TaskPriority(IntEnum):
    URGENT = 0
    IMPORTANT = 1
    UNIMPORTANT = 2

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"

class SubTask(BaseModel):
    id: str
    name: str
    description: str
    status: TaskStatus

class Task(BaseModel):
    id: str
    name: str
    description: str
    priority: TaskPriority
    status: TaskStatus
    subtasks: list[SubTask]  # embedded in task JSON file
    created_at: datetime
    updated_at: datetime

class Message(BaseModel):
    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime

class Session(BaseModel):
    id: str
    messages: list[Message]
    created_at: datetime
    updated_at: datetime
```

---

## Tool Definitions

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read file contents | `path: str` |
| `write_file` | Write content to file | `path: str, content: str` |
| `curl` | Make HTTP request | `url: str, method: str = "GET", body: str = None` |
| `create_task` | Create a task | `name: str, description: str, priority: int = 1` |
| `update_task` | Update task status | `task_id: str, status: str` |
| `get_pending_tasks` | List pending tasks | — |
| `get_active_session` | Get current session | — |

---

## File Structure

```
yapa/
├── pyproject.toml
├── README.md
└── src/
    └── yapa/
        ├── __init__.py
        ├── core/
        │   ├── __init__.py
        │   ├── agent.py          # Agent loop
        │   ├── main.py           # FastAPI entry point
        │   ├── router_task.py    # Task endpoints
        │   ├── router_session.py # Session endpoints
        │   ├── router_chat.py    # WebSocket chat
        │   └── worker.py         # Background task runner
        ├── tui/
        │   ├── __init__.py
        │   ├── app.py            # TUI app class
        │   ├── main.py           # Textual entry point
        │   ├── screens/
        │   │   └── chat_screen.py
        │   └── widgets/
        │       ├── chat_panel.py
        │       └── task_panel.py
        └── shared/
            ├── __init__.py
            ├── config.py
            ├── logging.py
            ├── models/
            │   ├── __init__.py
            │   ├── message.py
            │   ├── session.py
            │   └── task.py
            └── tools/
                ├── __init__.py
                ├── file_tools.py
                ├── http_tools.py
                ├── registry.py
                └── task_tools.py
```

---

## Logging

- Location: `~/.yapa/logs/`
- Format: `yapa-{date}.log`
- Levels: DEBUG, INFO, WARNING, ERROR
- Log to both file and stdout (for systemd journal)

---

## Running

**Development:**
```bash
# Terminal 1: Core
python -m yapa.core

# Terminal 2: TUI
python -m yapa.tui
```

**Production:**
```bash
systemctl --user start yapa-core
yapa-tui
```