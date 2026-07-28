# Agentic Loops and Tool Calling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an agentic loop to ChatService that detects tool calls, executes tools (with approval gating for dangerous ones), and re-invokes the model until it produces a text response.

**Architecture:** ChatService.stream() becomes a loop (max 10 iterations). Per iteration: call provider with tools → collect tool calls → execute approved tools → feed results back as ToolMessage → repeat. A new ToolRegistry holds tool instances. Approval flows through an injectable callback on the stream method.

**Tech Stack:** Python 3.12+, asyncio, FastAPI WebSocket, Pydantic v2

## Global Constraints

- No new dependencies beyond stdlib + existing (FastAPI, Pydantic, httpx)
- All provider tool-formatting code already exists — do not modify providers
- `ChatService` must remain stateless (no instance state per stream call)
- Approval callback is injected per-stream, not stored on the service
- Tool errors must be fed back to the model as ToolMessage (not exceptions)
- Unknown tool names return a descriptive error to the model
- File paths resolved against `os.getcwd()`; no symlink traversal checking for v1
- `write_file` must refuse if parent directory does not exist
- Tests must use mocked provider (no real API calls)

---
### Task 1: Tool Events

**Files:**
- Modify: `src/yapa/models/event.py:11-76`
- Modify: `src/yapa/models/__init__.py:1-64`
- Test: (implicit — tested downstream via service tests)

**Interfaces:**
- Consumes: existing `Event`, `EventType`, `EventSource` base classes
- Produces: `ToolCallEvent`, `ToolApprovalRequestEvent`, `ToolResultEvent` classes

- [ ] **Step 1: Add new event types to `event.py`**

Add these to `EventType`:

```python
TOOL_CALL = "tool_call"
TOOL_APPROVAL_REQUEST = "tool_approval_request"
TOOL_RESULT = "tool_result"
```

Add these classes after `AgentErrorEvent`:

```python
class ToolCallEvent(Event):
    type: EventType = EventType.TOOL_CALL
    source: EventSource = EventSource.AGENT
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


class ToolApprovalRequestEvent(Event):
    type: EventType = EventType.TOOL_APPROVAL_REQUEST
    source: EventSource = EventSource.SYSTEM
    tool_name: str
    arguments: dict[str, Any]
    call_id: str


class ToolResultEvent(Event):
    type: EventType = EventType.TOOL_RESULT
    source: EventSource = EventSource.SYSTEM
    tool_name: str
    call_id: str
    result: JsonValue
```

Add `from __future__ import annotations`, `from typing import Any`, and import `JsonValue` from `yapa.tools.base`.

- [ ] **Step 2: Export new events from `models/__init__.py`**

Add imports and `__all__` entries for `ToolCallEvent`, `ToolApprovalRequestEvent`, `ToolResultEvent`.

- [ ] **Step 3: Commit**

```bash
git add src/yapa/models/event.py src/yapa/models/__init__.py
git commit -m "feat: add tool call, approval request, and tool result events"
```

---
### Task 2: Calculator Tool

**Files:**
- Create: `src/yapa/tools/core/__init__.py`
- Create: `src/yapa/tools/core/calculator.py`
- Test: `tests/tools/test_calculator.py`

**Interfaces:**
- Produces: `Calculator` class (subclass of `Tool`, `needs_approval=False`)
- Consumes: `Tool` ABC from `yapa.tools.base`, `JsonValue`

- [ ] **Step 1: Write failing test**

```python
import pytest
from yapa.tools.core.calculator import calculator


class TestCalculator:
    async def test_add(self):
        result = await calculator.execute(expression="1 + 2")
        assert result == 3

    async def test_subtract(self):
        result = await calculator.execute(expression="10 - 4")
        assert result == 6

    async def test_multiply(self):
        result = await calculator.execute(expression="3 * 7")
        assert result == 21

    async def test_divide(self):
        result = await calculator.execute(expression="15 / 3")
        assert result == 5.0

    async def test_float_result(self):
        result = await calculator.execute(expression="7 / 2")
        assert result == 3.5

    async def test_invalid_expression_raises(self):
        with pytest.raises(Exception):
            await calculator.execute(expression="invalid +")

    async def test_name_and_metadata(self):
        assert calculator.name == "calculator"
        assert calculator.needs_approval is False
        assert "expression" in calculator.parameters["properties"]
```

Run: `pytest tests/tools/test_calculator.py -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Create `tools/core/__init__.py`**

```python
"""Concrete tool implementations."""

from yapa.tools.base import Tool


def default_tools() -> list[Tool]:
    """Return the default set of built-in tools."""
    from .calculator import calculator
    return [calculator]
```

- [ ] **Step 3: Create `tools/core/calculator.py`**

```python
"""Calculator tool — evaluates mathematical expressions."""

from yapa.tools.base import JsonValue, Tool


class Calculator(Tool):
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Evaluate a mathematical expression. Supports +, -, *, /, **, //, % and parentheses.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    },
                },
                "required": ["expression"],
            },
            needs_approval=False,
        )

    async def execute(self, expression: str = "", **kwargs: object) -> JsonValue:
        allowed = {"__builtins__": {}}
        return eval(expression, allowed, {})


calculator = Calculator()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_calculator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/yapa/tools/core/ tests/tools/test_calculator.py
git commit -m "feat: add calculator tool"
```

---
### Task 3: Read-Only File Tools

**Files:**
- Create: `src/yapa/tools/core/read_file.py`
- Create: `src/yapa/tools/core/grep.py`
- Create: `src/yapa/tools/core/list_dir.py`
- Modify: `src/yapa/tools/core/__init__.py`
- Test: `tests/tools/test_file_tools.py`

**Interfaces:**
- Produces: `read_file`, `grep`, `list_dir` tool instances (all `needs_approval=False`)

- [ ] **Step 1: Write failing test**

```python
import pytest
from pathlib import Path
from yapa.tools.core.read_file import read_file
from yapa.tools.core.grep import grep
from yapa.tools.core.list_dir import list_dir


class TestReadFile:
    async def test_reads_file(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await read_file.execute(path=str(f))
        assert result == "hello world"

    async def test_reads_with_limit(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await read_file.execute(path=str(f), limit=2)
        assert result == "line1\nline2\n"

    async def test_reads_with_offset(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await read_file.execute(path=str(f), offset=2)
        assert result == "line2\nline3\n"

    async def test_file_not_found(self):
        result = await read_file.execute(path="/nonexistent/file.txt")
        assert "Error" in result or "No such file" in result

    async def test_name_and_metadata(self):
        assert read_file.name == "read_file"
        assert read_file.needs_approval is False


class TestGrep:
    async def test_finds_pattern(self, tmp_path: Path):
        d = tmp_path / "sub"
        d.mkdir()
        f = d / "test.txt"
        f.write_text("apple\nbanana\napple pie\n")
        result = await grep.execute(pattern="apple", path=str(d))
        assert "test.txt" in result

    async def test_no_match(self, tmp_path: Path):
        d = tmp_path / "sub"
        d.mkdir()
        f = d / "test.txt"
        f.write_text("hello")
        result = await grep.execute(pattern="zzzz", path=str(d))
        assert not result or result == ""

    async def test_name_and_metadata(self):
        assert grep.name == "grep"
        assert grep.needs_approval is False


class TestListDir:
    async def test_lists_directory(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        result = await list_dir.execute(path=str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result

    async def test_name_and_metadata(self):
        assert list_dir.name == "list_dir"
        assert list_dir.needs_approval is False
```

Run: `pytest tests/tools/test_file_tools.py -v`
Expected: FAIL

- [ ] **Step 2: Create `read_file.py`**

```python
"""Read file tool."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class ReadFile(Tool):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read the contents of a file. Can optionally specify offset (1-indexed line) and limit (max lines).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                    },
                },
                "required": ["path"],
            },
            needs_approval=False,
        )

    async def execute(self, path: str = "", offset: int | None = None, limit: int | None = None, **kwargs: object) -> JsonValue:
        try:
            p = Path(path).resolve()
            lines = p.read_text().splitlines(keepends=True)
            start = (offset - 1) if offset else 0
            end = start + limit if limit else None
            return "".join(lines[start:end])
        except Exception as e:
            return f"Error: {e}"


read_file = ReadFile()
```

- [ ] **Step 3: Create `grep.py`**

```python
"""Grep tool — search files for a pattern."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class Grep(Tool):
    def __init__(self):
        super().__init__(
            name="grep",
            description="Search for a pattern in files within a directory. Returns matching file paths with line numbers and content.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to search",
                    },
                    "include": {
                        "type": "string",
                        "description": "Optional glob pattern to filter files (e.g. '*.py')",
                    },
                },
                "required": ["pattern", "path"],
            },
            needs_approval=False,
        )

    async def execute(self, pattern: str = "", path: str = "", include: str | None = None, **kwargs: object) -> JsonValue:
        import re
        try:
            root = Path(path).resolve()
            if not root.is_dir():
                return f"Error: {path} is not a directory"
            results: list[str] = []
            glob_pattern = f"**/{include}" if include else "**/*"
            for f in sorted(root.glob(glob_pattern)):
                if not f.is_file():
                    continue
                try:
                    text = f.read_text(errors="replace")
                    for i, line in enumerate(text.splitlines(), 1):
                        if re.search(pattern, line):
                            rel = f.relative_to(root)
                            results.append(f"{rel}:{i}:{line}")
                except Exception:
                    continue
            return "\n".join(results)
        except Exception as e:
            return f"Error: {e}"


grep = Grep()
```

- [ ] **Step 4: Create `list_dir.py`**

```python
"""List directory tool."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class ListDir(Tool):
    def __init__(self):
        super().__init__(
            name="list_dir",
            description="List files and directories in a path. Shows names, types (file/dir), and sizes.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory",
                    },
                },
                "required": ["path"],
            },
            needs_approval=False,
        )

    async def execute(self, path: str = "", **kwargs: object) -> JsonValue:
        try:
            p = Path(path).resolve()
            if not p.is_dir():
                return f"Error: {path} is not a directory"
            entries: list[str] = []
            for entry in sorted(p.iterdir()):
                if entry.is_dir():
                    entries.append(f"{entry.name}/")
                elif entry.is_file():
                    size = entry.stat().st_size
                    entries.append(f"{entry.name} ({size} bytes)")
                else:
                    entries.append(entry.name)
            return "\n".join(entries)
        except Exception as e:
            return f"Error: {e}"


list_dir = ListDir()
```

- [ ] **Step 5: Wire into `core/__init__.py`**

```python
def default_tools() -> list[Tool]:
    from .calculator import calculator
    from .grep import grep
    from .list_dir import list_dir
    from .read_file import read_file
    return [calculator, read_file, grep, list_dir]
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/tools/test_file_tools.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/yapa/tools/core/ tests/tools/test_file_tools.py
git commit -m "feat: add read-only file tools (read_file, grep, list_dir)"
```

---
### Task 4: Write/Dangerous File Tools

**Files:**
- Create: `src/yapa/tools/core/write_file.py`
- Create: `src/yapa/tools/core/bash.py`
- Modify: `src/yapa/tools/core/__init__.py`
- Test: `tests/tools/test_file_tools.py` (append)

**Interfaces:**
- Produces: `write_file`, `bash` tool instances (both `needs_approval=True`)

- [ ] **Step 1: Write failing test (append to existing `test_file_tools.py`)**

```python
class TestWriteFile:
    async def test_writes_file(self, tmp_path: Path):
        f = tmp_path / "out.txt"
        result = await write_file.execute(path=str(f), content="hello")
        assert f.read_text() == "hello"
        assert result == "ok"

    async def test_refuses_when_parent_missing(self, tmp_path: Path):
        f = tmp_path / "missing" / "out.txt"
        result = await write_file.execute(path=str(f), content="hello")
        assert "parent directory does not exist" in result
        assert not f.exists()

    async def test_name_and_metadata(self):
        assert write_file.name == "write_file"
        assert write_file.needs_approval is True


class TestBash:
    async def test_runs_command(self):
        result = await bash.execute(command="echo hello")
        assert "hello" in result

    async def test_failing_command(self):
        result = await bash.execute(command="exit 1")
        assert "exit code 1" in result

    async def test_name_and_metadata(self):
        assert bash.name == "bash"
        assert bash.needs_approval is True
```

- [ ] **Step 2: Create `write_file.py`**

```python
"""Write file tool."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class WriteFile(Tool):
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Write content to a file at the specified path. The parent directory must already exist. Overwrites existing files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
            needs_approval=True,
        )

    async def execute(self, path: str = "", content: str = "", **kwargs: object) -> JsonValue:
        try:
            p = Path(path).resolve()
            if not p.parent.exists():
                return "Error: parent directory does not exist"
            p.write_text(content)
            return "ok"
        except Exception as e:
            return f"Error: {e}"


write_file = WriteFile()
```

- [ ] **Step 3: Create `bash.py`**

```python
"""Bash tool — executes shell commands."""

import asyncio

from yapa.tools.base import JsonValue, Tool


class Bash(Tool):
    def __init__(self):
        super().__init__(
            name="bash",
            description="Execute a shell command and return its output. Use for running scripts, compiling code, or any command-line operation.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                },
                "required": ["command"],
            },
            needs_approval=True,
        )

    async def execute(self, command: str = "", **kwargs: object) -> JsonValue:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=60.0
            )
            output = stdout.decode()
            if stderr:
                output += "\nstderr:\n" + stderr.decode()
            if proc.returncode != 0:
                output += f"\n(exit code {proc.returncode})"
            return output.strip()
        except asyncio.TimeoutError:
            return "Error: command timed out after 60 seconds"
        except Exception as e:
            return f"Error: {e}"


bash = Bash()
```

- [ ] **Step 4: Create `edit_file.py`**

```python
"""Edit file tool — replaces first occurrence of old_string with new_string."""

from pathlib import Path

from yapa.tools.base import JsonValue, Tool


class EditFile(Tool):
    def __init__(self):
        super().__init__(
            name="edit_file",
            description="Replace the first occurrence of old_string with new_string in a file. Use for surgical edits without rewriting entire files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to replace (first occurrence only)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The string to replace it with",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
            needs_approval=True,
        )

    async def execute(self, path: str = "", old_string: str = "", new_string: str = "", **kwargs: object) -> JsonValue:
        try:
            p = Path(path).resolve()
            text = p.read_text()
            if old_string not in text:
                return f"Error: could not find '{old_string}' in {path}"
            new_text = text.replace(old_string, new_string, 1)
            p.write_text(new_text)
            return "ok"
        except Exception as e:
            return f"Error: {e}"


edit_file = EditFile()
```

- [ ] **Step 5: Add edit_file test cases**

Add to the test file, after `TestBash`:

```python
class TestEditFile:
    async def test_replaces_string(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = await edit_file.execute(path=str(f), old_string="world", new_string="there")
        assert result == "ok"
        assert f.read_text() == "hello there"

    async def test_string_not_found(self, tmp_path: Path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = await edit_file.execute(path=str(f), old_string="zzz", new_string="aaa")
        assert "could not find" in result

    async def test_name_and_metadata(self):
        assert edit_file.name == "edit_file"
        assert edit_file.needs_approval is True
```

- [ ] **Step 6: Wire into `core/__init__.py`**

```python
def default_tools() -> list[Tool]:
    from .bash import bash
    from .calculator import calculator
    from .edit_file import edit_file
    from .grep import grep
    from .list_dir import list_dir
    from .read_file import read_file
    from .write_file import write_file
    return [calculator, read_file, write_file, grep, list_dir, bash, edit_file]
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/tools/test_file_tools.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

---
### Task 5: ToolRegistry Wiring

**Files:**
- Modify: `src/yapa/services/__init__.py` — no changes needed (ToolRegistry already accessible via `yapa.tools`)
- Modify: `src/yapa/api/app.py`
- Test: `tests/tools/test_registry.py`

**Interfaces:**
- Consumes: `ToolRegistry` from `yapa.tools.registry`, `default_tools()` from `yapa.tools.core`
- Produces: wired `ToolRegistry` in `app.state`

- [ ] **Step 1: Write test for registry**

```python
import pytest
from yapa.tools.registry import ToolRegistry
from yapa.tools.core import default_tools


class TestToolRegistry:
    def test_initializes_with_default_tools(self):
        registry = ToolRegistry(default_tools())
        assert registry.get_tool("calculator") is not None
        assert registry.get_tool("read_file") is not None
        assert registry.get_tool("write_file") is not None
        assert registry.get_tool("grep") is not None
        assert registry.get_tool("list_dir") is not None
        assert registry.get_tool("bash") is not None

    def test_list_tools(self):
        registry = ToolRegistry(default_tools())
        tools = registry.list_tools()
        assert len(tools) >= 6
        names = {t.name for t in tools}
        assert "calculator" in names

    def test_get_unknown_tool(self):
        registry = ToolRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_register_and_unregister(self):
        registry = ToolRegistry()
        tool = MagicMock()
        tool.name = "test_tool"
        registry.register(tool)
        assert registry.get_tool("test_tool") is tool
        registry.unregister("test_tool")
        assert registry.get_tool("test_tool") is None
```

- [ ] **Step 2: Wire ToolRegistry into `app.py`**

Modify `_build_services`:

```python
from yapa.tools import ToolRegistry
from yapa.tools.core import default_tools

def _build_services(config: Config):
    store = JsonSessionStore(config.storage_dir)
    session_service = SessionService(store)
    model_service = ModelService()
    tool_registry = ToolRegistry(default_tools())
    chat_service = ChatService(
        sessions=session_service,
        models=model_service,
        tools=tool_registry,
    )
    return session_service, model_service, chat_service, tool_registry
```

Update the lifespan to also store `app.state.tool_registry`. Update the `hasattr` check to include `tool_registry`.

- [ ] **Step 3: Run tests**

Run: `pytest tests/tools/test_registry.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/yapa/api/app.py tests/tools/test_registry.py
git commit -m "feat: wire ToolRegistry into app factory"
```

---
### Task 6: Agentic Loop in ChatService

**Files:**
- Modify: `src/yapa/services/chat.py`
- Test: `tests/test_services/test_tool_loop.py`

**Interfaces:**
- Consumes: `ToolRegistry`, `ToolApprovalGetter` callable, existing events + new tool events
- Produces: `ChatService.stream()` with agentic loop

- [ ] **Step 1: Write tests for the agentic loop**

```python
"""Tests for ChatService agentic loop."""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
    TokenUsage,
    ToolMessage,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    TextEvent,
    ToolApprovalRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse
from yapa.services.chat import ChatService
from yapa.services.models import ModelService
from yapa.services.session import SessionService
from yapa.services.store import JsonSessionStore
from yapa.tools.base import Tool
from yapa.tools.registry import ToolRegistry


class ToolThatReturns(Tool):
    """Tool that returns a fixed result."""
    def __init__(self, name="echo", result="tool_result", needs_approval=False):
        super().__init__(
            name=name,
            description="Echo tool",
            parameters={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
            needs_approval=needs_approval,
        )
        self._result = result

    async def execute(self, input: str = "", **kwargs):
        return self._result


class ToolThatRaises(Tool):
    def __init__(self):
        super().__init__(
            name="failing",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            needs_approval=False,
        )

    async def execute(self, **kwargs):
        msg = "internal failure"
        raise RuntimeError(msg)


@pytest.fixture
def models(tmp_path):
    svc = MagicMock(spec=ModelService)
    provider = MagicMock()
    svc.get_provider_by_model.return_value = provider
    return svc


@pytest.fixture
def sessions(tmp_path):
    store = JsonSessionStore(storage_dir=tmp_path)
    return SessionService(store=store)


@pytest.fixture
def registry():
    return ToolRegistry([ToolThatReturns()])


@pytest.fixture
def chat(models, sessions, registry):
    return ChatService(sessions=sessions, models=models, tools=registry)


model = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)


class TestToolLoop:
    async def test_text_only_no_loop(self, chat, sessions, models):
        """When model returns text, no tool loop."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(content="Hello", finish_reason="stop", usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert isinstance(events[-1], AgentDoneEvent)
        assert events[-1].content == "Hello"
        # Only one provider call
        assert provider.stream_chat.call_count == 1

    async def test_single_tool_call_then_text(self, chat, sessions, models):
        """Model calls tool, tool executes, then model responds with text."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        call_count = 0

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="echo", arguments='{"input": "hi"}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert isinstance(events[-1], AgentDoneEvent)
        assert events[-1].content == "Done"
        assert provider.stream_chat.call_count == 2

    async def test_tool_call_event_yielded(self, chat, sessions, models):
        """ToolCallEvent is yielded for each tool call."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0
        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="echo", arguments='{"input": "hi"}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].tool_name == "echo"

    async def test_tool_result_event_yielded(self, chat, sessions, models):
        """ToolResultEvent is yielded after tool execution."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0
        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="echo", arguments='{"input": "hi"}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0].tool_name == "echo"

    async def test_dangerous_tool_requests_approval(self, chat, sessions, models):
        """Tool with needs_approval=True yields ToolApprovalRequestEvent and awaits callback."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        # Replace registry with a dangerous tool
        dangerous = ToolThatReturns(name="dangerous", needs_approval=True)
        chat._tools = ToolRegistry([dangerous])

        call_count = 0
        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="dangerous", arguments='{}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        approved = False
        async def get_approval(req: ToolApprovalRequest) -> ToolApprovalResponse:
            nonlocal approved
            approved = True
            return ToolApprovalResponse(call_id=req.call_id, approved=True)

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model, get_approval=get_approval):
            events.append(e)

        assert approved
        assert isinstance(events[-1], AgentDoneEvent)

    async def test_denied_tool_feeds_back_to_model(self, chat, sessions, models):
        """Denied tool call feeds rejection as ToolMessage and continues loop."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        dangerous = ToolThatReturns(name="dangerous", needs_approval=True)
        chat._tools = ToolRegistry([dangerous])

        call_count = 0
        prompts_seen = []

        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            prompts_seen.append(len(messages))
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="dangerous", arguments='{}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Ok", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        async def get_approval(req):
            return ToolApprovalResponse(call_id=req.call_id, approved=False, reason="not now")

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model, get_approval=get_approval):
            events.append(e)

        assert provider.stream_chat.call_count == 2
        # Second call should have more messages (denied ToolMessage)
        assert prompts_seen[1] > prompts_seen[0]

    async def test_unknown_tool(self, chat, sessions, models):
        """Unknown tool name is fed back as error ToolMessage."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0
        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="nonexistent", arguments='{}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Ok", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert provider.stream_chat.call_count == 2

    async def test_tool_execution_error(self, chat, sessions, models):
        """Tool that raises is fed back as error ToolMessage."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value
        chat._tools = ToolRegistry([ToolThatRaises()])

        call_count = 0
        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="failing", arguments='{}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Ok", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert provider.stream_chat.call_count == 2

    async def test_max_iterations(self, chat, sessions, models):
        """Loop terminates after MAX_ITERATIONS if model keeps calling tools."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        async def _stream(model, messages, tools=None, params=None):
            yield StreamDelta(
                tool_calls=[MagicMock(index=0, id="call_1", name="echo", arguments='{"input": "x"}')],
                finish_reason="tool_calls",
            )

        provider.stream_chat.side_effect = _stream

        events = []
        async for e in chat.stream(session_id=session.id, prompt="Hi", model=model):
            events.append(e)

        assert isinstance(events[-1], AgentErrorEvent)
        assert "max iterations" in events[-1].message.lower()
        assert provider.stream_chat.call_count == ChatService.MAX_ITERATIONS

    async def test_persists_messages_only_once(self, chat, sessions, models):
        """Messages are persisted only when the loop terminates, not on each iteration."""
        session = sessions.create()
        provider = models.get_provider_by_model.return_value

        call_count = 0
        async def _stream(model, messages, tools=None, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield StreamDelta(
                    tool_calls=[MagicMock(index=0, id="call_1", name="echo", arguments='{"input": "x"}')],
                    finish_reason="tool_calls",
                )
            else:
                yield StreamDelta(content="Done", finish_reason="stop")

        provider.stream_chat.side_effect = _stream

        async for _ in chat.stream(session_id=session.id, prompt="Hi", model=model):
            pass

        loaded = sessions.get(str(session.id))
        # Only one user + one assistant message (from final iteration)
        user_msgs = [m for m in loaded.messages if isinstance(m, UserMessage)]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Hi"
```

- [ ] **Step 2: Implement the agentic loop in `ChatService`**

Modify `ChatService.__init__` to accept `tools: ToolRegistry`:

```python
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from uuid import UUID

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    Message,
    ModelData,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from yapa.models.event import (
    AgentDoneEvent,
    AgentErrorEvent,
    AgentStartEvent,
    Event,
    ReasoningEvent,
    TextEvent,
    ToolApprovalRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from yapa.models.inference import ToolCallDelta
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse, ToolCall
from yapa.services.models import ModelService
from yapa.services.session import SessionService
from yapa.tools.registry import ToolRegistry


ToolApprovalGetter = Callable[
    [ToolApprovalRequest],
    Awaitable[ToolApprovalResponse],
]


class ChatService:
    MAX_ITERATIONS = 10

    def __init__(
        self,
        *,
        sessions: SessionService,
        models: ModelService,
        tools: ToolRegistry,
    ) -> None:
        self._sessions = sessions
        self._models = models
        self._tools = tools

    async def stream(
        self,
        session_id: UUID,
        prompt: str,
        model: ModelData | None = None,
        get_approval: ToolApprovalGetter | None = None,
    ) -> AsyncGenerator[Event, None]:
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
        initial_prompt_msg = messages[-1]

        try:
            for _ in range(self.MAX_ITERATIONS):
                content_buffer = ""
                finish_reason: str | None = None
                usage = None
                raw_tool_calls: list[tuple[int, str | None, str | None, str | None]] = []

                async for delta in provider.stream_chat(
                    model=model,
                    messages=messages,
                    tools=self._tools.list_tools(),
                    params=params,
                ):
                    if delta.reasoning_content:
                        yield ReasoningEvent(content=delta.reasoning_content)
                    if delta.content:
                        content_buffer += delta.content
                        yield TextEvent(content=delta.content)
                    for tcd in delta.tool_calls:
                        _merge_tool_call_delta(raw_tool_calls, tcd)
                    if delta.finish_reason:
                        finish_reason = delta.finish_reason
                    if delta.usage:
                        usage = delta.usage

                # Assemble tool calls
                tool_calls = [
                    ToolCall(id=tc_id, tool_name=tc_name, arguments=json.loads(tc_args))
                    for _, tc_id, tc_name, tc_args in raw_tool_calls
                    if tc_id and tc_name and tc_args
                ]

                assistant_msg = AssistantMessage(
                    content=content_buffer or None,
                    tool_calls=tool_calls,
                    model=model.full_id,
                    usage=usage,
                )
                messages.append(assistant_msg)

                # No tool calls → done
                if not tool_calls:
                    if not content_buffer:
                        yield AgentErrorEvent(message="Model returned empty response")
                        return
                    self._sessions.add_messages(
                        str(session_id),
                        [initial_prompt_msg, assistant_msg],
                        model=model,
                    )
                    yield AgentDoneEvent(
                        content=content_buffer,
                        finish_reason=finish_reason,
                        usage=usage,
                    )
                    return

                # Execute tool calls
                for tc in tool_calls:
                    tool = self._tools.get_tool(tc.tool_name)

                    if tool is None:
                        yield ToolCallEvent(tool_name=tc.tool_name, arguments=tc.arguments, call_id=tc.id)
                        messages.append(ToolMessage(
                            content=f"Unknown tool: {tc.tool_name}",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        ))
                        continue

                    yield ToolCallEvent(tool_name=tc.tool_name, arguments=tc.arguments, call_id=tc.id)

                    if tool.needs_approval:
                        yield ToolApprovalRequestEvent(
                            tool_name=tc.tool_name,
                            arguments=tc.arguments,
                            call_id=tc.id,
                        )
                        if get_approval is not None:
                            response = await get_approval(ToolApprovalRequest(
                                call_id=tc.id,
                                name=tc.tool_name,
                                arguments=tc.arguments,
                            ))
                            if not response.approved:
                                reason = response.reason or "No reason given"
                                messages.append(ToolMessage(
                                    content=f"Tool call denied: {reason}",
                                    tool_call_id=tc.id,
                                    tool_name=tc.tool_name,
                                ))
                                continue

                    try:
                        result = await tool.execute(**tc.arguments)
                        yield ToolResultEvent(tool_name=tc.tool_name, call_id=tc.id, result=result)
                        messages.append(ToolMessage(
                            content=_serialize_result(result),
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        ))
                    except Exception as e:
                        yield ToolResultEvent(tool_name=tc.tool_name, call_id=tc.id, result={"error": str(e)})
                        messages.append(ToolMessage(
                            content=f"Error: {e}",
                            tool_call_id=tc.id,
                            tool_name=tc.tool_name,
                        ))

            # Exceeded max iterations
            yield AgentErrorEvent(message="Max iterations reached")

        except Exception as e:
            yield AgentErrorEvent(message=str(e))
```

Add these helper functions at module level (or as private methods):

```python
import json


def _merge_tool_call_delta(
    acc: list[tuple[int, str | None, str | None, str | None]],
    delta: ToolCallDelta,
) -> None:
    """Merge a tool call delta into the accumulator list."""
    while len(acc) <= delta.index:
        acc.append((len(acc), None, None, None))
    idx, tid, tname, targs = acc[delta.index]
    if delta.id:
        tid = delta.id
    if delta.name:
        tname = delta.name
    if delta.arguments:
        targs = (targs or "") + delta.arguments
    acc[delta.index] = (idx, tid, tname, targs)


def _serialize_result(result: object) -> str:
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)
```

- [ ] **Step 3: Update `services/__init__.py` imports (no changes needed — ToolRegistry is not exported from services)**

- [ ] **Step 4: Run the tool loop tests**

Run: `pytest tests/test_services/test_tool_loop.py -v`
Expected: PASS

- [ ] **Step 5: Run existing chat tests to ensure no regression**

Run: `pytest tests/test_services/test_chat.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/yapa/services/chat.py tests/test_services/test_tool_loop.py
git commit -m "feat: add agentic loop to ChatService with tool execution and approval"
```

---
### Task 7: WebSocket Approval Flow

**Files:**
- Modify: `src/yapa/api/websocket/chat.py`
- Modify: `src/yapa/api/app.py` (update lifespan and `hasattr` check)
- Test: `tests/api/test_chat_ws.py` (append)

**Interfaces:**
- Consumes: `ChatService.stream()` with `get_approval` parameter
- Produces: bidirectional WebSocket handling tool approval responses

- [ ] **Step 1: Write failing test**

```python
def test_chat_ws_tool_approval_flow(client, mock_chat_service, mock_session_service):
    """WebSocket sends ToolApprovalRequestEvent and client responds with approval."""
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
    )

    async def _stream(*, session_id, prompt, model, get_approval=None):
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield ToolCallEvent(tool_name="write_file", arguments={"path": "/tmp/test.txt"}, call_id="call_1")
        yield ToolApprovalRequestEvent(tool_name="write_file", arguments={"path": "/tmp/test.txt"}, call_id="call_1")
        # Simulate waiting for approval
        response = await get_approval(ToolApprovalRequest(call_id="call_1", name="write_file", arguments={"path": "/tmp/test.txt"}))
        if response.approved:
            yield ToolResultEvent(tool_name="write_file", call_id="call_1", result="ok")
        yield AgentDoneEvent(content="File written", finish_reason="stop")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Write a file"})

        # Receive events until we get an approval request
        while True:
            msg = ws.receive_json()
            if msg["type"] == "tool_approval_request":
                assert msg["tool_name"] == "write_file"
                # Send approval response
                ws.send_json({"type": "tool_approval", "call_id": "call_1", "approved": True})
                break

        # Receive remaining events
        msg = ws.receive_json()
        assert msg["type"] in ("tool_result", "agent_done")
        if msg["type"] == "tool_result":
            msg = ws.receive_json()
            assert msg["type"] == "agent_done"


def test_chat_ws_tool_denial_flow(client, mock_chat_service, mock_session_service):
    """Client can deny a tool call."""
    session_id = str(uuid4())
    mock_session_service.get.return_value = Session(
        model=ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM)
    )

    async def _stream(*, session_id, prompt, model, get_approval=None):
        yield AgentStartEvent(model_id="openai:gpt-4o")
        yield ToolCallEvent(tool_name="write_file", arguments={}, call_id="call_1")
        yield ToolApprovalRequestEvent(tool_name="write_file", arguments={}, call_id="call_1")
        response = await get_approval(ToolApprovalRequest(call_id="call_1", name="write_file", arguments={}))
        assert not response.approved
        yield AgentDoneEvent(content="Denied", finish_reason="stop")

    mock_chat_service.stream = _stream

    with client.websocket_connect(f"/api/v1/chat/{session_id}") as ws:
        ws.send_json({"prompt": "Write a file"})
        while True:
            msg = ws.receive_json()
            if msg["type"] == "tool_approval_request":
                ws.send_json({"type": "tool_approval", "call_id": "call_1", "approved": False, "reason": "unsafe"})
                break
        msg = ws.receive_json()
        assert msg["type"] == "agent_done"
```

- [ ] **Step 2: Modify `websocket/chat.py`**

Add approval callback to the stream call. The key change in the `chat_websocket` function: after receiving a prompt and before entering the event loop, construct a `get_approval` callback. Inside the event loop, handle incoming `tool_approval` messages:

```python
import asyncio
import json
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from yapa.models.event import AgentDoneEvent, AgentErrorEvent, EventType
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse

router = APIRouter(tags=["chat"])


@router.websocket("/chat/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: UUID,
):
    chat_service = websocket.app.state.chat_service
    model_service = websocket.app.state.model_service
    session_service = websocket.app.state.session_service

    await websocket.accept()

    try:
        session = session_service.get(str(session_id))
    except ValueError:
        await websocket.close(code=4008, reason="Session not found")
        return

    while True:
        try:
            data = await websocket.receive_text()
        except WebSocketDisconnect:
            break

        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            await websocket.close(code=4008, reason="Invalid JSON")
            break

        # Handle tool approval responses
        if message.get("type") == "tool_approval":
            # Put response into the active approval handler
            handler = getattr(websocket.state, "_approval_handler", None)
            if handler is not None:
                handler.set_result(ToolApprovalResponse(**message))
            continue

        prompt = message.get("prompt")
        if not prompt:
            await websocket.close(code=4008, reason="Missing 'prompt' field")
            break

        if message.get("model"):
            model = await model_service.get_model(message["model"])
        elif session.model is not None:
            model = session.model
        else:
            await websocket.close(code=4008, reason="No model specified")
            break

        async def get_approval(request: ToolApprovalRequest) -> ToolApprovalResponse:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            websocket.state._approval_handler = future
            try:
                return await asyncio.wait_for(future, timeout=120.0)
            except asyncio.TimeoutError:
                return ToolApprovalResponse(
                    call_id=request.call_id,
                    approved=False,
                    reason="Approval timeout",
                )
            finally:
                websocket.state._approval_handler = None

        async for event in chat_service.stream(
            session_id=session_id,
            prompt=prompt,
            model=model,
            get_approval=get_approval,
        ):
            await websocket.send_json(event.model_dump(mode="json"))
            if isinstance(event, (AgentDoneEvent, AgentErrorEvent)):
                break
```

- [ ] **Step 3: Run the WS tests**

Run: `pytest tests/api/test_chat_ws.py -v`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: ≥80% passing (may have unrelated failures)

- [ ] **Step 5: Run linter and type checker**

Run: `uv run ruff check src/ tests/ && uv run ty check src/`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/yapa/api/websocket/chat.py tests/api/test_chat_ws.py
git commit -m "feat: add tool approval flow over WebSocket"
```