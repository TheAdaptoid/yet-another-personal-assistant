# Agentic Loops and Tool Calling

Date: 2026-07-27

## Overview

Add an agentic loop to ChatService that detects tool calls in model responses,
executes the corresponding tools, feeds results back to the model, and re-invokes
until the model produces a text response or hits a max iteration limit.

## Architecture

```
WebSocket Client
      ↕ JSON events
API Layer (websocket/chat.py)
      ↕ AsyncGenerator[Event] + approval callback
ChatService (services/chat.py)
      ↕ messages + tools
Provider (stream_chat)
      ↕ StreamDelta (tool_calls)
ToolRegistry (tools/registry.py)
      ↕ lookup & execute
Tool instances (tools/core/*.py)
```

### New dependencies

- `ChatService` gains a `tools: ToolRegistry` constructor parameter.
- `ChatService.stream()` gains an optional `get_approval: ToolApprovalGetter` callable.
- `ToolApprovalGetter = Callable[[ToolApprovalRequest], Awaitable[ToolApprovalResponse]]`

### What does not change

- `Session` model — no new fields. Tools are global, not per-session.
- `Provider` layer — already passes `tools` parameter, already parses tool call deltas.
- `SessionService`, `ModelService` — no changes.
- `Config` — no changes.

## Event Protocol

### New event types

| Event | Fields | When emitted |
|---|---|---|
| `ToolCallEvent` | `tool_name`, `arguments`, `call_id` | After model finishes a tool call delta stream |
| `ToolApprovalRequestEvent` | `tool_name`, `arguments`, `call_id` | Before executing a dangerous tool |
| `ToolResultEvent` | `tool_name`, `call_id`, `result` | After tool execution (success or error) |

### WebSocket message types (client → server)

| Type | Format | When |
|---|---|---|
| Chat prompt | `{"prompt": "...", "model": "..."}` | Normal prompt |
| Approval response | `{"type": "tool_approval", "call_id": "...", "approved": true, "reason": "..."}` | In response to `ToolApprovalRequestEvent` |

### Sequence: dangerous tool approved

```
Server → Client: ToolCallEvent {tool_name: "write_file", ...}
Server → Client: ToolApprovalRequestEvent {tool_name: "write_file", call_id: "call_1", ...}
Client → Server: {"type": "tool_approval", "call_id": "call_1", "approved": true}
Server → Client: ToolResultEvent {tool_name: "write_file", call_id: "call_1", result: "ok"}
Server → Client: TextEvent (model's next response)
```

### Sequence: dangerous tool denied

```
Server → Client: ToolCallEvent {tool_name: "write_file", ...}
Server → Client: ToolApprovalRequestEvent {tool_name: "write_file", ...}
Client → Server: {"type": "tool_approval", "call_id": "call_1", "approved": false, "reason": "wrong directory"}
Server → Client: ToolResultEvent {tool_name: "write_file", call_id: "call_1", result: "Tool call denied: wrong directory"}
Server → Client: TextEvent (model acknowledges denial)
```

## Agentic Loop

Inside `ChatService.stream()`:

```
loop (max 10 iterations):
    1. Call provider.stream_chat(model, messages, tools, params)
    2. Collect all deltas into content_buffer + tool_calls list
    3. Append AssistantMessage(content, tool_calls) to messages
    4. If no tool_calls → persist session, yield AgentDoneEvent, return
    5. For each tool_call:
       a. Look up tool in ToolRegistry
       b. Yield ToolCallEvent
       c. If tool.needs_approval:
            yield ToolApprovalRequestEvent
            response = await get_approval(request)
            if denied → append ToolMessage("denied: reason") to messages, continue
       d. Execute: result = await tool.execute(**args)
       e. Yield ToolResultEvent
       f. Append ToolMessage(result) to messages
    6. Loop back to step 1 with updated messages

After loop: yield AgentErrorEvent("max iterations reached")
```

### Key details

- Provider already supports `tools` parameter — pass `self._tools.list_tools()`.
- The model sees all tool results as `ToolMessage` in history.
- On approval timeout, the callback raises `asyncio.TimeoutError` → caught → `ToolMessage("Error: approval timeout")` fed back to model.
- The `prompt` (UserMessage) is persisted only once when the loop terminates, not on every iteration.
- Tool errors are caught and fed back as `ToolMessage(content=f"Error: {e}")` so the model can retry.
- Unknown tool names are fed back as `ToolMessage(content=f"Unknown tool: {name}")`.

## Tool Implementations

Seven tools in `tools/core/`, one file per tool:

| File | Tool | `needs_approval` | Signature |
|---|---|---|---|
| `tools/core/calculator.py` | `calculator` | `False` | `expression: str` |
| `tools/core/read_file.py` | `read_file` | `False` | `path: str, offset: int?, limit: int?` |
| `tools/core/write_file.py` | `write_file` | `True` | `path: str, content: str` — refuses if parent dir does not exist |
| `tools/core/edit_file.py` | `edit_file` | `True` | `path: str, old_string: str, new_string: str` |
| `tools/core/grep.py` | `grep` | `False` | `pattern: str, path: str, include: str?` |
| `tools/core/list_dir.py` | `list_dir` | `False` | `path: str` |
| `tools/core/bash.py` | `bash` | `True` | `command: str` — runs via `subprocess` |

All registered in `tools/core/__init__.py` via a `default_tools()` function that returns `list[Tool]`. The `ToolRegistry` in `app.py` is initialized with `default_tools()`.

## Testing

### New test files

| File | What it tests |
|---|---|
| `tests/tools/test_calculator.py` | calculator tool execution |
| `tests/tools/test_file_tools.py` | read, write, edit, grep, list_dir, bash tools |
| `tests/tools/test_registry.py` | ToolRegistry register/lookup/list |
| `tests/test_services/test_tool_loop.py` | ChatService agentic loop with mocked tools + approval |

### Key test scenarios for the agentic loop

1. Model returns text only (no tools) → normal flow, no loop
2. Model calls one safe tool → tool executed, result fed back, model responds with text
3. Model calls one dangerous tool → approval requested, approved → tool executed
4. Model calls one dangerous tool → approval requested, denied → model sees denial, responds
5. Model calls multiple tools in one response → all executed sequentially
6. Model calls unknown tool → error fed back, model can recover
7. Tool execution fails → error fed back, model can retry
8. Max iterations reached → AgentErrorEvent
9. Approval timeout → error fed back to model

All tests use mocked provider (no real API calls) and mocked approval callback.