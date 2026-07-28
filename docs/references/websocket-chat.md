# WebSocket Chat API

The YAPA server exposes a WebSocket endpoint for streaming chat
responses. The server sends each response as a sequence of typed JSON
events. This document explains the protocol and shows how to use it from
JavaScript and Python.

## Connect to the endpoint

The WebSocket URL is:

```
ws://<host>:<port>/api/v1/chat/<session_id>
```

The default host is `127.0.0.1` and the default port is `8000`. You must
create a session before you connect. Use `POST /api/v1/sessions` to
create one. The response contains the session ID in the `Location`
header.

## Send a message

Each message to the server is a JSON object with one required field and
one optional field:

```
{
  "prompt": "your text here",
  "model": "provider_id:model_id"
}
```

- `prompt` (required): The text you want to send to the model.
- `model` (optional): A full model ID like `openai:gpt-4o`. If you do
  not send this field, the server uses the model stored on the session.
  If the session has no model, the server closes the connection.

You can send multiple prompts over the same connection. The server
responds to each prompt with its own sequence of events. The connection
stays open until you or the server closes it.

### Tool approval responses

When the model calls a tool that requires approval, the server sends a
`ToolApprovalRequestEvent`. You must respond with a tool approval
message on the same connection:

```
{
  "type": "tool_approval",
  "call_id": "call_abc123",
  "approved": true
}
```

Fields:

- `type` (required): Must be `"tool_approval"`.
- `call_id` (required): The `call_id` from the `ToolApprovalRequestEvent`.
- `approved` (required): `true` to allow the tool, `false` to deny it.
- `reason` (optional): A string explaining why the tool was denied. The
  reason is fed back to the model so it can adjust its behavior.

The server waits up to 120 seconds for the approval response. If the
response does not arrive in time, the tool call is denied with reason
`"Approval timeout"`.

## Receive events

The server sends one JSON object per event. Each event has a `type`
field, a `source` field, and a `timestamp` field. The `source` is always
`"agent"`. The `timestamp` is an ISO 8601 UTC string.

### AgentStartEvent

The server sends this event first. It contains the model that will
generate the response.

```
{
  "type": "agent_start",
  "source": "agent",
  "timestamp": "2026-07-27T12:00:00Z",
  "model_id": "openai:gpt-4o"
}
```

### TextEvent

The server sends one or more text chunk events. Each chunk is a piece of
the model response. Concatenate them to get the full response.

```
{
  "type": "text_chunk",
  "source": "agent",
  "timestamp": "2026-07-27T12:00:01Z",
  "content": "Hello"
}
```

### ReasoningEvent

Some models send reasoning or thinking content. The server emits this
event before the related text chunks.

```
{
  "type": "reasoning_chunk",
  "source": "agent",
  "timestamp": "2026-07-27T12:00:01Z",
  "content": "The user asked a greeting question..."
}
```

### AgentDoneEvent

The server sends this event when the response is complete. It contains
the full response text, a finish reason, and optional token usage.

```
{
  "type": "agent_done",
  "source": "agent",
  "timestamp": "2026-07-27T12:00:02Z",
  "content": "Hello! How can I help you today?",
  "finish_reason": "stop",
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 10,
    "total_tokens": 35
  }
}
```

Possible values for `finish_reason`:

- `"stop"`: The model finished naturally.
- `"length"`: The model reached the maximum token limit.
- `"content_filter"`: A content filter stopped the response.
- `null`: The model did not provide a reason.

The `usage` field is `null` when the provider does not report token
usage.

### AgentErrorEvent

The server sends this event when an error occurs during generation. The
connection stays open so you can send another prompt if needed.

When an error occurs, the partial response received so far is discarded.
The session is not updated. Send the prompt again to retry from the
last saved state.

```
{
  "type": "agent_error",
  "source": "agent",
  "timestamp": "2026-07-27T12:00:02Z",
  "message": "Model returned empty response"
}
```

### ToolCallEvent

The server sends this event when the model decides to call a tool. One
`ToolCallEvent` is emitted per tool call in the response. The model may
call multiple tools in a single response.

```
{
  "type": "tool_call",
  "source": "agent",
  "timestamp": "2026-07-27T12:00:02Z",
  "tool_name": "read_file",
  "arguments": {"path": "/home/user/data.txt"},
  "call_id": "call_abc123"
}
```

### ToolApprovalRequestEvent

The server sends this event before executing a tool that requires
approval (`needs_approval=True`). The client must respond with a tool
approval message (see the "Send a message" section above). The server
waits for the response before executing the tool.

```
{
  "type": "tool_approval_request",
  "source": "system",
  "timestamp": "2026-07-27T12:00:02Z",
  "tool_name": "write_file",
  "arguments": {"path": "/home/user/output.txt", "content": "data"},
  "call_id": "call_def456"
}
```

### ToolResultEvent

The server sends this event after a tool has been executed. The `result`
field contains the tool output on success or an error description on
failure. Errors are also fed back to the model as tool messages so the
model can retry.

```
{
  "type": "tool_result",
  "source": "system",
  "timestamp": "2026-07-27T12:00:03Z",
  "tool_name": "read_file",
  "call_id": "call_abc123",
  "result": "file contents here"
}
```

On tool execution failure:

```
{
  "type": "tool_result",
  "source": "system",
  "timestamp": "2026-07-27T12:00:03Z",
  "tool_name": "bash",
  "call_id": "call_ghi789",
  "result": {"error": "Command timed out after 60 seconds"}
}
```

## Event sequence

### Text-only response (no tool calls)

Each prompt produces one complete event sequence:

```
AgentStartEvent -> (TextEvent | ReasoningEvent)* -> AgentDoneEvent
```

### Response with tool calls

When the model calls tools, the server runs an agentic loop. The loop
executes each tool, feeds the results back to the model, and re-invokes
the model until it produces a text response. The loop runs up to 10
iterations.

Safe tools (no approval needed):

```
AgentStartEvent
  -> (TextEvent | ReasoningEvent)*
  -> ToolCallEvent
  -> ToolResultEvent
  -> (TextEvent | ReasoningEvent)*
  -> AgentDoneEvent
```

Dangerous tools (approval required):

```
AgentStartEvent
  -> (TextEvent | ReasoningEvent)*
  -> ToolCallEvent
  -> ToolApprovalRequestEvent
  <- client responds with tool approval
  -> ToolResultEvent
  -> (TextEvent | ReasoningEvent)*
  -> AgentDoneEvent
```

Multiple tool calls in one response:

```
AgentStartEvent
  -> ToolCallEvent (tool_1)
  -> ToolCallEvent (tool_2)
  -> ToolApprovalRequestEvent (tool_1)
  <- client approves
  -> ToolResultEvent (tool_1)
  -> ToolResultEvent (tool_2)
  -> (TextEvent | ReasoningEvent)*
  -> AgentDoneEvent
```

Denied tool:

```
AgentStartEvent
  -> ToolCallEvent
  -> ToolApprovalRequestEvent
  <- client denies with reason "wrong file"
  -> ToolResultEvent (result contains "Tool call denied: wrong file")
  -> (TextEvent | ReasoningEvent)* (model acknowledges denial)
  -> AgentDoneEvent
```

Max iterations reached (no text response after 10 rounds):

```
AgentStartEvent
  -> ToolCallEvent (repeated 10 times)
  -> AgentErrorEvent (message: "Max iterations reached")
```

### Error sequences

On error:

```
AgentStartEvent -> AgentErrorEvent
```

The connection stays open after an error. Send another prompt on the
same connection. Do not reconnect.

## Connection errors

The server closes the connection in these cases:

- **Session not found**: The session ID does not exist. The server
  closes immediately after connect.
- **Invalid JSON**: The message is not valid JSON.
- **Missing prompt**: The JSON object does not contain a `prompt` field.
- **No model**: The request has no `model` field and the session has no
  stored model.

The close code is `4008` for all error cases.

## JavaScript example

This example uses the browser `WebSocket` API and handles tool approval
requests.

```javascript
const HOST = "127.0.0.1";
const PORT = 8000;
const SESSION_ID = "your-session-uuid";

const ws = new WebSocket(`ws://${HOST}:${PORT}/api/v1/chat/${SESSION_ID}`);

ws.onopen = () => {
  ws.send(JSON.stringify({ prompt: "Read the file data.txt" }));
};

const output = document.getElementById("output"); // assumes HTML element

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  switch (data.type) {
    case "agent_start":
      console.log("Model:", data.model_id);
      break;
    case "text_chunk":
      output.textContent += data.content;
      break;
    case "reasoning_chunk":
      console.log("Reasoning:", data.content);
      break;
    case "tool_call":
      console.log(`Tool call: ${data.tool_name}(${JSON.stringify(data.arguments)})`);
      break;
    case "tool_approval_request":
      console.log(`Approval needed: ${data.tool_name}`);
      // Show approval UI, then respond
      ws.send(JSON.stringify({
        type: "tool_approval",
        call_id: data.call_id,
        approved: true,
      }));
      break;
    case "tool_result":
      console.log(`Tool result: ${JSON.stringify(data.result)}`);
      break;
    case "agent_done":
      console.log("Done:", data.finish_reason);
      if (data.usage) {
        console.log(
          `Tokens: ${data.usage.prompt_tokens} prompt + ` +
          `${data.usage.completion_tokens} completion`
        );
      }
      break;
    case "agent_error":
      console.error("Error:", data.message);
      break;
  }
};

ws.onclose = (event) => {
  if (event.code !== 1000) {
    console.error("Connection closed with code", event.code);
  }
};
```

## Python example

This example uses the `websockets` library and handles tool approval
requests automatically.

```python
import asyncio
import json

import websockets


async def chat():
    host = "127.0.0.1"
    port = 8000
    session_id = "your-session-uuid"

    uri = f"ws://{host}:{port}/api/v1/chat/{session_id}"

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "prompt": "Read and summarize the file data.txt",
        }))

        async for raw in ws:
            event = json.loads(raw)
            event_type = event["type"]

            if event_type == "agent_start":
                print(f"Model: {event['model_id']}")
            elif event_type == "text_chunk":
                print(event["content"], end="", flush=True)
            elif event_type == "reasoning_chunk":
                print(f"\n[Reasoning]: {event['content']}")
            elif event_type == "tool_call":
                print(
                    f"\n[Tool call]: {event['tool_name']}"
                    f"({json.dumps(event['arguments'])})"
                )
            elif event_type == "tool_approval_request":
                print(
                    f"\n[Approval needed]: {event['tool_name']}"
                    f"({json.dumps(event['arguments'])})"
                )
                # Auto-approve. In a real client, show a UI prompt.
                await ws.send(json.dumps({
                    "type": "tool_approval",
                    "call_id": event["call_id"],
                    "approved": True,
                }))
            elif event_type == "tool_result":
                result = event["result"]
                if isinstance(result, dict) and "error" in result:
                    print(f"\n[Tool error]: {result['error']}")
                else:
                    print(f"\n[Tool result]: {result}")
            elif event_type == "agent_done":
                print()
                print(f"Finish reason: {event.get('finish_reason')}")
                usage = event.get("usage")
                if usage:
                    print(
                        f"Tokens: {usage['prompt_tokens']} prompt + "
                        f"{usage['completion_tokens']} completion"
                    )
                break
            elif event_type == "agent_error":
                print(f"Error: {event['message']}")
                break


asyncio.run(chat())
```

## Multi-turn conversation

You can send multiple prompts over the same WebSocket connection. The
server maintains the conversation history on the session.

```python
import asyncio
import json

import websockets


async def multi_turn_chat():
    uri = "ws://127.0.0.1:8000/api/v1/chat/your-session-uuid"

    async with websockets.connect(uri) as ws:
        prompts = [
            "My name is Alice.",
            "What is my name?",
        ]

        for prompt in prompts:
            await ws.send(json.dumps({"prompt": prompt}))

            async for raw in ws:
                event = json.loads(raw)

                if event["type"] == "text_chunk":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "agent_done":
                    print()
                    break
                elif event["type"] == "agent_error":
                    print(f"Error: {event['message']}")
                    break


asyncio.run(multi_turn_chat())
```

The second prompt ("What is my name?") uses the same session. The
server includes the previous messages in the context. The model
remembers that your name is Alice.

## Full client example

See the test file at `tests/api/test_chat_ws.py` for a complete
reference of the WebSocket behavior, including error cases.