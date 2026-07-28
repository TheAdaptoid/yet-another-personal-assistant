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

## Event sequence

Each prompt produces one complete event sequence:

```
AgentStartEvent -> (TextEvent | ReasoningEvent)* -> AgentDoneEvent
```

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

This example uses the browser `WebSocket` API.

```javascript
const HOST = "127.0.0.1";
const PORT = 8000;
const SESSION_ID = "your-session-uuid";

const ws = new WebSocket(`ws://${HOST}:${PORT}/api/v1/chat/${SESSION_ID}`);

ws.onopen = () => {
  ws.send(JSON.stringify({ prompt: "What is the weather in London?" }));
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

This example uses the `websockets` library.

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
            "prompt": "What is the weather in London?",
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