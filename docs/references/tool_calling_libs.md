# Tool Calling — OpenAI & OpenRouter SDKs

## Overview

Both OpenAI and OpenRouter use the **OpenAI-compatible Chat Completions format** for tool calling (also called function calling). Tools are declared as a JSON schema array in the request, the model responds with `tool_calls` when it wants to invoke one, and the caller executes the tool and feeds the result back as a `"role": "tool"` message.

OpenRouter's SDK also supports **server-side tools** (web search, datetime, image generation, etc.) via discriminated union types, but the standard function-calling path uses the same `type: "function"` format as OpenAI.

---

## 1. OpenAI Python SDK (`openai` v2.36.0)

### Request — Defining Tools

The `tools` parameter is `list[ChatCompletionToolParam]` — a TypedDict with shape:

```python
from openai.types.chat import ChatCompletionToolParam

tool: ChatCompletionToolParam = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {                           # JSON Schema object
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
        # "strict": True,  # optional — forces structured outputs
    },
}
```

The `function.parameters` field is `FunctionParameters` — typed as `dict[str, object]` at runtime, expected to be a valid JSON Schema object.

You can also use the older (deprecated) `functions` parameter, but `tools` is recommended.

### Request — Controlling Tool Choice

```python
from openai.types.chat import ChatCompletionToolChoiceOptionParam

tool_choice: ChatCompletionToolChoiceOptionParam
# Can be:
#   "auto"              — model decides (default)
#   "none"              — disable tools
#   "required"          — force tool use
#   {"type": "function", "function": {"name": "get_weather"}}  — force specific tool
```

### Non-Streaming Response

```python
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    tools=tools,
    tool_choice="auto",
    stream=False,
)

msg = response.choices[0].message
# msg.content         — str | None  (None when tool_calls is set)
# msg.tool_calls      — list[ChatCompletionMessageToolCall] | None

if msg.tool_calls:
    for tc in msg.tool_calls:
        tc.id              # str, e.g. "call_abc123"
        tc.type            # Literal["function"]
        tc.function.name       # str
        tc.function.arguments  # str (JSON — needs json.loads)
```

### Streaming Response

Streaming tool calls arrive incrementally across multiple chunks. The key type is `ChoiceDeltaToolCall`:

```python
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

# ChoiceDeltaToolCall fields:
#   index: int                          — position in the tool_calls array
#   id: str | None                      — set on the first chunk for each index
#   function: ChoiceDeltaToolCallFunction | None
#   type: Literal["function"] | None

# ChoiceDeltaToolCallFunction fields:
#   name: str | None                    — set on first chunk
#   arguments: str | None               — incremental, concatenate across chunks

# Usage:
async for chunk in response:
    delta = chunk.choices[0].delta
    if delta.tool_calls:
        for tc_delta in delta.tool_calls:
            # Accumulate per index:
            #   tc_delta.index      → which tool call this belongs to
            #   tc_delta.id         → set once per tool call
            #   tc_delta.function.name      → set once
            #   tc_delta.function.arguments → append across chunks
```

The `finish_reason` in the final chunk will be `"tool_calls"`.

### Message Format — Tool Result

After executing the tool, feed the result back:

```python
messages.append({
    "role": "tool",
    "tool_call_id": tc.id,       # must match the assistant's tool_call.id
    "content": json.dumps(result),
})
```

The `content` value should be a string (JSON-encoded if structured data).

---

## 2. OpenRouter Python SDK (`openrouter` v0.9.1)

### Two Approaches

OpenRouter can be used via **either**:

1. **The OpenAI client** with `base_url="https://openrouter.ai/api/v1"` — same tool format as OpenAI above. This is the approach their docs recommend for Python.
2. **The native OpenRouter SDK** — type-safe, uses `openrouter.chat.Chat.send()` / `send_async()`. The tool format uses Pydantic models.

Both hit the same backend (`/chat/completions`). The native SDK is auto-generated from OpenAPI specs and provides richer type hints.

### Approach 1: OpenAI Client (Recommended for YAPA's OpenAI-compatible path)

```python
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="...",
    base_url="https://openrouter.ai/api/v1",
)

# tools = same ChatCompletionToolParam list as OpenAI
# tool_choice = same as OpenAI
# response structure = same as OpenAI
```

This is already what LM Studio, Ollama, and OpenAI all use. OpenRouter's own Python docs use this approach, which means **the existing `OpenAILLMInferenceProtocol` could handle OpenRouter too** — but the user explicitly wants OpenRouter to use a separate implementation.

### Approach 2: Native OpenRouter SDK

#### Request — Defining Tools

Tools are `list[ChatFunctionTool]` — a discriminated union where `ChatFunctionToolFunction` is the standard function type:

```python
from openrouter.components import ChatFunctionToolFunction, ChatFunctionToolFunctionFunction

tool = ChatFunctionToolFunction(
    type="function",
    function=ChatFunctionToolFunctionFunction(
        name="get_weather",
        description="Get weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
        strict=None,  # OptionalNullable[bool]
    ),
    cache_control=None,  # Optional
)
```

The `tools` param accepts `list[ChatFunctionTool]` or `list[ChatFunctionToolTypedDict]`.

#### Request — Tool Choice

```python
from openrouter.components import ChatToolChoice

tool_choice = "auto"
# or "none"
# or ChatToolChoice(...) — but the type is a TypeAlias, not a Pydantic model
```

#### Non-Streaming Response (`send_async`)

```python
result = await client.chat.send_async(
    model="openai/gpt-4o",
    messages=messages,
    tools=tools,
    stream=False,
)
# Result type: ChatResult
#   choices: list[ChatChoice]
#     message: ChatAssistantMessage
#       content: str | None
#       tool_calls: list[ChatToolCall] | None

for choice in result.choices:
    msg = choice.message
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tc.id              # str
            tc.type            # Literal["function"]
            tc.function.name       # str
            tc.function.arguments  # str (JSON)
```

Note: `ChatChoice.finish_reason` is `Nullable[Union[Literal['tool_calls', 'stop', ...], str]]`.

#### Streaming Response (`send_async` with `stream=True`)

Returns `EventStreamAsync` — an async iterable yielding `ChatStreamChunk` objects:

```python
stream = await client.chat.send_async(
    model="openai/gpt-4o",
    messages=messages,
    tools=tools,
    stream=True,
)

async for chunk in stream:
    for choice in chunk.choices:
        delta = choice.delta
        # delta.content: str | None
        # delta.tool_calls: list[ChatStreamToolCall] | None
        if delta.tool_calls:
            for tc in delta.tool_calls:
                tc.index             # int
                tc.id                # str | None
                tc.function          # ChatStreamToolCallFunction | None
                    tc.function.name       # str | None
                    tc.function.arguments  # str | None (incremental!)
```

#### Message Format — OpenRouter tool result

Same as OpenAI:

```python
messages.append({
    "role": "tool",
    "tool_call_id": tc.id,
    "content": json.dumps(result),
})
```

---

## 3. SDK Comparison for Streaming Tool Calls

Both SDKs use the **same streaming chunk protocol** (`text/event-stream` with SSE). The delta structures are identical in shape:

| Concept | OpenAI | OpenRouter (native) |
|---|---|---|
| Tool type | `ChatCompletionToolParam` (TypedDict) | `ChatFunctionTool` (discriminated union) |
| Tool function spec | `FunctionDefinition` (TypedDict) | `ChatFunctionToolFunctionFunction` (Pydantic) |
| Non-streaming response | `ChatCompletionMessage.tool_calls` | `ChatAssistantMessage.tool_calls` |
| Streaming delta | `ChoiceDelta.tool_calls` | `ChatStreamDelta.tool_calls` |
| Delta tool call | `ChoiceDeltaToolCall` | `ChatStreamToolCall` |
| Delta tool call fn | `ChoiceDeltaToolCallFunction` | `ChatStreamToolCallFunction` |
| Tool result message | `{"role": "tool", "tool_call_id": ..., "content": ...}` | Same |

---

## 4. Key Differences

| Aspect | OpenAI Client | OpenRouter Native SDK |
|---|---|---|
| `tools` type | TypedDict dict literal | Pydantic model or TypedDict |
| Async interface | `await client.chat.completions.create()` with `stream=True/False` | `await client.chat.send_async()` with `stream=True/False` |
| Streaming return | `AsyncGenerator[ChatCompletionChunk, None]` (direct) | `EventStreamAsync` (async iterable wrapper) |
| Finish reason | On last `choices[0]` of the stream | On `choice.finish_reason` of `ChatStreamChoice` |
| Context manager | Not required | `async with OpenRouter(...) as client:` recommended |

---

## 5. Tool Call Delta Accumulation Pattern (Streaming)

Both SDKs require the same accumulation pattern for streaming tool calls:

```python
tool_call_deltas: dict[int, dict] = {}

async for chunk in stream:
    for choice in chunk.choices:
        if not choice.delta.tool_calls:
            continue
        for tc_delta in choice.delta.tool_calls:
            idx = tc_delta.index
            if idx not in tool_call_deltas:
                tool_call_deltas[idx] = {"id": "", "name": "", "arguments": ""}
            if tc_delta.id:
                tool_call_deltas[idx]["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    tool_call_deltas[idx]["name"] = tc_delta.function.name
                if tc_delta.function.arguments:
                    tool_call_deltas[idx]["arguments"] += tc_delta.function.arguments

# After stream ends:
complete_tool_calls = [
    ToolCall(id=val["id"], tool_name=val["name"],
             arguments=json.loads(val["arguments"]))
    for val in tool_call_deltas.values()
]
```
