# OpenAI API Reference

## Overview

- **Base URL**: `https://api.openai.com/v1`
- **Authentication**: Bearer token (`Authorization: Bearer <key>`)
- **API Style**: Native API with two surfaces — **Chat Completions API** (legacy, widely used) and **Responses API** (newer, recommended for agents)
- **OpenAPI Spec**: `https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml` (84,000 lines)

## Key Concepts

### Two API Surfaces

1. **Chat Completions API** (`/v1/chat/completions`) — Stateless, widely supported. Best for integration with existing toolchains and OpenAI-compatible endpoints.
2. **Responses API** (`/v1/responses`) — Newer, stateful API designed for agentic workflows. Includes built-in tool handling, conversation state, and turn management. Recommended for new agentic applications.

### Model Naming

| Family | Examples | Notes |
|--------|----------|-------|
| `gpt-4o` | `gpt-4o`, `gpt-4o-mini` | Omni (text+image+audio) |
| `o1` | `o1`, `o1-mini`, `o1-pro` | Reasoning models |
| `o3` | `o3`, `o3-mini` | Latest reasoning models |
| `gpt-3.5-turbo` | `gpt-3.5-turbo`, `gpt-3.5-turbo-instruct` | Legacy, cost-effective |
| Embeddings | `text-embedding-3-small`, `text-embedding-3-large` | Embedding models |
| Audio | `whisper-1`, `tts-1`, `tts-1-hd` | Speech-to-text, text-to-speech |
| Image | `dall-e-2`, `dall-e-3` | Image generation |
| Moderation | `text-moderation-latest`, `omni-moderation-latest` | Content moderation |

---

## Chat Completions API

### Create Chat Completion

**POST `/v1/chat/completions`**

Accepts messages and returns either a full response (non-streaming) or SSE chunks (streaming).

#### Request Body

```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "reasoning": true,
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": true,
  "temperature": 0.7,
  "top_p": 0.9,
  "n": 1,
  "max_tokens": 4096,
  "presence_penalty": 0.0,
  "frequency_penalty": 0.0,
  "logit_bias": {},
  "user": "user-123",
  "stop": ["END"]
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | Required | Model to use |
| `messages` | array | Required | Conversation messages |
| `stream` | boolean | `false` | Enable SSE streaming |
| `stream_options` | object | — | `{"include_usage": true}` for final usage in streaming |
| `reasoning` | boolean | `false` | Include reasoning content in response |
| `tools` | array | — | Function/tool definitions for tool calling |
| `tool_choice` | string\|object | `"none"` | `"none"`, `"auto"`, `"required"`, or `{"type":"function","function":{"name":"fn_name"}}` |
| `parallel_tool_calls` | boolean | `true` (max parallel = min(N_tools, 4)) | Allow multiple tool calls in parallel |
| `temperature` | float | 1.0 | 0–2. Lower = more deterministic |
| `top_p` | float | 1.0 | Nucleus sampling (0–1) |
| `n` | int | 1 | Number of completions (deprecated in Responses API) |
| `max_tokens` | int | — | Max tokens to generate |
| `presence_penalty` | float | 0.0 | -2 to 2 |
| `frequency_penalty` | float | 0.0 | -2 to 2 |
| `logit_bias` | object | — | Token-level logit adjustments |
| `user` | string | — | User identifier for rate limits |
| `stop` | string\|array | — | Up to 4 sequences that stop generation |

#### Messages Format

```typescript
type Message = {
  role: "system" | "user" | "assistant" | "tool" | "function"
  content: string | ContentItem[]
  name?: string        // for role="function"/"tool" (deprecated in Responses API)
  tool_call_id?: string // for role="tool"
  tool_calls?: ToolCall[] // for role="assistant"
}

type ContentItem =
  | { type: "text", text: string }
  | { type: "image_url", image_url: { url: string, detail?: "auto"|"low"|"high" } }
```

#### Non-Streaming Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1721000000,
  "model": "gpt-4o-mini",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?",
        "tool_calls": null
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 12,
    "total_tokens": 32,
    "prompt_tokens_details": {
      "cached_tokens": 0,
      "audio_tokens": 0
    },
    "completion_tokens_details": {
      "reasoning_tokens": 0,
      "audio_tokens": 0,
      "accepted_prediction_tokens": 0,
      "rejected_prediction_tokens": 0
    }
  },
  "system_fingerprint": "fp_f4o123"
}
```

#### Streaming Response

SSE with `data: ` JSON lines. Final frame includes `usage` if `stream_options.include_usage: true`. Ends with `data: [DONE]`.

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1721000000,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-abc123","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":12,"total_tokens":32}}

data: [DONE]
```

#### Reasoning Content in Streaming

For reasoning models (o1, o3, DeepSeek R1 via OpenRouter), reasoning content appears in a separate field:

```json
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "reasoning_content": "Let me think about this..."
      },
      "finish_reason": null
    }
  ]
}
```

**Important**: `reasoning_content` appears before `content` — reasoning is generated first, then the visible response.

#### Tool Call in Streaming

```json
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": ""
            }
          }
        ]
      },
      "finish_reason": null
    }
  ]
}
```

Subsequent argument chunks:

```json
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "function": {
              "arguments": "{\"loc"
            }
          }
        ]
      }
    }
  ]
}
```

After the final argument chunk, a `finish_reason: "tool_calls"` is sent, and the stream continues until `data: [DONE]`.

---

## Responses API

### Create Response

**POST `/v1/responses`**

The newer API designed for agentic workflows. Manages conversation state automatically.

#### Request Body

```json
{
  "model": "gpt-4o",
  "input": "Explain quantum computing.",
  "instructions": "You are a helpful assistant.",
  "tools": [...],
  "tool_choice": "auto",
  "reasoning": {
    "effort": "low",      // "low" | "medium" | "high"
    "summary": "concise"  // "concise" | "detailed" | "none" | "auto"
  },
  "parallel_tool_calls": true,
  "temperature": 0.7,
  "top_p": 0.9,
  "max_output_tokens": 4096,
  "truncation": "auto",   // "auto" | "middle" | "none"
  "previous_response_id": "resp_abc123"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Required |
| `input` | string\|array | User input (string or content blocks) |
| `instructions` | string | System instructions |
| `tools` | array | Tool definitions |
| `tool_choice` | string\|object | Same options as Chat Completions |
| `reasoning` | object | `effort`: "low"\|"medium"\|"high"`, `summary`: "concise"\|"detailed"\|"none"\|"auto" |
| `parallel_tool_calls` | boolean | Allow parallel tool calls |
| `temperature` | float | 0–2 |
| `top_p` | float | 0–1 |
| `max_output_tokens` | int | Max output |
| `truncation` | enum | How to handle context window overflow |
| `previous_response_id` | string | Chain responses for conversation state |

### Responses API Streaming

**POST `/v1/responses?stream=true`**

SSE with typed events:

```
event: response.created
data: {"response":{"id":"resp_abc123","object":"response","status":"in_progress"}}

event: response.in_progress.attempt.diffed
data: {"response_id":"resp_abc123","item_id":"item_abc123","output_index":0,"effect":{"type":"msg"},"content_index":0}

event: response.output_text.delta
data: {"response_id":"resp_abc123","item_id":"item_abc123","output_index":0,"content_index":0,"delta":"Hello"}

event: response.output_text.done
data: {"response_id":"resp_abc123","item_id":"item_abc123","output_index":0,"content_index":0,"text":"Hello world!"}

event: response.done
data: {"response":{"id":"resp_abc123","status":"completed"}}
```

### Tool Search (Deferred Tools)

The Responses API supports `tool_search` for tools that may not complete immediately:

```json
{
  "tools": [
    {
      "type": "tool_search",
      "name": "wait_for_human"
    }
  ]
}
```

When a model invokes a `tool_search` tool, the response enters a `response.waiting` status and the caller must provide the tool result.

---

## Image Inputs

### Chat Completions Format

```json
{
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ...",
            "detail": "auto"   // "auto" | "low" | "high"
          }
        }
      ]
    }
  ]
}
```

### Responses API Format

```json
{
  "model": "gpt-4o",
  "input": [
    {"type": "input_text", "text": "What is in this image?"},
    {"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/4AAQ..."}
  ]
}
```

### Supported Formats

- JPEG, PNG, GIF, WEBP
- Base64 data URIs or HTTP(S) URLs
- Max resolution depends on `detail` setting:
  - `"low"`: 512px, ~200 tokens
  - `"auto"`: ~1600px, ~1000 tokens
  - `"high"`: ~1600px, ~1000 tokens (splits large images for higher fidelity)

---

## Embeddings

**POST `/v1/embeddings`**

#### Request

```json
{
  "model": "text-embedding-3-small",
  "input": "The sky is blue.",
  "encoding_format": "float",    // "float" | "abase64" | "nul"
  "dimensions": 512,             // Reduce dimensionality (supported by newer models)
  "user": "user-123"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Required. Embedding model |
| `input` | string\|array[string] | Text to embed |
| `encoding_format` | enum | `"float"`, `"base64"`, or `"nul"` |
| `dimensions` | int | Output dimensions (384–4096, depends on model) |
| `user` | string | User identifier |

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0093, 0.0045, ...]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 6
  }
}
```

**Note**: OpenAI embeddings are **not** L2-normalized by default. The caller must normalize if needed for similarity search.

### Supported Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `text-embedding-3-small` | 1536 (default), 512–1536 (configurable) | Cost-effective, high quality |
| `text-embedding-3-large` | 3072 (default), 256–3072 (configurable) | Higher quality, more expensive |
| `text-embedding-ada-002` | 1536 | Legacy model, being deprecated |

---

## Models Endpoint

**GET `/v1/models`**

Returns list of available models. Note: returns minimal metadata.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-4o",
      "object": "model",
      "owned_by": "openai",
      "created": 1721000000
    },
    {
      "id": "gpt-3.5-turbo",
      "object": "model",
      "owned_by": "openai",
      "created": 1677610600
    }
  ]
}
```

For rich model metadata, use the `GET /v1/models/{model}` endpoint or refer to the published model list.

### Retrieve Model

**GET `/v1/models/{model_id}`**

```json
{
  "id": "gpt-4o",
  "object": "model",
  "owned_by": "system",
  "created": 1721000000
}
```

**Note**: OpenAI's model API returns minimal metadata compared to OpenRouter. Capabilities must be inferred from documentation.

---

## Moderation

**POST `/v1/moderations`**

#### Request

```json
{
  "model": "omni-moderation-latest",
  "input": "I want to harm someone."
}
```

#### Response

```json
{
  "id": "modr-abc123",
  "model": "omni-moderation-latest",
  "results": [
    {
      "flagged": true,
      "categories": {
        "harassment": true,
        "hateful_content": false,
        "violence": true,
        ...
      },
      "category_scores": {
        "harassment": 0.85,
        "hateful_content": 0.01,
        "violence": 0.92,
        ...
      },
      "category_applied_input_types": {
        "harassment": ["text"],
        "hateful_content": [],
        ...
      }
    }
  ]
}
```

---

## Audio

### Speech-to-Text

**POST `/v1/audio/transcriptions`**

```json
{
  "file": "audio.mp3",
  "model": "whisper-1",
  "language": "en",
  "prompt": "Optional context for the transcription",
  "response_format": "json",   // "json" | "text" | "srt" | "verbose_json" | "vtt"
  "temperature": 0.0,          // 0–1
  "timestamp_granularity": "segment"
}
```

### Text-to-Speech

**POST `/v1/audio/speech`**

```json
{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "nova",            // "alloy" | "echo" | "fable" | "onyx" | "nova" | "shimmer"
  "response_format": "mp3",   // "mp3" | "opus" | "aac" | "flac"
  "speed": 1.0                // 0.25–4.0
}
```

---

## Image Generation

### DALL·E 3

**POST `/v1/images/generations`**

```json
{
  "model": "dall-e-3",
  "prompt": "A cat astronaut on the moon",
  "n": 1,
  "size": "1024x1024",   // "1024x1024" | "1024x1792" | "1792x1024"
  "quality": "standard", // "standard" | "hd"
  "style": "vivid",      // "vivid" | "natural"
  "response_format": "url", // "url" | "b64_json"
  "user": "user-123"
}
```

---

## File Upload & Batch Operations

### Upload File

**POST `/v1/files`**

```json
{
  "file": "file.jsonl",
  "purpose": "batch" | "fine-tune" | "assistants"
}
```

### Create Batch

**POST `/v1/batches`**

For bulk API processing with async completion.

---

## Error Responses

Standard OpenAI error format:

```json
{
  "error": {
    "message": "The model `gpt-4o` does not exist",
    "type": "NotFoundError",
    "param": "model",
    "code": "model_not_found"
  }
}
```

Common error types:

| Error Type | HTTP Status | Description |
|------------|-------------|-------------|
| `AuthenticationError` | 401 | Invalid API key |
| `PermissionDenied` | 403 | Insufficient permissions |
| `NotFoundError` | 404 | Resource not found |
| `RateLimitError` | 429 | Rate limit exceeded |
| `BadRequestError` | 400 | Invalid request |
| `InternalServerError` | 500 | Server error |
| `APIConnectionError` | 500 | Network failure |
| `TimeoutError` | 504 | Request timed out |
| `InsufficientQuotaError` | 429 | Billing quota exceeded |

---

## Rate Limits

Rate limits vary by model tier and are communicated via response headers:

| Header | Description |
|--------|-------------|
| `x-ratelimit-limit-tokens` | Token limit per minute |
| `x-ratelimit-remaining-tokens` | Remaining tokens in window |
| `x-ratelimit-reset-tokens` | Reset time for token window |
| `x-ratelimit-limit-requests` | Request limit per minute |
| `x-ratelimit-remaining-requests` | Remaining requests |
| `x-ratelimit-reset-requests` | Reset time for request window |

---

## Tool Calling

### Tool Definition

Tools use JSON Schema (draft 2020-12 by default):

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

### Tool Call Response (Chat Completions)

```json
{
  "id": "chatcmpl-abc123",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"San Francisco, CA\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

### Providing Tool Results (Chat Completions)

To continue after a tool call, append the assistant's message and the tool result:

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temperature\": 68}"}
  ]
}
```

### Responses API Tool Handling

In the Responses API, tool calls and results are managed as items:

1. Model returns a `function_call` item
2. You provide a `function_call_output` item
3. The conversation continues automatically

---

## Reasoning Models

### Requesting Reasoning

#### Chat Completions API

```json
{
  "model": "o1",
  "messages": [...]
}
```

Reasoning is automatically included for o1/o3 models when available.

#### Including Reasoning in Response

```json
{
  "model": "o1",
  "include_reasoning": true,
  "messages": [...]
}
```

#### Reasoning Parameter (Responses API)

```json
{
  "model": "o3-mini",
  "input": "...",
  "reasoning": {
    "effort": "medium",    // "low" | "medium" | "high"
    "summary": "concise"   // "concise" | "detailed" | "none" | "auto"
  }
}
```

| Parameter | Values | Description |
|-----------|--------|-------------|
| `effort` | `"low"`, `"medium"`, `"high"` | Reasoning compute budget |
| `summary` | `"concise"`, `"detailed"`, `"none"`, `"auto"` | Whether to summarize reasoning |

### Streaming Reasoning Content

Reasoning content in Chat Completions appears as `choices[0].delta.reasoning_content`. In the Responses API, it appears as `response.reasoning_content.delta` events.

---

## Authentication

Set your API key in the `Authorization` header:

```
Authorization: Bearer sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

API keys are obtained from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

---

## Usage Limits

OpenAI enforces different limits based on account tier:

| Tier | Requirements | Limits |
|------|-------------|--------|
| Free | Email verification | 3 RPM, 16.67 KTPM |
| Tier 1 | $5 credit purchase | 60 RPM, 300 KTPM |
| Tier 2 | $50 total usage | 500 RPM, 1 MTPM |
| Tier 3 | $1000 total usage | 1000 RPM, 2 MTPM |
| Tier 4 | $10000 total usage | 2000 RPM, 4 MTPM |
| Tier 5 | $50000 total usage | 2000 RPM, 6 MTPM (varies by model) |

RPM = requests per minute, KTPM = thousand tokens per minute.

---

## Source

- Documentation: https://platform.openai.com/docs
- API reference: https://platform.openai.com/docs/api-reference
- OpenAPI spec: https://github.com/openai/openai-openapi
- Reasoning guide: https://platform.openai.com/docs/guides/reasoning
- Tool calling guide: https://platform.openai.com/docs/guides/function-calling
- Image input guide: https://platform.openai.com/docs/guides/vision
- Streaming guide: https://platform.openai.com/docs/guides/streaming
