# LM Studio API Reference

## Overview

- **Base URL (OpenAI compatible)**: `http://localhost:1234/v1`
- **Base URL (Native)**: `http://localhost:1234`
- **Authentication**: None (local application)
- **API Style**: Dual API surface — native REST endpoints + OpenAI-compatible endpoints
- **OpenAPI Spec**: Available at `http://localhost:1234/v1/swagger.json` (OpenAI compat layer)

## API Surfaces

LM Studio provides two API surfaces:

1. **OpenAI-Compatible API** (`/v1/`) — Emulates OpenAI's REST API exactly. Recommended for integration with existing OpenAI SDKs.
2. **Native API** (`/`) — LM Studio's own REST endpoints with richer metadata and typed SSE events.

---

## OpenAI-Compatible Endpoints

### Chat Completions

**POST `/v1/chat/completions`**

Accepts and returns standard OpenAI Chat Completions schema.

#### Request

```json
{
  "model": "llama-3.1-8b-instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "reasoning": "low",          // LM Studio extension: "off"|"low"|"medium"|"high"
  "tools": [...],              // OpenAI-compatible tools
  "tool_choice": "auto",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048,
  "stream_options": {
    "include_usage": true
  }
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Required. Model identifier (LM file name) |
| `messages` | array | Required. OpenAI message array |
| `stream` | boolean | Enable SSE streaming |
| `reasoning` | enum | LM Studio extension for reasoning. Values: `"off"`, `"low"`, `"medium"`, `"high"` |
| `tools` | array | OpenAI-compatible function definitions |
| `tool_choice` | string\|object | `"none"`, `"auto"`, `"required"`, or `{"type":"function","function":{"name":"fn"}}` |
| `temperature` | float | 0–2, default 0.7 |
| `top_p` | float | 0–1, default 1.0 |
| `max_tokens` | int | Max output tokens |
| `n` | int | Number of choices (default 1) |
| `user` | string | End-user identifier |

#### Image Inputs (OpenAI compat layer)

For vision-capable models (via OpenAI compatibility):

```json
{
  "model": "moondream",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ]
}
```

### Embeddings

**POST `/v1/embeddings`**

Standard OpenAI-compatible embeddings endpoint.

#### Request

```json
{
  "model": "text-embedding-nomic-embed-text-v1.5",
  "input": "The sky is blue.",
  "encoding_format": "float",
  "dimensions": 512
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Required. Embedding model |
| `input` | string\|array[string] | Text(s) to embed |
| `encoding_format` | enum | `"float"` (default) or `"base64"` |
| `dimensions` | int | Output dimensionality (if supported) |

#### Response

Standard OpenAI-compatible response:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0023, -0.0093, ...]}
  ],
  "model": "text-embedding-nomic-embed-text-v1.5",
  "usage": {"prompt_tokens": 6, "total_tokens": 6}
}
```

### Models

**GET `/v1/models`**

Returns all loaded/available models in OpenAI-compatible format.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.1-8b-instruct",
      "object": "model",
      "owned_by": "lms",
      "created": 1721000000
    }
  ]
}
```

### Completions (Legacy)

**POST `/v1/completions`**

Standard OpenAI text completions endpoint.

### Edits (Legacy)

**POST `/v1/edits`**

### Moderations

**POST `/v1/moderations`**

### Audio Transcriptions

**POST `/v1/audio/transcriptions`**

### Audio Translations

**POST `/v1/audio/translations`**

### Image Generation

**POST `/v1/images/generations`**

### Image Editing

**POST `/v1/images/edits`**

---

## Native Endpoints

### Root

**GET `/`** — Returns API metadata, version, and status.

#### Response

```json
{
  "version": "1.5.0",
  "api_version": "v1",
  "api": {
    "v1": "Available at /v1/",
    "native": "Available at /"
  }
}
```

### List Models (Native)

**GET `/models`** — Returns models with full metadata including capabilities.

#### Response

```json
{
  "models": [
    {
      "id": "llama-3.1-8b-instruct",
      "name": "LLaMA 3.1 8B Instruct",
      "sizeLabel": "8B",
      "size": 4752105074,
      "sizeHumanReadable": "4.42 GB",
      "format": "gguf",
      "type": "text-generation",
      "state": {
        "loaded": false
        // if loaded:
        "gpu": true,
        "cpu": false,
        "offloaded": true,
        "contextLength": 8192,
        "maxContextLength": 131072,
        "minContextLength": 4096,
        "ropeScale": 1.0,
        "ropeFreqScale": 0.0
      },
      "quant": "Q4_K_M",
      "file": "...",
      "hash": "...",
      "description": "LLaMA 3.1 8B",
      "model": {
        "preprocessor": {...},
        "arch": {
          "args": [...],
          "name": "llama",
          "type": "LLM"
        },
        "tensorParallel": {...}
      },
      "loading": false,
      "capabilities": [
        "autocomplete",
        "chat-completion",
        "embeddings",          // for embedding models
        "image-completion",     // for vision models
        "vocabulary",
        "tokenizer"
      ],
      "metadata": {
        "transformers_version": "4.40.0",
        "model_type": "llama",
        ...
      }
    }
  ]
}
```

### Get Model Info

**GET `/models/{model_id}`** — Returns detailed info for a single model.

### Load a Model

**POST `/models/{model_id}/load`**

Loads a model into GPU/CPU memory.

#### Response

```json
{
  "status": "loading" | "loaded" | "error",
  "reload": false,
  "unload": false,
  "model": {...}
}
```

### Unload a Model

**POST `/models/{model_id}/unload`**

Unloads a model from memory.

### Get Backend

**GET `/backend`**

Returns GPU/memory backend info:

```json
{
  "backends": [
    {
      "type": "GPU",
      "vendor": "NVIDIA",
      "name": "NVIDIA GeForce RTX 4090",
      "VRAM": {...}
    },
    {
      "type": "CPU",
      "vendor": "Intel",
      ...
    }
  ]
}
```

### Get Instance Config

**GET `/config/instance`**

Returns the LM Studio instance configuration:

```json
{
  "server": {
    "port": 1234,
    "host": "0.0.0.0",
    "verbose": false,
    ...
  },
  ...
}
```

### Get Config

**GET `/config`**

### Native Chat

**POST `/chat`**

LM Studio's native chat endpoint with typed SSE streaming events.

#### Request

```json
{
  "model": "llama-3.1-8b-instruct",
  "messages": [...],
  "stream": true,
  "reasoning": "low",        // "off"|"low"|"medium"|"high"
  "tools": [...],
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048
}
```

#### Native SSE Events

The native chat endpoint uses typed SSE events. Each event has an `event` type:

```
event: chat-setup
data: {"model":"llama-3.1-8b-instruct"}

event: user-message
data: {"message":{"role":"user","content":"Hello!"}}

event: assistant-database.update
data: {"type":"update","data":{...}}

event: memory-fragment.update
data: {...}

event: user-turn
data: {...}

event: assistant_turn.created
data: {"id":"turn_abc123","model":"llama-3.1-8b-instruct"}

event: conversation.model.loaded
data: {"model":"llama-3.1-8b-instruct"}

event: conversation.model.offloaded
data: {"model":"llama-3.1-8b-instruct"}

event: conversation.epoch.details
data: {"epoch":0,"details":{...}}
```

### Native Generate

**POST `/generate`**

Non-chat text generation.

### Native Embeddings

**POST `/embed`**

LM Studio native embeddings endpoint.

### Health

**GET `/health`**

Returns health status of the server.

### Memory

**GET `/memory`**

Returns memory usage info.

### Search Paths

**GET `/search-paths`**

Returns configured search paths for model files.

---

## Streaming

### OpenAI-Compatible Streaming

Uses standard SSE with `data: ` JSON lines and `data: [DONE]` sentinel at end.

Each chunk follows the OpenAI `chatCompletionChunk` schema:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1721000000,
  "model": "llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "delta": {"role": "assistant", "content": "Hello"},
      "finish_reason": null
    }
  ]
}
```

### Reasoning in Streaming

With `reasoning: "low"` (or higher), reasoning content appears in:

```json
{
  "choices": [
    {
      "delta": {
        "reasoning_content": "Let me think about this..."
      }
    }
  ]
}
```

### Native SSE Streaming

The native `/chat` endpoint emits typed SSE events with richer context:

```
event: conversation.epoch.details
data: {"epoch":0,"total_epochs":1,"details":{"model_name":"llama-3.1-8b-instruct"}}

event: assistant_turn.created
data: {"id":"turn_0","model":"llama-3.1-8b-instruct","stats":{"prompt_tokens":10,"completion_tokens":0}}

event: response.output_text.delta
data: {"text":"Hello"}

event: response.output_text.done
data: {"text":"Hello world!"}
```

---

## Tool Calling

### Tool Definition (OpenAI-Compatible)

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

### Tool Call in Response

OpenAI-compatible response format:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1721000000,
  "model": "llama-3.1-8b-instruct",
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

### Tool Calls in Streaming

```json
{
  "choices": [
    {
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_w",
              "arguments": ""
            }
          }
        ]
      }
    }
  ]
}
```

**Important**: LM Studio streams tool call arguments as partial strings. The `arguments` field is built incrementally — you must concatenate all chunks for a given tool call before parsing as JSON.

---

## Image Inputs

### Vision Models

LM Studio supports vision-capable models (LLaVA, Phi-3-vision, Moondream, etc.) via the OpenAI-compatible layer.

### OpenAI-Compatible Image Input

```json
{
  "model": "llama-3.1-8b-vision",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image:"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
      ]
    }
  ]
}
```

### Native Image Input

For native endpoints, use base64-encoded image data in the `images` field of the content:

```json
{
  "model": "llama-3.1-8b-vision",
  "messages": [
    {
      "role": "user",
      "content": "Describe this image:",
      "images": ["/9j/4AAQ..."]  // base64 without prefix
    }
  ]
}
```

---

## Reasoning

### Parameter

LM Studio extends the OpenAI-compatible API with a `reasoning` parameter:

| Value | Description |
|-------|-------------|
| `"off"` | Disable reasoning (default for most models) |
| `"low"` | Minimal reasoning computation |
| `"medium"` | Balanced reasoning |
| `"high"` | Maximum reasoning depth |

### Requesting Reasoning

```json
{
  "model": "deepseek-r1",
  "messages": [...],
  "stream": true,
  "reasoning": "high"
}
```

### Streaming Reasoning Output

Reasoning content appears in `choices[0].delta.reasoning_content`:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1721000000,
  "model": "deepseek-r1",
  "choices": [
    {
      "index": 0,
      "delta": {
        "reasoning_content": "Let me think step by step..."
      },
      "finish_reason": null
    }
  ]
}
```

After reasoning is complete, regular content follows in `delta.content`.

---

## Error Responses

Standard OpenAI-compatible error format:

```json
{
  "object": "error",
  "message": "Model 'xyz' not found",
  "type": "NotFoundError",
  "param": null,
  "code": 404
}
```

Common error types:
- `NotFoundError` — Model not found or not loaded
- `BadRequestError` — Invalid request parameters
- `InternalServerError` — Runtime/server errors

---

## Capabilities

LM Studio exposes model capabilities as an array of strings in the native `/models` endpoint:

| Capability | Description |
|------------|-------------|
| `autocomplete` | Text completion / generation |
| `chat-completion` | Chat-based interaction |
| `embeddings` | Vector embeddings generation |
| `image-completion` | Image input / vision processing |
| `vocabulary` | Vocabulary/tokenizer access |
| `tokenizer` | Tokenization |

---

## Environment & Configuration

### Ports

| Port | Purpose |
|------|---------|
| 1234 | API server (default) |
| 3912 | Model file download server |

### Command Line

```bash
# Start server on all interfaces
lms server start --port 1234 --host 0.0.0.0

# Start with GPU disabled
lms server start --n-gpu-layers 0

# Start with verbose logging
lms server start --verbose
```

---

## Source

- Documentation: https://lmstudio.ai/docs/
- API reference: https://lmstudio.ai/docs/dev
- OpenAPI spec: http://localhost:1234/v1/swagger.json (when server is running)
