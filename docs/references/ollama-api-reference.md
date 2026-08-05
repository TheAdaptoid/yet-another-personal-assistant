# Ollama API Reference

## Overview

- **Base URL**: `http://127.0.0.1:11434` (default, configurable via `OLLAMA_HOST`)
- **Authentication**: None (local service)
- **API Style**: Native REST API with OpenAI-compatible layer at `/v1/`
- **OpenAPI Spec**: Available at `http://localhost:11434/v1/swagger.json` (OpenAI compat) or via `ollama serve --openai-base-url`

## Endpoints

### Chat

Generate chat completion (non-streaming and streaming).

**POST `/api/chat`**

#### Request Body

```json
{
  "model": "llama3.1",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": true,
  "format": {},            // JSON schema for structured output
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "num_ctx": 4096,
    "num_predict": 128
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather info",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "think": true            // Enable reasoning traces
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Required. Model name (e.g., `"llama3.1"`, `"phi3"`, `"gemma2"`) |
| `messages` | array | Required. Conversation messages |
| `stream` | boolean | Default `false`. Stream response as SSE |
| `format` | object \| enum | JSON Schema object for structured output, or `"json"` for any JSON |
| `options` | object | Model hyperparameters (see below) |
| `tools` | array | JSON Schema functions for tool calling |
| `think` | boolean | Enable reasoning/thinking traces in `message.thinking` |
| `keep_alive` | string | How long to keep model in memory (e.g., `"5m"`, `"1h"`, `"-1"` for always) |
| `suffix` | string | (Deprecated) |

#### Options Object

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temperature` | float | 0.7 | 0–1, lower = more deterministic |
| `top_p` | float | 0.9 | Nucleus sampling probability |
| `top_k` | int | 40 | Top-k sampling |
| `min_p` | float | 0.0 | Minimum p for sampling |
| `num_ctx` | int | 2048 | Context window size (tokens) |
| `num_predict` | int | -1 | Max tokens to predict (-1 = unlimited) |
| `repeat_last_n` | int | 1 | Penalty for repetition |
| `repeat_penalty` | float | 1.1 | Repetition penalty factor |
| `seed` | int | 0 | Random seed (0 = random) |
| `tfs_z` | float | 1.0 | Tail-free sampling |
| `numa` | boolean | false | Enable NUMA support |
| `num_gpu` | int | 0 | Number of GPUs to use |
| `num_thread` | int | (auto) | Number of CPU threads |
| `batch` | int | 512 | Batch size |
| `flash_attention` | boolean | false | Use flash attention |
| `logit_bias` | object | — | Token-level logit bias |

### Generate

Generate completion from a prompt (non-chat, no conversational state).

**POST `/api/generate`**

#### Request Body

```json
{
  "model": "gemma2",
  "prompt": "Why is the sky blue?",
  "stream": true,
  "format": "json",
  "system": "You are a helpful assistant.",
  "template": "{{prompt}}",
  "context": [1, 2, 3],      // token IDs from previous call
  "raw": false,
  "keep_alive": "5m",
  "options": { "temperature": 0.7 }
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Required |
| `prompt` | string | Required |
| `stream` | boolean | Stream response |
| `format` | enum | `"json"`, `"json_object"`, or `"json_schema"` |
| `system` | string | System prompt |
| `template` | string | Custom template string |
| `context` | array[int] | Token IDs for multi-turn context |
| `raw` | boolean | Return raw response without formatting |
| `keep_alive` | string\|int | Same as `/api/chat` |
| `options` | object | Same options object as chat |

### Embeddings

Generate vector embeddings for input text.

**POST `/api/embed`**

#### Request Body

```json
{
  "model": "nomic-embed-text",
  "input": "The sky is blue.",
  "truncate": true,
  "normalize": true,
  "keep_alive": "5m"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | Required | Model name (e.g., `nomic-embed-text`) |
| `input` | string \| array[string] | Required | Text to embed |
| `truncate` | boolean | `true` | Truncate at model context length |
| `normalize` | boolean | `true` | L2-normalize the output vectors |
| `keep_alive` | string\|int | — | Model lifetime in memory |

#### Response

```json
{
  "model": "nomic-embed-text",
  "embeddings": [[-0.12, 0.34, ..., 0.56]],
  "prompt_eval_count": 8,
  "load_duration": 12345678
}
```

**Note**: When `input` is an array, returns multiple embedding arrays. Embeddings are L2-normalized by default.

### Create a Model

Create a model from a Modelfile.

**POST `/api/create`**

#### Request Body

```json
{
  "name": "my-finetuned-model",
  "path": "llama3.1",         // base model or OCI registry path
  "modfile": "...",           // Modelfile content
  "files": [
    {"path": "template", "content": "..."}
  ],
  "stream": false
}
```

### List Models

**GET `/api/tags`**

#### Response

```json
{
  "models": [
    {
      "name": "llama3.1:latest",
      "size": 2300000000000,
      "modified_at": "2024-07-24T10:00:00Z",
      "digest": "sha256:abcd1234...",
      "details": {
        "parent_model": "",
        "family": "llama",
        "families": ["llama"],
        "format": "gguf",
        "family_size": 8,         // in billions
        "parameter_size": "8B"
      }
    }
  ]
}
```

### Show Model Info

Get detailed info about a specific model.

**GET `/api/show?model=<name>&verbose=true`**

| Query Param | Description |
|-------------|-------------|
| `model` | Required. Model name |
| `verbose` | Include verbose output (parameters, modinfo) |

#### Response

```json
{
  "modinfo": {
    "format_version": "OMI_R1",
    "schema": {
      "model_format": "gguf",
      "model_type": "llama",
      "architecture": {
        "input": "string",
        "output": "string"
      }
    }
  },
  "parameters": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_p": 0.9,
    "num_ctx": 4096,
    "stop": ["</s>"]
  },
  "details": {
    "parent_model": "",
    "format": "gguf",
    "family": "llama",
    "families": ["llama"],
    "family_size": 8,
    "parameter_size": "8B",
    "quantization": "Q4_K_M"
  },
  "model_info": {
    "llama3.1": {
      "parent_model": "",
      "format": "gguf",
      "family": "llama",
      "families": ["llama"],
      "file_size": ...,
      "parameter_count": ...,
      ...
    }
  }
}
```

### Pull a Model

Download a model from the Ollama registry.

**POST `/api/pull`**

```json
{
  "name": "llama3.1:latest",
  "stream": true
}
```

### Push a Model

Push a model to the Ollama registry.

**POST `/api/push`**

```json
{
  "name": "myusername/my-model:latest",
  "stream": true
}
```

### Delete a Model

**DELETE `/api/delete`**

```json
{
  "name": "llama3.1:latest"
}
```

### List Running Models

**GET `/api/ps`**

Returns currently loaded models and their memory usage.

### Copy a Model

**POST `/api/copy`**

```json
{
  "source": "llama3.1:latest",
  "destination": "my-llama:copy"
}
```

## Streaming Format

Ollama uses Server-Sent Events (SSE) with `data: ` prefix. Each event contains a JSON object.

### Chat Stream Response

```
data: {"model":"llama3.1","created_at":"...","message":{"role":"assistant","content":"Hello"},"done":false}
data: {"model":"llama3.1","created_at":"...","message":{"role":"assistant","content":" world"},"done":false}
data: {"model":"llama3.1","created_at":"...","message":{"role":"assistant","content":""},"done":true,"context":[...],"eval_count":10,"prompt_eval_count":5,"eval_duration":12345678,"load_duration":9876543,"total_duration":22345678}
```

When `think: true`, reasoning appears as `message.thinking`:
```
data: {"model":"llama3.1","created_at":"...","message":{"role":"assistant","thinking":"Let me reason step by step...","content":""},"done":false}
```

### Generate Stream Response

```
data: {"model":"gemma2","created_at":"...","response":"The sky","done":false}
data: {"model":"gemma2","created_at":"...","response":" is blue.","done":false}
data: {"model":"gemma2","created_at":"...","response":null,"done":true,"context":[...],"eval_count":8,"prompt_eval_count":6,"eval_duration":12345678}
```

### Tool Call in Stream

Tool call chunks appear in `message.tool_calls` array within message deltas:
```json
{"message":{"role":"assistant","tool_calls":[{"index":0,"name":"get_weather","arguments":"{\"loc"}]}
```

### Embed Stream Response

```
data: {"model":"nomic-embed-text","created_at":"...","embedding":[0.1,0.2,...],"done":false}
data: {"model":"nomic-embed-text","created_at":"...","embedding":[0.3,0.4,...],"done":true,"prompt_eval_count":8,"total_duration":12345678}
```

## OpenAI-Compatible Layer

**POST `/v1/chat/completions`**

Fully OpenAI-compatible request/response schema. Streaming via standard SSE with `data: [DONE]` sentinel.

## Tool Calling

Tools are specified via the `tools` array in chat requests. Each tool must be a JSON Schema `function`:

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Search query"
            },
            "num_results": {
              "type": "integer",
              "minimum": 1,
              "maximum": 10
            }
          },
          "required": ["query"]
        }
      }
    }
  ]
}
```

Tool calls appear in response as:
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "index": 0,
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "search_web",
          "arguments": "{\"query\": \"current weather\"}"
        }
      }
    ]
  }
}
```

## Image Inputs

Ollama supports multimodal models (e.g., LLaVA, BakLLaVA, minicpm-v) via `images` array in the message:

```json
{
  "model": "llava",
  "messages": [
    {
      "role": "user",
      "content": "What's in this image?",
      "images": ["<base64-encoded-image-data>"]
    }
  ]
}
```

Image format: Base64-encoded string without data URI prefix.

## Reasoning / Thinking

Enable with `think: true` in chat requests. Reasoning text appears in `message.thinking` field of each streaming chunk:

```json
{"message":{"role":"assistant","thinking":"First, I need to evaluate...","content":""},"done":false}
```

Reasoning is **accumulated** in the `thinking` field of each successive chunk, not replaced.

## Error Responses

All errors return HTTP 4xx/5xx with JSON body:

```json
{
  "error": "model 'xyz' not found"
}
```

Common errors:
- `model not found` — Model not pulled locally
- `no slot available` — All workers busy
- `context length exceeded` — Input exceeds `num_ctx`

## Model Families

Common model families available via `ollama pull`:

| Family | Examples | Notes |
|--------|----------|-------|
| `llama3` | `llama3`, `llama3.1`, `llama3.2` | Latest Meta models, various sizes |
| `gemma2` | `gemma2`, `gemma2:9b` | Google's lightweight models |
| `gemma3` | `gemma3`, `gemma3:12b` | Latest Gemma generation |
| `phi3` | `phi3`, `phi3:mini`, `phi3:medium` | Microsoft's Phi series |
| `phi3.5` | `phi3.5` | Updated small model |
| `mistral` | `mistral`, `mistral-nemo` | Mistral AI models |
| `qwen2` | `qwen2`, `qwen2.5` | Alibaba's Qwen series |
| `qwen2.5-vl` | `qwen2.5vl` | Multimodal vision variant |
| `llava` | `llava`, `llava:13b` | Multimodal LLaVA models |
| `bakllava` | `bakllava` | LLaVA variant |
| `minicpm` | `minicpm-v`, `minicpm3` | Multilingual small models |
| `command-r` | `command-r`, `command-r-plus` | Cohere's command models |
| `dbrx` | `dbrx` | Databricks' MoE model |
| `deepseek-r1` | `deepseek-r1` | DeepSeek's reasoning model |
| `mxbai` | `mxbai-embed-large` | Embedding models |
| `nomic-embed` | `nomic-embed-text` | Embedding models |

## Modelfile Syntax

Custom models defined via Modelfiles (Dockerfile-like syntax):

```dockerfile
FROM llama3.1
SYSTEM "You are a helpful coding assistant."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
TEMPLATE """{{ .Prompt }}"""
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OLLAMA_HOST` | Host:port for API server (default `127.0.0.1:11434`) |
| `OLLAMA_MODELS` | Directory for models (default `~/.ollama`) |
| `OLLAMA_MAX_LOADED_MODEL_SIZE` | Max model VRAM in GB |
| `OLLAMA_NUM_PARALLEL` | Max parallel requests |
| `OLLAMA_MAX_LOADED_MODEL_COUNT` | Max models in memory |
| `OLLAMA_KEEP_ALIVE` | Default keep-alive duration |
| `OLLAMA_ORIGINS` | CORS origins (default `localhost,127.0.0.1`) |

## Source

- Documentation: https://docs.ollama.com/
- API reference: https://docs.ollama.com/api/
- OpenAPI spec: Available via `ollama serve`
