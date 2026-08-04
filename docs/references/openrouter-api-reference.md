# OpenRouter API Reference

## Overview

- **Base URL**: `https://openrouter.ai/api/v1`
- **Authentication**: Bearer token (`Authorization: Bearer <key>`)
- **API Style**: OpenAI-compatible extensions to `/v1/chat/completions`
- **API Spec**: OpenAPI available at `https://openrouter.ai/api/v1/openapi.json`
- **Key Differentiator**: Proxy/aggregator layer providing access to 70+ providers and models behind a single interface, with unified billing, rate limits, and provider routing controls.

## Endpoints

### Chat Completions

**POST `/v1/chat/completions`**

Fully OpenAI-compatible request/response schema with additional OpenRouter-specific parameters for provider routing and reasoning control.

#### Request Body

```json
{
  "model": "anthropic/claude-3-5-sonnet-20241022",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."}
  ],
  "stream": true,
  "include_reasoning": true,
  "reasoning": {
    "effort": "high",
    "max_tokens": 4000,
    "exclude": false,
    "enabled": true
  },
  "provider": {
    "order": ["anthropic", "deepseek"],      // Preferred providers (in order)
    "sort": "throughput:300",               // Sort by throughput, latency, or cost
    "ignore": ["together"]                  // Skip these providers
  },
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "calculate_pi",
        "description": "Calculate pi to N digits",
        "parameters": {
          "type": "object",
          "properties": {
            "digits": {"type": "integer"}
          },
          "required": ["digits"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 4000,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "stop": ["END"],
  "user": "user-123",
  "transforms": ["extended_tokens"],         // Extend context with prompt lookup
  "parallel_tool_calls": true
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Required. Fully-qualified model ID (e.g., `"openai/gpt-4o-mini"`, `"anthropic/claude-3-5-sonnet-20241022"`) |
| `messages` | array | Required. OpenAI message array |
| `stream` | boolean | Enable SSE streaming |
| `include_reasoning` | boolean | Include reasoning tokens in response (non-streaming). Default: `false` |
| `reasoning` | object | Reasoning control. See section 6 below |
| `provider` | object | Provider routing controls. See section 7 below |
| `tools` | array | OpenAI-compatible function definitions |
| `tool_choice` | string\|object | `"none"`, `"auto"`, `"required"`, or explicit |
| `parallel_tool_calls` | boolean | Allow multiple tool calls in parallel |
| `temperature` | float | 0–2 |
| `top_p` | float | 0–1 |
| `max_tokens` | int | Max completion tokens |
| `frequency_penalty` | float | -2 to 2 |
| `presence_penalty` | float | -2 to 2 |
| `stop` | string\|array[string] | Stop sequences |
| `transforms` | array[string] | Transformations like `"extended_tokens"` to extend context |
| `user` | string | End-user identifier for analytics |

#### Reasoning Parameter (OpenRouter Extension)

```json
{
  "model": "deepseek/deepseek-chat",
  "messages": [...],
  "reasoning": {
    "effort": "high",           // "low" | "medium" | "high" — reasoning depth
    "max_tokens": 4000,         // Reserve tokens for reasoning trace
    "exclude": false,           // Exclude reasoning from response (streaming only)
    "enabled": true             // Enable/disable reasoning
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `effort` | enum | `"low"`, `"medium"`, `"high"` — reasoning depth/effort level |
| `max_tokens` | int | Maximum tokens allocated to reasoning |
| `exclude` | boolean | Exclude reasoning content from streaming output |
| `enabled` | boolean | Enable or disable reasoning generation |

### Models

**GET `/v1/models`**

Returns all available models through OpenRouter with rich metadata.

#### Response Schema

Each model object in the response contains:

```json
{
  "id": "anthropic/claude-3-5-sonnet-20241022",
  "name": "Claude 3.5 Sonnet",
  "description": "Claude 3.5 Sonnet is the most advanced model...",
  "context_length": 200000,
  "token_limit": 200000,
  "max_completion_tokens": 4096,
  "created": 1729500000,
  "owned_by": "Anthropic",
  "object": "model",
  "architecture": {
    "modality": "text+text",      // e.g., "text+image", "text+text"
    "tokenizer": "Anthropic",
    "instruction_tokens": 3       // System message token overhead
  },
  "pricing": {
    "input": 0.000003,            // $ per 1K tokens
    "output": 0.000015,
    "request": 0.0,
    "image": 0.0,
    "request_fixed": 0.0,
    "image_fixed": 0.0,
    "web_search": 0.0,
    "internal_image": 0.0,
    "input_cache_read": 0.0,
    "cached": 0.0,
    "cache_read": 0.0,
    "cache_write": 0.0
  },
  "supported_parameters": [
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "logit_bias",
    "frequency_penalty",
    "presence_penalty",
    "tools",
    "stream",
    "json",
    "tool_choice",
    "n",
    "include_reasoning",
    "reasoning",
    "transforms",
    "parallel_tool_calls"
  ],
  "per_request_limits": {
    "prompt_tokens": 200000,
    "completion_tokens": 4096
  },
  "transform": [],
  "benchmarks": [
    "ai_blitz_xiii",
    "arena",
    "gpqa",
    "humaneval",
    "math_spelling",
    "mmlu",
    "sciencedraw",
    "webarena"
  ],
  "community": {},
  "status": "available"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Fully-qualified provider/model identifier |
| `name` | string | Human-readable model name |
| `description` | string | Model description |
| `context_length` | int | Max context window in tokens |
| `token_limit` | int | Same as context_length for most providers |
| `max_completion_tokens` | int | Max output tokens |
| `created` | int | Unix timestamp of model creation |
| `owned_by` | string | Provider/owner name |
| `architecture.modality` | string | Input modalities (e.g., `"text+text"`, `"text+image"`) |
| `pricing` | object | Per-1K-token costs in USD |
| `supported_parameters` | array[string] | Parameters this model supports |
| `per_request_limits` | object | Request-level token limits |
| `benchmarks` | array[string] | Benchmark names model was evaluated on |

### Get Model Endpoint

**GET `/v1/models/{model_id}`** — Returns a single model's metadata.

### Completions (Legacy)

**POST `/v1/completions`**

OpenAI-compatible text completions endpoint.

### Embeddings

**POST `/v1/embeddings`**

OpenAI-compatible embeddings endpoint. Delegates to underlying model's embedding support.

### Edits

**POST `/v1/edits`** — OpenAI-compatible edits endpoint.

### Moderations

**POST `/v1/moderations`** — OpenAI-compatible moderations endpoint.

### Chat Ids

**GET `/v1/chat`** — List recent chat sessions.

### Generation

**GET `/v1/generation`** — Get generation details by ID.

---

## Provider Routing

OpenRouter's key differentiator is its multi-provider routing system.

#### Provider Object

```json
{
  "provider": {
    "order": ["anthropic", "openai"],     // Try these providers first (priority order)
    "sort": "throughput:300",            // Sort by: latency, throughput, or cost
    "ignore": ["google"],                // Never use these providers
    "data_preference": "auto",           // "auto" | "allow" | "prefer" | "require"
    "ignore_addresses": []               // Specific provider URLs to skip
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `order` | array[string] | Providers to try first, in priority order. Names must match OpenRouter provider slugs |
| `sort` | string | Sort remaining providers by: `"latency"` (ms), `"throughput"` (tokens/sec), or `"cost"` (USD). Append `:<max_count>` to limit |
| `ignore` | array[string] | Provider slugs to exclude from routing |
| `data_preference` | enum | `"auto"`, `"allow"`, `"prefer"`, `"require"` — preference for providers with data-enabled pricing (lower cost) |
| `ignore_addresses` | array[string] | Specific provider endpoint addresses to skip |

#### Common Provider Slugs

| Provider Slug | Examples |
|---------------|----------|
| `anthropic` | Claude 3/3.5/3.7 models |
| `openai` | GPT-4o, GPT-4-turbo, o1, GPT-3.5 |
| `google` | Gemini 1.5/2.0 Flash, Gemini Pro |
| `deepseek` | DeepSeek Chat, DeepSeek Coder |
| `meta` | Llama 3/3.1 models |
| `cohere` | Command R, Command R+ |
| `mistral` | Mixtral, Mistral 7B/8x7B |
| `nvidia` | Nemotron, NVLM-D |
| `together` | Together-hosted models |
| `fireworks` | Firewalled.ai models |
| `nebius` | Nebius-hosted models |

---

## Streaming

### SSE Format

OpenRouter uses standard SSE with `data: ` JSON lines and `data: [DONE]` sentinel:

```
data: {"id":"chatcmpl-...", "object":"chat.completion.chunk", "choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...", "choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}

data: [DONE]
```

### Reasoning in Streaming

When `reasoning.enabled: true`, reasoning content streams as:

```json
{
  "choices": [
    {
      "delta": {
        "reasoning_content": "Let me think about this step by step..."
      }
    }
  ]
}
```

If `reasoning.exclude: true`, reasoning tokens are consumed but not streamed to the client.

### Tool Call Streaming

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
              "name": "search_web"
            }
          }
        ]
      }
    }
  ]
}
```

Followed by incremental argument chunks:

```json
{
  "choices": [
    {
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "function": {
              "arguments": "{\"query\": \"current "
            }
          }
        ]
      }
    }
  ]
}
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

```json
{
  "id": "req_abc123",
  "model": "openai/gpt-4o-mini",
  "object": "chat.completion",
  "created": 1729500000,
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

---

## Image Inputs

### Vision Models

OpenRouter supports image inputs for vision-capable models (e.g., Anthropic Claude 3 with vision, Google Gemini, OpenAI GPT-4V).

### Content Array Format

```json
{
  "model": "anthropic/claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What objects are in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQ..."
          }
        }
      ]
    }
  ]
}
```

Supported image content types:
- `data:image/jpeg;base64,...`
- `data:image/png;base64,...`
- `data:image/gif;base64,...`
- `data:image/webp;base64,...`
- Remote URLs (`https://...`) — depends on provider

Image input pricing is tracked separately in the `pricing.image` field of the model.

---

## Reasoning Guide

OpenRouter supports reasoning for certain models (e.g., DeepSeek R1, OpenAI o1, o3):

### Basic Reasoning

```json
{
  "model": "deepseek/deepseek-r1",
  "messages": [...]
}
```

### Explicit Reasoning Control

```json
{
  "model": "openai/o1-preview",
  "messages": [...],
  "reasoning": {
    "effort": "high",
    "max_tokens": 4000,
    "exclude": false
  }
}
```

### Including Reasoning in Non-Streaming Responses

```json
{
  "model": "deepseek/deepseek-r1",
  "include_reasoning": true,
  "messages": [...]
}
```

The `reasoning` field appears in the response as `choices[0].message.reasoning_content`.

---

## Transforms

OpenRouter supports transforms to modify prompt handling:

| Transform | Description |
|-----------|-------------|
| `"extended_tokens"` | Extends the context window via prompt lookup techniques |
| `"truncate"` | Truncates input to fit within context limits |

```json
{
  "transforms": ["extended_tokens"],
  "model": "anthropic/claude-3-5-sonnet-20241022"
}
```

---

## Error Responses

Standard OpenAI-compatible error format, with additional `provider` and `is_retryable` fields:

```json
{
  "error": {
    "message": "Model not found or not available",
    "type": "NotFoundError",
    "param": null,
    "code": 404,
    "provider": "openrouter",
    "is_retryable": false
  }
}
```

Common error types:
- `AuthenticationError` — Invalid or missing API key (401)
- `RateLimitError` — Rate limit exceeded (429)
- `NotFoundError` — Model not found or not available (404)
- `InsufficientBalanceError` — Account needs credits (402)
- `ProviderUnavailableError` — All selected providers failed (503)
- `ContextLengthViolationError` — Input exceeds model context (400)

---

## Rate Limiting & Headers

### Response Headers

All responses include OpenRouter-specific headers:

| Header | Description |
|--------|-------------|
| `x-ratelimit-limit-tokens` | Token rate limit |
| `x-ratelimit-remaining-tokens` | Remaining tokens in window |
| `x-ratelimit-reset-tokens` | Token window reset time |
| `x-request-id` | Unique request ID for debugging |
| `x-provider` | Which provider fulfilled the request |
| `x-usage-completed` | Tokens used by this completion |

### Free Tier

OpenRouter provides a free tier with rate limits. Paid usage removes limits. Rate limits are per-provider and per-token-based.

---

## Authentication

Include your API key in the `Authorization` header:

```
Authorization: Bearer sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

API keys are obtained from [openrouter.ai/keys](https://openrouter.ai/keys).

---

## Model ID Format

All model IDs follow the pattern `<provider>/<model-name>`:

| Provider Prefix | Description |
|-----------------|-------------|
| `anthropic/` | Claude models |
| `openai/` | GPT, o1, o3, embeddings |
| `google/` | Gemini models |
| `deepseek/` | DeepSeek chat and reasoning |
| `meta-llama/` | Llama 3/3.1 models |
| `cohere/` | Command R models |
| `mistralai/` | Mixtral and Mistral models |
| `nvidia/` | Nemotron models |
| `nebius/` | Nebius-hosted models |
| `microsoft/` | Phi models |
| `sophos/` | Various models |

---

## Source

- Documentation: https://openrouter.ai/docs
- API reference: https://openrouter.ai/docs/api-reference
- OpenAPI spec: https://openrouter.ai/api/v1/openapi.json
- Reasoning guide: https://openrouter.ai/docs/reasoning
- Provider guide: https://openrouter.ai/docs/providers
