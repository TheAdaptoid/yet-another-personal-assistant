# Provider Layer Enhancements — Design Spec

## Overview

Add four capabilities to the provider layer: ModelData metadata, token usage tracking,
finish/stop reason propagation, and retry logic. These enable cost tracking, context
window management, agentic loop decisions, and production reliability at the service
layer above.

## ModelData metadata

### Fields

```python
class ModelData(BaseModel):
    id: str
    provider_id: str
    type: ModelType
    context_length: int | None = None
    max_output: int | None = None
    supports_tools: bool = False
    supports_vision: bool = False
    pricing: dict[str, float] | None = None  # e.g. {"input": 2.50, "output": 10.00} / MTok
```

### Population strategy

| Provider | Source | Notes |
|----------|--------|-------|
| OpenRouter | `/v1/models` API response | Response includes context length, pricing by modality, and model architecture info. `_format_model` expanded to populate new fields. |
| OpenAI | Static lookup table | `/v1/models` only returns IDs. A module-level dict maps well-known model IDs (gpt-4o, gpt-4o-mini, etc.) to their metadata. Fields remain `None` for unknown models. |
| LM Studio, Ollama | None (deferred) | Fields remain at defaults. Can be addressed separately when local model metadata becomes a priority. |

### Lookup table approach (OpenAI)

A flat dict keyed by model ID prefix or exact ID:

```python
_MODEL_METADATA: dict[str, dict[str, Any]] = {
    "gpt-4o": {"context_length": 128000, "max_output": 16384, "supports_tools": True, "supports_vision": True},
    "gpt-4o-mini": {"context_length": 128000, "max_output": 16384, "supports_tools": True, "supports_vision": True},
    "gpt-4-turbo": {"context_length": 128000, "max_output": 4096, "supports_tools": True, "supports_vision": True},
    ...
}
```

`_format_model` checks the lookup after keyword-based type inference:

```python
def _format_model(self, model_id: str) -> ModelData:
    model_type = ...  # existing keyword logic
    meta = _MODEL_METADATA.get(model_id, {})
    return ModelData(id=model_id, provider_id=self.id, type=model_type, **meta)
```

## Token usage tracking

### TokenUsage model

```python
class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

### StreamDelta additions

```python
class StreamDelta(BaseModel):
    ...
    finish_reason: str | None = None      # "stop" | "length" | "content_filter" | "tool_calls"
    usage: TokenUsage | None = None
```

### AssistantMessage addition

```python
class AssistantMessage(BaseMessage):
    ...
    usage: TokenUsage | None = None
```

### How streaming populates it

1. Pass `stream_options={"include_usage": True}` to `chat.completions.create()`.
   This makes the OpenAI API include a final usage chunk in the stream.

2. In `_stream_chat_impl`, on each chunk:
   - Read `chunk.choices[0].finish_reason` if present (set on final contentful chunk).
   - Read `chunk.usage` if present (set on final usage chunk).

3. Yield a final delta with both fields populated before the stream ends.

### How static chat populates it

`response.usage` is directly available on `ChatCompletion` responses.

```python
if response.usage:
    usage = TokenUsage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )
```

## Finish reason

No separate model — `finish_reason` is a `str | None` field on `StreamDelta`.

Values mirror the OpenAI API: `"stop"`, `"length"`, `"content_filter"`, `"tool_calls"`.

The agent loop above the provider can check `delta.finish_reason == "length"` to
decide whether the model hit its token limit and the conversation needs continuation.

## Retry logic

### Config

Add to `Config`:

```python
provider_max_retries: int = Field(default=2, ge=0)
```

### Provider

Pass to `AsyncOpenAI` constructor:

```python
self._client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=timeout,
    max_retries=max_retries,
)
```

All four current providers (OpenAI, LM Studio, Ollama, OpenRouter) use
`OpenAICompatibleProvider` and inherit this change automatically.

The OpenAI SDK retries on 429 (rate limit), 500, 502, 503 with exponential backoff
and jitter. No custom retry logic is needed for MVP.

## User cancellation

No changes to the provider module. The existing `AsyncGenerator` protocol provides
`aclose()`, which the conversation/service layer calls when the user presses cancel.
`GeneratorExit` propagates through the generator, which closes the OpenAI `AsyncStream`
and cancels the HTTP request. The conversation service only persists messages after
the generator completes, so interrupted streams leave no partial state.

## Files changed

| File | Change |
|------|--------|
| `src/yapa/config.py` | Add `provider_max_retries` |
| `src/yapa/models/inference.py` | Add `TokenUsage`; extend `ModelData`; extend `StreamDelta` |
| `src/yapa/models/message.py` | Add `usage` to `AssistantMessage` |
| `src/yapa/providers/openai_compat.py` | Accept `max_retries`; populate `finish_reason` + `usage`; pass `stream_options` |
| `src/yapa/providers/openrouter/provider.py` | Expand `_format_model` for OpenRouter metadata |
| (new file) `src/yapa/providers/openai/model_data.py` | Static lookup table for well-known OpenAI models |

## Out of scope

- Vision/multimodal message types
- `response_format` and `stop` in `InferenceParams`
- LM Studio / Ollama model metadata (deferred)
- Service-layer cost tracking or budgets
