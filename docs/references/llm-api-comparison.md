# LLM Inference API Comparison Report

## Executive Summary

This report compares four LLM inference APIs for designing a unified provider interface: **Ollama**, **LM Studio**, **OpenRouter**, and **OpenAI**. All four support OpenAI-compatible `/v1/chat/completions` schemas, but differ significantly in architecture, streaming formats, reasoning/trace support, and model metadata richness.

---

## 1. Provider Classification

| Provider | Architecture | Cloud vs Local | Base URL Pattern | Authentication |
|----------|-------------|----------------|-----------------|----------------|
| **Ollama** | Native REST API (OpenAI-compatible layer) | Local | `http://localhost:11434` | None |
| **LM Studio** | Native REST + OpenAI-compatible layer (dual) | Local | `http://localhost:1234/v1` (OpenAI compat) or `http://localhost:1234` (native) | None |
| **OpenRouter** | Proxy/aggregator over OpenAI-compatible schema | Cloud | `https://openrouter.ai/api/v1` | Bearer token |
| **OpenAI** | Native API (Responses API + Chat Completions) | Cloud | `https://api.openai.com/v1` | Bearer token |

**Key Insight**: Ollama and LM Studio are local-first with no auth; OpenRouter is a proxy aggregator; OpenAI is the reference implementation.

---

## 2. Feature Matrix

| Feature | Ollama | LM Studio | OpenRouter | OpenAI |
|---------|--------|-----------|------------|--------|
| **Streaming** | Yes (SSE, `data: ` JSON lines) | Yes (SSE, typed events in native; OpenAI SSE in compat) | Yes (SSE, `data: ` JSON lines, `[DONE]`) | Yes (SSE, `data: ` JSON lines, `[DONE]`) |
| **Tool Calling** | Yes (`tools` array in `format`) | Yes (OpenAI compat + native) | Yes (OpenAI-compatible `tools`) | Yes (Chat Completions + Responses API) |
| **Image Inputs** | Yes (`images` array in `format`) | Yes (OpenAI compat `content` array) | Yes (OpenAI-compatible `content` array) | Yes (Chat Completions + Responses `image_url`) |
| **Reasoning Traces** | Yes (`think: true` → `thinking` field) | Partial (`reasoning` param: "low"\|"medium"\|"high") | Yes (`reasoning` object in request) | Yes (separate `reasoning` param, `reasoning_content` in response) |
| **Embeddings** | Yes (`/api/embed`, L2-normalized) | Yes (`/v1/embeddings`, OpenAI-compatible) | Yes (delegates to underlying models) | Yes (`/v1/embeddings`, multiple models) |

---

## 3. Streaming Formats

### 3.1 Ollama — NDJSON over SSE

Ollama streams as Server-Sent Events where each event's `data` field contains a JSON object. Each `message` chunk includes:

- `model`, `created_at`, `message` (`role`, `content`, `thinking` when `think=true`)
- `done` boolean — when `true`, `context` (token IDs) and final usage stats are included
- No `data: [DONE]` sentinel

```
data: {"model":"llama3.1","message":{"role":"assistant","content":"Hello"},"done":false}
data: {"model":"llama3.1","message":{"role":"assistant","content":" world"},"done":false}
data: {"model":"llama3.1","message":{},"done":true,"context":[...],"eval_count":2,"prompt_eval_count":5,"total_duration":...}
```

When `think=true`, reasoning appears as `message.thinking`:
```json
{"message":{"role":"assistant","thinking":"Let me reason...","content":""},"done":false}
```

### 3.2 LM Studio — Dual Streaming

- **OpenAI-compatible layer**: Standard SSE with `data: ` JSON chunks matching `chatCompletionChunk` schema, ends with `data: [DONE]`
- **Native SSE**: Typed events with event names:
  - `conversation.model.loaded`
  - `conversation.model.offloaded`
  - `conversation.epoch.details`
  - `chat-setup`
  - `user-message`
  - `assistant-database.*`
  - `memory-fragment.*`
  - `user-turn`
  - `assistant_turn.created`

### 3.3 OpenRouter — Standard SSE

- `data: ` JSON lines matching OpenAI `chatCompletionChunk` schema
- `data: [DONE]` sentinel at end
- Includes `providers` array in each chunk for multi-provider routing visibility

### 3.4 OpenAI — Standard SSE

**Chat Completions API**: `data: ` JSON lines matching `chatCompletionChunk` schema, `data: [DONE]` sentinel. Reasoning content appears in `choices[0].delta.reasoning_content` (not `content`).

**Responses API**: Different SSE format with `response.created`, `response.in_progress.attempt.diffed`, `response.output_text.delta`, etc.

---

## 4. Reasoning / Thinking Traces

| Provider | Request Parameter | Response Field | Format |
|----------|------------------|----------------|--------|
| **Ollama** | `think: true` | `message.thinking` | Plain text accumulated in streaming chunks |
| **LM Studio** | `reasoning: "low"\|"medium"\|"high"` | Via `reasoning_content` in compat layer; native events vary | Opaque in native mode |
| **OpenRouter** | `reasoning: {effort, max_tokens, exclude, enabled}` | `choices[0].delta.reasoning_content` | Plain text in streaming chunks |
| **OpenAI** | `reasoning: {effort, summary}` or `reasoning_effort` param | `choices[0].delta.reasoning_content` (Chat Completions); separate objects in Responses API | Plain text or structured summary |

**Design Note**: Reasoning traces are the most divergent. Ollama uses a boolean flag; OpenRouter/OpenAI use objects with effort levels. A unified model should expose `reasoning_effect: "off"|"low"|"medium"|"high"` and map to each provider's equivalent.

---

## 5. Tool Calling

### 5.1 Schema Shape

All four use JSON Schema `function` objects: `{name, description, parameters}` where `parameters` is a JSON Schema object (not a Pydantic model).

### 5.2 Streaming Tool Use

| Provider | Response Field | Notes |
|----------|----------------|-------|
| **Ollama** | `message.tool_calls` (accumulated) | Full tool_calls array appears in the final chunk; no incremental deltas |
| **LM Studio** | `choices[0].delta.tool_calls` | OpenAI-compatible streaming deltas |
| **OpenRouter** | `choices[0].delta.tool_calls` | OpenAI-compatible streaming deltas |
| **OpenAI** | `choices[0].delta.tool_calls` | Chat Completions: OpenAI-compatible. Responses API: different event types (`tool_call.delta`) |

### 5.3 Tool Execution Flow

OpenAI's **Responses API** introduces "tool_use" and "tool_result" as first-class objects. After a model requests a tool call, it returns a `response.failed` state and waits for the `tool_result`. This differs from Chat Completions where the caller must manually append results.

OpenRouter and LM Studio (via OpenAI compat layer) use the Chat Completions flow where caller appends results.

---

## 6. Model Metadata and Capability Representation

### 6.1 Ollama

- `GET /api/tags` returns list of models with `name`, `size`, `modified_at`, `digest`
- `GET /api/show?model=<name>` returns `modinfo` (model file config), `parameters` (default template params), `details` (family, format, family size)
- No structured "capabilities" field. Capabilities inferred from model family/config.
- `format` options in config.yaml indicate supported features (e.g., `parameter` defaults)

### 6.2 LM Studio

- `GET /v1/models` (OpenAI compat): Returns `{id, object, owned_by, created}`
- Native `GET /`: Returns models with `id`, `name`, `sizeLabel`, `quant`, `type`, `state`, `capabilities` (array of strings like `appbar-vision`, `function-calling`, etc.)
- Also `GET /health`, `GET /config`, `GET /config/instance`
- Capabilities expressed as string arrays — very explicit

### 6.3 OpenRouter

- `GET /v1/models` returns rich model objects:
  - `id`, `name`, `description`, `context_length`, `token_limit`, `max_completion_tokens`
  - `architecture`: `{modality, tokenizer, instruction_tokens}`
  - `pricing`: `{input, output, request, image, request_fixed, image_fixed}`
  - `supported_parameters`: array of strings (e.g., `temperature`, `top_p`, `reasoning`, `tool_choice`, etc.)
  - `per_request_limits`
  - `transform`
  - `benchmarks`: array of benchmark IDs
  - `community`
  - `created`, `owned_by`
- Very rich metadata — best for capability introspection

### 6.4 OpenAI

- `GET /v1/models` returns `{id, object, owned_by, created}` — minimal
- No per-model capabilities endpoint; capabilities are documented in API reference per model
- Model families inferred from ID naming conventions (e.g., `gpt-4o`, `o1-*`, `text-embedding-*`)`

**Design Note**: OpenRouter provides the richest structured metadata. LM Studio's native API provides explicit `capabilities` arrays. Ollama exposes raw config. OpenAI provides the least structured metadata.

---

## 7. Embeddings

| Provider | Endpoint | Output | Normalization |
|----------|----------|--------|---------------|
| **Ollama** | `POST /api/embed` | `embeddings` array (list of floats) | L2-normalized (documented) |
| **LM Studio** | `POST /v1/embeddings` | OpenAI-compatible `{object, data:[{embedding, index}], model, usage}` | Not explicitly documented |
| **OpenRouter** | `POST /v1/embeddings` | OpenAI-compatible | Delegates to underlying model |
| **OpenAI** | `POST /v1/embeddings` | OpenAI-compatible `{object, data:[{embedding, index}], model, usage}` | Not normalized — caller must L2-normalize |

**Design Note**: Ollama explicitly states L2 normalization; OpenAI does not normalize (common integration gotcha).

---

## 8. Unified Provider Model Recommendations

### 8.1 Core Schema Mapping

```python
class ModelData:
    id: str           # unique identifier (provider_id/model_name)
    name: str         # display name
    provider: str     # provider identifier
    description: Optional[str]
    context_length: int
    max_completion_tokens: int
    supports_streaming: bool = True
    supports_tool_calling: bool
    supports_image_input: bool
    supports_reasoning: bool
    supports_embeddings: bool
    reasoning_levels: List[str]  # ["low","medium","high"] or ["on","off"] or ["none"]
    embedding_dimensions: Optional[int]
    pricing: Optional[PricingInfo]
    raw_metadata: dict  # provider-specific extras
```

### 8.2 Reasoning Parameter Mapping

| Unified Value | Ollama | LM Studio | OpenRouter | OpenAI |
|---------------|--------|-----------|------------|--------|
| `off` | `think: false` | `reasoning: "off"` | `reasoning: {enabled: false}` | omit param |
| `low` | `think: true` | `reasoning: "low"` | `reasoning: {effort: "low"}` | `reasoning: {effort: "low"}` |
| `medium` | `think: true` | `reasoning: "medium"` | `reasoning: {effort: "medium"}` | `reasoning: {effort: "medium"}` |
| `high` | `think: true` | `reasoning: "high"` | `reasoning: {effort: "high"}` | `reasoning: {effort: "high"}` |

### 8.3 Streaming Event Normalization

Map all streaming formats to a common event shape:

```python
# Normalized streaming output
class StreamEvent(TypedDict):
    type: Literal["content.delta", "thinking.delta", "tool_call.delta", "tool_call.done", "metadata", "stream_end"]
    delta: Optional[str]  # incremental text content
    thinking: Optional[str]  # incremental reasoning text
    tool_call: Optional[PartialToolCall]
    usage: Optional[Usage]
    model: Optional[str]
```

### 8.4 Model Listing Strategy

1. Use OpenRouter/OpenAI `GET /v1/models` for cloud providers
2. Use LM Studio native `GET /` for local models with capabilities
3. Use Ollama `GET /api/tags` + `GET /api/show` for model details
4. Cache results per provider; refresh on demand

---

## 9. Key Divergences Requiring Abstraction

1. **Reasoning parameter shape** — boolean, enum, object, or absent
2. **Model metadata richness** — from empty objects to full specs
3. **Streaming format** — SSE JSON-lines vs typed SSE events vs NDJSON
4. **Authentication** — none (local) vs bearer tokens (cloud)
5. **API surfaces** — Ollama has a unique native schema; all others converge on OpenAI-compatible schema
6. **Embeddings normalization** — inconsistent across providers

---

## 10. Source URLs

- Ollama: docs.ollama.com
- LM Studio: lmstudio.ai/docs/developer
- OpenRouter: openrouter.ai/docs
- OpenAI: platform.openai.com/docs
