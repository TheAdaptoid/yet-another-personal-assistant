# Live API Response Schemas — Discrepancy Report

> Captured 2026-08-05 from real OpenRouter, LM Studio, and Ollama APIs.
> Base: `docs/specs/2026-07-31-provider-rework-requirements.md`

## 1. Reasoning Field Names (CRITICAL — SDK-verified)

Verified via the OpenAI SDK's parsed objects (not just raw JSON):

| Provider | SDK message field | SDK delta field | Current code reads | Status |
|---|---|---|---|---|
| **OpenRouter** | `msg.reasoning` | `delta.reasoning` | `reasoning_content` | ⚠️ **MISMATCH — reasoning dropped** |
| **LM Studio** | `msg.reasoning_content` | `delta.reasoning_content` | `reasoning_content` | ✅ OK |
| **Ollama (OpenAI-compat)** | `msg.reasoning` | `delta.reasoning` | `reasoning_content` | ⚠️ **MISMATCH — reasoning dropped** |
| **Ollama (native)** | `msg.thinking` | `msg.thinking` | `thinking` | ✅ OK |

**SDK evidence (OpenRouter):**
```
SDK-parsed message model_dump():
  {"reasoning": "The user just says \"", "reasoning_details": [...]}
  # NO reasoning_content key at all

SDK streaming delta model_dump():
  {'reasoning': 'Okay', 'reasoning_details': [...]}
  # NO reasoning_content key at all
```

**Impact:** The current `_extract_reasoning` only reads `reasoning_content` (via `getattr`). For OpenRouter and Ollama-OpenAI-compat, the SDK never populates `reasoning_content` — it uses `reasoning`. Real reasoning content from these providers would be **silently dropped**.

**Spec conflict:** REQ-PROV-18 AC4 pins OpenAI/LM Studio/OpenRouter to `reasoning_content` only, with "NO `reasoning`-first fallback." The real OpenRouter SDK populates `reasoning`, not `reasoning_content`. This is a direct spec-vs-reality conflict.

**OpenRouter also returns** `reasoning_details` (array of `{type, text, format, index}`) — not currently captured.

## 2. OpenRouter — Real Response Shapes

### Model listing (`GET /v1/models`)
```json
{
  "data": [{
    "id": "qwen/qwen3.8-max",
    "name": "Qwen: Qwen3.8 Max",
    "context_length": 1000000,
    "architecture": {
      "modality": "text+image+video->text",
      "input_modalities": ["text", "image", "video"],
      "output_modalities": ["text"]
    },
    "pricing": {
      "prompt": "0.000002",
      "completion": "0.000006",
      "input_cache_read": "0.00000025"
    },
    "top_provider": {
      "context_length": 1000000,
      "max_completion_tokens": 131072
    },
    "supported_parameters": ["tools", "reasoning", "reasoning_effort", ...]
  }]
}
```
- `pricing.prompt`/`completion` are **per-1K strings** (e.g. `"0.000002"` = $0.000002/1K tokens). Current code does `float(p.get("prompt", 0)) * 1_000_000` → converts to per-million. ✅ OK.
- Extra pricing fields (`input_cache_read`, etc.) are dropped. ✅ OK.
- `architecture.modality` is a string like `"text+image+video->text"`. Current code checks `"image" in modality`. ✅ Works.
- `supported_parameters` includes `"tools"`. Current code checks `"tools" in supported`. ✅ OK.

### Static chat (`POST /v1/chat/completions`)
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The user said \"Say",
      "reasoning": "The user said \"Say",
      "reasoning_details": [{"type": "reasoning.text", "text": "...", "format": "unknown", "index": 0}]
    },
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 18, "completion_tokens": 5, "total_tokens": 23,
    "completion_tokens_details": {"reasoning_tokens": 5}
  }
}
```
- Reasoning is in `reasoning` (not `reasoning_content`). ⚠️ **MISMATCH**
- `reasoning_details` array present. Not captured.

### Streaming chat
- Delta has `reasoning` field (not `reasoning_content`). ⚠️ **MISMATCH**
- Final usage-only chunk has `choices` with delta + `usage`. ✅ Matches code expectation.

## 3. LM Studio — Real Response Shapes

### Native model listing (`GET /api/v1/models`)
```json
{
  "models": [{
    "type": "llm",
    "key": "qwen3.5-9b-claude-4.6-opus-reasoning-distilled",
    "display_name": "Qwen3.5 9B",
    "max_context_length": 262144,
    "capabilities": {
      "vision": true,
      "trained_for_tool_use": true,
      "reasoning": {"allowed_options": ["off", "on"], "default": "on"}
    }
  }]
}
```
- `capabilities.vision` is a **boolean** (not `"image-completion"` string). Current code checks `"image-completion" in caps`. ⚠️ **MISMATCH** — vision would not be detected.
- `capabilities.trained_for_tool_use` is a boolean. Current code reads `caps.get("trained_for_tool_use", False)`. ✅ OK.
- `capabilities.reasoning` is an object (not used for classification). Current code doesn't read it. Acceptable.
- `max_context_length` field. Current code reads `raw.get("max_context_length")`. ✅ OK.

### Static chat (`POST /v1/chat/completions`)
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "",
      "reasoning_content": "The user is asking me",
      "tool_calls": []
    },
    "finish_reason": "length"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
}
```
- Uses `reasoning_content`. ✅ Matches code.
- `tool_calls: []` in response. Not an issue.

## 4. Ollama — Real Response Shapes

### Native listing (`GET /api/tags`)
```json
{
  "models": [{
    "name": "qwen3.5:0.8b",
    "model": "qwen3.5:0.8b",
    "details": {
      "format": "gguf",
      "family": "qwen35",
      "parameter_size": "873.44M",
      "context_length": 262144,
      "embedding_length": 1024
    },
    "capabilities": ["vision", "completion", "tools", "thinking"]
  }]
}
```
- `details.context_length` (not `max_context_length`). Current code reads `raw.get("max_context_length")`. ⚠️ **MISMATCH** — context_length would be `None`.
- `capabilities` is an array of strings including `"thinking"`. Current code doesn't read this. Acceptable.

### Native chat (`POST /api/chat`, `think: true`)
```json
{
  "message": {"role": "assistant", "content": "", "thinking": "Thinking Process:\n\n1..."},
  "done": true,
  "done_reason": "stop",
  "prompt_eval_count": 14,
  "eval_count": 24
}
```
- `message.thinking` field. Current code reads `msg.get("thinking")`. ✅ OK.
- `prompt_eval_count` / `eval_count`. Current code reads these. ✅ OK.
- When `think: false`, no `thinking` field is returned. ✅ OK.

### Native streaming chat
- Each chunk: `{"message": {"content": "", "thinking": "Okay"}, "done": false}`
- `thinking` accumulates across chunks. Current code reads `msg.get("thinking")`. ✅ OK.

### OpenAI-compat chat (`POST /v1/chat/completions`)
```json
{
  "choices": [{
    "message": {"role": "assistant", "content": "", "reasoning": "Thinking Process:\n\n1"},
    "finish_reason": "length"
  }]
}
```
- Uses `reasoning` (not `reasoning_content`). ⚠️ **MISMATCH**

### Native embed (`POST /api/embed`)
- This server returned: `{"error": "This server does not support embeddings. Start it with `--embeddings`"}`. Cannot test embed shape here.

## 5. Summary of Discrepancies

| # | Severity | Provider | Issue |
|---|---|---|---|
| 1 | **HIGH** | OpenRouter | Reasoning in `reasoning` not `reasoning_content` — would be dropped |
| 2 | **HIGH** | Ollama OpenAI-compat | Reasoning in `reasoning` not `reasoning_content` — would be dropped |
| 3 | MEDIUM | LM Studio | `capabilities.vision` is boolean, code checks for `"image-completion"` string |
| 4 | MEDIUM | Ollama native | `details.context_length` vs code's `max_context_length` |
| 5 | LOW | OpenRouter | `reasoning_details` array not captured |

## 6. Recommendations

1. **Reasoning (OpenRouter/Ollama):** The spec pins these to `reasoning_content`, but the real APIs use `reasoning`. Decide: (a) follow the spec strictly (reasoning dropped for these providers), or (b) add `reasoning` as a fallback for OpenRouter/Ollama. The OpenAI SDK may map `reasoning` → `reasoning_content` internally — this needs verification against the SDK's parsed objects, not raw JSON.

2. **LM Studio vision:** Change detection from `"image-completion" in caps` to `caps.get("vision", False)`.

3. **Ollama context_length:** Read `details.context_length` instead of (or in addition to) `max_context_length`.

4. **OpenRouter reasoning_details:** Consider capturing if useful for display.
