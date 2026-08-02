# Provider Module Rework — Testable Requirements

Status: draft
Date: 2026-07-31

## Purpose

This document defines the behavioral requirements for the provider module rework
(Ollama, LM Studio, and OpenRouter moving to their official APIs; `openai_compat.py`
folded into the OpenAI provider; `InferenceProvider` and data models subject to
change). It exists to guarantee that the defects identified in the provider code
review (2026-07-31) do not persist in the new implementation.

Architecture requirements — official clients, module layout, base-class contract,
and the enriched data models — were added on 2026-08-01 (sections 12-16).

## How to read this spec

- **MUST** — mandatory behavior. A requirement is satisfied only when every
  acceptance criterion (AC) holds.
- Requirements are written **behaviorally**, not against specific method names or
  line numbers. The rework may rename methods, restructure modules, or change the
  base class; the requirements persist as written.
- **Test target** — a suggested location for the tests that verify the ACs. Not
  binding; tests may live anywhere the implementation deems appropriate.
- Method names used in requirements (e.g. `get_model`, `stream_chat`) refer to the
  current API and should be read as "or its equivalent in the new architecture".
- Requirement IDs are prefixed by the module directory they concern:
  `REQ-PROV-*` for `providers/`, `REQ-MODEL-*` for `models/`, `REQ-SERV-*` for
  `services/`, `REQ-CLI-*` for `cli/`, `REQ-API-*` for `api/`.

## Traceability

| Review issue (2026-07-31) | Requirement(s) |
|---|---|
| 1. Streaming crash on final usage chunk | REQ-PROV-01, REQ-PROV-02 |
| 2. `json.loads` crash on tool-call arguments | REQ-PROV-03 |
| 3. Model listing bypasses timeout/retries | REQ-PROV-04 |
| 4. Registry swallows exceptions, inconsistent keys, no logging | REQ-PROV-05, REQ-PROV-06 |
| 5. `get_model` fabricates models; heuristic type fragility | REQ-PROV-08, REQ-PROV-09 |
| 6. No provider-id guard before invocation | REQ-PROV-10 |
| 7. Assistant reasoning content dropped on replay | REQ-PROV-11 |
| 8. Parser contracts defined by self-authored fixtures | REQ-PROV-12 |
| 9. Explicit `None` inference params sent | REQ-PROV-14 |
| 10. `stream_options` sent unconditionally | REQ-PROV-15 |
| 11. Reasoning field precedence undocumented | REQ-PROV-18 |
| 12. OpenAI metadata table drift / untested | REQ-PROV-17 |
| 13. `base_url` string surgery | REQ-PROV-13 |
| 14. Registry `get()` misleading error message | REQ-PROV-07 |
| 15. Empty `api_key` sent as auth header | REQ-PROV-16 |
| 16. Test gaps (None-key config, non-2xx paths) | REQ-PROV-19, REQ-PROV-20 |

---

## 1. Streaming integrity

### REQ-PROV-01 — Streaming MUST complete without error after the final content chunk

A provider's streaming implementation MUST NOT raise when the API sends a final
chunk with no choices (e.g. a usage-only chunk produced by
`stream_options: {"include_usage": true}`-style options).

- AC1: A chunk with an empty `choices` list and usage populated yields a
  `StreamDelta` and does not raise.
- AC2: A stream of content chunks followed by a usage-only final chunk completes
  cleanly; the `finish_reason` from the last contentful chunk is preserved in the
  emitted deltas.
- AC3: An exception raised mid-stream still surfaces as `ModelInvocationError`
  (error-wrapping behavior does not regress).

Test target: `tests/providers/test_openai.py` — streaming tests.

### REQ-PROV-02 — Streaming token usage MUST be delivered to the caller

When the provider API includes token usage in a stream (e.g. a final usage chunk),
the provider MUST emit a `StreamDelta` carrying that usage as `TokenUsage`.
The usage MUST NOT be dropped.

- AC1: A usage-only final chunk produces a `StreamDelta` with `usage` populated
  (prompt/completion/total tokens).
- AC2: When no usage chunk is sent, every emitted delta has `usage=None`.

Test target: `tests/providers/test_openai.py` — streaming usage tests.

## 2. Tool-call parsing

### REQ-PROV-03 — Tool-call argument parsing MUST NOT crash invocation on malformed arguments

When a provider returns tool-call arguments that are empty, whitespace-only, or
invalid JSON, the provider MUST either normalize them (e.g. to an empty arguments
dict) or raise a typed error (`ModelInvocationError`). A raw `JSONDecodeError`
MUST NOT escape the provider.

- AC1: Empty-string arguments produce a `ToolCall` with an empty arguments dict and
  no exception.
- AC2: Invalid JSON arguments raise `ModelInvocationError`.
- AC3: Valid JSON arguments are parsed into a dict as before.

Test target: `tests/providers/test_openai.py` — static-chat tool-call tests.

## 3. Configuration adherence

### REQ-PROV-04 — All provider API calls MUST honor configured timeout and retries

Every network call a provider makes — model listing, model retrieval, chat,
streaming, and any auxiliary HTTP requests — MUST use the configured
`provider_timeout` and `provider_max_retries` (or the official client's equivalent
settings).

- AC1: The official API client is constructed with the configured timeout and
  max retries.
- AC2: Any auxiliary HTTP requests retained by the new architecture pass the
  configured timeout and retry per configuration.
- AC3: Defaults remain `provider_timeout=120`, `provider_max_retries=2`.

Test target: provider constructor tests; any native-API call tests.

## 4. Registry robustness

### REQ-PROV-05 — Registry MUST log provider initialization failures

When a provider class raises during initialization, the failure MUST be recorded
in the registry's failure map AND logged (error level, including the exception).

- AC1: A provider whose constructor raises produces a failure record and an error
  log containing the exception detail.
- AC2: Successful initialization produces no error log for that provider.

Test target: `tests/providers/test_registry.py`.

### REQ-PROV-06 — Registry failure records MUST be keyed consistently with availability

The failure map MUST use the same keys as the availability map (provider id)
wherever the provider id can be determined. Class name is a documented fallback
only for constructors that fail before an id exists.

- AC1: A failure whose provider id is determinable is keyed by that id.
- AC2: A constructor failure that never produces an instance is keyed by class
  name, and this fallback is documented.

Test target: `tests/providers/test_registry.py`.

### REQ-PROV-07 — Registry `get()` MUST include the initialization failure reason

When a provider is registered but failed initialization, `get()` MUST raise
`ProviderNotAvailableError` whose message includes the recorded failure reason.
For unregistered ids, the message MUST indicate the provider is unknown.

- AC1: `get()` on a provider that failed init raises with the stored failure text
  in the message.
- AC2: `get()` on an unregistered id raises with a message indicating an unknown
  provider.

Test target: `tests/providers/test_registry.py`.

## 5. Model lookup and invocation guards

### REQ-PROV-08 — `get_model` MUST NOT fabricate models

`get_model` for an id the provider has no data for MUST raise a typed error
(`ModelsFetchError` or a documented subclass). It MUST NOT return a guessed
`ModelData` built from heuristics.

- AC1: A known id returns the full `ModelData` from provider data.
- AC2: An unknown id raises a typed error and returns no model.

Test target: per-provider model tests (all four providers).

### REQ-PROV-09 — Model type classification MUST prefer provider-native type information

When a provider's API reports a model's type natively (e.g. "llm"/"embedding"),
that value MUST take precedence. Substring heuristics MUST apply only when no
native type exists, and every heuristic rule MUST be tested with representative
cases.

- AC1: A native `llm` type overrides a heuristic keyword match (e.g. an id
  containing "embed") → classified as `ModelType.LLM`.
- AC2: Without a native type, an id containing a heuristic keyword (e.g. "embed",
  "audio", "image") → `ModelType.OTHER`.
- AC3: Without a native type and without keyword matches → `ModelType.LLM`.

Test target: per-provider classification tests; OpenAI `_format_model` tests.

### REQ-PROV-10 — Invocation MUST reject models from other providers before any network call

Chat invocation MUST raise a typed error when the model's `provider_id` does not
match the provider's id, and the API client MUST NOT be called.

- AC1: A model whose `provider_id` differs from the provider raises a typed error
  and the client call is never awaited.
- AC2: A model whose `provider_id` matches proceeds normally.
- AC3: The existing non-LLM type rejection (typed error, no network call) is
  preserved.

Test target: `tests/providers/test_base.py` (contract) or per-provider tests.

## 6. Message round-trip

### REQ-PROV-11 — Assistant reasoning content MUST round-trip

When an `AssistantMessage` containing reasoning content is formatted for the API,
the request MUST include the reasoning field expected by that provider's official
API (e.g. `reasoning_content` for OpenAI; the documented field for other
providers).

- AC1: OpenAI-formatted assistant messages include `reasoning_content` when
  present.
- AC2: Providers whose official API uses a different field name include their
  documented field.
- AC3: Assistant messages without reasoning content are unchanged.

Test target: message-formatting tests per provider.

## 7. API contract fidelity

### REQ-PROV-12 — Provider response parsing MUST be validated against real API responses

Each provider's response parsing MUST have at least one test fixture derived from
a recorded or live official-API response — not solely hand-authored shapes written
to match the code. Malformed or empty response bodies MUST raise typed errors,
not `AttributeError`/`KeyError`.

- AC1: Every provider has at least one recorded-response fixture test for model
  listing.
- AC2: An empty or malformed response body raises a typed error
  (`ModelsFetchError`).

Test target: per-provider model-listing tests.

### REQ-PROV-13 — Endpoint derivation MUST use URL parsing, not string surgery

Endpoints derived from the configured `base_url` MUST be computed by URL parsing
such that the documented `base_url` variants (with/without trailing slash,
with/without a `/v1` suffix, custom paths) all yield correct endpoints.

- AC1: Parametrized tests over the documented `base_url` variants produce the
  correct endpoint in every case.

Test target: per-provider endpoint-derivation tests (LM Studio, OpenRouter).

## 8. Request payload hygiene

### REQ-PROV-14 — Unset inference params MUST be omitted, not sent as null

When `InferenceParams` fields are unset (None), the API request MUST NOT contain
those keys.

- AC1: With all params unset, the request body contains no `temperature`,
  `max_tokens`, or `top_p` keys.
- AC2: Set values are sent as before.

Test target: request-building tests (OpenAI provider).

### REQ-PROV-15 — Streaming MUST work on every provider without unsupported parameters

Usage-requesting options (e.g. `stream_options` / `include_usage`) MUST be sent
only where the endpoint supports them. Providers whose endpoints do not support
them MUST stream normally and complete without error.

- AC1: A supported endpoint receives the usage option.
- AC2: An unsupported endpoint receives no usage option; the stream completes and
  every delta has `usage=None`.

Test target: per-provider streaming tests.

### REQ-PROV-16 — Providers without API keys MUST NOT send empty Authorization headers

`api_key` MUST be handled as `str | None`. A `None` or empty key MUST produce
requests without an `Authorization` header; a set key MUST produce the documented
bearer header.

- AC1: With a `None`/empty key, requests carry no `Authorization` header.
- AC2: With a set key, requests carry the bearer header.

Test target: constructor and request tests per provider.

## 9. Model metadata table

### REQ-PROV-17 — Metadata table entries MUST be test-covered; unknown models MUST degrade gracefully

Every entry in the OpenAI metadata table MUST be exercised by a test asserting its
field values. Ids not in the table MUST yield default metadata (None/False)
without error.

- AC1: A parametrized test covers every table entry and asserts its
  `context_length`, `max_output`, `supports_tools`, and `supports_vision`.
- AC2: An id not in the table yields default metadata and no error.

Test target: `tests/providers/test_openai.py` — metadata tests.

## 10. Reasoning extraction

### REQ-PROV-18 — Reasoning extraction MUST match each provider's official field with documented precedence

Reasoning extraction MUST use the field names of each provider's official API, in
both streaming and static responses. When multiple candidate fields exist, the
precedence MUST be documented and tested.

- AC1: The OpenAI-style `reasoning_content` field is extracted (streaming delta
  and static message).
- AC2: A provider-documented alternative field (e.g. `reasoning`) is extracted
  where applicable.
- AC3: Empty or whitespace-only reasoning yields `None`.
- AC4: Precedence when multiple fields are present is documented and tested.

Test target: per-provider reasoning tests.

## 11. Constructor and error paths

### REQ-PROV-19 — Constructor tests MUST cover both missing-key and explicit-None key configs

For every provider that requires an API key, both the absent-from-config case and
the explicit-`None` case MUST raise the documented error and be tested
independently.

- AC1: Absent key (not in config) raises the documented error; test exists.
- AC2: Explicit `None` key raises the documented error; test exists.

Test target: `tests/providers/test_init.py`.

### REQ-PROV-20 — Non-2xx responses MUST raise typed errors

Non-2xx responses (auth failures, 429, 5xx) from any provider API call MUST
surface as the documented typed errors — `ModelsFetchError` for model listing and
retrieval, `ModelInvocationError` for chat — with tests for each call type.

- AC1: A non-2xx response on model listing raises `ModelsFetchError`.
- AC2: A non-2xx response on a chat call raises `ModelInvocationError`.

Test target: per-provider error-path tests.

## 12. Architecture — Data models (`yapa/models/`)

### REQ-MODEL-01 — ModelType MUST expose three distinct values

`ModelType` MUST contain exactly `llm`, `embedding`, and `other` values.

- AC1: `ModelType.LLM`, `ModelType.EMBED`, and `ModelType.OTHER` exist with
  the values `"llm"`, `"embedding"`, `"other"`.
- AC2: The API `model_type` query parameter and the CLI `--type` option accept
  the `embedding` value.

Test target: `tests/models/` — model type tests; `tests/api/`; `tests/cli/`.

### REQ-MODEL-02 — Model data MUST be a discriminated hierarchy

`ModelData` MUST be the base type carrying `id`, `provider_id`, `type`, `name`,
and `description`. `LanguageModel` and `EmbedModel` MUST be subtypes
discriminated by the `type` field. Parsing a model record MUST yield the
correct subtype without explicit consumer dispatch.

- AC1: A record with `type="llm"` parses as `LanguageModel`.
- AC2: A record with `type="embedding"` parses as `EmbedModel`.
- AC3: A record with `type="other"` parses as bare `ModelData`.
- AC4: `name` and `description` default to `None` for every subtype when
  absent.

Test target: `tests/models/` — model data tests.

### REQ-MODEL-03 — Pricing MUST be a structured model

Pricing MUST be a Pydantic model (`ModelPricing`) with optional `input`,
`output`, and `request` fields (USD per million tokens), each `None` when
unknown.

- AC1: Pricing serializes as a structured object, not a bare dict.
- AC2: Unknown pricing fields default to `None` without error.
- AC3: Unknown pricing on a model is represented as a `None` value, not an
  empty object.

Test target: `tests/models/` — pricing tests.

### REQ-MODEL-04 — LanguageModel MUST carry LLM-specific capability fields

`LanguageModel` MUST expose `context_length`, `max_output`, `supports_tools`,
`supports_vision`, `supports_reasoning`, `reasoning_levels`,
`supports_streaming`, and `pricing`.

- AC1: All fields default safely (`None`/`False`/empty list) when absent.
- AC2: `reasoning_levels` is a list of strings (e.g. `["low","medium","high"]`,
  `["on","off"]`, or empty).

Test target: `tests/models/` — language model tests.

### REQ-MODEL-05 — EmbedModel MUST carry embedding-specific fields

`EmbedModel` MUST expose `embedding_dimensions`, `normalized`, and `pricing`.

- AC1: `embedding_dimensions` defaults to `None`; `normalized` defaults to
  `False`.
- AC2: An embedding model reports its native dimensions when the provider
  API exposes them.

Test target: `tests/models/` — embed model tests.

### REQ-MODEL-06 — InferenceParams MUST be a curated typed set

`InferenceParams` MUST contain the typed fields `temperature`, `max_tokens`,
`top_p`, `presence_penalty`, `frequency_penalty`, `stop`, `seed`, `top_k`,
`min_p`, `repeat_penalty`, and `reasoning_effort`. Every field MUST default to
`None` and be omitted from API requests when unset (see REQ-PROV-14).

- AC1: A params object with all fields unset serializes without those keys.
- AC2: A params object with a subset set serializes only the set keys.
- AC3: `stop` accepts a single string or a list of strings.

Test target: `tests/models/` — inference params tests.

### REQ-MODEL-07 — Reasoning effort MUST be a unified enum

`ReasoningEffort` MUST contain `OFF`, `LOW`, `MEDIUM`, and `HIGH`. The mapping
to each provider's request parameter MUST be:

| Unified | OpenAI | OpenRouter | LM Studio | Ollama |
|---|---|---|---|---|
| `OFF` | omit | omit | `reasoning: "off"` | `think: false` |
| `LOW` | `reasoning: {"effort": "low"}` | same | `reasoning: "low"` | `think: true` |
| `MEDIUM` | `reasoning: {"effort": "medium"}` | same | `reasoning: "medium"` | `think: true` |
| `HIGH` | `reasoning: {"effort": "high"}` | same | `reasoning: "high"` | `think: true` |
| `None` | omit | omit | omit | omit |

- AC1: Each provider maps every `ReasoningEffort` value per the table.
- AC2: `None` produces no reasoning-related request parameter.

Test target: per-provider request-building tests.

### REQ-MODEL-08 — Message content MUST support text and image parts

User message content MUST be `str | list[ContentPart]` where `ContentPart` is
a discriminated union of `TextPart` and `ImagePart`. `ImagePart` MUST carry a
`url` (http(s) or data URL) and an optional `detail` hint.

- AC1: A plain string message parses unchanged.
- AC2: A mixed text/image part list parses and round-trips through storage.
- AC3: An unknown part type fails validation.

Test target: `tests/models/` — message tests.

### REQ-MODEL-09 — Embedding results MUST be a structured result

`EmbeddingResult` MUST contain `vectors` (list of float lists aligned with the
inputs), `model_id`, and optional `usage` (`TokenUsage`).

- AC1: There is one vector per input, in input order.
- AC2: `usage` is `None` when the provider does not report it.

Test target: `tests/models/` — embedding result tests.

### REQ-MODEL-10 — Session model MUST be typed as LanguageModel

`Session.model` MUST be `LanguageModel | None`. A session record whose model
is not an LLM MUST fail validation.

- AC1: A session with an LLM model loads normally.
- AC2: A session record with `type="embedding"` in `model` fails validation.

Test target: `tests/models/` — session tests.

### REQ-MODEL-11 — Streaming MUST emit a discriminated event union

The provider streaming boundary MUST be `StreamEvent`, a discriminated union
of `ContentDelta`, `ReasoningDelta`, `ToolCallDeltaEvent`, and `StreamEndEvent`
keyed by a `type` discriminator.

- AC1: Content deltas carry `content` only.
- AC2: Reasoning deltas carry `content` only.
- AC3: Tool-call deltas carry `index`, `id`, `name`, and `arguments`
  (raw JSON fragments).
- AC4: The union rejects unknown event types at construction.

Test target: `tests/models/` — stream event tests.

## 13. Architecture — Streaming contract (provider boundary)

### REQ-PROV-21 — A stream MUST end with exactly one StreamEndEvent

Every provider stream MUST emit exactly one `StreamEndEvent` as its final
event. `finish_reason`, `usage`, and `model_id` MUST appear only on
`StreamEndEvent`, never on content, reasoning, or tool-call deltas.

- AC1: A stream of content chunks followed by a usage-only final chunk ends
  with one `StreamEndEvent` carrying the usage and does not raise.
- AC2: A stream without a usage chunk ends with one `StreamEndEvent` with
  `usage=None` and the preserved `finish_reason`.
- AC3: A mid-stream provider exception surfaces as `ModelInvocationError`
  and no `StreamEndEvent` is emitted (error-wrapping does not regress).

Test target: per-provider streaming tests; `tests/providers/test_base.py`.

### REQ-PROV-22 — Stream events MUST NOT carry errors

Errors during streaming MUST be raised as typed exceptions
(`ModelInvocationError`); the `StreamEvent` union MUST contain no error event.

- AC1: A provider API failure during streaming raises `ModelInvocationError`.
- AC2: The union has exactly the four members of REQ-MODEL-11, none of which
  is an error.

Test target: per-provider streaming error tests.

## 14. Architecture — Provider base class and module layout

### REQ-PROV-23 — InferenceProvider MUST expose the full contract

The base class MUST expose `list_models`, `get_model`, `stream_chat`,
`static_chat`, and `embed`. Public methods MUST wrap private `_impl` methods
with logging and error conversion.

- AC1: Each public method logs at info level and wraps unexpected exceptions
  into the documented typed error.
- AC2: `embed` wraps failures as `ModelInvocationError` and never leaks
  client-level exceptions.
- AC3: `get_model`/`list_models` return the model subtypes of section 12 and
  never fabricate data for unknown ids (REQ-PROV-08 does not regress).

Test target: `tests/providers/test_base.py`.

### REQ-PROV-24 — Invocation MUST guard model subtypes before network calls

`stream_chat`/`static_chat` MUST reject non-`LanguageModel` arguments and
`embed` MUST reject non-`EmbedModel` arguments with a typed error before any
client call is made.

- AC1: `stream_chat` with an `EmbedModel` raises a typed error; the client is
  never called.
- AC2: `embed` with a `LanguageModel` raises a typed error; the client is
  never called.
- AC3: Matching subtypes proceed normally.

Test target: `tests/providers/test_base.py`; per-provider tests.

### REQ-PROV-25 — Client strategy MUST follow the official SDKs

OpenAI, OpenRouter, and LM Studio MUST use the official OpenAI SDK
(`AsyncOpenAI`) for chat, streaming, embeddings, and model listing, with the
configured `provider_timeout` and `provider_max_retries`. Ollama MUST use the
official Ollama SDK for chat, streaming, embeddings, and model list/show.
httpx MUST be used only for endpoints without SDK support (LM Studio native
model listing), and those calls MUST pass the configured timeout.

- AC1: `AsyncOpenAI` is constructed with the configured timeout and max
  retries.
- AC2: The Ollama client is constructed with the configured timeout.
- AC3: LM Studio native listing passes the configured timeout.
- AC4: No provider uses a client or endpoint other than documented above.

Test target: per-provider constructor and call tests.

### REQ-PROV-26 — Ollama retries MUST honor provider_max_retries

Because the official Ollama SDK has no built-in retry configuration, Ollama
provider calls MUST be wrapped in a retry layer honoring
`provider_max_retries`, retrying only retryable failures.

- AC1: A transient failure is retried up to `provider_max_retries` times.
- AC2: Non-retryable failures are not retried.
- AC3: Total attempts never exceed `provider_max_retries + 1`.

Test target: `tests/providers/test_ollama.py` — retry tests.

### REQ-PROV-27 — Multimodal messages MUST map to each provider's schema

User messages containing image parts MUST be formatted per provider: the
openai-SDK family uses the content-array form; Ollama uses its documented
image field mapping.

- AC1: A message with an image part produces the provider's documented image
  representation.
- AC2: Plain string messages are unchanged.

Test target: per-provider message-formatting tests.

### REQ-PROV-28 — Embedding calls MUST normalize usage and errors

`embed` MUST return `EmbeddingResult`. Token usage MUST be normalized to
`TokenUsage` when the provider reports it (missing categories default to 0)
and be `None` otherwise. Non-2xx and malformed responses MUST raise
`ModelInvocationError`.

- AC1: OpenAI-family embeddings map usage to `TokenUsage`.
- AC2: Ollama embeddings with token counts map to `TokenUsage`; without
  counts, `usage=None`.
- AC3: A failing embed call raises `ModelInvocationError`.

Test target: per-provider embedding tests.

### REQ-PROV-29 — Model type classification MUST map embed keywords to EMBED

Per REQ-PROV-09, without a native type, an id containing `embed` MUST
classify as `ModelType.EMBED`; `audio`/`image` keywords MUST classify as
`ModelType.OTHER`.

- AC1: `text-embedding-3-large` (no native type) classifies as EMBED.
- AC2: A native `llm` type still overrides the `embed` keyword match.
- AC3: A native `embedding` type classifies as EMBED.

Test target: per-provider classification tests; OpenAI `_format_model` tests.

## 15. Architecture — Consumer integration

### REQ-SERV-01 — ChatService MUST consume the StreamEvent union

ChatService MUST dispatch on `StreamEvent` types rather than inspecting
optional fields, and MUST assemble `AssistantMessage` from accumulated
content, tool calls, finish reason, and usage.

- AC1: Content, reasoning, and tool-call deltas accumulate as before;
  agentic-loop behavior does not regress.
- AC2: A usage-only final chunk yields no error and the usage lands on the
  final `AgentDoneEvent`.
- AC3: The assembled `AssistantMessage` carries `usage` and `finish_reason`.

Test target: `tests/services/` — chat tests.

### REQ-SERV-02 — ModelService MUST preserve subtype fidelity

`ModelService.list_models` and `get_model` MUST return the model subtypes
without flattening. `get_provider_by_model` MUST dispatch on
`model.provider_id` unchanged.

- AC1: Listing returns `LanguageModel`/`EmbedModel` instances as produced by
  providers.
- AC2: Per-provider error isolation (one failing provider does not fail the
  listing) is preserved.

Test target: `tests/services/` — model service tests.

### REQ-CLI-01 — CLI model listing MUST accept the embedding type

The `models` command's `--type` option MUST accept `llm`, `embedding`, and
`other`, with help text listing all three.

- AC1: `--type embedding` lists only embedding models.
- AC2: An invalid value is rejected with the documented error.

Test target: `tests/cli/`.

### REQ-CLI-02 — CLI MUST display pricing from ModelPricing

Model listing output MUST render pricing from the structured `ModelPricing`
fields where present, and show a placeholder when absent.

- AC1: A model with pricing renders input/output/request values.
- AC2: A model without pricing renders a placeholder without error.

Test target: `tests/cli/`.

### REQ-API-01 — API model routes MUST accept and serialize subtypes

`GET /models` and `GET /models/{full_id}` MUST accept the `embedding` model
type and serialize subtype-specific fields without error.

- AC1: `?model_type=embedding` returns only embedding models.
- AC2: The JSON response includes subtype fields (e.g. `embedding_dimensions`,
  `supports_tools`) with absent fields as `null`.

Test target: `tests/api/` — model route tests.

## 16. Amendments to existing requirements

| Requirement | Change |
|---|---|
| REQ-PROV-01 | Superseded by REQ-PROV-21: the usage-only final chunk is expressed as the final `StreamEndEvent`; the stream must still complete cleanly. |
| REQ-PROV-02 | Superseded by REQ-PROV-21: token usage is delivered on the `StreamEndEvent` as `TokenUsage`; absent usage → `usage=None`. |
| REQ-PROV-04 | Extended by REQ-PROV-25/26: official SDK clients carry timeout/retries; the Ollama SDK retry gap is closed by a retry layer; httpx calls pass the configured timeout. |
| REQ-PROV-09 | AC2 amended by REQ-PROV-29: `embed` keyword → `ModelType.EMBED`; `audio`/`image` → `ModelType.OTHER`. |
| REQ-PROV-10 | Extended by REQ-PROV-24: invocation guards use model subtypes. |
| REQ-PROV-20 | Extended by REQ-PROV-28: `embed` failures raise `ModelInvocationError`. |

---

## Open questions

- None. Sections 12-16 record the architecture requirements agreed on
  2026-08-01; behavioral requirements 1-11 remain complete for the 2026-07-31
  review issues.
