# Provider Module Rework — Testable Requirements

Status: draft
Date: 2026-07-31

## Purpose

This document defines the behavioral requirements for the provider module rework
(Ollama, LM Studio, and OpenRouter moving to their official APIs; `openai_compat.py`
folded into the OpenAI provider; `InferenceProvider` and data models subject to
change). It exists to guarantee that the defects identified in the provider code
review (2026-07-31) do not persist in the new implementation.

Architecture requirements (official clients, module layout, base-class contract)
are intentionally out of scope for now and will be added to this spec incrementally.

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

---

## Open questions

- None. Requirements above are considered complete for the 2026-07-31 review
  issues; architecture requirements for the provider rework are deferred by design
  and will be appended to this document.
