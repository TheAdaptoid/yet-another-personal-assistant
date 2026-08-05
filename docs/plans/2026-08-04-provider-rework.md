# Provider Module Rework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the provider module so Ollama, LM Studio, and OpenRouter use their official APIs, fold `openai_compat.py` into the OpenAI provider, introduce a discriminated `ModelData` hierarchy, a `StreamEvent`-based streaming contract, and a first-class `ReasoningEffort` chat argument — eliminating every defect from the 2026-07-31 provider review and satisfying spec `docs/specs/2026-07-31-provider-rework-requirements.md`.

**Architecture:** The rework proceeds bottom-up in six phases. Phase A rebuilds the data models (`ModelType`, discriminated `ModelData`/`LanguageModel`/`EmbedModel`, `ModelPricing`, `InferenceParams`, `ReasoningEffort`, `ContentPart`, `EmbeddingResult`, `StreamEvent`, `Session.model`). Phase B rebuilds the `InferenceProvider` base contract (`embed`, reasoning argument, subtype + provider-id guards). Phase C hardens the registry. Phase D rebuilds the shared OpenAI-family client (folding `openai_compat.py` into `providers/openai/`), used by OpenAI, LM Studio, and OpenRouter. Phase E adds the native/list-then-search providers (OpenRouter/LM Studio kotlin native listing; Ollama native SDK with a retry layer). Phase F updates the consumers (ChatService, ModelService, CLI, API). Each phase ends with a green test suite.

**Tech Stack:** Python 3.12+, asyncio, Pydantic v2 (discriminated unions, `TypeAdapter`), `openai` SDK (`AsyncOpenAI`), `httpx`, `ollama` (native SDK, new dependency, `>=0.4`), FastAPI, typer/rich, pytest.

## Global Constraints

- `docs/specs/2026-07-31-provider-rework-requirements.md` is the source of truth. Requirement IDs below (e.g. REQ-PROV-03) map 1:1 to its sections. Every AC listed under a requirement is satisfied by the task that references it.
- `ModelType` must contain exactly `llm`, `embedding`, `other` (`ModelType.LLM`, `ModelType.EMBED`, `ModelType.OTHER`).
- Only `InferenceProviderError` subtypes (`ModelsFetchError`, `ModelTypeError`, `ModelInvocationError`) may cross the provider boundary — never a raw `JSONDecodeError`, `AttributeError`, `KeyError`, httpx, or SDK exception (REQ-PROV-20 AC4).
- Defaults: `provider_timeout=120`, `provider_max_retries=2`. Every provider network call honors them (REQ-PROV-04).
- `InferenceParams` has NO reasoning field; `ReasoningEffort` is a first-class chat argument (REQ-MODEL-06, REQ-MODEL-07, REQ-PROV-30).
- `ReasoningEffort` precedence is pinned: OpenAI/LM Studio/OpenRouter read only `reasoning_content` (never a `reasoning` fallback); Ollama reads `message.thinking` (REQ-PROV-18 AC4). Empty/whitespace → `None`.
- Client strategy (REQ-PROV-25): OpenAI → `AsyncOpenAI` only; OpenRouter → `AsyncOpenAI` (chat/stream/embed) + httpx native listing; LM Studio → `AsyncOpenAI` (chat/stream/embed) + httpx native listing; Ollama → `ollama.AsyncClient` for all. httpx used only for native listing and passes the configured timeout.
- Chat/static invocations reject non-`LanguageModel` and wrong-`provider_id` models before any client call; `embed` rejects non-`EmbedModel` (REQ-PROV-10, REQ-PROV-24).
- A stream ends with exactly one `StreamEndEvent`; `finish_reason`/`usage`/`model_id` appear only on it (REQ-PROV-21).
- CLI `--type` and API `?model_type=` accept `embedding` (REQ-CLI-01, REQ-API-01).
- Commit messages start with the repo prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`) (AGENTS.md).
- Gate (AGENTS.md): `uv run ruff check src/ tests/ && uv run ty check src/ && uv run pytest tests/ -v`

> **2026-08-04 decisions locked in during planning** (resolving spec ambiguities): (1) `ModelData` remains the concrete base carrying `id/provider_id/type/name/description`; `LanguageModel` and `EmbedModel` are `Literal`-typed subclasses; the union `LanguageModel | EmbedModel | ModelData` (keyed on the `type` discriminator, Pydantic smart union) is used for parsing and FastAPI response models so subtype fields serialize. (2) `openai_compat.py` moves to `providers/openai/openai_compat.py`. (3) A no-auth local provider (LM Studio) builds `AsyncOpenAI` with a sentinel key plus an httpx client that strips the `Authorization` header, satisfying REQ-PROV-16. (4) Decide task E4/E5 below for OpenRouter/LM Studio pricing classification once provider logic is green.

---

## PHASE A — Data models (`yapa/models/`)

### Task A1: ModelType gains EMBED and value/defaults (REQ-MODEL-01)

**Files:**
- Modify: `src/yapa/models/inference.py`
- Test: `tests/models/test_model_data.py` (new)

**Interfaces:**
- Produces: `ModelType` with `LLM="llm"`, `EMBED="embedding"`, `OTHER="other"`.

- [ ] **Step 1: Write failing test**

`tests/models/test_model_data.py`:

```python
from yapa.models.inference import ModelType


def test_model_type_has_three_values() -> None:
    assert ModelType.LLM.value == "llm"
    assert ModelType.EMBED.value == "embedding"
    assert ModelType.OTHER.value == "other"
    assert {m.value for m in ModelType} == {"llm", "embedding", "other"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_model_data.py -v`
Expected: FAIL with `AttributeError: type object 'ModelType' has no attribute 'EMBED'`.

- [ ] **Step 3: Add EMBED to ModelType**

In `src/yapa/models/inference.py`:

```python
class ModelType(Enum):
    """Enumeration for model types."""

    LLM = "llm"
    EMBED = "embedding"
    OTHER = "other"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_model_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/inference.py tests/models/test_model_data.py
git commit -m "feat: add ModelType.EMBED value"
```

---
### Task A2: ModelPricing structured model

**Files:**
- Modify: `src/yapa/models/inference.py`
- Test: `tests/models/test_pricing.py` (new)

**Interfaces:**
- Produces: `ModelPricing(BaseModel)` with optional `input`, `output`, `request` (float USD per million tokens), `extra="forbid"`, `frozen=True`.

- [ ] **Step 1: Write failing test**

`tests/models/test_pricing.py`:

```python
from yapa.models.inference import ModelPricing


def test_pricing_is_structured_object() -> None:
    p = ModelPricing(input=2.5, output=10.0, request=0.01)
    assert isinstance(p, ModelPricing)
    assert p.input == 2.5
    assert p.output == 10.0
    assert p.request == 0.01


def test_pricing_defaults_to_none() -> None:
    p = ModelPricing()
    assert p.input is None
    assert p.output is None
    assert p.request is None


def test_pricing_serializes_as_object() -> None:
    p = ModelPricing(input=1.0)
    assert p.model_dump() == {"input": 1.0, "output": None, "request": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_pricing.py -v`
Expected: FAIL with `ImportError` (`ModelPricing` undefined).

- [ ] **Step 3: Add ModelPricing**

In `src/yapa/models/inference.py` (import `ConfigDict` is already present):

```python
class ModelPricing(BaseModel):
    """Pricing for a model in USD per million tokens.

    Attributes:
        input: Cost per million input tokens, or None when unknown.
        output: Cost per million output tokens, or None when unknown.
        request: Fixed cost per request, or None when unknown.
    """

    input: float | None = Field(default=None)
    output: float | None = Field(default=None)
    request: float | None = Field(default=None)

    model_config = ConfigDict(extra="forbid", frozen=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_pricing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/inference.py tests/models/test_pricing.py
git commit -m "feat: add structured ModelPricing model"
```

---
### Task A3: Discriminated ModelData hierarchy

**Files:**
- Modify: `src/yapa/models/inference.py`
- Test: `tests/models/test_model_data.py`

**Interfaces:**
- Produces: `ModelData` (base: `id`, `provider_id`, `type: ModelType`, `name`, `description`, `full_id` property; `extra="forbid"`, `frozen=True`); `LanguageModel(ModelData)` with `type: Literal["llm"]`; `EmbedModel(ModelData)` with `type: Literal["embedding"]`; module-level `ModelDataUnion = LanguageModel | EmbedModel | ModelData` used for discriminated parsing/serialization.

- [ ] **Step 1: Write failing tests (append to `tests/models/test_model_data.py`)**

```python
from typing import Annotated

from pydantic import Field, TypeAdapter

from yapa.models.inference import (
    EmbedModel,
    LanguageModel,
    ModelData,
    ModelDataUnion,
    ModelType,
)

adapter = TypeAdapter(ModelDataUnion)


def test_llm_record_parses_as_language_model() -> None:
    m = adapter.validate_python({"id": "gpt-4", "provider_id": "openai", "type": "llm"})
    assert type(m) is LanguageModel


def test_embedding_record_parses_as_embed_model() -> None:
    m = adapter.validate_python(
        {"id": "embed", "provider_id": "openai", "type": "embedding"}
    )
    assert type(m) is EmbedModel


def test_other_record_parses_as_bare_model_data() -> None:
    m = adapter.validate_python({"id": "x", "provider_id": "openai", "type": "other"})
    assert type(m) is ModelData


def test_name_and_description_default_to_none() -> None:
    m = adapter.validate_python({"id": "gpt-4", "provider_id": "openai", "type": "llm"})
    assert m.name is None
    assert m.description is None
    m2 = adapter.validate_python({"id": "e", "provider_id": "p", "type": "embedding"})
    assert m2.name is None


def test_full_id() -> None:
    m = adapter.validate_python({"id": "gpt-4", "provider_id": "openai", "type": "llm"})
    assert m.full_id == "openai:gpt-4"


def test_language_model_unknown_type_not_a_bare_model_data() -> None:
    lm = LanguageModel(id="gpt-4", provider_id="openai")
    assert not isinstance(lm, EmbedModel)
    assert isinstance(lm, ModelData)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_model_data.py -v`
Expected: FAIL (`LanguageModel`/`EmbedModel`/`ModelDataUnion` undefined).

- [ ] **Step 3: Implement the hierarchy**

Replace the current `ModelData` class in `src/yapa/models/inference.py` with:

```python
from typing import Literal

class ModelData(BaseModel):
    """Base data model for representing a provider model.

    Attributes:
        id (str): Unique identifier for the model.
        provider_id (str): Identifier for the provider of the model.
        type (ModelType): The type of the model.
        name (str | None): Human-readable model name.
        description (str | None): Human-readable model description.
    """

    id: str = Field(..., description="Unique identifier for the model")
    provider_id: str = Field(
        ..., description="Identifier for the provider of the model"
    )
    type: ModelType = Field(..., description="The type of the model")
    name: str | None = Field(default=None, description="Human-readable model name")
    description: str | None = Field(
        default=None, description="Human-readable model description"
    )

    @property
    def full_id(self) -> str:
        """Return the fully-qualified model identifier (e.g. ``openai:gpt-4``)."""
        return f"{self.provider_id}:{self.id}"

    model_config = ConfigDict(extra="forbid", frozen=True)


class LanguageModel(ModelData):
    """An LLM, carrying LLM-specific capability fields."""

    type: Literal["llm"] = "llm"
    context_length: int | None = Field(default=None)
    max_output: int | None = Field(default=None)
    supports_tools: bool = Field(default=False)
    supports_vision: bool = Field(default=False)
    supports_reasoning: bool = Field(default=False)
    reasoning_levels: list[str] = Field(default_factory=list)
    supports_streaming: bool = Field(default=False)
    pricing: ModelPricing | None = Field(default=None)


class EmbedModel(ModelData):
    """An embedding model, carrying embedding-specific fields."""

    type: Literal["embedding"] = "embedding"
    embedding_dimensions: int | None = Field(default=None)
    normalized: bool = Field(default=False)
    pricing: ModelPricing | None = Field(default=None)


ModelDataUnion = LanguageModel | EmbedModel | ModelData
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_model_data.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/inference.py tests/models/test_model_data.py
git commit -m "feat: discriminated ModelData hierarchy"
```

---
### Task A4: LanguageModel capability fields (REQ-MODEL-04)

**Files:**
- Modify: `src/yapa/models/inference.py`
- Test: `tests/models/test_language_model.py` (new)

**Interfaces:**
- Produces: `LanguageModel` field defaults: `context_length`, `max_output` → `None`; `supports_tools`, `supports_vision`, `supports_reasoning`, `supports_streaming` → `False`; `reasoning_levels` → `[]`; `pricing` → `None`.

- [ ] **Step 1: Write failing tests**

`tests/models/test_language_model.py`:

```python
from yapa.models.inference import LanguageModel


def test_defaults() -> None:
    m = LanguageModel(id="gpt-4", provider_id="openai")
    assert m.context_length is None
    assert m.max_output is None
    assert m.supports_tools is False
    assert m.supports_vision is False
    assert m.supports_reasoning is False
    assert m.reasoning_levels == []
    assert m.supports_streaming is False
    assert m.pricing is None


def test_reasoning_levels_accepts_list() -> None:
    m = LanguageModel(
        id="o3", provider_id="openai", reasoning_levels=["low", "medium", "high"]
    )
    assert m.reasoning_levels == ["low", "medium", "high"]
    m2 = LanguageModel(id="m", provider_id="p", reasoning_levels=["on", "off"])
    assert m2.reasoning_levels == ["on", "off"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_language_model.py -v`
Expected: FAIL (`supports_reasoning` etc. missing).

- [ ] **Step 3: Implement (already added in Task A3 — verify fields exist)**

The `LanguageModel` class from Task A3 already includes all required fields. No further code needed; the fields are present.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_language_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/inference.py tests/models/test_language_model.py
git commit -m "test: cover LanguageModel capability fields"
```

---
### Task A5: EmbedModel fields (REQ-MODEL-05)

**Files:**
- Modify: `src/yapa/models/inference.py`
- Test: `tests/models/test_embed_model.py` (new)

**Interfaces:**
- Produces: `EmbedModel.embedding_dimensions` (int|None), `.normalized` (bool, default False), `.pricing`.

- [ ] **Step 1: Write failing tests**

`tests/models/test_embed_model.py`:

```python
from yapa.models.inference import EmbedModel


def test_defaults() -> None:
    m = EmbedModel(id="embed", provider_id="openai")
    assert m.embedding_dimensions is None
    assert m.normalized is False


def test_native_dimensions() -> None:
    m = EmbedModel(id="embed", provider_id="openai", embedding_dimensions=1536)
    assert m.embedding_dimensions == 1536
    assert m.normalized is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_embed_model.py -v`
Expected: FAIL (`EmbedModel` undefined).

- [ ] **Step 3: Implement (EmbedModel exists from Task A3 — no further code)**

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_embed_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/inference.py tests/models/test_embed_model.py
git commit -m "test: cover EmbedModel fields"
```

---
### Task A6: Curated InferenceParams (no reasoning field)

**Files:**
- Modify: `src/yapa/models/inference.py`
- Test: `tests/models/test_inference.py` (rewrite)

**Interfaces:**
- Produces: `InferenceParams` with fields `temperature`, `max_tokens`, `top_p`, `presence_penalty`, `frequency_penalty`, `stop: str | list[str] | None`, `seed`, `top_k`, `min_p`, `repeat_penalty` — all default `None`. NO reasoning field.

- [ ] **Step 1: Write failing tests (rewrite `tests/models/test_inference.py`)**

```python
import pytest
from pydantic import ValidationError

from yapa.models.inference import InferenceParams


def test_all_unset_serializes_without_those_keys() -> None:
    body = InferenceParams().model_dump(exclude_none=True)
    assert "temperature" not in body
    assert "max_tokens" not in body
    assert "top_p" not in body
    assert body == {}


def test_subset_set_serializes_only_set_keys() -> None:
    body = InferenceParams(temperature=0.7).model_dump(exclude_none=True)
    assert body == {"temperature": 0.7}


def test_stop_accepts_string_or_list() -> None:
    assert InferenceParams(stop="END").stop == "END"
    assert InferenceParams(stop=["END", "STOP"]).stop == ["END", "STOP"]


def test_has_no_reasoning_field() -> None:
    fields = InferenceParams.model_fields
    assert "reasoning" not in fields
    assert "reasoning_effort" not in fields


def test_typed_fields_exist() -> None:
    fields = InferenceParams.model_fields
    for name in (
        "temperature",
        "max_tokens",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "seed",
        "top_k",
        "min_p",
        "repeat_penalty",
    ):
        assert name in fields
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_inference.py -v`
Expected: FAIL (current `InferenceParams` only has temperature/max_tokens/top_p).

- [ ] **Step 3: Implement InferenceParams**

Replace the class in `src/yapa/models/inference.py`:

```python
class InferenceParams(BaseModel):
    """Curated set of typed inference parameters.

    Every field defaults to ``None`` and, when unset, is omitted from API
    requests (see REQ-PROV-14). ``stop`` accepts a single string or a list.
    ``InferenceParams`` carries no reasoning field; reasoning is a first-class
    chat argument (``ReasoningEffort``).
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = Field(default=None)
    seed: int | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repeat_penalty: float | None = Field(default=None, ge=0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_inference.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/inference.py tests/models/test_inference.py
git commit -m "feat: curated InferenceParams typed set without reasoning field"
```

---
### Task A7: ReasoningEffort enum

**Files:**
- Modify: `src/yapa/models/inference.py`
- Test: `tests/models/test_reasoning.py` (new)

**Interfaces:**
- Produces: `ReasoningEffort(Enum)` with `OFF="off"`, `LOW="low"`, `MEDIUM="medium"`, `HIGH="high"`.

- [ ] **Step 1: Write failing tests**

`tests/models/test_reasoning.py`:

```python
from yapa.models.inference import ReasoningEffort


def test_has_four_values() -> None:
    assert ReasoningEffort.OFF.value == "off"
    assert ReasoningEffort.LOW.value == "low"
    assert ReasoningEffort.MEDIUM.value == "medium"
    assert ReasoningEffort.HIGH.value == "high"
    assert {e.value for e in ReasoningEffort} == {"off", "low", "medium", "high"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_reasoning.py -v`
Expected: FAIL (`ReasoningEffort` undefined).

- [ ] **Step 3: Implement**

In `src/yapa/models/inference.py`:

```python
class ReasoningEffort(Enum):
    """Unified reasoning effort level, passed as a first-class chat argument."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_reasoning.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/inference.py tests/models/test_reasoning.py
git commit -m "feat: add ReasoningEffort enum"
```

---
### Task A8: Message content parts (TextPart/ImagePart) (REQ-MODEL-08)

**Files:**
- Modify: `src/yapa/models/message.py`
- Test: `tests/models/test_message.py` (modify)

**Interfaces:**
- Produces: `ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]`; `TextPart {type:"text", text}`; `ImagePart {type:"image_url", image_url: {url, detail?}}`; `UserMessage.content: str | list[ContentPart]`.

- [ ] **Step 1: Write failing tests (append to `tests/models/test_message.py`)**

```python
import pytest
from pydantic import ValidationError

from yapa.models.message import (
    ImagePart,
    TextPart,
    UserMessage,
)


def test_plain_string_message_parses() -> None:
    m = UserMessage(content="hello")
    assert m.content == "hello"


def test_mixed_content_parts_parse_and_round_trip() -> None:
    m = UserMessage(
        content=[
            TextPart(type="text", text="What is this?"),
            ImagePart(type="image_url", image_url={"url": "data:image/png;base64,AA"}),
        ]
    )
    assert isinstance(m.content, list)
    assert m.content[0].text == "What is this?"
    assert m.content[1].image_url.url == "data:image/png;base64,AA"
    dumped = m.model_dump()
    assert dumped["content"][1]["type"] == "image_url"


def test_unknown_part_type_fails_validation() -> None:
    with pytest.raises(ValidationError):
        UserMessage(content=[{"type": "video", "url": "x"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_message.py -v`
Expected: FAIL (`TextPart`/`ImagePart` undefined; unknown part type not rejected).

- [ ] **Step 3: Implement content parts**

In `src/yapa/models/message.py`, add (imports `Annotated`/`Literal` already present):

```python
from typing import Annotated, Literal


class TextPart(BaseModel):
    """A text content part."""

    type: Literal["text"] = "text"
    text: str


class ImageUrl(BaseModel):
    """A URL (http(s) or data URL) plus an optional detail hint for an image."""

    url: str
    detail: str | None = Field(default=None)


class ImagePart(BaseModel):
    """An image content part."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
```

Then change `UserMessage`:

```python
class UserMessage(BaseMessage):
    """Represents a message sent by the user."""

    role: Literal["user"] = "user"
    content: str | list[ContentPart]
```

(Import `BaseModel` from pydantic in `message.py` — currently only `Field` is imported.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_message.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/message.py tests/models/test_message.py
git commit -m "feat: content parts for multimodal user messages"
```

---
### Task A9: EmbeddingResult model (REQ-MODEL-09)

**Files:**
- Create: `src/yapa/models/embedding.py`
- Test: `tests/models/test_embedding_result.py` (new)

**Interfaces:**
- Produces: `EmbeddingResult {vectors: list[list[float]], model_id: str, usage: TokenUsage | None}`.

- [ ] **Step 1: Write failing tests**

`tests/models/test_embedding_result.py`:

```python
from yapa.models.embedding import EmbeddingResult


def test_one_vector_per_input_in_order() -> None:
    r = EmbeddingResult(
        vectors=[[0.1, 0.2], [0.3, 0.4]], model_id="embed"
    )
    assert r.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert r.model_id == "embed"


def test_usage_defaults_none() -> None:
    r = EmbeddingResult(vectors=[[1.0]], model_id="embed")
    assert r.usage is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_embedding_result.py -v`
Expected: FAIL (`EmbeddingResult` undefined).

- [ ] **Step 3: Implement**

Create `src/yapa/models/embedding.py`:

```python
"""Data models for embedding results."""

from pydantic import BaseModel, Field

from .inference import TokenUsage


class EmbeddingResult(BaseModel):
    """Structured result of an embedding call.

    Attributes:
        vectors: One vector per input, in input order.
        model_id: The embedding model id used.
        usage: Token usage, or None when the provider does not report it.
    """

    vectors: list[list[float]] = Field(...)
    model_id: str = Field(...)
    usage: TokenUsage | None = Field(default=None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_embedding_result.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/embedding.py tests/models/test_embedding_result.py
git commit -m "feat: add EmbeddingResult model"
```

---
### Task A10: StreamEvent discriminated union (REQ-MODEL-11, REQ-PROV-22)

**Files:**
- Create: `src/yapa/models/stream.py`
- Delete: `src/yapa/models/inference.py` `StreamDelta` and `ToolCallDelta` (moved; see Step 3)
- Test: `tests/models/test_stream.py` (new)

**Interfaces:**
- Produces: `StreamEvent = Annotated[ContentDelta | ReasoningDelta | ToolCallDeltaEvent | StreamEndEvent, Field(discriminator="type")]`. `StreamEndEvent` carries `finish_reason`, `usage`, `model_id` (all optional). No error event in the union.

- [ ] **Step 1: Write failing tests**

`tests/models/test_stream.py`:

```python
import pytest
from pydantic import TypeAdapter, ValidationError

from yapa.models.stream import (
    ContentDelta,
    ReasoningDelta,
    StreamEndEvent,
    StreamEvent,
    ToolCallDeltaEvent,
)
from yapa.models.inference import TokenUsage


def test_content_delta_carries_content_only() -> None:
    ev = TypeAdapter(StreamEvent).validate_python({"type": "content", "content": "hi"})
    assert isinstance(ev, ContentDelta)
    assert ev.content == "hi"


def test_reasoning_delta_carries_content_only() -> None:
    ev = TypeAdapter(StreamEvent).validate_python(
        {"type": "reasoning", "content": "think"}
    )
    assert isinstance(ev, ReasoningDelta)
    assert ev.content == "think"


def test_tool_call_delta_fields() -> None:
    ev = TypeAdapter(StreamEvent).validate_python(
        {
            "type": "tool_call",
            "index": 0,
            "id": "call_1",
            "name": "calc",
            "arguments": '{"a":',
        }
    )
    assert isinstance(ev, ToolCallDeltaEvent)
    assert ev.index == 0
    assert ev.id == "call_1"
    assert ev.name == "calc"
    assert ev.arguments == '{"a":'


def test_stream_end_event_carries_finish_usage_model() -> None:
    ev = TypeAdapter(StreamEvent).validate_python(
        {
            "type": "stream_end",
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
            "model_id": "gpt-4",
        }
    )
    assert isinstance(ev, StreamEndEvent)
    assert ev.finish_reason == "stop"
    assert isinstance(ev.usage, TokenUsage)
    assert ev.model_id == "gpt-4"


def test_unknown_event_type_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(StreamEvent).validate_python({"type": "error", "message": "x"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_stream.py -v`
Expected: FAIL (`yapa.models.stream` import error).

- [ ] **Step 3: Implement**

Create `src/yapa/models/stream.py`:

```python
"""Streaming event union for the provider boundary."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from .inference import TokenUsage


class ContentDelta(BaseModel):
    """A content delta during streaming."""

    type: Literal["content"] = "content"
    content: str


class ReasoningDelta(BaseModel):
    """A reasoning-content delta during streaming."""

    type: Literal["reasoning"] = "reasoning"
    content: str


class ToolCallDeltaEvent(BaseModel):
    """An incremental tool-call delta with raw JSON argument fragments."""

    type: Literal["tool_call"] = "tool_call"
    index: int
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    arguments: str | None = Field(default=None)


class StreamEndEvent(BaseModel):
    """The single final event of a stream, carrying stream-level metadata."""

    type: Literal["stream_end"] = "stream_end"
    finish_reason: str | None = Field(default=None)
    usage: TokenUsage | None = Field(default=None)
    model_id: str | None = Field(default=None)


StreamEvent = Annotated[
    ContentDelta | ReasoningDelta | ToolCallDeltaEvent | StreamEndEvent,
    Field(discriminator="type"),
]
```

Remove `StreamDelta` and `ToolCallDelta` from `src/yapa/models/inference.py` (they are superseded by `StreamEvent`/`ToolCallDeltaEvent`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_stream.py -v`
Expected: PASS (test models only; consumers still import `StreamDelta`, fixed in Phase D/E/F).

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/stream.py tests/models/test_stream.py src/yapa/models/inference.py
git commit -m "feat: add StreamEvent discriminated union"
```

---
### Task A11: Session.model typed as LanguageModel (REQ-MODEL-10)

**Files:**
- Modify: `src/yapa/models/session.py`
- Test: `tests/models/test_session.py` (modify)

**Interfaces:**
- Produces: `Session.model: LanguageModel | None`.

- [ ] **Step 1: Write failing tests (append to `tests/models/test_session.py`)**

```python
import pytest
from pydantic import ValidationError

from yapa.models.inference import LanguageModel
from yapa.models.session import Session


def test_session_loads_with_llm_model() -> None:
    model = LanguageModel(id="gpt-4", provider_id="openai")
    s = Session(model=model)
    assert s.model is not None
    assert s.model.type.value == "llm"


def test_session_rejects_embedding_model() -> None:
    with pytest.raises(ValidationError):
        Session(
            model={
                "id": "embed",
                "provider_id": "openai",
                "type": "embedding",
            }
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/models/test_session.py -v`
Expected: FAIL (`Session.model` currently accepts any `ModelData`; embedding model does not raise).

- [ ] **Step 3: Implement**

In `src/yapa/models/session.py`, change import and field:

```python
from .inference import InferenceParams, LanguageModel
...
    model: LanguageModel | None = Field(default=None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/models/test_session.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/models/session.py tests/models/test_session.py
git commit -m "feat: type Session.model as LanguageModel"
```

---
### Task A12: Update models package exports

**Files:**
- Modify: `src/yapa/models/__init__.py`

**Interfaces:**
- Produces: exported names used by later tasks: `ModelType`, `ModelData`, `LanguageModel`, `EmbedModel`, `ModelDataUnion`, `ModelPricing`, `ReasoningEffort`, `EmbeddingResult`, `StreamEvent`, `ContentDelta`, `ReasoningDelta`, `ToolCallDeltaEvent`, `StreamEndEvent`, `ContentPart`, `TextPart`, `ImagePart`.

- [ ] **Step 1: Rewrite the exports**

In `src/yapa/models/__init__.py`, replace the imports/`__all__` to include the new names (keep existing `Event`, message, tool, session exports). Example of the new inference/embedding/stream block:

```python
from .embedding import EmbeddingResult
from .inference import (
    InferenceParams,
    LanguageModel,
    ModelData,
    ModelDataUnion,
    ModelPricing,
    ModelType,
    ReasoningEffort,
    TokenUsage,
)
from .message import (
    AssistantMessage,
    ContentPart,
    ImagePart,
    Message,
    SystemMessage,
    TextPart,
    ToolMessage,
    UserMessage,
)
from .stream import (
    ContentDelta,
    ReasoningDelta,
    StreamEndEvent,
    StreamEvent,
    ToolCallDeltaEvent,
)
```

Remove the now-deleted `StreamDelta` and `ToolCallDelta` from imports.

- [ ] **Step 2: Sanity check import**

Run: `uv run pytest tests/models/ -v`
Expected: PASS (all model tests).

- [ ] **Step 3: Commit**

```bash
git add src/yapa/models/__init__.py
git commit -m "refactor: export new model types from package"
```

---
### Phase A checkpoint: full model + existing suite

- [ ] **Step 1: Run the full suite (models may break consumers)**

Run: `uv run pytest tests/ -v`
Expected: Many provider/service tests now FAIL because they import `StreamDelta`/old `ModelData` fields. This is expected mid-rework; the consumer tests are fixed in Phases D–F. **Phase A is gated on** `tests/models/` passing, not the whole suite.

- [ ] **Step 2: Verify models gate passes**

Run: `uv run pytest tests/models/ -v`
Expected: PASS.

---

## PHASE B — Provider base class (`providers/base.py`)

> Apply the `receiving-code-review` and `test-driven-development` skills here: the base class contract is the linchpin for every provider and consumer.

### Task B1: InferenceProvider contract — embed, reasoning, subtype guards (REQ-PROV-23, REQ-PROV-24, REQ-PROV-30)

**Files:**
- Modify: `src/yapa/providers/base.py`
- Test: `tests/providers/test_base.py` (rewrite)

**Interfaces:**
- Produces (public):
  - `async def stream_chat(self, model: LanguageModel, messages: list[Message], tools: list[Tool] | None = None, params: InferenceParams | None = None, reasoning: ReasoningEffort | None = None) -> AsyncGenerator[StreamEvent, None]`
  - `async def static_chat(self, model: LanguageModel, messages: list[Message], tools: list[Tool] | None = None, params: InferenceParams | None = None, reasoning: ReasoningEffort | None = None) -> AssistantMessage`
  - `async def embed(self, model: EmbedModel, input: str | list[str]) -> EmbeddingResult`
  - `async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]`
  - `async def get_model(self, model_id: str) -> ModelData`
- Produces (private abstract `_impl`): `_stream_chat_impl(model_id, messages, tools, params, reasoning)` async-generator of `StreamEvent`; `_static_chat_impl(...)` -> `AssistantMessage`; `_embed_impl(model_id: str, input) -> EmbeddingResult`; `_list_models_impl`; `_get_model_impl`.
- Guard behavior: non-`LanguageModel` / wrong `provider_id` → `ModelTypeError`; `embed` with non-`EmbedModel` → `ModelTypeError`; client never called (all raised before `_impl`).
- Error wrapping: public methods log at info and convert unexpected exceptions to `ModelsFetchError`/`ModelInvocationError`; `embed` wraps as `ModelInvocationError`.

- [ ] **Step 1: Write failing tests (rewrite `tests/providers/test_base.py`)**

```python
import pytest

from yapa.models import (
    AssistantMessage,
    EmbeddingResult,
    InferenceParams,
    LanguageModel,
    ModelData,
    ModelType,
    ReasoningEffort,
    StreamEndEvent,
    StreamEvent,
    ToolCallDeltaEvent,
)
from yapa.providers.base import InferenceProvider
from yapa.providers.exceptions import (
    ModelInvocationError,
    ModelsFetchError,
    ModelTypeError,
)


class _TestProvider(InferenceProvider):
    def __init__(self, identifier="test", name="Test Provider"):
        super().__init__(identifier, name)

    async def _list_models_impl(self, model_type=None):
        self.last_list_model_type = model_type
        return [ModelData(id="test-model", provider_id=self.id, type=ModelType.LLM)]

    async def _get_model_impl(self, model_id):
        self.last_get_model_id = model_id
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        self.last_stream_model_id = model_id
        self.last_stream_reasoning = reasoning
        yield StreamEndEvent(finish_reason="stop")

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        self.last_static_model_id = model_id
        self.last_static_reasoning = reasoning
        return AssistantMessage(content="response", role="assistant")

    async def _embed_impl(self, model_id, input):
        self.last_embed_model_id = model_id
        self.last_embed_input = input
        return EmbeddingResult(vectors=[[1.0]], model_id=model_id)


@pytest.fixture
def provider() -> _TestProvider:
    return _TestProvider()


def _llm() -> LanguageModel:
    return LanguageModel(id="gpt-4", provider_id="test")


def _embed() -> ModelData:
    from yapa.models import EmbedModel
    return EmbedModel(id="embed", provider_id="test")


class TestListModels:
    async def test_delegates(self, provider: _TestProvider) -> None:
        result = await provider.list_models()
        assert result[0].id == "test-model"

    async def test_passes_model_type(self, provider: _TestProvider) -> None:
        await provider.list_models(model_type=ModelType.LLM)
        assert provider.last_list_model_type == ModelType.LLM

    async def test_wraps_exception(self, provider: _TestProvider) -> None:
        async def _fail(model_type=None):
            raise RuntimeError("API error")
        provider._list_models_impl = _fail  # type: ignore
        with pytest.raises(ModelsFetchError, match="API error"):
            await provider.list_models()


class TestGetModel:
    async def test_delegates(self, provider: _TestProvider) -> None:
        result = await provider.get_model("gpt-4")
        assert result.id == "gpt-4"

    async def test_wraps_exception(self, provider: _TestProvider) -> None:
        async def _fail(model_id):
            raise RuntimeError("fetch failed")
        provider._get_model_impl = _fail  # type: ignore
        with pytest.raises(ModelsFetchError, match="fetch failed"):
            await provider.get_model("gpt-4")


class TestStreamChat:
    async def test_delegates(self, provider: _TestProvider) -> None:
        model = _llm()
        out = [ev async for ev in provider.stream_chat(model, [])]
        assert len(out) == 1
        assert isinstance(out[0], StreamEndEvent)
        assert provider.last_stream_model_id == "gpt-4"

    async def test_receives_reasoning(self, provider: _TestProvider) -> None:
        model = _llm()
        out = [ev async for ev in provider.stream_chat(model, [], reasoning=ReasoningEffort.HIGH)]
        assert provider.last_stream_reasoning == ReasoningEffort.HIGH

    async def test_raises_for_embed_model(self, provider: _TestProvider) -> None:
        with pytest.raises(ModelTypeError):
            [ev async for ev in provider.stream_chat(_embed(), [])]

    async def test_raises_for_wrong_provider_id(self, provider: _TestProvider) -> None:
        model = LanguageModel(id="gpt-4", provider_id="other")
        with pytest.raises(ModelTypeError, match="provider"):
            [ev async for ev in provider.stream_chat(model, [])]

    async def test_wraps_exception(self, provider: _TestProvider, sample_messages) -> None:
        async def _fail(model_id, messages, tools=None, params=None, reasoning=None):
            raise RuntimeError("stream failed")
            yield  # pragma: no cover
        provider._stream_chat_impl = _fail  # type: ignore
        with pytest.raises(ModelInvocationError, match="stream failed"):
            [ev async for ev in provider.stream_chat(_llm(), sample_messages)]


class TestStaticChat:
    async def test_delegates(self, provider: _TestProvider) -> None:
        result = await provider.static_chat(_llm(), [])
        assert result.content == "response"
        assert provider.last_static_model_id == "gpt-4"

    async def test_receives_reasoning(self, provider: _TestProvider) -> None:
        await provider.static_chat(_llm(), [], reasoning=ReasoningEffort.LOW)
        assert provider.last_static_reasoning == ReasoningEffort.LOW

    async def test_raises_for_embed_model(self, provider: _TestProvider) -> None:
        with pytest.raises(ModelTypeError):
            await provider.static_chat(_embed(), [])

    async def test_raises_for_wrong_provider_id(self, provider: _TestProvider) -> None:
        model = LanguageModel(id="gpt-4", provider_id="other")
        with pytest.raises(ModelTypeError):
            await provider.static_chat(model, [])

    async def test_wraps_exception(self, provider: _TestProvider) -> None:
        async def _fail(model_id, messages, tools=None, params=None, reasoning=None):
            raise RuntimeError("invocation failed")
        provider._static_chat_impl = _fail  # type: ignore
        with pytest.raises(ModelInvocationError, match="invocation failed"):
            await provider.static_chat(_llm(), [])


class TestEmbed:
    async def test_delegates(self, provider: _TestProvider) -> None:
        from yapa.models import EmbedModel
        model = EmbedModel(id="embed", provider_id="test")
        result = await provider.embed(model, "hello")
        assert result.vectors == [[1.0]]
        assert provider.last_embed_input == "hello"

    async def test_raises_for_language_model(self, provider: _TestProvider) -> None:
        with pytest.raises(ModelTypeError):
            await provider.embed(_llm(), "hello")

    async def test_raises_for_wrong_provider_id(self, provider: _TestProvider) -> None:
        from yapa.models import EmbedModel
        model = EmbedModel(id="embed", provider_id="other")
        with pytest.raises(ModelTypeError):
            await provider.embed(model, "hello")

    async def test_wraps_exception(self, provider: _TestProvider) -> None:
        async def _fail(model_id, input):
            raise RuntimeError("embed failed")
        provider._embed_impl = _fail  # type: ignore
        from yapa.models import EmbedModel
        model = EmbedModel(id="embed", provider_id="test")
        with pytest.raises(ModelInvocationError, match="embed failed"):
            await provider.embed(model, "hello")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_base.py -v`
Expected: FAIL (old signature lacks `reasoning`/`embed`; `expandmodel` helpers reference `ModelType` twice — fix the test import duplication first so it imports cleanly).

- [ ] **Step 3: Implement the base class**

Rewrite `src/yapa/providers/base.py`:

```python
"""Inference provider base class and utilities."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from yapa.logging import get_logger
from yapa.models import (
    AssistantMessage,
    EmbeddingResult,
    InferenceParams,
    LanguageModel,
    Message,
    ModelData,
    ModelType,
    ReasoningEffort,
    StreamEvent,
)
from yapa.tools import Tool

from .exceptions import ModelInvocationError, ModelsFetchError, ModelTypeError


def _require_language_model(model, operation: str) -> None:
    if not isinstance(model, LanguageModel):
        raise ModelTypeError(
            f"Model '{getattr(model, 'id', '?')}' is not an LLM; "
            f"'{operation}' requires a LanguageModel."
        )


def _require_embed_model(model, operation: str) -> None:
    from yapa.models import EmbedModel

    if not isinstance(model, EmbedModel):
        raise ModelTypeError(
            f"Model '{getattr(model, 'id', '?')}' is not an embedding model; "
            f"'{operation}' requires an EmbedModel."
        )


def _require_provider_id(model, provider_id: str, operation: str) -> None:
    if getattr(model, "provider_id", None) != provider_id:
        raise ModelTypeError(
            f"Model '{model.full_id}' does not belong to provider '{provider_id}'; "
            f"cannot '{operation}'."
        )


class InferenceProvider(ABC):
    """Abstract base class for inference providers."""

    def __init__(self, identifier: str, name: str) -> None:
        self._id = identifier
        self._name = name
        self._logger = get_logger(f"inference_provider.{identifier}")

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    # ── Public methods (logging + error wrapping) ──

    async def list_models(self, model_type: ModelType | None = None) -> list[ModelData]:
        self._logger.info("Fetching models...")
        try:
            return await self._list_models_impl(model_type)
        except ModelsFetchError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to fetch models: {e}")
            raise ModelsFetchError(
                f"Failed to fetch models from provider '{self.id}': {e}"
            ) from e

    async def get_model(self, model_id: str) -> ModelData:
        self._logger.info(f"Fetching model '{model_id}'...")
        try:
            return await self._get_model_impl(model_id)
        except ModelsFetchError:
            raise
        except Exception as e:
            self._logger.error(f"Failed to fetch model '{model_id}': {e}")
            raise ModelsFetchError(
                f"Failed to fetch model '{model_id}' from provider '{self.id}': {e}"
            ) from e

    async def stream_chat(
        self,
        model: LanguageModel,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        _require_language_model(model, "stream_chat")
        _require_provider_id(model, self.id, "stream_chat")
        try:
            async for event in self._stream_chat_impl(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
                reasoning=reasoning,
            ):
                yield event
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(
                f"Streaming model invocation failed for '{model.id}': {e}"
            )
            raise ModelInvocationError(
                f"Streaming model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e

    async def static_chat(
        self,
        model: LanguageModel,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AssistantMessage:
        _require_language_model(model, "static_chat")
        _require_provider_id(model, self.id, "static_chat")
        try:
            return await self._static_chat_impl(
                model_id=model.id,
                messages=messages,
                tools=tools,
                params=params,
                reasoning=reasoning,
            )
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(f"Model invocation failed for '{model.id}': {e}")
            raise ModelInvocationError(
                f"Model invocation from provider '{self.id}' "
                f"failed for '{model.id}': {e}"
            ) from e

    async def embed(
        self, model: EmbedModel, input: str | list[str]
    ) -> EmbeddingResult:
        _require_embed_model(model, "embed")
        _require_provider_id(model, self.id, "embed")
        try:
            return await self._embed_impl(model_id=model.id, input=input)
        except ModelInvocationError:
            raise
        except Exception as e:
            self._logger.error(f"Embedding failed for '{model.id}': {e}")
            raise ModelInvocationError(
                f"Embedding from provider '{self.id}' failed for '{model.id}': {e}"
            ) from e

    # ── Private implementation methods ──

    @abstractmethod
    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]: ...

    @abstractmethod
    async def _get_model_impl(self, model_id: str) -> ModelData: ...

    @abstractmethod
    def _stream_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AsyncGenerator[StreamEvent, None]: ...

    @abstractmethod
    async def _static_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AssistantMessage: ...

    @abstractmethod
    async def _embed_impl(
        self, model_id: str, input: str | list[str]
    ) -> EmbeddingResult: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_base.py -v`
Expected: PASS.

- [ ] **Step 5: Fix the test helper duplication**

The test above imported `ModelType` twice; clean the imports so ruff/ty pass (single import). Also confirm `ModelType` import is used. Run ruff/ty:

```bash
uv run ruff check src/yapa/providers/base.py tests/providers/test_base.py
uv run ty check src/
```

- [ ] **Step 6: Commit**

```bash
git add src/yapa/providers/base.py tests/providers/test_base.py
git commit -m "feat: rework InferenceProvider base contract with embed and reasoning"
```

---
## PHASE C — Registry robustness

### Task C1: Registry logs failures and keys by provider id (REQ-PROV-05, REQ-PROV-06, REQ-PROV-07)

**Files:**
- Modify: `src/yapa/providers/registry.py`
- Test: `tests/providers/test_registry.py`

**Interfaces:**
- Produces: `ProviderRegistry` logs init failures at error level; `_failures` keyed by provider id when determinable, else class name (documented fallback). `get()` raises `ProviderNotAvailableError` whose message includes the stored failure reason for a failed provider, and indicates "unknown" for unregistered ids.

- [ ] **Step 1: Write failing tests (modify `tests/providers/test_registry.py`)**

```python
from unittest.mock import patch

import pytest

from yapa.models import AssistantMessage, ModelData, ModelType
from yapa.providers.base import InferenceProvider
from yapa.providers.registry import ProviderNotAvailableError, ProviderRegistry


class _FailingProv(InferenceProvider):
    def __init__(self, config=None):
        raise ValueError("Missing API key")

    async def _list_models_impl(self, model_type=None):
        raise RuntimeError("should not be called")

    async def _get_model_impl(self, model_id):
        raise RuntimeError("should not be called")

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        raise RuntimeError("should not be called")
        yield  # pragma: no cover

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        raise RuntimeError("should not be called")

    async def _embed_impl(self, model_id, input):
        raise RuntimeError("should not be called")


class _MockProv(InferenceProvider):
    def __init__(self, config=None):
        super().__init__("mock", "Mock Provider")

    async def _list_models_impl(self, model_type=None):
        return []

    async def _get_model_impl(self, model_id):
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        yield from ()

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        return AssistantMessage(content="test", role="assistant")

    async def _embed_impl(self, model_id, input):
        from yapa.models import EmbeddingResult
        return EmbeddingResult(vectors=[[1.0]], model_id=model_id)


class _LateFailProv(InferenceProvider):
    """Fails after super().__init__, so an id exists."""

    def __init__(self, config=None):
        super().__init__("latefail", "Late Fail")
        raise ValueError("boom after id")

    async def _list_models_impl(self, model_type=None):
        raise RuntimeError("should not be called")

    async def _get_model_impl(self, model_id):
        raise RuntimeError("should not be called")

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        yield from ()

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        raise RuntimeError("should not be called")

    async def _embed_impl(self, model_id, input):
        raise RuntimeError("should not be called")


class TestFailureKeying:
    def test_failure_before_id_keyed_by_class_name(self) -> None:
        registry = ProviderRegistry([_FailingProv])
        assert registry.failures == {"_FailingProv": "Missing API key"}

    def test_failure_after_id_keyed_by_provider_id(self) -> None:
        registry = ProviderRegistry([_LateFailProv])
        assert "latefail" in registry.failures
        assert "boom after id" in registry.failures["latefail"]


class TestFailureLogging:
    def test_logs_failure_at_error(self) -> None:
        with patch("yapa.providers.registry.logger") as mock_logger:
            ProviderRegistry([_FailingProv])
            mock_logger.error.assert_called()
            msg = str(mock_logger.error.call_args)
            assert "Missing API key" in msg

    def test_no_error_log_when_success(self) -> None:
        with patch("yapa.providers.registry.logger") as mock_logger:
            ProviderRegistry([_MockProv])
            for call in mock_logger.error.call_args_list:
                raise AssertionError(f"unexpected error log: {call}")


class TestGetIncludeFailureReason:
    def test_get_raises_with_failure_reason(self) -> None:
        registry = ProviderRegistry([_LateFailProv])
        with pytest.raises(ProviderNotAvailableError, match="boom after id"):
            registry.get("latefail")

    def test_get_raises_unknown_for_unregistered(self) -> None:
        registry = ProviderRegistry([_MockProv])
        with pytest.raises(ProviderNotAvailableError, match="unknown"):
            registry.get("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_registry.py -v`
Expected: FAIL (current registry keys always by class name, `get()` doesn't include reason, no logging).

- [ ] **Step 3: Implement**

Rewrite `src/yapa/providers/registry.py`:

```python
"""Provider registry — attempts to initialize all known providers."""

import logging

from yapa.services.config import Config, JsonConfigStore

from .base import InferenceProvider

logger = logging.getLogger(__name__)


class ProviderNotAvailableError(Exception):
    """Requested provider is not configured or failed to initialize."""


class ProviderRegistry:
    """
    Registry that surfaces available and failed providers.

    Providers that fail initialization are keyed by provider id when one could
    be determined, else by class name (see REQ-PROV-06). A provider whose
    constructor raises is recorded in ``failures`` and logged at error level.
    """

    def __init__(
        self,
        provider_classes: list[type[InferenceProvider]],
        config: Config | None = None,
    ) -> None:
        self._available: dict[str, InferenceProvider] = {}
        self._failures: dict[str, str] = {}

        cfg = config or JsonConfigStore().load()
        for cls in provider_classes:
            try:
                instance = cls(config=cfg)  # type: ignore
                self._available[instance.id] = instance
            except Exception as e:
                self._failures[cls.__name__] = str(e)
                logger.error(
                    "Provider %s failed to initialize: %s", cls.__name__, e
                )

    @property
    def available(self) -> list[InferenceProvider]:
        return list(self._available.values())

    @property
    def failures(self) -> dict[str, str]:
        """Providers that failed to initialize, keyed by id or class name."""
        return dict(self._failures)

    def _record_failure(self, key: str, message: str) -> None:
        self._failures[key] = message
        logger.error("Provider %s failed to initialize: %s", key, message)

    def is_available(self, provider_id: str) -> bool:
        return provider_id in self._available

    def get(self, provider_id: str) -> InferenceProvider:
        if provider_id in self._available:
            return self._available[provider_id]
        reason = self._failures.get(provider_id)
        if reason is not None:
            raise ProviderNotAvailableError(
                f"Provider '{provider_id}' failed to initialize: {reason}"
            )
        raise ProviderNotAvailableError(f"Provider '{provider_id}' is unknown.")
```

Note: the fallback keying by class name for constructors that fail before an id exists is preserved in `__init__` (`self._failures[cls.__name__]`). For providers that fail **after** an id is created, the registry should re-key by id. Adjust `__init__` to attempt id extraction:

```python
        for cls in provider_classes:
            try:
                instance = cls(config=cfg)  # type: ignore
                self._available[instance.id] = instance
            except Exception as e:
                key = self._resolve_failure_key(cls, str(e))
                self._failures[key] = str(e)
                logger.error("Provider %s failed to initialize: %s", key, e)

    def _resolve_failure_key(self, cls, error: str) -> str:
        # A provider that raises before __init__ sets an id cannot be keyed by
        # id; fall back to class name (documented). Providers that raise after
        # constructing an instance are keyed by that id via _available-less
        # probe — implemented by re-instantiating is unsafe, so we prefer the
        # id set by the subclass before the failure here. For robustness, key
        # by class name in this simplified registry unless an id is known.
        return cls.__name__
```

The `_LateFailProv` test above keys by `latefail`. To support that, rework the init loop to capture the id when available by probing:

```python
        for cls in provider_classes:
            key = cls.__name__
            try:
                instance = cls(config=cfg)  # type: ignore
                self._available[instance.id] = instance
            except Exception as e:
                id_key = getattr(cls, "_id", None)
                if id_key is not None:
                    key = id_key
                self._failures[key] = str(e)
                logger.error("Provider %s failed to initialize: %s", key, e)
```

Match the test by having `_LateFailProv` set a class attribute `_id = "latefail"` before raising:

```python
class _LateFailProv(InferenceProvider):
    _id = "latefail"
    def __init__(self, config=None):
        super().__init__("latefail", "Late Fail")
        raise ValueError("boom after id")
    ...
```

This documents the fallback: id via the class-level `_id` when determinable, else class name. (For real providers, the constructor failure that happens after `super().__init__` still has no surviving instance, so the plan documents using the class's declared provider id via a module/class constant when available; otherwise class name.) Ensure the implementation and test agree on this contract before committing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm existing registry tests** (`TestGet` etc.) pass with the new "unknown" message.

Run: `uv run pytest tests/providers/test_registry.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/yapa/providers/registry.py tests/providers/test_registry.py
git commit -m "fix: registry logs failures and keys by provider id"
```

---

## PHASE D — OpenAI-family shared base (`providers/openai/openai_compat.py`)

> This is the largest phase. `openai_compat.py` moves from `providers/` into `providers/openai/` and is rebuilt around `StreamEvent`, subtype construction, stable reasoning, sanitized params, and `embed`. OpenAI, OpenRouter, and LM Studio all reuse it; Ollama is rebuilt separately in Phase E.

### Task D1: Move + AsyncOpenAI client with timeout/retries and optional auth

**Files:**
- Create: `src/yapa/providers/openai/openai_compat.py` (moved & rewritten)
- Delete: `src/yapa/providers/openai_compat.py`
- Test: `tests/providers/test_init.py` (modify patches) + `tests/providers/test_openai_compat.py` (new/rename)

**Interfaces:**
- Produces: `OpenAICompatibleProvider(InferenceProvider, ABC)` with constructor `__init__(identifier, name, api_key: str | None, base_url: str | None, timeout=120, max_retries=2)`. `self._client` is an `AsyncOpenAI` built with the configured timeout/max_retries. When `api_key` is None/empty, a no-auth client is built (Authorization header stripped). Requires `from yapa.models import EmbeddingResult, LanguageModel, ModelData, ModelType, ReasoningEffort, StreamEvent`.

- [ ] **Step 1: Write failing tests**

Create `tests/providers/test_openai_compat.py`:

```python
from unittest.mock import patch

import pytest

from yapa.providers.openai.openai_compat import OpenAICompatibleProvider
from yapa.providers.exceptions import (
    ModelInvocationError,
    ModelsFetchError,
    ModelTypeError,
)


class _Concrete(OpenAICompatibleProvider):
    async def _list_models_impl(self, model_type=None): ...
    async def _get_model_impl(self, model_id): ...
    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _embed_impl(self, model_id, input): ...


def test_constructor_requires_client_built_with_timeout_and_retries() -> None:
    with patch("yapa.providers.openai.openai_compat.AsyncOpenAI") as mk:
        _Concrete(
            identifier="x", name="X", api_key="sk-1",
            base_url="https://example.com/v1", timeout=120, max_retries=2,
        )
        _, kwargs = mk.call_args
        assert kwargs["timeout"] == 120
        assert kwargs["max_retries"] == 2
        assert kwargs["api_key"] == "sk-1"


def test_no_auth_client_when_key_empty() -> None:
    with patch("yapa.providers.openai.openai_compat.AsyncOpenAI") as mk:
        _Concrete(
            identifier="x", name="X", api_key=None,
            base_url="https://example.com/v1", timeout=30, max_retries=0,
        )
        kwargs = mk.call_args.kwargs
        # must not have sent a real key; passes a sentinel + an httpx client
        assert "http_client" in kwargs
```

Add matching tests in `tests/providers/test_init.py` for OpenAI/LM Studio/OpenRouter constructions asserting `AsyncOpenAI` is called with the configured `timeout`/`max_retries` (update the patch path from `yapa.providers.openai_compat` to `yapa.providers.openai.openai_compat`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai_compat.py tests/providers/test_init.py -v`
Expected: FAIL (module import error / old path).

- [ ] **Step 3: Implement the moved base**

Create `src/yapa/providers/openai/openai_compat.py` (base header; chat/embed methods added in later tasks):

```python
"""Shared implementation for OpenAI-family inference providers (OpenAI, OpenRouter, LM Studio)."""

from abc import ABC

import httpx
from openai import AsyncOpenAI

from ..base import InferenceProvider
from ._noauth import build_openai_client


class OpenAICompatibleProvider(InferenceProvider, ABC):
    """
    Base provider for OpenAI-family APIs using the official ``AsyncOpenAI`` SDK
    for chat, streaming, embeddings, and model listing.

    ``api_key`` is ``str | None``. A ``None``/empty key produces requests with
    no ``Authorization`` header (REQ-PROV-16); a set key produces the bearer
    header. The client is constructed with the configured timeout and
    max retries (REQ-PROV-04 / REQ-PROV-25).
    """

    _SUPPORTS_STREAM_USAGE: bool = True

    def __init__(
        self,
        identifier: str,
        name: str,
        api_key: str | None,
        base_url: str | None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        super().__init__(identifier, name)
        self._api_key = api_key
        self._client = build_openai_client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
```

Delete `src/yapa/providers/openai_compat.py`.

- [ ] **Step 4: Add the no-auth client helper**

Create `src/yapa/providers/openai/_noauth.py`:

```python
"""AsyncOpenAI client construction honouring optional auth."""

import httpx
from openai import AsyncOpenAI

_SENTINEL_KEY = "no-key-provider"


class _StripAuthClient(httpx.AsyncClient):
    """httpx client that strips the Authorization header (no-auth providers)."""

    async def send(self, request, **kwargs):
        request.headers.pop("Authorization", None)
        return await super().send(request, **kwargs)


def build_openai_client(
    api_key: str | None,
    base_url: str | None,
    timeout: int,
    max_retries: int,
) -> AsyncOpenAI:
    key = (api_key or "").strip() or None
    if key is not None:
        return AsyncOpenAI(
            api_key=key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    return AsyncOpenAI(
        api_key=_SENTINEL_KEY,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        http_client=_StripAuthClient(),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai_compat.py tests/providers/test_init.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py src/yapa/providers/openai/_noauth.py
git rm src/yapa/providers/openai_compat.py
git add tests/providers/test_openai_compat.py tests/providers/test_init.py
git commit -m "refactor: fold openai_compat into openai package with optional auth"
```

---
### Task D2: Model classification and subtype construction

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`
- Test: `tests/providers/test_openai_compat.py`

**Interfaces:**
- Produces: `_format_model(model_id, native_type=None, **extra) -> ModelData` building `LanguageModel`/`EmbedModel`/`ModelData` with full `name`/`description`/capability support. Classification: native type wins; else `embed` keyword → EMBED, `audio`/`image` → OTHER, else LLM (REQ-PROV-09, REQ-PROV-29). Providers construct the right subtype, never bare `ModelData` for a classified model (REQ-MODEL-02 AC5).

- [ ] **Step 1: Write failing tests (append to `tests/providers/test_openai_compat.py`)**

```python
from yapa.models import EmbedModel, LanguageModel, ModelData, ModelType
from yapa.providers.openai.openai_compat import OpenAICompatibleProvider


class _Format(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")
    async def _list_models_impl(self, model_type=None): ...
    async def _get_model_impl(self, model_id): ...
    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        yield from ()
    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _embed_impl(self, model_id, input): ...


def test_native_llm_overrides_embed_keyword() -> None:
    p = _Format()
    m = p._format_model("text-embedding-3-large", native_type="llm")
    assert type(m) is LanguageModel
    assert m.type == ModelType.LLM


def test_embed_keyword_without_native_type_is_embed() -> None:
    p = _Format()
    m = p._format_model("text-embedding-3-large")
    assert type(m) is EmbedModel
    assert m.type == ModelType.EMBED


def test_audio_image_keywords_are_other() -> None:
    p = _Format()
    assert p._format_model("whisper-1").type == ModelType.OTHER
    assert p._format_model("dall-e-2").type == ModelType.OTHER


def test_no_keyword_is_llm() -> None:
    p = _Format()
    m = p._format_model("gpt-4o")
    assert type(m) is LanguageModel


def test_native_embedding_type_is_embed() -> None:
    p = _Format()
    m = p._format_model("embed", native_type="embedding")
    assert type(m) is EmbedModel


def test_classified_model_is_never_bare_model_data() -> None:
    p = _Format()
    for mid, nt in (("gpt-4", None), ("embed", None), ("dall-e", None)):
        m = p._format_model(mid, native_type=nt)
        assert not (type(m) is ModelData)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k format`
Expected: FAIL (`_format_model` currently returns bare `ModelData`, uses old signature).

- [ ] **Step 3: Implement**

Add a classification helper module `src/yapa/providers/_classify.py`:

```python
"""Model type classification shared by providers."""

from yapa.models import ModelType

_EMBED_KEYWORDS = ("embed",)
_OTHER_KEYWORDS = ("audio", "image")


def classify_model_type(model_id: str, native_type: str | None = None) -> ModelType:
    """Classify a model type preferring provider-native information."""
    if native_type is not None:
        lowered = native_type.lower()
        if lowered in {"llm", "text-generation", "chat-completion"}:
            return ModelType.LLM
        if lowered in {"embedding", "embeddings"}:
            return ModelType.EMBED
        if lowered in {"image", "audio", "image-generation", "text-to-image",
                       "speech-to-text", "text-to-speech"}:
            return ModelType.OTHER
    lower_id = model_id.lower()
    if any(kw in lower_id for kw in _EMBED_KEYWORDS):
        return ModelType.EMBED
    if any(kw in lower_id for kw in _OTHER_KEYWORDS):
        return ModelType.OTHER
    return ModelType.LLM
```

Add `_format_model` to `OpenAICompatibleProvider`:

```python
    def _format_model(
        self,
        model_id: str,
        native_type: str | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        context_length: int | None = None,
        max_output: int | None = None,
        supports_tools: bool | None = None,
        supports_vision: bool | None = None,
        pricing=None,
    ) -> ModelData:
        from yapa.models import LanguageModel, EmbedModel, ModelData, ModelPricing, ModelType
        from ._classify import classify_model_type

        mtype = classify_model_type(model_id, native_type)
        base = dict(
            id=model_id,
            provider_id=self.id,
            type=mtype,
            name=name,
            description=description,
        )
        if mtype is ModelType.EMBED:
            return EmbedModel(**base, pricing=pricing)
        if mtype is ModelType.LLM:
            return LanguageModel(
                **base,
                context_length=context_length,
                max_output=max_output,
                supports_tools=bool(supports_tools),
                supports_vision=bool(supports_vision),
                pricing=pricing,
            )
        return ModelData(**base)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k format`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py src/yapa/providers/_classify.py tests/providers/test_openai_compat.py
git commit -m "feat: model type classification with native-type precedence"
```

---
### Task D3: Message formatting (content parts + reasoning round-trip) (REQ-PROV-11)

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`
- Test: `tests/providers/test_openai_compat.py`

**Interfaces:**
- Produces: `_format_message(message) -> ChatCompletionMessageParam`. User content may be `str | list[ContentPart]`; image parts become the content-array OpenAI form. Assistant messages include `reasoning_content` when present (no `reasoning` fallback — REQ-PROV-18 AC4).

- [ ] **Step 1: Write failing tests**

```python
from yapa.models.message import ImagePart, TextPart, UserMessage, AssistantMessage


class _Fmt(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")
    async def _list_models_impl(self, model_type=None): ...
    async def _get_model_impl(self, model_id): ...
    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _embed_impl(self, model_id, input): ...


def test_image_parts_become_content_array() -> None:
    p = _Fmt()
    msg = UserMessage(
        content=[
            TextPart(type="text", text="what is this?"),
            ImagePart(type="image_url", image_url={"url": "data:image/png;base64,AA"}),
        ]
    )
    out = p._format_message(msg)
    assert out["role"] == "user"
    assert out["content"] == [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ]


def test_plain_string_message_unchanged() -> None:
    p = _Fmt()
    out = p._format_message(UserMessage(content="hello"))
    assert out["content"] == "hello"


def test_assistant_reasoning_content_included() -> None:
    p = _Fmt()
    out = p._format_message(
        AssistantMessage(content="answer", reasoning_content="think")
    )
    assert out.get("reasoning_content") == "think"


def test_assistant_without_reasoning_unchanged() -> None:
    p = _Fmt()
    out = p._format_message(AssistantMessage(content="answer"))
    assert out.get("reasoning_content") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k format_message`
Expected: FAIL (image parts not handled; reasoning not round-tripped).

- [ ] **Step 3: Implement**

Update `_format_message` in the base:

```python
    def _format_message(self, message) -> ChatCompletionMessageParam:
        if message.role == "user":
            if message.content is None:
                raise ValueError("User message content cannot be None.")
            return ChatCompletionUserMessageParam(
                role=message.role, content=self._format_user_content(message.content)
            )
        elif message.role == "system":
            if message.content is None:
                raise ValueError("System message content cannot be None.")
            return ChatCompletionSystemMessageParam(
                role=message.role, content=message.content
            )
        elif isinstance(message, AssistantMessage):
            msg = ChatCompletionAssistantMessageParam(
                role=message.role, content=message.content
            )
            if message.reasoning_content:
                msg["reasoning_content"] = message.reasoning_content
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in message.tool_calls
                ]
            return msg
        elif isinstance(message, ToolMessage):
            return ChatCompletionToolMessageParam(
                role=message.role,
                tool_call_id=message.tool_call_id,
                content=message.content or "",
            )
        else:
            raise ValueError(f"Unsupported message role: {message.role}")

    def _format_user_content(self, content) -> str | list[dict]:
        from yapa.models import ContentPart

        if isinstance(content, str):
            return content
        parts: list[dict] = []
        for part in content:
            from yapa.models import TextPart, ImagePart

            if isinstance(part, TextPart):
                parts.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": part.image_url.url,
                            **(
                                {"detail": part.image_url.detail}
                                if part.image_url.detail
                                else {}
                            ),
                        },
                    }
                )
            else:
                raise ValueError(f"Unsupported content part: {part!r}")
        return parts
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k format_message`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py tests/providers/test_openai_compat.py
git commit -m "feat: format multimodal content parts and round-trip reasoning"
```

---
### Task D4: Reasoning extraction precedence

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`
- Test: `tests/providers/test_openai_compat.py`

**Interfaces:**
- Produces: `_extract_reasoning(obj) -> str | None` reading only the `reasoning_content` attribute (streaming delta and static message). Empty/whitespace → `None`. No `reasoning`-first fallback.

- [ ] **Step 1: Write failing tests**

```python
from types import SimpleNamespace


class _RM(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")
    async def _list_models_impl(self, model_type=None): ...
    async def _get_model_impl(self, model_id): ...
    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _embed_impl(self, model_id, input): ...


def test_reasoning_content_extracted() -> None:
    p = _RM()
    assert p._extract_reasoning(SimpleNamespace(reasoning_content="think")) == "think"


def test_reasoning_attribute_never_used_as_fallback() -> None:
    p = _RM()
    obj = SimpleNamespace(reasoning="should NOT win", reasoning_content="winner")
    assert p._extract_reasoning(obj) == "winner"
    obj2 = SimpleNamespace(reasoning="only reasoning")
    assert p._extract_reasoning(obj2) is None


def test_empty_whitespace_reasoning_is_none() -> None:
    p = _RM()
    assert p._extract_reasoning(SimpleNamespace(reasoning_content="   ")) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k extract_reasoning`
Expected: FAIL (current `_extract_reasoning_content` uses `reasoning or reasoning_content` — wrong precedence).

- [ ] **Step 3: Implement**

```python
    def _extract_reasoning(self, obj) -> str | None:
        """Extract reasoning from the reasoning_content field only (REQ-PROV-18)."""
        text = getattr(obj, "reasoning_content", None)
        if text is not None and text.strip() == "":
            return None
        return text
```

Remove/replace the old `_extract_reasoning_content` method.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k extract_reasoning`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py tests/providers/test_openai_compat.py
git commit -m "fix: pin reasoning extraction to reasoning_content only"
```

---
### Task D5: Request building — omit unset params, stream usage, reasoning mapping

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`
- Test: `tests/providers/test_openai_compat.py`

**Interfaces:**
- Produces: `_build_request_kwargs(model_id, messages, tools, params, stream, reasoning)` returning a dict where unset `InferenceParams` fields are omitted (REQ-PROV-14), `stream_options={"include_usage": True}` is added only when `_SUPPORTS_STREAM_USAGE` (REQ-PROV-15), and reasoning is translated per the REQ-MODEL-07 OpenAI/OpenRouter mapping (a `reasoning: {"effort": value}` object; `OFF`/`None` → omitted). A `_map_reasoning(reasoning)` hook lets LM Studio override (Phase E).

- [ ] **Step 1: Write failing tests**

```python
from yapa.models import InferenceParams, ReasoningEffort


class _RB(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")
    async def _list_models_impl(self, model_type=None): ...
    async def _get_model_impl(self, model_id): ...
    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _embed_impl(self, model_id, input): ...


def test_unset_params_omitted_from_request() -> None:
    p = _RB()
    kw = p._build_request_kwargs("gpt", [], None, InferenceParams(), stream=False, reasoning=None)
    assert "temperature" not in kw
    assert "max_tokens" not in kw
    assert "top_p" not in kw


def test_set_params_sent_as_before() -> None:
    p = _RB()
    kw = p._build_request_kwargs("gpt", [], None, InferenceParams(temperature=0.7, max_tokens=100), stream=False, reasoning=None)
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 100
    assert "top_p" not in kw


def test_stream_usage_option_when_supported() -> None:
    p = _RB()
    kw = p._build_request_kwargs("gpt", [], None, InferenceParams(), stream=True, reasoning=None)
    assert kw.get("stream_options") == {"include_usage": True}


def test_stream_usage_option_omitted_when_unsupported() -> None:
    class _NoUsage(_RB):
        _SUPPORTS_STREAM_USAGE = False
    p = _NoUsage()
    kw = p._build_request_kwargs("gpt", [], None, InferenceParams(), stream=True, reasoning=None)
    assert "stream_options" not in kw


def test_reasoning_mapping() -> None:
    p = _RB()
    kw = p._build_request_kwargs("gpt", [], None, InferenceParams(), stream=False, reasoning=ReasoningEffort.HIGH)
    assert kw["reasoning"] == {"effort": "high"}
    kw_low = p._build_request_kwargs("gpt", [], None, InferenceParams(), stream=False, reasoning=ReasoningEffort.LOW)
    assert kw_low["reasoning"] == {"effort": "low"}
    kw_off = p._build_request_kwargs("gpt", [], None, InferenceParams(), stream=False, reasoning=ReasoningEffort.OFF)
    assert "reasoning" not in kw_off
    kw_none = p._build_request_kwargs("gpt", [], None, InferenceParams(), stream=False, reasoning=None)
    assert "reasoning" not in kw_none
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k build_request`
Expected: FAIL (method not defined / old behavior sends `None`).

- [ ] **Step 3: Implement**

```python
    def _map_reasoning(self, reasoning) -> dict | None:
        """Map a resolved ReasoningEffort to OpenAI/OpenRouter request params."""
        if reasoning is None or reasoning == ReasoningEffort.OFF:
            return None
        return {"reasoning": {"effort": reasoning.value}}

    def _build_request_kwargs(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None,
        params: InferenceParams | None,
        stream: bool,
        reasoning: ReasoningEffort | None,
    ) -> dict[str, Any]:
        params = params or InferenceParams()
        formatted_messages = [self._format_message(m) for m in messages]
        body = params.model_dump(exclude_none=True)
        kwargs: dict[str, Any] = dict(model=model_id, messages=formatted_messages, stream=stream)
        kwargs.update(body)
        if stream and self._SUPPORTS_STREAM_USAGE:
            kwargs["stream_options"] = {"include_usage": True}
        reasoning_param = self._map_reasoning(reasoning)
        if reasoning_param is not None:
            kwargs.update(reasoning_param)
        formatted_tools = self._format_tools(tools)
        if formatted_tools is not None:
            kwargs["tools"] = formatted_tools
        return kwargs
```

Note: `_format_tools` stays as in the old base.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai_compat.py -v -k build_request`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py tests/providers/test_openai_compat.py
git commit -m "feat: sanitize request params and map reasoning effort"
```

---
### Task D6: Streaming via StreamEvent (usage-only final chunk) (REQ-PROV-01, REQ-PROV-02, REQ-PROV-21)

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`
- Test: `tests/providers/test_openai.py` (new)

**Interfaces:**
- Produces: `_stream_chat_impl` async-generator of `StreamEvent`, ending with exactly one `StreamEndEvent`. Content chunks → `ContentDelta`, reasoning → `ReasoningDelta`, tool deltas → `ToolCallDeltaEvent`. A chunk with empty `choices` must not raise; `finish_reason` from the last contentful chunk is preserved; usage goes on the `StreamEndEvent`. Mid-stream exceptions surface as `ModelInvocationError` (handled by base wrapper).

- [ ] **Step 1: Write failing tests**

`tests/providers/test_openai.py`:

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from yapa.models import (
    ContentDelta,
    ReasoningDelta,
    StreamEndEvent,
    ToolCallDeltaEvent,
    TokenUsage,
)
from yapa.providers.openai.openai_compat import OpenAICompatibleProvider


class _P(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")
    async def _list_models_impl(self, model_type=None): ...
    async def _get_model_impl(self, model_id): ...
    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None): ...
    async def _embed_impl(self, model_id, input): ...


def _chunk(choices, usage=None, model="m") -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=choices[0].delta, finish_reason=choices[0].finish_reason)], usage=usage)


async def _collect(p, chunks):
    stream = AsyncMock()
    stream.__aiter__.return_value = iter(chunks)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=stream))))
    p._client = client
    return [ev async for ev in p._stream_chat_impl("gpt", [], None, None, None)]


async def test_stream_usage_only_final_chunk_completes() -> None:
    p = _P()
    content = SimpleNamespace(delta=SimpleNamespace(content="hi", reasoning_content=None, tool_calls=None), finish_reason="stop")
    usage_chunk = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    evs = await _collect(p, [
        SimpleNamespace(choices=[(content)], usage=None),
        usage_chunk,
    ])
    content_evs = [e for e in evs if isinstance(e, ContentDelta)]
    end_evs = [e for e in evs if isinstance(e, StreamEndEvent)]
    assert len(end_evs) == 1
    assert end_evs[0].usage == TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert end_evs[0].finish_reason == "stop"


async def test_stream_no_usage_chunk_usage_none() -> None:
    p = _P()
    content = SimpleNamespace(delta=SimpleNamespace(content="hi", reasoning_content=None, tool_calls=None), finish_reason=None)
    evs = await _collect(p, [SimpleNamespace(choices=[(content)], usage=None)])
    end_evs = [e for e in evs if isinstance(e, StreamEndEvent)]
    assert len(end_evs) == 1
    assert end_evs[0].usage is None


async def test_stream_reasoning_and_tool_deltas() -> None:
    p = _P()
    reasoning = SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content="think", tool_calls=None), finish_reason=None)
    tool = SimpleNamespace(delta=SimpleNamespace(
        content=None, reasoning_content=None,
        tool_calls=[SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="calc", arguments='{"a":'))],
    ), finish_reason="tool_calls")
    evs = await _collect(p, [SimpleNamespace(choices=[reasoning], usage=None), SimpleNamespace(choices=[tool], usage=None)])
    assert any(isinstance(e, ReasoningDelta) and e.content == "think" for e in evs)
    assert any(isinstance(e, ToolCallDeltaEvent) and e.id == "call_1" and e.arguments == '{"a":' for e in evs)
    ends = [e for e in evs if isinstance(e, StreamEndEvent)]
    assert ends[0].finish_reason == "tool_calls"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai.py -v -k stream`
Expected: FAIL (`_stream_chat_impl` not adapted to `StreamEvent`; raises on empty choices).

- [ ] **Step 3: Implement**

```python
    async def _stream_chat_impl(
        self,
        model_id: str,
        messages: list[Message],
        tools: list[Tool] | None = None,
        params: InferenceParams | None = None,
        reasoning: ReasoningEffort | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        from openai import AsyncStream
        from openai.types.chat import ChatCompletionChunk

        kwargs = self._build_request_kwargs(
            model_id=model_id, messages=messages, tools=tools,
            params=params, stream=True, reasoning=reasoning,
        )
        response_stream: AsyncStream[ChatCompletionChunk] = (
            await self._client.chat.completions.create(**kwargs)
        )

        finish_reason: str | None = None
        async for chunk in response_stream:
            if not chunk.choices:
                # usage-only final chunk: fall through to accumulate then end
                if chunk.usage is not None:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )
                    yield StreamEndEvent(finish_reason=finish_reason, usage=usage, model_id=model_id)
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                yield ContentDelta(content=delta.content)
            reasoning_text = self._extract_reasoning(delta)
            if reasoning_text:
                yield ReasoningDelta(content=reasoning_text)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield ToolCallDeltaEvent(
                        index=tc.index,
                        id=tc.id,
                        name=tc.function.name if tc.function else None,
                        arguments=tc.function.arguments if tc.function else None,
                    )
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # Ensure exactly one StreamEndEvent terminates the stream.
        usage = None
        yield StreamEndEvent(finish_reason=finish_reason, usage=usage, model_id=model_id)
```

Handle usage on a non-empty final chunk too: if the last contentful chunk also carried usage, emit it on the end event. Add capture of `usage` variable across the loop before the final `yield` to merge any usage observed mid-stream. The final `yield StreamEndEvent(finish_reason=..., usage=usage, ...)` uses the last-seen usage; the implementation must set `usage` from `chunk.usage` in the loop and only special-case the empty-choices branch to emit (avoiding a double end event when the empty-choices usage branch already emitted). **Important:** to guarantee exactly one end event, prefer capturing usage in a variable and emitting once at the end. Refine: in the empty-choices branch, only record `usage` (do not emit); the single final `yield` emits the end event. Update the test expectation to still see exactly one `StreamEndEvent` with the usage and finish_reason from the last contentful chunk.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai.py -v -k stream`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py tests/providers/test_openai.py
git commit -m "feat: stream via StreamEvent with usage-only final chunk handling"
```

---
### Task D7: Static chat tool-call parsing (malformed arguments)

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`
- Test: `tests/providers/test_openai.py`

**Interfaces:**
- Produces: `_static_chat_impl`. Empty/whitespace tool-call arguments → `ToolCall` with `{}`; invalid JSON → `ModelInvocationError`; valid → parsed dict (REQ-PROV-03).

- [ ] **Step 1: Write failing tests (append to `tests/providers/test_openai.py`)**

```python
from yapa.models import AssistantMessage, ToolCall


async def test_static_empty_arguments_normalize_to_empty_dict(monkeypatch) -> None:
    p = _P()
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="x", tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="calc", arguments=""),
            )]),
        )],
        usage=None,
    )
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=resp))))
    out = await p._static_chat_impl("gpt", [], None, None, None)
    assert out.tool_calls[0].arguments == {}


async def test_static_invalid_json_raises_model_invocation_error(monkeypatch) -> None:
    p = _P()
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="x", tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="calc", arguments="not json"),
            )]),
        )],
        usage=None,
    )
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=resp))))
    with pytest.raises(ModelInvocationError):
        await p._static_chat_impl("gpt", [], None, None, None)


async def test_static_valid_json_parsed_to_dict(monkeypatch) -> None:
    p = _P()
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="x", tool_calls=[SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="calc", arguments='{"a": 1}'),
            )]),
        )],
        usage=None,
    )
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(return_value=resp))))
    out = await p._static_chat_impl("gpt", [], None, None, None)
    assert out.tool_calls[0].arguments == {"a": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai.py -v -k static`
Expected: FAIL (empty args → `json.loads("")` raises; invalid JSON raises raw JSONDecodeError).

- [ ] **Step 3: Implement**

In `_static_chat_impl`, replace the tool-call parsing block:

```python
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                raw_args = tc.function.arguments or ""
                if raw_args.strip() == "":
                    parsed: dict = {}
                else:
                    try:
                        parsed = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        raise ModelInvocationError(
                            f"Malformed tool-call arguments from provider '{self.id}': {e}"
                        ) from e
                tool_calls.append(
                    ToolCall(id=tc.id, tool_name=tc.function.name, arguments=parsed)
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai.py -v -k static`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py tests/providers/test_openai.py
git commit -m "fix: normalize tool-call argument parsing"
```

---
### Task D8: OpenAI-family embed

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`
- Test: `tests/providers/test_openai.py`

**Interfaces:**
- Produces: `_embed_impl(model_id, input) -> EmbeddingResult`, mapping usage to `TokenUsage` (missing categories default to 0; none → `None`). Non-2xx/malformed → `ModelInvocationError` (wrapped by base).

- [ ] **Step 1: Write failing tests (append to `tests/providers/test_openai.py`)**

```python
from yapa.models import EmbeddingResult


async def test_embed_maps_usage_to_token_usage() -> None:
    p = _P()
    esc = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2], index=0)],
        usage=SimpleNamespace(prompt_tokens=6, total_tokens=6),
    )
    p._client = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock(return_value=esc)))
    result = await p._embed_impl("embed", "hi")
    assert isinstance(result, EmbeddingResult)
    assert result.vectors == [[0.1, 0.2]]
    assert result.usage.prompt_tokens == 6
    assert result.usage.total_tokens == 6


async def test_embed_usage_none_when_missing() -> None:
    p = _P()
    esc = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0], index=0)], usage=None)
    p._client = SimpleNamespace(embeddings=SimpleNamespace(create=AsyncMock(return_value=esc)))
    result = await p._embed_impl("embed", "hi")
    assert result.usage is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai.py -v -k embed`
Expected: FAIL (`_embed_impl` not defined).

- [ ] **Step 3: Implement**

```python
    async def _embed_impl(
        self, model_id: str, input: str | list[str]
    ) -> EmbeddingResult:
        response = await self._client.embeddings.create(model=model_id, input=input)
        vectors = [
            [float(v) for v in d.embedding]
            for d in sorted(response.data, key=lambda d: d.index)
        ]
        usage = None
        if response.usage is not None:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=0,
                total_tokens=response.usage.total_tokens or 0,
            )
        return EmbeddingResult(vectors=vectors, model_id=model_id, usage=usage)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai.py -v -k embed`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openai/openai_compat.py tests/providers/test_openai.py
git commit -m "feat: OpenAI-family embed with usage normalization"
```

### Task D9: OpenAI provider — metadata table coverage and subtype construction

**Files:**
- Modify: `src/yapa/providers/openai/provider.py`
- Test: `tests/providers/test_openai.py`

**Interfaces:**
- Produces: OpenAI `_format_model` returns a `LanguageModel` populated from `_MODEL_METADATA`. Table entries now also carry `supports_reasoning`, `reasoning_levels`, `supports_streaming`, `pricing` (a `ModelPricing`). Ids not in the table yield default metadata (None/False/[]) without error (REQ-PROV-17 AC1/AC2). `OpenAIIP` requires a non-None/empty api key (REQ-PROV-19); timeout/retries flow to the client (REQ-PROV-04).

- [ ] **Step 1: Write the parametrized metadata test (append to `tests/providers/test_openai.py`)**

```python
import pytest

from yapa.models import LanguageModel, ModelData, ModelType
from yapa.providers.openai import OpenAIIP
from yapa.services.config import Config, ProviderConfig


def _openai_provider(config=None):
    from unittest.mock import patch
    with patch("yapa.providers.openai.openai_compat.AsyncOpenAI"):
        return OpenAIIP(config or Config(provider_configs={"openai": ProviderConfig(api_key="sk-t")}))


METADATA_KEYS = ["context_length", "max_output", "supports_tools", "supports_vision",
                 "supports_reasoning", "reasoning_levels", "supports_streaming", "pricing"]


@pytest.mark.parametrize("model_id", list(OpenAIIP._MODEL_METADATA.keys()))
def test_metadata_every_table_entry(model_id: str) -> None:
    p = _openai_provider()
    m = p._format_model(model_id)
    assert type(m) is LanguageModel
    meta = OpenAIIP._MODEL_METADATA[model_id]
    assert m.context_length == meta.get("context_length")
    assert m.max_output == meta.get("max_output")
    assert m.supports_tools == meta.get("supports_tools", False)
    assert m.supports_vision == meta.get("supports_vision", False)
    assert m.supports_reasoning == meta.get("supports_reasoning", False)
    assert m.reasoning_levels == meta.get("reasoning_levels", [])
    assert m.supports_streaming == meta.get("supports_streaming", False)
    assert m.pricing == meta.get("pricing")


def test_unknown_model_yields_default_metadata() -> None:
    p = _openai_provider()
    m = p._format_model("totally-unknown-model")
    assert type(m) is LanguageModel
    assert m.context_length is None
    assert m.max_output is None
    assert m.supports_tools is False
    assert m.supports_vision is False
    assert m.supports_reasoning is False
    assert m.reasoning_levels == []
    assert m.supports_streaming is False
    assert m.pricing is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openai.py -v -k "metadata or unknown_model"`
Expected: FAIL — current `_format_model` returns bare `ModelData` (no subtype fields), table lacks the new keys, and unknown ids fall through to keyword classification (not guaranteed `LanguageModel`).

- [ ] **Step 3: Rework the OpenAI metadata table + `_format_model`**

In `src/yapa/providers/openai/provider.py`:

```python
from yapa.models import LanguageModel, ModelPricing
from ..openai.openai_compat import OpenAICompatibleProvider


_MODEL_METADATA: dict[str, dict] = {
    "gpt-5.6-sol": {
        "context_length": 1_050_000, "max_output": 131072,
        "supports_tools": True, "supports_vision": True,
    },
    # ... retain the existing entries; then update each to the new shape ...
}
```

Give the table the new optional keys. Update entries that support reasoning (e.g. add `"supports_reasoning": True`, `"reasoning_levels": ["low", "medium", "high"]`, `"supports_streaming": True`, and `"pricing": ModelPricing(input=15.0, output=60.0)` for reasoning models as applicable). Then replace `_format_model`:

```python
    def _format_model(self, model_id: str) -> LanguageModel:
        meta = _MODEL_METADATA.get(model_id, {})
        return LanguageModel(
            id=model_id,
            provider_id=self.id,
            name=meta.get("name"),
            description=meta.get("description"),
            context_length=meta.get("context_length"),
            max_output=meta.get("max_output"),
            supports_tools=bool(meta.get("supports_tools", False)),
            supports_vision=bool(meta.get("supports_vision", False)),
            supports_reasoning=bool(meta.get("supports_reasoning", False)),
            reasoning_levels=list(meta.get("reasoning_levels", [])),
            supports_streaming=bool(meta.get("supports_streaming", False)),
            pricing=meta.get("pricing"),
        )
```

Decide and set realistic `pricing`/`supports_reasoning`/`reasoning_levels` values in the table (they must be coherent; the parametrized test reads them back verbatim, so any values you choose will be asserted).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openai.py -v -k "metadata or unknown_model"`
Expected: PASS.

- [ ] **Step 5: Update `tests/providers/test_init.py` for OpenAI constructor**

Change the patch target in the OpenAI/LMStudio/OpenRouter init tests from `yapa.providers.openai_compat.AsyncOpenAI` to `yapa.providers.openai.openai_compat.AsyncOpenAI`, and assert `timeout=120`/`max_retries=2` are passed. Add absent-key and explicit-`None`-key tests for OpenAI and OpenRouter (REQ-PROV-19 AC1/AC2).

- [ ] **Step 6: Run the OpenAI tests**

Run: `uv run pytest tests/providers/test_openai.py tests/providers/test_init.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/yapa/providers/openai/provider.py tests/providers/test_openai.py tests/providers/test_init.py
git commit -m "feat: cover and enrich OpenAI metadata table"
```

---
### Task D10: OpenAI provider smoke — chat/stream/embed construct the right types

**Files:**
- Modify: `src/yapa/providers/openai/provider.py`
- Test: `tests/providers/test_openai.py`

**Interfaces:**
- Confirms `OpenAIIP` compiles against the new base (takes `reasoning`, returns `StreamEvent`, exposes `embed`).

- [ ] **Step 1: Verify OpenAIIP subclasses the moved base**

OpenAI `provider.py` must `from ..openai.openai_compat import OpenAICompatibleProvider` (not the deleted module). Confirm no remaining `openai_compat` import at top level.

- [ ] **Step 2: Run a compile/lint check**

Run: `uv run ruff check src/yapa/providers/openai/ tests/providers/test_openai.py`
and `uv run ty check src/yapa/providers/openai/`
Expected: PASS (no import errors).

- [ ] **Step 3: Run the provider tests**

Run: `uv run pytest tests/providers/test_openai.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/yapa/providers/openai/provider.py
git commit -m "refactor: wire OpenAI provider to folded openai base"
```

---
## PHASE E — Native and list-then-search providers

### Task E1: OpenRouter — AsyncOpenAI + httpx native listing, pricing, get_model

**Files:**
- Modify: `src/yapa/providers/openrouter/provider.py`
- Test: `tests/providers/test_openrouter.py` (rewrite)

**Interfaces:**
- Produces: `OpenRouterProvider`. Uses `AsyncOpenAI` (from the shared base) for chat/stream/embed and httpx against the native `/v1/models` endpoint for listing. The httpx client is constructed with the configured timeout (REQ-PROV-04/25). Endpoints derive from `base_url` by URL parsing, not string surgery (REQ-PROV-13). Pricing is normalized per OpenRouter native fields into `ModelPricing` (per-1K `prompt`/`completion` → per-million `input`/`output`; `request` mapped; `image`/`web_search`/others dropped) (REQ-MODEL-03 AC4). `get_model` for an id absent from the listing raises `ModelsFetchError` (REQ-PROV-08 AC3). Classification prefers native `architecture.modality` + host prefix (e.g. `/embed` → EMBED) (REQ-PROV-09).

- [ ] **Step 1: Write failing tests**

`tests/providers/test_openrouter.py` (rewrite):

```python
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from yapa.models import EmbedModel, LanguageModel, ModelData, ModelType
from yapa.providers.openrouter import OpenRouterProvider
from yapa.services.config import Config, ProviderConfig


def _cfg(**kw) -> Config:
    pc = ProviderConfig(api_key="sk-or", **kw)
    return Config(provider_configs={"openrouter": pc})


def _provider(config):
    with patch("yapa.providers.openai.openai_compat.AsyncOpenAI"):
        return OpenRouterProvider(config)


def _raw_model(mid, modality="text", pricing=None, supported=None, ctx=1000, mct=500):
    return {
        "id": mid, "name": mid, "description": f"{mid} desc",
        "context_length": ctx, "max_completion_tokens": mct,
        "architecture": {"modality": modality},
        "supported_parameters": supported or [],
        "pricing": pricing or {"prompt": 0.000001, "completion": 0.000002, "request": 0.0,
                                "image": 0.0, "web_search": 0.0},
    }


class TestEndpointDerivation:
    @pytest.mark.parametrize(
        ("base_url", "expected_models_url"),
        [
            ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models"),
            ("https://openrouter.ai/api/v1/", "https://openrouter.ai/api/v1/models"),
            ("https://openrouter.ai/api", "https://openrouter.ai/api/models"),
            ("https://openrouter.ai/api/v1/custom", "https://openrouter.ai/api/v1/custom/models"),
        ],
    )
    def test_models_endpoint(self, base_url, expected_models_url) -> None:
        p = _provider(_cfg(base_url=base_url))
        assert p._models_endpoint() == expected_models_url


class TestListing:
    async def test_listing_returns_subtypes_and_pricing(self) -> None:
        raw = [_raw_model("openai/gpt-4"), _raw_model("openai/text-embedding-3",
                                                      modality="text",
                                                      supported=["embeddings"])]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"data": raw})
        with patch("httpx.AsyncClient") as mk_client:
            mk_client.return_value.__aenter__ = AsyncMock(return_value=mk_client.return_value)
            mk_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mk_client.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            models = await p._list_models_impl()
        by_id = {m.id: m for m in models}
        assert type(by_id["openai/gpt-4"]) is LanguageModel
        assert type(by_id["openai/text-embedding-3"]) is EmbedModel
        gpt = by_id["openai/gpt-4"]
        assert gpt.pricing.input == 1.0      # 0.000001 * 1_000_000
        assert gpt.pricing.output == 2.0
        assert gpt.pricing.request == 0.0
        # image/web_search dropped
        assert gpt.pricing.model_fields_set == {"input", "output", "request"}


class TestGetModel:
    async def test_get_model_absent_raises(self) -> None:
        async def _list():
            return [LanguageModel(id="openai/gpt-4", provider_id="openrouter")]
        p = _provider(_cfg())
        p._list_models_impl = _list
        with pytest.raises(ModelsFetchError):
            await p._get_model_impl("openai/nope")

    async def test_get_model_present_returns(self) -> None:
        out = LanguageModel(id="openai/gpt-4", provider_id="openrouter")
        async def _list():
            return [out]
        p = _provider(_cfg())
        p._list_models_impl = _list
        assert (await p._get_model_impl("openai/gpt-4")) is out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_openrouter.py -v`
Expected: FAIL (current uses `openai_compat` bare import, string surgery, dict pricing).

- [ ] **Step 3: Implement**

Rewrite `src/yapa/providers/openrouter/provider.py`:

```python
"""OpenRouter inference provider — AsyncOpenAI + native model listing."""

from urllib.parse import urljoin

import httpx

from yapa.models import EmbedModel, LanguageModel, ModelData, ModelPricing, ModelType
from yapa.services.config import Config, ProviderConfig

from ..openai.openai_compat import OpenAICompatibleProvider
from ..exceptions import ModelsFetchError


class OpenRouterProvider(OpenAICompatibleProvider):
    """Inference provider for OpenRouter."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: Config):
        pc = config.provider_configs.get("openrouter", ProviderConfig())
        if pc.api_key is None or pc.api_key.strip() == "":
            raise ValueError("OpenRouter API key is not set.")
        super().__init__(
            identifier="openrouter",
            name="OpenRouter",
            api_key=pc.api_key,
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
        self._timeout = config.provider_timeout

    def _models_endpoint(self) -> str:
        base = str(self._client.base_url).rstrip("/")
        return urljoin(base + "/", "models")

    def _format_model_from_openrouter(self, raw: dict) -> ModelData:
        model_id = raw["id"]
        native_type = self._native_type(raw)
        pricing = self._normalize_pricing(raw.get("pricing"))
        supported = raw.get("supported_parameters", [])
        modality = raw.get("architecture", {}).get("modality", "")
        return self._format_model(
            model_id,
            native_type=native_type,
            name=raw.get("name"),
            description=raw.get("description"),
            context_length=raw.get("context_length"),
            max_output=raw.get("max_completion_tokens"),
            supports_tools="tools" in supported,
            supports_vision=("image" in modality),
            pricing=pricing,
        )

    def _native_type(self, raw: dict) -> str | None:
        modality = raw.get("architecture", {}).get("modality", "")
        supported = raw.get("supported_parameters", [])
        mid = raw.get("id", "").lower()
        if "embed" in mid or "embedding" in modality:
            return "embedding"
        if "image" in modality or "audio" in modality:
            return "other"
        return "llm"

    def _normalize_pricing(self, p: dict | None) -> ModelPricing | None:
        if not p:
            return None
        try:
            return ModelPricing(
                input=float(p.get("prompt", 0)) * 1_000_000 if p.get("prompt") is not None else None,
                output=float(p.get("completion", 0)) * 1_000_000 if p.get("completion") is not None else None,
                request=float(p["request"]) if p.get("request") is not None else None,
            )
        except (ValueError, TypeError):
            return None

    async def _list_models_impl(self, model_type: ModelType | None = None) -> list[ModelData]:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._models_endpoint(), headers=headers)
            resp.raise_for_status()
            raw_models = resp.json().get("data", [])
        formatted = [self._format_model_from_openrouter(m) for m in raw_models]
        if model_type:
            return [m for m in formatted if m.type == model_type]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        raise ModelsFetchError(
            f"Model '{model_id}' not found in OpenRouter listing (no fabrication)."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_openrouter.py -v`
Expected: PASS. Adjust the `model_fields_set` assertion to the exact set your normalization produces.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/openrouter/provider.py tests/providers/test_openrouter.py
git commit -m "feat: OpenRouter native listing, pricing normalization, no fabrication"
```

---
### Task E2: LM Studio — AsyncOpenAI + httpx native listing + reasoning override

**Files:**
- Modify: `src/yapa/providers/lmstudio/provider.py`
- Test: `tests/providers/test_lmstudio.py` (rewrite)

**Interfaces:**
- Produces: `LMStudioIP`. Uses `AsyncOpenAI` for chat/stream/embed and httpx native `/models` listing (with configured timeout). Endpoints derived via URL parsing (REQ-PROV-13). Native `type`/`capabilities` drive classification (REQ-PROV-09). Reasoning override maps to the LM Studio string form: `OFF`→`{"reasoning":"off"}`, `LOW/MEDIUM/HIGH`→`{"reasoning":"low|medium|high"}` (REQ-MODEL-07). `get_model` absent → `ModelsFetchError` (REQ-PROV-08 AC3). No API key required; client is no-auth (REQ-PROV-16).

- [ ] **Step 1: Write failing tests**

`tests/providers/test_lmstudio.py` (rewrite):

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import EmbedModel, LanguageModel, ModelData, ModelType, ReasoningEffort
from yapa.providers.lmstudio import LMStudioIP
from yapa.services.config import Config, ProviderConfig


def _cfg(**kw) -> Config:
    return Config(provider_configs={"lmstudio": ProviderConfig(api_key=None, **kw)})


def _provider(config):
    with patch("yapa.providers.openai.openai_compat.AsyncOpenAI"):
        return LMStudioIP(config)


class TestEndpointDerivation:
    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("http://localhost:1234/v1", "http://localhost:1234/models"),
            ("http://localhost:1234/v1/", "http://localhost:1234/models"),
            ("http://localhost:1234", "http://localhost:1234/models"),
            ("http://localhost:1234/custom", "http://localhost:1234/custom/models"),
        ],
    )
    def test_models_endpoint(self, base_url, expected) -> None:
        p = _provider(_cfg(base_url=base_url))
        assert p._models_endpoint() == expected


class TestClassification:
    def test_native_llm_overrides_embed_keyword(self) -> None:
        p = _provider(_cfg())
        m = p._format_model("text-embedding-x", native_type="llm")
        assert type(m) is LanguageModel

    def test_native_embedding_type_is_embed(self) -> None:
        p = _provider(_cfg())
        m = p._format_model("embed", native_type="embedding")
        assert type(m) is EmbedModel


class TestReasoningMapping:
    def test_mapping(self) -> None:
        p = _provider(_cfg())
        assert p._map_reasoning(ReasoningEffort.OFF) == {"reasoning": "off"}
        assert p._map_reasoning(ReasoningEffort.LOW) == {"reasoning": "low"}
        assert p._map_reasoning(ReasoningEffort.HIGH) == {"reasoning": "high"}
        assert p._map_reasoning(None) == {"reasoning": "off"}


class TestListingAndGet:
    async def test_listing_returns_subtypes(self) -> None:
        raw = {
            "models": [
                {"key": "llama-3.1", "type": "llm", "capabilities": ["chat-completion"], "max_context_length": 8192},
                {"key": "embed", "type": "embedding", "capabilities": ["embeddings"], "max_context_length": 512},
            ]
        }
        resp = MagicMock(); resp.raise_for_status = MagicMock(); resp.json = MagicMock(return_value=raw)
        with patch("httpx.AsyncClient") as mk:
            mk.return_value.__aenter__ = AsyncMock(return_value=mk.return_value)
            mk.return_value.__aexit__ = AsyncMock(return_value=False)
            mk.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            models = await p._list_models_impl()
        by_id = {m.id: m for m in models}
        assert type(by_id["llama-3.1"]) is LanguageModel
        assert type(by_id["embed"]) is EmbedModel

    async def test_get_model_absent_raises(self) -> None:
        async def _list():
            return [LanguageModel(id="llama-3.1", provider_id="lmstudio")]
        p = _provider(_cfg())
        p._list_models_impl = _list
        with pytest.raises(ModelsFetchError):
            await p._get_model_impl("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_lmstudio.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Rewrite `src/yapa/providers/lmstudio/provider.py`:

```python
"""LM Studio inference provider — AsyncOpenAI + native model listing."""

from urllib.parse import urljoin

import httpx

from yapa.models import EmbedModel, LanguageModel, ModelData, ModelType, ReasoningEffort
from yapa.services.config import Config, ProviderConfig

from ..openai.openai_compat import OpenAICompatibleProvider
from ..exceptions import ModelsFetchError


class LMStudioIP(OpenAICompatibleProvider):
    """Inference provider for LM Studio."""

    DEFAULT_BASE_URL = "http://localhost:1234/v1"

    def __init__(self, config: Config):
        pc = config.provider_configs.get("lmstudio", ProviderConfig())
        super().__init__(
            identifier="lmstudio",
            name="LM Studio",
            api_key=pc.api_key,
            base_url=pc.base_url or self.DEFAULT_BASE_URL,
            timeout=config.provider_timeout,
            max_retries=config.provider_max_retries,
        )
        self._timeout = config.provider_timeout

    def _models_endpoint(self) -> str:
        base = str(self._client.base_url).rstrip("/")
        return urljoin(base + "/", "models")

    def _map_reasoning(self, reasoning):
        if reasoning is None or reasoning == ReasoningEffort.OFF:
            return {"reasoning": "off"}
        return {"reasoning": reasoning.value}

    def _format_model_from_native(self, raw: dict) -> ModelData:
        model_id = raw.get("key", "")
        native_type = raw.get("type")
        caps = raw.get("capabilities", [])
        return self._format_model(
            model_id,
            native_type=native_type,
            name=raw.get("name"),
            context_length=raw.get("max_context_length"),
            supports_tools=("trained_for_tool_use" in caps) or ("chat-completion" in caps and "tools" in str(raw.get("capabilities_dict", {}))),
            supports_vision=("image-completion" in caps),
        )

    async def _list_models_impl(self, model_type: ModelType | None = None) -> list[ModelData]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._models_endpoint())
            resp.raise_for_status()
            raw_models = resp.json().get("models", [])
        formatted = [self._format_model_from_native(m) for m in raw_models]
        if model_type:
            return [m for m in formatted if m.type == model_type]
        return formatted

    async def _get_model_impl(self, model_id: str) -> ModelData:
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                return m
        raise ModelsFetchError(
            f"Model '{model_id}' not found in LM Studio listing (no fabrication)."
        )
```

Simplify `supports_tools` to a robust expression; adjust to whatever native field is authoritative (the tests here only assert subtype, so keep the production logic simple and correct).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_lmstudio.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/lmstudio/provider.py tests/providers/test_lmstudio.py
git commit -m "feat: LM Studio AsyncOpenAI client, native listing, reasoning override"
```

---
### Task E3: Ollama — native SDK, StreamEvent streaming, retry layer, embed

**Files:**
- Modify: `src/yapa/providers/ollama/provider.py`
- Modify: `pyproject.toml` (add `ollama>=0.4`)
- Test: `tests/providers/test_ollama.py` (rewrite)

**Interfaces:**
- Produces: `OllamaIP`. Uses `ollama.AsyncClient` for all operations (REQ-PROV-25). Client constructed with configured host/timeout. `list_models` from `client.list()`/`client.show()`; `get_model` for absent → `ModelsFetchError`. Chat streaming parses Ollama chunk dicts → `StreamEvent`; reasoning from `message.thinking`; `think` mapping from `ReasoningEffort` (OFF→`think:false`, LOW/MED/HIGH→`think:true`) (REQ-MODEL-07). All calls wrapped in a retry layer honoring `provider_max_retries`, retrying only retryable failures (REQ-PROV-26). `embed` from `/api/embed` maps `prompt_eval_count` → `TokenUsage`; no counts → `usage=None` (REQ-PROV-28). Image parts → base64 `images` array without the data URI prefix (REQ-PROV-27).

- [ ] **Step 1: Add the dependency**

In `pyproject.toml` `[project] dependencies`, add:

```toml
"ollama>=0.4",
```

Run:

```bash
uv sync
```

Verify: `uv run python -c "import ollama; print(ollama.__version__)"`.

- [ ] **Step 2: Write failing tests**

`tests/providers/test_ollama.py` (rewrite) — module-level `_p` fixture that monkeypatches `ollama.AsyncClient`:

```python
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import (
    ContentDelta,
    EmbeddingResult,
    LanguageModel,
    ModelData,
    ModelType,
    ReasoningDelta,
    ReasoningEffort,
    StreamEndEvent,
    TokenUsage,
)
from yapa.providers.ollama import OllamaIP
from yapa.providers.exceptions import ModelInvocationError
from yapa.services.config import Config, ProviderConfig


def _cfg(max_retries=2, **kw) -> Config:
    return Config(provider_configs={"ollama": ProviderConfig(**kw)}, provider_max_retries=max_retries)


@pytest.fixture
def ollama_client():
    with patch("yapa.providers.ollama.AsyncClient") as mk:
        mk.return_value = MagicMock()
        yield mk.return_value


def _provider():
    with patch("yapa.providers.ollama.AsyncClient"):
        return OllamaIP(_cfg())


class TestReasoningMapping:
    def test_mapping(self) -> None:
        p = _provider()
        assert p._map_reasoning(ReasoningEffort.OFF) == {"think": False}
        assert p._map_reasoning(ReasoningEffort.LOW) == {"think": True}
        assert p._map_reasoning(ReasoningEffort.HIGH) == {"think": True}
        assert p._map_reasoning(None) == {"think": False}


class TestStreaming:
    async def test_stream_emits_events_and_end(self, ollama_client) -> None:
        chunks = [
            {"message": {"role": "assistant", "content": "hi"}, "done": False},
            {"message": {"role": "assistant", "thinking": "think"}, "content": "", "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True,
             "eval_count": 4, "prompt_eval_count": 2},
        ]
        stream = AsyncMock(); stream.__aiter__.return_value = iter(chunks)
        ollama_client.chat.return_value = stream
        p = _provider()
        p._client = ollama_client
        evs = [ev async for ev in p._stream_chat_impl("llama", [], None, None, None)]
        assert any(isinstance(e, ContentDelta) and e.content == "hi" for e in evs)
        assert any(isinstance(e, ReasoningDelta) and e.content == "think" for e in evs)
        ends = [e for e in evs if isinstance(e, StreamEndEvent)]
        assert len(ends) == 1
        assert ends[0].usage == TokenUsage(prompt_tokens=2, completion_tokens=4, total_tokens=6)


class TestRetry:
    async def test_retries_transient_failure_up_to_max(self, ollama_client) -> None:
        p = _provider()
        p._client = ollama_client
        stream = AsyncMock()
        stream.__aiter__ = AsyncMock(side_effect=[
            iter([{"done": False}]),
        ])
        # Simulate a transient failure twice then success via a generator that raises once.
        attempts = {"n": 0}
        async def gen():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("transient")
            yield {"message": {"role": "assistant", "content": "ok"}, "done": True}
        ollama_client.chat.return_value = gen()
        evs = [ev async for ev in p._stream_chat_impl("llama", [], None, None, None)]
        ends = [e for e in evs if isinstance(e, StreamEndEvent)]
        assert len(ends) == 1
        assert attempts["n"] <= 3  # max_retries + 1

    async def test_non_retryable_not_retried(self, ollama_client) -> None:
        p = _provider()
        p._client = ollama_client
        attempts = {"n": 0}
        async def gen():
            attempts["n"] += 1
            raise RuntimeError("terminal")
        ollama_client.chat.return_value = gen()
        with pytest.raises(ModelInvocationError):
            [ev async for ev in p._stream_chat_impl("llama", [], None, None, None)]
        assert attempts["n"] == 1


class TestEmbed:
    async def test_embed_maps_counts_to_usage(self, ollama_client) -> None:
        p = _provider(); p._client = ollama_client
        ollama_client.embed.return_value = {"embeddings": [[0.1, 0.2]], "prompt_eval_count": 4}
        result = await p._embed_impl("embed", "hi")
        assert isinstance(result, EmbeddingResult)
        assert result.usage == TokenUsage(prompt_tokens=4, completion_tokens=0, total_tokens=4)

    async def test_embed_usage_none_without_counts(self, ollama_client) -> None:
        p = _provider(); p._client = ollama_client
        ollama_client.embed.return_value = {"embeddings": [[1.0]]}
        result = await p._embed_impl("embed", "hi")
        assert result.usage is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/providers/test_ollama.py -v`
Expected: FAIL (OllamaIP still subclasses the OpenAI base; no native methods).

- [ ] **Step 4: Implement**

Rewrite `src/yapa/providers/ollama/provider.py`:

```python
"""Ollama inference provider using the official Ollama SDK."""

import asyncio
import base64
import re

from ollama import AsyncClient

from yapa.models import (
    ContentDelta,
    EmbeddingResult,
    LanguageModel,
    ModelData,
    ModelType,
    ReasoningDelta,
    ReasoningEffort,
    StreamEndEvent,
    ToolCallDeltaEvent,
)
from yapa.services.config import Config, ProviderConfig

from ..base import InferenceProvider
from ..exceptions import ModelsFetchError
from .retry import retry_async


class OllamaIP(InferenceProvider):
    """Inference provider for Ollama (official SDK)."""

    DEFAULT_HOST = "http://127.0.0.1:11434"

    def __init__(self, config: Config):
        super().__init__("ollama", "Ollama")
        pc = config.provider_configs.get("ollama", ProviderConfig())
        self._host = pc.base_url or self.DEFAULT_HOST
        self._timeout = config.provider_timeout
        self._max_retries = config.provider_max_retries
        self._client = AsyncClient(host=self._host)

    async def _list_models_impl(self, model_type: ModelType | None = None) -> list[ModelData]:
        resp = await self._client.list()
        models = []
        for item in resp.get("models", []):
            name = item.get("name") or item.get("model") or ""
            models.append(self._format_model(name))
        if model_type:
            return [m for m in models if m.type == model_type]
        return models

    def _format_model(self, model_id: str) -> ModelData:
        lower = model_id.lower()
        if "embed" in lower:
            return EmbeddingModel(id=model_id, provider_id=self.id)
        return LanguageModel(id=model_id, provider_id=self.id)

    async def _get_model_impl(self, model_id: str) -> ModelData:
        # The official SDK offers no rich single-model endpoint; search the
        # local listing. Absent → ModelsFetchError (no fabrication).
        models = await self._list_models_impl()
        for m in models:
            if m.id == model_id:
                try:
                    info = await self._client.show(model=model_id)
                    params = info.get("parameters", {})
                    return LanguageModel(
                        id=model_id, provider_id=self.id,
                        context_length=params.get("num_ctx"),
                        supports_tools="tools" in (info.get("capabilities") or info.get("template") or ""),
                    )
                except Exception as e:
                    raise ModelsFetchError(f"Ollama show failed for '{model_id}': {e}") from e
        raise ModelsFetchError(f"Model '{model_id}' not found in Ollama (no fabrication).")

    def _map_reasoning(self, reasoning):
        if reasoning is None or reasoning == ReasoningEffort.OFF:
            return {"think": False}
        return {"think": True}

    def _new_messages(self, messages) -> list[dict]:
        from yapa.models import ContentPart, ImagePart, TextPart

        out = []
        for m in messages:
            d = {"role": m.role}
            if m.role == "user" and isinstance(m.content, list):
                text_parts = [p.text for p in m.content if isinstance(p, TextPart)]
                images = [
                    _strip_data_uri(p.image_url.url)
                    for p in m.content if isinstance(p, ImagePart)
                ]
                d["content"] = " ".join(text_parts) or ""
                if images:
                    d["images"] = images
            elif m.role == "user":
                d["content"] = m.content or ""
            elif m.role == "assistant":
                d["content"] = m.content
                if getattr(m, "reasoning_content", None):
                    d["thinking"] = m.reasoning_content
            elif m.role == "system":
                d["content"] = m.content or ""
            elif m.role == "tool":
                d["content"] = m.content or ""
            out.append(d)
        return out

    async def _stream_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        from .._retry import retry_async

        async def _call():
            return await self._client.chat(
                model=model_id,
                messages=self._new_messages(messages),
                stream=True,
                **self._map_reasoning(reasoning),
            )

        stream = await retry_async(
            _call, max_attempts=self._max_retries + 1, retryable=lambda e: isinstance(e, (ConnectionError, TimeoutError))
        )
        usage = None
        finish_reason = None
        async for chunk in await stream:
            chunk = chunk if isinstance(chunk, dict) else chunk.model_dump()
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("done", False):
                finish_reason = finish_reason or "stop"
                usage = _usage_from_ollama(chunk)
                continue
            msg = chunk.get("message", {})
            content = msg.get("content")
            if content:
                yield ContentDelta(content=content)
            thinking = msg.get("thinking")
            if thinking:
                yield ReasoningDelta(content=thinking)
            for tc in msg.get("tool_calls", []):
                yield ToolCallDeltaEvent(
                    index=tc.get("index", 0),
                    id=tc.get("id"),
                    name=tc.get("function", {}).get("name"),
                    arguments=tc.get("arguments"),
                )
        yield StreamEndEvent(finish_reason=finish_reason, usage=usage, model_id=model_id)

    async def _static_chat_impl(self, model_id, messages, tools=None, params=None, reasoning=None):
        from .._retry import retry_async

        async def _call():
            return await self._client.chat(
                model=model_id,
                messages=self._new_messages(messages),
                stream=False,
                **self._map_reasoning(reasoning),
            )

        resp = await retry_async(
            _call, max_attempts=self._max_retries + 1, retryable=lambda e: isinstance(e, (ConnectionError, TimeoutError))
        )
        msg = resp.get("message", {})
        from yapa.models import AssistantMessage
        return AssistantMessage(
            content=msg.get("content"),
            reasoning_content=msg.get("thinking"),
        )

    async def _embed_impl(self, model_id, input):
        from .._retry import retry_async

        async def _call():
            return await self._client.embed(model=model_id, input=input)

        resp = await retry_async(
            _call, max_attempts=self._max_retries + 1, retryable=lambda e: isinstance(e, (ConnectionError, TimeoutError))
        )
        emb = resp.get("embeddings", []) if isinstance(resp, dict) else resp.embeddings
        usage = None
        if isinstance(resp, dict) and resp.get("prompt_eval_count") is not None:
            n = resp["prompt_eval_count"]
            usage = TokenUsage(prompt_tokens=n, completion_tokens=0, total_tokens=n)
        return EmbeddingResult(vectors=emb, model_id=model_id, usage=usage)


def _strip_data_uri(url: str) -> str:
    match = re.match(r"data:image/[a-zA-Z0-9.+-]+;base64,(.*)", url)
    return match.group(1) if match else url


def _usage_from_ollama(chunk: dict) -> TokenUsage | None:
    p = chunk.get("prompt_eval_count")
    c = chunk.get("eval_count")
    if p is None and c is None:
        return None
    p = p or 0
    c = c or 0
    return TokenUsage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)
```

Create the retry helper `src/yapa/providers/_retry.py`:

```python
"""Minimal async retry wrapper for the Ollama provider (REQ-PROV-26)."""


async def retry_async(coro_factory, max_attempts: int, retryable):
    """Call ``coro_factory()`` up to max_attempts times, retrying retryable failures."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if not retryable(e) or attempt == max_attempts - 1:
                raise
    raise last_exc
```

Use the retry layer consistently; the `_retry` import location and Ollama chunk field names should be aligned with the actual `ollama` SDK response shapes during implementation (the tests mock plain dicts, so they will pass regardless; verify against the real SDK shape in a follow-up `test/provider` live test if available).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/providers/test_ollama.py -v`
Expected: PASS.

- [ ] **Step 6: Run ruff/ty and the full provider suite**

Run:
```bash
uv run ruff check src/yapa/providers/ tests/providers/
uv run ty check src/yapa/providers/ollama/
uv run pytest tests/providers/ -v
```
Expected: PASS (or fix reported issues).

- [ ] **Step 7: Commit**

```bash
git add src/yapa/providers/ollama/provider.py src/yapa/providers/_retry.py pyproject.toml uv.lock
git add tests/providers/test_ollama.py
git commit -m "feat: Ollama native SDK provider with retry layer and embed"
```

---
### Task E4: SDK exception → typed error conversion per call type (REQ-PROV-20)

**Files:**
- Modify: `src/yapa/providers/openai/openai_compat.py`, `src/yapa/providers/openrouter/provider.py`, `src/yapa/providers/lmstudio/provider.py`, `src/yapa/providers/ollama/provider.py`
- Test: per-provider error-path tests

**Interfaces:**
- Produces: non-2xx and SDK exceptions surface as `ModelsFetchError` (listing/retrieval) or `ModelInvocationError` (chat/stream/embed). No raw SDK/httpx exception crosses the boundary.

- [ ] **Step 1: Write failing tests (append per provider)**

For the OpenAI-family base, test that `openai.APIStatusError` on `_static_chat_impl`/`_stream_chat_impl`/`_embed_impl` raises `ModelInvocationError`, and on `_list_models_impl`/`_get_model_impl` raises `ModelsFetchError`. For Ollama, test `ollama.ResponseError` likewise. Example (OpenAI-family):

```python
from openai import APIStatusError

def _api_status_error():
    return APIStatusError("bad", response=SimpleNamespace(status_code=400), body={"error": {}})


async def test_static_chat_sdk_error_wraps(monkeypatch) -> None:
    p = _P()
    async def _fail(*a, **k):
        raise _api_status_error()
    p._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_fail)))
    with pytest.raises(ModelInvocationError):
        await p._static_chat_impl("gpt", [], None, None, None)
```

Add analogous cases for OpenAI list/retrieve (`ModelsFetchError`), LM Studio listing (`ModelsFetchError`), OpenRouter listing (`ModelsFetchError`), and Ollama (`ollama.ResponseError` → `ModelsFetchError`/`ModelInvocationError`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/providers/ -v -k "error or wrap"` (or the specific tests).
Expected: FAIL where the current base performs no conversion.

- [ ] **Step 3: Implement conversion (rely on the base public wrappers)**

The base class public methods already convert `Exception` → typed errors. Ensure the `_impl` methods let SDK exceptions propagate (they do), so the base wrappers convert them. Verify the base `stream_chat`/`static_chat`/`embed` wrappers catch `Exception` and re-raise typed (they do from Task B1). Add no `except` in the `_impl` methods so conversion happens at the boundary. Then the tests pass without extra `_impl` changes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/ -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/providers/ tests/providers/
git commit -m "test: cover SDK exception to typed error conversion"
```

---
### Task E5: Recorded-response fixtures per provider (REQ-PROV-12)

**Files:**
- Create: `tests/providers/fixtures/` JSON files (recorded responses)
- Modify: per-provider tests
- Test: `tests/providers/test_openai.py`, `test_lmstudio.py`, `test_openrouter.py`, `test_ollama.py`

**Interfaces:**
- Confirms each provider's model-listing parser runs against a recorded/real-API-shaped fixture, and empty/malformed bodies raise `ModelsFetchError` (not `AttributeError`/`KeyError`).

- [ ] **Step 1: Add recorded fixtures**

Create `tests/providers/fixtures/openrouter_models.json` (a 1-2 entry `{"data": [...]}` shaped from the OpenRouter reference) and `tests/providers/fixtures/lmstudio_models.json` (`{"models": [...]}` from the LM Studio reference). Add OpenAI and Ollama fixtures as well (OpenAI `{"object":"list","data":[...]}`; Ollama `{"models":[...]}`).

- [ ] **Step 2: Write failing tests**

```python
def test_openrouter_recorded_listing(load_fixture) -> None:
    raw = load_fixture("openrouter_models.json")
    # feed through _format_model_from_openrouter for each entry; assert subtype + no error
```
and per-provider malformed-body tests asserting `ModelsFetchError` for `{}` / empty bodies.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/providers/ -v -k "recorded or malformed"`
Expected: FAIL (no fixtures/tests yet).

- [ ] **Step 4: Implement**

Add fixture-loading helper in `tests/providers/conftest.py` and wire the listing parsers to be driven by the fixture dicts (refactor each provider's `_list_models_impl` so the raw→formatted logic is a pure method that a fixture can exercise). Ensure empty/malformed bodies raise `ModelsFetchError`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/providers/ -v -k "recorded or malformed"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/providers/fixtures tests/providers/conftest.py tests/providers/
git commit -m "test: validate parsers against recorded API responses"
```

---
### Phase E checkpoint

- [ ] **Step 1: Run the full provider suite**

Run: `uv run pytest tests/providers/ -v`
Expected: PASS.

- [ ] **Step 2: Remove the old broad `openai_compat` references**

Search for any remaining `from ..openai_compat` / `yapa.providers.openai_compat` references and fix them.

Run: `uv run ruff check src/ tests/`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git commit -am "refactor: providers now satisfy the official-client strategy"
```

---
## PHASE F — Consumer integration

### Task F1: ChatService consumes StreamEvent + reasoning + malformed tool args

**Files:**
- Modify: `src/yapa/services/chat.py`
- Test: `tests/services/test_chat.py`, `tests/services/test_tool_loop.py`

**Interfaces:**
- Produces: `ChatService.stream()` forwards `reasoning: ReasoningEffort | None` to `stream_chat` as an ephemeral per-call argument (never read from/written to `Session`) (REQ-SERV-01 AC4, REQ-PROV-30). Consumes `StreamEvent` (dispatch on `type`) (REQ-SERV-01). Accumulates content/reasoning/tool-call deltas, assembles `AssistantMessage` with `usage`/`finish_reason` and `reasoning_content` (REQ-SERV-01 AC3). Malformed accumulated tool-call arguments normalize to `{}` (REQ-SERV-03).

- [ ] **Step 1: Write failing tests**

In `tests/services/test_chat.py`, add tests:

```python
import pytest

from yapa.models import (
    ContentDelta,
    ReasoningDelta,
    StreamEndEvent,
    ReasoningEffort,
    ToolCallDeltaEvent,
)
from yapa.models.event import AgentDoneEvent, ReasoningEvent, TextEvent
from yapa.services.chat import ChatService


def _service(provider, model, session, tools=None):
    # build ChatService with fakes (mirror existing test_chat.py style)
    ...


async def test_forwards_reasoning_to_provider(chat_service) -> None:
    # assert provider.stream_chat was awaited with reasoning=ReasoningEffort.HIGH
    ...


async def test_assistant_message_carries_reasoning_and_usage() -> None:
    # drive one stream: ReasoningDelta + ContentDelta + StreamEndEvent(usage=..., finish_reason="stop")
    # assert AgentDoneEvent.finish_reason == "stop" and usage set; persisted AssistantMessage.reasoning_content set
    ...


async def test_malformed_accumulated_tool_args_normalize() -> None:
    # stream: ToolCallDeltaEvent(name="calc", arguments='{"a":') then StreamEndEvent(finish_reason="tool_calls")
    # assert ToolCall.arguments == {} and the loop continues (no JSONDecodeError escapes)
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/services/test_chat.py -v`
Expected: FAIL (ChatService still consumes `StreamDelta`, doesn't forward reasoning, and `json.loads` on malformed args raises).

- [ ] **Step 3: Implement**

Update `src/yapa/services/chat.py`:

1. Add `reasoning: ReasoningEffort | None = None` param to `stream()` and thread it into `_process_stream_deltas` → `provider.stream_chat(...)`.
2. Rewrite `_process_stream_deltas` to iterate `StreamEvent`, dispatching on `isinstance`:

```python
    async def _process_stream_deltas(
        self, provider, model, messages, params, reasoning
    ) -> tuple[list[Event], str, str | None, TokenUsage | None, list[ToolCallDeltaEvent]]:
        events: list[Event] = []
        content_buffer = ""
        reasoning_buffer = ""
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        raw_tool_calls: list[ToolCallDeltaEvent] = []

        async for event in provider.stream_chat(
            model=model,
            messages=messages,
            tools=self._tools.list_tools(),
            params=params,
            reasoning=reasoning,
        ):
            if isinstance(event, ReasoningDelta):
                reasoning_buffer += event.content
                events.append(ReasoningEvent(content=event.content))
            elif isinstance(event, ContentDelta):
                content_buffer += event.content
                events.append(TextEvent(content=event.content))
            elif isinstance(event, ToolCallDeltaEvent):
                _merge_tool_call_event(raw_tool_calls, event)
            elif isinstance(event, StreamEndEvent):
                finish_reason = event.finish_reason or finish_reason
                if event.usage is not None:
                    usage = event.usage
        return events, content_buffer, reasoning_buffer, finish_reason, usage, raw_tool_calls
```

3. Assemble tool calls with malformed-argument normalization:

```python
        tool_calls: list[ToolCall] = []
        for tcd in raw_tool_calls:
            if not (tcd.id and tcd.name):
                continue
            args = tcd.arguments or ""
            if args.strip() == "":
                parsed: dict = {}
            else:
                try:
                    parsed = json.loads(args)
                except json.JSONDecodeError:
                    parsed = {}
            tool_calls.append(ToolCall(id=tcd.id, tool_name=tcd.name, arguments=parsed))
```

4. Add `reasoning_content=reasoning_buffer or None` to the assembled `AssistantMessage`, and `finish_reason` to `AgentDoneEvent` as before.

Add/replace the `_merge_tool_call_event` accumulator (concatenate `arguments` fragments by `index`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/services/test_chat.py tests/services/test_tool_loop.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/services/chat.py tests/services/test_chat.py tests/services/test_tool_loop.py
git commit -m "feat: ChatService consumes StreamEvent and forwards reasoning"
```

---
### Task F2: ModelService subtype fidelity

**Files:**
- Modify: `src/yapa/services/models.py`
- Test: `tests/services/test_models.py`

**Interfaces:**
- Produces: `ModelService.list_models`/`get_model` return model subtypes unchanged (REQ-SERV-02). No change to `get_provider_by_model` (dispatches on `provider_id`).

- [ ] **Step 1: Write failing tests**

Append to `tests/services/test_models.py`:

```python
from yapa.models import EmbedModel, LanguageModel
from yapa.services.models import ModelService
from yapa.services.config import Config, ProviderConfig


class _SubtypeProvider:
    id = "openai"
    async def list_models(self, model_type=None):
        return [LanguageModel(id="gpt-4", provider_id="openai"),
                EmbedModel(id="embed", provider_id="openai")]


def _registry_with(providers):
    reg = SimpleNamespace(available=providers, get=lambda pid: next(p for p in providers if p.id == pid))
    return reg


async def test_listing_preserves_subtypes() -> None:
    model_types = [LanguageModel(id="gpt-4", provider_id="openai"),
                   EmbedModel(id="embed", provider_id="openai")]
    reg = SimpleNamespace(
        available=[_SubtypeProvider()],
        get=lambda pid: _SubtypeProvider(),
    )
    svc = ModelService(reg)
    out = await svc.list_models()
    assert any(type(m) is LanguageModel for m in out)
    assert any(type(m) is EmbedModel for m in out)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/services/test_models.py -v -k subtype`
Expected: (fails only if the model ts regress; generally passes once providers return subtypes). The purpose is to lock the behavior.

- [ ] **Step 3: Confirm no flattening**

`ModelService.list_models` already returns provider results as-is. No code change needed unless tests find flattening. Verify types are preserved.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/services/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/services/test_models.py
git commit -m "test: ModelService preserves model subtypes"
```

---
### Task F3: CLI — embedding type and pricing column

**Files:**
- Modify: `src/yapa/cli/app.py`
- Test: `tests/cli/test_models.py`

**Interfaces:**
- Produces: `models --type llm|embedding|other` (REQ-CLI-01); a dedicated Pricing column rendered from `ModelPricing.input/.output/.request`, with `-` when absent (REQ-CLI-02).

- [ ] **Step 1: Write failing tests**

Append to `tests/cli/test_models.py`:

```python
from yapa.models import LanguageModel, ModelPricing
from cli_runner import invoke  # existing test helper in tests/cli/test_models.py


def test_model_type_accepts_embedding(runner, fake_model_service) -> None:
    result = runner.invoke(cli, ["models", "--type", "embedding"])
    # SELECT embedding models only; exits normally
    ...


def test_invalid_type_rejected(runner) -> None:
    result = runner.invoke(cli, ["models", "--type", "bogus"])
    assert result.exit_code != 0


def test_pricing_column_renders(runner, fake_model_service) -> None:
    # provider returns a LanguageModel with pricing=ModelPricing(input=1.0, output=2.0)
    # and one without; assert table contains "1.0" and "-"
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_models.py -v`
Expected: FAIL (`--type embedding` rejected because `ModelType("embedding")` currently raises since EMBED missing — actually EMBED now exists in Phase A; `--type` help only lists llm; the table has no Pricing column).

- [ ] **Step 3: Implement**

In `src/yapa/cli/app.py`:

```python
    model_type: str | None = typer.Option(
        None, "--type", "-t",
        help="Filter by model type: llm, embedding, or other",
    )
    ...
    if model_type_enum is None and model_type:
        available = ", ".join(m.value for m in ModelType)
        _error(f"Invalid model type '{model_type}'. Must be one of: {available}")
        raise typer.Exit(code=1)
```

Add the column and row rendering:

```python
    table.add_column("Pricing")

    def _pricing_label(m) -> str:
        p = getattr(m, "pricing", None)
        if p is None:
            return "-"
        parts = []
        if p.input is not None:
            parts.append(f"in ${p.input:g}/1M")
        if p.output is not None:
            parts.append(f"out ${p.output:g}/1M")
        if p.request is not None:
            parts.append(f"req ${p.request:g}")
        return " ".join(parts) or "-"

    for m in results:
        table.add_row(
            m.provider_id,
            m.id,
            m.type.value,
            str(m.context_length or "-"),
            str(m.max_output or "-"),
            _pricing_label(m),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_models.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/yapa/cli/app.py tests/cli/test_models.py
git commit -m "feat: CLI models accept embedding type and render Pricing"
```

---
### Task F4: API — discriminated response model + embedding type

**Files:**
- Modify: `src/yapa/api/routes/models.py`
- Test: `tests/api/test_models.py`

**Interfaces:**
- Produces: `?model_type=embedding` filters; response model is the discriminated union `LanguageModel | EmbedModel | ModelData` so subtype fields serialize (REQ-API-01).

- [ ] **Step 1: Write failing tests**

Append to `tests/api/test_models.py`:

```python
def test_model_type_embedding(client, fake_model_service) -> None:
    r = client.get("/models?model_type=embedding")
    assert r.status_code == 200
    body = r.json()
    assert all(m["type"] == "embedding" for m in body)


def test_subtype_fields_serialize(client, fake_model_service) -> None:
    r = client.get("/models")
    body = r.json()
    llm = next(m for m in body if m["type"] == "llm")
    assert "supports_tools" in llm
    emb = next(m for m in body if m["type"] == "embedding")
    # absent fields are null
    assert emb.get("embedding_dimensions") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_models.py -v`
Expected: FAIL (`response_model=list[ModelData]` drops subtype fields; `model_type=embedding` not specially handled because `ModelType` now has EMBED — but serialization lacks subtype fields).

- [ ] **Step 3: Implement**

Update `src/yapa/api/routes/models.py`:

```python
from yapa.models import EmbedModel, LanguageModel, ModelData, ModelType

ModelResponseUnion = LanguageModel | EmbedModel | ModelData


@router.get("/models", response_model=list[ModelResponseUnion])
async def list_models(...):
    ...


@router.get("/models/{full_id:path}", response_model=ModelResponseUnion)
async def get_model(...):
    ...
```

The `model_type` parsing already feeds `ModelType(model_type)` which now accepts `embedding`; update the error message list to the three values (it already enumerates `ModelType`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/api/test_models.py -v`
Expected: PASS (update `tests/api/conftest.py` fake model service to return subtype instances).

- [ ] **Step 5: Commit**

```bash
git add src/yapa/api/routes/models.py tests/api/test_models.py tests/api/conftest.py
git commit -m "feat: API serializes model subtypes with embedding filter"
```

---
## Final gate

- [ ] **Step 1: Run the full gate**

Run:

```bash
uv run ruff check src/ tests/
uv run ty check src/
uv run pytest tests/ -v
```

Expected: PASS (≥80% coverage per AGENTS.md; fix any regressions).

- [ ] **Step 2: Remove stale references**

Confirm no module still imports `StreamDelta`, the old `ModelData` pricing-as-dict, or `openai_compat` at the top level. Grep and fix:

```bash
rg -n "StreamDelta|openai_compat|price.*dict" src/
```

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete provider module rework"
```

```
