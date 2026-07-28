# Provider Layer Enhancements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ModelData metadata (context length, capabilities), token usage tracking, finish/stop reason, and retry logic to the provider layer.

**Architecture:** Extend existing Pydantic models (`ModelData`, `StreamDelta`, `AssistantMessage`) with new optional fields. Add a static lookup table for well-known OpenAI model metadata. Use the OpenAI SDK's built-in `max_retries` parameter for retry. Extend the OpenAI streaming implementation to request and parse usage data from the API.

**Tech Stack:** Python 3.13+, Pydantic, openai SDK

## Global Constraints

- All new `ModelData` fields must be optional with sensible defaults (None for scalars, False for booleans) to maintain backward compatibility
- `TokenUsage` must be a Pydantic `BaseModel` with typed `int` fields
- `provider_max_retries` must default to `2` and accept `ge=0`
- OpenAI model lookup table should be a module-level `dict` in `openai/provider.py`
- All existing tests must continue to pass

---

### Task 1: Config and model changes

**Files:**
- Modify: `src/yapa/config.py:63-66`
- Modify: `src/yapa/models/inference.py:8,30-47,72-100`
- Modify: `src/yapa/models/message.py:47-63`
- Test: `tests/models/test_inference.py`
- Test: `tests/models/test_message.py`

**Interfaces:**
- Consumes: existing `Config`, `ModelData`, `StreamDelta`, `AssistantMessage`, `InferenceParams` definitions
- Produces: `TokenUsage(BaseModel)` with `prompt_tokens`, `completion_tokens`, `total_tokens` fields; `ModelData` with new optional fields; `StreamDelta` with new optional fields; `AssistantMessage` with new optional `usage` field; `Config` with `provider_max_retries`

- [ ] **Step 1: Add `provider_max_retries` to Config**

  In `src/yapa/config.py`, add a new field to the `Config` class after `provider_timeout`:

  ```python
  provider_timeout: int = Field(
      default=DEFAULT_PROVIDER_TIMEOUT, ge=1,
      description="Timeout in seconds for provider API calls",
  )
  provider_max_retries: int = Field(
      default=2, ge=0,
      description="Maximum number of retries for provider API calls",
  )
  ```

- [ ] **Step 2: Add `TokenUsage` model and extend `ModelData` and `StreamDelta` in `inference.py`**

  In `src/yapa/models/inference.py`:

  Add `TokenUsage` class after the imports:

  ```python
  class TokenUsage(BaseModel):
      prompt_tokens: int
      completion_tokens: int
      total_tokens: int
  ```

  Extend `ModelData` with new optional fields:

  ```python
  class ModelData(BaseModel):
      id: str = Field(..., description="Unique identifier for the model")
      provider_id: str = Field(
          ..., description="Identifier for the provider of the model"
      )
      type: ModelType = Field(..., description="The type of the model (e.g., 'llm')")
      context_length: int | None = Field(
          default=None, description="Maximum context length in tokens"
      )
      max_output: int | None = Field(
          default=None, description="Maximum output tokens"
      )
      supports_tools: bool = Field(
          default=False, description="Whether the model supports tool/function calling"
      )
      supports_vision: bool = Field(
          default=False, description="Whether the model supports image inputs"
      )
      pricing: dict[str, float] | None = Field(
          default=None, description="Per-token pricing in USD per million tokens, e.g. {'input': 2.50, 'output': 10.00}"
      )
  ```

  Extend `StreamDelta` with `finish_reason` and `usage`:

  ```python
  class StreamDelta(BaseModel):
      content: str | None = Field(default=None)
      reasoning_content: str | None = Field(default=None)
      tool_calls: list[ToolCallDelta] = Field(default_factory=list)
      error: str | None = Field(default=None)
      done: bool = Field(default=False)
      finish_reason: str | None = Field(
          default=None, description="Why the stream finished: stop, length, content_filter, tool_calls"
      )
      usage: TokenUsage | None = Field(
          default=None, description="Token usage for the completed stream"
      )
  ```

  Update the `__all__` in `src/yapa/models/__init__.py` to export `TokenUsage`.

- [ ] **Step 3: Extend `AssistantMessage` with `usage` in `message.py`**

  In `src/yapa/models/message.py`, add to `AssistantMessage`:

  ```python
  class AssistantMessage(BaseMessage):
      role: Literal["assistant"] = "assistant"
      reasoning_content: str | None = Field(default=None)
      model: str | None = Field(default=None)
      tool_calls: list[ToolCall] = Field(default_factory=list)
      usage: TokenUsage | None = Field(
          default=None, description="Token usage for this response"
      )
  ```

  Add the import at the top:

  ```python
  from .inference import TokenUsage
  ```

- [ ] **Step 4: Write tests for model changes**

  In `tests/models/test_inference.py`, add tests:

  ```python
  class TestTokenUsage:
      """TokenUsage — fields and validation."""

      def test_fields(self):
          u = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
          assert u.prompt_tokens == 10
          assert u.completion_tokens == 20
          assert u.total_tokens == 30

      def test_json_round_trip(self):
          u = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
          data = u.model_dump(mode="json")
          restored = TokenUsage(**data)
          assert restored.total_tokens == 30

  class TestModelDataMetadata:
      """ModelData — new optional metadata fields."""

      def test_defaults(self):
          md = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
          assert md.context_length is None
          assert md.max_output is None
          assert md.supports_tools is False
          assert md.supports_vision is False
          assert md.pricing is None

      def test_can_set_metadata(self):
          md = ModelData(
              id="gpt-4o",
              provider_id="openai",
              type=ModelType.LLM,
              context_length=128000,
              max_output=16384,
              supports_tools=True,
              supports_vision=True,
              pricing={"input": 2.50, "output": 10.00},
          )
          assert md.context_length == 128000
          assert md.supports_tools is True

      def test_json_round_trip_with_metadata(self):
          md = ModelData(
              id="gpt-4o",
              provider_id="openai",
              type=ModelType.LLM,
              context_length=128000,
          )
          data = md.model_dump(mode="json")
          restored = ModelData(**data)
          assert restored.context_length == 128000

  class TestStreamDeltaMetadata:
      """StreamDelta — new finish_reason and usage fields."""

      def test_defaults(self):
          sd = StreamDelta()
          assert sd.finish_reason is None
          assert sd.usage is None

      def test_can_set_finish_reason(self):
          sd = StreamDelta(finish_reason="stop")
          assert sd.finish_reason == "stop"

      def test_can_set_usage(self):
          usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
          sd = StreamDelta(usage=usage)
          assert sd.usage.prompt_tokens == 10
          assert sd.usage.total_tokens == 30

      def test_json_round_trip(self):
          usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
          sd = StreamDelta(content="hi", finish_reason="stop", usage=usage)
          data = sd.model_dump(mode="json")
          restored = StreamDelta(**data)
          assert restored.finish_reason == "stop"
          assert restored.usage is not None
          assert restored.usage.total_tokens == 30
  ```

  In `tests/models/test_message.py`, add a test to `TestDiscriminator`:

  ```python
  def test_assistant_message_with_usage(self):
      from yapa.models import TokenUsage

      usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
      msg = AssistantMessage(content="hi", usage=usage)
      data = msg.model_dump(mode="json")
      restored = _adapter.validate_python(data)
      assert isinstance(restored, AssistantMessage)
      assert restored.usage is not None
      assert restored.usage.total_tokens == 30
  ```

- [ ] **Step 5: Run model tests**

  Run: `uv run pytest tests/models/test_inference.py tests/models/test_message.py -v`

  Expected: all tests pass

- [ ] **Step 6: Commit**

  ```bash
  git add src/yapa/config.py src/yapa/models/ tests/models/
  git commit -m "feat: add TokenUsage model and extend ModelData/StreamDelta/AssistantMessage with metadata fields"
  ```

---

### Task 2: OpenAI lookup table

**Files:**
- Modify: `src/yapa/providers/openai/provider.py`
- Test: `tests/providers/test_openai_compat.py`

**Interfaces:**
- Consumes: `ModelData` from Task 1
- Produces: `_MODEL_METADATA` lookup dict; `OpenAIIP._format_model()` override that populates metadata from the lookup on top of the base class implementation

- [ ] **Step 1: Add the lookup table and override `_format_model`**

  In `src/yapa/providers/openai/provider.py`, add a module-level dict:

  ```python
  _MODEL_METADATA: dict[str, dict[str, object]] = {
      "gpt-4o": {"context_length": 128000, "max_output": 16384, "supports_tools": True, "supports_vision": True},
      "gpt-4o-mini": {"context_length": 128000, "max_output": 16384, "supports_tools": True, "supports_vision": True},
      "gpt-4-turbo": {"context_length": 128000, "max_output": 4096, "supports_tools": True, "supports_vision": True},
      "gpt-4": {"context_length": 8192, "max_output": 4096, "supports_tools": True, "supports_vision": False},
      "gpt-3.5-turbo": {"context_length": 16385, "max_output": 4096, "supports_tools": True, "supports_vision": False},
      "o1": {"context_length": 200000, "max_output": 100000, "supports_tools": True, "supports_vision": True},
      "o1-mini": {"context_length": 128000, "max_output": 65536, "supports_tools": True, "supports_vision": True},
      "o3-mini": {"context_length": 200000, "max_output": 100000, "supports_tools": True, "supports_vision": True},
  }
  ```

  Then add an override of `_format_model`:

  ```python
  class OpenAIIP(OpenAICompatibleProvider):
      def __init__(self, config: Config):
          ...

      def _format_model(self, model_id: str) -> ModelData:
          model = super()._format_model(model_id)
          meta = _MODEL_METADATA.get(model_id)
          if meta is not None:
              return ModelData(
                  id=model.id,
                  provider_id=model.provider_id,
                  type=model.type,
                  context_length=meta["context_length"],  # type: ignore
                  max_output=meta["max_output"],  # type: ignore
                  supports_tools=meta["supports_tools"],  # type: ignore
                  supports_vision=meta["supports_vision"],  # type: ignore
              )
          return model
  ```

- [ ] **Step 2: Write tests for the lookup table**

  In `tests/providers/test_openai_compat.py`, add a `TestOpenAIModelMetadata` class:

  ```python
  class TestOpenAIModelMetadata:
      @pytest.fixture
      def openai_provider(self, mock_openai_client):
          from yapa.config import Config
          from yapa.providers.openai import OpenAIIP
          with patch("yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client):
              provider = OpenAIIP(config=Config(openai_api_key="sk-test"))
              provider._client = mock_openai_client
              return provider

      def test_known_model_gets_metadata(self, openai_provider):
          model = openai_provider._format_model("gpt-4o")
          assert model.context_length == 128000
          assert model.supports_tools is True
          assert model.supports_vision is True

      def test_unknown_model_defaults(self, openai_provider):
          model = openai_provider._format_model("unknown-model")
          assert model.context_length is None
          assert model.supports_tools is False

      def test_embed_model_gets_other_type(self, openai_provider):
          model = openai_provider._format_model("text-embedding-3")
          assert model.type == ModelType.OTHER
          assert model.supports_tools is False
  ```

- [ ] **Step 3: Run tests**

  Run: `uv run pytest tests/providers/test_openai_compat.py::TestOpenAIModelMetadata tests/providers/test_init.py::TestOpenAIIPInit -v`

  Expected: all pass

- [ ] **Step 4: Commit**

  ```bash
  git add src/yapa/providers/openai/provider.py tests/providers/test_openai_compat.py
  git commit -m "feat: add OpenAI model metadata lookup table"
  ```

---

### Task 3: Retry logic

**Files:**
- Modify: `src/yapa/providers/openai_compat.py:44-63,152-175`
- Modify: `src/yapa/providers/openai/provider.py`
- Modify: `src/yapa/providers/lmstudio/provider.py`
- Modify: `src/yapa/providers/ollama/provider.py`
- Modify: `src/yapa/providers/openrouter/provider.py`
- Test: `tests/providers/test_init.py`

**Interfaces:**
- Consumes: `Config.provider_max_retries` from Task 1
- Produces: `OpenAICompatibleProvider.__init__` accepts `max_retries: int = 2` parameter; all four concrete providers pass `config.provider_max_retries`

- [ ] **Step 1: Add `max_retries` parameter to `OpenAICompatibleProvider.__init__`**

  In `src/yapa/providers/openai_compat.py`, change the constructor:

  ```python
  def __init__(
      self,
      identifier: str,
      name: str,
      api_key: str,
      base_url: str | None,
      timeout: int = DEFAULT_PROVIDER_TIMEOUT,
      max_retries: int = 2,
  ) -> None:
      super().__init__(identifier, name)
      self._client = AsyncOpenAI(
          api_key=api_key,
          base_url=base_url,
          timeout=timeout,
          max_retries=max_retries,
      )
  ```

  Add `max_retries` import — it's already a parameter of `AsyncOpenAI`, no additional imports needed.

- [ ] **Step 2: Update all concrete providers to pass `max_retries`**

  In `src/yapa/providers/openai/provider.py`:

  ```python
  super().__init__(
      identifier="openai",
      name="OpenAI",
      api_key=config.openai_api_key,
      base_url=config.openai_base_url,
      timeout=config.provider_timeout,
      max_retries=config.provider_max_retries,
  )
  ```

  In `src/yapa/providers/lmstudio/provider.py`:

  ```python
  super().__init__(
      identifier="lmstudio",
      name="LM Studio",
      api_key=config.lmstudio_api_key,
      base_url=config.lmstudio_base_url,
      timeout=config.provider_timeout,
      max_retries=config.provider_max_retries,
  )
  ```

  In `src/yapa/providers/ollama/provider.py`:

  ```python
  super().__init__(
      identifier="ollama",
      name="Ollama",
      api_key=config.ollama_api_key,
      base_url=config.ollama_base_url,
      timeout=config.provider_timeout,
      max_retries=config.provider_max_retries,
  )
  ```

  In `src/yapa/providers/openrouter/provider.py`:

  ```python
  super().__init__(
      identifier="openrouter",
      name="OpenRouter",
      api_key=config.openrouter_api_key,
      base_url=config.openrouter_base_url,
      timeout=config.provider_timeout,
      max_retries=config.provider_max_retries,
  )
  ```

- [ ] **Step 3: Write tests**

  In `tests/providers/test_init.py`, update `TestOpenAIIPInit.test_initializes_with_valid_key` to verify `mock_openai_client` was called with `max_retries`. Add an assertion:

  ```python
  def test_initializes_with_valid_key(self, mock_openai_client) -> None:
      config = Config(openai_api_key="sk-test", provider_max_retries=3)
      with patch("yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client) as mock_client:
          provider = OpenAIIP(config=config)
      assert provider.id == "openai"
      assert provider.name == "OpenAI"
      mock_client.assert_called_once()
      assert mock_client.call_args.kwargs["max_retries"] == 3
  ```

  Add similar assertion to `TestOpenRouterProviderInit.test_initializes_with_valid_key`:

  ```python
  def test_initializes_with_valid_key(self, mock_openai_client) -> None:
      config = Config(openrouter_api_key="sk-or-test", provider_max_retries=0)
      with patch("yapa.providers.openai_compat.AsyncOpenAI", return_value=mock_openai_client) as mock_client:
          provider = OpenRouterProvider(config=config)
      assert provider.id == "openrouter"
      mock_client.assert_called_once()
      assert mock_client.call_args.kwargs["max_retries"] == 0
  ```

  For `TestLMStudioIPInit.test_initializes_with_config` and `TestOllamaIPInit.test_initializes_with_config`, add the same `mock_client.call_args.kwargs["max_retries"]` assertion with `== 2` (default).

- [ ] **Step 4: Run tests**

  Run: `uv run pytest tests/providers/test_init.py -v`

  Expected: all tests pass

- [ ] **Step 5: Commit**

  ```bash
  git add src/yapa/providers/openai_compat.py src/yapa/providers/openai/provider.py src/yapa/providers/lmstudio/provider.py src/yapa/providers/ollama/provider.py src/yapa/providers/openrouter/provider.py tests/providers/test_init.py
  git commit -m "feat: add retry logic via OpenAI SDK max_retries"
  ```

---

### Task 4: Streaming metadata (finish reason + token usage)

**Files:**
- Modify: `src/yapa/providers/openai_compat.py:177-217,219-260`
- Test: `tests/providers/test_openai_compat.py`

**Interfaces:**
- Consumes: `StreamDelta.finish_reason`, `StreamDelta.usage`, `TokenUsage` from Task 1
- Produces: `_stream_chat_impl` yields final delta with `finish_reason` and `usage`; `_static_chat_impl` returns `AssistantMessage` with `usage`

- [ ] **Step 1: Update `_stream_chat_impl` to pass `stream_options` and extract metadata**

  In `src/yapa/providers/openai_compat.py`, modify `_stream_chat_impl`:

  Update `_common_pre_invoke` kwargs to include `stream_options`:

  ```python
  kwargs: dict[str, Any] = dict(
      model=model_id,
      messages=formatted_messages,
      temperature=params.temperature,
      max_tokens=params.max_tokens,
      top_p=params.top_p,
      stream=stream,
  )
  if stream:
      kwargs["stream_options"] = {"include_usage": True}
  ```

  Update the streaming loop in `_stream_chat_impl` to extract finish_reason and usage:

  ```python
  async for chunk in response_stream:
      delta = chunk.choices[0].delta
      content = delta.content
      reasoning_content = self._extract_reasoning_content(delta)

      tool_call_deltas: list[ToolCallDelta] = []
      if delta.tool_calls:
          for tc in delta.tool_calls:
              tool_call_deltas.append(...)

      finish_reason: str | None = chunk.choices[0].finish_reason

      usage: TokenUsage | None = None
      if chunk.usage is not None:
          usage = TokenUsage(
              prompt_tokens=chunk.usage.prompt_tokens,
              completion_tokens=chunk.usage.completion_tokens,
              total_tokens=chunk.usage.total_tokens,
          )

      yield StreamDelta(
          content=content,
          reasoning_content=reasoning_content,
          tool_calls=tool_call_deltas,
          finish_reason=finish_reason,
          usage=usage,
      )
  ```

  Add import at top:

  ```python
  from yapa.models import (
      ...
      TokenUsage,
      ...
  )
  ```

- [ ] **Step 2: Update `_static_chat_impl` to return usage**

  In `src/yapa/providers/openai_compat.py`, modify `_static_chat_impl`:

  After extracting tool_calls:

  ```python
  usage: TokenUsage | None = None
  if response.usage is not None:
      usage = TokenUsage(
          prompt_tokens=response.usage.prompt_tokens,
          completion_tokens=response.usage.completion_tokens,
          total_tokens=response.usage.total_tokens,
      )

  return AssistantMessage(
      role="assistant",
      content=content,
      reasoning_content=reasoning_content,
      tool_calls=tool_calls,
      usage=usage,
  )
  ```

- [ ] **Step 3: Write tests for streaming metadata**

  In `tests/providers/test_openai_compat.py`, add tests to existing classes or create new ones.

  In `TestStreamChatImpl`, add:

  ```python
  async def test_yields_finish_reason(self, compat_provider) -> None:
      chunk = _chunk(content="done", reasoning_content=None)
      chunk.choices[0].finish_reason = "stop"
      stream = _stream(chunk)
      mock_create = AsyncMock(return_value=stream)
      compat_provider._client.chat.completions.create = mock_create

      results: list[StreamDelta] = []
      async for delta in compat_provider._stream_chat_impl(
          model_id="gpt-4",
          messages=[UserMessage(content="hi")],
      ):
          results.append(delta)

      assert results[0].finish_reason == "stop"

  async def test_yields_usage(self, compat_provider) -> None:
      from yapa.models.token import TokenUsage

      chunk = _chunk(content="done", reasoning_content=None)
      chunk.choices[0].finish_reason = "stop"
      chunk.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
      stream = _stream(chunk)
      mock_create = AsyncMock(return_value=stream)
      compat_provider._client.chat.completions.create = mock_create

      results: list[StreamDelta] = []
      async for delta in compat_provider._stream_chat_impl(
          model_id="gpt-4",
          messages=[UserMessage(content="hi")],
      ):
          results.append(delta)

      assert results[0].usage is not None
      assert results[0].usage.prompt_tokens == 10
      assert results[0].usage.total_tokens == 30

  async def test_no_usage_when_not_in_response(self, compat_provider) -> None:
      stream = _stream(_chunk(content="ok", reasoning_content=None))
      mock_create = AsyncMock(return_value=stream)
      compat_provider._client.chat.completions.create = mock_create

      results: list[StreamDelta] = []
      async for delta in compat_provider._stream_chat_impl(
          model_id="gpt-4",
          messages=[UserMessage(content="hi")],
      ):
          results.append(delta)

      assert results[0].usage is None
      assert results[0].finish_reason is None
  ```

  In `TestStaticChatImpl`, add:

  ```python
  async def test_returns_usage(self, compat_provider) -> None:
      from yapa.models import TokenUsage

      response = self._make_response(content="Hello")
      response.usage = SimpleNamespace(
          prompt_tokens=10, completion_tokens=20, total_tokens=30
      )
      mock_create = AsyncMock(return_value=response)
      compat_provider._client.chat.completions.create = mock_create

      result = await compat_provider._static_chat_impl(
          model_id="gpt-4",
          messages=[UserMessage(content="hi")],
      )

      assert result.usage is not None
      assert result.usage.total_tokens == 30

  async def test_no_usage_when_not_in_response(self, compat_provider) -> None:
      response = self._make_response(content="Hello")
      response.usage = None
      mock_create = AsyncMock(return_value=response)
      compat_provider._client.chat.completions.create = mock_create

      result = await compat_provider._static_chat_impl(
          model_id="gpt-4",
          messages=[UserMessage(content="hi")],
      )

      assert result.usage is None
  ```

  Also add the `TokenUsage` import:

  ```python
  from yapa.models import (
      ...
      TokenUsage,
      ...
  )
  ```

  (If it's not exported, add it to `src/yapa/models/__init__.py`.)

- [ ] **Step 4: Run tests**

  Run: `uv run pytest tests/providers/test_openai_compat.py -v`

  Expected: all tests pass

- [ ] **Step 5: Run full test suite**

  Run: `uv run pytest tests/ -q`

  Expected: all tests pass, coverage >= 80%

- [ ] **Step 6: Commit**

  ```bash
  git add src/yapa/providers/openai_compat.py src/yapa/models/__init__.py tests/providers/test_openai_compat.py
  git commit -m "feat: populate finish_reason and token usage from streaming and static responses"
  ```

---

## Out of scope (deferred)

- LM Studio / Ollama model metadata (no lookup table yet, fields remain None/False)
- OpenRouter model metadata via `/v1/models` extended fields
- Vision/multimodal message support
- `response_format` / `stop` in `InferenceParams`
- Service-layer cost tracking, context window management, or cancellation UI
