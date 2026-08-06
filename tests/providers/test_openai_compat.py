"""Tests for OpenAICompatibleProvider base: classification, formatting, requests."""

from types import SimpleNamespace
from unittest.mock import patch

from yapa.models import (
    AssistantMessage,
    EmbedModel,
    InferenceParams,
    LanguageModel,
    ModelData,
    ModelType,
    ReasoningEffort,
    UserMessage,
)
from yapa.models.message import ImagePart, TextPart
from yapa.providers.openai.openai_compat import OpenAICompatibleProvider


class _Concrete(OpenAICompatibleProvider):
    pass


def _make(*args, **kwargs) -> _Concrete:
    return _Concrete(*args, **kwargs)


# ── Client construction ──


def test_client_built_with_timeout_and_retries() -> None:
    with patch("yapa.providers.openai._noauth.AsyncOpenAI") as mk:
        _make(
            identifier="x",
            name="X",
            api_key="sk-1",
            base_url="https://example.com/v1",
            timeout=120,
            max_retries=2,
        )
        kwargs = mk.call_args.kwargs
        assert kwargs["timeout"] == 120
        assert kwargs["max_retries"] == 2
        assert kwargs["api_key"] == "sk-1"


def test_no_auth_client_when_key_empty() -> None:
    with (
        patch("yapa.providers.openai._noauth.AsyncOpenAI") as mk,
        patch("yapa.providers.openai._noauth.httpx"),
    ):
        _make(
            identifier="x",
            name="X",
            api_key=None,
            base_url="https://example.com/v1",
            timeout=30,
            max_retries=0,
        )
        kwargs = mk.call_args.kwargs
        # sentinel key passed; an httpx client is supplied to strip Authorization
        assert kwargs["api_key"] == "no-key-provider"
        assert "http_client" in kwargs


# ── _format_model classification ──


class _Format(_Concrete):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")


def test_native_llm_overrides_embed_keyword() -> None:
    p = _Format()
    m = p._format_model("text-embedding-3-large", native_type="llm")
    assert type(m) is LanguageModel
    assert m.type == "llm"


def test_embed_keyword_without_native_type_is_embed() -> None:
    p = _Format()
    m = p._format_model("text-embedding-3-large")
    assert type(m) is EmbedModel
    assert m.type == "embedding"


def test_audio_image_keywords_are_other() -> None:
    p = _Format()
    assert p._format_model("my-audio-model").type == ModelType.OTHER
    assert p._format_model("my-image-model").type == ModelType.OTHER


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
        assert type(m) is not ModelData


# ── _format_message ──


class _Fmt(_Concrete):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")


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


# ── _extract_reasoning precedence ──


class _RM(_Concrete):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")


def test_reasoning_content_extracted() -> None:
    p = _RM()
    assert p._extract_reasoning(SimpleNamespace(reasoning_content="think")) == "think"


def test_reasoning_attribute_never_used_as_fallback() -> None:
    p = _RM()
    obj = SimpleNamespace(reasoning="should NOT win", reasoning_content="winner")
    assert p._extract_reasoning(obj) == "winner"


def test_reasoning_fallback_to_reasoning_field() -> None:
    """When reasoning_content is absent, fall back to reasoning (e.g. OpenRouter)."""
    p = _RM()
    obj = SimpleNamespace(reasoning="fallback only")
    assert p._extract_reasoning(obj) == "fallback only"
    obj2 = SimpleNamespace(reasoning="r", reasoning_content=None)
    assert p._extract_reasoning(obj2) == "r"
    obj3 = SimpleNamespace(reasoning=None, reasoning_content="   ")
    assert p._extract_reasoning(obj3) is None


def test_empty_whitespace_reasoning_is_none() -> None:
    p = _RM()
    assert p._extract_reasoning(SimpleNamespace(reasoning_content="   ")) is None


# ── _build_request_kwargs ──


class _RB(_Concrete):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")


def test_unset_params_omitted_from_request() -> None:
    p = _RB()
    kw = p._build_request_kwargs(
        "gpt", [], None, InferenceParams(), stream=False, reasoning=None
    )
    assert "temperature" not in kw
    assert "max_tokens" not in kw
    assert "top_p" not in kw


def test_set_params_sent() -> None:
    p = _RB()
    kw = p._build_request_kwargs(
        "gpt",
        [],
        None,
        InferenceParams(temperature=0.7, max_tokens=100),
        stream=False,
        reasoning=None,
    )
    assert kw["temperature"] == 0.7
    assert kw["max_tokens"] == 100
    assert "top_p" not in kw


def test_unsupported_params_omitted_from_request() -> None:
    p = _RB()
    kw = p._build_request_kwargs(
        "gpt",
        [],
        None,
        InferenceParams(top_k=5, min_p=0.5, repeat_penalty=1.2, temperature=0.7),
        stream=False,
        reasoning=None,
    )
    assert kw["temperature"] == 0.7
    assert "top_k" not in kw
    assert "min_p" not in kw
    assert "repeat_penalty" not in kw


def test_stream_usage_option_when_supported() -> None:
    p = _RB()
    kw = p._build_request_kwargs(
        "gpt", [], None, InferenceParams(), stream=True, reasoning=None
    )
    assert kw.get("stream_options") == {"include_usage": True}


def test_stream_usage_option_omitted_when_unsupported() -> None:
    class _NoUsage(_RB):
        _SUPPORTS_STREAM_USAGE = False

    p = _NoUsage()
    kw = p._build_request_kwargs(
        "gpt", [], None, InferenceParams(), stream=True, reasoning=None
    )
    assert "stream_options" not in kw


def test_reasoning_mapping() -> None:
    p = _RB()
    kw = p._build_request_kwargs(
        "gpt", [], None, InferenceParams(), stream=False, reasoning=ReasoningEffort.HIGH
    )
    assert kw["reasoning"] == {"effort": "high"}
    kw_low = p._build_request_kwargs(
        "gpt", [], None, InferenceParams(), stream=False, reasoning=ReasoningEffort.LOW
    )
    assert kw_low["reasoning"] == {"effort": "low"}
    kw_off = p._build_request_kwargs(
        "gpt", [], None, InferenceParams(), stream=False, reasoning=ReasoningEffort.OFF
    )
    assert "reasoning" not in kw_off
    kw_none = p._build_request_kwargs(
        "gpt", [], None, InferenceParams(), stream=False, reasoning=None
    )
    assert "reasoning" not in kw_none
