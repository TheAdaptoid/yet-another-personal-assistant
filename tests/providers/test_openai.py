"""Tests for OpenAICompatibleProvider streaming, static chat, and embed."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APIStatusError

from yapa.models import (
    ContentDelta,
    EmbeddingResult,
    LanguageModel,
    ModelType,
    ReasoningDelta,
    StreamEndEvent,
    TokenUsage,
    ToolCallDeltaEvent,
)
from yapa.providers.exceptions import ModelInvocationError, ModelsFetchError
from yapa.providers.openai import OpenAIIP
from yapa.providers.openai.openai_compat import OpenAICompatibleProvider


class _P(OpenAICompatibleProvider):
    def __init__(self):
        super().__init__("x", "X", api_key="k", base_url="http://x/v1")


def _chunk(choices, usage=None, model="m") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=choices[0].delta, finish_reason=choices[0].finish_reason
            )
        ],
        usage=usage,
    )


async def _collect(p, chunks):
    stream = AsyncMock()
    stream.__aiter__.return_value = iter(chunks)
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=stream))
        )
    )
    p._client = client
    return [ev async for ev in p._stream_chat_impl("gpt", [], None, None, None)]


async def test_stream_usage_only_final_chunk_completes() -> None:
    p = _P()
    content = SimpleNamespace(
        delta=SimpleNamespace(content="hi", reasoning_content=None, tool_calls=None),
        finish_reason="stop",
    )
    usage_chunk = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )
    evs = await _collect(
        p,
        [
            SimpleNamespace(choices=[content], usage=None),
            usage_chunk,
        ],
    )
    content_evs = [e for e in evs if isinstance(e, ContentDelta)]
    end_evs = [e for e in evs if isinstance(e, StreamEndEvent)]
    assert content_evs and content_evs[0].content == "hi"
    assert len(end_evs) == 1
    assert end_evs[0].usage == TokenUsage(
        prompt_tokens=1, completion_tokens=2, total_tokens=3
    )
    assert end_evs[0].finish_reason == "stop"


async def test_stream_no_usage_chunk_usage_none() -> None:
    p = _P()
    content = SimpleNamespace(
        delta=SimpleNamespace(content="hi", reasoning_content=None, tool_calls=None),
        finish_reason=None,
    )
    evs = await _collect(p, [SimpleNamespace(choices=[content], usage=None)])
    end_evs = [e for e in evs if isinstance(e, StreamEndEvent)]
    assert len(end_evs) == 1
    assert end_evs[0].usage is None


async def test_stream_reasoning_and_tool_deltas() -> None:
    p = _P()
    reasoning = SimpleNamespace(
        delta=SimpleNamespace(content=None, reasoning_content="think", tool_calls=None),
        finish_reason=None,
    )
    tool = SimpleNamespace(
        delta=SimpleNamespace(
            content=None,
            reasoning_content=None,
            tool_calls=[
                SimpleNamespace(
                    index=0,
                    id="call_1",
                    function=SimpleNamespace(name="calc", arguments='{"a":'),
                )
            ],
        ),
        finish_reason="tool_calls",
    )
    evs = await _collect(
        p,
        [
            SimpleNamespace(choices=[reasoning], usage=None),
            SimpleNamespace(choices=[tool], usage=None),
        ],
    )
    assert any(isinstance(e, ReasoningDelta) and e.content == "think" for e in evs)
    assert any(
        isinstance(e, ToolCallDeltaEvent)
        and e.id == "call_1"
        and e.arguments == '{"a":'
        for e in evs
    )
    ends = [e for e in evs if isinstance(e, StreamEndEvent)]
    assert ends[0].finish_reason == "tool_calls"


async def test_static_empty_arguments_normalize_to_empty_dict() -> None:
    p = _P()
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="x",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="calc", arguments=""),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    p._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=resp))
        )
    )
    out = await p._static_chat_impl("gpt", [], None, None, None)
    assert out.tool_calls[0].arguments == {}


async def test_static_invalid_json_raises_model_invocation_error() -> None:
    p = _P()
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="x",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="calc", arguments="not json"),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    p._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=resp))
        )
    )
    with pytest.raises(ModelInvocationError):
        await p._static_chat_impl("gpt", [], None, None, None)


async def test_static_valid_json_parsed_to_dict() -> None:
    p = _P()
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="x",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="calc", arguments='{"a": 1}'),
                        )
                    ],
                ),
            )
        ],
        usage=None,
    )
    p._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=resp))
        )
    )
    out = await p._static_chat_impl("gpt", [], None, None, None)
    assert out.tool_calls[0].arguments == {"a": 1}


async def test_embed_maps_usage_to_token_usage() -> None:
    p = _P()
    esc = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2], index=0)],
        usage=SimpleNamespace(prompt_tokens=6, total_tokens=6),
    )
    p._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock(return_value=esc))
    )
    result = await p._embed_impl("embed", "hi")
    assert isinstance(result, EmbeddingResult)
    assert result.vectors == [[0.1, 0.2]]
    assert result.usage.prompt_tokens == 6
    assert result.usage.total_tokens == 6


async def test_embed_usage_none_when_missing() -> None:
    p = _P()
    esc = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0], index=0)], usage=None)
    p._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=AsyncMock(return_value=esc))
    )
    result = await p._embed_impl("embed", "hi")
    assert result.usage is None


async def test_list_models_filters_by_type() -> None:
    p = _P()
    resp = SimpleNamespace(
        data=[SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="text-embedding-3")]
    )
    p._client = SimpleNamespace(
        models=SimpleNamespace(list=AsyncMock(return_value=resp))
    )

    llms = await p._list_models_impl(model_type=ModelType.LLM)
    embeds = await p._list_models_impl(model_type=ModelType.EMBED)

    assert [m.id for m in llms] == ["gpt-4o"]
    assert [m.id for m in embeds] == ["text-embedding-3"]


def _openai_provider(config=None):
    from unittest.mock import patch

    from yapa.providers.openai import OpenAIIP
    from yapa.services.config import Config, ProviderConfig

    with patch("yapa.providers.openai._noauth.AsyncOpenAI"):
        return OpenAIIP(
            config
            or Config(provider_configs={"openai": ProviderConfig(api_key="sk-t")})
        )


@pytest.mark.parametrize("model_id", list(OpenAIIP._MODEL_METADATA.keys()))
def test_metadata_every_table_entry(model_id: str) -> None:
    from yapa.models import LanguageModel

    p = _openai_provider()
    m = p._format_model(model_id)
    assert type(m) is LanguageModel
    meta = OpenAIIP._MODEL_METADATA[model_id]
    assert m.context_length == meta.get("context_length")
    assert m.max_output == meta.get("max_output")
    assert m.supports_tools == meta.get("supports_tools", False)
    assert m.supports_vision == meta.get("supports_vision", False)
    assert m.supports_reasoning == meta.get("supports_reasoning", False)
    assert m.pricing == meta.get("pricing")


def test_unknown_model_yields_default_metadata() -> None:
    from yapa.models import LanguageModel

    p = _openai_provider()
    m = p._format_model("totally-unknown-model")
    assert type(m) is LanguageModel
    assert m.context_length is None
    assert m.max_output is None
    assert m.supports_tools is False
    assert m.supports_vision is False
    assert m.supports_reasoning is False
    assert m.pricing is None


def test_format_preserves_explicit_falsy_values_over_metadata() -> None:
    p = _openai_provider()
    m = p._format_model(
        "gpt-5.6",
        name="",
        description="",
        context_length=0,
        max_output=0,
    )
    assert type(m) is LanguageModel
    assert m.name == ""
    assert m.description == ""
    assert m.context_length == 0
    assert m.max_output == 0


def _make_api_status_error(
    status_code: int,
    message: str = "API error",
) -> APIStatusError:
    """Construct an openai.APIStatusError with a stub httpx.Response."""
    url = "https://api.openai.com/v1/test"
    response = httpx.Response(
        status_code=status_code, request=httpx.Request("POST", url)
    )
    return APIStatusError(
        message, response=response, body={"error": {"message": message}}
    )


class TestErrorConversion:
    """Tests that SDK/HTTP exceptions are converted to typed provider errors."""

    async def test_list_models_sdk_error_wraps_as_models_fetch_error(self) -> None:
        p = _P()

        async def _fail(*a, **k):
            raise _make_api_status_error(500, "server error")

        p._client = SimpleNamespace(
            models=SimpleNamespace(list=AsyncMock(side_effect=_fail))
        )
        with pytest.raises(ModelsFetchError, match="server error"):
            await p.list_models()

    async def test_get_model_sdk_error_wraps_as_models_fetch_error(self) -> None:
        p = _P()

        async def _fail(*a, **k):
            raise _make_api_status_error(404, "not found")

        p._client = SimpleNamespace(
            models=SimpleNamespace(retrieve=AsyncMock(side_effect=_fail))
        )
        with pytest.raises(ModelsFetchError, match="not found"):
            await p.get_model("gpt-4")

    async def test_stream_chat_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _P()

        async def _fail(*a, **k):
            raise _make_api_status_error(400, "bad request")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_fail))
            )
        )
        model = LanguageModel(id="gpt", provider_id="x")
        with pytest.raises(ModelInvocationError, match="bad request"):
            [ev async for ev in p.stream_chat(model, [])]

    async def test_static_chat_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _P()

        async def _fail(*a, **k):
            raise _make_api_status_error(401, "unauthorized")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_fail))
            )
        )
        model = LanguageModel(id="gpt", provider_id="x")
        with pytest.raises(ModelInvocationError, match="unauthorized"):
            await p.static_chat(model, [])

    async def test_embed_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _P()

        async def _fail(*a, **k):
            raise _make_api_status_error(429, "rate limited")

        p._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=AsyncMock(side_effect=_fail))
        )
        from yapa.models import EmbedModel

        embed_model = EmbedModel(id="embed", provider_id="x")
        with pytest.raises(ModelInvocationError, match="rate limited"):
            await p.embed(embed_model, "hi")

    async def test_stream_chat_httpx_error_wraps_as_invocation_error(self) -> None:
        p = _P()

        async def _fail(*a, **k):
            raise httpx.RequestError("connection failed")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_fail))
            )
        )
        model = LanguageModel(id="gpt", provider_id="x")
        with pytest.raises(ModelInvocationError, match="connection failed"):
            [ev async for ev in p.stream_chat(model, [])]

    async def test_embed_httpx_error_wraps_as_model_invocation_error(self) -> None:
        p = _P()

        async def _fail(*a, **k):
            raise httpx.ConnectTimeout("timeout")

        p._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=AsyncMock(side_effect=_fail))
        )
        from yapa.models import EmbedModel

        embed_model = EmbedModel(id="embed", provider_id="x")
        with pytest.raises(ModelInvocationError, match="timeout"):
            await p.embed(embed_model, "hi")
