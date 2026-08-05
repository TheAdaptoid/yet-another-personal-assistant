"""LM Studio provider tests — AsyncOpenAI client, native listing, reasoning."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from yapa.models import EmbedModel, LanguageModel, ModelType, ReasoningEffort
from yapa.providers.exceptions import ModelInvocationError, ModelsFetchError
from yapa.providers.lmstudio import LMStudioIP
from yapa.services.config import Config, ProviderConfig


def _cfg(**kw) -> Config:
    return Config(provider_configs={"lmstudio": ProviderConfig(api_key=None, **kw)})


def _provider(config):
    with patch("yapa.providers.openai._noauth.AsyncOpenAI"):
        return LMStudioIP(config)


class TestEndpointDerivation:
    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("http://localhost:1234/v1", "http://localhost:1234/api/v1/models"),
            ("http://localhost:1234/v1/", "http://localhost:1234/api/v1/models"),
            ("http://localhost:1234", "http://localhost:1234/api/v1/models"),
            (
                "http://localhost:1234/custom",
                "http://localhost:1234/custom/api/v1/models",
            ),
        ],
    )
    def test_models_endpoint(self, base_url, expected) -> None:
        p = _provider(_cfg(base_url=base_url))
        p._client = SimpleNamespace(base_url=base_url)
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
                {
                    "key": "llama-3.1",
                    "type": "llm",
                    "capabilities": ["chat-completion"],
                    "max_context_length": 8192,
                },
                {
                    "key": "embed",
                    "type": "embedding",
                    "capabilities": ["embeddings"],
                    "max_context_length": 512,
                },
            ]
        }
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=raw)
        with patch("httpx.AsyncClient") as mk:
            mk.return_value.__aenter__ = AsyncMock(return_value=mk.return_value)
            mk.return_value.__aexit__ = AsyncMock(return_value=False)
            mk.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            models = await p._list_models_impl()
        by_id = {m.id: m for m in models}
        assert type(by_id["llama-3.1"]) is LanguageModel
        assert type(by_id["embed"]) is EmbedModel

    async def test_listing_filters_by_model_type(self) -> None:
        raw = {
            "models": [
                {
                    "key": "llama-3.1",
                    "type": "llm",
                    "capabilities": ["chat-completion"],
                    "max_context_length": 8192,
                },
                {
                    "key": "embed",
                    "type": "embedding",
                    "capabilities": ["embeddings"],
                    "max_context_length": 512,
                },
            ]
        }
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=raw)
        with patch("httpx.AsyncClient") as mk:
            mk.return_value.__aenter__ = AsyncMock(return_value=mk.return_value)
            mk.return_value.__aexit__ = AsyncMock(return_value=False)
            mk.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            llms = await p._list_models_impl(ModelType.LLM)
            embeds = await p._list_models_impl(ModelType.EMBED)
        assert [m.id for m in llms] == ["llama-3.1"]
        assert all(type(m) is LanguageModel for m in llms)
        assert [m.id for m in embeds] == ["embed"]
        assert all(type(m) is EmbedModel for m in embeds)

    async def test_listing_sets_supports_reasoning(self) -> None:
        raw = {
            "models": [
                {
                    "key": "reasoning-model",
                    "type": "llm",
                    "capabilities": {"reasoning": {"allowed_options": ["off", "on"]}},
                    "max_context_length": 8192,
                },
                {
                    "key": "plain-model",
                    "type": "llm",
                    "capabilities": {"vision": True},
                    "max_context_length": 4096,
                },
            ]
        }
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=raw)
        with patch("httpx.AsyncClient") as mk:
            mk.return_value.__aenter__ = AsyncMock(return_value=mk.return_value)
            mk.return_value.__aexit__ = AsyncMock(return_value=False)
            mk.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            models = await p._list_models_impl()
        by_id = {m.id: m for m in models}
        assert by_id["reasoning-model"].supports_reasoning is True
        assert by_id["plain-model"].supports_reasoning is False

    async def test_get_model_absent_raises(self) -> None:
        async def _list():
            return [LanguageModel(id="llama-3.1", provider_id="lmstudio")]

        p = _provider(_cfg())
        p._list_models_impl = _list
        with pytest.raises(ModelsFetchError):
            await p._get_model_impl("nope")


def _make_api_status_error(
    status_code: int,
    message: str = "API error",
) -> APIStatusError:
    """Construct an openai.APIStatusError with a stub httpx.Response."""
    url = "http://localhost:1234/v1/test"
    response = httpx.Response(
        status_code=status_code, request=httpx.Request("POST", url)
    )
    return APIStatusError(
        message, response=response, body={"error": {"message": message}}
    )


class TestErrorConversion:
    """Tests that SDK/HTTP exceptions are converted to typed provider errors."""

    async def test_list_models_httpx_error_wraps_as_models_fetch_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise httpx.RequestError("connection failed")

        with patch("httpx.AsyncClient") as mk_client:
            client = mk_client.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            client.get = AsyncMock(side_effect=_fail)
            with pytest.raises(ModelsFetchError, match="connection failed"):
                await p.list_models()

    async def test_get_model_httpx_error_wraps_as_models_fetch_error(self) -> None:
        p = _provider(_cfg())

        async def _fail_list(*a, **k):
            raise httpx.RequestError("connection failed")

        with patch("httpx.AsyncClient") as mk_client:
            client = mk_client.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            client.get = AsyncMock(side_effect=_fail_list)
            with pytest.raises(ModelsFetchError, match="connection failed"):
                await p.get_model("llama-3.1")

    async def test_list_models_sdk_error_wraps_as_models_fetch_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise httpx.RequestError("server error")

        with patch("httpx.AsyncClient") as mk_client:
            client = mk_client.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            client.get = AsyncMock(side_effect=_fail)
            with pytest.raises(ModelsFetchError, match="server error"):
                await p.list_models()

    async def test_get_model_sdk_error_wraps_as_models_fetch_error(self) -> None:
        p = _provider(_cfg())

        async def _fail_list(*a, **k):
            raise httpx.RequestError("not found")

        with patch("httpx.AsyncClient") as mk_client:
            client = mk_client.return_value
            client.__aenter__ = AsyncMock(return_value=client)
            client.__aexit__ = AsyncMock(return_value=False)
            client.get = AsyncMock(side_effect=_fail_list)
            with pytest.raises(ModelsFetchError, match="not found"):
                await p.get_model("llama-3.1")

    async def test_stream_chat_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise _make_api_status_error(400, "bad request")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_fail))
            )
        )
        model = LanguageModel(id="llama-3.1", provider_id="lmstudio")
        with pytest.raises(ModelInvocationError, match="bad request"):
            [ev async for ev in p.stream_chat(model, [])]

    async def test_static_chat_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise _make_api_status_error(401, "unauthorized")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_fail))
            )
        )
        model = LanguageModel(id="llama-3.1", provider_id="lmstudio")
        with pytest.raises(ModelInvocationError, match="unauthorized"):
            await p.static_chat(model, [])

    async def test_embed_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise _make_api_status_error(429, "rate limited")

        p._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=AsyncMock(side_effect=_fail))
        )
        embed_model = EmbedModel(id="embed", provider_id="lmstudio")
        with pytest.raises(ModelInvocationError, match="rate limited"):
            await p.embed(embed_model, "hi")

    async def test_stream_chat_httpx_error_wraps_as_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise httpx.RequestError("connection failed")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_fail))
            )
        )
        model = LanguageModel(id="llama-3.1", provider_id="lmstudio")
        with pytest.raises(ModelInvocationError, match="connection failed"):
            [ev async for ev in p.stream_chat(model, [])]

    async def test_embed_httpx_error_wraps_as_model_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise httpx.ConnectTimeout("timeout")

        p._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=AsyncMock(side_effect=_fail))
        )
        embed_model = EmbedModel(id="embed", provider_id="lmstudio")
        with pytest.raises(ModelInvocationError, match="timeout"):
            await p.embed(embed_model, "hi")
