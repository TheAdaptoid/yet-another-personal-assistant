from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from yapa.models import EmbedModel, LanguageModel, ModelType
from yapa.providers.exceptions import ModelInvocationError, ModelsFetchError
from yapa.providers.openrouter import OpenRouterProvider
from yapa.services.config import Config, ProviderConfig

_ASYNC_OPENAI = "yapa.providers.openai._noauth.AsyncOpenAI"


def _cfg(**kw) -> Config:
    pc = ProviderConfig(api_key="sk-or", **kw)
    return Config(provider_configs={"openrouter": pc})


def _provider(config):
    with patch(_ASYNC_OPENAI):
        return OpenRouterProvider(config)


def _raw_model(mid, modality="text", pricing=None, supported=None, ctx=1000, mct=500):
    return {
        "id": mid,
        "name": mid,
        "description": f"{mid} desc",
        "context_length": ctx,
        "max_completion_tokens": mct,
        "architecture": {"modality": modality},
        "supported_parameters": supported or [],
        "pricing": pricing
        or {
            "prompt": 0.000001,
            "completion": 0.000002,
            "request": 0.0,
            "image": 0.0,
            "web_search": 0.0,
        },
    }


class TestEndpointDerivation:
    @pytest.mark.parametrize(
        ("base_url", "expected_models_url"),
        [
            ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/models"),
            ("https://openrouter.ai/api/v1/", "https://openrouter.ai/api/v1/models"),
            ("https://openrouter.ai/api", "https://openrouter.ai/api/models"),
            (
                "https://openrouter.ai/api/v1/custom",
                "https://openrouter.ai/api/v1/custom/models",
            ),
        ],
    )
    def test_models_endpoint(self, base_url, expected_models_url) -> None:
        p = _provider(_cfg(base_url=base_url))
        assert p._models_endpoint() == expected_models_url


class TestListing:
    async def test_listing_returns_subtypes_and_pricing(self) -> None:
        raw = [
            _raw_model("openai/gpt-4"),
            _raw_model(
                "openai/text-embedding-3", modality="text", supported=["embeddings"]
            ),
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"data": raw})
        with patch("httpx.AsyncClient") as mk_client:
            mk_client.return_value.__aenter__ = AsyncMock(
                return_value=mk_client.return_value
            )
            mk_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mk_client.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            models = await p._list_models_impl()
        by_id = {m.id: m for m in models}
        assert type(by_id["openai/gpt-4"]) is LanguageModel
        assert type(by_id["openai/text-embedding-3"]) is EmbedModel
        gpt = by_id["openai/gpt-4"]
        assert gpt.pricing.input == 0.001  # 0.000001 * 1000 (per-1M)
        assert gpt.pricing.output == 0.002
        assert gpt.pricing.request == 0.0
        # image/web_search dropped
        assert gpt.pricing.model_fields_set == {"input", "output", "request"}

    async def test_pricing_accepts_input_output_keys(self) -> None:
        raw = [
            _raw_model(
                "openai/x",
                pricing={
                    "input": 0.001,
                    "output": 0.002,
                    "request": 0.0,
                    "image": 0.0,
                },
            )
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"data": raw})
        with patch("httpx.AsyncClient") as mk_client:
            mk_client.return_value.__aenter__ = AsyncMock(
                return_value=mk_client.return_value
            )
            mk_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mk_client.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            models = await p._list_models_impl()
        by_id = {m.id: m for m in models}
        assert by_id["openai/x"].pricing.input == 1.0  # 0.001 * 1000
        assert by_id["openai/x"].pricing.output == 2.0  # 0.002 * 1000

    async def test_listing_filters_by_model_type(self) -> None:
        raw = [
            _raw_model("openai/gpt-4"),
            _raw_model(
                "openai/text-embedding-3", modality="text", supported=["embeddings"]
            ),
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"data": raw})
        with patch("httpx.AsyncClient") as mk_client:
            mk_client.return_value.__aenter__ = AsyncMock(
                return_value=mk_client.return_value
            )
            mk_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mk_client.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            llms = await p._list_models_impl(ModelType.LLM)
            embeds = await p._list_models_impl(ModelType.EMBED)
        assert [m.id for m in llms] == ["openai/gpt-4"]
        assert all(type(m) is LanguageModel for m in llms)
        assert [m.id for m in embeds] == ["openai/text-embedding-3"]
        assert all(type(m) is EmbedModel for m in embeds)


class TestListingReasoning:
    async def test_listing_sets_supports_reasoning(self) -> None:
        raw = [
            _raw_model(
                "reasoner/reasoner",
                supported=["tools", "reasoning", "reasoning_effort"],
            ),
            _raw_model("plain/llama", supported=["tools"]),
        ]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={"data": raw})
        with patch("httpx.AsyncClient") as mk_client:
            mk_client.return_value.__aenter__ = AsyncMock(
                return_value=mk_client.return_value
            )
            mk_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mk_client.return_value.get = AsyncMock(return_value=resp)
            p = _provider(_cfg())
            models = await p._list_models_impl()
        by_id = {m.id: m for m in models}
        assert by_id["reasoner/reasoner"].supports_reasoning is True
        assert by_id["plain/llama"].supports_reasoning is False


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


def _make_api_status_error(
    status_code: int,
    message: str = "API error",
) -> APIStatusError:
    """Construct an openai.APIStatusError with a stub httpx.Response."""
    url = "https://openrouter.ai/api/v1/test"
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
                await p.get_model("openai/gpt-4")

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
                await p.get_model("openai/gpt-4")

    async def test_stream_chat_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise _make_api_status_error(400, "bad request")

        p._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(side_effect=_fail))
            )
        )
        model = LanguageModel(id="openai/gpt-4", provider_id="openrouter")
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
        model = LanguageModel(id="openai/gpt-4", provider_id="openrouter")
        with pytest.raises(ModelInvocationError, match="unauthorized"):
            await p.static_chat(model, [])

    async def test_embed_sdk_error_wraps_as_model_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise _make_api_status_error(429, "rate limited")

        p._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=AsyncMock(side_effect=_fail))
        )
        embed_model = EmbedModel(id="openai/text-embedding-3", provider_id="openrouter")
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
        model = LanguageModel(id="openai/gpt-4", provider_id="openrouter")
        with pytest.raises(ModelInvocationError, match="connection failed"):
            [ev async for ev in p.stream_chat(model, [])]

    async def test_embed_httpx_error_wraps_as_model_invocation_error(self) -> None:
        p = _provider(_cfg())

        async def _fail(*a, **k):
            raise httpx.ConnectTimeout("timeout")

        p._client = SimpleNamespace(
            embeddings=SimpleNamespace(create=AsyncMock(side_effect=_fail))
        )
        embed_model = EmbedModel(id="openai/text-embedding-3", provider_id="openrouter")
        with pytest.raises(ModelInvocationError, match="timeout"):
            await p.embed(embed_model, "hi")
