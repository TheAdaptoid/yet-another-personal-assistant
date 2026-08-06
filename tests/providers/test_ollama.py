"""Tests for the Ollama provider implemented with the official SDK."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.models import (
    ContentDelta,
    EmbeddingResult,
    EmbedModel,
    LanguageModel,
    ModelType,
    ReasoningDelta,
    ReasoningEffort,
    StreamEndEvent,
    TokenUsage,
)
from yapa.providers.exceptions import ModelInvocationError, ModelsFetchError
from yapa.providers.ollama import OllamaIP
from yapa.services.config import Config, ProviderConfig


def _cfg(max_retries=2, **kw) -> Config:
    return Config(
        provider_configs={"ollama": ProviderConfig(**kw)},
        provider_max_retries=max_retries,
    )


async def _stream(*chunks):
    """Turn plain chunks into an async iteration for mock chat() streams."""
    for c in chunks:
        yield c


_ASYNC_CLIENT = "yapa.providers.ollama.provider.AsyncClient"


@pytest.fixture
def ollama_client():
    """Return a mocked ollama.AsyncClient wired into OllamaIP."""
    client = MagicMock()
    for name in ("chat", "embed", "list", "show"):
        setattr(client, name, AsyncMock())
    with patch(_ASYNC_CLIENT, return_value=client) as mk:
        provider = OllamaIP(_cfg())
    provider._client = client
    return provider, client, mk


def test_new_messages_serializes_assistant_tool_calls(ollama_client) -> None:
    from yapa.models import AssistantMessage, ToolCall

    provider, _, _ = ollama_client
    msg = AssistantMessage(
        content=None,
        tool_calls=[
            ToolCall(id="t1", tool_name="read_file", arguments={"path": "/x"})
        ],
    )
    out = provider._new_messages([msg])
    assert out[0]["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "/x"}}}
    ]


class TestReasoningMapping:
    def test_mapping(self) -> None:
        with patch(_ASYNC_CLIENT):
            p = OllamaIP(_cfg())
        assert p._map_reasoning(ReasoningEffort.OFF) == {"think": False}
        assert p._map_reasoning(ReasoningEffort.LOW) == {"think": True}
        assert p._map_reasoning(ReasoningEffort.MEDIUM) == {"think": True}
        assert p._map_reasoning(ReasoningEffort.HIGH) == {"think": True}
        assert p._map_reasoning(None) == {"think": False}


class TestStreaming:
    async def test_stream_emits_events_and_single_end(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        chunks = [
            {"message": {"role": "assistant", "content": "hi"}, "done": False},
            {"message": {"role": "assistant", "thinking": "think"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "eval_count": 4,
                "prompt_eval_count": 2,
            },
        ]
        client.chat.return_value = _stream(*chunks)
        evs = [
            ev async for ev in provider._stream_chat_impl("llama", [], None, None, None)
        ]
        assert any(isinstance(e, ContentDelta) and e.content == "hi" for e in evs)
        assert any(isinstance(e, ReasoningDelta) for e in evs)
        ends = [e for e in evs if isinstance(e, StreamEndEvent)]
        assert len(ends) == 1
        assert ends[0].usage == TokenUsage(
            prompt_tokens=2, completion_tokens=4, total_tokens=6
        )
        assert ends[0].finish_reason == "stop"

    async def test_stream_blank_thinking_becomes_none(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        chunks = [
            {"message": {"role": "assistant", "thinking": "  "}, "done": False},
            {"message": {"role": "assistant", "content": "x"}, "done": True},
        ]
        client.chat.return_value = _stream(*chunks)
        evs = [
            ev async for ev in provider._stream_chat_impl("llama", [], None, None, None)
        ]
        assert not any(isinstance(e, ReasoningDelta) for e in evs)

    async def test_stream_emits_tool_call_delta(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        chunks = [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": {"path": "/x"},
                            }
                        }
                    ],
                },
                "done": False,
            },
            {"message": {"role": "assistant"}, "done": True},
        ]
        client.chat.return_value = _stream(*chunks)
        evs = [
            ev async for ev in provider._stream_chat_impl("llama", [], None, None, None)
        ]
        from yapa.models import ToolCallDeltaEvent

        tcs = [e for e in evs if isinstance(e, ToolCallDeltaEvent)]
        assert tcs and tcs[0].name == "read_file"


class TestRetry:
    async def test_retries_transient_failure_up_to_max(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.chat.side_effect = [
            ConnectionError("transient"),
            ConnectionError("transient"),
            _stream({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        ]
        evs = [
            ev async for ev in provider._stream_chat_impl("llama", [], None, None, None)
        ]
        ends = [e for e in evs if isinstance(e, StreamEndEvent)]
        assert len(ends) == 1
        assert client.chat.call_count == 3  # max_retries(2) + 1

    async def test_non_retryable_not_retried(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.chat.side_effect = [RuntimeError("terminal")]
        with pytest.raises(RuntimeError, match="terminal"):
            [
                ev
                async for ev in provider._stream_chat_impl(
                    "llama", [], None, None, None
                )
            ]
        assert client.chat.call_count == 1

    async def test_retries_exhaust_after_max_attempts(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.embed.side_effect = [
            TimeoutError("boom"),
            TimeoutError("boom"),
            TimeoutError("boom"),
            {"embeddings": [[1.0]]},
        ]
        with pytest.raises(TimeoutError):
            await provider._embed_impl("embed", "hi")
        assert client.embed.call_count == 3  # never exceeds max_retries + 1


class TestEmbed:
    async def test_embed_maps_counts_to_usage(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.embed.return_value = {"embeddings": [[0.1, 0.2]], "prompt_eval_count": 4}
        result = await provider._embed_impl("embed", "hi")
        assert isinstance(result, EmbeddingResult)
        assert result.usage == TokenUsage(
            prompt_tokens=4, completion_tokens=0, total_tokens=4
        )

    async def test_embed_usage_none_without_counts(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.embed.return_value = {"embeddings": [[1.0]]}
        result = await provider._embed_impl("embed", "hi")
        assert result.usage is None


class TestModelListing:
    async def test_list_models_formats(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.list.return_value = {
            "models": [{"name": "llama3"}, {"name": "nomic-embed-text"}]
        }
        models = await provider._list_models_impl()
        by_id = {m.id: m for m in models}
        assert isinstance(by_id["llama3"], LanguageModel)
        assert by_id["llama3"].type == "llm"
        assert by_id["llama3"].provider_id == "ollama"
        assert by_id["nomic-embed-text"].type == "embedding"

    async def test_list_models_sets_supports_reasoning(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.list.return_value = {
            "models": [
                {"name": "qwen3.5", "capabilities": ["thinking", "tools"]},
                {"name": "llama3", "capabilities": ["completion"]},
            ]
        }
        models = await provider._list_models_impl()
        by_id = {m.id: m for m in models}
        assert by_id["qwen3.5"].supports_reasoning is True
        assert by_id["llama3"].supports_reasoning is False

    async def test_get_model_sets_supports_reasoning(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.list.return_value = {"models": [{"name": "qwen3.5"}]}
        client.show.return_value = {
            "template": "...",
            "parameters": {"num_ctx": 4096},
            "capabilities": ["thinking"],
        }
        model = await provider._get_model_impl("qwen3.5")
        assert isinstance(model, LanguageModel)
        assert model.supports_reasoning is True

    async def test_listing_filters_by_model_type(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        client.list.return_value = {
            "models": [
                {"name": "llama3"},
                {"name": "nomic-embed-text"},
            ]
        }
        llms = await provider._list_models_impl(ModelType.LLM)
        embeds = await provider._list_models_impl(ModelType.EMBED)
        assert [m.id for m in llms] == ["llama3"]
        assert all(m.type == "llm" for m in llms)
        assert [m.id for m in embeds] == ["nomic-embed-text"]
        assert all(m.type == "embedding" for m in embeds)
        provider, client, _ = ollama_client
        client.list.return_value = SimpleNamespace(
            models=[
                SimpleNamespace(name="llama3", model="llama3"),
                SimpleNamespace(name=None, model="nomic-embed-text"),
            ]
        )
        models = await provider._list_models_impl()
        by_id = {m.id: m for m in models}
        assert isinstance(by_id["llama3"], LanguageModel)
        assert by_id["llama3"].type == "llm"
        assert by_id["nomic-embed-text"].type == "embedding"

    async def test_get_model_missing_raises_models_fetch_error(
        self, ollama_client
    ) -> None:
        provider, client, _ = ollama_client
        client.list.return_value = {"models": [{"name": "llama3"}]}
        with pytest.raises(ModelsFetchError):
            await provider._get_model_impl("nonexistent")


class TestErrorConversion:
    """Tests that SDK/network exceptions are converted to typed provider errors."""

    async def test_list_models_sdk_error_wraps_as_models_fetch_error(
        self, ollama_client
    ) -> None:
        provider, client, _ = ollama_client
        client.list.side_effect = RuntimeError("list failed")
        with pytest.raises(ModelsFetchError, match="list failed"):
            await provider.list_models()

    async def test_get_model_sdk_error_wraps_as_models_fetch_error(
        self, ollama_client
    ) -> None:
        provider, client, _ = ollama_client
        client.list.side_effect = RuntimeError("list failed")
        with pytest.raises(ModelsFetchError, match="list failed"):
            await provider.get_model("llama3")

    async def test_stream_chat_sdk_error_wraps_as_model_invocation_error(
        self, ollama_client
    ) -> None:
        provider, client, _ = ollama_client

        async def _fail(*a, **k):
            raise RuntimeError("chat connection lost")

        client.chat.side_effect = _fail
        model = LanguageModel(id="llama", provider_id="ollama")
        with pytest.raises(ModelInvocationError, match="chat connection lost"):
            [ev async for ev in provider.stream_chat(model, [])]

    async def test_static_chat_sdk_error_wraps_as_model_invocation_error(
        self, ollama_client
    ) -> None:
        provider, client, _ = ollama_client

        async def _fail(*a, **k):
            raise RuntimeError("chat failed")

        client.chat.side_effect = _fail
        model = LanguageModel(id="llama", provider_id="ollama")
        with pytest.raises(ModelInvocationError, match="chat failed"):
            await provider.static_chat(model, [])

    async def test_embed_sdk_error_wraps_as_model_invocation_error(
        self, ollama_client
    ) -> None:
        provider, client, _ = ollama_client

        async def _fail(*a, **k):
            raise RuntimeError("embed timeout")

        client.embed.side_effect = _fail
        embed_model = EmbedModel(id="embed", provider_id="ollama")
        with pytest.raises(ModelInvocationError, match="embed timeout"):
            await provider.embed(embed_model, "hi")

    async def test_stream_chat_transient_error_retries(self, ollama_client) -> None:
        provider, client, _ = ollama_client
        call_count = {"n": 0}

        async def _fail_then_succeed(*a, **k):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ConnectionError("transient")
            return _stream(
                {"message": {"role": "assistant", "content": "ok"}, "done": True}
            )

        client.chat.side_effect = _fail_then_succeed
        evs = [
            ev async for ev in provider._stream_chat_impl("llama", [], None, None, None)
        ]
        ends = [e for e in evs if isinstance(e, StreamEndEvent)]
        assert len(ends) == 1
        assert call_count["n"] == 2
