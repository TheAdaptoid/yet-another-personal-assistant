"""Tests for InferenceProvider base class."""

from unittest.mock import AsyncMock

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
)
from yapa.providers.exceptions import ModelInvocationError, ModelsFetchError


class TestInit:
    """Tests for InferenceProvider.__init__()."""

    def test_stores_attributes(
        self, provider, mock_model_fetcher, mock_model_invoker
    ) -> None:
        assert provider._identifier == "test_prov"
        assert provider._name == "Test Provider"
        assert provider.id == "test_prov"
        assert provider.name == "Test Provider"
        assert provider._model_fetcher is mock_model_fetcher
        assert provider._model_invoker is mock_model_invoker


class TestListModels:
    """Tests for InferenceProvider.list_models()."""

    async def test_delegates_to_fetcher(
        self, provider, mock_model_fetcher
    ) -> None:
        expected = [
            ModelData(id="gpt-4", provider_id="test_prov", type=ModelType.LLM),
        ]
        mock_model_fetcher.list_models = AsyncMock(return_value=expected)

        result = await provider.list_models()

        assert result == expected
        mock_model_fetcher.list_models.assert_awaited_once_with(model_type=None)

    async def test_passes_model_type(
        self, provider, mock_model_fetcher
    ) -> None:
        await provider.list_models(model_type=ModelType.LLM)
        mock_model_fetcher.list_models.assert_awaited_once_with(
            model_type=ModelType.LLM
        )

    async def test_raises_models_fetch_error_on_failure(
        self, provider, mock_model_fetcher
    ) -> None:
        api_error = Exception("API failure")
        mock_model_fetcher.list_models = AsyncMock(side_effect=api_error)

        with pytest.raises(ModelsFetchError) as exc_info:
            await provider.list_models()

        assert "API failure" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_error


class TestGetModel:
    """Tests for InferenceProvider.get_model()."""

    async def test_delegates_to_fetcher(
        self, provider, mock_model_fetcher
    ) -> None:
        expected = ModelData(id="gpt-4", provider_id="test_prov", type=ModelType.LLM)
        mock_model_fetcher.get_model = AsyncMock(return_value=expected)

        result = await provider.get_model(model_id="gpt-4")

        assert result == expected
        mock_model_fetcher.get_model.assert_awaited_once_with(model_id="gpt-4")

    async def test_raises_models_fetch_error_on_failure(
        self, provider, mock_model_fetcher
    ) -> None:
        api_error = Exception("not found")
        mock_model_fetcher.get_model = AsyncMock(side_effect=api_error)

        with pytest.raises(ModelsFetchError) as exc_info:
            await provider.get_model(model_id="gpt-4")

        assert "not found" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_error


class TestInvokeLlmStream:
    """Tests for InferenceProvider.invoke_llm_stream()."""

    async def test_raises_error_on_non_llm_model(
        self, provider, sample_messages
    ) -> None:
        model = ModelData(
            id="text-embedding-3", provider_id="test_prov", type=ModelType.OTHER
        )
        with pytest.raises(ModelInvocationError, match="not an LLM"):
            async for _ in provider.invoke_llm_stream(model, sample_messages):
                pass

    async def test_streams_content_from_invoker(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        results: list[StreamDelta] = []
        async for chunk in provider.invoke_llm_stream(sample_model, sample_messages):
            results.append(chunk)

        assert results[0] == StreamDelta(
            content="Hello", reasoning_content=None, done=False
        )

    async def test_yields_final_done_delta(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        results: list[StreamDelta] = []
        async for chunk in provider.invoke_llm_stream(sample_model, sample_messages):
            results.append(chunk)

        assert results[-1] == StreamDelta(
            content=None, reasoning_content=None, done=True
        )

    async def test_passes_model_id_and_messages_to_invoker(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        captured = None

        async def _gen(model_id, messages, params=None):
            nonlocal captured
            captured = (model_id, messages, params)
            yield StreamDelta(content=None, done=False)
            yield StreamDelta(content=None, done=True)

        mock_model_invoker.invoke_llm_stream = _gen

        async for _ in provider.invoke_llm_stream(sample_model, sample_messages):
            pass

        assert captured is not None
        assert captured[0] == sample_model.id
        assert captured[1] == sample_messages
        assert captured[2] is None

    async def test_passes_inference_params(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        captured = None

        async def _gen(model_id, messages, params=None):
            nonlocal captured
            captured = (model_id, messages, params)
            yield StreamDelta(content=None, done=False)
            yield StreamDelta(content=None, done=True)

        mock_model_invoker.invoke_llm_stream = _gen
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        async for _ in provider.invoke_llm_stream(
            sample_model, sample_messages, params=params
        ):
            pass

        assert captured is not None
        assert captured[0] == sample_model.id
        assert captured[1] == sample_messages
        assert captured[2] == params

    async def test_raises_model_invocation_error_on_failure(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        api_error = Exception("stream failed")

        async def _fail(*_args, **_kwargs):
            raise api_error
            yield  # pragma: no cover

        mock_model_invoker.invoke_llm_stream = _fail

        with pytest.raises(ModelInvocationError) as exc_info:
            async for _ in provider.invoke_llm_stream(sample_model, sample_messages):
                pass

        assert "stream failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_error


class TestInvokeLlm:
    """Tests for InferenceProvider.invoke_llm() (non-streaming)."""

    async def test_raises_error_on_non_llm_model(
        self, provider, sample_messages
    ) -> None:
        model = ModelData(
            id="text-embedding-3", provider_id="test_prov", type=ModelType.OTHER
        )
        with pytest.raises(ModelInvocationError, match="not an LLM"):
            await provider.invoke_llm(model, sample_messages)

    async def test_returns_assistant_message(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        expected = AssistantMessage(content="Hello", role="assistant")
        mock_model_invoker.invoke_llm = AsyncMock(return_value=expected)

        result = await provider.invoke_llm(sample_model, sample_messages)

        assert result == expected

    async def test_passes_model_id_messages_and_params(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        mock_model_invoker.invoke_llm = AsyncMock(
            return_value=AssistantMessage(content="ok", role="assistant")
        )
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)

        await provider.invoke_llm(sample_model, sample_messages, params=params)

        mock_model_invoker.invoke_llm.assert_awaited_once_with(
            model_id=sample_model.id,
            messages=sample_messages,
            params=params,
        )

    async def test_raises_model_invocation_error_on_failure(
        self, provider, mock_model_invoker, sample_model, sample_messages
    ) -> None:
        api_error = Exception("invocation failed")
        mock_model_invoker.invoke_llm = AsyncMock(side_effect=api_error)

        with pytest.raises(ModelInvocationError) as exc_info:
            await provider.invoke_llm(sample_model, sample_messages)

        assert "invocation failed" in str(exc_info.value)
        assert exc_info.value.__cause__ is api_error
