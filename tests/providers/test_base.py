"""Tests for InferenceProvider base class."""

import pytest

from yapa.models import (
    AssistantMessage,
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
)
from yapa.providers.base import InferenceProvider
from yapa.providers.exceptions import (
    ModelInvocationError,
    ModelsFetchError,
    ModelTypeError,
)


class _TestProvider(InferenceProvider):
    """Minimal concrete provider for testing the base class."""

    def __init__(self, identifier: str = "test", name: str = "Test Provider") -> None:
        super().__init__(identifier, name)

    async def _list_models_impl(
        self, model_type: ModelType | None = None
    ) -> list[ModelData]:
        self.last_list_model_type = model_type
        return [ModelData(id="test-model", provider_id=self.id, type=ModelType.LLM)]

    async def _get_model_impl(self, model_id: str) -> ModelData:
        self.last_get_model_id = model_id
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(
        self,
        model_id: str,
        messages: list,
        tools=None,
        params=None,
    ):
        self.last_stream_model_id = model_id
        self.last_stream_messages = messages
        self.last_stream_tools = tools
        self.last_stream_params = params
        yield StreamDelta(content="hello")

    async def _static_chat_impl(
        self,
        model_id: str,
        messages: list,
        tools=None,
        params=None,
    ) -> AssistantMessage:
        self.last_static_model_id = model_id
        self.last_static_messages = messages
        self.last_static_tools = tools
        self.last_static_params = params
        return AssistantMessage(content="response", role="assistant")


class TestProperties:
    """Tests for InferenceProvider properties."""

    def test_id(self) -> None:
        provider = _TestProvider(identifier="my_id", name="My Name")
        assert provider.id == "my_id"

    def test_name(self) -> None:
        provider = _TestProvider(identifier="my_id", name="My Name")
        assert provider.name == "My Name"


class TestListModels:
    """Tests for InferenceProvider.list_models()."""

    @pytest.fixture
    def provider(self) -> _TestProvider:
        return _TestProvider()

    async def test_delegates_to_impl(self, provider: _TestProvider) -> None:
        result = await provider.list_models()
        assert len(result) == 1
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

    async def test_passes_through_models_fetch_error(
        self, provider: _TestProvider
    ) -> None:
        async def _fail(model_type=None):
            raise ModelsFetchError("original")

        provider._list_models_impl = _fail  # type: ignore
        with pytest.raises(ModelsFetchError, match="original"):
            await provider.list_models()


class TestGetModel:
    """Tests for InferenceProvider.get_model()."""

    @pytest.fixture
    def provider(self) -> _TestProvider:
        return _TestProvider()

    async def test_delegates_to_impl(self, provider: _TestProvider) -> None:
        result = await provider.get_model("gpt-4")
        assert result.id == "gpt-4"

    async def test_passes_model_id(self, provider: _TestProvider) -> None:
        await provider.get_model("gpt-4")
        assert provider.last_get_model_id == "gpt-4"

    async def test_wraps_exception(self, provider: _TestProvider) -> None:
        async def _fail(model_id):
            raise RuntimeError("fetch failed")

        provider._get_model_impl = _fail  # type: ignore
        with pytest.raises(ModelsFetchError, match="fetch failed"):
            await provider.get_model("gpt-4")

    async def test_passes_through_models_fetch_error(
        self, provider: _TestProvider
    ) -> None:
        async def _fail(model_id):
            raise ModelsFetchError("original")

        provider._get_model_impl = _fail  # type: ignore
        with pytest.raises(ModelsFetchError, match="original"):
            await provider.get_model("gpt-4")


class TestStreamChat:
    """Tests for InferenceProvider.stream_chat()."""

    @pytest.fixture
    def provider(self) -> _TestProvider:
        return _TestProvider()

    async def test_delegates_to_impl(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        results: list[StreamDelta] = []
        async for delta in provider.stream_chat(model, sample_messages):
            results.append(delta)
        assert len(results) == 1
        assert results[0].content == "hello"

    async def test_passes_arguments(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)
        async for _ in provider.stream_chat(model, sample_messages, params=params):
            pass
        assert provider.last_stream_model_id == "gpt-4"
        assert provider.last_stream_params == params

    async def test_raises_model_type_error_for_non_llm(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        model = ModelData(id="embed-3", provider_id="test", type=ModelType.OTHER)
        with pytest.raises(ModelTypeError, match="not an LLM"):
            async for _ in provider.stream_chat(model, sample_messages):
                pass

    async def test_wraps_exception(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        async def _fail(model_id, messages, tools=None, params=None):
            raise RuntimeError("stream failed")
            yield  # pragma: no cover

        provider._stream_chat_impl = _fail  # type: ignore
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        with pytest.raises(ModelInvocationError, match="stream failed"):
            async for _ in provider.stream_chat(model, sample_messages):
                pass

    async def test_passes_through_model_invocation_error(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        async def _fail(model_id, messages, tools=None, params=None):
            raise ModelInvocationError("original")
            yield  # pragma: no cover

        provider._stream_chat_impl = _fail  # type: ignore
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        with pytest.raises(ModelInvocationError, match="original"):
            async for _ in provider.stream_chat(model, sample_messages):
                pass


class TestStaticChat:
    """Tests for InferenceProvider.static_chat()."""

    @pytest.fixture
    def provider(self) -> _TestProvider:
        return _TestProvider()

    async def test_delegates_to_impl(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        result = await provider.static_chat(model, sample_messages)
        assert result.content == "response"

    async def test_passes_arguments(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        params = InferenceParams(temperature=0.7, max_tokens=100, top_p=0.9)
        await provider.static_chat(model, sample_messages, params=params)
        assert provider.last_static_model_id == "gpt-4"
        assert provider.last_static_params == params

    async def test_raises_model_type_error_for_non_llm(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        model = ModelData(id="embed-3", provider_id="test", type=ModelType.OTHER)
        with pytest.raises(ModelTypeError, match="not an LLM"):
            await provider.static_chat(model, sample_messages)

    async def test_wraps_exception(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        async def _fail(model_id, messages, tools=None, params=None):
            raise RuntimeError("invocation failed")

        provider._static_chat_impl = _fail  # type: ignore
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        with pytest.raises(ModelInvocationError, match="invocation failed"):
            await provider.static_chat(model, sample_messages)

    async def test_passes_through_model_invocation_error(
        self, provider: _TestProvider, sample_messages: list
    ) -> None:
        async def _fail(model_id, messages, tools=None, params=None):
            raise ModelInvocationError("original")

        provider._static_chat_impl = _fail  # type: ignore
        model = ModelData(id="gpt-4", provider_id="test", type=ModelType.LLM)
        with pytest.raises(ModelInvocationError, match="original"):
            await provider.static_chat(model, sample_messages)
