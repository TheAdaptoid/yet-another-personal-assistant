import pytest

from yapa.models import (
    AssistantMessage,
    EmbeddingResult,
    EmbedModel,
    LanguageModel,
    ModelData,
    ModelDataUnion,
    ModelType,
    ReasoningEffort,
    StreamEndEvent,
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

    async def _list_models_impl(self, model_type=None) -> list[ModelDataUnion]:
        self.last_list_model_type = model_type
        return [ModelData(id="test-model", provider_id=self.id, type=ModelType.LLM)]

    async def _get_model_impl(self, model_id) -> ModelDataUnion:
        self.last_get_model_id = model_id
        return ModelData(id=model_id, provider_id=self.id, type=ModelType.LLM)

    async def _stream_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        self.last_stream_model_id = model_id
        self.last_stream_reasoning = reasoning
        yield StreamEndEvent(finish_reason="stop")

    async def _static_chat_impl(
        self, model_id, messages, tools=None, params=None, reasoning=None
    ):
        self.last_static_model_id = model_id
        self.last_static_reasoning = reasoning
        return AssistantMessage(content="response", role="assistant")

    async def _embed_impl(self, model_id, input):
        self.last_embed_model_id = model_id
        self.last_embed_input = input
        return EmbeddingResult(vectors=[[1.0]], model_id=model_id)


@pytest.fixture
def provider():
    return _TestProvider()


def _llm():
    return LanguageModel(id="gpt-4", provider_id="test")


def _embed():
    return EmbedModel(id="embed", provider_id="test")


class TestListModels:
    async def test_delegates(self, provider):
        result = await provider.list_models()
        assert result[0].id == "test-model"

    async def test_passes_model_type(self, provider):
        await provider.list_models(model_type=ModelType.LLM)
        assert provider.last_list_model_type == ModelType.LLM

    async def test_wraps_exception(self, provider):
        async def _fail(model_type=None):
            raise RuntimeError("API error")

        provider._list_models_impl = _fail  # type: ignore
        with pytest.raises(ModelsFetchError, match="API error"):
            await provider.list_models()


class TestGetModel:
    async def test_delegates(self, provider):
        result = await provider.get_model("gpt-4")
        assert result.id == "gpt-4"

    async def test_wraps_exception(self, provider):
        async def _fail(model_id):
            raise RuntimeError("fetch failed")

        provider._get_model_impl = _fail  # type: ignore
        with pytest.raises(ModelsFetchError, match="fetch failed"):
            await provider.get_model("gpt-4")


class TestStreamChat:
    async def test_delegates(self, provider):
        out = [ev async for ev in provider.stream_chat(_llm(), [])]
        assert len(out) == 1
        assert isinstance(out[0], StreamEndEvent)
        assert provider.last_stream_model_id == "gpt-4"

    async def test_receives_reasoning(self, provider):
        async for _ev in provider.stream_chat(
            _llm(), [], reasoning=ReasoningEffort.HIGH
        ):
            pass
        assert provider.last_stream_reasoning == ReasoningEffort.HIGH

    async def test_raises_for_embed_model(self, provider):
        with pytest.raises(ModelTypeError):
            [ev async for ev in provider.stream_chat(_embed(), [])]

    async def test_raises_for_wrong_provider_id(self, provider):
        model = LanguageModel(id="gpt-4", provider_id="other")
        with pytest.raises(ModelTypeError, match="provider"):
            [ev async for ev in provider.stream_chat(model, [])]

    async def test_wraps_exception(self, provider):
        async def _fail(model_id, messages, tools=None, params=None, reasoning=None):
            raise RuntimeError("stream failed")
            yield  # pragma: no cover

        provider._stream_chat_impl = _fail  # type: ignore
        with pytest.raises(ModelInvocationError, match="stream failed"):
            [ev async for ev in provider.stream_chat(_llm(), [])]


class TestStaticChat:
    async def test_delegates(self, provider):
        result = await provider.static_chat(_llm(), [])
        assert result.content == "response"
        assert provider.last_static_model_id == "gpt-4"

    async def test_receives_reasoning(self, provider):
        await provider.static_chat(_llm(), [], reasoning=ReasoningEffort.LOW)
        assert provider.last_static_reasoning == ReasoningEffort.LOW

    async def test_raises_for_embed_model(self, provider):
        with pytest.raises(ModelTypeError):
            await provider.static_chat(_embed(), [])

    async def test_raises_for_wrong_provider_id(self, provider):
        model = LanguageModel(id="gpt-4", provider_id="other")
        with pytest.raises(ModelTypeError):
            await provider.static_chat(model, [])

    async def test_wraps_exception(self, provider):
        async def _fail(model_id, messages, tools=None, params=None, reasoning=None):
            raise RuntimeError("invocation failed")

        provider._static_chat_impl = _fail  # type: ignore
        with pytest.raises(ModelInvocationError, match="invocation failed"):
            await provider.static_chat(_llm(), [])


class TestEmbed:
    async def test_delegates(self, provider):
        model = EmbedModel(id="embed", provider_id="test")
        result = await provider.embed(model, "hello")
        assert result.vectors == [[1.0]]
        assert provider.last_embed_input == "hello"

    async def test_raises_for_language_model(self, provider):
        with pytest.raises(ModelTypeError):
            await provider.embed(_llm(), "hello")

    async def test_raises_for_wrong_provider_id(self, provider):
        model = EmbedModel(id="embed", provider_id="other")
        with pytest.raises(ModelTypeError):
            await provider.embed(model, "hello")

    async def test_wraps_exception(self, provider):
        async def _fail(model_id, input):
            raise RuntimeError("embed failed")

        provider._embed_impl = _fail  # type: ignore
        model = EmbedModel(id="embed", provider_id="test")
        with pytest.raises(ModelInvocationError, match="embed failed"):
            await provider.embed(model, "hello")
