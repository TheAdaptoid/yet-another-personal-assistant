"""Tests for inference models — serialization, validation, and immutability."""

import pytest
from pydantic import ValidationError

from yapa.models import (
    InferenceParams,
    ModelData,
    ModelType,
    TokenUsage,
)
from yapa.models.inference import LanguageModel, ModelPricing


class TestModelData:
    """ModelData — JSON round-trip, full_id, frozen, extra forbidden."""

    def test_json_round_trip(self):
        md = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        data = md.model_dump(mode="json")
        restored = ModelData(**data)
        assert restored.id == "gpt-4"
        assert restored.provider_id == "openai"
        assert restored.type == ModelType.LLM

    def test_full_id_property(self):
        md = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        assert md.full_id == "openai:gpt-4"

    def test_immutable(self):
        md = ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM)
        with pytest.raises(ValidationError):
            md.id = "new"

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            ModelData(id="gpt-4", provider_id="openai", type=ModelType.LLM, extra="x")


class TestStreamEvent:
    """StreamEvent discriminated union — parse and round-trip each variant."""

    def test_content_delta_parses(self):
        from pydantic import TypeAdapter

        from yapa.models.stream import ContentDelta, StreamEvent

        adapter = TypeAdapter(StreamEvent)
        ev = adapter.validate_python({"type": "content", "content": "hello"})
        assert isinstance(ev, ContentDelta)
        assert ev.content == "hello"

    def test_stream_end_event_parses_with_usage(self):
        from pydantic import TypeAdapter

        from yapa.models.stream import StreamEndEvent, StreamEvent

        adapter = TypeAdapter(StreamEvent)
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        ev = adapter.validate_python(
            {"type": "stream_end", "finish_reason": "stop", "usage": usage}
        )
        assert isinstance(ev, StreamEndEvent)
        assert ev.finish_reason == "stop"
        assert ev.usage.total_tokens == 30

    def test_union_rejects_unknown_type(self):
        from pydantic import TypeAdapter, ValidationError

        from yapa.models.stream import StreamEvent

        adapter = TypeAdapter(StreamEvent)
        with pytest.raises(ValidationError):
            adapter.validate_python({"type": "error", "message": "x"})


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


class TestLanguageModelMetadata:
    """LanguageModel — LLM-specific optional metadata fields."""

    def test_defaults(self):
        md = LanguageModel(id="gpt-4", provider_id="openai")
        assert md.context_length is None
        assert md.max_output is None
        assert md.supports_tools is False
        assert md.supports_vision is False
        assert md.supports_reasoning is False
        assert md.pricing is None

    def test_can_set_metadata(self):
        md = LanguageModel(
            id="gpt-4o",
            provider_id="openai",
            context_length=128000,
            max_output=16384,
            supports_tools=True,
            supports_vision=True,
            pricing=ModelPricing(input=2.50, output=10.00),
        )
        assert md.context_length == 128000
        assert md.supports_tools is True
        assert md.pricing == ModelPricing(input=2.50, output=10.00)

    def test_json_round_trip_with_metadata(self):
        md = LanguageModel(
            id="gpt-4o",
            provider_id="openai",
            context_length=128000,
        )
        data = md.model_dump(mode="json")
        restored = LanguageModel(**data)
        assert restored.context_length == 128000


class TestInferenceParams:
    """InferenceParams — field validation."""

    def test_all_fields_default_to_none(self):
        params = InferenceParams()
        fields = (
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
        )
        assert {name: getattr(params, name) for name in fields} == {
            name: None for name in fields
        }

    def test_all_unset_serializes_to_empty_dict(self):
        assert InferenceParams().model_dump(exclude_none=True) == {}

    def test_subset_serializes_only_set_fields(self):
        params = InferenceParams(temperature=0.7, stop="END")
        assert params.model_dump(exclude_none=True) == {
            "temperature": 0.7,
            "stop": "END",
        }

    def test_stop_accepts_string_or_list(self):
        assert InferenceParams(stop="END").stop == "END"
        assert InferenceParams(stop=["END", "STOP"]).stop == ["END", "STOP"]

    def test_has_exact_curated_fields_without_reasoning(self):
        expected = {
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
        }
        fields = InferenceParams.model_fields
        assert set(fields) == expected
        assert "reasoning" not in fields
        assert "reasoning_effort" not in fields

    def test_rejects_out_of_range(self):
        with pytest.raises(ValidationError):
            InferenceParams(temperature=-1)
        with pytest.raises(ValidationError):
            InferenceParams(temperature=3)
        with pytest.raises(ValidationError):
            InferenceParams(max_tokens=0)
        with pytest.raises(ValidationError):
            InferenceParams(top_p=-1)
        with pytest.raises(ValidationError):
            InferenceParams(top_p=2)

    def test_json_round_trip(self):
        params = InferenceParams(temperature=0.5, max_tokens=100, top_p=0.9)
        data = params.model_dump(mode="json")
        restored = InferenceParams(**data)
        assert restored.temperature == 0.5
        assert restored.max_tokens == 100
        assert restored.top_p == 0.9
