"""Tests for inference models — serialization, validation, and immutability."""

import pytest
from pydantic import ValidationError

from yapa.models import (
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
    TokenUsage,
)


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


class TestStreamDelta:
    """StreamDelta — JSON round-trip."""

    def test_json_round_trip(self):
        sd = StreamDelta(content="hello", reasoning_content="thinking")
        data = sd.model_dump(mode="json")
        restored = StreamDelta(**data)
        assert restored.content == "hello"
        assert restored.reasoning_content == "thinking"


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


class TestInferenceParams:
    """InferenceParams — field validation."""

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
