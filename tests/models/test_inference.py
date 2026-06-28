"""Tests for inference models — serialization, validation, and immutability."""

import pytest
from pydantic import ValidationError

from yapa.models import (
    InferenceParams,
    ModelData,
    ModelType,
    StreamDelta,
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
        sd = StreamDelta(content="hello", reasoning_content="thinking", done=True)
        data = sd.model_dump(mode="json")
        restored = StreamDelta(**data)
        assert restored.content == "hello"
        assert restored.reasoning_content == "thinking"
        assert restored.done is True


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
