import pytest
from pydantic import TypeAdapter, ValidationError

from yapa.models.inference import TokenUsage
from yapa.models.stream import (
    ContentDelta,
    ReasoningDelta,
    StreamEndEvent,
    StreamEvent,
    ToolCallDeltaEvent,
)


def test_content_delta_carries_content_only() -> None:
    ev = TypeAdapter(StreamEvent).validate_python({"type": "content", "content": "hi"})
    assert isinstance(ev, ContentDelta)
    assert ev.content == "hi"


def test_reasoning_delta_carries_content_only() -> None:
    ev = TypeAdapter(StreamEvent).validate_python(
        {"type": "reasoning", "content": "think"}
    )
    assert isinstance(ev, ReasoningDelta)
    assert ev.content == "think"


def test_tool_call_delta_fields() -> None:
    ev = TypeAdapter(StreamEvent).validate_python(
        {
            "type": "tool_call",
            "index": 0,
            "id": "call_1",
            "name": "calc",
            "arguments": '{"a":',
        }
    )
    assert isinstance(ev, ToolCallDeltaEvent)
    assert ev.index == 0
    assert ev.id == "call_1"
    assert ev.name == "calc"
    assert ev.arguments == '{"a":'


def test_stream_end_event_carries_finish_usage_model() -> None:
    ev = TypeAdapter(StreamEvent).validate_python(
        {
            "type": "stream_end",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "model_id": "gpt-4",
        }
    )
    assert isinstance(ev, StreamEndEvent)
    assert ev.finish_reason == "stop"
    assert isinstance(ev.usage, TokenUsage)
    assert ev.model_id == "gpt-4"


def test_unknown_event_type_rejected() -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(StreamEvent).validate_python({"type": "error", "message": "x"})
