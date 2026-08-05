"""Tests for Message discriminated union resolution."""

import pytest
from pydantic import TypeAdapter, ValidationError

from yapa.models import (
    AssistantMessage,
    Message,
    SystemMessage,
    TokenUsage,
    UserMessage,
)
from yapa.models.message import ImagePart, TextPart

_adapter = TypeAdapter(Message)


class TestDiscriminator:
    """Message union resolves to the correct type based on role."""

    def test_resolves_user(self):
        msg = _adapter.validate_python({"role": "user", "content": "hello"})
        assert isinstance(msg, UserMessage)

    def test_resolves_system(self):
        msg = _adapter.validate_python({"role": "system", "content": "instruction"})
        assert isinstance(msg, SystemMessage)

    def test_resolves_assistant(self):
        msg = _adapter.validate_python(
            {"role": "assistant", "content": "response", "model": "gpt-4"}
        )
        assert isinstance(msg, AssistantMessage)
        assert msg.model == "gpt-4"

    def test_assistant_model_round_trip(self):
        msg = AssistantMessage(content="hi", model="gpt-4")
        data = msg.model_dump(mode="json")
        restored = _adapter.validate_python(data)
        assert isinstance(restored, AssistantMessage)
        assert restored.model == "gpt-4"

    def test_assistant_message_with_usage(self):
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        msg = AssistantMessage(content="hi", usage=usage)
        data = msg.model_dump(mode="json")
        restored = _adapter.validate_python(data)
        assert isinstance(restored, AssistantMessage)
        assert restored.usage is not None
        assert restored.usage.total_tokens == 30


def test_plain_string_message_parses() -> None:
    m = UserMessage(content="hello")
    assert m.content == "hello"


def test_mixed_content_parts_parse_and_round_trip() -> None:
    m = UserMessage(
        content=[
            TextPart(type="text", text="What is this?"),
            ImagePart(type="image_url", image_url={"url": "data:image/png;base64,AA"}),
        ]
    )
    assert isinstance(m.content, list)
    assert m.content[0].text == "What is this?"
    assert m.content[1].image_url.url == "data:image/png;base64,AA"
    dumped = m.model_dump()
    assert dumped["content"][1]["type"] == "image_url"


def test_unknown_part_type_fails_validation() -> None:
    with pytest.raises(ValidationError):
        UserMessage(content=[{"type": "video", "url": "x"}])
