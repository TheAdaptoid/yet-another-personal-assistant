"""Tests for Message discriminated union resolution."""

from pydantic import TypeAdapter

from yapa.models import AssistantMessage, Message, SystemMessage, UserMessage

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
