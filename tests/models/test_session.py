"""Tests for Session model — serialization and message discrimination."""

from yapa.models import (
    AssistantMessage,
    Session,
    SystemMessage,
    UserMessage,
)


class TestJsonRoundTrip:
    """JSON round-trip preserves Session fields and embedded messages."""

    def test_messages_serialize_as_array(self):
        session = Session(title="test")
        session.messages.append(UserMessage(content="hello"))
        data = session.model_dump(mode="json")
        assert data["title"] == "test"
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "hello"

    def test_round_trip(self):
        session = Session(title="chat")
        session.messages.append(UserMessage(content="hi"))
        session.messages.append(AssistantMessage(content="hello", model="m"))
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert restored.title == "chat"
        assert len(restored.messages) == 2
        assert isinstance(restored.messages[0], UserMessage)
        assert isinstance(restored.messages[1], AssistantMessage)

    def test_discriminated_messages(self):
        session = Session()
        session.messages.append(SystemMessage(content="system prompt"))
        session.messages.append(UserMessage(content="user text"))
        session.messages.append(AssistantMessage(content="response", model="m"))
        data = session.model_dump(mode="json")
        restored = Session(**data)
        assert isinstance(restored.messages[0], SystemMessage)
        assert isinstance(restored.messages[1], UserMessage)
        assert isinstance(restored.messages[2], AssistantMessage)
