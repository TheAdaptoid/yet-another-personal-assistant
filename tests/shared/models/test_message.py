import orjson
import pytest
from pydantic import TypeAdapter, ValidationError

from yapa.shared.models.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)


class TestCreateUserMessage:
    def test_returns_correct_type_and_role(self):
        msg = UserMessage(content="hello")
        assert isinstance(msg, UserMessage)
        assert msg.role == "user"

    def test_sets_content(self):
        msg = UserMessage(content="test content")
        assert msg.content == "test content"

    def test_sets_id_and_timestamp(self):
        msg = UserMessage(content="hello")
        assert len(msg.id) == 32  # uuid4().hex is 32 chars
        assert isinstance(msg.timestamp, int)
        assert msg.timestamp > 0


class TestCreateSystemMessage:
    def test_returns_correct_type_and_role(self):
        msg = SystemMessage(content="system prompt")
        assert isinstance(msg, SystemMessage)
        assert msg.role == "system"

    def test_without_name_defaults_to_none(self):
        msg = SystemMessage(content="be helpful")
        assert msg.name is None

    def test_with_name(self):
        msg = SystemMessage(content="you are a bot", name="assistant")
        assert msg.name == "assistant"


class TestCreateAssistantMessage:
    def test_returns_correct_type_and_role(self):
        msg = AssistantMessage(content="I think therefore I am")
        assert isinstance(msg, AssistantMessage)
        assert msg.role == "assistant"

    def test_without_model_defaults_to_none(self):
        msg = AssistantMessage(content="hello")
        assert msg.model is None

    def test_with_model(self):
        msg = AssistantMessage(content="hello", model="gpt-4o")
        assert msg.model == "gpt-4o"


class TestUniqueIds:
    def test_consecutive_messages_have_different_ids(self):
        msg1 = UserMessage(content="first")
        msg2 = UserMessage(content="second")
        assert msg1.id != msg2.id


class TestDiscriminatedUnion:
    def test_parses_user_message(self):
        adapter = TypeAdapter(Message)
        parsed = adapter.validate_python({"role": "user", "content": "hi"})
        assert isinstance(parsed, UserMessage)
        assert parsed.content == "hi"

    def test_parses_system_message(self):
        adapter = TypeAdapter(Message)
        parsed = adapter.validate_python(
            {"role": "system", "content": "system prompt", "name": "bot"}
        )
        assert isinstance(parsed, SystemMessage)
        assert parsed.content == "system prompt"
        assert parsed.name == "bot"

    def test_parses_assistant_message(self):
        adapter = TypeAdapter(Message)
        parsed = adapter.validate_python(
            {"role": "assistant", "content": "response", "model": "gpt-4o"}
        )
        assert isinstance(parsed, AssistantMessage)
        assert parsed.content == "response"
        assert parsed.model == "gpt-4o"

    def test_invalid_role_raises_error(self):
        adapter = TypeAdapter(Message)
        with pytest.raises(ValidationError):
            adapter.validate_python({"role": "invalid", "content": "hello"})


class TestSerialization:
    def test_roundtrip_user_message(self):
        msg = UserMessage(content="Hello, world!")
        data = msg.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        adapter = TypeAdapter(Message)
        parsed = adapter.validate_python(restored)

        assert isinstance(parsed, UserMessage)
        assert parsed.content == "Hello, world!"
        assert parsed.id == msg.id
        assert parsed.timestamp == msg.timestamp

    def test_roundtrip_system_message_with_name(self):
        msg = SystemMessage(content="be helpful", name="bot")
        data = msg.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        adapter = TypeAdapter(Message)
        parsed = adapter.validate_python(restored)

        assert isinstance(parsed, SystemMessage)
        assert parsed.content == "be helpful"
        assert parsed.name == "bot"

    def test_roundtrip_assistant_message_with_model(self):
        msg = AssistantMessage(content="response", model="claude-3")
        data = msg.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        adapter = TypeAdapter(Message)
        parsed = adapter.validate_python(restored)

        assert isinstance(parsed, AssistantMessage)
        assert parsed.content == "response"
        assert parsed.model == "claude-3"

    def test_mixed_list_roundtrip(self):
        messages = [
            UserMessage(content="User message"),
            SystemMessage(content="System message", name="bot"),
            AssistantMessage(content="Assistant response", model="gpt-4o"),
        ]

        data = [m.model_dump(mode="python") for m in messages]
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        adapter = TypeAdapter(list[Message])
        parsed = adapter.validate_python(restored)

        assert len(parsed) == 3
        assert isinstance(parsed[0], UserMessage)
        assert isinstance(parsed[1], SystemMessage)
        assert isinstance(parsed[2], AssistantMessage)
        assert parsed[0].content == "User message"
        assert parsed[1].name == "bot"
        assert parsed[2].model == "gpt-4o"


class TestEdgeCases:
    def test_empty_content(self):
        msg = UserMessage(content="")
        assert msg.content == ""

    def test_unicode_content(self):
        content = "你好 Hallo 😊"
        msg = UserMessage(content=content)
        data = msg.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        adapter = TypeAdapter(Message)
        parsed = adapter.validate_python(restored)

        assert parsed.content == content
