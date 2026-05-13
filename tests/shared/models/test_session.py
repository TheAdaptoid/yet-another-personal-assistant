import orjson
import pytest
from pydantic import TypeAdapter

from yapa.shared.models import Session
from yapa.shared.models.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)


class TestCreateSession:
    def test_default_title_when_no_args(self):
        session = Session.create()
        assert session.title == "New Session"

    def test_default_title_when_none(self):
        session = Session.create(title=None)
        assert session.title == "New Session"

    def test_default_title_when_empty_string(self):
        session = Session.create(title="")
        assert session.title == "New Session"

    def test_default_title_when_whitespace(self):
        session = Session.create(title="   ")
        assert session.title == "New Session"

    def test_custom_title(self):
        session = Session.create(title="My Chat")
        assert session.title == "My Chat"

    def test_unique_ids(self):
        s1 = Session.create()
        s2 = Session.create()
        assert s1.id != s2.id


class TestSessionDefaults:
    def test_default_title_constant(self):
        from yapa.shared.models.session import DEFAULT_SESSION_TITLE
        session = Session.create()
        assert session.title == DEFAULT_SESSION_TITLE

    def test_default_messages_empty(self):
        session = Session.create()
        assert session.messages == []

    def test_timestamps_positive(self):
        session = Session.create()
        assert session.created_at > 0
        assert session.updated_at > 0


class TestAddMessage:
    def test_add_message_append(self):
        session = Session.create()
        msg = UserMessage(content="Hello")
        session.add_message(msg)

        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello"

    def test_add_message_multiple(self):
        session = Session.create()
        session.add_message(UserMessage(content="First"))
        session.add_message(AssistantMessage(content="Second"))

        assert len(session.messages) == 2

    def test_add_message_updates_timestamp(self):
        session = Session.create()
        original_updated = session.updated_at
        session.add_message(UserMessage(content="Test"))

        assert session.updated_at >= original_updated

    def test_add_message_preserves_type(self):
        session = Session.create()
        session.add_message(UserMessage(content="user"))
        session.add_message(SystemMessage(content="system"))
        session.add_message(AssistantMessage(content="assistant"))

        adapter = TypeAdapter(Message)
        for i, msg in enumerate(session.messages):
            if i == 0:
                assert isinstance(msg, UserMessage)
            elif i == 1:
                assert isinstance(msg, SystemMessage)
            else:
                assert isinstance(msg, AssistantMessage)


class TestSerialization:
    def test_empty_messages_roundtrip(self):
        session = Session.create(title="Test Title")
        data = session.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        parsed = Session.model_validate(restored)

        assert parsed.title == "Test Title"
        assert parsed.messages == []

    def test_messages_list_roundtrip(self):
        session = Session.create(title="Chat")
        session.add_message(UserMessage(content="User message"))
        session.add_message(AssistantMessage(content="Assistant response", model="gpt-4o"))

        data = session.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        parsed = Session.model_validate(restored)

        assert len(parsed.messages) == 2
        assert isinstance(parsed.messages[0], UserMessage)
        assert isinstance(parsed.messages[1], AssistantMessage)
        assert parsed.messages[1].model == "gpt-4o"

    def test_mixed_message_types_roundtrip(self):
        session = Session.create()
        session.add_message(SystemMessage(content="System prompt", name="bot"))
        session.add_message(UserMessage(content="User input"))
        session.add_message(AssistantMessage(content="Response"))

        data = session.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        parsed = Session.model_validate(restored)

        assert isinstance(parsed.messages[0], SystemMessage)
        assert parsed.messages[0].name == "bot"
        assert isinstance(parsed.messages[1], UserMessage)
        assert isinstance(parsed.messages[2], AssistantMessage)