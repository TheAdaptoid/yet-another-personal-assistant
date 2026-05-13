import orjson
import pytest
from pydantic import TypeAdapter

from yapa.shared.models import Session, create_session
from yapa.shared.models.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
    create_assistant_message,
    create_system_message,
    create_user_message,
)


class TestCreateSession:
    def test_default_title_when_no_args(self):
        session = create_session()
        assert session.title == "New Session"

    def test_default_title_when_none(self):
        session = create_session(None)
        assert session.title == "New Session"

    def test_default_title_when_empty_string(self):
        session = create_session("")
        assert session.title == "New Session"

    def test_default_title_when_whitespace(self):
        session = create_session("   ")
        assert session.title == "New Session"

    def test_custom_title(self):
        session = create_session("My Chat")
        assert session.title == "My Chat"

    def test_unique_ids(self):
        s1 = create_session()
        s2 = create_session()
        assert s1.id != s2.id


class TestSessionDefaults:
    def test_default_title_constant(self):
        from yapa.shared.models.session import DEFAULT_SESSION_TITLE
        session = Session()
        assert session.title == DEFAULT_SESSION_TITLE

    def test_default_messages_empty(self):
        session = Session()
        assert session.messages == []

    def test_timestamps_positive(self):
        session = Session()
        assert session.created_at > 0
        assert session.updated_at > 0


class TestAddMessage:
    def test_add_message_append(self):
        session = create_session()
        msg = create_user_message("Hello")
        session.add_message(msg)

        assert len(session.messages) == 1
        assert session.messages[0].content == "Hello"

    def test_add_message_multiple(self):
        session = create_session()
        session.add_message(create_user_message("First"))
        session.add_message(create_assistant_message("Second"))

        assert len(session.messages) == 2

    def test_add_message_updates_timestamp(self):
        session = create_session()
        original_updated = session.updated_at
        session.add_message(create_user_message("Test"))

        assert session.updated_at >= original_updated

    def test_add_message_preserves_type(self):
        session = create_session()
        session.add_message(create_user_message("user"))
        session.add_message(create_system_message("system"))
        session.add_message(create_assistant_message("assistant"))

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
        session = create_session("Test Title")
        data = session.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        parsed = Session.model_validate(restored)

        assert parsed.title == "Test Title"
        assert parsed.messages == []

    def test_messages_list_roundtrip(self):
        session = create_session("Chat")
        session.add_message(create_user_message("User message"))
        session.add_message(create_assistant_message("Assistant response", model="gpt-4o"))

        data = session.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        parsed = Session.model_validate(restored)

        assert len(parsed.messages) == 2
        assert isinstance(parsed.messages[0], UserMessage)
        assert isinstance(parsed.messages[1], AssistantMessage)
        assert parsed.messages[1].model == "gpt-4o"

    def test_mixed_message_types_roundtrip(self):
        session = create_session()
        session.add_message(create_system_message("System prompt", name="bot"))
        session.add_message(create_user_message("User input"))
        session.add_message(create_assistant_message("Response"))

        data = session.model_dump(mode="python")
        json_bytes = orjson.dumps(data)
        restored = orjson.loads(json_bytes)

        parsed = Session.model_validate(restored)

        assert isinstance(parsed.messages[0], SystemMessage)
        assert parsed.messages[0].name == "bot"
        assert isinstance(parsed.messages[1], UserMessage)
        assert isinstance(parsed.messages[2], AssistantMessage)