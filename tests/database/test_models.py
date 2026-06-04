"""Tests for database model classes (no DB required)."""

from __future__ import annotations

from yapa.database.models import MessageTable, SessionTable
from yapa.models.message import AssistantMessage, SystemMessage, UserMessage


class TestSessionTableCreate:
    """Tests for SessionTable.create()."""

    def test_default_title(self) -> None:
        """Should default to 'New Session' when no title provided."""
        session = SessionTable.create()
        assert session.title == "New Session"

    def test_custom_title(self) -> None:
        """Should use the provided title."""
        session = SessionTable.create(title="My Chat")
        assert session.title == "My Chat"


class TestMessageTableFromPydantic:
    """Tests for MessageTable.from_pydantic()."""

    def test_user_message(self) -> None:
        """Should map role, content, and set model to None."""
        msg = UserMessage(content="hello")
        table = MessageTable.from_pydantic(msg, session_id="s1")
        assert table.role == "user"
        assert table.content == "hello"
        assert table.model is None
        assert table.session_id == "s1"

    def test_assistant_message_with_model(self) -> None:
        """Should preserve the model field from AssistantMessage."""
        msg = AssistantMessage(content="hi", model="gpt-4")
        table = MessageTable.from_pydantic(msg, session_id="s1")
        assert table.role == "assistant"
        assert table.content == "hi"
        assert table.model == "gpt-4"
        assert table.session_id == "s1"

    def test_assistant_message_no_model(self) -> None:
        """Should set model to None when AssistantMessage has no model."""
        msg = AssistantMessage(content="hi")
        table = MessageTable.from_pydantic(msg, session_id="s1")
        assert table.role == "assistant"
        assert table.model is None

    def test_system_message(self) -> None:
        """Should map system message and set model to None."""
        msg = SystemMessage(content="be helpful")
        table = MessageTable.from_pydantic(msg, session_id="s1")
        assert table.role == "system"
        assert table.content == "be helpful"
        assert table.model is None
        assert table.session_id == "s1"
