"""Tests for SessionRepository CRUD operations."""

from __future__ import annotations

import pytest

from yapa.database.repositories import SessionRepository
from yapa.models.message import AssistantMessage, SystemMessage, UserMessage


class TestCreate:
    """Tests for SessionRepository.create()."""

    def test_default_title(self) -> None:
        """Should create a session with 'New Session' as the default title."""
        session = SessionRepository.create()
        assert session.title == "New Session"
        assert session.id is not None
        assert session.created_at is not None

    def test_custom_title(self) -> None:
        """Should create a session with the provided title."""
        session = SessionRepository.create(title="My Chat")
        assert session.title == "My Chat"


class TestGet:
    """Tests for SessionRepository.get()."""

    def test_returns_existing_session(self) -> None:
        """Should return the session matching the given ID."""
        created = SessionRepository.create(title="test")
        loaded = SessionRepository.get(created.id)
        assert loaded.id == created.id
        assert loaded.title == created.title

    def test_raises_value_error_for_missing(self) -> None:
        """Should raise ValueError when the session does not exist."""
        with pytest.raises(ValueError, match="not found"):
            SessionRepository.get("nonexistent")


class TestListAll:
    """Tests for SessionRepository.list_all()."""

    def test_empty_when_no_sessions(self) -> None:
        """Should return an empty list when there are no sessions."""
        assert SessionRepository.list_all() == []

    def test_returns_all_sessions_newest_first(self) -> None:
        """Should return sessions ordered by created_at descending."""
        s1 = SessionRepository.create(title="first")
        s2 = SessionRepository.create(title="second")
        s3 = SessionRepository.create(title="third")
        all_sessions = SessionRepository.list_all()
        assert [s.id for s in all_sessions] == [s3.id, s2.id, s1.id]


class TestDelete:
    """Tests for SessionRepository.delete()."""

    def test_deletes_session(self) -> None:
        """Should remove the session from the database."""
        session = SessionRepository.create()
        SessionRepository.delete(session.id)
        assert SessionRepository.list_all() == []

    def test_cascade_deletes_messages(self) -> None:
        """Should delete all messages belonging to the session."""
        session = SessionRepository.create()
        SessionRepository.add_message(session.id, UserMessage(content="hello"))
        SessionRepository.add_message(session.id, AssistantMessage(content="hi"))
        SessionRepository.delete(session.id)
        with pytest.raises(ValueError, match="not found"):
            SessionRepository.get_messages(session.id)

    def test_raises_value_error_for_missing(self) -> None:
        """Should raise ValueError when the session does not exist."""
        with pytest.raises(ValueError, match="not found"):
            SessionRepository.delete("nonexistent")


class TestAddMessage:
    """Tests for SessionRepository.add_message()."""

    def test_adds_user_message(self) -> None:
        """Should persist a user message with the correct fields."""
        session = SessionRepository.create()
        msg = UserMessage(content="hello")
        table = SessionRepository.add_message(session.id, msg)
        assert table.role == "user"
        assert table.content == "hello"
        assert table.model is None
        assert table.session_id == session.id

    def test_adds_assistant_message_with_model(self) -> None:
        """Should persist an assistant message including the model field."""
        session = SessionRepository.create()
        msg = AssistantMessage(content="hi", model="gpt-4")
        table = SessionRepository.add_message(session.id, msg)
        assert table.role == "assistant"
        assert table.content == "hi"
        assert table.model == "gpt-4"

    def test_adds_system_message(self) -> None:
        """Should persist a system message."""
        session = SessionRepository.create()
        msg = SystemMessage(content="be helpful")
        table = SessionRepository.add_message(session.id, msg)
        assert table.role == "system"
        assert table.content == "be helpful"

    def test_raises_value_error_for_missing_session(self) -> None:
        """Should raise ValueError when the session does not exist."""
        msg = UserMessage(content="hello")
        with pytest.raises(ValueError, match="not found"):
            SessionRepository.add_message("nonexistent", msg)


class TestGetMessages:
    """Tests for SessionRepository.get_messages()."""

    def test_returns_messages_oldest_first(self) -> None:
        """Should return messages ordered by created_at ascending."""
        session = SessionRepository.create()
        m1 = SessionRepository.add_message(session.id, UserMessage(content="first"))
        m2 = SessionRepository.add_message(session.id, UserMessage(content="second"))
        m3 = SessionRepository.add_message(session.id, UserMessage(content="third"))
        messages = SessionRepository.get_messages(session.id)
        assert [m.id for m in messages] == [m1.id, m2.id, m3.id]

    def test_returns_empty_for_no_messages(self) -> None:
        """Should return an empty list when the session has no messages."""
        session = SessionRepository.create()
        assert SessionRepository.get_messages(session.id) == []

    def test_raises_value_error_for_missing_session(self) -> None:
        """Should raise ValueError when the session does not exist."""
        with pytest.raises(ValueError, match="not found"):
            SessionRepository.get_messages("nonexistent")
