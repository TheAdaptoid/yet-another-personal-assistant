"""Tests for SessionService."""

import time
from datetime import datetime, timezone

import pytest

from yapa.models import AssistantMessage, UserMessage
from yapa.services.session import SessionService


class TestCreate:
    """Tests for SessionService.create()."""

    def test_default_title(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        assert session.title == "Untitled Session"
        assert session.id is not None

    def test_custom_title(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create(title="My Chat")
        assert session.title == "My Chat"

    def test_persists_to_disk(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create(title="persist")
        loaded = service.get_session(str(session.id))
        assert loaded.id == session.id
        assert loaded.title == "persist"


class TestGetSession:
    """Tests for SessionService.get_session()."""

    def test_returns_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        created = service.create(title="test")
        loaded = service.get_session(str(created.id))
        assert loaded.id == created.id
        assert loaded.title == "test"

    def test_missing_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            service.get_session("nonexistent")

    def test_corrupt_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{bad json")
        with pytest.raises(ValueError, match="Failed to load"):
            service.get_session("corrupt")


class TestListSessions:
    """Tests for SessionService.list_sessions()."""

    def test_empty_when_no_sessions(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        assert service.list_sessions() == []

    def test_ordered_newest_first(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        s1 = service.create(title="old")
        s2 = service.create(title="newer")
        s3 = service.create(title="newest")

        sessions = service.list_sessions()
        assert [s.id for s in sessions] == [s3.id, s2.id, s1.id]

    def test_min_date_filter(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        service.create(title="old")
        time.sleep(0.005)
        cutoff = datetime.now(timezone.utc)
        s2 = service.create(title="new")
        sessions = service.list_sessions(min_date=cutoff)
        assert sessions == [s2]

    def test_max_date_filter(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        s1 = service.create(title="old")
        cutoff = datetime.now(timezone.utc)
        time.sleep(0.005)
        service.create(title="new")
        sessions = service.list_sessions(max_date=cutoff)
        assert sessions == [s1]

    def test_oldest_first_flag(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        s1 = service.create(title="old")
        s2 = service.create(title="new")
        sessions = service.list_sessions(newest_first=False)
        assert [s.id for s in sessions] == [s1.id, s2.id]


class TestRename:
    """Tests for SessionService.rename()."""

    def test_updates_title(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create(title="old")
        updated = service.rename(str(session.id), "new")
        assert updated.title == "new"
        assert updated.id == session.id

    def test_persists_rename(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create(title="old")
        service.rename(str(session.id), "persisted")
        loaded = service.get_session(str(session.id))
        assert loaded.title == "persisted"

    def test_missing_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            service.rename("nonexistent", "new title")

    def test_corrupt_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{bad json")
        with pytest.raises(ValueError, match="Failed to load"):
            service.rename("corrupt", "new title")


class TestDelete:
    """Tests for SessionService.delete()."""

    def test_removes_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        service.delete(str(session.id))
        assert service.list_sessions() == []

    def test_missing_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            service.delete("nonexistent")


class TestAddMessage:
    """Tests for SessionService.add_message()."""

    def test_adds_message(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        msg = UserMessage(content="hello")
        updated = service.add_message(str(session.id), msg)
        assert len(updated.messages) == 1
        assert updated.messages[0].content == "hello"

    def test_persists_message(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        msg = UserMessage(content="hello")
        service.add_message(str(session.id), msg)
        loaded = service.get_session(str(session.id))
        assert len(loaded.messages) == 1
        assert loaded.messages[0].content == "hello"

    def test_adds_multiple_individually(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        service.add_message(str(session.id), UserMessage(content="hi"))
        service.add_message(
            str(session.id), AssistantMessage(content="hey", model="m")
        )
        loaded = service.get_session(str(session.id))
        assert len(loaded.messages) == 2

    def test_missing_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            service.add_message("nonexistent", UserMessage(content="hi"))


class TestAddMessages:
    """Tests for SessionService.add_messages()."""

    def test_adds_multiple_atomically(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        msgs = [
            UserMessage(content="hello"),
            AssistantMessage(content="world", model="m"),
        ]
        updated = service.add_messages(str(session.id), msgs)
        assert len(updated.messages) == 2
        assert updated.messages[0].content == "hello"
        assert updated.messages[1].content == "world"

    def test_persists_all(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        msgs = [
            UserMessage(content="q1"),
            AssistantMessage(content="a1", model="m"),
            UserMessage(content="q2"),
            AssistantMessage(content="a2", model="m"),
        ]
        service.add_messages(str(session.id), msgs)
        loaded = service.get_session(str(session.id))
        assert len(loaded.messages) == 4

    def test_missing_session(self, tmp_path):
        service = SessionService(storage_dir=tmp_path)
        with pytest.raises(ValueError, match="not found"):
            service.add_messages("nonexistent", [UserMessage(content="hi")])

    def test_does_not_mutate_on_missing_session(self, tmp_path):
        """Should not persist anything if the session does not exist."""
        service = SessionService(storage_dir=tmp_path)
        session = service.create()
        msgs = [UserMessage(content="should-not-save")]
        with pytest.raises(ValueError):
            service.add_messages("nonexistent", msgs)
        loaded = service.get_session(str(session.id))
        assert len(loaded.messages) == 0
