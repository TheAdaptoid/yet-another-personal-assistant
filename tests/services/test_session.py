"""Tests for SessionService."""

import time
from datetime import datetime, timezone

import pytest

from yapa.services import SessionService


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
