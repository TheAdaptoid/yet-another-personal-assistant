"""Tests for SessionService."""

import pytest

from yapa.models import AssistantMessage, UserMessage
from yapa.services import SessionService


class TestListAll:
    """Tests for SessionService.list_all()."""

    def test_empty_when_no_sessions(self, repo):
        service = SessionService(session_repo=repo)
        assert service.list_all() == []

    def test_returns_summaries_ordered_by_updated_at(self, repo):
        service = SessionService(session_repo=repo)
        s1 = repo.create(title="old")
        s2 = repo.create(title="newer")
        s3 = repo.create(title="newest")

        summaries = service.list_all()
        assert [s.id for s in summaries] == [s3.id, s2.id, s1.id]

    def test_summary_fields_match_session(self, repo):
        service = SessionService(session_repo=repo)
        created = repo.create(title="My Chat")

        summaries = service.list_all()
        assert len(summaries) == 1
        s = summaries[0]
        assert s.id == created.id
        assert s.title == "My Chat"
        assert s.created_at == created.created_at
        assert s.updated_at == created.updated_at


class TestGet:
    """Tests for SessionService.get()."""

    def test_returns_summary(self, repo):
        service = SessionService(session_repo=repo)
        created = repo.create(title="test")
        summary = service.get(created.id)
        assert summary.id == created.id
        assert summary.title == "test"

    def test_raises_value_error_for_missing(self, repo):
        service = SessionService(session_repo=repo)
        with pytest.raises(ValueError, match="not found"):
            service.get("nonexistent")


class TestGetMessages:
    """Tests for SessionService.get_messages()."""

    def test_returns_pydantic_messages(self, repo):
        service = SessionService(session_repo=repo)
        session = repo.create()
        repo.add_message(session.id, UserMessage(content="hello"))
        repo.add_message(session.id, AssistantMessage(content="hi", model="m"))

        messages = service.get_messages(session.id)
        assert len(messages) == 2
        assert isinstance(messages[0], UserMessage)
        assert messages[0].content == "hello"
        assert isinstance(messages[1], AssistantMessage)
        assert messages[1].content == "hi"
        assert messages[1].model == "m"

    def test_raises_value_error_for_missing_session(self, repo):
        service = SessionService(session_repo=repo)
        with pytest.raises(ValueError, match="not found"):
            service.get_messages("nonexistent")


class TestCreate:
    """Tests for SessionService.create()."""

    def test_returns_summary_with_default_title(self, repo):
        service = SessionService(session_repo=repo)
        summary = service.create()
        assert summary.title == "New Session"
        assert summary.id is not None

    def test_returns_summary_with_custom_title(self, repo):
        service = SessionService(session_repo=repo)
        summary = service.create(title="Custom")
        assert summary.title == "Custom"


class TestRename:
    """Tests for SessionService.rename()."""

    def test_updates_title(self, repo):
        service = SessionService(session_repo=repo)
        summary = service.create(title="old")
        updated = service.rename(summary.id, "new")
        assert updated.title == "new"
        assert updated.id == summary.id

    def test_raises_value_error_for_missing(self, repo):
        service = SessionService(session_repo=repo)
        with pytest.raises(ValueError, match="not found"):
            service.rename("nonexistent", "new title")


class TestDelete:
    """Tests for SessionService.delete()."""

    def test_removes_session(self, repo):
        service = SessionService(session_repo=repo)
        summary = service.create()
        service.delete(summary.id)
        assert service.list_all() == []

    def test_raises_value_error_for_missing(self, repo):
        service = SessionService(session_repo=repo)
        with pytest.raises(ValueError, match="not found"):
            service.delete("nonexistent")


class TestPurge:
    """Tests for SessionService.purge()."""

    def test_removes_sessions_with_fewer_than_2_messages(self, repo):
        svc = SessionService(session_repo=repo)
        empty = repo.create()
        single = repo.create()
        repo.add_message(single.id, UserMessage(content="hi"))
        keep = repo.create()
        repo.add_message(keep.id, UserMessage(content="a"))
        repo.add_message(keep.id, AssistantMessage(content="b", model="m"))

        purged = svc.purge()

        assert len(purged) == 2
        assert empty.id in purged
        assert single.id in purged
        assert repo.get(keep.id)

    def test_returns_empty_list_when_nothing_to_purge(self, repo):
        svc = SessionService(session_repo=repo)
        s = repo.create()
        repo.add_message(s.id, UserMessage(content="a"))
        repo.add_message(s.id, AssistantMessage(content="b", model="m"))
        assert svc.purge() == []
