"""Tests for session command handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yapa.cli.sessions import (
    _auto_rename_session,
    delete_session,
    list_sessions,
    purge_sessions,
    rename_session,
)
from yapa.database.repositories import SessionRepository
from yapa.models import UserMessage

_session_repo = SessionRepository()


def test_list_sessions_empty(capsys):
    """list_sessions prints a dim message when no sessions exist."""
    list_sessions()
    captured = capsys.readouterr()
    assert "No sessions" in captured.out


def test_list_sessions_with_sessions(capsys, seeded_session):
    """list_sessions prints a table when sessions exist."""
    list_sessions()
    captured = capsys.readouterr()
    assert seeded_session.id[:8] in captured.out
    assert seeded_session.title in captured.out


def test_rename_session_success(seeded_session):
    """rename_session updates the title and prints confirmation."""
    rename_session(seeded_session.id, "My Title")
    updated = _session_repo.get(seeded_session.id)
    assert updated.title == "My Title"


def test_rename_session_missing(capsys):
    """rename_session prints an error for a nonexistent session."""
    rename_session("nonexistent", "New Title")
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_delete_session_success(seeded_session):
    """delete_session removes the session and prints confirmation."""
    delete_session(seeded_session.id)
    assert _session_repo.list_all() == []


def test_delete_session_missing(capsys):
    """delete_session prints an error for a nonexistent session."""
    delete_session("nonexistent")
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_purge_nothing_to_purge(capsys, seeded_session):
    """purge_sessions prints a dim message when nothing qualifies."""
    purge_sessions()
    captured = capsys.readouterr()
    assert "No empty sessions" in captured.out


def test_purge_requires_confirmation(capsys, monkeypatch, seeded_session):
    """purge_sessions prompts and cancels when confirmation is denied."""
    repo = SessionRepository()
    repo.create()
    monkeypatch.setattr("yapa.cli.sessions.Confirm.ask", lambda *a, **kw: False)
    purge_sessions()
    captured = capsys.readouterr()
    assert "Cancelled" in captured.out


def test_purge_removes_empty_sessions(capsys, monkeypatch, seeded_session):
    """purge_sessions deletes sessions with < 2 messages after confirmation."""
    repo = SessionRepository()
    empty = repo.create()
    monkeypatch.setattr("yapa.cli.sessions.Confirm.ask", lambda *a, **kw: True)

    purge_sessions()
    captured = capsys.readouterr()

    assert "Purged 1 session" in captured.out
    repo.get(seeded_session.id)
    with pytest.raises(ValueError):
        repo.get(empty.id)


class TestAutoRenameSession:
    """Tests for _auto_rename_session."""

    async def test_auto_rename_success(self, seeded_session):
        """_auto_rename_session generates title and renames."""
        mock_svc = MagicMock()
        mock_svc.start = AsyncMock()
        mock_svc.messages = [UserMessage(content="Hello world")]
        mock_svc.generate_title = AsyncMock(return_value="My Title")

        with patch("yapa.services.ConversationService", return_value=mock_svc):
            result = await _auto_rename_session(seeded_session.id)

        assert result == "My Title"
        mock_svc.start.assert_awaited_once_with(session_id=seeded_session.id)
        mock_svc.generate_title.assert_awaited_once_with("Hello world")
        updated = _session_repo.get(seeded_session.id)
        assert updated.title == "My Title"

    async def test_auto_rename_title_fails(self, seeded_session):
        """_auto_rename_session returns None when generate_title fails."""
        mock_svc = MagicMock()
        mock_svc.start = AsyncMock()
        mock_svc.messages = [UserMessage(content="Hello world")]
        mock_svc.generate_title = AsyncMock(return_value=None)

        with patch("yapa.services.ConversationService", return_value=mock_svc):
            result = await _auto_rename_session(seeded_session.id)

        assert result is None
        updated = _session_repo.get(seeded_session.id)
        assert updated.title == "New Session"

    async def test_auto_rename_missing_session(self):
        """_auto_rename_session returns None for missing session."""
        mock_svc = MagicMock()
        mock_svc.start = AsyncMock(side_effect=ValueError("not found"))

        with patch("yapa.services.ConversationService", return_value=mock_svc):
            result = await _auto_rename_session("nonexistent-id")

        assert result is None

    async def test_auto_rename_no_user_messages(self, seeded_session):
        """_auto_rename_session returns None when no user messages."""
        mock_svc = MagicMock()
        mock_svc.start = AsyncMock()
        mock_svc.messages = []
        mock_svc.generate_title = AsyncMock(return_value="My Title")

        with patch("yapa.services.ConversationService", return_value=mock_svc):
            result = await _auto_rename_session(seeded_session.id)

        assert result is None
        mock_svc.generate_title.assert_not_called()
