"""Tests for session command handlers."""

import pytest

from yapa.cli.sessions import (
    delete_session,
    list_sessions,
    purge_sessions,
    rename_session,
)
from yapa.database.repositories import SessionRepository

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
