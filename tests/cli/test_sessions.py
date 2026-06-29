"""Tests for session command handlers."""

from yapa.cli.sessions import delete_session, list_sessions, rename_session


def test_list_sessions_empty(capsys):
    """list_sessions prints a dim message when no sessions exist."""
    list_sessions()
    captured = capsys.readouterr()
    assert "No sessions" in captured.out


def test_list_sessions_with_sessions(capsys, seeded_session):
    """list_sessions prints a table when sessions exist."""
    list_sessions()
    captured = capsys.readouterr()
    assert str(seeded_session.id)[:8] in captured.out
    assert seeded_session.title in captured.out


def test_rename_session_missing(capsys):
    """rename_session prints an error for a nonexistent session."""
    rename_session("nonexistent", "New Title")
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_delete_session_missing(capsys):
    """delete_session prints an error for a nonexistent session."""
    delete_session("nonexistent")
    captured = capsys.readouterr()
    assert "not found" in captured.out

