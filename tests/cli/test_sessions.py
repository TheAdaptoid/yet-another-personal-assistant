"""Tests for CLI sessions commands."""

from uuid import uuid4

from yapa.models import Session


def test_sessions_list(runner, mock_store):
    mock_store.list.return_value = [Session(title="Test")]

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "list"])
    assert result.exit_code == 0
    assert "Test" in result.stdout


def test_sessions_list_empty(runner, mock_store):
    mock_store.list.return_value = []

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "list"])
    assert result.exit_code == 0
    assert "No sessions" in result.stdout


def test_sessions_get(runner, mock_store):
    session = Session(title="Test")
    mock_store.load.return_value = session

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "get", str(session.id)])
    assert result.exit_code == 0
    assert "Test" in result.stdout


def test_sessions_get_not_found(runner, mock_store):
    mock_store.load.side_effect = FileNotFoundError("not found")

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "get", str(uuid4())])
    assert result.exit_code == 1
    assert "Error:" in result.stdout


def test_sessions_delete(runner, mock_store):
    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "delete", str(uuid4())])
    assert result.exit_code == 0
    assert "✓" in result.stdout


def test_sessions_delete_not_found(runner, mock_store):
    mock_store.delete.side_effect = FileNotFoundError("not found")

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "delete", str(uuid4())])
    assert result.exit_code == 1


def test_sessions_rename(runner, mock_store):
    session = Session(title="New Title")
    mock_store.load.return_value = session

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "rename", str(session.id), "New Title"])
    assert result.exit_code == 0
    assert "✓" in result.stdout


def test_sessions_rename_not_found(runner, mock_store):
    mock_store.load.side_effect = FileNotFoundError("not found")

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["sessions", "rename", str(uuid4()), "Nope"])
    assert result.exit_code == 1
