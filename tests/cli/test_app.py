"""Tests for CLI app routing via CliRunner."""

from typer.testing import CliRunner

from yapa.cli import cli

runner = CliRunner()


class TestSessionsCommands:
    """Tests for the `sessions` subcommand group."""

    def test_list_empty(self):
        result = runner.invoke(cli, ["sessions", "list"])
        assert result.exit_code == 0

    def test_list_with_session(self, seeded_session):
        result = runner.invoke(cli, ["sessions", "list"])
        assert result.exit_code == 0
        assert seeded_session.id[:8] in result.stdout

    def test_rename(self, seeded_session):
        result = runner.invoke(
            cli, ["sessions", "rename", seeded_session.id, "New Title"]
        )
        assert result.exit_code == 0
        assert "New Title" in result.stdout

    def test_rename_missing(self):
        result = runner.invoke(cli, ["sessions", "rename", "bad-id", "X"])
        assert result.exit_code == 0
        assert "Error" in result.stdout

    def test_delete(self, seeded_session):
        result = runner.invoke(cli, ["sessions", "delete", seeded_session.id])
        assert result.exit_code == 0
        assert "deleted" in result.stdout

    def test_delete_missing(self):
        result = runner.invoke(cli, ["sessions", "delete", "bad-id"])
        assert result.exit_code == 0
        assert "Error" in result.stdout
