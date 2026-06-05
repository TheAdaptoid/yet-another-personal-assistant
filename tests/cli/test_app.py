"""Tests for CLI app routing via CliRunner."""

from unittest.mock import AsyncMock

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
        assert "not found" in result.stdout

    def test_delete(self, seeded_session):
        result = runner.invoke(cli, ["sessions", "delete", seeded_session.id])
        assert result.exit_code == 0
        assert "Deleted session" in result.stdout

    def test_delete_missing(self):
        result = runner.invoke(cli, ["sessions", "delete", "bad-id"])
        assert result.exit_code == 0
        assert "not found" in result.stdout

    def test_delete_no_args(self):
        """Delete without session_id or --purge prints an error."""
        result = runner.invoke(cli, ["sessions", "delete"])
        assert result.exit_code == 0
        assert "Specify a session ID or use --purge" in result.stdout

    def test_purge_flag(self, monkeypatch):
        """Delete with --purge invokes the purge flow."""
        monkeypatch.setattr("yapa.cli.sessions.Confirm.ask", lambda *a, **kw: True)
        result = runner.invoke(cli, ["sessions", "delete", "--purge"])
        assert result.exit_code == 0

    def test_purge_short_flag(self, monkeypatch):
        """Delete with -p invokes the purge flow."""
        monkeypatch.setattr("yapa.cli.sessions.Confirm.ask", lambda *a, **kw: True)
        result = runner.invoke(cli, ["sessions", "delete", "-p"])
        assert result.exit_code == 0


class TestModelsCommands:
    """Tests for the `models` command routing."""

    def test_models_set_flag(self, monkeypatch):
        """--set routes to set_default_model."""
        mock_set = AsyncMock()
        monkeypatch.setattr("yapa.cli.app.set_default_model", mock_set)
        result = runner.invoke(cli, ["models", "--set=openai/gpt-4o"])
        assert result.exit_code == 0
        mock_set.assert_called_once_with("openai/gpt-4o", None)

    def test_models_set_with_provider(self, monkeypatch):
        """--set + --provider scopes the lookup."""
        mock_set = AsyncMock()
        monkeypatch.setattr("yapa.cli.app.set_default_model", mock_set)
        result = runner.invoke(
            cli, ["models", "--set=openai/gpt-4o", "--provider=openrouter"]
        )
        assert result.exit_code == 0
        mock_set.assert_called_once_with("openai/gpt-4o", "openrouter")
