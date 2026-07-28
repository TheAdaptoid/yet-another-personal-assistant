"""Tests for CLI server command."""

from unittest.mock import patch


def test_server_command(runner):
    with patch("yapa.cli.app.uvicorn.run") as mock_run:
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["server"])
        assert result.exit_code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == "yapa.api.app:create_app"
        assert mock_run.call_args.kwargs["factory"] is True


def test_server_with_custom_host_port(runner):
    with patch("yapa.cli.app.uvicorn.run") as mock_run:
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["server", "--host", "0.0.0.0", "--port", "9000"])
        assert result.exit_code == 0
        _, kwargs = mock_run.call_args
        assert kwargs.get("host") == "0.0.0.0"
        assert kwargs.get("port") == 9000
        assert kwargs.get("factory") is True
