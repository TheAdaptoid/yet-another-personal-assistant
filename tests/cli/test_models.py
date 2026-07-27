"""Tests for CLI models command."""

from unittest.mock import AsyncMock, patch

from yapa.models import ModelData, ModelType


def test_models_list(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.stdout


def test_models_list_by_provider(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(return_value=[])

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "--provider", "openai"])
        assert result.exit_code == 1  # no models found
        mock_model_service.list_models.assert_called_once_with(provider_id="openai")


def test_models_empty(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(return_value=[])

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 1  # no models found
