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
        mock_model_service.list_models.assert_called_once_with(
            provider_id="openai", model_type=None
        )


def test_models_empty(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(return_value=[])

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 1  # no models found


def test_models_filter_by_type_llm(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM),
            ModelData(id="llama-3", provider_id="ollama", type=ModelType.LLM),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "--model-type", "llm"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.stdout
        assert "llama-3" in result.stdout
        mock_model_service.list_models.assert_called_once_with(
            provider_id=None, model_type=ModelType.LLM
        )


def test_models_filter_by_type_other(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            ModelData(
                id="text-embedding-ada-002",
                provider_id="openai",
                type=ModelType.OTHER,
            ),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "--model-type", "other"])
        assert result.exit_code == 0
        assert "text-embedding-ada-002" in result.stdout
        assert "other" in result.stdout
        mock_model_service.list_models.assert_called_once_with(
            provider_id=None, model_type=ModelType.OTHER
        )


def test_models_filter_by_type_no_results(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(return_value=[])

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "--model-type", "llm"])
        assert result.exit_code == 1
        mock_model_service.list_models.assert_called_once_with(
            provider_id=None, model_type=ModelType.LLM
        )


def test_models_filter_by_type_invalid(runner, mock_model_service):
    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "--model-type", "invalid"])
        assert result.exit_code != 0
        assert "Invalid" in result.stdout or "invalid" in str(result.exception)


def test_models_filter_by_provider_and_type(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(
            cli, ["models", "--provider", "openai", "--model-type", "llm"]
        )
        assert result.exit_code == 0
        assert "gpt-4o" in result.stdout
        mock_model_service.list_models.assert_called_once_with(
            provider_id="openai", model_type=ModelType.LLM
        )


def test_models_filter_by_type_uses_short_flag(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            ModelData(id="gpt-4o", provider_id="openai", type=ModelType.LLM),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "-t", "llm"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.stdout
        mock_model_service.list_models.assert_called_once_with(
            provider_id=None, model_type=ModelType.LLM
        )
