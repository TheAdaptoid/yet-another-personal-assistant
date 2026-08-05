"""Tests for CLI models command."""

from unittest.mock import AsyncMock, patch

from yapa.models import EmbedModel, LanguageModel, ModelData, ModelPricing, ModelType


def test_models_list(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            LanguageModel(id="gpt-4o", provider_id="openai"),
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
            LanguageModel(id="gpt-4o", provider_id="openai"),
            LanguageModel(id="llama-3", provider_id="ollama"),
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


def test_models_filter_by_type_embedding(runner, mock_model_service):
    mock_model_service.list_models = AsyncMock(
        return_value=[
            LanguageModel(id="text-embedding-ada-002", provider_id="openai"),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models", "--model-type", "embedding"])
        assert result.exit_code == 0
        assert "text-embedding-ada-002" in result.stdout
        mock_model_service.list_models.assert_called_once_with(
            provider_id=None, model_type=ModelType.EMBED
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
            LanguageModel(id="gpt-4o", provider_id="openai"),
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
            LanguageModel(id="gpt-4o", provider_id="openai"),
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


def test_models_list_mixed_types(runner, mock_model_service):
    """EmbedModel and LanguageModel both render without AttributeError."""
    mock_model_service.list_models = AsyncMock(
        return_value=[
            LanguageModel(id="gpt-4o", provider_id="openai", context_length=128000),
            EmbedModel(id="text-embedding-ada-002", provider_id="openai"),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.stdout
        assert "text-embedding-ada-002" in result.stdout


def test_models_list_base_model_data(runner, mock_model_service):
    """Base ModelData (no context_length/max_output) renders without crash."""
    mock_model_service.list_models = AsyncMock(
        return_value=[
            ModelData(id="custom-model", provider_id="custom", type=ModelType.OTHER),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "custom-model" in result.stdout


def test_models_pricing_column(runner, mock_model_service):
    """Pricing column renders input/output/request values and a placeholder."""
    mock_model_service.list_models = AsyncMock(
        return_value=[
            LanguageModel(
                id="gpt-4o",
                provider_id="openai",
                pricing=ModelPricing(input=2.5, output=10.0, request=0.01),
            ),
            LanguageModel(id="free-model", provider_id="openai"),
        ]
    )

    with patch("yapa.cli.app.ModelService", return_value=mock_model_service):
        from yapa.cli.app import cli

        result = runner.invoke(cli, ["models"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.stdout
        assert "free-model" in result.stdout
