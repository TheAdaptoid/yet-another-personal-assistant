"""Tests for model command handler and grouped display."""

from unittest.mock import AsyncMock

from yapa.cli.models import _strip_group, display_models, set_default_model
from yapa.config import Config
from yapa.models import ModelData, ModelType


class TestStripGroup:
    """Tests for _strip_group helper."""

    def test_strips_prefix(self):
        assert _strip_group("openai/gpt-4", "openai") == "gpt-4"

    def test_no_prefix_match(self):
        assert _strip_group("anthropic/claude", "openai") == "anthropic/claude"

    def test_other_group(self):
        assert _strip_group("codellama-7b", "other") == "codellama-7b"


class TestDisplayModels:
    """Tests for display_models Tree output."""

    def test_empty_models(self, capsys):
        display_models("test-provider", [])
        captured = capsys.readouterr()
        assert "Models for" in captured.out
        assert "(0)" in captured.out

    def test_single_group(self, capsys):
        models = [
            ModelData(id="openai/gpt-4o", provider_id="p", type=ModelType.LLM),
            ModelData(id="openai/gpt-4-turbo", provider_id="p", type=ModelType.LLM),
        ]
        display_models("openrouter", models)
        captured = capsys.readouterr()
        assert "openai (2)" in captured.out
        assert "gpt-4o" in captured.out
        assert "gpt-4-turbo" in captured.out

    def test_tree_connectors_single_group(self, capsys):
        models = [
            ModelData(id="openai/gpt-4o", provider_id="p", type=ModelType.LLM),
            ModelData(id="openai/gpt-4-turbo", provider_id="p", type=ModelType.LLM),
        ]
        display_models("openrouter", models)
        captured = capsys.readouterr()
        assert "└──" in captured.out
        assert "├──" in captured.out

    def test_multiple_groups(self, capsys):
        models = [
            ModelData(id="openai/gpt-4o", provider_id="p", type=ModelType.LLM),
            ModelData(id="anthropic/claude-3", provider_id="p", type=ModelType.LLM),
            ModelData(id="mistral/mistral-small", provider_id="p", type=ModelType.LLM),
        ]
        display_models("openrouter", models)
        captured = capsys.readouterr()
        assert "anthropic (1)" in captured.out
        assert "openai (1)" in captured.out
        assert "mistral (1)" in captured.out

    def test_other_group(self, capsys):
        models = [
            ModelData(id="codellama-7b", provider_id="p", type=ModelType.LLM),
            ModelData(id="openai/gpt-4o", provider_id="p", type=ModelType.LLM),
        ]
        display_models("test", models)
        captured = capsys.readouterr()
        assert "other (1)" in captured.out
        assert "codellama-7b" in captured.out
        assert "openai (1)" in captured.out
        assert "gpt-4o" in captured.out


class TestSetDefaultModel:
    """Tests for set_default_model."""

    async def test_sets_default_model(self, monkeypatch):
        models = [
            ModelData(
                id="openai/gpt-4o", provider_id="openrouter", type=ModelType.LLM
            )
        ]
        mock_list_models = AsyncMock(return_value={"openrouter": models})
        monkeypatch.setattr(
            "yapa.cli.models.ProviderService.list_models", mock_list_models
        )

        config = Config(default_model="old:old/model")
        monkeypatch.setattr("yapa.cli.models.get_config", lambda: config)

        save_calls: list[Config] = []
        monkeypatch.setattr(
            "yapa.cli.models.save_config", lambda c: save_calls.append(c)
        )

        await set_default_model("openai/gpt-4o")

        assert config.default_model == "openrouter:openai/gpt-4o"
        assert len(save_calls) == 1

    async def test_model_not_found(self, monkeypatch, capsys):
        mock_list_models = AsyncMock(return_value={"openrouter": []})
        monkeypatch.setattr(
            "yapa.cli.models.ProviderService.list_models", mock_list_models
        )

        await set_default_model("nonexistent/model")

        captured = capsys.readouterr()
        assert "not found" in captured.out

    async def test_scoped_to_provider(self, monkeypatch):
        models_a = [
            ModelData(
                id="prov_a/model-a", provider_id="prov_a", type=ModelType.LLM
            )
        ]
        models_b = [
            ModelData(
                id="prov_b/model-b", provider_id="prov_b", type=ModelType.LLM
            )
        ]
        mock_list_models = AsyncMock(
            return_value={"prov_a": models_a, "prov_b": models_b}
        )
        monkeypatch.setattr(
            "yapa.cli.models.ProviderService.list_models", mock_list_models
        )

        config = Config(default_model="old:old/model")
        monkeypatch.setattr("yapa.cli.models.get_config", lambda: config)

        save_calls: list[Config] = []
        monkeypatch.setattr(
            "yapa.cli.models.save_config", lambda c: save_calls.append(c)
        )

        await set_default_model("prov_a/model-a", "prov_a")

        assert config.default_model == "prov_a:prov_a/model-a"
        assert len(save_calls) == 1

    async def test_scoped_not_found_outside_provider(self, monkeypatch, capsys):
        models_a = [
            ModelData(
                id="prov_a/model-a", provider_id="prov_a", type=ModelType.LLM
            )
        ]
        models_b = [
            ModelData(
                id="prov_b/model-b", provider_id="prov_b", type=ModelType.LLM
            )
        ]
        mock_list_models = AsyncMock(
            return_value={"prov_a": models_a, "prov_b": models_b}
        )
        monkeypatch.setattr(
            "yapa.cli.models.ProviderService.list_models", mock_list_models
        )

        await set_default_model("prov_b/model-b", "prov_a")

        captured = capsys.readouterr()
        assert "not found" in captured.out
