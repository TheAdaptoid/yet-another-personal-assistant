"""Tests for model command handler and grouped display."""

from yapa.cli.models import _group_key, _strip_group, display_models
from yapa.models import ModelData


class TestGroupKey:
    """Tests for _group_key helper."""

    def test_vendor_prefixed(self):
        assert _group_key("openai/gpt-4") == "openai"

    def test_nested_prefix(self):
        assert _group_key("huggingface/TheBloke/Mistral-7B") == "huggingface"

    def test_no_prefix(self):
        assert _group_key("codellama-7b") == "other"

    def test_single_component(self):
        assert _group_key("gpt4") == "other"


class TestStripGroup:
    """Tests for _strip_group helper."""

    def test_strips_prefix(self):
        assert _strip_group("openai/gpt-4", "openai") == "gpt-4"

    def test_no_prefix_match(self):
        assert _strip_group("anthropic/claude", "openai") == "anthropic/claude"

    def test_other_group(self):
        assert _strip_group("codellama-7b", "other") == "codellama-7b"


class TestDisplayModels:
    """Tests for display_models output."""

    def test_empty_models(self, capsys):
        display_models("test-provider", [])
        captured = capsys.readouterr()
        assert "0 total" in captured.out

    def test_single_group(self, capsys):
        models = [
            ModelData(id="openai/gpt-4o", provider_id="p"),
            ModelData(id="openai/gpt-4-turbo", provider_id="p"),
        ]
        display_models("openrouter", models)
        captured = capsys.readouterr()
        assert "openai (2):" in captured.out
        assert "gpt-4o" in captured.out
        assert "gpt-4-turbo" in captured.out

    def test_multiple_groups(self, capsys):
        models = [
            ModelData(id="openai/gpt-4o", provider_id="p"),
            ModelData(id="anthropic/claude-3", provider_id="p"),
            ModelData(id="mistral/mistral-small", provider_id="p"),
        ]
        display_models("openrouter", models)
        captured = capsys.readouterr()
        assert "anthropic (1):" in captured.out
        assert "openai (1):" in captured.out
        assert "mistral (1):" in captured.out

    def test_other_group(self, capsys):
        models = [
            ModelData(id="codellama-7b", provider_id="p"),
            ModelData(id="openai/gpt-4o", provider_id="p"),
        ]
        display_models("test", models)
        captured = capsys.readouterr()
        assert "other (1):" in captured.out
        assert "codellama-7b" in captured.out
        assert "openai (1):" in captured.out
        assert "gpt-4o" in captured.out
