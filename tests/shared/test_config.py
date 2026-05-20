import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from yapa.shared.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_LMSTUDIO_BASE_URL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    Config,
    get_config,
    load_config,
    save_config,
)


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset the global config cache before and after each test."""
    import yapa.shared.config as config_module

    config_module._config = None
    yield
    config_module._config = None


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path):
        """Test that defaults are used when no config file exists."""
        config_path = tmp_path / "config.json"

        config = load_config(path=config_path)

        assert config.default_model == DEFAULT_MODEL
        assert config.openrouter_api_key is None
        assert config.data_dir == DEFAULT_DATA_DIR
        assert config.log_level == DEFAULT_LOG_LEVEL

    def test_loads_from_file(self, tmp_path):
        """Test that config is loaded from JSON file."""
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "openrouter_api_key": "test-key-123",
                    "default_model": "gpt-4",
                    "log_level": "DEBUG",
                }
            )
        )

        config = load_config(path=config_path)

        assert config.openrouter_api_key == "test-key-123"
        assert config.default_model == "gpt-4"
        assert config.log_level == "DEBUG"

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        """Test that environment variables take precedence over file."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"default_model": "from-file"}))

        monkeypatch.setenv("YAPA_DEFAULT_MODEL", "from-env")

        config = load_config(path=config_path)

        assert config.default_model == "from-env"

    def test_empty_env_does_not_override(self, tmp_path, monkeypatch):
        """Test that empty string env var does not override file."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"default_model": "from-file"}))

        monkeypatch.setenv("YAPA_DEFAULT_MODEL", "")

        config = load_config(path=config_path)

        assert config.default_model == "from-file"

    def test_resolves_data_dir_to_absolute(self, tmp_path):
        """Test that data_dir is resolved to absolute path."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"data_dir": "~/yapa-test"}))

        config = load_config(path=config_path)

        assert config.data_dir.is_absolute()
        assert "yapa-test" in str(config.data_dir)

    def test_data_dir_created_if_missing(self, tmp_path):
        """Test that data directory is created if it doesn't exist."""
        new_data_dir = tmp_path / "new-yapa-dir"

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"data_dir": str(new_data_dir)}))

        config = load_config(path=config_path)

        assert new_data_dir.exists()
        assert config.data_dir == new_data_dir


class TestGetConfig:
    def test_returns_cached_config(self, tmp_path):
        """Test that get_config returns the same instance after loading."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2


class TestSaveConfig:
    def test_saves_to_custom_path(self, tmp_path):
        """Test saving to custom path."""
        custom_path = tmp_path / "custom-config.json"

        config = Config(default_model="custom-model", log_level="ERROR")

        save_config(config, path=custom_path)

        assert custom_path.exists()

        with open(custom_path) as f:
            saved = json.load(f)

        assert saved["default_model"] == "custom-model"
        assert saved["log_level"] == "ERROR"

    def test_saves_to_default_path(self, tmp_path):
        """Test saving to default path location."""
        config_path = tmp_path / "config.json"

        config = Config(
            openrouter_api_key="saved-key",
            default_model="claude-3",
            log_level="WARNING",
        )

        save_config(config, path=config_path)

        assert config_path.exists()

        with open(config_path) as f:
            saved = json.load(f)

        assert saved["openrouter_api_key"] == "saved-key"
        assert saved["default_model"] == "claude-3"
        assert saved["log_level"] == "WARNING"

    def test_overwrites_existing(self, tmp_path):
        """Test that save overwrites existing file."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"default_model": "old-model"}))

        config = Config(default_model="new-model")
        save_config(config, path=config_path)

        with open(config_path) as f:
            saved = json.load(f)

        assert saved["default_model"] == "new-model"


class TestConfigModel:
    def test_default_values(self):
        """Test that Config has correct defaults."""
        config = Config()

        assert config.default_model == DEFAULT_MODEL
        assert config.openrouter_api_key is None
        assert config.data_dir == DEFAULT_DATA_DIR
        assert config.log_level == DEFAULT_LOG_LEVEL

    def test_custom_values(self):
        """Test that custom values are accepted."""
        config = Config(
            openrouter_api_key="my-key",
            default_model="my-model",
            data_dir=Path("/tmp/test-yapa"),
            log_level="DEBUG",
        )

        assert config.openrouter_api_key == "my-key"
        assert config.default_model == "my-model"
        assert config.data_dir == Path("/tmp/test-yapa")
        assert config.log_level == "DEBUG"

    def test_valid_log_levels(self):
        """Test that valid log levels are accepted."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = Config(log_level=level)
            assert config.log_level == level

    def test_invalid_log_level_rejected(self):
        """Test that invalid log level raises error."""
        with pytest.raises(ValidationError):
            Config(log_level="INVALID")

    def test_new_fields_default_values(self):
        """Test that new LM Studio and Ollama fields have correct defaults."""
        config = Config()
        
        assert config.openrouter_base_url == DEFAULT_OPENROUTER_BASE_URL
        assert config.lmstudio_api_key is None
        assert config.lmstudio_base_url == DEFAULT_LMSTUDIO_BASE_URL
        assert config.ollama_api_key is None
        assert config.ollama_base_url == DEFAULT_OLLAMA_BASE_URL

    def test_new_fields_custom_values(self):
        """Test that new LM Studio and Ollama fields accept custom values."""
        config = Config(
            openrouter_base_url="https://custom.openrouter.ai/v1",
            lmstudio_api_key="lmstudio-key",
            lmstudio_base_url="http://custom-lmstudio:1234/v1",
            ollama_api_key="ollama-key",
            ollama_base_url="http://custom-ollama:11434/api/v1",
        )
        
        assert config.openrouter_base_url == "https://custom.openrouter.ai/v1"
        assert config.lmstudio_api_key == "lmstudio-key"
        assert config.lmstudio_base_url == "http://custom-lmstudio:1234/v1"
        assert config.ollama_api_key == "ollama-key"
        assert config.ollama_base_url == "http://custom-ollama:11434/api/v1"

    def test_env_overrides_for_new_api_keys(self, tmp_path, monkeypatch):
        """Test that LM Studio and Ollama API keys can be overridden via environment variables."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({}))  # Empty config file

        monkeypatch.setenv("LMSTUDIO_API_KEY", "lmstudio-env-key")
        monkeypatch.setenv("OLLAMA_API_KEY", "ollama-env-key")

        config = load_config(path=config_path)

        assert config.lmstudio_api_key == "lmstudio-env-key"
        assert config.ollama_api_key == "ollama-env-key"

    def test_empty_env_does_not_override_new_api_keys(self, tmp_path, monkeypatch):
        """Test that empty string env vars do not override file values for new API keys."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "lmstudio_api_key": "lmstudio-file-key",
            "ollama_api_key": "ollama-file-key",
        }))

        monkeypatch.setenv("LMSTUDIO_API_KEY", "")
        monkeypatch.setenv("OLLAMA_API_KEY", "")

        config = load_config(path=config_path)

        assert config.lmstudio_api_key == "lmstudio-file-key"
        assert config.ollama_api_key == "ollama-file-key"