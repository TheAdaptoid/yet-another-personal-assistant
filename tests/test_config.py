"""Tests for configuration management."""

import json

from yapa.config import (
    DEFAULT_MODEL_ID,
    DEFAULT_PROVIDER_ID,
    Config,
    load_config,
    save_config,
)


class TestConfigDefaults:
    """Default values for Config fields."""

    def test_default_model_id(self):
        assert Config().default_model_id == DEFAULT_MODEL_ID

    def test_default_provider_id(self):
        assert Config().default_provider_id == DEFAULT_PROVIDER_ID


class TestLoadConfig:
    """Loading config from disk."""

    def test_load_from_json(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "default_model_id": "custom/model",
                    "default_provider_id": "custom",
                }
            )
        )
        cfg = load_config(path=path)
        assert cfg.default_model_id == "custom/model"
        assert cfg.default_provider_id == "custom"

    def test_old_string_field_ignored(self, tmp_path):
        """A config with the old 'default_model' string uses defaults."""
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"default_model": "openrouter/free"}))
        cfg = load_config(path=path)
        assert cfg.default_model_id == DEFAULT_MODEL_ID
        assert cfg.default_provider_id == DEFAULT_PROVIDER_ID

    def test_missing_file_uses_defaults(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cfg = load_config(path=path)
        assert cfg.default_model_id == DEFAULT_MODEL_ID
        assert cfg.default_provider_id == DEFAULT_PROVIDER_ID

    def test_empty_json_uses_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{}")
        cfg = load_config(path=path)
        assert cfg.default_model_id == DEFAULT_MODEL_ID
        assert cfg.default_provider_id == DEFAULT_PROVIDER_ID

    @staticmethod
    def _write_json(path, data):
        path.write_text(json.dumps(data))

    def test_env_override_model_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YAPA_DEFAULT_MODEL_ID", "env/model")
        path = tmp_path / "config.json"
        self._write_json(
            path,
            {"default_model_id": "file/model", "default_provider_id": "file"},
        )
        cfg = load_config(path=path)
        assert cfg.default_model_id == "env/model"
        assert cfg.default_provider_id == "file"

    def test_env_override_provider_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YAPA_DEFAULT_PROVIDER_ID", "env-provider")
        path = tmp_path / "config.json"
        self._write_json(
            path,
            {"default_model_id": "file/model", "default_provider_id": "file"},
        )
        cfg = load_config(path=path)
        assert cfg.default_model_id == "file/model"
        assert cfg.default_provider_id == "env-provider"

    def test_env_override_both(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YAPA_DEFAULT_MODEL_ID", "env/model")
        monkeypatch.setenv("YAPA_DEFAULT_PROVIDER_ID", "env")
        path = tmp_path / "config.json"
        self._write_json(
            path,
            {"default_model_id": "file/model", "default_provider_id": "file"},
        )
        cfg = load_config(path=path)
        assert cfg.default_model_id == "env/model"
        assert cfg.default_provider_id == "env"


class TestSaveConfig:
    """Saving config to disk."""

    def test_saves_model_fields(self, tmp_path):
        path = tmp_path / "config.json"
        cfg = Config(
            default_model_id="saved/model",
            default_provider_id="saved",
        )
        save_config(cfg, path=path)
        data = json.loads(path.read_text())
        assert data["default_model_id"] == "saved/model"
        assert data["default_provider_id"] == "saved"

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "config.json"
        original = Config(
            default_model_id="roundtrip/model",
            default_provider_id="roundtrip",
        )
        save_config(original, path=path)
        loaded = load_config(path=path)
        assert loaded.default_model_id == "roundtrip/model"
        assert loaded.default_provider_id == "roundtrip"

    def test_data_dir_created(self, tmp_path):
        nested = tmp_path / "a" / "b"
        path = nested / "config.json"
        cfg = Config(data_dir=tmp_path / "data")
        save_config(cfg, path=path)
        assert path.exists()
        assert path.parent == nested
