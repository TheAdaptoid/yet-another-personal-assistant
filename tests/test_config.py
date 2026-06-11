"""Tests for configuration management."""

import json

from yapa.config import DEFAULT_MODEL, Config, load_config, save_config


class TestConfigDefaults:
    """Default values for Config fields."""

    def test_default_model(self):
        assert Config().default_model == DEFAULT_MODEL


class TestLoadConfig:
    """Loading config from disk."""

    def test_load_from_json(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"default_model": "custom:custom/model"}))
        cfg = load_config(path=path)
        assert cfg.default_model == "custom:custom/model"

    def test_old_two_field_strings_ignored(self, tmp_path):
        """Old default_model_id/default_provider_id keys are silently ignored."""
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "default_model_id": "old/model",
                    "default_provider_id": "old",
                }
            )
        )
        cfg = load_config(path=path)
        assert cfg.default_model == DEFAULT_MODEL

    def test_missing_file_uses_defaults(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        cfg = load_config(path=path)
        assert cfg.default_model == DEFAULT_MODEL

    def test_empty_json_uses_defaults(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{}")
        cfg = load_config(path=path)
        assert cfg.default_model == DEFAULT_MODEL

    @staticmethod
    def _write_json(path, data):
        path.write_text(json.dumps(data))

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YAPA_DEFAULT_MODEL", "env:env/model")
        path = tmp_path / "config.json"
        self._write_json(path, {"default_model": "file:file/model"})
        cfg = load_config(path=path)
        assert cfg.default_model == "env:env/model"

    def test_env_override_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YAPA_DEFAULT_MODEL", "env:env/model")
        path = tmp_path / "nonexistent.json"
        cfg = load_config(path=path)
        assert cfg.default_model == "env:env/model"


class TestSaveConfig:
    """Saving config to disk."""

    def test_saves_default_model(self, tmp_path):
        path = tmp_path / "config.json"
        cfg = Config(default_model="saved:saved/model")
        save_config(cfg, path=path)
        data = json.loads(path.read_text())
        assert data["default_model"] == "saved:saved/model"

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "config.json"
        original = Config(default_model="rt:rt/model")
        save_config(original, path=path)
        loaded = load_config(path=path)
        assert loaded.default_model == "rt:rt/model"

    def test_data_dir_created(self, tmp_path):
        nested = tmp_path / "a" / "b"
        path = nested / "config.json"
        cfg = Config(data_dir=tmp_path / "data")
        save_config(cfg, path=path)
        assert path.exists()
        assert path.parent == nested
