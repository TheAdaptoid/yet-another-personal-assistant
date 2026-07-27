"""Tests for config models and JsonConfigStore."""

import json
from pathlib import Path

from yapa.services.config import (
    Config,
    JsonConfigStore,
    ProviderConfig,
)


class TestProviderConfig:
    def test_defaults(self):
        pc = ProviderConfig()
        assert pc.api_key is None
        assert pc.base_url is None

    def test_custom_values(self):
        pc = ProviderConfig(api_key="sk-abc", base_url="https://example.com/v1")
        assert pc.api_key == "sk-abc"
        assert pc.base_url == "https://example.com/v1"


class TestConfigDefaults:
    def test_provider_configs_defaults_to_empty(self):
        cfg = Config()
        assert cfg.provider_configs == {}

    def test_storage_dir_default(self):
        cfg = Config()
        assert cfg.storage_dir == Path.home() / ".yapa" / "storage"

    def test_log_level_default(self):
        cfg = Config()
        assert cfg.log_level == "INFO"

    def test_provider_timeout_default(self):
        cfg = Config()
        assert cfg.provider_timeout == 120

    def test_provider_max_retries_default(self):
        cfg = Config()
        assert cfg.provider_max_retries == 2


class TestJsonConfigStore:
    def test_load_returns_config_with_defaults_when_no_file(self, tmp_path):
        store = JsonConfigStore(path=tmp_path / "config.json")
        cfg = store.load()
        assert cfg.log_level == "INFO"
        assert cfg.storage_dir == Path.home() / ".yapa" / "storage"

    def test_load_reads_from_file(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"log_level": "DEBUG", "storage_dir": str(tmp_path)})
        )
        store = JsonConfigStore(path=config_path)
        cfg = store.load()
        assert cfg.log_level == "DEBUG"
        assert cfg.storage_dir == tmp_path

    def test_env_override_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YAPA_LOG_LEVEL", "ERROR")
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"log_level": "DEBUG"}))
        store = JsonConfigStore(path=config_path)
        cfg = store.load()
        assert cfg.log_level == "ERROR"

    def test_save_writes_config(self, tmp_path):
        config_path = tmp_path / "config.json"
        store = JsonConfigStore(path=config_path)
        cfg = Config(log_level="WARNING", provider_timeout=60)
        store.save(cfg)
        data = json.loads(config_path.read_text())
        assert data["log_level"] == "WARNING"
        assert data["provider_timeout"] == 60

    def test_save_creates_parent_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "config.json"
        store = JsonConfigStore(path=nested)
        store.save(Config())
        assert nested.exists()

    def test_roundtrip(self, tmp_path):
        config_path = tmp_path / "config.json"
        store = JsonConfigStore(path=config_path)
        original = Config(log_level="WARNING", provider_configs={
            "openai": ProviderConfig(api_key="sk-abc"),
        })
        store.save(original)
        loaded = store.load()
        assert loaded.log_level == "WARNING"
        assert loaded.provider_configs["openai"].api_key == "sk-abc"

    def test_empty_json_uses_defaults(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{}")
        store = JsonConfigStore(path=config_path)
        cfg = store.load()
        assert cfg.log_level == "INFO"

    def test_reloads_from_disk(self, tmp_path):
        config_path = tmp_path / "config.json"
        store = JsonConfigStore(path=config_path)
        _ = store.load()
        # Modify file behind the scenes
        config_path.write_text(json.dumps({"log_level": "DEBUG"}))
        cfg2 = store.load()
        # Should re-read from disk
        assert cfg2.log_level == "DEBUG"
