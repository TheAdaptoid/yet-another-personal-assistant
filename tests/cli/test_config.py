"""Tests for CLI config commands."""

from pathlib import Path

import pytest

from yapa.services.config import Config


def test_config_show_empty(runner, mock_config_store):
    mock_config_store.load.return_value = Config()

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0


def test_config_show_with_providers(runner, mock_config_store):
    cfg = Config(provider_configs={"openai": {"api_key": "sk-..."}})
    mock_config_store.load.return_value = cfg

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "openai" in result.stdout


def test_config_set(runner, mock_config_store):
    cfg = Config()
    mock_config_store.load.return_value = cfg

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["config", "set", "log_level", "DEBUG"])
    assert result.exit_code == 0
    assert mock_config_store.save.called


def test_config_set_nested_provider_key(runner, mock_config_store):
    cfg = Config()
    mock_config_store.load.return_value = cfg

    from yapa.cli.app import cli

    result = runner.invoke(
        cli,
        ["config", "set", "provider_configs.openai.api_key", "sk-test"],
    )
    assert result.exit_code == 0
    assert cfg.provider_configs["openai"].api_key == "sk-test"


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("provider_timeout", "240", 240),
        ("provider_max_retries", "5", 5),
        ("storage_dir", "/tmp/yapa-storage", Path("/tmp/yapa-storage")),
        ("api_prefix", "/custom", "/custom"),
    ],
)
def test_config_set_typed_values(runner, mock_config_store, key, value, expected):
    cfg = Config()
    mock_config_store.load.return_value = cfg

    from yapa.cli.app import cli

    result = runner.invoke(cli, ["config", "set", key, value])
    assert result.exit_code == 0
    assert getattr(cfg, key) == expected
