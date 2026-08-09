"""Tests for application logging configuration."""

import logging

from yapa.api.app import create_app
from yapa.logging import get_logger
from yapa.services.config import Config, JsonConfigStore


def test_create_app_applies_configured_log_level(tmp_path, monkeypatch):
    """An app config should update loggers created before app startup."""
    monkeypatch.setattr("yapa.logging.Path.home", lambda: tmp_path)
    logger = get_logger("test_configured_log_level")

    assert logger.level == logging.INFO

    create_app(Config(storage_dir=tmp_path, log_level="DEBUG"))

    assert logger.level == logging.DEBUG


def test_loading_config_applies_configured_log_level(tmp_path):
    """Loading persisted config should update existing loggers."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"log_level": "DEBUG"}', encoding="utf-8")
    logger = get_logger("test_loaded_configured_log_level", level="INFO")

    JsonConfigStore(config_path).load()

    assert logger.level == logging.DEBUG
