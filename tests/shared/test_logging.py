"""Tests for logging module."""
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from yapa.shared.logging import LOGGER_NAMES, get_logger


@pytest.fixture(autouse=True)
def reset_loggers():
    """Reset loggers after each test."""
    import yapa.shared.logging as logging_module

    for name in list(logging_module._loggers.keys()):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.NOTSET)

    logging_module._loggers.clear()
    yield


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a mock data directory."""
    return tmp_path / "yapa-test"


class TestGetLogger:
    def test_creates_logger_with_correct_name(self, mock_data_dir):
        """Test that get_logger returns a Logger with the correct name."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            logger = get_logger("core")

            assert isinstance(logger, logging.Logger)
            assert logger.name == "core"

    def test_creates_file_in_date_subdir(self, mock_data_dir):
        """Test that log files are created in YYYY-MM-DD subdirectory."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            logger = get_logger("core")

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            expected_dir = mock_data_dir / "logs" / today
            expected_file = expected_dir / "core.log"

            assert expected_dir.exists()
            assert expected_file.exists()

    def test_multiple_loggers_have_separate_files(self, mock_data_dir):
        """Test that different logger names create separate files."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            logger1 = get_logger("core")
            logger2 = get_logger("tui")

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            log_dir = mock_data_dir / "logs" / today

            assert (log_dir / "core.log").exists()
            assert (log_dir / "tui.log").exists()

    def test_log_level_from_config(self, mock_data_dir):
        """Test that log level is set from config."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "WARNING"
            mock_config.return_value.data_dir = mock_data_dir

            logger = get_logger("core")

            assert logger.level == logging.WARNING

    def test_log_level_override(self, mock_data_dir):
        """Test that explicit level parameter overrides config."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            logger = get_logger("core", level="DEBUG")

            assert logger.level == logging.DEBUG

    def test_console_handler_added_when_requested(self, mock_data_dir):
        """Test that console handler is added when console=True."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            logger = get_logger("core", console=True)

            handler_types = [type(h).__name__ for h in logger.handlers]
            assert "FileHandler" in handler_types
            assert "StreamHandler" in handler_types

    def test_console_handler_not_added_by_default(self, mock_data_dir):
        """Test that console handler is not added by default."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            logger = get_logger("core", console=False)

            handler_types = [type(h).__name__ for h in logger.handlers]
            assert "FileHandler" in handler_types
            assert "StreamHandler" not in handler_types

    def test_duplicate_calls_return_same_logger(self, mock_data_dir):
        """Test that repeated get_logger calls return same instance."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            logger1 = get_logger("core")
            logger2 = get_logger("core")

            assert logger1 is logger2

    def test_invalid_logger_name_raises_error(self, mock_data_dir):
        """Test that invalid logger name raises ValueError."""
        with patch("yapa.shared.logging.get_config") as mock_config:
            mock_config.return_value.log_level = "INFO"
            mock_config.return_value.data_dir = mock_data_dir

            with pytest.raises(ValueError) as exc_info:
                get_logger("invalid_name")

            assert "Invalid logger name" in str(exc_info.value)


class TestLoggerNames:
    def test_contains_expected_names(self):
        """Test that LOGGER_NAMES contains expected values."""
        assert "core" in LOGGER_NAMES
        assert "agent" in LOGGER_NAMES
        assert "tui" in LOGGER_NAMES

    def test_is_frozen(self):
        """Test that LOGGER_NAMES is immutable."""
        with pytest.raises((TypeError, AttributeError)):
            LOGGER_NAMES.add("new_name")