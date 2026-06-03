"""YAPA logging configuration."""

import logging
from datetime import datetime, timezone

from yapa.config import get_config

_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    console: bool = False,
    level: str | None = None,
) -> logging.Logger:
    """
    Get a logger with file handler (always) and optional console handler.

    Log files are stored in date-based subdirectories:
    ~/.yapa/logs/{YYYY-MM-DD}/{name}.log

    Args:
        name: Component name (e.g., "core", "agent", "tui").
        console: Whether to add console handler. Defaults to False.
        level: Optional log level override. Defaults to config.log_level.

    Returns:
        logging.Logger: Configured logger.

    Raises:
        ValueError: If logger name is not in the allowlist.
    """
    if name in _loggers:
        return _loggers[name]

    config = get_config()
    log_level = level or config.log_level

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.handlers.clear()

    log_dir = config.data_dir / "logs" / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger
