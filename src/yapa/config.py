"""YAPA configuration management."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DEFAULT_DATA_DIR = Path.home() / ".yapa"
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "config.json"
DEFAULT_DATABASE_PATH = DEFAULT_DATA_DIR / "yapa.db"
DEFAULT_MODEL_ID = "openrouter/free"
DEFAULT_PROVIDER_ID = "openrouter"
DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

UNSET = "NOT_SET"


class Config(BaseModel):
    """
    YAPA configuration.

    Attributes:
        openrouter_api_key (str | None): OpenRouter API key for LLM calls.
        lmstudio_base_url (str): Base URL for LM Studio API.
        default_model_id (str): Model identifier to use by default.
        default_provider_id (str): Provider of the default model.
        data_dir (Path): Directory for YAPA data storage (sessions, tasks, logs).
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR).
    """

    openrouter_api_key: str = UNSET
    openrouter_base_url: str = DEFAULT_OPENROUTER_BASE_URL
    lmstudio_api_key: str = UNSET
    lmstudio_base_url: str = DEFAULT_LMSTUDIO_BASE_URL
    default_model_id: str = DEFAULT_MODEL_ID
    default_provider_id: str = DEFAULT_PROVIDER_ID
    data_dir: Path = Field(default_factory=lambda: DEFAULT_DATA_DIR)
    database_path: Path = Field(default_factory=lambda: DEFAULT_DATABASE_PATH)
    log_level: str = Field(
        default=DEFAULT_LOG_LEVEL, pattern="^(DEBUG|INFO|WARNING|ERROR)$"
    )


def load_config(path: Path | None = None) -> Config:
    """
    Load configuration from file with environment variable overrides.

    Reads ~/.yapa/config.json if it exists, then applies environment variable
    overrides. Creates the data directory if it doesn't exist.

    Args:
        path (Path | None): Optional custom config file path. Defaults to
            ~/.yapa/config.json.

    Returns:
        Config: The loaded configuration.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    config_data: dict[str, Any] = {}

    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f) or {}

    env_overrides = {
        "openrouter_api_key": os.environ.get("OPENROUTER_API_KEY"),
        "openrouter_base_url": os.environ.get("OPENROUTER_BASE_URL"),
        "lmstudio_api_key": os.environ.get("LMSTUDIO_API_KEY"),
        "lmstudio_base_url": os.environ.get("LMSTUDIO_BASE_URL"),
        "default_model_id": os.environ.get("YAPA_DEFAULT_MODEL_ID"),
        "default_provider_id": os.environ.get("YAPA_DEFAULT_PROVIDER_ID"),
        "data_dir": os.environ.get("YAPA_DATA_DIR"),
        "database_path": os.environ.get("YAPA_DATABASE_PATH"),
        "log_level": os.environ.get("YAPA_LOG_LEVEL"),
    }

    for key, value in env_overrides.items():
        if value is not None and value != "":
            config_data[key] = value

    if "data_dir" in config_data and isinstance(config_data["data_dir"], str):
        config_data["data_dir"] = Path(config_data["data_dir"])
    if "database_path" in config_data and isinstance(config_data["database_path"], str):
        config_data["database_path"] = Path(config_data["database_path"])

    config = Config(**config_data)

    config.data_dir = config.data_dir.expanduser().resolve()
    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.database_path = config.database_path.expanduser().resolve()
    config.database_path.parent.mkdir(parents=True, exist_ok=True)

    return config


_config: Config | None = None


def get_config() -> Config:
    """
    Get the cached configuration, loading if necessary.

    Returns:
        Config: The cached configuration.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def save_config(config: Config, path: Path | None = None) -> None:
    """
    Save configuration to a JSON file.

    Args:
        config (Config): The configuration to save.
        path (Path | None): Optional custom path. Defaults to ~/.yapa/config.json.

    Raises:
        OSError: If the directory cannot be created or file cannot be written.
    """
    save_path = path or DEFAULT_CONFIG_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump(mode="python")

    if config_dict.get("data_dir"):
        config_dict["data_dir"] = str(config_dict["data_dir"])
    if config_dict.get("database_path"):
        config_dict["database_path"] = str(config_dict["database_path"])

    with open(save_path, "w") as f:
        json.dump(config_dict, f, indent=2)
