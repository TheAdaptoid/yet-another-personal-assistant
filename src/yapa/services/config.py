"""Config models, ConfigStore protocol, and JsonConfigStore implementation."""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_DATA_DIR = Path.home() / ".yapa"
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "config.json"
DEFAULT_STORAGE_DIR = DEFAULT_DATA_DIR / "storage"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_PROVIDER_TIMEOUT = 120

ENV_OVERRIDES: dict[str, str] = {
    "log_level": "YAPA_LOG_LEVEL",
    "storage_dir": "YAPA_STORAGE_DIR",
    "provider_timeout": "YAPA_PROVIDER_TIMEOUT",
    "provider_max_retries": "YAPA_PROVIDER_MAX_RETRIES",
}


class ProviderConfig(BaseModel):
    """Configuration for a single provider."""

    api_key: str | None = None
    base_url: str | None = None


class Config(BaseModel):
    """Application configuration."""

    provider_configs: dict[str, ProviderConfig] = Field(default_factory=dict)
    storage_dir: Path = DEFAULT_STORAGE_DIR
    log_level: str = DEFAULT_LOG_LEVEL
    provider_timeout: int = DEFAULT_PROVIDER_TIMEOUT
    provider_max_retries: int = 2


@runtime_checkable
class ConfigStore(Protocol):
    """Protocol for config persistence."""

    def load(self) -> Config: ...
    def save(self, config: Config) -> None: ...


class JsonConfigStore:
    """JSON-file-backed config store with env variable overrides."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_CONFIG_PATH
        self._cache: Config | None = None

    def load(self) -> Config:
        """Read config file, apply env overrides, cache and return."""
        if self._cache is not None:
            return self._cache
        load_dotenv()
        config_data: dict[str, Any] = {}
        if self._path.exists():
            with open(self._path) as f:
                config_data = json.load(f) or {}

        for key, env_var in ENV_OVERRIDES.items():
            value = os.environ.get(env_var)
            if value is not None and value != "":
                if key == "storage_dir":
                    config_data[key] = Path(value)
                elif key in ("provider_timeout", "provider_max_retries"):
                    try:
                        config_data[key] = int(value)
                    except ValueError:
                        warnings.warn(f"Invalid integer for {key}: {value}")
                else:
                    config_data[key] = value

        self._cache = Config(**config_data)
        return self._cache

    def save(self, config: Config) -> None:
        """Persist config to JSON file."""
        self._cache = config
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            f.write(config.model_dump_json(indent=2))
