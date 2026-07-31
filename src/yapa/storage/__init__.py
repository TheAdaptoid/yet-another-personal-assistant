"""Storage module — generic entity persistence via JSON files."""

from .exceptions import StorageDeleteError, StorageReadError, StorageWriteError
from .json_store import GenericJSONStore

__all__ = [
    "GenericJSONStore",
    "StorageDeleteError",
    "StorageReadError",
    "StorageWriteError",
]
