"""Storage module — generic entity persistence via JSON files."""

from .exceptions import StorageDeleteError, StorageReadError, StorageWriteError
from .store import GenericStore

__all__ = [
    "GenericStore",
    "StorageDeleteError",
    "StorageReadError",
    "StorageWriteError",
]
