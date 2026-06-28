"""Custom exceptions for the storage module."""


class StorageWriteError(Exception):
    """Raised when an error occurs while writing an entity to the storage."""

    pass


class StorageReadError(Exception):
    """Raised when an error occurs while reading an entity from the storage."""

    pass


class StorageDeleteError(Exception):
    """Raised when an error occurs while deleting an entity from the storage."""

    pass
