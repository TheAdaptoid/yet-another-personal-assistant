"""A generic store for managing entities of a specific type."""

import json
import os
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import ValidationError

from yapa.logging import get_logger
from yapa.models import TrackedEntity

from .exceptions import StorageDeleteError, StorageReadError, StorageWriteError

logger = get_logger(__name__)


TE = TypeVar("TE", bound=TrackedEntity)


class GenericJSONStore(Generic[TE]):
    """A generic JSON store for managing entities of a specific type."""

    def __init__(self, storage_dir: Path, entity_type: type[TE]):
        """
        Initialize the GenericJSONStore.

        Args:
            storage_dir: The directory to store the entities in.
            entity_type: The type of the entities to store.
        """
        self._dir = storage_dir
        self._entity_type = entity_type

        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, entity: TE, *, overwrite: bool = False) -> None:
        """
        Save an entity.

        This method persists the entity to disk and updates its metadata.

        Args:
            entity: The entity to save.
            overwrite: If True, overwrite the file if it already exists.

        Raises:
            FileExistsError: If the file already exists and overwrite is False.
            StorageWriteError: If the entity fails to be saved.
        """
        file_path = self._dir / f"{entity.id}.json"

        if not overwrite and file_path.exists():
            raise FileExistsError("File already exists")

        try:
            entity.touch()
            tmp_file_path = file_path.with_suffix(".tmp")
            tmp_file_path.write_text(
                json.dumps(entity.model_dump(mode="json")), encoding="utf-8"
            )
            os.replace(tmp_file_path, file_path)
        except Exception as e:
            err_msg = f"Failed to save entity: {e}"
            logger.error(err_msg, exc_info=True)
            try:  # If error, cleanup the temporary file
                tmp_file_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise StorageWriteError(err_msg)
        else:
            logger.debug(
                f"Entity of type {self._entity_type.__name__} saved: {entity.id}"
            )

    def load(self, entity_id: str) -> TE:
        """
        Load an entity from the storage.

        Args:
            entity_id: The ID of the entity to load.

        Returns:
            The loaded entity.

        Raises:
            FileNotFoundError: If the file is not found.
            StorageReadError: If the entity fails to be loaded.
        """
        file_path = self._dir / f"{entity_id}.json"

        if not file_path.exists():
            raise FileNotFoundError("File not found")

        try:
            data = file_path.read_text(encoding="utf-8")
            entity = self._entity_type(**json.loads(data))
        except (json.JSONDecodeError, ValidationError, OSError) as e:
            err_msg = f"Failed to load entity: {e}"
            logger.error(err_msg, exc_info=True)
            raise StorageReadError(err_msg) from e

        logger.debug(f"Entity of type {self._entity_type.__name__} loaded: {entity_id}")
        return entity

    def list(self) -> list[TE]:
        """
        List all entities in the storage.

        Entities are returned without any particular sorting.

        Returns:
            A list of all entities in the storage.
        """
        entities = []
        for file_path in self._dir.glob("*.json"):
            try:
                entity_id = file_path.stem
                entities.append(self.load(entity_id))
            except (FileNotFoundError, StorageReadError) as e:
                logger.warning(f"Failed to list entity: {e}")
        return entities

    def exists(self, entity_id: str) -> bool:
        """Check if an entity exists in the storage."""
        return (self._dir / f"{entity_id}.json").exists()

    def count(self) -> int:
        """Return the number of entities in the storage."""
        return len(list(self._dir.glob("*.json")))

    def delete(self, entity_id: str) -> None:
        """
        Delete an entity from the storage.

        Args:
            entity_id: The ID of the entity to delete.

        Raises:
            FileNotFoundError: If the file is not found.
            StorageDeleteError: If the entity fails to be deleted.
        """
        file_path = self._dir / f"{entity_id}.json"

        if not file_path.exists():
            raise FileNotFoundError("File not found")

        try:
            file_path.unlink()
        except Exception as e:
            err_msg = f"Failed to delete entity: {e}"
            logger.error(err_msg, exc_info=True)
            raise StorageDeleteError(err_msg)
        else:
            logger.debug(
                f"Entity of type {self._entity_type.__name__} deleted: {entity_id}"
            )
