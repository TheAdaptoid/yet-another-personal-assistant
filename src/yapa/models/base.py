"""Base classes for data models within the application."""

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TrackedEntity(BaseModel):
    """
    A base class for all entities that intend to be persisted.

    Attributes:
        id (UUID): A unique identifier for the entity.
        created_at (datetime): The timestamp when the entity was created.
        updated_at (datetime): The timestamp when the entity was last updated.
    """

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def touch(self):
        """Update the `updated_at` timestamp."""
        self.updated_at = datetime.now()
