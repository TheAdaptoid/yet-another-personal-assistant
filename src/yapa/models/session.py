"""Session related models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SessionSummary(BaseModel):
    """
    Read-only summary of a session for list display.

    Attributes:
        id (str): The unique identifier of the session.
        title (str): The title of the session.
        created_at (datetime): The timestamp when the session was created.
        updated_at (datetime): The timestamp when the session was last updated.
        message_count (int): The number of messages in the session.
    """

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    model_config = ConfigDict(
        frozen=True,
    )
