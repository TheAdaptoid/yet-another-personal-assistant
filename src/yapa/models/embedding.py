"""Data models for embedding results."""

from pydantic import BaseModel, Field

from .inference import TokenUsage


class EmbeddingResult(BaseModel):
    """Structured result of an embedding call."""

    vectors: list[list[float]] = Field(...)
    model_id: str = Field(...)
    usage: TokenUsage | None = Field(default=None)
