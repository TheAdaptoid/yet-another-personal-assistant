"""Data models for inference-related data."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ModelType(Enum):
    """Enumeration for model types."""

    LLM = "llm"
    OTHER = "other"


class InferenceParams(BaseModel):
    """
    Parameters for model inference.

    Attributes:
        temperature: Sampling temperature (0.0 to 2.0). Higher = more creative.
        max_tokens: Maximum tokens to generate. None = use model default.
        top_p: Nucleus sampling threshold. Lower = more focused.
    """

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelData(BaseModel):
    """
    Data model for representing a language model.

    Attributes:
        id (str): Unique identifier for the model.
        provider_id (str): Identifier for the provider of the model.
        type (ModelType): The type of the model.
    """

    id: str = Field(..., description="Unique identifier for the model")
    provider_id: str = Field(
        ..., description="Identifier for the provider of the model"
    )
    type: ModelType = Field(..., description="The type of the model (e.g., 'llm')")

    # Immutable and strict model configuration
    model_config = ConfigDict(extra="forbid", frozen=True)


class StreamDelta(BaseModel):
    """
    Data model for representing a delta in a streaming response.

    Attributes:
        content (str | None): The content of the delta, if any.
        reasoning_content (str | None): The reasoning content of the delta, if any.
        done (bool): Whether this delta represents the end of the stream.
    """

    content: str | None = Field(
        default=None, description="The content of the delta, if any"
    )
    reasoning_content: str | None = Field(
        default=None, description="The reasoning content of the delta, if any"
    )
    done: bool = Field(
        default=False, description="Whether this delta represents the end of the stream"
    )
