"""Data models for inference-related data."""

from pydantic import BaseModel, Field


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
        name (str): Human-readable name for the model.
        provider_id (str): Identifier for the provider of the model.
        provider_name (str): Name of the provider of the model.
    """

    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name for the model")
    provider_id: str = Field(
        ..., description="Identifier for the provider of the model"
    )
    provider_name: str = Field(..., description="Name of the provider of the model")
