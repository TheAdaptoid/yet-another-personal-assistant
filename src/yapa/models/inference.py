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

    @property
    def full_id(self) -> str:
        """Returns a combined identifier for the provider and model IDs."""
        return f"{self.provider_id}:{self.id}"


class ToolCallDelta(BaseModel):
    """
    Represents a delta in a tool call response.

    Attributes:
        index (int): The index of the tool call in the response sequence.
        id (str | None): The unique identifier of the tool call, if available.
        name (str | None): The name of the tool being called, if available.
        arguments (str | None): The arguments passed to the tool, if available.
    """

    index: int
    id: str | None = Field(default=None)
    name: str | None = Field(default=None)
    arguments: str | None = Field(default=None)


class StreamDelta(BaseModel):
    """
    Represents a delta in a streaming response.

    Attributes:
        content (str | None): The content of the delta, if any.
        reasoning_content (str | None): The reasoning content of the delta, if any.
        tool_calls (list[ToolCallDelta]): A list of tool call deltas associated with
            this stream delta.
        error (str | None): Error message if an error occurred during streaming.
        done (bool): Whether this delta represents the end of the stream.
    """

    content: str | None = Field(
        default=None, description="The content of the delta, if any"
    )
    reasoning_content: str | None = Field(
        default=None, description="The reasoning content of the delta, if any"
    )
    tool_calls: list[ToolCallDelta] = Field(
        default_factory=list,
        description="A list of tool call deltas associated with this stream delta",
    )
    error: str | None = Field(
        default=None, description="Error message if an error occurred during streaming"
    )
    done: bool = Field(
        default=False, description="Whether this delta represents the end of the stream"
    )
