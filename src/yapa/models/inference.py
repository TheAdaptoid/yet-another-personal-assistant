"""Data models for inference-related data."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelType(Enum):
    """Enumeration for model types."""

    LLM = "llm"
    EMBED = "embedding"
    OTHER = "other"


class TokenUsage(BaseModel):
    """
    Token usage for a model response.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total number of tokens used (prompt + completion).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class InferenceParams(BaseModel):
    """Curated set of typed inference parameters."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = Field(default=None)
    seed: int | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    repeat_penalty: float | None = Field(default=None, ge=0.0)


class ModelPricing(BaseModel):
    """Pricing for a model in USD per million tokens."""

    input: float | None = Field(default=None)
    output: float | None = Field(default=None)
    request: float | None = Field(default=None)

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelData(BaseModel):
    """
    Base data model for representing a provider model.

    Attributes:
        id (str): Unique identifier for the model.
        provider_id (str): Identifier for the provider of the model.
        type (ModelType): The type of the model.
        name (str | None): Human-readable model name.
        description (str | None): Human-readable model description.
    """

    id: str = Field(..., description="Unique identifier for the model")
    provider_id: str = Field(
        ..., description="Identifier for the provider of the model"
    )
    type: ModelType = Field(..., description="The type of the model")
    name: str | None = Field(default=None, description="Human-readable model name")
    description: str | None = Field(
        default=None, description="Human-readable model description"
    )

    @property
    def full_id(self) -> str:
        """Return the fully-qualified model identifier (e.g. ``openai:gpt-4``)."""
        return f"{self.provider_id}:{self.id}"

    model_config = ConfigDict(extra="forbid", frozen=True)


class LanguageModel(ModelData):
    """An LLM, carrying LLM-specific capability fields."""

    type: Literal["llm"] = "llm"
    context_length: int | None = Field(default=None)
    max_output: int | None = Field(default=None)
    supports_tools: bool = Field(default=False)
    supports_vision: bool = Field(default=False)
    supports_reasoning: bool = Field(default=False)


    pricing: ModelPricing | None = Field(default=None)


class EmbedModel(ModelData):
    """An embedding model, carrying embedding-specific fields."""

    type: Literal["embedding"] = "embedding"
    embedding_dimensions: int | None = Field(default=None)
    normalized: bool = Field(default=False)
    pricing: ModelPricing | None = Field(default=None)


ModelDataUnion = LanguageModel | EmbedModel | ModelData


class ReasoningEffort(Enum):
    """Unified reasoning effort level, passed as a first-class chat argument."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
