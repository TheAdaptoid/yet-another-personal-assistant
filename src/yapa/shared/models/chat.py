"""Chat request/response models."""

from pydantic import BaseModel, Field

from yapa.shared.models.inference import ModelData


class ChatRequest(BaseModel):
    """Request model for a chat message."""

    model: ModelData = Field(
        ..., description="Model to use for generating the response"
    )
    message: str = Field(..., description="The chat message from the client")
    retry: bool = Field(
        False,
        description="If True, skip re-adding the message and re-run inference "
        "on the existing session (used after a failed inference attempt)",
    )


class ChatResponse(BaseModel):
    """Response model for a chat message."""

    response: str = Field(..., description="The generated response from the chat model")
    done: bool = Field(
        False, description="Indicates if the chat response is complete (for streaming)"
    )
    error: str | None = Field(
        None, description="Error message if the chat processing failed"
    )
