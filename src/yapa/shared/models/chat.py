from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="ID of the chat session")
    model: str = Field(..., description="Model to use for generating the response")
    message: str = Field(..., description="The chat message from the client")


class ChatResponse(BaseModel):
    response: str = Field(..., description="The generated response from the chat model")
    done: bool = Field(
        False, description="Indicates if the chat response is complete (for streaming)"
    )
    error: str | None = Field(
        None, description="Error message if the chat processing failed"
    )
