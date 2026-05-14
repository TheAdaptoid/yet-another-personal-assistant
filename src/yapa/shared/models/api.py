class BaseSessionResponse(BaseModel):
    error: str | None = Field(
        None,
        description="Error message if the session operation failed.",
    )


class SessionCreateRequest(BaseModel):
    title: str | None = Field(
        None,
        description="Optional title for the session.",
    )


class SessionCreateResponse(BaseSessionResponse):
    session: SessionData = Field(
        ...,
        description="The created session object.",
    )


class SessionListResponse(BaseSessionResponse):
    sessions: list[SessionData] = Field(
        ...,
        description="An unordered list of all sessions.",
    )


class SessionRenameRequest(BaseModel):
    title: str = Field(
        ...,
        description="New title for the session.",
    )


class SessionRenameResponse(BaseSessionResponse):
    id: str = Field(
        ...,
        description="Unique identifier for the renamed session.",
    )
