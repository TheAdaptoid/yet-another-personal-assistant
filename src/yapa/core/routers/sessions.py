"""Session API routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from yapa.core.services.session_service import SessionService
from yapa.shared import Config, get_config, get_logger
from yapa.shared.models import Session

router = APIRouter(tags=["sessions"])


class SessionCreate(BaseModel):
    """Request body for creating a session."""

    title: str | None = None


class SessionRename(BaseModel):
    """Request body for renaming a session."""

    title: str


def get_session_service(
    config: Annotated[Config, Depends(get_config)],
    logger: Annotated[logging.Logger, Depends(lambda: get_logger("core"))],
) -> SessionService:
    """
    Dependency that provides a SessionService instance.

    Args:
        config: Application configuration.
        logger: Logger instance.

    Returns:
        Configured SessionService instance.
    """
    return SessionService.with_file_repository(config, logger)


@router.post(
    "/",
    response_model=Session,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new session",
)
async def create_session(
    body: SessionCreate | None = None,
    service: SessionService = Depends(get_session_service),
) -> Session:
    """
    Create a new session with an optional title.

    If no title is provided, a default title will be assigned.
    """
    title = body.title if body else None
    return await service.create_session(title=title)


@router.get(
    "/",
    response_model=list[Session],
    summary="List all sessions",
)
async def list_sessions(
    service: SessionService = Depends(get_session_service),
) -> list[Session]:
    """Retrieve all sessions ordered by creation time (newest first)."""
    return await service.list_sessions()


@router.get(
    "/{session_id}",
    response_model=Session,
    summary="Get a specific session",
)
async def get_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> Session:
    """
    Retrieve a session by its ID.

    Raises 404 if the session doesn't exist.
    """
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return session


@router.patch(
    "/{session_id}",
    response_model=Session,
    summary="Rename a session",
)
async def rename_session(
    session_id: str,
    body: SessionRename,
    service: SessionService = Depends(get_session_service),
) -> Session:
    """
    Rename an existing session.

    Raises 404 if the session doesn't exist.
    """
    success = await service.rename_session(session_id, body.title)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    session = await service.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    return session


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a session",
)
async def delete_session(
    session_id: str,
    service: SessionService = Depends(get_session_service),
) -> None:
    """
    Delete a session by its ID.

    Raises 404 if the session doesn't exist.
    """
    success = await service.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
