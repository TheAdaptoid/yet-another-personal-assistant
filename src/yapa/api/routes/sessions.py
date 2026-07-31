"""Session CRUD routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from yapa.models import InferenceParams, Session
from yapa.services import SessionService

from ..dependencies import get_session_service

router = APIRouter(tags=["sessions"])

MAX_PER_PAGE = 100
DEFAULT_PER_PAGE = 10


@router.get("/sessions", response_model=list[Session])
async def list_sessions(
    response: Response,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
    session_service: SessionService = Depends(get_session_service),
):
    """List all sessions with pagination."""
    response.headers["X-Total-Count"] = str(session_service.count())
    all_sessions = session_service.list(newest_first=True)
    per_page = min(per_page, MAX_PER_PAGE)
    start = (page - 1) * per_page
    return all_sessions[start : start + per_page]


@router.post("/sessions", response_model=Session, status_code=201)
async def create_session(
    request: Request,
    response: Response,
    session_service: SessionService = Depends(get_session_service),
):
    """Create a new session."""
    session = session_service.create()
    response.headers["Location"] = (
        f"{request.app.state.config.api_prefix}/sessions/{session.id}"
    )
    return session


@router.get("/sessions/{session_id}", response_model=Session)
async def get_session(
    session_id: UUID,
    session_service: SessionService = Depends(get_session_service),
):
    """Get a session by ID."""
    return session_service.get(str(session_id))


@router.patch("/sessions/{session_id}", response_model=Session)
async def patch_session_title(
    session_id: UUID,
    body: dict,
    session_service: SessionService = Depends(get_session_service),
):
    """Rename a session."""
    title = body.get("title")
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    return session_service.rename(str(session_id), title)


@router.delete("/sessions/{session_id}", response_model=None, status_code=204)
async def delete_session(
    session_id: UUID,
    session_service: SessionService = Depends(get_session_service),
):
    """Delete a session."""
    session_service.delete(str(session_id))


@router.patch("/sessions/{session_id}/system-prompt", response_model=Session)
async def patch_system_prompt(
    session_id: UUID,
    body: dict,
    session_service: SessionService = Depends(get_session_service),
):
    """Set or clear the system prompt for a session."""
    prompt = body.get("system_prompt")
    return session_service.update_system_prompt(str(session_id), prompt)


@router.patch("/sessions/{session_id}/inference-params", response_model=Session)
async def patch_inference_params(
    session_id: UUID,
    body: dict,
    session_service: SessionService = Depends(get_session_service),
):
    """Set or clear the inference params for a session."""
    params = InferenceParams(**body) if body is not None else None
    return session_service.update_inference_params(str(session_id), params)
