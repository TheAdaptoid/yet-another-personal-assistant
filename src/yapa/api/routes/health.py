"""Health check endpoint."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", response_model=dict[str, str])
async def health() -> dict[str, str]:
    """Return a simple health check response."""
    return {"status": "ok"}
