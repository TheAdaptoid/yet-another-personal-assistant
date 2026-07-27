"""Model listing and lookup routes."""

from fastapi import APIRouter, Depends, HTTPException

from yapa.models import ModelData
from yapa.services import ModelService

from ..dependencies import get_model_service

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelData])
async def list_models(
    provider_id: str | None = None,
    model_service: ModelService = Depends(get_model_service),
):
    """List all models, optionally filtered by provider."""
    return await model_service.list_models(provider_id=provider_id)


@router.get("/models/{full_id:path}", response_model=ModelData)
async def get_model(
    full_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Get a model by its full ID (e.g. openai:gpt-4o)."""
    try:
        return await model_service.get_model(full_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Model '{full_id}' not found")
