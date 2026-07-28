"""Model listing and lookup routes."""

from fastapi import APIRouter, Depends, HTTPException

from yapa.models import ModelData, ModelType
from yapa.services import ModelService

from ..dependencies import get_model_service

router = APIRouter(tags=["models"])


@router.get("/models", response_model=list[ModelData])
async def list_models(
    provider_id: str | None = None,
    model_type: str | None = None,
    model_service: ModelService = Depends(get_model_service),
):
    """List all models, optionally filtered by provider and model type."""
    try:
        model_type_enum = ModelType(model_type) if model_type else None
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model type: '{model_type}'. "
                f"Must be one of {[e.value for e in ModelType]}"
            ),
        )
    return await model_service.list_models(
        provider_id=provider_id, model_type=model_type_enum
    )


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
