"""Router for model listing endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from yapa.core.services import InferenceService
from yapa.shared import get_logger
from yapa.shared.models import ModelData

router = APIRouter(prefix="/models", tags=["models"])


def get_inference_service(
    logger: Annotated[logging.Logger, Depends(lambda: get_logger("core"))],
) -> InferenceService:
    """
    Dependency that provides an instance of the InferenceService.

    Args:
        logger (logging.Logger): The logger to use for the InferenceService.

    Returns:
        InferenceService: An instance of the InferenceService.
    """
    return InferenceService.create_default(logger=logger)


@router.get("/")
async def get_models(
    inference_service: InferenceService = Depends(get_inference_service),
) -> list[ModelData]:
    """
    Get the list of available models.

    Args:
        inference_service (InferenceService): The inference service to use.

    Returns:
        list[ModelData]: A list of available model data. Returns an empty list
            if all providers are unreachable or have no models loaded.
    """
    return await inference_service.get_models()
