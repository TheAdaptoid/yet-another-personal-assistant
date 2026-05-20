"""Chat WebSocket routes."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from yapa.core.services import ChatService, InferenceService, SessionService
from yapa.shared import get_logger
from yapa.shared.models import ChatRequest

from .models import get_inference_service
from .sessions import get_session_service

router = APIRouter(prefix="/chat", tags=["chat"])


async def get_chat_service(
    logger: Annotated[logging.Logger, Depends(lambda: get_logger("core"))],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    inference_service: Annotated[InferenceService, Depends(get_inference_service)],
) -> ChatService:
    """
    Dependency that provides a ChatService instance.

    Args:
        logger: Logger instance.
        session_service: Service for session management.
        inference_service: Service for model inference.

    Returns:
        Configured ChatService instance.
    """
    return ChatService(
        logger=logger,
        session_service=session_service,
        inference_service=inference_service,
    )


@router.websocket("/ws/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: str,
    chat_service: ChatService = Depends(get_chat_service),
):
    """
    Handle a WebSocket chat connection for a specific session.

    Each message is processed as a ChatRequest and results in a ChatResponse.
    Errors are communicated via the ChatResponse.error field without closing
    the connection.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            chat_request = ChatRequest(**data)

            response = await chat_service.process_message(
                session_id=session_id,
                model=chat_request.model,
                message=chat_request.message,
                retry=chat_request.retry,
            )

            await websocket.send_json(response.model_dump())

    except WebSocketDisconnect:
        pass
