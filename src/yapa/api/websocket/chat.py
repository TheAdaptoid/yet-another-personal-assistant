"""WebSocket chat handler — streams events per prompt."""

import json
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from yapa.models.event import AgentDoneEvent, AgentErrorEvent

router = APIRouter(tags=["chat"])


async def _resolve_model(
    model_service,
    session,
    requested_model: str | None,
):
    if requested_model:
        return await model_service.get_model(requested_model), None

    if session.model is None:
        return None, "No model specified"

    return session.model, None


@router.websocket("/chat/{session_id}")
async def chat_websocket(
    websocket: WebSocket,
    session_id: UUID,
):
    """Stream chat events over WebSocket for a given session."""
    chat_service = websocket.app.state.chat_service
    model_service = websocket.app.state.model_service
    session_service = websocket.app.state.session_service

    await websocket.accept()

    try:
        session = session_service.get(str(session_id))
    except ValueError:
        await websocket.close(code=4008, reason="Session not found")
        return

    while True:
        try:
            data = await websocket.receive_text()
        except WebSocketDisconnect:
            break

        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            await websocket.close(code=4008, reason="Invalid JSON")
            break

        prompt = message.get("prompt")
        if not prompt:
            await websocket.close(code=4008, reason="Missing 'prompt' field")
            break

        model, error = await _resolve_model(
            model_service,
            session,
            message.get("model"),
        )
        if error is not None:
            await websocket.close(code=4008, reason=error)
            break

        async for event in chat_service.stream(
            session_id=session_id,
            prompt=prompt,
            model=model,
        ):
            await websocket.send_json(event.model_dump(mode="json"))

            if isinstance(event, (AgentDoneEvent, AgentErrorEvent)):
                break
