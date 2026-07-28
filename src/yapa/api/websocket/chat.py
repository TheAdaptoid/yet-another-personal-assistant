"""WebSocket chat handler — streams events per prompt with tool approval support."""

import asyncio
import json
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from yapa.models.event import AgentDoneEvent, AgentErrorEvent
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse

router = APIRouter(tags=["chat"])


async def _approval_listener(
    websocket: WebSocket,
    request: ToolApprovalRequest,
    *,
    timeout: float = 120.0,
) -> ToolApprovalResponse:
    """Wait for a tool approval response from the client over WebSocket."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    websocket.state._approval_handler = future

    async def _wait_for_approval():
        while True:
            try:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "tool_approval":
                    future.set_result(ToolApprovalResponse(**msg))
                    return
            except WebSocketDisconnect:
                if not future.done():
                    future.set_exception(RuntimeError("Connection closed"))
                return
            except json.JSONDecodeError:
                continue

    task = asyncio.create_task(_wait_for_approval())
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        return ToolApprovalResponse(
            call_id=request.call_id,
            approved=False,
            reason="Approval timeout",
        )
    finally:
        task.cancel()
        websocket.state._approval_handler = None


async def _resolve_model(
    websocket: WebSocket,
    message: dict,
    model_service,
    session,
) -> object | None:
    """Resolve model from message, session, or return None on failure."""
    if message.get("model"):
        return await model_service.get_model(message["model"])
    if session.model is not None:
        return session.model
    await websocket.close(code=4008, reason="No model specified")
    return None


async def _handle_message(
    websocket: WebSocket,
    message: dict,
    session_id: UUID,
    session,
    chat_service,
    model_service,
) -> bool:
    """Process a single WebSocket message. Returns False to stop the loop."""
    if message.get("type") == "tool_approval":
        handler = getattr(websocket.state, "_approval_handler", None)
        if handler is not None:
            handler.set_result(ToolApprovalResponse(**message))
        return True

    prompt = message.get("prompt")
    if not prompt:
        await websocket.close(code=4008, reason="Missing 'prompt' field")
        return False

    model = await _resolve_model(websocket, message, model_service, session)
    if model is None:
        return False

    async def get_approval(request: ToolApprovalRequest) -> ToolApprovalResponse:
        return await _approval_listener(websocket, request)

    async for event in chat_service.stream(
        session_id=session_id,
        prompt=prompt,
        model=model,
        get_approval=get_approval,
    ):
        await websocket.send_json(event.model_dump(mode="json"))
        if isinstance(event, (AgentDoneEvent, AgentErrorEvent)):
            break
    return True


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

        ok = await _handle_message(
            websocket, message, session_id, session, chat_service, model_service
        )
        if not ok:
            break
