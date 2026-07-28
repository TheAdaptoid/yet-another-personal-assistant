"""WebSocket chat handler — streams events per prompt with tool approval support."""

import asyncio
import json
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from yapa.models.event import AgentDoneEvent, AgentErrorEvent
from yapa.models.tool import ToolApprovalRequest, ToolApprovalResponse

router = APIRouter(tags=["chat"])


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

        # Handle tool approval responses
        if message.get("type") == "tool_approval":
            handler = getattr(websocket.state, "_approval_handler", None)
            if handler is not None:
                handler.set_result(ToolApprovalResponse(**message))
            continue

        prompt = message.get("prompt")
        if not prompt:
            await websocket.close(code=4008, reason="Missing 'prompt' field")
            break

        if message.get("model"):
            model = await model_service.get_model(message["model"])
        elif session.model is not None:
            model = session.model
        else:
            await websocket.close(code=4008, reason="No model specified")
            break

        async def get_approval(request: ToolApprovalRequest) -> ToolApprovalResponse:
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
                return await asyncio.wait_for(future, timeout=120.0)
            except asyncio.TimeoutError:
                return ToolApprovalResponse(
                    call_id=request.call_id,
                    approved=False,
                    reason="Approval timeout",
                )
            finally:
                task.cancel()
                websocket.state._approval_handler = None

        async for event in chat_service.stream(
            session_id=session_id,
            prompt=prompt,
            model=model,
            get_approval=get_approval,
        ):
            await websocket.send_json(event.model_dump(mode="json"))

            if isinstance(event, (AgentDoneEvent, AgentErrorEvent)):
                break