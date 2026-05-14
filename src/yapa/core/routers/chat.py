import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from yapa.shared.models import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.websocket("/ws")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            chat_request = ChatRequest(**data)

            # TODO: Process the chat request and generate a response
            # For now, we echo the received message back to the client
            await asyncio.sleep(1)  # Simulate processing delay
            await websocket.send_json(
                ChatResponse(response="Model is thinking...", done=False).model_dump()
            )
            await asyncio.sleep(1)  # Simulate thinking delay
            await websocket.send_json(
                ChatResponse(response="Model responded", done=True).model_dump()
            )

    except WebSocketDisconnect:
        print("WebSocket disconnected")
