from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from novasight.runtime.telemetry import get_telemetry_summary


router = APIRouter()


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    status = websocket.app.state.license.status()
    if not status.configured or not status.valid:
        await websocket.close(code=4401, reason="license required")
        return
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(get_telemetry_summary(websocket.app.state))
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        return


__all__ = ["router", "websocket_stream"]
