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
            # Human-facing metrics do not need a 20 Hz full runtime snapshot.
            # Five updates per second remain responsive without repeatedly
            # competing with the capture/inference/control path for CPU time.
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return


__all__ = ["router", "websocket_stream"]
