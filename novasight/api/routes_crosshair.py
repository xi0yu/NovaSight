from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response


router = APIRouter(prefix="/api/crosshair", tags=["crosshair"])


@router.get("")
def get_crosshair(request: Request) -> dict:
    return request.app.state.runtime.crosshair.status()


@router.post("/learn")
def learn_crosshair(request: Request) -> dict:
    try:
        template = request.app.state.runtime.crosshair.learn()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "learned": True,
        "template": template,
        "status": request.app.state.runtime.crosshair.status(),
    }


@router.delete("/template")
def clear_crosshair_template(request: Request) -> dict:
    return request.app.state.runtime.crosshair.clear_template()


@router.get("/template.png")
def get_crosshair_template_preview(request: Request) -> Response:
    payload = request.app.state.runtime.crosshair.template_preview_png()
    if payload is None:
        raise HTTPException(status_code=404, detail="crosshair template unavailable")
    return Response(payload, media_type="image/png", headers={"Cache-Control": "no-store"})
