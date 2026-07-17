from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/motion", tags=["motion-profile"])


def _repo(request: Request):
    return request.app.state.motion_profiles


@router.get("/sessions")
def list_sessions(request: Request) -> list[dict[str, Any]]:
    return _repo(request).list_sessions()


@router.post("/sessions")
async def create_session(request: Request) -> dict[str, Any]:
    payload = await request.json()
    return _repo(request).create_session(str(payload.get("name", "未命名训练")))


@router.post("/sessions/{session_id}/samples")
async def add_sample(request: Request, session_id: str) -> dict[str, Any]:
    try:
        return _repo(request).add_sample(session_id, await request.json())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/profiles/train")
async def train_profile(request: Request) -> dict[str, Any]:
    payload = await request.json()
    try:
        return _repo(request).train_profile(str(payload.get("session_id", "")), str(payload.get("name", "真人画像")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/profiles")
def list_profiles(request: Request) -> list[dict[str, Any]]:
    return _repo(request).list_profiles()


@router.post("/profiles/{profile_id}/activate")
def activate_profile(request: Request, profile_id: str) -> dict[str, Any]:
    profile = next((item for item in _repo(request).list_profiles() if item.get("profile_id") == profile_id), None)
    if profile is None:
        raise HTTPException(status_code=404, detail="motion profile not found")
    report = __import__("novasight.runtime.reconfigurator", fromlist=["RuntimeReconfigurator"]).RuntimeReconfigurator(request.app).apply_field(
        "control", "humanized_motion", {"enabled": True, "active_profile": profile_id}
    )
    return {"profile": profile, "config": report.asdict(include_schema=False)}


@router.post("/profiles/disable")
def disable_profile(request: Request) -> dict[str, Any]:
    report = __import__("novasight.runtime.reconfigurator", fromlist=["RuntimeReconfigurator"]).RuntimeReconfigurator(request.app).apply_field(
        "control", "humanized_motion", {"enabled": False, "active_profile": ""}
    )
    return report.asdict(include_schema=False)
