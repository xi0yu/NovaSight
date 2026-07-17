from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import time
from typing import Any


@dataclass(frozen=True)
class MotionPoint:
    t_us: int
    x: float
    y: float
    dx: float = 0.0
    dy: float = 0.0


@dataclass(frozen=True)
class MotionSample:
    sample_id: str
    session_id: str
    spawn_x: float
    spawn_y: float
    target_x: float
    target_y: float
    radius_px: float
    points: tuple[MotionPoint, ...]
    quality: str = "valid"


class MotionProfileRepository:
    """Small JSON repository for the first usable training workflow."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.sessions = self.root / "sessions"
        self.profiles = self.root / "profiles"
        self.sessions.mkdir(parents=True, exist_ok=True)
        self.profiles.mkdir(parents=True, exist_ok=True)

    def create_session(self, name: str = "未命名训练") -> dict[str, Any]:
        session_id = f"session_{time.time_ns()}"
        payload = {"session_id": session_id, "name": name, "created_at": time.time(), "samples": []}
        self._write(self.sessions / f"{session_id}.json", payload)
        return payload

    def list_sessions(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.sessions.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                result.append({k: payload.get(k) for k in ("session_id", "name", "created_at")})
                result[-1]["sample_count"] = len(payload.get("samples", []))
            except (OSError, ValueError, TypeError):
                continue
        return result

    def add_sample(self, session_id: str, sample: dict[str, Any]) -> dict[str, Any]:
        path = self.sessions / f"{session_id}.json"
        if not path.is_file():
            raise ValueError("motion training session not found")
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(sample.get("sample_id") or f"sample_{time.time_ns()}")
        points = sample.get("points") or []
        if len(points) < 2:
            raise ValueError("motion sample requires at least two points")
        normalized = {
            "sample_id": sample_id,
            "spawn_x": float(sample.get("spawn_x", 0.0)),
            "spawn_y": float(sample.get("spawn_y", 0.0)),
            "target_x": float(sample.get("target_x", 0.0)),
            "target_y": float(sample.get("target_y", 0.0)),
            "radius_px": float(sample.get("radius_px", 1.0)),
            "points": points,
            "quality": str(sample.get("quality", "valid")),
        }
        payload.setdefault("samples", []).append(normalized)
        self._write(path, payload)
        return normalized

    def list_profiles(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.profiles.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                result.append(payload)
            except (OSError, ValueError, TypeError):
                continue
        return result

    def train_profile(self, session_id: str, name: str) -> dict[str, Any]:
        path = self.sessions / f"{session_id}.json"
        if not path.is_file():
            raise ValueError("motion training session not found")
        session = json.loads(path.read_text(encoding="utf-8"))
        samples = [s for s in session.get("samples", []) if s.get("quality") == "valid"]
        if not samples:
            raise ValueError("no valid motion samples")
        durations = []
        peak_speeds = []
        for sample in samples:
            points = sample.get("points", [])
            if len(points) < 2:
                continue
            durations.append(max(0.0, (float(points[-1].get("t_us", 0)) - float(points[0].get("t_us", 0))) / 1000.0))
            peak = 0.0
            for prev, current in zip(points, points[1:]):
                dt = max(1.0, float(current.get("t_us", 0)) - float(prev.get("t_us", 0))) / 1000.0
                peak = max(peak, math.hypot(float(current.get("x", 0)) - float(prev.get("x", 0)), float(current.get("y", 0)) - float(prev.get("y", 0))) / dt)
            peak_speeds.append(peak)
        profile = {
            "profile_version": 1,
            "profile_id": f"profile_{time.time_ns()}",
            "name": name,
            "session_id": session_id,
            "sample_count": len(samples),
            "quality_score": min(100, int(55 + min(40, len(samples) * 0.45))),
            "features": {
                "median_duration_ms": _median(durations),
                "median_peak_speed_px_ms": _median(peak_speeds),
            },
            "runtime_parameters": {
                "startup_duration_ratio": 0.15,
                "startup_gain": 0.82,
                "cruise_gain": 1.0,
                "braking_start_ratio": 0.68,
                "braking_gain": 0.72,
                "fine_correction_gain": 0.86,
                "curve_strength": 0.0,
            },
        }
        self._write(self.profiles / f"{profile['profile_id']}.json", profile)
        return profile

    def _write(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0

