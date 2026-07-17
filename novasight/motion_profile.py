from __future__ import annotations

from dataclasses import dataclass
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
        features: list[dict[str, Any]] = []
        for sample in samples:
            feature = _analyze_sample(sample)
            if feature is not None:
                features.append(feature)
        if not features:
            raise ValueError("valid samples do not contain a usable trajectory")
        fitts_a_ms, fitts_b_ms = _fit_fitts(features)
        progress_curve = _median_curve([item["progress_curve"] for item in features])
        distance_profiles: dict[str, dict[str, Any]] = {}
        for group in ("micro", "near", "mid", "far"):
            group_items = [item for item in features if item["distance_group"] == group]
            if group_items:
                distance_profiles[group] = {
                    "sample_count": len(group_items),
                    "progress_curve": _median_curve([item["progress_curve"] for item in group_items]),
                    "median_duration_ms": _median([item["duration_ms"] for item in group_items]),
                }
        direction_scales = _direction_duration_scales(features, fitts_a_ms, fitts_b_ms)
        correction_start = _median([item["correction_start_ratio"] for item in features])
        correction_gain = _median([_terminal_curve_gain(item["progress_curve"]) for item in features])
        direction_coverage = len({item["direction_group"] for item in features})
        distance_coverage = len({item["distance_group"] for item in features})
        quality_score = min(100, int(
            25
            + min(35, len(features) * 0.7)
            + direction_coverage * 3
            + distance_coverage * 4
        ))
        profile = {
            "profile_version": 2,
            "profile_id": f"profile_{time.time_ns()}",
            "name": name,
            "session_id": session_id,
            "sample_count": len(features),
            "quality_score": quality_score,
            "features": {
                "median_duration_ms": _median([item["duration_ms"] for item in features]),
                "median_peak_speed_px_ms": _median([item["peak_speed_px_ms"] for item in features]),
                "direction_coverage": direction_coverage,
                "distance_coverage": distance_coverage,
            },
            "timing": {
                "model": "fitts",
                "fitts_a_ms": fitts_a_ms,
                "fitts_b_ms": fitts_b_ms,
            },
            "progress_curve": progress_curve,
            "distance_profiles": distance_profiles,
            "direction_duration_scales": direction_scales,
            "runtime_parameters": {
                "correction_start_ratio": min(0.95, max(0.55, correction_start)),
                "correction_gain": min(0.70, max(0.15, correction_gain)),
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


CURVE_POINTS = 16


def _analyze_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    points = [item for item in sample.get("points", []) if isinstance(item, dict)]
    if len(points) < 3:
        return None
    points.sort(key=lambda item: float(item.get("t_us", 0.0)))
    start_t = float(points[0].get("t_us", 0.0))
    end_t = float(points[-1].get("t_us", 0.0))
    duration_ms = (end_t - start_t) / 1000.0
    if not 20.0 <= duration_ms <= 2500.0:
        return None
    start_x = float(points[0].get("x", 0.0))
    start_y = float(points[0].get("y", 0.0))
    target_x = float(sample.get("target_x", 0.0))
    target_y = float(sample.get("target_y", 0.0))
    vector_x = target_x - start_x
    vector_y = target_y - start_y
    distance = math.hypot(vector_x, vector_y)
    if distance < 2.0:
        return None
    unit_x = vector_x / distance
    unit_y = vector_y / distance
    time_ratios: list[float] = []
    along_ratios: list[float] = []
    peak_speed = 0.0
    monotonic_along = 0.0
    for index, point in enumerate(points):
        elapsed = max(0.0, float(point.get("t_us", 0.0)) - start_t)
        along = ((float(point.get("x", 0.0)) - start_x) * unit_x + (float(point.get("y", 0.0)) - start_y) * unit_y) / distance
        monotonic_along = max(monotonic_along, min(1.0, max(0.0, along)))
        time_ratios.append(min(1.0, elapsed / max(1.0, end_t - start_t)))
        along_ratios.append(monotonic_along)
        if index:
            previous = points[index - 1]
            dt_ms = max(0.001, (float(point.get("t_us", 0.0)) - float(previous.get("t_us", 0.0))) / 1000.0)
            speed = math.hypot(
                float(point.get("x", 0.0)) - float(previous.get("x", 0.0)),
                float(point.get("y", 0.0)) - float(previous.get("y", 0.0)),
            ) / dt_ms
            peak_speed = max(peak_speed, speed)
    curve = [_sample_series(time_ratios, along_ratios, index / (CURVE_POINTS - 1)) for index in range(CURVE_POINTS)]
    curve[0] = 0.0
    curve[-1] = 1.0
    correction_start = next(
        (index / (CURVE_POINTS - 1) for index, value in enumerate(curve) if value >= 0.82),
        0.82,
    )
    radius = max(1.0, float(sample.get("radius_px", 1.0)))
    return {
        "duration_ms": duration_ms,
        "distance_px": distance,
        "width_px": radius * 2.0,
        "index_of_difficulty": math.log2(distance / (radius * 2.0) + 1.0),
        "peak_speed_px_ms": peak_speed,
        "progress_curve": curve,
        "correction_start_ratio": correction_start,
        "distance_group": _distance_group(distance),
        "direction_group": _direction_group(vector_x, vector_y),
    }


def _fit_fitts(features: list[dict[str, Any]]) -> tuple[float, float]:
    xs = [float(item["index_of_difficulty"]) for item in features]
    ys = [float(item["duration_ms"]) for item in features]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((value - mean_x) ** 2 for value in xs)
    if variance <= 1e-6:
        return max(20.0, mean_y * 0.45), max(15.0, mean_y * 0.25)
    b_ms = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / variance
    b_ms = min(500.0, max(15.0, b_ms))
    a_ms = min(1000.0, max(0.0, mean_y - b_ms * mean_x))
    return a_ms, b_ms


def _median_curve(curves: list[list[float]]) -> list[float]:
    result = [_median([float(curve[index]) for curve in curves]) for index in range(CURVE_POINTS)]
    result[0] = 0.0
    for index in range(1, len(result)):
        result[index] = max(result[index - 1], min(1.0, result[index]))
    result[-1] = 1.0
    return result


def _sample_series(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    for index in range(1, len(xs)):
        if x <= xs[index]:
            span = max(1e-9, xs[index] - xs[index - 1])
            weight = (x - xs[index - 1]) / span
            return ys[index - 1] + (ys[index] - ys[index - 1]) * weight
    return ys[-1]


def _direction_duration_scales(features: list[dict[str, Any]], a_ms: float, b_ms: float) -> dict[str, float]:
    scales: dict[str, float] = {}
    for group in ("left", "right", "up", "down", "diagonal"):
        values = []
        for item in features:
            if item["direction_group"] != group:
                continue
            predicted = max(1.0, a_ms + b_ms * float(item["index_of_difficulty"]))
            values.append(float(item["duration_ms"]) / predicted)
        scales[group] = min(1.45, max(0.65, _median(values))) if values else 1.0
    return scales


def _distance_group(distance: float) -> str:
    return "micro" if distance < 50.0 else "near" if distance < 150.0 else "mid" if distance < 350.0 else "far"


def _direction_group(x: float, y: float) -> str:
    if abs(x) >= abs(y) * 1.5:
        return "right" if x >= 0.0 else "left"
    if abs(y) >= abs(x) * 1.5:
        return "down" if y >= 0.0 else "up"
    return "diagonal"


def _terminal_curve_gain(curve: list[float]) -> float:
    slopes = [max(0.0, curve[index] - curve[index - 1]) for index in range(1, len(curve))]
    peak = max(slopes, default=0.0)
    if peak <= 1e-9:
        return 0.35
    terminal = _median(slopes[max(0, len(slopes) - 4):])
    return terminal / peak
