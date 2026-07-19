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
        points = _normalize_points(sample.get("points"))
        if len(points) < 2:
            raise ValueError("motion sample requires at least two points")
        target_spawn_us = _finite_int(sample.get("target_spawn_us"), points[0]["t_us"])
        click_us = _finite_int(sample.get("click_us"), points[-1]["t_us"])
        first_motion_us = _optional_finite_int(sample.get("first_motion_us"))
        if first_motion_us is None:
            first_motion_us = _detect_first_motion_us(points)
        capture = _normalize_capture(sample.get("capture"))
        normalized = {
            "sample_id": sample_id,
            "spawn_x": _finite_float(sample.get("spawn_x"), points[0]["x"]),
            "spawn_y": _finite_float(sample.get("spawn_y"), points[0]["y"]),
            "target_x": _finite_float(sample.get("target_x"), 0.0),
            "target_y": _finite_float(sample.get("target_y"), 0.0),
            "radius_px": max(1.0, _finite_float(sample.get("radius_px"), 1.0)),
            "target_spawn_us": target_spawn_us,
            "first_motion_us": first_motion_us,
            "click_us": max(click_us, points[-1]["t_us"]),
            "points": points,
            "capture": capture,
        }
        feature = _analyze_sample(normalized)
        quality_reasons = _sample_quality_reasons(normalized, feature)
        normalized["quality"] = "valid" if not quality_reasons else "low_quality"
        normalized["quality_reasons"] = quality_reasons
        normalized["metrics"] = _public_sample_metrics(feature, capture)
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
        all_samples = session.get("samples", [])
        samples = [s for s in all_samples if s.get("quality") == "valid"]
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
        side_offset_samples = _median_side_curve(
            [item["side_offset_curve"] for item in features]
        )
        side_offset_curve = _fit_side_bezier(side_offset_samples)
        distance_profiles: dict[str, dict[str, Any]] = {}
        for group in ("micro", "near", "mid", "far"):
            group_items = [item for item in features if item["distance_group"] == group]
            if group_items:
                distance_profiles[group] = {
                    "sample_count": len(group_items),
                    "progress_curve": _median_curve([item["progress_curve"] for item in group_items]),
                    "side_offset_curve": _fit_side_bezier(
                        _median_side_curve(
                            [item["side_offset_curve"] for item in group_items]
                        )
                    ),
                    "median_duration_ms": _median([item["movement_duration_ms"] for item in group_items]),
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
            "profile_version": 4,
            "profile_id": f"profile_{time.time_ns()}",
            "name": name,
            "session_id": session_id,
            "sample_count": len(features),
            "quality_score": quality_score,
            "features": {
                "median_reaction_ms": _median([item["reaction_time_ms"] for item in features]),
                "median_duration_ms": _median([item["movement_duration_ms"] for item in features]),
                "median_peak_speed_px_ms": _median([item["peak_speed_px_ms"] for item in features]),
                "median_path_efficiency": _median([item["path_efficiency"] for item in features]),
                "direction_coverage": direction_coverage,
                "distance_coverage": distance_coverage,
                "rejected_sample_count": max(0, len(all_samples) - len(features)),
            },
            "timing": {
                "model": "fitts",
                "fitts_a_ms": fitts_a_ms,
                "fitts_b_ms": fitts_b_ms,
            },
            "progress_curve": progress_curve,
            "side_offset_curve": side_offset_curve,
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
MAX_SAMPLE_POINTS = 8192


def _finite_float(value: object, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _finite_int(value: object, fallback: int) -> int:
    return int(round(_finite_float(value, float(fallback))))


def _optional_finite_int(value: object) -> int | None:
    if value is None:
        return None
    number = _finite_float(value, math.nan)
    return int(round(number)) if math.isfinite(number) else None


def _normalize_points(raw: object) -> list[dict[str, float | int]]:
    if not isinstance(raw, list):
        return []
    points: list[dict[str, float | int]] = []
    last_t = -1
    last_x = 0.0
    last_y = 0.0
    for item in raw[:MAX_SAMPLE_POINTS]:
        if not isinstance(item, dict):
            continue
        x = _finite_float(item.get("x"), math.nan)
        y = _finite_float(item.get("y"), math.nan)
        t_us = _finite_int(item.get("t_us"), last_t + 1)
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        t_us = max(last_t + 1, t_us)
        points.append({
            "t_us": t_us,
            "x": x,
            "y": y,
            "dx": _finite_float(item.get("dx"), x - last_x if points else 0.0),
            "dy": _finite_float(item.get("dy"), y - last_y if points else 0.0),
        })
        last_t = t_us
        last_x = x
        last_y = y
    return points


def _normalize_capture(raw: object) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    result: dict[str, Any] = {}
    for key in (
        "event_count", "coalesced_event_count", "dropped_event_count",
        "boundary_hit_count", "miss_click_count", "canvas_width", "canvas_height",
    ):
        result[key] = max(0, _finite_int(source.get(key), 0))
    for key in ("max_event_gap_ms", "max_dispatch_delay_ms", "device_pixel_ratio"):
        result[key] = max(0.0, _finite_float(source.get(key), 0.0))
    for key in ("planned_distance_group", "planned_direction"):
        result[key] = str(source.get(key, ""))[:32]
    return result


def _detect_first_motion_us(points: list[dict[str, Any]]) -> int | None:
    if len(points) < 2:
        return None
    start_x = float(points[0].get("x", 0.0))
    start_y = float(points[0].get("y", 0.0))
    for point in points[1:]:
        if math.hypot(float(point.get("x", 0.0)) - start_x, float(point.get("y", 0.0)) - start_y) >= 1.5:
            return int(point.get("t_us", 0))
    return None


def _sample_quality_reasons(sample: dict[str, Any], feature: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    points = sample.get("points", [])
    capture = sample.get("capture", {})
    if sample.get("first_motion_us") is None:
        reasons.append("no_motion")
    if len(points) < 4:
        reasons.append("too_few_points")
    if feature is None and sample.get("first_motion_us") is not None:
        movement_duration_ms = (
            float(sample.get("click_us", points[-1].get("t_us", 0.0)))
            - float(sample["first_motion_us"])
        ) / 1000.0
        if movement_duration_ms < 20.0:
            reasons.append("movement_too_short")
        elif movement_duration_ms > 2000.0:
            reasons.append("movement_too_long")
        else:
            reasons.append("unusable_trajectory")
    elif feature is not None:
        duration = float(feature["movement_duration_ms"])
        if duration < 20.0:
            reasons.append("movement_too_short")
        elif duration > 2000.0:
            reasons.append("movement_too_long")
        if float(feature["path_efficiency"]) < 0.35:
            reasons.append("inefficient_path")
        if float(feature["click_error_px"]) > float(sample.get("radius_px", 1.0)) * 1.2:
            reasons.append("click_outside_target")
        if float(feature["reaction_time_ms"]) > 2000.0:
            reasons.append("reaction_too_long")
    if float(capture.get("max_dispatch_delay_ms", 0.0)) > 100.0:
        reasons.append("dispatch_delay")
    if int(capture.get("boundary_hit_count", 0)) > 3:
        reasons.append("boundary_hits")
    return list(dict.fromkeys(reasons))


def _public_sample_metrics(feature: dict[str, Any] | None, capture: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "max_event_gap_ms": float(capture.get("max_event_gap_ms", 0.0)),
        "max_dispatch_delay_ms": float(capture.get("max_dispatch_delay_ms", 0.0)),
    }
    if feature is not None:
        metrics.update({
            "reaction_time_ms": float(feature["reaction_time_ms"]),
            "movement_duration_ms": float(feature["movement_duration_ms"]),
            "path_efficiency": float(feature["path_efficiency"]),
            "peak_speed_px_ms": float(feature["peak_speed_px_ms"]),
            "click_error_px": float(feature["click_error_px"]),
        })
    return metrics


def _analyze_sample(sample: dict[str, Any]) -> dict[str, Any] | None:
    points = [item for item in sample.get("points", []) if isinstance(item, dict)]
    if len(points) < 3:
        return None
    target_spawn_us = float(sample.get("target_spawn_us", points[0].get("t_us", 0.0)))
    first_motion_us = sample.get("first_motion_us")
    if first_motion_us is None:
        first_motion_us = _detect_first_motion_us(points)
    if first_motion_us is None:
        return None
    start_t = float(first_motion_us)
    end_t = float(sample.get("click_us", points[-1].get("t_us", 0.0)))
    movement_duration_ms = (end_t - start_t) / 1000.0
    if not 10.0 <= movement_duration_ms <= 2500.0:
        return None
    reaction_time_ms = max(0.0, (start_t - target_spawn_us) / 1000.0)
    start_x = float(points[0].get("x", 0.0))
    start_y = float(points[0].get("y", 0.0))
    target_x = float(sample.get("target_x", 0.0))
    target_y = float(sample.get("target_y", 0.0))
    click_x = float(points[-1].get("x", 0.0))
    click_y = float(points[-1].get("y", 0.0))
    target_vector_x = target_x - start_x
    target_vector_y = target_y - start_y
    target_distance = math.hypot(target_vector_x, target_vector_y)
    movement_vector_x = click_x - start_x
    movement_vector_y = click_y - start_y
    movement_distance = math.hypot(movement_vector_x, movement_vector_y)
    if min(target_distance, movement_distance) < 2.0:
        return None
    unit_x = movement_vector_x / movement_distance
    unit_y = movement_vector_y / movement_distance
    time_ratios: list[float] = []
    along_ratios: list[float] = []
    peak_speed = 0.0
    monotonic_along = 0.0
    motion_points = [{"t_us": start_t, "x": start_x, "y": start_y}]
    motion_points.extend(point for point in points if float(point.get("t_us", 0.0)) > start_t)
    if len(motion_points) < 3:
        return None
    path_length = 0.0
    for index, point in enumerate(motion_points):
        elapsed = max(0.0, float(point.get("t_us", 0.0)) - start_t)
        along = ((float(point.get("x", 0.0)) - start_x) * unit_x + (float(point.get("y", 0.0)) - start_y) * unit_y) / movement_distance
        monotonic_along = max(monotonic_along, min(1.0, max(0.0, along)))
        time_ratios.append(min(1.0, elapsed / max(1.0, end_t - start_t)))
        along_ratios.append(monotonic_along)
        if index:
            previous = motion_points[index - 1]
            step_distance = math.hypot(
                float(point.get("x", 0.0)) - float(previous.get("x", 0.0)),
                float(point.get("y", 0.0)) - float(previous.get("y", 0.0)),
            )
            path_length += step_distance
            window_index = index - 1
            while window_index > 0 and float(point.get("t_us", 0.0)) - float(motion_points[window_index].get("t_us", 0.0)) < 2000.0:
                window_index -= 1
            window_point = motion_points[window_index]
            dt_ms = max(0.25, (float(point.get("t_us", 0.0)) - float(window_point.get("t_us", 0.0))) / 1000.0)
            window_distance = math.hypot(
                float(point.get("x", 0.0)) - float(window_point.get("x", 0.0)),
                float(point.get("y", 0.0)) - float(window_point.get("y", 0.0)),
            )
            peak_speed = max(peak_speed, window_distance / dt_ms)
    curve = [_sample_series(time_ratios, along_ratios, index / (CURVE_POINTS - 1)) for index in range(CURVE_POINTS)]
    curve[0] = 0.0
    curve[-1] = 1.0
    side_ratios: list[float] = []
    for point in motion_points:
        px = float(point.get("x", 0.0)) - start_x
        py = float(point.get("y", 0.0)) - start_y
        side_ratios.append((px * (-unit_y) + py * unit_x) / movement_distance)
    side_curve = [_sample_series(time_ratios, side_ratios, index / (CURVE_POINTS - 1)) for index in range(CURVE_POINTS)]
    side_curve[0] = 0.0
    side_curve[-1] = 0.0
    correction_start = next(
        (index / (CURVE_POINTS - 1) for index, value in enumerate(curve) if value >= 0.82),
        0.82,
    )
    radius = max(1.0, float(sample.get("radius_px", 1.0)))
    return {
        "reaction_time_ms": reaction_time_ms,
        "movement_duration_ms": movement_duration_ms,
        "distance_px": target_distance,
        "width_px": radius * 2.0,
        "index_of_difficulty": math.log2(target_distance / (radius * 2.0) + 1.0),
        "peak_speed_px_ms": peak_speed,
        "path_length_px": path_length,
        "path_efficiency": min(1.0, movement_distance / max(movement_distance, path_length)),
        "click_error_px": math.hypot(click_x - target_x, click_y - target_y),
        "progress_curve": curve,
        "side_offset_curve": side_curve,
        "correction_start_ratio": correction_start,
        "distance_group": _distance_group(target_distance),
        "direction_group": _direction_group(target_vector_x, target_vector_y),
    }


def _fit_fitts(features: list[dict[str, Any]]) -> tuple[float, float]:
    xs = [float(item["index_of_difficulty"]) for item in features]
    ys = [float(item["movement_duration_ms"]) for item in features]
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


def _median_side_curve(curves: list[list[float]]) -> list[float]:
    """Aggregate signed lateral motion without forcing monotonic progress."""

    result = [
        max(-1.0, min(1.0, _median([float(curve[index]) for curve in curves])))
        for index in range(CURVE_POINTS)
    ]
    result[0] = 0.0
    result[-1] = 0.0
    return result


def _fit_side_bezier(samples: list[float]) -> dict[str, Any]:
    """Fit signed lateral samples to a cubic Bezier with zero endpoints."""

    aa = ab = bb = ay = by = 0.0
    denominator = max(1, len(samples) - 1)
    for index, value in enumerate(samples):
        t = index / denominator
        u = 1.0 - t
        a = 3.0 * u * u * t
        b = 3.0 * u * t * t
        aa += a * a
        ab += a * b
        bb += b * b
        ay += a * float(value)
        by += b * float(value)
    determinant = aa * bb - ab * ab
    if abs(determinant) <= 1e-9:
        control_1 = control_2 = 0.0
    else:
        control_1 = (ay * bb - by * ab) / determinant
        control_2 = (by * aa - ay * ab) / determinant
    control_1 = max(-1.0, min(1.0, control_1))
    control_2 = max(-1.0, min(1.0, control_2))
    return {
        "model": "cubic_bezier_side",
        "control_points": [0.0, control_1, control_2, 0.0],
        "samples": samples,
    }


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
    for group in ("left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right"):
        values = []
        for item in features:
            if item["direction_group"] != group:
                continue
            predicted = max(1.0, a_ms + b_ms * float(item["index_of_difficulty"]))
            values.append(float(item["movement_duration_ms"]) / predicted)
        scales[group] = min(1.45, max(0.65, _median(values))) if values else 1.0
    return scales


def _distance_group(distance: float) -> str:
    return "micro" if distance < 50.0 else "near" if distance < 150.0 else "mid" if distance < 350.0 else "far"


def _direction_group(x: float, y: float) -> str:
    if abs(x) >= abs(y) * 1.5:
        return "right" if x >= 0.0 else "left"
    if abs(y) >= abs(x) * 1.5:
        return "down" if y >= 0.0 else "up"
    vertical = "down" if y >= 0.0 else "up"
    horizontal = "right" if x >= 0.0 else "left"
    return f"{vertical}_{horizontal}"


def _terminal_curve_gain(curve: list[float]) -> float:
    slopes = [max(0.0, curve[index] - curve[index - 1]) for index in range(1, len(curve))]
    peak = max(slopes, default=0.0)
    if peak <= 1e-9:
        return 0.35
    terminal = _median(slopes[max(0, len(slopes) - 4):])
    return terminal / peak
