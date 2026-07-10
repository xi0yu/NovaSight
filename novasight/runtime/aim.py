from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from novasight.coordinates import CoordinateTransform
from novasight.contracts import Track


@dataclass(frozen=True)
class AimPointConfig:
    horizontal_percent: float = 50.0
    vertical_percent_from_top: float = 42.0
    offset_x_px: float = 0.0
    offset_y_px: float = 0.0
    ema_enabled: bool = True
    ema_alpha: float = 0.65
    max_anchor_jump_ratio: float = 0.15


@dataclass(frozen=True)
class LatencyCompensationConfig:
    enabled: bool = True
    scale: float = 0.70
    max_compensation_ms: float = 35.0
    reject_if_age_exceeds_ms: float = 55.0
    max_compensation_px: float = 80.0
    min_velocity_px_s: float = 30.0
    max_velocity_px_s: float = 2500.0
    min_velocity_measurements: int = 3
    min_velocity_confidence: float = 0.65
    extra_prediction_delay_ms: float = 2.0


@dataclass(frozen=True)
class EstimatedTargetState:
    track_id: int
    state_ts_ns: int
    capture_ts_ns: int | None
    x: float
    y: float
    vx: float
    vy: float
    valid: bool
    predicted: bool = False
    prediction_confidence: float = 1.0
    position_sigma_px: float = 0.0
    cov_trace: float = 0.0
    nis: float = 0.0
    identity_confidence: float = 1.0
    velocity_measurements: int = 1


@dataclass(frozen=True)
class AimPointState:
    track_id: int
    state_ts_ns: int
    raw_x: float
    raw_y: float
    smoothed_x: float
    smoothed_y: float
    center_x: float
    center_y: float
    anchor_jump_norm: float
    aim_confidence: float
    ema_reset_reason: str = ""


@dataclass(frozen=True)
class CompensatedTarget:
    track_id: int
    source_frame_id: int
    capture_ts_ns: int | None
    state_ts_ns: int
    roi_x: float
    roi_y: float
    control_x: float
    control_y: float
    raw_x: float
    raw_y: float
    smoothed_x: float
    smoothed_y: float
    delta_x: float
    delta_y: float
    applied: bool
    reason: str
    measurement_age_ms: float
    compensation_ms: float
    prediction_confidence: float
    predicted_source: bool
    control_allowed: bool

    def debug_payload(self) -> dict:
        return {
            "track_id": self.track_id,
            "source_frame_id": self.source_frame_id,
            "capture_ts_ns": self.capture_ts_ns,
            "state_ts_ns": self.state_ts_ns,
            "roi_x": self.roi_x,
            "roi_y": self.roi_y,
            "control_x": self.control_x,
            "control_y": self.control_y,
            "raw_x": self.raw_x,
            "raw_y": self.raw_y,
            "smoothed_x": self.smoothed_x,
            "smoothed_y": self.smoothed_y,
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "applied": self.applied,
            "reason": self.reason,
            "measurement_age_ms": self.measurement_age_ms,
            "compensation_ms": self.compensation_ms,
            "prediction_confidence": self.prediction_confidence,
            "predicted_source": self.predicted_source,
            "control_allowed": self.control_allowed,
        }


class AimPointGenerator:
    def __init__(self) -> None:
        self._last_by_track: dict[int, AimPointState] = {}
        self._last_raw_by_track: dict[int, tuple[float, float, float, float, int]] = {}

    def reset(self) -> None:
        self._last_by_track.clear()
        self._last_raw_by_track.clear()

    def update(
        self,
        *,
        track: Track,
        estimate: EstimatedTargetState,
        config: AimPointConfig,
    ) -> AimPointState:
        previous = self._last_by_track.get(track.track_id)
        if previous is not None and previous.state_ts_ns == estimate.state_ts_ns:
            return previous

        ax = _clamp(float(config.horizontal_percent), 0.0, 100.0) / 100.0
        ay = _clamp(float(config.vertical_percent_from_top), 0.0, 100.0) / 100.0
        raw_x = float(track.x1) + float(track.w) * ax + float(config.offset_x_px)
        raw_y = float(track.y1) + float(track.h) * ay + float(config.offset_y_px)
        center_x = float(track.cx)
        center_y = float(track.cy)
        anchor_jump_norm = 0.0
        confidence = 1.0
        reset_reason = ""

        previous_raw = self._last_raw_by_track.get(track.track_id)
        if previous_raw is not None:
            prev_raw_x, prev_raw_y, prev_center_x, prev_center_y, _prev_ts = previous_raw
            jump_x = (raw_x - prev_raw_x) - (center_x - prev_center_x)
            jump_y = (raw_y - prev_raw_y) - (center_y - prev_center_y)
            anchor_jump_norm = hypot(jump_x, jump_y) / max(1.0, float(track.h))
            if anchor_jump_norm > max(0.0, float(config.max_anchor_jump_ratio)):
                confidence = 0.35
                reset_reason = "ANCHOR_JUMP"

        if reset_reason == "ANCHOR_JUMP" and previous is not None:
            smoothed_x = previous.smoothed_x
            smoothed_y = previous.smoothed_y
        elif bool(config.ema_enabled) and previous is not None:
            alpha = _clamp(float(config.ema_alpha), 0.0, 1.0)
            smoothed_x = alpha * raw_x + (1.0 - alpha) * previous.smoothed_x
            smoothed_y = alpha * raw_y + (1.0 - alpha) * previous.smoothed_y
        else:
            smoothed_x = raw_x
            smoothed_y = raw_y
            reset_reason = reset_reason or "EMA_INITIALIZED"

        state = AimPointState(
            track_id=int(track.track_id),
            state_ts_ns=int(estimate.state_ts_ns),
            raw_x=raw_x,
            raw_y=raw_y,
            smoothed_x=smoothed_x,
            smoothed_y=smoothed_y,
            center_x=center_x,
            center_y=center_y,
            anchor_jump_norm=anchor_jump_norm,
            aim_confidence=confidence,
            ema_reset_reason=reset_reason,
        )
        self._last_by_track[track.track_id] = state
        self._last_raw_by_track[track.track_id] = (
            raw_x,
            raw_y,
            center_x,
            center_y,
            int(estimate.state_ts_ns),
        )
        return state


class LatencyCompensator:
    def compensate(
        self,
        *,
        aim: AimPointState,
        estimate: EstimatedTargetState,
        source_frame_id: int,
        compute_ts_ns: int,
        roi_offset_x: float,
        roi_offset_y: float,
        roi_width: float,
        roi_height: float,
        control_width: float | None = None,
        control_height: float | None = None,
        config: LatencyCompensationConfig,
        coordinate_transform: CoordinateTransform | None = None,
    ) -> CompensatedTarget:
        measurement_age_ms = (
            max(0.0, (int(compute_ts_ns) - int(estimate.capture_ts_ns)) / 1e6)
            if estimate.capture_ts_ns is not None
            else 0.0
        )
        roi_width = float(roi_width)
        roi_height = float(roi_height)
        roi_offset_x = float(roi_offset_x)
        roi_offset_y = float(roi_offset_y)
        if not _all_finite(roi_offset_x, roi_offset_y, roi_width, roi_height) or roi_width <= 0.0 or roi_height <= 0.0:
            return self._result(
                aim=aim,
                estimate=estimate,
                source_frame_id=source_frame_id,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                coordinate_transform=coordinate_transform,
                measurement_age_ms=measurement_age_ms,
                reason="ROI_GEOMETRY_INVALID",
                control_allowed=False,
            )
        if not _all_finite(aim.raw_x, aim.raw_y, aim.smoothed_x, aim.smoothed_y):
            return self._result(
                aim=aim,
                estimate=estimate,
                source_frame_id=source_frame_id,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                coordinate_transform=coordinate_transform,
                measurement_age_ms=measurement_age_ms,
                reason="AIM_POINT_INVALID",
                control_allowed=False,
            )
        if measurement_age_ms > float(config.reject_if_age_exceeds_ms):
            return self._result(
                aim=aim,
                estimate=estimate,
                source_frame_id=source_frame_id,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                coordinate_transform=coordinate_transform,
                measurement_age_ms=measurement_age_ms,
                reason="MEASUREMENT_TOO_OLD",
                control_allowed=False,
            )
        if not estimate.valid or not _all_finite(estimate.x, estimate.y, estimate.vx, estimate.vy):
            return self._result(
                aim=aim,
                estimate=estimate,
                source_frame_id=source_frame_id,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                coordinate_transform=coordinate_transform,
                measurement_age_ms=measurement_age_ms,
                reason="ESTIMATE_INVALID",
                control_allowed=False,
            )
        if not (0.0 <= float(estimate.x) <= roi_width and 0.0 <= float(estimate.y) <= roi_height):
            return self._result(
                aim=aim,
                estimate=estimate,
                source_frame_id=source_frame_id,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                coordinate_transform=coordinate_transform,
                measurement_age_ms=measurement_age_ms,
                reason="ESTIMATE_OUT_OF_ROI",
                control_allowed=False,
            )
        if not (0.0 <= float(aim.smoothed_x) <= roi_width and 0.0 <= float(aim.smoothed_y) <= roi_height):
            return self._result(
                aim=aim,
                estimate=estimate,
                source_frame_id=source_frame_id,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                coordinate_transform=coordinate_transform,
                measurement_age_ms=measurement_age_ms,
                reason="AIM_OUT_OF_ROI",
                control_allowed=False,
            )

        dx = 0.0
        dy = 0.0
        applied = False
        reason = "DISABLED"
        compensation_ms = 0.0
        velocity_norm = hypot(float(estimate.vx), float(estimate.vy))
        velocity_confidence = _clamp(float(estimate.prediction_confidence), 0.0, 1.0)
        if bool(config.enabled):
            allowed = (
                velocity_norm >= float(config.min_velocity_px_s)
                and velocity_norm <= float(config.max_velocity_px_s)
                and estimate.velocity_measurements >= int(config.min_velocity_measurements)
                and velocity_confidence >= float(config.min_velocity_confidence)
            )
            if allowed:
                prediction_ts_ns = int(compute_ts_ns) + int(
                    max(0.0, float(config.extra_prediction_delay_ms)) * 1e6
                )
                used_ns = min(
                    max(0, prediction_ts_ns - int(estimate.state_ts_ns)),
                    int(max(0.0, float(config.max_compensation_ms)) * 1e6),
                )
                compensation_ms = used_ns / 1e6
                scale = max(0.0, min(1.5, float(config.scale))) * velocity_confidence
                dx = float(estimate.vx) * (used_ns / 1e9) * scale
                dy = float(estimate.vy) * (used_ns / 1e9) * scale
                dx, dy = _clamp_vector(dx, dy, max(0.0, float(config.max_compensation_px)))
                applied = abs(dx) > 0.0 or abs(dy) > 0.0
                reason = "APPLIED" if applied else "ZERO_DELTA"
            else:
                reason = "VELOCITY_GATE"

        roi_x = float(aim.smoothed_x) + dx
        roi_y = float(aim.smoothed_y) + dy
        if not (0.0 <= roi_x <= roi_width and 0.0 <= roi_y <= roi_height):
            return self._result(
                aim=aim,
                estimate=estimate,
                source_frame_id=source_frame_id,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                coordinate_transform=coordinate_transform,
                measurement_age_ms=measurement_age_ms,
                reason="COMPENSATED_OUT_OF_ROI",
                control_allowed=False,
            )
        control_point = _roi_to_control_point(
            coordinate_transform,
            roi_x,
            roi_y,
            roi_offset_x=roi_offset_x,
            roi_offset_y=roi_offset_y,
        )
        control_x = control_point[0]
        control_y = control_point[1]
        if _has_positive_finite_bounds(control_width, control_height):
            assert control_width is not None
            assert control_height is not None
            if not (0.0 <= control_x <= float(control_width) and 0.0 <= control_y <= float(control_height)):
                return self._result(
                    aim=aim,
                    estimate=estimate,
                    source_frame_id=source_frame_id,
                    roi_offset_x=roi_offset_x,
                    roi_offset_y=roi_offset_y,
                    measurement_age_ms=measurement_age_ms,
                    reason="COMPENSATED_OUT_OF_CONTROL",
                    control_allowed=False,
                )
        return CompensatedTarget(
            track_id=int(aim.track_id),
            source_frame_id=int(source_frame_id),
            capture_ts_ns=estimate.capture_ts_ns,
            state_ts_ns=int(estimate.state_ts_ns),
            roi_x=roi_x,
            roi_y=roi_y,
            control_x=control_x,
            control_y=control_y,
            raw_x=aim.raw_x,
            raw_y=aim.raw_y,
            smoothed_x=aim.smoothed_x,
            smoothed_y=aim.smoothed_y,
            delta_x=dx,
            delta_y=dy,
            applied=applied,
            reason=reason,
            measurement_age_ms=measurement_age_ms,
            compensation_ms=compensation_ms,
            prediction_confidence=float(estimate.prediction_confidence) * float(aim.aim_confidence),
            predicted_source=bool(estimate.predicted or applied),
            control_allowed=True,
        )

    def _result(
        self,
        *,
        aim: AimPointState,
        estimate: EstimatedTargetState,
        source_frame_id: int,
        roi_offset_x: float,
        roi_offset_y: float,
        measurement_age_ms: float,
        reason: str,
        control_allowed: bool,
        coordinate_transform: CoordinateTransform | None = None,
    ) -> CompensatedTarget:
        control_point = _roi_to_control_point(
            coordinate_transform,
            aim.smoothed_x,
            aim.smoothed_y,
            roi_offset_x=roi_offset_x,
            roi_offset_y=roi_offset_y,
        )
        return CompensatedTarget(
            track_id=int(aim.track_id),
            source_frame_id=int(source_frame_id),
            capture_ts_ns=estimate.capture_ts_ns,
            state_ts_ns=int(estimate.state_ts_ns),
            roi_x=aim.smoothed_x,
            roi_y=aim.smoothed_y,
            control_x=control_point[0],
            control_y=control_point[1],
            raw_x=aim.raw_x,
            raw_y=aim.raw_y,
            smoothed_x=aim.smoothed_x,
            smoothed_y=aim.smoothed_y,
            delta_x=0.0,
            delta_y=0.0,
            applied=False,
            reason=reason,
            measurement_age_ms=measurement_age_ms,
            compensation_ms=0.0,
            prediction_confidence=0.0 if not control_allowed else float(estimate.prediction_confidence),
            predicted_source=bool(estimate.predicted),
            control_allowed=bool(control_allowed),
        )


def estimated_state_from_debug(
    *,
    track: Track,
    capture_ts_ns: int | None,
    tracker_debug: dict,
) -> EstimatedTargetState:
    track_payload = _find_track_payload(track, tracker_debug)
    estimate = track_payload.get("estimate") if isinstance(track_payload, dict) else None
    if isinstance(estimate, dict):
        return EstimatedTargetState(
            track_id=int(track.track_id),
            state_ts_ns=_int(estimate.get("state_ts_ns"), capture_ts_ns or 0),
            capture_ts_ns=capture_ts_ns,
            x=_float(estimate.get("x"), track.cx),
            y=_float(estimate.get("y"), track.cy),
            vx=_float(estimate.get("vx"), 0.0),
            vy=_float(estimate.get("vy"), 0.0),
            valid=bool(estimate.get("valid", True)),
            predicted=bool(estimate.get("predicted", False)),
            prediction_confidence=_float(estimate.get("prediction_confidence"), 1.0),
            position_sigma_px=_float(estimate.get("position_sigma_px"), 0.0),
            cov_trace=_float(estimate.get("cov_trace"), 0.0),
            nis=_float(estimate.get("nis"), 0.0),
            identity_confidence=_float(track_payload.get("identity_confidence"), 1.0),
            velocity_measurements=max(1, _int(track_payload.get("hits"), 1)),
        )
    return EstimatedTargetState(
        track_id=int(track.track_id),
        state_ts_ns=int(capture_ts_ns or 0),
        capture_ts_ns=capture_ts_ns,
        x=float(track.cx),
        y=float(track.cy),
        vx=0.0,
        vy=0.0,
        valid=True,
        prediction_confidence=1.0,
        identity_confidence=1.0,
        velocity_measurements=1,
    )


def _find_track_payload(track: Track, tracker_debug: dict) -> dict:
    tracker = tracker_debug.get("tracker") if isinstance(tracker_debug, dict) else None
    tracks = tracker.get("tracks") if isinstance(tracker, dict) else None
    if not isinstance(tracks, list):
        return {}
    for item in tracks:
        if isinstance(item, dict) and int(item.get("track_id", -1)) == int(track.track_id):
            return item
    return {}


def _clamp_vector(x: float, y: float, limit: float) -> tuple[float, float]:
    norm = hypot(x, y)
    if limit <= 0.0 or norm <= limit or norm <= 0.0:
        return x, y
    scale = limit / norm
    return x * scale, y * scale


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _all_finite(*values: float) -> bool:
    return all(isfinite(float(value)) for value in values)


def _has_positive_finite_bounds(width: float | None, height: float | None) -> bool:
    if width is None or height is None:
        return False
    return _all_finite(float(width), float(height)) and float(width) > 0.0 and float(height) > 0.0


def _roi_to_control_point(
    transform: CoordinateTransform | None,
    x: float,
    y: float,
    *,
    roi_offset_x: float,
    roi_offset_y: float,
) -> tuple[float, float]:
    if transform is not None:
        capture_point = transform.roi_to_capture_point(x, y)
        point = transform.capture_to_control_point(capture_point.x, capture_point.y)
        return point.x, point.y
    return float(roi_offset_x) + float(x), float(roi_offset_y) + float(y)


def _float(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(fallback)


def _int(value: object, fallback: int) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return int(fallback)
