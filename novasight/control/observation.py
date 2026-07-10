from __future__ import annotations

from dataclasses import dataclass
import math

from novasight.contracts import Track
from novasight.coordinates import CoordinateTransform


def normalize_aim_y_ratio(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("aim y ratio must be finite")
    return round(max(0.0, min(1.0, numeric)), 2)


@dataclass(frozen=True, slots=True)
class RawAimObservation:
    frame_id: int
    target_id: int
    capture_ts_ns: int | None
    bbox_left_roi_px: float
    bbox_top_roi_px: float
    bbox_width_roi_px: float
    bbox_height_roi_px: float
    y_ratio: float
    aim_roi_x_px: float
    aim_roi_y_px: float
    aim_control_x_px: float
    aim_control_y_px: float
    control_width_px: float
    control_height_px: float
    source_geometry_trusted: bool
    valid: bool
    invalid_reason: str = ""

    def debug_payload(self) -> dict[str, int | float | bool | str | None]:
        return {
            "frame_id": self.frame_id,
            "target_id": self.target_id,
            "capture_ts_ns": self.capture_ts_ns,
            "bbox_left_roi_px": self.bbox_left_roi_px,
            "bbox_top_roi_px": self.bbox_top_roi_px,
            "bbox_width_roi_px": self.bbox_width_roi_px,
            "bbox_height_roi_px": self.bbox_height_roi_px,
            "y_ratio": self.y_ratio,
            "aim_roi_x_px": self.aim_roi_x_px,
            "aim_roi_y_px": self.aim_roi_y_px,
            "aim_control_x_px": self.aim_control_x_px,
            "aim_control_y_px": self.aim_control_y_px,
            "control_width_px": self.control_width_px,
            "control_height_px": self.control_height_px,
            "source_geometry_trusted": self.source_geometry_trusted,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True, slots=True)
class TargetMotionEstimate:
    track_id: int
    state_ts_ns: int
    x: float
    y: float
    vx: float
    vy: float
    valid: bool
    predicted: bool
    prediction_confidence: float
    identity_confidence: float


def target_motion_estimate_from_debug(
    *,
    track: Track,
    capture_ts_ns: int | None,
    tracker_debug: dict,
) -> TargetMotionEstimate:
    tracker = tracker_debug.get("tracker") if isinstance(tracker_debug, dict) else None
    tracks = tracker.get("tracks") if isinstance(tracker, dict) else None
    track_payload = {}
    if isinstance(tracks, list):
        track_payload = next(
            (
                item
                for item in tracks
                if isinstance(item, dict) and int(item.get("track_id", -1)) == int(track.track_id)
            ),
            {},
        )
    estimate = track_payload.get("estimate") if isinstance(track_payload, dict) else None
    if not isinstance(estimate, dict):
        return TargetMotionEstimate(
            track_id=int(track.track_id),
            state_ts_ns=int(capture_ts_ns or 0),
            x=float(track.cx),
            y=float(track.cy),
            vx=0.0,
            vy=0.0,
            valid=True,
            predicted=False,
            prediction_confidence=1.0,
            identity_confidence=1.0,
        )
    return TargetMotionEstimate(
        track_id=int(track.track_id),
        state_ts_ns=_int_value(estimate.get("state_ts_ns"), capture_ts_ns or 0),
        x=_float_value(estimate.get("x"), track.cx),
        y=_float_value(estimate.get("y"), track.cy),
        vx=_float_value(estimate.get("vx"), 0.0),
        vy=_float_value(estimate.get("vy"), 0.0),
        valid=bool(estimate.get("valid", True)),
        predicted=bool(estimate.get("predicted", False)),
        prediction_confidence=_float_value(estimate.get("prediction_confidence"), 1.0),
        identity_confidence=_float_value(track_payload.get("identity_confidence"), 1.0),
    )


class RawAimPointProjector:
    def project(
        self,
        *,
        track: Track,
        frame_id: int,
        capture_ts_ns: int | None,
        y_ratio: float,
        coordinate_transform: CoordinateTransform | None,
        control_width_px: float,
        control_height_px: float,
        source_geometry_trusted: bool,
    ) -> RawAimObservation:
        ratio = normalize_aim_y_ratio(y_ratio)
        left = float(track.x1)
        top = float(track.y1)
        width = float(track.w)
        height = float(track.h)
        aim_roi_x = left + width * 0.5
        aim_roi_y = top + height * ratio
        base = {
            "frame_id": int(frame_id),
            "target_id": int(track.track_id),
            "capture_ts_ns": int(capture_ts_ns) if capture_ts_ns is not None else None,
            "bbox_left_roi_px": left,
            "bbox_top_roi_px": top,
            "bbox_width_roi_px": width,
            "bbox_height_roi_px": height,
            "y_ratio": ratio,
            "aim_roi_x_px": aim_roi_x,
            "aim_roi_y_px": aim_roi_y,
            "control_width_px": float(control_width_px),
            "control_height_px": float(control_height_px),
            "source_geometry_trusted": bool(source_geometry_trusted),
        }

        if not _all_finite(left, top, width, height, aim_roi_x, aim_roi_y) or width <= 0.0 or height <= 0.0:
            return self._invalid(base, "BBOX_INVALID")
        if coordinate_transform is None or not source_geometry_trusted:
            return self._invalid(base, "SOURCE_GEOMETRY_UNTRUSTED")
        if not _positive_finite(control_width_px) or not _positive_finite(control_height_px):
            return self._invalid(base, "CONTROL_GEOMETRY_INVALID")
        if not (
            0.0 <= left
            and 0.0 <= top
            and left + width <= coordinate_transform.roi_width
            and top + height <= coordinate_transform.roi_height
        ):
            return self._invalid(base, "BBOX_OUT_OF_ROI")

        capture_point = coordinate_transform.roi_to_capture_point(aim_roi_x, aim_roi_y)
        control_point = coordinate_transform.capture_to_control_point(capture_point.x, capture_point.y)
        if not _all_finite(control_point.x, control_point.y):
            return self._invalid(base, "AIM_CONTROL_INVALID")
        if not (
            0.0 <= control_point.x <= float(control_width_px)
            and 0.0 <= control_point.y <= float(control_height_px)
        ):
            return self._invalid(base, "AIM_OUT_OF_CONTROL")

        return RawAimObservation(
            **base,
            aim_control_x_px=control_point.x,
            aim_control_y_px=control_point.y,
            valid=True,
        )

    @staticmethod
    def _invalid(
        base: dict[str, int | float | bool | None],
        reason: str,
    ) -> RawAimObservation:
        return RawAimObservation(
            **base,
            aim_control_x_px=0.0,
            aim_control_y_px=0.0,
            valid=False,
            invalid_reason=reason,
        )


def _positive_finite(value: float) -> bool:
    return math.isfinite(float(value)) and float(value) > 0.0


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _float_value(value: object, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float(fallback)


def _int_value(value: object, fallback: int) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    return int(fallback)
