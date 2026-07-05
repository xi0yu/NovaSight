from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from novasight.contracts import Detection, Track
from novasight.hardware import BoxInputState

from .strategy import MOVE_KINDS, MoveCommand


Target = Detection | Track


@dataclass(frozen=True, slots=True)
class IsolatedMouseConfig:
    kp_x: float = 0.35
    kp_y: float = 0.24
    max_x: float = 80.0
    max_y: float = 60.0
    deadzone_px: float = 2.0
    aim_ratio: float = 40.0
    smoothing: float = 0.0
    prediction: float = 0.0
    fov_deg: float = 105.0
    counts_per_revolution_x: float = 9980.0
    counts_per_revolution_y: float = 9980.0
    move_kind: str = "raw"
    move_ms: int = 0
    trace_ms: int = 0


class IsolatedMouseStrategy:
    """Independent mouse movement algorithm with no legacy PID state or modules."""

    def __init__(self, config: IsolatedMouseConfig | None = None) -> None:
        self.config = config or IsolatedMouseConfig()
        self._last_error: tuple[float, float] | None = None
        self._last_smooth: tuple[float, float] | None = None
        self._last_s = 0.0

    def calculate(
        self,
        target: Target,
        current_pos: tuple[float, float],
        box_input: BoxInputState,
    ) -> MoveCommand:
        cfg = self.config
        now_s = time.monotonic()
        center_x, center_y = current_pos
        aim_x, aim_y = _aim_point(target, cfg.aim_ratio)
        error_x = float(aim_x - center_x)
        error_y = float(center_y - aim_y)

        predicted_x = error_x
        predicted_y = error_y
        if self._last_error is not None and cfg.prediction > 0:
            previous_x, previous_y = self._last_error
            predicted_x += (error_x - previous_x) * cfg.prediction
            predicted_y += (error_y - previous_y) * cfg.prediction

        smooth_x, smooth_y = predicted_x, predicted_y
        smoothing = max(0.0, min(0.95, cfg.smoothing))
        if self._last_smooth is not None and smoothing > 0:
            last_x, last_y = self._last_smooth
            smooth_x = last_x * smoothing + predicted_x * (1.0 - smoothing)
            smooth_y = last_y * smoothing + predicted_y * (1.0 - smoothing)

        self._last_error = (error_x, error_y)
        self._last_smooth = (smooth_x, smooth_y)
        dt_ms = (now_s - self._last_s) * 1000.0 if self._last_s > 0 else 0.0
        self._last_s = now_s

        counts_x = _pixels_to_counts(
            smooth_x,
            frame_width=max(1.0, center_x * 2.0),
            fov_deg=cfg.fov_deg,
            counts_per_revolution=max(1.0, cfg.counts_per_revolution_x),
        )
        counts_y = _pixels_to_counts(
            smooth_y,
            frame_width=max(1.0, center_x * 2.0),
            fov_deg=cfg.fov_deg,
            counts_per_revolution=max(1.0, cfg.counts_per_revolution_y),
        )

        if abs(error_x) < max(0.0, cfg.deadzone_px):
            counts_x = 0.0
        if abs(error_y) < max(0.0, cfg.deadzone_px):
            counts_y = 0.0

        output_x = _clamp(counts_x * cfg.kp_x, -abs(cfg.max_x), abs(cfg.max_x))
        output_y = _clamp(counts_y * cfg.kp_y, -abs(cfg.max_y), abs(cfg.max_y))
        move_kind = cfg.move_kind if cfg.move_kind in MOVE_KINDS else "raw"
        return MoveCommand(
            dx=output_x,
            dy=output_y,
            confidence=float(getattr(target, "score", 1.0)),
            reason="isolated mouse strategy",
            move_kind=move_kind,
            move_ms=max(0, int(cfg.move_ms)),
            trace_ms=max(0, int(cfg.trace_ms)),
            debug={
                "stage": "isolated_mouse_pipeline",
                "algorithm": "isolated_mouse",
                "coordinate_y": "cartesian_up_positive",
                "aim_ratio": cfg.aim_ratio,
                "aim_x": aim_x,
                "aim_y": aim_y,
                "raw_px_x": error_x,
                "raw_px_y": error_y,
                "predicted_x": predicted_x,
                "predicted_y": predicted_y,
                "smooth_x": smooth_x,
                "smooth_y": smooth_y,
                "fov_counts_x": counts_x,
                "fov_counts_y": counts_y,
                "kp_x": cfg.kp_x,
                "kp_y": cfg.kp_y,
                "deadzone_px": cfg.deadzone_px,
                "max_x": cfg.max_x,
                "max_y": cfg.max_y,
                "smoothing": cfg.smoothing,
                "prediction": cfg.prediction,
                "dt_ms": dt_ms,
                "trigger_active": bool(box_input.active),
            },
        )

    def reset(self) -> None:
        self._last_error = None
        self._last_smooth = None
        self._last_s = 0.0


def _aim_point(target: Target, aim_ratio: float) -> tuple[float, float]:
    ratio = max(0.0, min(100.0, float(aim_ratio))) / 100.0
    y = float(target.y) + float(target.h) * ratio
    return float(target.cx), y


def _pixels_to_counts(
    error_pixels: float,
    *,
    frame_width: float,
    fov_deg: float,
    counts_per_revolution: float,
) -> float:
    focal = (frame_width * 0.5) / math.tan(math.radians(max(1.0, min(179.0, fov_deg))) * 0.5)
    angle = math.degrees(math.atan(float(error_pixels) / focal)) if focal > 0 else 0.0
    return angle * (counts_per_revolution / 360.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))
