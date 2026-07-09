from __future__ import annotations

import math
from dataclasses import dataclass

from novasight.contracts import ControlIntent


@dataclass(frozen=True)
class ControlOutput:
    dx: int
    dy: int
    action: str | None
    confidence: float
    source_id: str
    accepted: bool
    clipped: bool
    reason: str
    move_kind: str = "raw"
    move_ms: int = 0
    trace_ms: int = 0
    bezier_ctrl: tuple[int, int, int, int] | None = None
    source_frame_id: int | None = None
    source_track_id: int | None = None
    predicted_source: bool = False
    trajectory_generation: int | None = None


class ControlOutputPolicy:
    def __init__(
        self,
        *,
        max_abs_dx: int = 120,
        max_abs_dy: int = 120,
        min_confidence: float = 0.0,
    ) -> None:
        if max_abs_dx < 0:
            raise ValueError("max_abs_dx must be >= 0")
        if max_abs_dy < 0:
            raise ValueError("max_abs_dy must be >= 0")
        self.max_abs_dx = max_abs_dx
        self.max_abs_dy = max_abs_dy
        self.min_confidence = min_confidence

    def apply(self, intent: ControlIntent) -> ControlOutput:
        if (
            not math.isfinite(intent.dx)
            or not math.isfinite(intent.dy)
            or not math.isfinite(intent.confidence)
        ):
            return ControlOutput(
                0,
                0,
                intent.action,
                intent.confidence,
                intent.source_id,
                False,
                False,
                "non-finite control value",
                intent.move_kind,
                intent.move_ms,
                intent.trace_ms,
                intent.bezier_ctrl,
                intent.source_frame_id,
                intent.source_track_id,
                intent.predicted_source,
                intent.trajectory_generation,
            )
        if intent.confidence < self.min_confidence:
            return ControlOutput(
                0,
                0,
                intent.action,
                intent.confidence,
                intent.source_id,
                False,
                False,
                "confidence below threshold",
                intent.move_kind,
                intent.move_ms,
                intent.trace_ms,
                intent.bezier_ctrl,
                intent.source_frame_id,
                intent.source_track_id,
                intent.predicted_source,
                intent.trajectory_generation,
            )
        requested_dx = int(round(intent.dx))
        requested_dy = int(round(intent.dy))
        dx = max(-self.max_abs_dx, min(self.max_abs_dx, requested_dx))
        dy = max(-self.max_abs_dy, min(self.max_abs_dy, requested_dy))
        clipped = dx != requested_dx or dy != requested_dy
        reason = "clamped to configured limits" if clipped else intent.reason
        return ControlOutput(
            dx,
            dy,
            intent.action,
            intent.confidence,
            intent.source_id,
            True,
            clipped,
            reason,
            intent.move_kind,
            intent.move_ms,
            intent.trace_ms,
            intent.bezier_ctrl,
            intent.source_frame_id,
            intent.source_track_id,
            intent.predicted_source,
            intent.trajectory_generation,
        )
