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
    trigger_required: bool | None = None
    trigger_active: bool | None = None
    command_expires_ts_ns: int | None = None


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
                dx=0,
                dy=0,
                action=intent.action,
                confidence=intent.confidence,
                source_id=intent.source_id,
                accepted=False,
                clipped=False,
                reason="non-finite control value",
                move_kind=intent.move_kind,
                move_ms=intent.move_ms,
                trace_ms=intent.trace_ms,
                bezier_ctrl=intent.bezier_ctrl,
                source_frame_id=intent.source_frame_id,
                source_track_id=intent.source_track_id,
                predicted_source=intent.predicted_source,
                trajectory_generation=intent.trajectory_generation,
                trigger_required=intent.trigger_required,
                trigger_active=intent.trigger_active,
                command_expires_ts_ns=intent.command_expires_ts_ns,
            )
        if intent.confidence < self.min_confidence:
            return ControlOutput(
                dx=0,
                dy=0,
                action=intent.action,
                confidence=intent.confidence,
                source_id=intent.source_id,
                accepted=False,
                clipped=False,
                reason="confidence below threshold",
                move_kind=intent.move_kind,
                move_ms=intent.move_ms,
                trace_ms=intent.trace_ms,
                bezier_ctrl=intent.bezier_ctrl,
                source_frame_id=intent.source_frame_id,
                source_track_id=intent.source_track_id,
                predicted_source=intent.predicted_source,
                trajectory_generation=intent.trajectory_generation,
                trigger_required=intent.trigger_required,
                trigger_active=intent.trigger_active,
                command_expires_ts_ns=intent.command_expires_ts_ns,
            )
        requested_dx = int(round(intent.dx))
        requested_dy = int(round(intent.dy))
        dx = max(-self.max_abs_dx, min(self.max_abs_dx, requested_dx))
        dy = max(-self.max_abs_dy, min(self.max_abs_dy, requested_dy))
        clipped = dx != requested_dx or dy != requested_dy
        reason = "clamped to configured limits" if clipped else intent.reason
        return ControlOutput(
            dx=dx,
            dy=dy,
            action=intent.action,
            confidence=intent.confidence,
            source_id=intent.source_id,
            accepted=True,
            clipped=clipped,
            reason=reason,
            move_kind=intent.move_kind,
            move_ms=intent.move_ms,
            trace_ms=intent.trace_ms,
            bezier_ctrl=intent.bezier_ctrl,
            source_frame_id=intent.source_frame_id,
            source_track_id=intent.source_track_id,
            predicted_source=intent.predicted_source,
            trajectory_generation=intent.trajectory_generation,
            trigger_required=intent.trigger_required,
            trigger_active=intent.trigger_active,
            command_expires_ts_ns=intent.command_expires_ts_ns,
        )
