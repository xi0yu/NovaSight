from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import time
from typing import Any

from novasight.control.output import ControlOutput


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    output: ControlOutput | None
    metadata: dict[str, Any]

    @property
    def should_send(self) -> bool:
        return self.output is not None


class CommandScheduler:
    """Production command scheduler.

    The scheduler keeps the latest command only. This matches the
    remediation contract: a new measurement supersedes old pending motion
    instead of merging old counts into the next send.
    """

    def __init__(
        self,
        *,
        min_interval_s: float = 0.0,
        ttl_s: float = 0.050,
        predicted_ttl_s: float | None = None,
        cancel_on_new_frame: bool = True,
        cancel_on_direction_change: bool = True,
        cancel_on_track_change: bool = True,
        max_step_x: int = 20,
        max_step_y: int = 20,
        queue_hard_limit: int = 64,
        device_error_cooldown_s: float = 0.050,
    ) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.ttl_s = max(0.0, float(ttl_s))
        self.predicted_ttl_s = (
            max(0.0, float(predicted_ttl_s))
            if predicted_ttl_s is not None
            else max(0.0, float(ttl_s))
        )
        self.cancel_on_new_frame = bool(cancel_on_new_frame)
        self.cancel_on_direction_change = bool(cancel_on_direction_change)
        self.cancel_on_track_change = bool(cancel_on_track_change)
        self.max_step_x = max(1, int(max_step_x))
        self.max_step_y = max(1, int(max_step_y))
        self.queue_hard_limit = max(1, int(queue_hard_limit))
        self.device_error_cooldown_s = max(0.0, float(device_error_cooldown_s))
        self._last_emit_s = 0.0
        self._pending_steps: deque[ControlOutput] = deque()
        self._pending_parent: ControlOutput | None = None
        self._pending_created_s = 0.0
        self._pending_command_id = 0
        self._pending_expires_s = 0.0
        self._cancelled_pending = 0
        self._expired_pending = 0
        self._throttled_since_emit = 0
        self._command_seq = 0
        self._cooldown_until_s = 0.0
        self._last_cancel_reason = ""
        self._last_error = ""

    def submit(self, output: ControlOutput, *, now_s: float | None = None) -> ScheduleDecision:
        now = time.monotonic() if now_s is None else float(now_s)
        if self._cooldown_until_s > now:
            return ScheduleDecision(
                output=None,
                metadata={
                    "stage": "scheduler",
                    "action": "cooldown",
                    "command_status": "cooldown",
                    "sent_allowed": False,
                    "cooldown_remaining_ms": (self._cooldown_until_s - now) * 1000.0,
                    "last_error": self._last_error,
                },
            )
        if not output.accepted:
            self.clear("rejected_output")
            return ScheduleDecision(
                output=output,
                metadata={
                    "stage": "scheduler",
                    "action": "pass_rejected",
                    "command_status": "rejected",
                    "sent_allowed": True,
                    "reason": "rejected outputs bypass scheduler state",
                },
            )

        expired_reason = self._drop_expired(now)
        cancel_reason = self._cancel_pending_for_new_output(output)
        elapsed = now - self._last_emit_s if self._last_emit_s > 0 else self.min_interval_s
        if _interval_due(elapsed, self.min_interval_s):
            cancelled = len(self._pending_steps)
            self._cancelled_pending += cancelled
            self._pending_steps.clear()
            self._pending_parent = None
            self._pending_created_s = 0.0
            self._pending_command_id = 0
            self._pending_expires_s = 0.0
            self._last_emit_s = now
            throttled = self._throttled_since_emit
            self._throttled_since_emit = 0
            command_id = self._next_command_id()
            expires_s = now + self._ttl_for(output)
            step, pending_steps, split_metadata = self._split_output(output)
            self._pending_steps = deque(pending_steps)
            self._pending_parent = output if pending_steps else None
            self._pending_command_id = command_id if pending_steps else 0
            self._pending_created_s = now if pending_steps else 0.0
            self._pending_expires_s = expires_s if pending_steps else 0.0
            return ScheduleDecision(
                output=step,
                metadata={
                    "stage": "scheduler",
                    "action": "emit_step",
                    "command_id": command_id,
                    "source_frame_id": output.source_frame_id,
                    "source_track_id": output.source_track_id,
                    "trajectory_generation": output.trajectory_generation,
                    "created_ts_ns": int(now * 1_000_000_000),
                    "expires_ts_ns": int(expires_s * 1_000_000_000),
                    "command_status": "ready",
                    "cancel_reason": cancel_reason or ("LATEST_SUPERSEDES_PENDING" if cancelled else ""),
                    "expired_reason": expired_reason,
                    "sent_allowed": True,
                    "elapsed_ms": elapsed * 1000.0,
                    "min_interval_ms": self.min_interval_s * 1000.0,
                    "ttl_ms": self._ttl_for(output) * 1000.0,
                    "cancelled_pending": cancelled,
                    "expired_pending": self._expired_pending,
                    "throttled_since_emit": throttled,
                    **split_metadata,
                },
            )

        replaced = bool(self._pending_steps)
        command_id = self._next_command_id()
        first_step, tail_steps, split_metadata = self._split_output(output)
        steps = [first_step, *tail_steps]
        self._pending_steps = deque(steps)
        self._pending_parent = output
        self._pending_created_s = now
        self._pending_command_id = command_id
        self._pending_expires_s = now + self._ttl_for(output)
        self._throttled_since_emit += 1
        if replaced:
            self._cancelled_pending += 1
        return ScheduleDecision(
            output=None,
            metadata={
                "stage": "scheduler",
                "action": "hold_latest",
                "command_id": command_id,
                "source_frame_id": output.source_frame_id,
                "source_track_id": output.source_track_id,
                "trajectory_generation": output.trajectory_generation,
                "created_ts_ns": int(now * 1_000_000_000),
                "expires_ts_ns": int(self._pending_expires_s * 1_000_000_000),
                "command_status": "pending",
                "cancel_reason": cancel_reason or ("REPLACED_PENDING" if replaced else ""),
                "expired_reason": expired_reason,
                "sent_allowed": False,
                "reason": "minimum interval not elapsed",
                "elapsed_ms": elapsed * 1000.0,
                "min_interval_ms": self.min_interval_s * 1000.0,
                "ttl_ms": self._ttl_for(output) * 1000.0,
                "replaced_pending": replaced,
                **split_metadata,
                "pending_dx": float(sum(step.dx for step in self._pending_steps)),
                "pending_dy": float(sum(step.dy for step in self._pending_steps)),
                "pending_steps": len(self._pending_steps),
                "throttled_since_emit": self._throttled_since_emit,
            },
        )

    def clear(self, reason: str = "") -> None:
        if reason:
            self._last_cancel_reason = reason
        if self._pending_steps:
            self._cancelled_pending += len(self._pending_steps)
        self._pending_steps.clear()
        self._pending_parent = None
        self._pending_created_s = 0.0
        self._pending_command_id = 0
        self._pending_expires_s = 0.0
        self._throttled_since_emit = 0

    def tick(self, *, now_s: float | None = None) -> ScheduleDecision:
        """Emit the next pending step when pacing allows it.

        Output loops can call this at a higher frequency than the HID device
        rate. The scheduler keeps the latest pending command and only releases
        one already-split step per tick once the minimum interval has elapsed.
        """

        now = time.monotonic() if now_s is None else float(now_s)
        if self._cooldown_until_s > now:
            return ScheduleDecision(
                output=None,
                metadata={
                    "stage": "scheduler",
                    "action": "cooldown",
                    "command_status": "cooldown",
                    "sent_allowed": False,
                    "cooldown_remaining_ms": (self._cooldown_until_s - now) * 1000.0,
                    "last_error": self._last_error,
                },
            )

        expired_reason = self._drop_expired(now)
        if not self._pending_steps:
            return ScheduleDecision(
                output=None,
                metadata={
                    "stage": "scheduler",
                    "action": "idle",
                    "command_status": "idle",
                    "sent_allowed": False,
                    "expired_reason": expired_reason,
                },
            )

        elapsed = now - self._last_emit_s if self._last_emit_s > 0 else self.min_interval_s
        if not _interval_due(elapsed, self.min_interval_s):
            return ScheduleDecision(
                output=None,
                metadata={
                    "stage": "scheduler",
                    "action": "hold_pending",
                    "command_id": self._pending_command_id,
                    "source_frame_id": self._pending_parent.source_frame_id
                    if self._pending_parent is not None
                    else None,
                    "source_track_id": self._pending_parent.source_track_id
                    if self._pending_parent is not None
                    else None,
                    "trajectory_generation": self._pending_parent.trajectory_generation
                    if self._pending_parent is not None
                    else None,
                    "command_status": "pending",
                    "sent_allowed": False,
                    "reason": "minimum interval not elapsed",
                    "elapsed_ms": elapsed * 1000.0,
                    "min_interval_ms": self.min_interval_s * 1000.0,
                    "pending_dx": float(sum(step.dx for step in self._pending_steps)),
                    "pending_dy": float(sum(step.dy for step in self._pending_steps)),
                    "pending_steps": len(self._pending_steps),
                    "expired_reason": expired_reason,
                },
            )

        step = self._pending_steps.popleft()
        pending_dx = float(sum(pending.dx for pending in self._pending_steps))
        pending_dy = float(sum(pending.dy for pending in self._pending_steps))
        pending_steps = len(self._pending_steps)
        command_id = self._pending_command_id
        parent = self._pending_parent
        self._last_emit_s = now
        if not self._pending_steps:
            self._pending_parent = None
            self._pending_created_s = 0.0
            self._pending_command_id = 0
            self._pending_expires_s = 0.0
        return ScheduleDecision(
            output=step,
            metadata={
                "stage": "scheduler",
                "action": "emit_pending_step",
                "command_id": command_id,
                "source_frame_id": parent.source_frame_id if parent is not None else step.source_frame_id,
                "source_track_id": parent.source_track_id if parent is not None else step.source_track_id,
                "trajectory_generation": parent.trajectory_generation if parent is not None else step.trajectory_generation,
                "command_status": "ready",
                "sent_allowed": True,
                "elapsed_ms": elapsed * 1000.0,
                "min_interval_ms": self.min_interval_s * 1000.0,
                "emitted_step_dx": int(step.dx),
                "emitted_step_dy": int(step.dy),
                "pending_dx": pending_dx,
                "pending_dy": pending_dy,
                "pending_steps": pending_steps,
                "expired_reason": expired_reason,
            },
        )

    def record_execution_result(
        self,
        *,
        sent: bool,
        message: str = "",
        now_s: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic() if now_s is None else float(now_s)
        if sent:
            return {
                "stage": "scheduler",
                "command_status": "sent",
                "cooldown": False,
            }
        self.clear("DEVICE_ERROR")
        self._last_error = str(message or "device send failed")
        if self.device_error_cooldown_s > 0:
            self._cooldown_until_s = now + self.device_error_cooldown_s
        return {
            "stage": "scheduler",
            "command_status": "error",
            "cancel_reason": "DEVICE_ERROR",
            "cooldown": self.device_error_cooldown_s > 0,
            "cooldown_until_ts_ns": int(self._cooldown_until_s * 1_000_000_000),
            "error": self._last_error,
        }

    def status(self, *, now_s: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now_s is None else float(now_s)
        self._drop_expired(now)
        return {
            "enabled": self.min_interval_s > 0,
            "min_interval_ms": self.min_interval_s * 1000.0,
            "ttl_ms": self.ttl_s * 1000.0,
            "predicted_ttl_ms": self.predicted_ttl_s * 1000.0,
            "has_pending": bool(self._pending_steps),
            "pending_command_id": self._pending_command_id,
            "pending_dx": float(sum(step.dx for step in self._pending_steps)),
            "pending_dy": float(sum(step.dy for step in self._pending_steps)),
            "pending_steps": len(self._pending_steps),
            "pending_created_ts_ns": int(self._pending_created_s * 1_000_000_000),
            "pending_age_ms": (
                max(0.0, (now - self._pending_created_s) * 1000.0)
                if self._pending_created_s > 0.0 and self._pending_steps
                else 0.0
            ),
            "max_step_x": self.max_step_x,
            "max_step_y": self.max_step_y,
            "queue_hard_limit": self.queue_hard_limit,
            "pending_source_frame_id": self._pending_parent.source_frame_id if self._pending_parent is not None else None,
            "pending_source_track_id": self._pending_parent.source_track_id if self._pending_parent is not None else None,
            "pending_trajectory_generation": self._pending_parent.trajectory_generation if self._pending_parent is not None else None,
            "pending_expires_ts_ns": int(self._pending_expires_s * 1_000_000_000),
            "cancelled_pending": self._cancelled_pending,
            "expired_pending": self._expired_pending,
            "throttled_since_emit": self._throttled_since_emit,
            "cooldown_active": self._cooldown_until_s > now,
            "cooldown_until_ts_ns": int(self._cooldown_until_s * 1_000_000_000),
            "last_cancel_reason": self._last_cancel_reason,
            "last_error": self._last_error,
        }

    def _drop_expired(self, now: float) -> str:
        if not self._pending_steps or self.ttl_s <= 0:
            return ""
        if now >= self._pending_expires_s:
            self._expired_pending += 1
            self.clear("pending_expired")
            return "EXPIRED"
        return ""

    def _cancel_pending_for_new_output(self, output: ControlOutput) -> str:
        pending = self._pending_parent
        if pending is None:
            return ""
        if (
            pending.trajectory_generation is not None
            and output.trajectory_generation is not None
            and pending.trajectory_generation != output.trajectory_generation
        ):
            self.clear("TRAJECTORY_GENERATION")
            return "TRAJECTORY_GENERATION"
        if (
            self.cancel_on_track_change
            and pending.source_track_id is not None
            and output.source_track_id is not None
            and pending.source_track_id != output.source_track_id
        ):
            self.clear("TRACK_CHANGE")
            return "TRACK_CHANGE"
        if (
            self.cancel_on_new_frame
            and pending.source_frame_id is not None
            and output.source_frame_id is not None
            and pending.source_frame_id != output.source_frame_id
        ):
            self.clear("NEW_FRAME")
            return "NEW_FRAME"
        if self.cancel_on_direction_change and _direction_changed(pending, output):
            self.clear("DIRECTION_CHANGE")
            return "DIRECTION_CHANGE"
        return ""

    def cancel_pending_before_generation(self, generation: int) -> bool:
        pending = self._pending_parent
        if pending is None or pending.trajectory_generation is None:
            return False
        if int(pending.trajectory_generation) >= int(generation):
            return False
        self.clear("TRAJECTORY_GENERATION")
        return True

    def _ttl_for(self, output: ControlOutput) -> float:
        return self.predicted_ttl_s if output.predicted_source else self.ttl_s

    def _next_command_id(self) -> int:
        self._command_seq += 1
        return self._command_seq

    def _split_output(self, output: ControlOutput) -> tuple[ControlOutput, list[ControlOutput], dict[str, Any]]:
        steps = _split_steps(
            output,
            max_step_x=self.max_step_x,
            max_step_y=self.max_step_y,
            queue_hard_limit=self.queue_hard_limit,
        )
        first = steps[0]
        tail = steps[1:]
        return first, tail, {
            "split_steps_total": len(steps),
            "emitted_step_dx": int(first.dx),
            "emitted_step_dy": int(first.dy),
            "pending_steps": len(tail),
            "pending_dx": float(sum(step.dx for step in tail)),
            "pending_dy": float(sum(step.dy for step in tail)),
            "max_step_x": self.max_step_x,
            "max_step_y": self.max_step_y,
            "queue_hard_limit": self.queue_hard_limit,
        }


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _interval_due(elapsed_s: float, min_interval_s: float) -> bool:
    if min_interval_s <= 0:
        return True
    return elapsed_s + 1e-12 >= min_interval_s


def _direction_changed(previous: ControlOutput, current: ControlOutput) -> bool:
    prev_x = _sign(int(previous.dx))
    prev_y = _sign(int(previous.dy))
    next_x = _sign(int(current.dx))
    next_y = _sign(int(current.dy))
    return (prev_x != 0 and next_x != 0 and prev_x != next_x) or (
        prev_y != 0 and next_y != 0 and prev_y != next_y
    )


def _split_steps(
    output: ControlOutput,
    *,
    max_step_x: int,
    max_step_y: int,
    queue_hard_limit: int,
) -> list[ControlOutput]:
    dx = int(output.dx)
    dy = int(output.dy)
    if dx == 0 and dy == 0:
        return [output]
    step_count = max(
        1,
        math.ceil(abs(dx) / max(1, int(max_step_x))),
        math.ceil(abs(dy) / max(1, int(max_step_y))),
    )
    step_count = min(step_count, max(1, int(queue_hard_limit)))
    steps: list[ControlOutput] = []
    previous_x = 0
    previous_y = 0
    for index in range(1, step_count + 1):
        target_x = int(round(dx * index / step_count))
        target_y = int(round(dy * index / step_count))
        step_x = target_x - previous_x
        step_y = target_y - previous_y
        previous_x = target_x
        previous_y = target_y
        steps.append(replace(output, dx=step_x, dy=step_y))
    return steps or [output]


Scheduler = CommandScheduler
