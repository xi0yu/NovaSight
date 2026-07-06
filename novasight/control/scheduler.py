from __future__ import annotations

from dataclasses import dataclass
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
    ) -> None:
        self.min_interval_s = max(0.0, float(min_interval_s))
        self.ttl_s = max(0.0, float(ttl_s))
        self._last_emit_s = 0.0
        self._pending: ControlOutput | None = None
        self._pending_created_s = 0.0
        self._cancelled_pending = 0
        self._throttled_since_emit = 0

    def submit(self, output: ControlOutput, *, now_s: float | None = None) -> ScheduleDecision:
        now = time.monotonic() if now_s is None else float(now_s)
        if not output.accepted:
            self.clear("rejected_output")
            return ScheduleDecision(
                output=output,
                metadata={
                    "stage": "scheduler",
                    "action": "pass_rejected",
                    "sent_allowed": True,
                    "reason": "rejected outputs bypass scheduler state",
                },
            )

        self._drop_expired(now)
        elapsed = now - self._last_emit_s if self._last_emit_s > 0 else self.min_interval_s
        if self.min_interval_s <= 0 or elapsed >= self.min_interval_s:
            cancelled = 1 if self._pending is not None else 0
            self._cancelled_pending += cancelled
            self._pending = None
            self._pending_created_s = 0.0
            self._last_emit_s = now
            throttled = self._throttled_since_emit
            self._throttled_since_emit = 0
            return ScheduleDecision(
                output=output,
                metadata={
                    "stage": "scheduler",
                    "action": "emit_latest",
                    "sent_allowed": True,
                    "elapsed_ms": elapsed * 1000.0,
                    "min_interval_ms": self.min_interval_s * 1000.0,
                    "cancelled_pending": cancelled,
                    "throttled_since_emit": throttled,
                },
            )

        replaced = self._pending is not None
        self._pending = output
        self._pending_created_s = now
        self._throttled_since_emit += 1
        if replaced:
            self._cancelled_pending += 1
        return ScheduleDecision(
            output=None,
            metadata={
                "stage": "scheduler",
                "action": "hold_latest",
                "sent_allowed": False,
                "reason": "minimum interval not elapsed",
                "elapsed_ms": elapsed * 1000.0,
                "min_interval_ms": self.min_interval_s * 1000.0,
                "replaced_pending": replaced,
                "pending_dx": float(output.dx),
                "pending_dy": float(output.dy),
                "throttled_since_emit": self._throttled_since_emit,
            },
        )

    def clear(self, reason: str = "") -> None:
        if self._pending is not None:
            self._cancelled_pending += 1
        self._pending = None
        self._pending_created_s = 0.0
        self._throttled_since_emit = 0

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.min_interval_s > 0,
            "min_interval_ms": self.min_interval_s * 1000.0,
            "ttl_ms": self.ttl_s * 1000.0,
            "has_pending": self._pending is not None,
            "pending_dx": float(self._pending.dx) if self._pending is not None else 0.0,
            "pending_dy": float(self._pending.dy) if self._pending is not None else 0.0,
            "cancelled_pending": self._cancelled_pending,
            "throttled_since_emit": self._throttled_since_emit,
        }

    def _drop_expired(self, now: float) -> None:
        if self._pending is None or self.ttl_s <= 0:
            return
        if now - self._pending_created_s > self.ttl_s:
            self.clear("pending_expired")
