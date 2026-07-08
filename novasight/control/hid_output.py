from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Protocol, runtime_checkable

from novasight.control.output import ControlOutput
from novasight.control.scheduler import CommandScheduler


@runtime_checkable
class ControlOutputExecutor(Protocol):
    def execute(self, output: ControlOutput) -> Any:
        ...


@runtime_checkable
class RelativeMoveDevice(Protocol):
    def send_move(self, dx: int, dy: int) -> Any:
        ...


class HidOutput:
    """Small relative-move HID facade for scheduler output loops."""

    def __init__(
        self,
        device: ControlOutputExecutor | RelativeMoveDevice,
        *,
        source_id: str = "hid_output",
        move_kind: str = "raw",
    ) -> None:
        self.device = device
        self.source_id = str(source_id)
        self.move_kind = str(move_kind or "raw")

    def send(self, dx: int, dy: int) -> Any:
        dx_i = int(dx)
        dy_i = int(dy)
        if isinstance(self.device, ControlOutputExecutor):
            output = ControlOutput(
                dx=dx_i,
                dy=dy_i,
                action="move",
                confidence=1.0,
                source_id=self.source_id,
                accepted=True,
                clipped=False,
                reason="HID relative move",
                move_kind=self.move_kind,
            )
            return self.device.execute(output)
        if isinstance(self.device, RelativeMoveDevice):
            return self.device.send_move(dx_i, dy_i)
        raise TypeError("HidOutput device must expose execute(ControlOutput) or send_move(dx, dy)")


@dataclass
class HidOutputThreadStats:
    ticks: int = 0
    sent_steps: int = 0
    failed_steps: int = 0
    last_error: str = ""
    last_sent_ts_ns: int = 0


class HidOutputThread(threading.Thread):
    def __init__(
        self,
        *,
        scheduler: CommandScheduler,
        output: HidOutput,
        poll_interval_s: float = 0.0005,
        name: str = "novasight-hid-output",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.scheduler = scheduler
        self.output = output
        self.poll_interval_s = max(0.0001, float(poll_interval_s))
        self.stats = HidOutputThreadStats()
        self._stop_event = threading.Event()

    def stop(self, *, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=max(0.0, float(timeout_s)))

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.stats.ticks += 1
            decision = self.scheduler.tick()
            if decision.output is not None:
                self._send(decision.output)
            self._stop_event.wait(self.poll_interval_s)

    def _send(self, output: ControlOutput) -> None:
        try:
            result = self.output.send(output.dx, output.dy)
        except Exception as exc:
            self.stats.failed_steps += 1
            self.stats.last_error = str(exc)
            self.scheduler.record_execution_result(sent=False, message=str(exc))
            return
        sent = bool(getattr(result, "sent", True))
        message = str(getattr(result, "message", ""))
        metadata = self.scheduler.record_execution_result(sent=sent, message=message)
        if sent:
            self.stats.sent_steps += 1
            self.stats.last_sent_ts_ns = time.monotonic_ns()
            self.stats.last_error = ""
        else:
            self.stats.failed_steps += 1
            self.stats.last_error = str(metadata.get("error") or message or "HID output failed")


__all__ = [
    "ControlOutputExecutor",
    "HidOutput",
    "HidOutputThread",
    "HidOutputThreadStats",
    "RelativeMoveDevice",
]
