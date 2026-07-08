from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from novasight.control.output import ControlOutput


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


__all__ = ["ControlOutputExecutor", "HidOutput", "RelativeMoveDevice"]
