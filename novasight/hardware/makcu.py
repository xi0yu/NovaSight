from __future__ import annotations

from .contracts import BoxInputState


class MAKCUAdapter:
    def __init__(self, port: str, *, baudrate: int = 115200, timeout_s: float = 0.02) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout_s = timeout_s
        self._serial = None
        self._last_input = BoxInputState()

    def connect(self) -> None:
        import serial

        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout_s)

    def send_move(self, dx: int, dy: int) -> None:
        self._write(f"move {dx} {dy}\n")

    def send_click(self, button: str) -> None:
        self._write(f"click {button}\n")

    def get_input_state(self) -> BoxInputState:
        serial_obj = self._require_serial()
        data = serial_obj.readline()
        if not data:
            return self._last_input
        text = data.decode("utf-8", errors="ignore").strip().lower()
        state = BoxInputState(
            left="left=1" in text or "l=1" in text,
            right="right=1" in text or "r=1" in text,
            side="side=1" in text or "x=1" in text,
            raw={"packet": text},
        )
        self._last_input = state
        return state

    def _write(self, command: str) -> None:
        serial_obj = self._require_serial()
        serial_obj.write(command.encode("ascii"))

    def _require_serial(self):
        if self._serial is None:
            raise RuntimeError("hardware box is not connected")
        return self._serial
