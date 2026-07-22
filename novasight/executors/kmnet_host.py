"""Narrow crash-isolation host for the proprietary kmNet CPython extension.

The Rust daemon owns this process and communicates only through one JSON object
per line on stdin/stdout. Diagnostics go to stderr so protocol framing cannot be
corrupted by application logs.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .kmnet_loader import load_kmnet_driver


PROTOCOL_VERSION = 1


class HostProtocol:
    def __init__(self) -> None:
        loaded = load_kmnet_driver()
        self.driver = loaded.module
        self.driver_source = loaded.source
        self.driver_error = loaded.reason
        self.connected = False

    def handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        request_id = _required_integer(request, "id")
        operation = _required_string(request, "op")
        if operation == "hello":
            protocol = _required_integer(request, "protocol")
            if protocol != PROTOCOL_VERSION:
                raise ValueError(
                    f"protocol version mismatch: expected {PROTOCOL_VERSION}, got {protocol}"
                )
            return (
                _success(
                    request_id,
                    {
                        "protocol": PROTOCOL_VERSION,
                        "driver_available": self.driver is not None,
                        "driver_source": self.driver_source,
                    },
                ),
                False,
            )
        if operation == "shutdown":
            self.connected = False
            return _success(request_id, {"stopped": True}), True
        if operation == "connect":
            self._connect(request)
            return _success(request_id, {"connected": True}), False
        if operation == "move":
            self._require_connected()
            dx = _signed_16(request, "dx")
            dy = _signed_16(request, "dy")
            result = self._call("move", dx, dy)
            return _success(request_id, {"driver_rc": result}), False
        if operation == "buttons":
            self._require_connected()
            left_exists, left = self._button("isdown_left")
            right_exists, right = self._button("isdown_right")
            return (
                _success(
                    request_id,
                    {
                        "available": left_exists or right_exists,
                        "left": left,
                        "right": right,
                    },
                ),
                False,
            )
        raise ValueError(f"unsupported operation: {operation}")

    def _connect(self, request: dict[str, Any]) -> None:
        if self.driver is None:
            raise RuntimeError(self.driver_error or "kmNet driver unavailable")
        host = _required_string(request, "host")
        port = _required_integer(request, "port")
        uuid = _required_string(request, "uuid")
        monitor_port = _required_integer(request, "monitor_port")
        if not 1 <= port <= 65535:
            raise ValueError("port must be within 1..65535")
        if not 0 <= monitor_port <= 65535:
            raise ValueError("monitor_port must be within 0..65535")
        self._call("init", host, str(port), uuid)
        if monitor_port > 0:
            self._call("monitor", monitor_port)
        self.connected = True

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("kmNet driver is not connected")

    def _call(self, name: str, *args: Any) -> Any:
        if self.driver is None:
            raise RuntimeError("kmNet driver unavailable")
        function = getattr(self.driver, name, None)
        if not callable(function):
            raise RuntimeError(f"driver function unavailable: {name}")
        result = function(*args)
        if result not in (None, 0):
            raise RuntimeError(f"{name} failed rc={result}")
        return result

    def _button(self, name: str) -> tuple[bool, bool]:
        if self.driver is None:
            return False, False
        function = getattr(self.driver, name, None)
        if not callable(function):
            return False, False
        result = function()
        if result not in (0, 1, False, True):
            raise RuntimeError(f"{name} failed rc={result}")
        return True, bool(result)


def _required_integer(request: dict[str, Any], field: str) -> int:
    value = request.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _required_string(request: dict[str, Any], field: str) -> str:
    value = request.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _signed_16(request: dict[str, Any], field: str) -> int:
    value = _required_integer(request, field)
    if not -(2**15) <= value <= 2**15 - 1:
        raise ValueError(f"{field} is outside signed 16-bit range")
    return value


def _success(request_id: int, result: dict[str, Any]) -> dict[str, Any]:
    return {"id": request_id, "ok": True, "result": result, "error": None}


def _failure(request_id: int, error: BaseException) -> dict[str, Any]:
    return {
        "id": request_id,
        "ok": False,
        "result": None,
        "error": f"{type(error).__name__}: {error}",
    }


def main() -> int:
    protocol = HostProtocol()
    for line in sys.stdin:
        request_id = 0
        stop = False
        try:
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            raw_id = payload.get("id")
            if isinstance(raw_id, int) and not isinstance(raw_id, bool):
                request_id = raw_id
            response, stop = protocol.handle(payload)
        except Exception as error:
            response = _failure(request_id, error)
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        if stop:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
