from __future__ import annotations

import logging
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import ControlOutput
from novasight.executors.contracts import ExecutionResult
from novasight.executors.kmnet_loader import load_kmnet_driver

logger = logging.getLogger("novasight.executors.kmnet")


class KmNetExecutor:
    executor_id = "kmnet"

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        uuid: str = "",
        monitor_port: int = 0,
        flip_dy: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.uuid = uuid
        self.monitor_port = monitor_port
        self.flip_dy = flip_dy
        self.connected = False
        self.monitoring = False
        self.move_count = 0
        self.last_dx = 0
        self.last_dy = 0
        self.last_error = ""
        result = load_kmnet_driver()
        self._driver: Any | None = result.module
        self.driver_source = result.source
        self.driver_platform = result.platform
        self.driver_machine = result.machine
        self.driver_python = result.python_tag
        if not result.available:
            self.last_error = result.reason

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> KmNetExecutor:
        return cls(
            host=config.hardware.host,
            port=config.hardware.port,
            uuid=config.hardware.uuid,
            monitor_port=config.hardware.monitor_port,
            flip_dy=config.hardware.flip_dy,
        )

    def available(self) -> bool:
        return self._driver is not None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available(),
            "connected": self.connected,
            "monitoring": self.monitoring,
            "move_count": self.move_count,
            "last_dx": self.last_dx,
            "last_dy": self.last_dy,
            "last_error": self.last_error,
            "driver_source": self.driver_source,
            "driver_platform": self.driver_platform,
            "driver_machine": self.driver_machine,
            "driver_python": self.driver_python,
            "host": self.host,
            "port": self.port,
            "monitor_port": self.monitor_port,
            "has_move_auto": self._driver is not None and hasattr(self._driver, "move_auto"),
            "has_enc_move": self._driver is not None and hasattr(self._driver, "enc_move"),
            "has_enc_move_auto": self._driver is not None and hasattr(self._driver, "enc_move_auto"),
            "has_move_bezier": self._driver is not None and hasattr(self._driver, "move_beizer"),
            "has_trace": self._driver is not None and hasattr(self._driver, "trace"),
            "has_left_button": self._driver is not None and hasattr(self._driver, "isdown_left"),
            "has_right_button": self._driver is not None and hasattr(self._driver, "isdown_right"),
        }

    def read_buttons(self) -> dict[str, Any]:
        if self._driver is None:
            return {"available": False, "left": False, "right": False, "reason": self.last_error}
        if not self.monitoring:
            return {"available": False, "left": False, "right": False, "reason": "kmNet monitor is not enabled"}
        try:
            left = self._read_button("isdown_left")
            right = self._read_button("isdown_right")
        except Exception as exc:
            self.last_error = f"kmNet button read failed: {exc}"
            return {"available": False, "left": False, "right": False, "reason": self.last_error}
        return {"available": True, "left": left, "right": right, "reason": ""}

    def diagnostic_move(self, dx: int, dy: int, move_kind: str | None = None) -> ExecutionResult:
        output = ControlOutput(
            dx=int(dx),
            dy=int(dy),
            action="move",
            confidence=1.0,
            source_id="doctor.kmnet",
            accepted=True,
            clipped=False,
            reason="kmNet diagnostic move",
            move_kind=move_kind or "raw",
            move_ms=12,
        )
        return self.execute(output)

    def connect(self) -> dict[str, Any]:
        self._connect()
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        self.connected = False
        self.monitoring = False
        self.last_error = ""
        return self.status()

    def execute(self, output: ControlOutput) -> ExecutionResult:
        if not output.accepted:
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=output,
                message="control output rejected",
            )
        if self._driver is None:
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=output,
                message="kmNet driver unavailable",
            )
        if not self.connected:
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=output,
                message=self.last_error or "kmNet not connected",
            )

        dx = int(output.dx)
        dy = int(-output.dy if self.flip_dy else output.dy)
        api_name = "move"
        try:
            if output.action == "left_down":
                api_name = "left"
                self._call_driver("left", 1)
            elif output.action == "left_up":
                api_name = "left"
                self._call_driver("left", 0)
            elif output.move_kind in {"auto", "enc_auto"}:
                api_name = self._move_auto(dx, dy, output.move_ms, encrypted=output.move_kind == "enc_auto")
            elif output.move_kind in {"bezier", "enc_bezier"} and output.bezier_ctrl is not None:
                api_name = self._move_bezier(
                    dx,
                    dy,
                    output.move_ms,
                    output.bezier_ctrl,
                    encrypted=output.move_kind == "enc_bezier",
                )
            else:
                api_name = "enc_move" if output.move_kind == "enc_raw" else "move"
                self._call_driver(api_name, dx, dy)
            self.move_count += 1
            self.last_dx = dx
            self.last_dy = dy
            self.last_error = ""
            if output.trace_ms > 0 and hasattr(self._driver, "trace"):
                self._call_driver("trace", 0, int(output.trace_ms))
            logger.info(
                "kmNet output sent api=%s action=%s kind=%s dx=%s dy=%s source=%s",
                api_name,
                output.action,
                output.move_kind,
                dx,
                dy,
                output.source_id,
            )
        except Exception as exc:
            self.last_error = f"kmNet send failed: {exc}"
            logger.warning(
                "kmNet output failed api=%s action=%s kind=%s dx=%s dy=%s source=%s error=%s",
                api_name,
                output.action,
                output.move_kind,
                dx,
                dy,
                output.source_id,
                exc,
            )
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=False,
                intent=output,
                message=self.last_error,
            )
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=True,
            intent=output,
            message="sent output to kmNet driver",
        )

    def _connect(self) -> None:
        if self._driver is None:
            return
        if self.connected:
            return
        if self.port <= 0:
            self.last_error = "kmNet port is not configured"
            self.connected = False
            return
        try:
            self._call_init_driver()
            self.connected = True
            self.last_error = ""
            if self.monitor_port > 0:
                self._call_driver("monitor", int(self.monitor_port))
                self.monitoring = True
            logger.info(
                "kmNet connected host=%s port=%s monitor_port=%s",
                self.host,
                self.port,
                self.monitor_port,
            )
        except Exception as exc:
            self.last_error = f"kmNet init failed: {exc}"
            self.connected = False
            logger.warning(
                "kmNet connect failed host=%s port=%s monitor_port=%s error=%s",
                self.host,
                self.port,
                self.monitor_port,
                exc,
            )

    def _move_auto(self, dx: int, dy: int, move_ms: int, *, encrypted: bool = False) -> str:
        name = "enc_move_auto" if encrypted else "move_auto"
        if getattr(self._driver, name, None) is None:
            name = "enc_move" if encrypted else "move"
        if getattr(self._driver, name, None) is None:
            fallback = "move"
            self._call_driver(fallback, dx, dy)
            return fallback
        if name.endswith("move_auto"):
            self._call_driver(name, dx, dy, int(move_ms))
        else:
            self._call_driver(name, dx, dy)
        return name

    def _move_bezier(
        self,
        dx: int,
        dy: int,
        move_ms: int,
        ctrl: tuple[int, int, int, int],
        *,
        encrypted: bool = False,
    ) -> str:
        name = "enc_move_beizer" if encrypted else "move_beizer"
        alternate = "enc_move_bezier" if encrypted else "move_bezier"
        if getattr(self._driver, name, None) is None:
            name = alternate
        if getattr(self._driver, name, None) is None:
            fallback = "enc_move" if encrypted else "move"
            self._call_driver(fallback, dx, dy)
            return fallback
        x1, y1, x2, y2 = ctrl
        if self.flip_dy:
            y1 = -y1
            y2 = -y2
        self._call_driver(name, dx, dy, int(move_ms), int(x1), int(y1), int(x2), int(y2))
        return name

    def _call_driver(self, name: str, *args: Any) -> Any:
        if self._driver is None:
            raise RuntimeError("driver unavailable")
        fn = getattr(self._driver, name, None)
        if fn is None:
            raise RuntimeError(f"driver function unavailable: {name}")
        rc = fn(*args)
        if rc not in (None, 0):
            raise RuntimeError(f"{name} failed rc={rc}")
        return rc

    def _call_init_driver(self) -> Any:
        return self._call_driver("init", self.host, str(self.port), self.uuid)

    def _read_button(self, name: str) -> bool:
        fn = getattr(self._driver, name, None)
        if fn is None:
            return False
        return int(fn()) == 1
