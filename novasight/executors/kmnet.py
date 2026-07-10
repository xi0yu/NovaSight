from __future__ import annotations

import logging
import threading
import time
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
    ) -> None:
        self.host = host
        self.port = port
        self.uuid = uuid
        self.monitor_port = monitor_port
        self.connected = False
        self.connecting = False
        self.monitoring = False
        self.move_count = 0
        self.last_dx = 0
        self.last_dy = 0
        self.last_error = ""
        self.last_button_left = False
        self.last_button_right = False
        self.last_button_available = False
        self.last_button_reason = ""
        self.last_button_raw: dict[str, Any] = {}
        self._last_button_log_signature = ""
        self._last_button_log_s = 0.0
        self._connection_lock = threading.Lock()
        self._connection_generation = 0
        self._connect_thread: threading.Thread | None = None
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
        )

    def available(self) -> bool:
        return self._driver is not None

    def status(self, *, refresh_buttons: bool = False) -> dict[str, Any]:
        if refresh_buttons and self._driver is not None and self.monitoring:
            self.read_buttons()
        return {
            "available": self.available(),
            "connected": self.connected,
            "connecting": self.connecting,
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
            "button_available": self.last_button_available,
            "button_left": self.last_button_left,
            "button_right": self.last_button_right,
            "button_reason": self.last_button_reason,
            "button_raw": self.last_button_raw,
        }

    def read_buttons(self) -> dict[str, Any]:
        if self._driver is None:
            return self._record_buttons(False, False, False, self.last_error)
        if not self.monitoring:
            if not self.connected:
                return self._record_buttons(False, False, False, "kmNet is not connected")
            if self.monitor_port <= 0:
                return self._record_buttons(False, False, False, "kmNet monitor_port is not configured")
            try:
                self._call_driver("monitor", int(self.monitor_port))
                self.monitoring = True
                logger.info("kmNet monitor auto-started port=%s", self.monitor_port)
            except Exception as exc:
                self.last_error = f"kmNet monitor start failed: {exc}"
                return self._record_buttons(False, False, False, self.last_error)
        try:
            left_raw = self._read_button_raw("isdown_left")
            right_raw = self._read_button_raw("isdown_right")
            left = left_raw["pressed"]
            right = right_raw["pressed"]
            raw = {
                "left": left_raw,
                "right": right_raw,
            }
        except Exception as exc:
            self.last_error = f"kmNet button read failed: {exc}"
            return self._record_buttons(False, False, False, self.last_error)
        return self._record_buttons(True, left, right, "", raw=raw)

    def _record_buttons(
        self,
        available: bool,
        left: bool,
        right: bool,
        reason: str,
        *,
        raw: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.last_button_available = bool(available)
        self.last_button_left = bool(left)
        self.last_button_right = bool(right)
        self.last_button_reason = str(reason or "")
        self.last_button_raw = raw or {}
        signature = f"available={available}|left={left}|right={right}|reason={reason}|raw={self.last_button_raw}"
        now = time.monotonic()
        if signature != self._last_button_log_signature or now - self._last_button_log_s >= 1.0:
            self._last_button_log_signature = signature
            self._last_button_log_s = now
            logger.info(
                "kmNet buttons available=%s left=%s right=%s reason=%s raw=%s",
                available,
                left,
                right,
                reason or "",
                self.last_button_raw,
            )
        return {
            "available": bool(available),
            "left": bool(left),
            "right": bool(right),
            "reason": str(reason or ""),
            "raw": self.last_button_raw,
        }

    def diagnostic_move(
        self,
        dx: int,
        dy: int,
        move_kind: str | None = None,
        move_ms: int = 12,
        bezier_ctrl: tuple[int, int, int, int] | None = None,
    ) -> ExecutionResult:
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
            move_ms=max(0, int(move_ms)),
            bezier_ctrl=bezier_ctrl,
        )
        return self.execute(output)

    def connect(self) -> dict[str, Any]:
        generation = self._begin_connect()
        if generation is not None:
            self._run_connect_attempt(generation)
        return self.status()

    def connect_async(self) -> dict[str, Any]:
        generation = self._begin_connect()
        if generation is None:
            return self.status()
        thread = threading.Thread(
            target=self._run_connect_attempt,
            args=(generation,),
            name="novasight-kmnet-connect",
            daemon=True,
        )
        with self._connection_lock:
            self._connect_thread = thread
        try:
            thread.start()
        except Exception:
            with self._connection_lock:
                if generation == self._connection_generation:
                    self.connecting = False
                    self._connect_thread = None
            raise
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        with self._connection_lock:
            self._connection_generation += 1
            self.connected = False
            self.connecting = False
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
        dy = int(output.dy)
        api_name = "move"
        driver_rc: Any | None = None
        try:
            if output.action == "left_down":
                api_name = "left"
                driver_rc = self._call_driver("left", 1)
            elif output.action == "left_up":
                api_name = "left"
                driver_rc = self._call_driver("left", 0)
            elif output.move_kind in {"auto", "enc_auto"}:
                api_name, driver_rc = self._move_auto(dx, dy, output.move_ms, encrypted=output.move_kind == "enc_auto")
            elif output.move_kind in {"bezier", "enc_bezier"} and output.bezier_ctrl is not None:
                api_name, driver_rc = self._move_bezier(
                    dx,
                    dy,
                    output.move_ms,
                    output.bezier_ctrl,
                    encrypted=output.move_kind == "enc_bezier",
                )
            else:
                api_name = "enc_move" if output.move_kind == "enc_raw" else "move"
                driver_rc = self._call_driver(api_name, dx, dy)
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
                metadata={
                    "stage": "driver",
                    "api_name": api_name,
                    "driver_rc": None,
                    "driver_dx": dx,
                    "driver_dy": dy,
                    "error": str(exc),
                },
            )
        return ExecutionResult(
            executor_id=self.executor_id,
            sent=True,
            intent=output,
            message="sent output to kmNet driver",
            metadata={
                "stage": "driver",
                "api_name": api_name,
                "driver_rc": driver_rc,
                "driver_dx": dx,
                "driver_dy": dy,
                "move_count": self.move_count,
            },
        )

    def _begin_connect(self) -> int | None:
        with self._connection_lock:
            if self.connected or self.connecting:
                return None
            self._connection_generation += 1
            self.connecting = True
            self.last_error = ""
            return self._connection_generation

    def _run_connect_attempt(self, generation: int) -> None:
        try:
            self._connect(generation)
        finally:
            current_thread = threading.current_thread()
            with self._connection_lock:
                if generation == self._connection_generation:
                    self.connecting = False
                if self._connect_thread is current_thread:
                    self._connect_thread = None

    def _connect(self, generation: int) -> None:
        if self._driver is None:
            return
        if self.port <= 0:
            with self._connection_lock:
                if generation == self._connection_generation:
                    self.last_error = "kmNet port is not configured"
                    self.connected = False
            return
        try:
            self._call_init_driver()
            monitoring = False
            if self.monitor_port > 0:
                self._call_driver("monitor", int(self.monitor_port))
                monitoring = True
            with self._connection_lock:
                if generation != self._connection_generation:
                    logger.info("kmNet connect result ignored after disconnect")
                    return
                self.connected = True
                self.monitoring = monitoring
                self.last_error = ""
            logger.info(
                "kmNet connected host=%s port=%s monitor_port=%s",
                self.host,
                self.port,
                self.monitor_port,
            )
        except Exception as exc:
            with self._connection_lock:
                if generation != self._connection_generation:
                    return
                self.last_error = f"kmNet init failed: {exc}"
                self.connected = False
                self.monitoring = False
            logger.warning(
                "kmNet connect failed host=%s port=%s monitor_port=%s error=%s",
                self.host,
                self.port,
                self.monitor_port,
                exc,
            )

    def _move_auto(self, dx: int, dy: int, move_ms: int, *, encrypted: bool = False) -> tuple[str, Any]:
        name = "enc_move_auto" if encrypted else "move_auto"
        if getattr(self._driver, name, None) is None:
            name = "enc_move" if encrypted else "move"
        if getattr(self._driver, name, None) is None:
            fallback = "move"
            rc = self._call_driver(fallback, dx, dy)
            return fallback, rc
        if name.endswith("move_auto"):
            rc = self._call_driver(name, dx, dy, int(move_ms))
        else:
            rc = self._call_driver(name, dx, dy)
        return name, rc

    def _move_bezier(
        self,
        dx: int,
        dy: int,
        move_ms: int,
        ctrl: tuple[int, int, int, int],
        *,
        encrypted: bool = False,
    ) -> tuple[str, Any]:
        name = "enc_move_beizer" if encrypted else "move_beizer"
        alternate = "enc_move_bezier" if encrypted else "move_bezier"
        if getattr(self._driver, name, None) is None:
            name = alternate
        if getattr(self._driver, name, None) is None:
            fallback = "enc_move" if encrypted else "move"
            rc = self._call_driver(fallback, dx, dy)
            return fallback, rc
        x1, y1, x2, y2 = ctrl
        rc = self._call_driver(name, dx, dy, int(move_ms), int(x1), int(y1), int(x2), int(y2))
        return name, rc

    def _call_driver(self, name: str, *args: Any) -> Any:
        if self._driver is None:
            raise RuntimeError("driver unavailable")
        fn = getattr(self._driver, name, None)
        if fn is None:
            raise RuntimeError(f"driver function unavailable: {name}")
        rc = fn(*args)
        if name in {
            "init",
            "monitor",
            "move",
            "enc_move",
            "move_auto",
            "enc_move_auto",
            "move_beizer",
            "move_bezier",
            "enc_move_beizer",
            "enc_move_bezier",
            "left",
            "trace",
        }:
            logger.info("kmNet driver call name=%s args=%s rc=%s", name, args, rc)
        if rc not in (None, 0):
            raise RuntimeError(f"{name} failed rc={rc}")
        return rc

    def _call_init_driver(self) -> Any:
        return self._call_driver("init", self.host, str(self.port), self.uuid)

    def _read_button_raw(self, name: str) -> dict[str, Any]:
        fn = getattr(self._driver, name, None)
        if fn is None:
            return {"function": name, "exists": False, "value": None, "pressed": False}
        value = fn()
        try:
            pressed = int(value) == 1
        except (TypeError, ValueError):
            pressed = bool(value)
        return {"function": name, "exists": True, "value": value, "pressed": pressed}
