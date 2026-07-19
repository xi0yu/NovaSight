from __future__ import annotations

import logging
import threading
import time
from typing import Any

from novasight.config import RuntimeConfig
from novasight.control import ControlOutput
from novasight.executors.contracts import ExecutionResult
from novasight.executors.kmnet_diagnostics import probe_kmnet_route
from novasight.executors.kmnet_loader import load_kmnet_driver
from novasight.executors.kmnet_process import KmNetDriverProcess

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
        button_poll_interval_s: float = 0.004,
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
        self.connection_stage = "idle"
        self.last_connect_error_stage = ""
        self.last_connect_error_type = ""
        self.last_driver_call = ""
        self.last_driver_call_duration_ms = 0.0
        self.last_driver_rc: Any | None = None
        self.last_driver_error = ""
        self.route_available: bool | None = None
        self.route_resolved_ip = ""
        self.route_local_ip = ""
        self.route_error = ""
        self.last_button_left = False
        self.last_button_right = False
        self.last_button_available = False
        self.last_button_reason = ""
        self.last_button_raw: dict[str, Any] = {}
        self.last_button_sample_ts_ns = 0
        self.last_button_poll_ts_ns = 0
        self.left_pressed_since_ts_ns = 0
        self.right_pressed_since_ts_ns = 0
        self.button_poll_interval_s = max(0.001, min(0.050, float(button_poll_interval_s)))
        self._button_state_lock = threading.Lock()
        self._button_poll_stop = threading.Event()
        self._button_poll_thread: threading.Thread | None = None
        self._connection_lock = threading.Lock()
        self._connection_generation = 0
        self._connect_thread: threading.Thread | None = None
        result = load_kmnet_driver()
        self._driver: Any | None = result.module
        self._driver_process = (
            KmNetDriverProcess(driver_loader=load_kmnet_driver)
            if result.available and result.source != "test"
            else None
        )
        self.driver_source = result.source
        self.driver_platform = result.platform
        self.driver_machine = result.machine
        self.driver_python = result.python_tag
        self.driver_load_error = result.reason
        if not result.available:
            self.last_error = result.reason
            self.connection_stage = "driver_unavailable"
            self.last_connect_error_stage = "driver_load"
            self.last_connect_error_type = "unavailable"

    @classmethod
    def from_config(cls, config: RuntimeConfig) -> KmNetExecutor:
        return cls(
            host=config.hardware.host,
            port=config.hardware.port,
            uuid=config.hardware.uuid,
            monitor_port=(
                config.hardware.monitor_port
                if (
                    config.control.trigger_mode == "hardware"
                    or bool(config.control.recoil.enabled)
                )
                else 0
            ),
            button_poll_interval_s=float(config.control.scheduler_interval_ms) / 1000.0,
        )

    def available(self) -> bool:
        return self._driver is not None

    def status(self, *, refresh_buttons: bool = False) -> dict[str, Any]:
        del refresh_buttons
        buttons = self.read_buttons()
        connection_state = self._connection_state()
        sample_ts_ns = int(buttons["sample_ts_ns"])
        sample_age_ms = (
            max(0.0, (time.monotonic_ns() - sample_ts_ns) / 1_000_000.0)
            if sample_ts_ns > 0
            else None
        )
        return {
            "available": self.available(),
            "connected": self.connected,
            "connecting": self.connecting,
            "connection_state": connection_state,
            "retryable": connection_state == "failed" and self.available(),
            "connection_stage": self.connection_stage,
            "last_connect_error_stage": self.last_connect_error_stage,
            "last_connect_error_type": self.last_connect_error_type,
            "monitoring": self.monitoring,
            "move_count": self.move_count,
            "last_dx": self.last_dx,
            "last_dy": self.last_dy,
            "last_error": self.last_error,
            "last_driver_call": self.last_driver_call,
            "last_driver_call_duration_ms": self.last_driver_call_duration_ms,
            "last_driver_rc": self.last_driver_rc,
            "last_driver_error": self.last_driver_error,
            "route_available": self.route_available,
            "route_resolved_ip": self.route_resolved_ip,
            "route_local_ip": self.route_local_ip,
            "route_error": self.route_error,
            "driver_source": self.driver_source,
            "driver_platform": self.driver_platform,
            "driver_machine": self.driver_machine,
            "driver_python": self.driver_python,
            "host": self.host,
            "port": self.port,
            "monitor_port": self.monitor_port,
            "has_move_auto": self._driver_has("move_auto"),
            "has_enc_move": self._driver_has("enc_move"),
            "has_enc_move_auto": self._driver_has("enc_move_auto"),
            "has_move_bezier": self._driver_has("move_beizer"),
            "has_trace": self._driver_has("trace"),
            "has_left_button": self._driver_has("isdown_left"),
            "has_right_button": self._driver_has("isdown_right"),
            "button_available": buttons["available"],
            "button_left": buttons["left"],
            "button_right": buttons["right"],
            "button_reason": buttons["reason"],
            "button_raw": buttons["raw"],
            "button_sample_ts_ns": sample_ts_ns,
            "button_sample_age_ms": sample_age_ms,
            "button_poll_ts_ns": buttons["poll_ts_ns"],
            "left_pressed_since_ts_ns": buttons["left_pressed_since_ts_ns"],
            "right_pressed_since_ts_ns": buttons["right_pressed_since_ts_ns"],
            "button_polling": (
                self._button_poll_thread is not None
                and self._button_poll_thread.is_alive()
            ),
            "button_poll_interval_ms": self.button_poll_interval_s * 1000.0,
        }

    def _connection_state(self) -> str:
        if self.connecting:
            return "connecting"
        if self.connected:
            return "degraded" if self.connection_stage == "connected_without_monitor" else "connected"
        if self.connection_stage in {"failed", "driver_unavailable"}:
            return "failed"
        return "disconnected"

    def read_buttons(self) -> dict[str, Any]:
        with self._button_state_lock:
            return {
                "available": bool(self.last_button_available),
                "left": bool(self.last_button_left),
                "right": bool(self.last_button_right),
                "reason": str(self.last_button_reason),
                "raw": dict(self.last_button_raw),
                "sample_ts_ns": int(self.last_button_sample_ts_ns),
                "poll_ts_ns": int(self.last_button_poll_ts_ns),
                "left_pressed_since_ts_ns": int(self.left_pressed_since_ts_ns),
                "right_pressed_since_ts_ns": int(self.right_pressed_since_ts_ns),
            }

    def _record_buttons(
        self,
        available: bool,
        left: bool,
        right: bool,
        reason: str,
        *,
        raw: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now_ns = time.monotonic_ns()
        with self._button_state_lock:
            previous_left = bool(self.last_button_left)
            previous_right = bool(self.last_button_right)
            previous_trigger_active = bool(
                self.last_button_available and (previous_left or previous_right)
            )
            self.last_button_poll_ts_ns = now_ns
            self.last_button_available = bool(available)
            self.last_button_left = bool(left)
            self.last_button_right = bool(right)
            self.last_button_reason = str(reason or "")
            self.last_button_raw = dict(raw or {})
            if available and left and not previous_left:
                self.left_pressed_since_ts_ns = now_ns
            elif not available or not left:
                self.left_pressed_since_ts_ns = 0
            if available and right and not previous_right:
                self.right_pressed_since_ts_ns = now_ns
            elif not available or not right:
                self.right_pressed_since_ts_ns = 0
            if available:
                self.last_button_sample_ts_ns = now_ns
            trigger_active = bool(available and (left or right))
            should_log = bool(
                trigger_active
                and (
                    not previous_trigger_active
                    or bool(left) != previous_left
                    or bool(right) != previous_right
                )
            )
            result = {
                "available": bool(available),
                "left": bool(left),
                "right": bool(right),
                "reason": str(reason or ""),
                "raw": dict(self.last_button_raw),
                "sample_ts_ns": int(self.last_button_sample_ts_ns),
                "poll_ts_ns": int(self.last_button_poll_ts_ns),
                "left_pressed_since_ts_ns": int(self.left_pressed_since_ts_ns),
                "right_pressed_since_ts_ns": int(self.right_pressed_since_ts_ns),
            }
        if should_log:
            logger.info(
                "kmNet trigger active left=%s right=%s raw=%s",
                left,
                right,
                result["raw"],
            )
        return result

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
            self.connection_stage = "idle"
            self.last_error = ""
            button_poll_stop = self._button_poll_stop
            button_poll_thread = self._button_poll_thread
            self._button_poll_thread = None
        button_poll_stop.set()
        if self._driver_process is not None:
            self._driver_process.abort()
        if button_poll_thread is not None and button_poll_thread is not threading.current_thread():
            button_poll_thread.join(timeout=0.5)
        self._record_buttons(False, False, False, "kmNet is not connected")
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
            if output.trace_ms > 0 and self._driver_has("trace"):
                self._call_driver("trace", 0, int(output.trace_ms))
            logger.debug(
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
            if self._driver is None:
                self.connection_stage = "driver_unavailable"
                self.last_connect_error_stage = "driver_load"
                self.last_connect_error_type = "unavailable"
                self.last_error = self.driver_load_error or "kmNet driver unavailable"
                return None
            self._connection_generation += 1
            self.connecting = True
            self.connection_stage = "network"
            self.last_connect_error_stage = ""
            self.last_connect_error_type = ""
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
                    self.connection_stage = "failed"
                    self.last_connect_error_stage = "config"
                    self.last_connect_error_type = "invalid_port"
                    self.last_error = "kmNet port is not configured"
                    self.connected = False
            return
        route = probe_kmnet_route(self.host, self.port)
        with self._connection_lock:
            if generation != self._connection_generation:
                return
            self.route_available = route.available
            self.route_resolved_ip = route.resolved_ip
            self.route_local_ip = route.local_ip
            self.route_error = route.error
            self.connection_stage = "init" if route.available else "failed"
        if not route.available:
            self._record_connect_failure(
                generation,
                "network",
                RuntimeError(route.error or "no IPv4 route to kmNet host"),
                error_type="unreachable",
            )
            return
        try:
            self._call_init_driver()
        except Exception as exc:
            self._record_connect_failure(generation, "init", exc)
            return
        with self._connection_lock:
            if generation != self._connection_generation:
                logger.info("kmNet connect result ignored after disconnect")
                return
            self.connected = True
            self.monitoring = False
            self.connection_stage = "monitor" if self.monitor_port > 0 else "connected"
            self.last_error = ""
        if self.monitor_port > 0:
            try:
                self._call_driver("monitor", int(self.monitor_port))
            except Exception as exc:
                with self._connection_lock:
                    if generation != self._connection_generation:
                        return
                    self.monitoring = False
                    self.connection_stage = "connected_without_monitor"
                    self.last_connect_error_stage = "monitor"
                    self.last_connect_error_type = self._connect_error_type(exc)
                    self.last_error = f"kmNet monitor failed: {exc}"
                logger.warning(
                    "kmNet connected but monitor failed host=%s port=%s monitor_port=%s error=%s",
                    self.host,
                    self.port,
                    self.monitor_port,
                    exc,
                )
                self._record_buttons(False, False, False, self.last_error)
                return
            with self._connection_lock:
                if generation != self._connection_generation:
                    return
                self.connected = True
                self.monitoring = True
                self.connection_stage = "connected"
                self.last_connect_error_stage = ""
                self.last_connect_error_type = ""
                self.last_error = ""
            self._record_buttons(False, False, False, "awaiting first kmNet button sample")
            self._start_button_poller(generation)
        else:
            self._record_buttons(False, False, False, "kmNet monitor_port is not configured")
        logger.info(
            "kmNet connected host=%s port=%s monitor_port=%s monitoring=%s",
            self.host,
            self.port,
            self.monitor_port,
            self.monitoring,
        )

    def _start_button_poller(self, generation: int) -> None:
        stop = threading.Event()
        thread = threading.Thread(
            target=self._button_poll_loop,
            args=(generation, stop),
            name="novasight-kmnet-buttons",
            daemon=True,
        )
        with self._connection_lock:
            if generation != self._connection_generation or not self.monitoring:
                return
            previous_stop = self._button_poll_stop
            self._button_poll_stop = stop
            self._button_poll_thread = thread
        previous_stop.set()
        thread.start()

    def _button_poll_loop(self, generation: int, stop: threading.Event) -> None:
        while not stop.is_set():
            with self._connection_lock:
                active = (
                    generation == self._connection_generation
                    and self.connected
                    and self.monitoring
                )
            if not active:
                return
            try:
                left_raw, right_raw = self._read_buttons_raw()
                with self._connection_lock:
                    if generation != self._connection_generation:
                        return
                available = bool(left_raw["exists"] or right_raw["exists"])
                reason = "" if available else "kmNet button query functions are unavailable"
                self._record_buttons(
                    available,
                    bool(left_raw["pressed"]),
                    bool(right_raw["pressed"]),
                    reason,
                    raw={"left": left_raw, "right": right_raw},
                )
            except Exception as exc:
                with self._connection_lock:
                    if generation != self._connection_generation:
                        return
                reason = f"kmNet button read failed: {exc}"
                self.last_error = reason
                self._record_buttons(False, False, False, reason)
                if isinstance(exc, TimeoutError):
                    with self._connection_lock:
                        if generation == self._connection_generation:
                            self.connected = False
                            self.monitoring = False
                            self.connection_stage = "failed"
                            self.last_connect_error_stage = "button_poll"
                            self.last_connect_error_type = "timeout"
                    return
            stop.wait(self.button_poll_interval_s)

    def _record_connect_failure(
        self,
        generation: int,
        stage: str,
        exc: Exception,
        *,
        error_type: str | None = None,
    ) -> None:
        with self._connection_lock:
            if generation != self._connection_generation:
                return
            self.connection_stage = "failed"
            self.last_connect_error_stage = stage
            self.last_connect_error_type = error_type or self._connect_error_type(exc)
            self.last_error = f"kmNet {stage} failed: {exc}"
            self.connected = False
            self.monitoring = False
        logger.warning(
            "kmNet connect failed stage=%s type=%s host=%s port=%s monitor_port=%s error=%s",
            stage,
            self.last_connect_error_type,
            self.host,
            self.port,
            self.monitor_port,
            exc,
        )

    @staticmethod
    def _connect_error_type(exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return "timeout"
        if "failed rc=" in str(exc):
            return "driver_return_code"
        return "driver_exception"

    def _move_auto(self, dx: int, dy: int, move_ms: int, *, encrypted: bool = False) -> tuple[str, Any]:
        name = "enc_move_auto" if encrypted else "move_auto"
        if not self._driver_has(name):
            name = "enc_move" if encrypted else "move"
        if not self._driver_has(name):
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
        if not self._driver_has(name):
            name = alternate
        if not self._driver_has(name):
            fallback = "enc_move" if encrypted else "move"
            rc = self._call_driver(fallback, dx, dy)
            return fallback, rc
        x1, y1, x2, y2 = ctrl
        rc = self._call_driver(name, dx, dy, int(move_ms), int(x1), int(y1), int(x2), int(y2))
        return name, rc

    def _call_driver(self, name: str, *args: Any) -> Any:
        if self._driver is None:
            raise RuntimeError("driver unavailable")
        if not self._driver_has(name):
            raise RuntimeError(f"driver function unavailable: {name}")
        started_ns = time.monotonic_ns()
        self.last_driver_call = name
        self.last_driver_rc = None
        self.last_driver_error = ""
        try:
            if self._driver_process is not None:
                timeout_s = 3.0 if name == "init" else 2.0 if name == "monitor" else 1.0
                rc = self._driver_process.call(name, *args, timeout_s=timeout_s)
            else:
                rc = getattr(self._driver, name)(*args)
            self.last_driver_rc = rc
            if self._driver_return_code_is_error(name, rc):
                raise RuntimeError(f"{name} failed rc={rc}")
        except Exception as exc:
            self.last_driver_error = str(exc)
            raise
        finally:
            self.last_driver_call_duration_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
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
            logger.debug("kmNet driver call name=%s args=%s rc=%s", name, args, rc)
        return rc

    @staticmethod
    def _driver_return_code_is_error(name: str, rc: Any) -> bool:
        if name in {"isdown_left", "isdown_right"}:
            try:
                return int(rc) not in {0, 1}
            except (TypeError, ValueError):
                return True
        return rc not in (None, 0)

    def _call_init_driver(self) -> Any:
        return self._call_driver("init", self.host, str(self.port), self.uuid)

    def _read_button_raw(self, name: str) -> dict[str, Any]:
        if not self._driver_has(name):
            return {"function": name, "exists": False, "value": None, "pressed": False}
        value = self._call_driver(name)
        return self._button_payload(name, value)

    def _read_buttons_raw(self) -> tuple[dict[str, Any], dict[str, Any]]:
        names = ("isdown_left", "isdown_right")
        available_names = [name for name in names if self._driver_has(name)]
        if self._driver_process is None or len(available_names) < 2:
            return (
                self._read_button_raw(names[0]),
                self._read_button_raw(names[1]),
            )

        started_ns = time.monotonic_ns()
        self.last_driver_call = "+".join(available_names)
        self.last_driver_rc = None
        self.last_driver_error = ""
        try:
            values = self._driver_process.call_many(
                [(name, ()) for name in available_names],
                timeout_s=1.0,
            )
            for name, value in zip(available_names, values, strict=True):
                if self._driver_return_code_is_error(name, value):
                    raise RuntimeError(f"{name} failed rc={value}")
            self.last_driver_rc = tuple(values)
        except Exception as exc:
            self.last_driver_error = str(exc)
            raise
        finally:
            self.last_driver_call_duration_ms = (
                time.monotonic_ns() - started_ns
            ) / 1_000_000.0
        payloads = {
            name: self._button_payload(name, value)
            for name, value in zip(available_names, values, strict=True)
        }
        return payloads[names[0]], payloads[names[1]]

    @staticmethod
    def _button_payload(name: str, value: Any) -> dict[str, Any]:
        try:
            pressed = int(value) == 1
        except (TypeError, ValueError):
            pressed = bool(value)
        return {"function": name, "exists": True, "value": value, "pressed": pressed}

    def _driver_has(self, name: str) -> bool:
        return self._driver is not None and hasattr(self._driver, name)
