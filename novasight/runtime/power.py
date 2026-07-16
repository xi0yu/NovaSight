from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol


logger = logging.getLogger("novasight.runtime.power")


class RuntimeLifecycle(Protocol):
    @property
    def running(self) -> bool: ...

    def start(self, reason: str) -> None: ...

    def stop(self, reason: str) -> None: ...


class CallbackRuntimeLifecycle:
    def __init__(
        self,
        *,
        is_running: Callable[[], bool],
        start: Callable[[str], None],
        stop: Callable[[str], None],
    ) -> None:
        self._is_running = is_running
        self._start = start
        self._stop = stop

    @property
    def running(self) -> bool:
        return bool(self._is_running())

    def start(self, reason: str) -> None:
        self._start(str(reason))

    def stop(self, reason: str) -> None:
        self._stop(str(reason))


class RuntimePowerSupervisor:
    """Own runtime run intent and host-presence-driven cold standby policy."""

    def __init__(
        self,
        *,
        lifecycle: RuntimeLifecycle,
        enabled: bool,
        target_host_id: str,
        heartbeat_timeout_s: float,
        offline_grace_s: float,
        auto_resume: bool,
        now_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._lifecycle = lifecycle
        self._enabled = bool(enabled)
        self._target_host_id = str(target_host_id).strip()
        if self._enabled and not self._target_host_id:
            raise ValueError("enabled host presence requires target_host_id")
        self._heartbeat_timeout_ns = int(max(0.1, heartbeat_timeout_s) * 1e9)
        self._offline_grace_ns = int(max(0.0, offline_grace_s) * 1e9)
        self._auto_resume = bool(auto_resume)
        self._now_ns = now_ns
        self._lock = threading.RLock()
        self._run_intent = False
        self._suspended_by_policy = False
        self._last_heartbeat_ns = 0
        self._offline_since_ns = 0
        self._host_id = ""
        self._reason = "power saving disabled" if not self._enabled else "runtime stopped"
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None

    def request_start(self) -> dict[str, Any]:
        with self._lock:
            self._run_intent = True
            if self._enabled and not self._host_online_locked(int(self._now_ns())):
                self._suspended_by_policy = True
                self._reason = "waiting for target host heartbeat"
                return self._status_locked()
            self._lifecycle.start("USER_START")
            self._suspended_by_policy = False
            self._reason = "runtime active"
            return self._status_locked()

    def request_stop(self) -> dict[str, Any]:
        with self._lock:
            self._run_intent = False
            self._suspended_by_policy = False
            self._offline_since_ns = 0
            self._lifecycle.stop("USER_STOP")
            self._reason = "runtime stopped by user"
            return self._status_locked()

    def heartbeat(self, host_id: str) -> dict[str, Any]:
        normalized_host_id = str(host_id).strip()
        if not normalized_host_id:
            raise ValueError("host heartbeat requires a non-empty host_id")
        if self._target_host_id and normalized_host_id != self._target_host_id:
            raise ValueError(
                f"heartbeat host_id {normalized_host_id!r} does not match configured "
                f"target_host_id {self._target_host_id!r}"
            )
        with self._lock:
            self._host_id = normalized_host_id
            self._last_heartbeat_ns = int(self._now_ns())
            self._offline_since_ns = 0
            if (
                self._enabled
                and self._run_intent
                and self._auto_resume
                and self._suspended_by_policy
                and not self._lifecycle.running
            ):
                self._lifecycle.start("HOST_ONLINE")
                self._suspended_by_policy = False
            if self._lifecycle.running:
                self._reason = "runtime active"
            elif self._run_intent and not self._suspended_by_policy:
                self._reason = "runtime stopped outside power-saving policy"
            else:
                self._reason = "target host online"
            return self._status_locked()

    def tick(self) -> dict[str, Any]:
        with self._lock:
            now_ns = int(self._now_ns())
            if not self._enabled or not self._run_intent or self._last_heartbeat_ns <= 0:
                return self._status_locked(now_ns=now_ns)
            if now_ns - self._last_heartbeat_ns <= self._heartbeat_timeout_ns:
                self._offline_since_ns = 0
                return self._status_locked(now_ns=now_ns)
            if self._offline_since_ns <= 0:
                self._offline_since_ns = now_ns
                self._reason = "target host heartbeat expired; offline grace active"
                if self._offline_grace_ns > 0:
                    return self._status_locked(now_ns=now_ns)
            if (
                self._lifecycle.running
                and now_ns - self._offline_since_ns >= self._offline_grace_ns
            ):
                self._lifecycle.stop("HOST_OFFLINE")
                self._suspended_by_policy = True
                self._reason = "target host offline; runtime suspended"
            return self._status_locked(now_ns=now_ns)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def start_monitoring(self, *, interval_s: float = 0.5) -> None:
        with self._lock:
            if not self._enabled:
                return
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return
            self._monitor_stop.clear()
            thread = threading.Thread(
                target=self._monitor_loop,
                args=(max(0.1, float(interval_s)),),
                name="novasight-runtime-power",
                daemon=True,
            )
            self._monitor_thread = thread
            thread.start()

    def close(self) -> None:
        self._monitor_stop.set()
        with self._lock:
            thread = self._monitor_thread
            self._monitor_thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _monitor_loop(self, interval_s: float) -> None:
        while not self._monitor_stop.wait(interval_s):
            try:
                self.tick()
            except Exception as exc:
                # Lifecycle failures remain visible through runtime fatal/status reporting;
                # keep monitoring so a later heartbeat can recover the requested runtime.
                logger.warning("runtime power transition failed: %s", exc)
                continue

    def _host_online_locked(self, now_ns: int) -> bool:
        return bool(
            self._last_heartbeat_ns > 0
            and int(now_ns) - self._last_heartbeat_ns <= self._heartbeat_timeout_ns
        )

    def _status_locked(self, *, now_ns: int | None = None) -> dict[str, Any]:
        current_ns = int(self._now_ns()) if now_ns is None else int(now_ns)
        running = bool(self._lifecycle.running)
        if not self._enabled:
            mode = "disabled"
        elif running and self._offline_since_ns > 0:
            mode = "grace"
        elif running:
            mode = "active"
        elif self._run_intent and self._suspended_by_policy:
            mode = "cold_standby"
        elif self._run_intent:
            mode = "interrupted"
        else:
            mode = "stopped"
        return {
            "enabled": self._enabled,
            "mode": mode,
            "run_intent": self._run_intent,
            "suspended_by_policy": self._suspended_by_policy,
            "running": running,
            "host_id": self._host_id,
            "target_host_id": self._target_host_id,
            "host_online": self._host_online_locked(current_ns),
            "heartbeat_age_ms": (
                max(0.0, (current_ns - self._last_heartbeat_ns) / 1e6)
                if self._last_heartbeat_ns > 0
                else None
            ),
            "auto_resume": self._auto_resume,
            "reason": self._reason,
        }


__all__ = [
    "CallbackRuntimeLifecycle",
    "RuntimeLifecycle",
    "RuntimePowerSupervisor",
]
