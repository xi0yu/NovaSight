from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Any

from novasight.executors.kmnet_loader import KmNetLoadResult, load_kmnet_driver


DriverLoader = Callable[[], KmNetLoadResult]


def _run_kmnet_driver_worker(connection: Connection, driver_loader: DriverLoader) -> None:
    result = driver_loader()
    driver = result.module
    try:
        while True:
            try:
                request_id, name, args = connection.recv()
            except EOFError:
                return
            if name == "__shutdown__":
                return
            if driver is None:
                connection.send((request_id, False, None, result.reason or "kmNet driver unavailable"))
                continue
            try:
                function = getattr(driver, name)
                value = function(*args)
            except Exception as exc:
                connection.send((request_id, False, None, f"{type(exc).__name__}: {exc}"))
            else:
                connection.send((request_id, True, value, ""))
    finally:
        connection.close()


class KmNetDriverProcess:
    def __init__(
        self,
        *,
        driver_loader: DriverLoader = load_kmnet_driver,
        context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        self._driver_loader = driver_loader
        self._context = context or multiprocessing.get_context("spawn")
        self._request_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._connection: Connection | None = None
        self._process: multiprocessing.Process | None = None
        self._request_id = 0

    def call(self, name: str, *args: Any, timeout_s: float) -> Any:
        with self._request_lock:
            connection = self._ensure_worker()
            self._request_id += 1
            request_id = self._request_id
            try:
                connection.send((request_id, name, args))
                if not connection.poll(max(0.01, float(timeout_s))):
                    raise TimeoutError(f"kmNet driver call timed out: {name}")
                response_id, succeeded, value, error = connection.recv()
            except (EOFError, OSError, TimeoutError):
                self.abort()
                raise
            if response_id != request_id:
                self.abort()
                raise RuntimeError(
                    f"kmNet driver response mismatch: expected {request_id}, received {response_id}"
                )
            if not succeeded:
                raise RuntimeError(str(error or f"kmNet driver call failed: {name}"))
            return value

    def abort(self) -> None:
        with self._lifecycle_lock:
            connection = self._connection
            process = self._process
            self._connection = None
            self._process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
            process.join(timeout=0.5)

    def _ensure_worker(self) -> Connection:
        with self._lifecycle_lock:
            if (
                self._connection is not None
                and self._process is not None
                and self._process.is_alive()
            ):
                return self._connection
            if self._connection is not None:
                self._connection.close()
            if self._process is not None:
                self._process.join(timeout=0.1)
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_run_kmnet_driver_worker,
                args=(child_connection, self._driver_loader),
                name="novasight-kmnet-driver",
                daemon=True,
            )
            try:
                process.start()
            except Exception:
                parent_connection.close()
                child_connection.close()
                raise
            child_connection.close()
            self._connection = parent_connection
            self._process = process
            return parent_connection


__all__ = ["KmNetDriverProcess"]
