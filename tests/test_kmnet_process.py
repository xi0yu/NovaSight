from __future__ import annotations

import multiprocessing
import os
import time

import pytest

from novasight.executors.kmnet_loader import KmNetLoadResult
from novasight.executors.kmnet_process import KmNetDriverProcess


class TestDriver:
    def init(self, *_: object) -> int:
        time.sleep(1.0)
        return 0

    def move(self, _dx: int, _dy: int) -> int:
        return 0

    def crash(self) -> None:
        os._exit(17)


def _load_test_driver() -> KmNetLoadResult:
    return KmNetLoadResult(
        module=TestDriver(),
        available=True,
        source="test",
        reason="",
        platform="test",
        machine="test",
        python_tag="test",
    )


def _spawn_context() -> multiprocessing.context.BaseContext:
    if "spawn" not in multiprocessing.get_all_start_methods():
        pytest.skip("spawn multiprocessing context is unavailable")
    return multiprocessing.get_context("spawn")


def test_driver_process_returns_successful_call_result() -> None:
    driver = KmNetDriverProcess(
        driver_loader=_load_test_driver,
        context=_spawn_context(),
    )
    try:
        assert driver.call("move", 3, -2, timeout_s=0.5) == 0
    finally:
        driver.abort()


def test_driver_process_terminates_worker_after_call_timeout() -> None:
    driver = KmNetDriverProcess(
        driver_loader=_load_test_driver,
        context=_spawn_context(),
    )
    started_at = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="init"):
            driver.call("init", "192.0.2.1", "8888", "test", timeout_s=0.05)
        assert time.monotonic() - started_at < 0.8

        # A later command starts a clean worker instead of reusing the blocked one.
        assert driver.call("move", 1, 0, timeout_s=0.5) == 0
    finally:
        driver.abort()


def test_driver_process_reports_native_worker_exit_code() -> None:
    driver = KmNetDriverProcess(
        driver_loader=_load_test_driver,
        context=_spawn_context(),
    )
    try:
        with pytest.raises(RuntimeError, match=r"crash: exitcode=17"):
            driver.call("crash", timeout_s=0.5)
    finally:
        driver.abort()
