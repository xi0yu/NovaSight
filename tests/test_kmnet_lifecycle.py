from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from novasight.api.app import (
    _auto_connect_kmnet,
    _disconnect_kmnet,
    _start_auto_connect_kmnet,
    _start_auto_restore_capture,
)
from novasight.api.routes_executors import disconnect_kmnet
from novasight.config import RuntimeConfig
from novasight.executors.kmnet import KmNetExecutor


class FakeKmNet:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self) -> dict[str, object]:
        self.connect_calls += 1
        if self.fail:
            raise RuntimeError("connect failed")
        self.connected = True
        return {
            "connected": True,
            "host": "192.168.2.188",
            "port": 8888,
            "monitor_port": 5001,
        }

    def disconnect(self) -> dict[str, object]:
        self.disconnect_calls += 1
        self.connected = False
        return {"connected": False}


class FakeScheduler:
    def __init__(self) -> None:
        self.clear_reasons: list[str] = []

    def clear(self, reason: str) -> None:
        self.clear_reasons.append(reason)


class BlockingKmNet(FakeKmNet):
    def __init__(self) -> None:
        super().__init__()
        self.connect_started = threading.Event()
        self.release_connect = threading.Event()

    def connect(self) -> dict[str, object]:
        self.connect_calls += 1
        self.connect_started.set()
        self.release_connect.wait(timeout=2.0)
        self.connected = True
        return {"connected": True}


class BlockingCapture:
    def __init__(self) -> None:
        self.configure_started = threading.Event()
        self.release_configure = threading.Event()

    def configure(self, **_: object) -> SimpleNamespace:
        self.configure_started.set()
        self.release_configure.wait(timeout=2.0)
        return SimpleNamespace(available=False)


class BlockingDriver:
    def __init__(self) -> None:
        self.init_started = threading.Event()
        self.release_init = threading.Event()

    def init(self, *_: object) -> int:
        self.init_started.set()
        self.release_init.wait(timeout=2.0)
        return 0


def _registry(kmnet: FakeKmNet, scheduler: FakeScheduler | None = None) -> SimpleNamespace:
    return SimpleNamespace(executors={"kmnet": kmnet}, scheduler=scheduler)


def test_backend_startup_auto_connects_kmnet_by_default() -> None:
    kmnet = FakeKmNet()

    _auto_connect_kmnet(_registry(kmnet), RuntimeConfig())

    assert kmnet.connect_calls == 1
    assert kmnet.connected is True


def test_backend_startup_does_not_wait_for_blocking_kmnet_connect() -> None:
    kmnet = BlockingKmNet()

    started_at = time.monotonic()
    thread = _start_auto_connect_kmnet(_registry(kmnet), RuntimeConfig())
    elapsed_s = time.monotonic() - started_at

    try:
        assert elapsed_s < 0.2
        assert thread is not None
        assert kmnet.connect_started.wait(timeout=0.2)
        assert kmnet.connected is False
    finally:
        kmnet.release_connect.set()
        if thread is not None:
            thread.join(timeout=1.0)

    assert kmnet.connected is True


def test_backend_startup_does_not_wait_for_blocking_capture_restore() -> None:
    capture = BlockingCapture()
    config = RuntimeConfig()
    config.source.default = "capture"

    started_at = time.monotonic()
    thread = _start_auto_restore_capture(capture, config)
    elapsed_s = time.monotonic() - started_at

    try:
        assert elapsed_s < 0.2
        assert thread is not None
        assert capture.configure_started.wait(timeout=0.2)
    finally:
        capture.release_configure.set()
        if thread is not None:
            thread.join(timeout=1.0)


def test_manual_disconnect_invalidates_inflight_connect_result() -> None:
    driver = BlockingDriver()
    executor = KmNetExecutor(host="192.0.2.1", port=8888, uuid="test")
    executor._driver = driver
    executor._driver_process = None

    status = executor.connect_async()

    assert status["connecting"] is True
    assert driver.init_started.wait(timeout=0.2)
    disconnected = executor.disconnect()
    assert disconnected["connecting"] is False
    assert disconnected["connected"] is False

    driver.release_init.set()
    thread = executor._connect_thread
    if thread is not None:
        thread.join(timeout=1.0)

    assert executor.status()["connected"] is False


def test_backend_startup_respects_disabled_auto_connect() -> None:
    config = RuntimeConfig()
    config.hardware.auto_connect = False
    kmnet = FakeKmNet()

    _auto_connect_kmnet(_registry(kmnet), config)

    assert kmnet.connect_calls == 0
    assert kmnet.connected is False


def test_backend_startup_survives_kmnet_connect_failure() -> None:
    kmnet = FakeKmNet(fail=True)

    _auto_connect_kmnet(_registry(kmnet), RuntimeConfig())

    assert kmnet.connect_calls == 1
    assert kmnet.connected is False


def test_manual_disconnect_cancels_pending_and_returns_device_state() -> None:
    kmnet = FakeKmNet()
    kmnet.connected = True
    scheduler = FakeScheduler()
    registry = _registry(kmnet, scheduler)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(executors=registry)))

    status = disconnect_kmnet(request)

    assert status["connected"] is False
    assert scheduler.clear_reasons == ["KMNET_MANUAL_DISCONNECT"]
    assert kmnet.disconnect_calls == 1


def test_backend_shutdown_disconnects_kmnet() -> None:
    kmnet = FakeKmNet()
    kmnet.connected = True

    _disconnect_kmnet(_registry(kmnet))

    assert kmnet.disconnect_calls == 1
    assert kmnet.connected is False
