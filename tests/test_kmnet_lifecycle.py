from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from novasight.api.app import (
    _auto_restore_capture,
    _auto_connect_kmnet,
    _disconnect_kmnet,
    _start_auto_connect_kmnet,
    _start_auto_restore_capture,
)
from novasight.api.routes_executors import disconnect_kmnet
from novasight.config import RuntimeConfig
from novasight.executors.kmnet import KmNetExecutor
from novasight.executors.kmnet_diagnostics import KmNetRouteStatus
from novasight.executors.kmnet_loader import KmNetLoadResult


@pytest.fixture(autouse=True)
def _available_kmnet_route(monkeypatch) -> None:
    monkeypatch.setattr(
        "novasight.executors.kmnet.probe_kmnet_route",
        lambda host, _port: KmNetRouteStatus(
            available=True,
            resolved_ip=str(host),
            local_ip="192.0.2.2",
        ),
    )


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

    def configure_profile_only(self, *_: object, **__: object) -> SimpleNamespace:
        self.configure_started.set()
        self.release_configure.wait(timeout=2.0)
        return SimpleNamespace(available=False)


class AvailableCapture:
    def __init__(self) -> None:
        self.configure_calls = 0

    def configure_profile_only(self, *_: object, **__: object) -> SimpleNamespace:
        self.configure_calls += 1
        return SimpleNamespace(
            available=True,
            profile=SimpleNamespace(pixel_format="MJPG", width=1920, height=1080, fps=120),
        )


class BlockingDriver:
    def __init__(self) -> None:
        self.init_started = threading.Event()
        self.release_init = threading.Event()

    def init(self, *_: object) -> int:
        self.init_started.set()
        self.release_init.wait(timeout=2.0)
        return 0


class MonitorFailureDriver:
    def init(self, *_: object) -> int:
        return 0

    def monitor(self, _port: int) -> int:
        return 7


class ButtonDriver:
    def __init__(self) -> None:
        self.left_reads = 0
        self.right_reads = 0

    def init(self, *_: object) -> int:
        return 0

    def monitor(self, _port: int) -> int:
        return 0

    def isdown_left(self) -> int:
        self.left_reads += 1
        return 1

    def isdown_right(self) -> int:
        self.right_reads += 1
        return 0


class BlockingButtonDriver(ButtonDriver):
    def __init__(self) -> None:
        super().__init__()
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def isdown_left(self) -> int:
        self.read_started.set()
        self.release_read.wait(timeout=2.0)
        return super().isdown_left()


class TimeoutDriverProcess:
    def call(self, name: str, *_args: object, timeout_s: float) -> object:
        raise TimeoutError(f"kmNet driver call timed out: {name}")

    def abort(self) -> None:
        return None


def _registry(kmnet: FakeKmNet, scheduler: FakeScheduler | None = None) -> SimpleNamespace:
    return SimpleNamespace(executors={"kmnet": kmnet}, scheduler=scheduler)


def test_backend_startup_auto_connects_kmnet_by_default() -> None:
    kmnet = FakeKmNet()

    _auto_connect_kmnet(_registry(kmnet), RuntimeConfig())

    assert kmnet.connect_calls == 1
    assert kmnet.connected is True


def test_target_driven_control_does_not_start_unused_kmnet_button_monitor() -> None:
    config = RuntimeConfig()

    target_driven = KmNetExecutor.from_config(config)
    config.control.trigger_mode = "hardware"
    hardware_triggered = KmNetExecutor.from_config(config)

    assert target_driven.monitor_port == 0
    assert hardware_triggered.monitor_port == config.hardware.monitor_port


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


def test_backend_startup_restores_capture_profile_without_starting_mainline() -> None:
    config = RuntimeConfig()
    config.source.default = "capture"
    capture = AvailableCapture()

    _auto_restore_capture(capture, config)  # type: ignore[arg-type]

    assert capture.configure_calls == 1


def test_monitor_failure_does_not_hide_successful_device_connection() -> None:
    executor = KmNetExecutor(host="192.0.2.1", port=8888, uuid="test", monitor_port=5001)
    executor._driver = MonitorFailureDriver()
    executor._driver_process = None

    status = executor.connect()

    assert status["connected"] is True
    assert status["monitoring"] is False
    assert status["connection_stage"] == "connected_without_monitor"
    assert status["last_connect_error_stage"] == "monitor"
    assert status["last_connect_error_type"] == "driver_return_code"
    assert status["last_driver_call"] == "monitor"
    assert status["last_driver_rc"] == 7
    assert "monitor failed" in str(status["last_error"])


def test_button_value_one_means_pressed_instead_of_driver_failure() -> None:
    executor = KmNetExecutor(
        host="192.0.2.1",
        port=8888,
        uuid="test",
        monitor_port=5001,
        button_poll_interval_s=0.050,
    )
    driver = ButtonDriver()
    executor._driver = driver
    executor._driver_process = None

    executor.connect()
    deadline = time.monotonic() + 0.5
    while not executor.read_buttons()["available"] and time.monotonic() < deadline:
        time.sleep(0.005)

    try:
        buttons = executor.read_buttons()
        assert buttons["available"] is True
        assert buttons["left"] is True
        assert buttons["right"] is False
        assert buttons["reason"] == ""
    finally:
        executor.disconnect()


def test_button_poll_logging_only_reports_trigger_activation(caplog) -> None:
    executor = KmNetExecutor()

    with caplog.at_level("INFO", logger="novasight.executors.kmnet"):
        executor._record_buttons(True, False, False, "", raw={"sample": "idle"})
        executor._record_buttons(True, False, False, "", raw={"sample": "idle"})
        executor._record_buttons(True, True, False, "", raw={"sample": "left"})
        executor._record_buttons(True, True, False, "", raw={"sample": "left"})
        executor._record_buttons(True, False, False, "", raw={"sample": "idle"})
        executor._record_buttons(True, True, False, "", raw={"sample": "left"})

    trigger_logs = [
        record
        for record in caplog.records
        if record.name == "novasight.executors.kmnet"
        and record.getMessage().startswith("kmNet trigger active")
    ]
    assert len(trigger_logs) == 2
    assert all("left=True right=False" in record.getMessage() for record in trigger_logs)


def test_read_buttons_returns_cached_state_without_waiting_for_driver() -> None:
    executor = KmNetExecutor(
        host="192.0.2.1",
        port=8888,
        uuid="test",
        monitor_port=5001,
    )
    driver = BlockingButtonDriver()
    executor._driver = driver
    executor._driver_process = None

    executor.connect()
    assert driver.read_started.wait(timeout=0.2)

    started_at = time.monotonic()
    buttons = executor.read_buttons()
    elapsed_s = time.monotonic() - started_at

    driver.release_read.set()
    executor.disconnect()

    assert elapsed_s < 0.020
    assert buttons["available"] is False
    assert "awaiting first" in str(buttons["reason"])


def test_init_timeout_reports_exact_connection_failure_stage() -> None:
    executor = KmNetExecutor(host="192.0.2.1", port=8888, uuid="test")
    executor._driver = MonitorFailureDriver()
    executor._driver_process = TimeoutDriverProcess()

    status = executor.connect()

    assert status["connected"] is False
    assert status["connection_stage"] == "failed"
    assert status["last_connect_error_stage"] == "init"
    assert status["last_connect_error_type"] == "timeout"
    assert status["last_driver_call"] == "init"
    assert "timed out" in str(status["last_driver_error"])


def test_unavailable_driver_reason_survives_connect_attempt(monkeypatch) -> None:
    monkeypatch.setattr(
        "novasight.executors.kmnet.load_kmnet_driver",
        lambda: KmNetLoadResult(
            module=None,
            available=False,
            source="test",
            reason="Python ABI mismatch",
            platform="linux",
            machine="aarch64",
            python_tag="cpython-311",
        ),
    )
    executor = KmNetExecutor()

    status = executor.connect()

    assert status["connection_stage"] == "driver_unavailable"
    assert status["last_connect_error_stage"] == "driver_load"
    assert status["last_connect_error_type"] == "unavailable"
    assert status["last_error"] == "Python ABI mismatch"


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
