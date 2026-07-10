from __future__ import annotations

from types import SimpleNamespace

from novasight.api.app import _auto_connect_kmnet, _disconnect_kmnet
from novasight.api.routes_executors import disconnect_kmnet
from novasight.config import RuntimeConfig


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


def _registry(kmnet: FakeKmNet, scheduler: FakeScheduler | None = None) -> SimpleNamespace:
    return SimpleNamespace(executors={"kmnet": kmnet}, scheduler=scheduler)


def test_backend_startup_auto_connects_kmnet_by_default() -> None:
    kmnet = FakeKmNet()

    _auto_connect_kmnet(_registry(kmnet), RuntimeConfig())

    assert kmnet.connect_calls == 1
    assert kmnet.connected is True


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
