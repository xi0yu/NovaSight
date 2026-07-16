from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from novasight.api import routes_runtime
from novasight.runtime.power import RuntimePowerSupervisor


class FakeLifecycle:
    running = False

    def start(self, reason: str) -> None:
        raise AssertionError(f"pipeline must not start without heartbeat: {reason}")

    def stop(self, reason: str) -> None:
        self.running = False


class RecordingLifecycle:
    def __init__(self) -> None:
        self.running = False
        self.events: list[tuple[str, str]] = []

    def start(self, reason: str) -> None:
        self.events.append(("start", reason))
        self.running = True

    def stop(self, reason: str) -> None:
        self.events.append(("stop", reason))
        self.running = False


def test_runtime_start_accepts_intent_in_host_offline_cold_standby(monkeypatch) -> None:
    app = FastAPI()
    app.include_router(routes_runtime.router)
    app.state.runtime = SimpleNamespace(pipeline=None, running=False, fatal_error=None)
    app.state.capture = SimpleNamespace()
    app.state.runtime_power = RuntimePowerSupervisor(
        lifecycle=FakeLifecycle(),
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
    )
    monkeypatch.setattr(
        routes_runtime,
        "create_runtime_pipeline",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("pipeline factory must not run in cold standby")
        ),
    )

    response = TestClient(app).post("/api/runtime/start")

    assert response.status_code == 200
    assert response.json() == {
        "running": False,
        "accepted": True,
        "failed": False,
        "standby": True,
        "power_saving": {
            "enabled": True,
            "mode": "cold_standby",
            "run_intent": True,
            "suspended_by_policy": True,
            "running": False,
            "host_id": "",
            "target_host_id": "gaming-pc",
            "host_online": False,
            "heartbeat_age_ms": None,
            "auto_resume": True,
            "reason": "waiting for target host heartbeat",
        },
    }


def test_host_heartbeat_endpoint_resumes_requested_runtime() -> None:
    lifecycle = RecordingLifecycle()
    app = FastAPI()
    app.include_router(routes_runtime.router)
    app.state.runtime = SimpleNamespace(pipeline=None, running=False, fatal_error=None)
    app.state.capture = SimpleNamespace()
    app.state.runtime_power = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
    )
    client = TestClient(app)
    client.post("/api/runtime/start")

    response = client.post(
        "/api/runtime/presence/heartbeat",
        json={"host_id": "gaming-pc"},
    )

    assert response.status_code == 200
    assert lifecycle.events == [("start", "HOST_ONLINE")]
    assert response.json()["power_saving"]["mode"] == "active"
    assert response.json()["power_saving"]["host_id"] == "gaming-pc"
