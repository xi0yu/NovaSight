from __future__ import annotations

import pytest

from novasight.runtime.power import RuntimePowerSupervisor


class FakeLifecycle:
    def __init__(self) -> None:
        self.running = False
        self.events: list[tuple[str, str]] = []

    def start(self, reason: str) -> None:
        self.events.append(("start", reason))
        self.running = True

    def stop(self, reason: str) -> None:
        self.events.append(("stop", reason))
        self.running = False


def test_enabled_host_presence_holds_start_intent_in_cold_standby() -> None:
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
        now_ns=lambda: 1_000_000_000,
    )

    status = supervisor.request_start()

    assert lifecycle.events == []
    assert status["run_intent"] is True
    assert status["mode"] == "cold_standby"
    assert status["reason"] == "waiting for target host heartbeat"


def test_host_heartbeat_resumes_requested_runtime() -> None:
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
        now_ns=lambda: 2_000_000_000,
    )
    supervisor.request_start()

    status = supervisor.heartbeat("gaming-pc")

    assert lifecycle.events == [("start", "HOST_ONLINE")]
    assert status["mode"] == "active"
    assert status["host_id"] == "gaming-pc"


def test_expired_heartbeat_stops_runtime_after_offline_grace() -> None:
    now_ns = [1_000_000_000]
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
        now_ns=lambda: now_ns[0],
    )
    supervisor.request_start()
    supervisor.heartbeat("gaming-pc")

    now_ns[0] = 7_100_000_000
    grace_status = supervisor.tick()
    assert lifecycle.running is True
    assert grace_status["mode"] == "grace"

    now_ns[0] = 22_200_000_000
    standby_status = supervisor.tick()

    assert lifecycle.events[-1] == ("stop", "HOST_OFFLINE")
    assert standby_status["mode"] == "cold_standby"
    assert standby_status["run_intent"] is True


def test_explicit_stop_prevents_future_heartbeat_auto_resume() -> None:
    now_ns = [1_000_000_000]
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
        now_ns=lambda: now_ns[0],
    )
    supervisor.request_start()
    supervisor.heartbeat("gaming-pc")

    stopped = supervisor.request_stop()
    now_ns[0] = 2_000_000_000
    resumed = supervisor.heartbeat("gaming-pc")

    assert lifecycle.events == [
        ("start", "HOST_ONLINE"),
        ("stop", "USER_STOP"),
    ]
    assert stopped["run_intent"] is False
    assert resumed["mode"] == "stopped"


def test_disabled_policy_preserves_manual_runtime_start() -> None:
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=False,
        target_host_id="",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
    )

    status = supervisor.request_start()

    assert lifecycle.events == [("start", "USER_START")]
    assert status["mode"] == "disabled"
    assert status["running"] is True


def test_stale_heartbeat_does_not_authorize_runtime_start() -> None:
    now_ns = [1_000_000_000]
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
        now_ns=lambda: now_ns[0],
    )
    supervisor.heartbeat("gaming-pc")
    now_ns[0] = 8_000_000_000

    status = supervisor.request_start()

    assert lifecycle.events == []
    assert status["mode"] == "cold_standby"
    assert status["host_online"] is False


def test_zero_offline_grace_stops_on_first_expired_tick() -> None:
    now_ns = [1_000_000_000]
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=0.0,
        auto_resume=True,
        now_ns=lambda: now_ns[0],
    )
    supervisor.request_start()
    supervisor.heartbeat("gaming-pc")
    now_ns[0] = 7_100_000_000

    status = supervisor.tick()

    assert lifecycle.events[-1] == ("stop", "HOST_OFFLINE")
    assert status["mode"] == "cold_standby"


def test_heartbeat_does_not_restart_runtime_after_non_policy_failure() -> None:
    now_ns = [1_000_000_000]
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
        now_ns=lambda: now_ns[0],
    )
    supervisor.request_start()
    supervisor.heartbeat("gaming-pc")
    lifecycle.running = False
    now_ns[0] = 2_000_000_000

    status = supervisor.heartbeat("gaming-pc")

    assert lifecycle.events == [("start", "HOST_ONLINE")]
    assert status["mode"] == "interrupted"
    assert status["suspended_by_policy"] is False


def test_explicit_stop_cleans_up_non_running_failed_pipeline() -> None:
    lifecycle = FakeLifecycle()
    supervisor = RuntimePowerSupervisor(
        lifecycle=lifecycle,
        enabled=False,
        target_host_id="",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
    )
    supervisor.request_start()
    lifecycle.running = False

    supervisor.request_stop()

    assert lifecycle.events == [
        ("start", "USER_START"),
        ("stop", "USER_STOP"),
    ]


def test_heartbeat_rejects_unconfigured_host_identity() -> None:
    supervisor = RuntimePowerSupervisor(
        lifecycle=FakeLifecycle(),
        enabled=True,
        target_host_id="gaming-pc",
        heartbeat_timeout_s=6.0,
        offline_grace_s=15.0,
        auto_resume=True,
    )

    with pytest.raises(ValueError, match="does not match configured target_host_id"):
        supervisor.heartbeat("other-host")
