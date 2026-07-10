from __future__ import annotations

from types import SimpleNamespace

from novasight.config import RuntimeConfig
from novasight.runtime.service import RuntimeService
from novasight.runtime.status import StatusHub


def test_status_hub_updates_interactive_state_at_twenty_hz() -> None:
    hub = StatusHub(SimpleNamespace(state=lambda: SimpleNamespace()))

    assert hub.interval_s == 0.05


def test_runtime_state_takes_control_snapshot_under_control_lock() -> None:
    service = RuntimeService(
        RuntimeConfig(),
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(status=lambda: {}),
    )

    class ProbeLock:
        entered = False

        def __enter__(self):
            self.entered = True

        def __exit__(self, *_args):
            return False

    probe = ProbeLock()
    service._control_lock = probe

    service.state()

    assert probe.entered is True
