from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from novasight.config import RuntimeConfig
from novasight.runtime.service import RuntimeService
from novasight.runtime.status import StatusHub


@dataclass
class EmptyState:
    pass


def test_status_hub_updates_interactive_state_at_five_hz() -> None:
    hub = StatusHub(SimpleNamespace(state=lambda: SimpleNamespace()))

    assert hub.interval_s == 0.2


def test_status_hub_shares_one_pump_across_subscribers() -> None:
    async def scenario() -> None:
        hub = StatusHub(SimpleNamespace(state=EmptyState))
        first = await hub.subscribe()
        pump = hub._pump_task
        second = await hub.subscribe()

        assert pump is not None
        assert hub._pump_task is pump

        hub.unsubscribe(first)
        assert hub._pump_task is pump
        hub.unsubscribe(second)
        assert hub._pump_task is None

    asyncio.run(scenario())


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
