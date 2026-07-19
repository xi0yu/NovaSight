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


def test_status_hub_updates_interactive_state_at_two_hz() -> None:
    hub = StatusHub(SimpleNamespace(state=lambda: SimpleNamespace()))

    assert hub.interval_s == 0.5


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


def test_status_hub_groups_page_scoped_snapshots_by_topic() -> None:
    class TopicRuntime:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def state(self) -> EmptyState:
            return EmptyState()

        def status_snapshot(self, topic: str) -> dict:
            self.calls.append(topic)
            return {"kind": "runtime_snapshot", "topic": topic, "state": {}}

    async def scenario() -> None:
        runtime = TopicRuntime()
        hub = StatusHub(runtime)
        summary_a = await hub.subscribe("summary")
        summary_b = await hub.subscribe("summary")
        capture = await hub.subscribe("capture")
        runtime.calls.clear()

        await hub.broadcast()

        assert runtime.calls.count("summary") == 1
        assert runtime.calls.count("capture") == 1
        hub.unsubscribe(summary_a)
        hub.unsubscribe(summary_b)
        hub.unsubscribe(capture)

    asyncio.run(scenario())


def test_runtime_summary_snapshot_avoids_full_pipeline_and_model_registry() -> None:
    class SummaryPipeline:
        stats = SimpleNamespace(
            processed_frames=7,
            control_observations=6,
        )

        def status(self) -> dict:
            raise AssertionError("full pipeline status must not run for summary snapshots")

        def summary_status(self) -> dict:
            return {
                "running": True,
                "selected": "deepstream_nvinfer",
                "deepstream": {
                    "available": True,
                    "running": True,
                    "capture_frames": 12,
                    "capture_fps": 10.0,
                    "input_frames": 11,
                    "input_fps": 9.0,
                    "output_buffers": 10,
                    "output_fps": 8.0,
                    "published_batches": 9,
                    "published_fps": 7.0,
                    "detection_batch_mailbox": {"overwritten_batches": 1},
                },
            }

    class Models:
        def get_active_deployment(self):
            raise AssertionError("model registry must not run for summary snapshots")

    service = RuntimeService(
        RuntimeConfig(),
        models=Models(),
        executors=SimpleNamespace(status=lambda: {"selected": "kmnet"}),
    )
    service.pipeline = SummaryPipeline()

    snapshot = service.status_snapshot("capture")

    assert snapshot["full"] is False
    assert "active_model" not in snapshot["state"]
    assert snapshot["state"]["statistics"]["capture_fps"] == 10.0
    assert snapshot["state"]["statistics"]["inference_fps"] == 8.0
    assert "configured" not in snapshot["state"]["inference"]


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


def test_runtime_state_exposes_power_saving_supervisor_status() -> None:
    service = RuntimeService(
        RuntimeConfig(),
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(status=lambda: {}),
    )
    service.power_supervisor = SimpleNamespace(
        status=lambda: {
            "enabled": True,
            "mode": "cold_standby",
            "run_intent": True,
            "reason": "target host offline; runtime suspended",
        }
    )

    state = service.state()

    assert state.power_saving["mode"] == "cold_standby"
    assert state.power_saving["run_intent"] is True
