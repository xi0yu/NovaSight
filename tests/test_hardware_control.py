"""Tests for the mouse controller route and command dispatch.

Each test focuses on one observable contract:
- RuntimeService owns the only production mouse-control route.
- ExecutorRegistry must route production output through CommandScheduler
  before any device call.
"""
from __future__ import annotations

import copy
import importlib
from pathlib import Path

from novasight.config import RuntimeConfig
from novasight.contracts import ControlIntent
from novasight.control import (
    CommandScheduler,
    ControlOutput,
)
from novasight.executors import ExecutionResult, ExecutorRegistry
from novasight.executors.kmnet import KmNetExecutor
from novasight.executors.kmnet_loader import KmNetLoadResult
from novasight.runtime import CONTROL_FRAME_FIELDS, RuntimeService, run_replay_acceptance


def _intent(dx: float, dy: float, reason: str = "test") -> ControlIntent:
    return ControlIntent(
        dx=dx, dy=dy, action="move", confidence=1.0,
        reason=reason, source_id="control.test",
    )


def _output(
    dx: int,
    dy: int,
    *,
    frame_id: int = 1,
    track_id: int = 1,
    predicted: bool = False,
    trajectory_generation: int | None = None,
) -> ControlOutput:
    return ControlOutput(
        dx=dx,
        dy=dy,
        action="move",
        confidence=1.0,
        source_id="test",
        accepted=True,
        clipped=False,
        reason="test",
        source_frame_id=frame_id,
        source_track_id=track_id,
        predicted_source=predicted,
        trajectory_generation=trajectory_generation,
    )


def test_removed_control_routes_are_not_importable() -> None:
    for module_name in (
        "novasight.control.angular",
        "novasight.control.controller",
        "novasight.control.hid_output",
        "novasight.control.latency_compensator",
        "novasight.control.strategy",
        "novasight.control.dynamic_pid",
        "novasight.control.isolated_mouse",
        "novasight.runtime.aim",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            assert exc.name == module_name
        else:
            raise AssertionError(f"legacy module still importable: {module_name}")


def test_removed_parallel_hardware_adapter_is_not_importable() -> None:
    for module_name in (
        "novasight.hardware.kmbox_net",
        "novasight.hardware.factory",
        "novasight.hardware.heartbeat",
        "novasight.hardware.makcu",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            assert exc.name == module_name
        else:
            raise AssertionError(f"parallel hardware path still importable: {module_name}")


def test_executor_registry_requires_scheduler_before_device_send() -> None:
    class FakeExecutor:
        executor_id = "kmnet"

        def __init__(self) -> None:
            self.calls = 0

        def available(self) -> bool:
            return True

        def execute(self, output: ControlOutput):  # pragma: no cover - guarded by assertion below
            del output
            self.calls += 1
            raise AssertionError("executor must not be called without CommandScheduler")

    fake = FakeExecutor()
    registry = ExecutorRegistry([fake], default="kmnet", scheduler=None)

    result = registry.execute(_intent(4, 2))

    assert result.sent is False
    assert result.message == "command scheduler required"
    assert result.metadata["stage"] == "scheduler_required"
    assert fake.calls == 0


def test_scheduler_disabled_sends_each_observation_directly_without_splitting() -> None:
    class FakeExecutor:
        executor_id = "kmnet"

        def __init__(self) -> None:
            self.outputs: list[ControlOutput] = []

        def available(self) -> bool:
            return True

        def execute(self, output: ControlOutput) -> ExecutionResult:
            self.outputs.append(output)
            return ExecutionResult(self.executor_id, True, output, "sent")

    config = RuntimeConfig()
    config.control.scheduler_enabled = False
    fake = FakeExecutor()
    registry = ExecutorRegistry.from_config(config)
    registry.executors["kmnet"] = fake

    result = registry.execute(_intent(60, 16))

    assert result.sent is True
    assert result.metadata["stage"] == "direct_output"
    assert result.metadata["scheduler_enabled"] is False
    assert [(output.dx, output.dy) for output in fake.outputs] == [(60, 16)]
    assert registry.status()["scheduler"] == {"enabled": False, "direct_output": True}


def test_new_observation_only_replaces_plan_and_scheduler_tick_owns_send() -> None:
    class FakeExecutor:
        executor_id = "kmnet"

        def __init__(self) -> None:
            self.calls = 0

        def available(self) -> bool:
            return True

        def execute(self, output: ControlOutput) -> ExecutionResult:
            self.calls += 1
            return ExecutionResult(
                executor_id=self.executor_id,
                sent=True,
                intent=output,
                message="sent",
            )

    fake = FakeExecutor()
    registry = ExecutorRegistry(
        [fake],
        default="kmnet",
        scheduler=CommandScheduler(min_interval_s=0.0),
    )

    submitted = registry.execute(_intent(4, 2))

    assert submitted.sent is False
    assert submitted.message == "control command scheduled"
    assert fake.calls == 0
    assert registry.status()["scheduler"]["has_pending"] is True

    sent = registry.tick_pending()

    assert sent.sent is True
    assert fake.calls == 1


def test_zero_movement_never_reaches_kmnet_executor() -> None:
    class FakeExecutor:
        executor_id = "kmnet"

        def __init__(self) -> None:
            self.calls = 0

        def available(self) -> bool:
            return True

        def execute(self, output: ControlOutput) -> ExecutionResult:
            self.calls += 1
            return ExecutionResult(self.executor_id, True, output)

    fake = FakeExecutor()
    registry = ExecutorRegistry(
        [fake],
        default="kmnet",
        scheduler=CommandScheduler(min_interval_s=0.0),
    )

    result = registry.execute(_intent(0, 0))

    assert result.sent is False
    assert result.message == "zero control command ignored"
    assert result.metadata["action"] == "zero_output"
    assert registry.status()["scheduler"]["has_pending"] is False
    assert fake.calls == 0


def test_runtime_service_does_not_directly_call_kmnet_device_layer() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "novasight" / "runtime" / "service.py").read_text(encoding="utf-8")

    assert "self.executors.execute(intent)" in source
    for forbidden in (
        "KmNetExecutor",
        "load_kmnet_driver",
        "diagnostic_move",
        "move_auto",
        "move_beizer",
        "enc_move",
    ):
        assert forbidden not in source


def test_production_modules_do_not_call_kmnet_or_diagnostics_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    allowed = {
        "novasight/main.py",  # doctor kmnet CLI
        "novasight/api/routes_executors.py",  # explicit diagnostic API, not production runtime control
    }
    forbidden_tokens = (
        "KmNetExecutor",
        "load_kmnet_driver",
        "diagnostic_move",
        "move_auto",
        "move_beizer",
        "enc_move",
        "enc_move_auto",
        "enc_move_beizer",
    )

    violations: list[str] = []
    for path in (root / "novasight").rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in allowed or rel.startswith("novasight/executors/"):
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            if token in text:
                violations.append(f"{rel}: {token}")

    assert violations == []


def test_executor_registry_does_not_apply_legacy_y_down_compensation() -> None:
    root = Path(__file__).resolve().parents[1]
    executor_source = (root / "novasight" / "executors" / "runtime.py").read_text(
        encoding="utf-8"
    )

    for forbidden in (
        "YAxisWindowLimiter",
        "y_rate_limiter",
        "drop_counts",
        "Y axis periodic down compensation",
    ):
        assert forbidden not in executor_source


def test_runtime_service_production_control_contract_is_static() -> None:
    root = Path(__file__).resolve().parents[1]
    service_source = (root / "novasight" / "runtime" / "service.py").read_text(encoding="utf-8")
    executor_source = (root / "novasight" / "executors" / "runtime.py").read_text(encoding="utf-8")
    mouse_source = (root / "novasight" / "control" / "mouse.py").read_text(encoding="utf-8")
    observation_source = (root / "novasight" / "control" / "observation.py").read_text(encoding="utf-8")

    assert '"CommandScheduler"' in service_source
    assert '"DirectObservationSend"' in service_source
    assert "MouseController(" in service_source
    assert "MouseObservation(" in service_source
    assert "self.executors.execute(intent)" in service_source
    assert "ControlOutput(" not in service_source
    assert "diagnostic_move" not in service_source

    assert "predicted_error_x_rad" in mouse_source
    assert "observed_error_x_rad" in mouse_source
    assert "d_raw_x" in mouse_source
    assert "CalibratedAngularController" in mouse_source
    assert "UniversalSaturatedController" in mouse_source
    assert "ControllerFactory" in mouse_source
    assert "RawAimPointProjector" in observation_source

    submit_index = executor_source.index(
        "decision = self.scheduler.submit(bounded, emit_immediately=False)"
    )
    send_index = executor_source.index("self.executors[self.selected].execute(bounded)")
    assert submit_index < send_index
    assert "command scheduler required" in executor_source


def test_control_mode_switch_replaces_controller_and_clears_scheduler_state() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.clear_reasons: list[str] = []

        def clear(self, reason: str = "") -> None:
            self.clear_reasons.append(reason)

    class FakeExecutors:
        selected = "kmnet"

        def __init__(self) -> None:
            self.scheduler = FakeScheduler()

        def update_runtime_config(self, _config: RuntimeConfig) -> None:
            return None

    config = RuntimeConfig()
    executors = FakeExecutors()
    service = RuntimeService(config, models=object(), executors=executors)  # type: ignore[arg-type]
    previous_controller = service.mouse_controller
    previous_controller.state.residual_x_counts = 0.75
    updated = copy.deepcopy(config)
    updated.control.mode = "calibrated_angular"

    service.update_config(updated)

    assert previous_controller.mode == "universal_saturated"
    assert service.mouse_controller is not previous_controller
    assert service.mouse_controller.mode == "calibrated_angular"
    assert service.mouse_controller.state.residual_x_counts == 0.0
    assert "CONTROL_MODE_CHANGED" in executors.scheduler.clear_reasons


def test_runtime_service_does_not_bypass_selector_or_cache_bbox_for_control() -> None:
    root = Path(__file__).resolve().parents[1]
    service_source = (root / "novasight" / "runtime" / "service.py").read_text(encoding="utf-8")

    assert "selection = self._select_control_target(context)" in service_source
    assert "target = selection.target" in service_source
    assert "self.target_selector.select(" in service_source
    for forbidden in (
        "context.detections[0]",
        "context.detections[-1]",
        "list(context.detections)[0]",
        "next(iter(context.detections",
        "_last_target_box",
        "_cached_target",
        "box_cache",
        "bbox_cache",
        "last_detection",
        "last_detections",
    ):
        assert forbidden not in service_source


def test_mouse_controller_does_not_recreate_target_selection_or_bbox_cache() -> None:
    root = Path(__file__).resolve().parents[1]
    mouse_source = (root / "novasight" / "control" / "mouse.py").read_text(encoding="utf-8")

    for forbidden in (
        "TargetSelector",
        "KalmanEstimator",
        "_last_target_box",
        "_cached_target",
        "bbox_cache",
        "last_detection",
    ):
        assert forbidden not in mouse_source


def test_replay_acceptance_and_recorder_schema_do_not_reintroduce_legacy_control_paths() -> None:
    report = run_replay_acceptance()

    assert report.passed is True
    required_fields = {
        "candidate_count",
        "track_id",
        "control_width_px",
        "observed_aim_x_px",
        "observed_error_x_rad",
        "planned_x_counts",
    }
    assert required_fields.issubset(CONTROL_FRAME_FIELDS)
    for forbidden in (
        "pixel_linear",
        "pixel_to_count_scale",
        "mouse_ratio",
        "bbox_cache",
        "cached_target",
        "last_target_box",
        "local_trigger",
        "dry_run",
        "silent",
    ):
        assert forbidden not in CONTROL_FRAME_FIELDS


def test_command_scheduler_cancels_pending_on_new_frame() -> None:
    scheduler = CommandScheduler(min_interval_s=0.1, ttl_s=0.05)

    first = scheduler.submit(_output(1, 0, frame_id=1), now_s=1.0)
    held = scheduler.submit(_output(2, 0, frame_id=2), now_s=1.01)
    replaced = scheduler.submit(_output(3, 0, frame_id=3), now_s=1.02)

    assert first.output is not None
    assert first.metadata["command_status"] == "ready"
    assert held.output is None
    assert held.metadata["command_status"] == "pending"
    assert replaced.output is None
    assert replaced.metadata["cancel_reason"] == "NEW_FRAME"
    assert scheduler.status(now_s=1.02)["pending_source_frame_id"] == 3


def test_command_scheduler_replaces_pending_on_new_trajectory_generation() -> None:
    scheduler = CommandScheduler(min_interval_s=0.1, ttl_s=0.05)

    scheduler.submit(_output(1, 0, frame_id=1, trajectory_generation=1), now_s=1.0)
    scheduler.submit(_output(2, 0, frame_id=1, trajectory_generation=1), now_s=1.01)
    replaced = scheduler.submit(_output(3, 0, frame_id=1, trajectory_generation=2), now_s=1.02)
    status = scheduler.status(now_s=1.02)

    assert replaced.output is None
    assert replaced.metadata["cancel_reason"] == "TRAJECTORY_GENERATION"
    assert replaced.metadata["trajectory_generation"] == 2
    assert status["pending_trajectory_generation"] == 2
    assert status["pending_dx"] == 3


def test_command_scheduler_can_cancel_pending_before_generation() -> None:
    scheduler = CommandScheduler(min_interval_s=0.1, ttl_s=0.05)

    scheduler.submit(_output(1, 0, frame_id=1, trajectory_generation=4), now_s=1.0)
    scheduler.submit(_output(2, 0, frame_id=1, trajectory_generation=4), now_s=1.01)

    assert scheduler.cancel_pending_before_generation(4) is False
    assert scheduler.status(now_s=1.01)["has_pending"] is True
    assert scheduler.cancel_pending_before_generation(5) is True
    assert scheduler.status(now_s=1.01)["has_pending"] is False
    assert scheduler.status(now_s=1.01)["last_cancel_reason"] == "TRAJECTORY_GENERATION"


def test_command_scheduler_cancels_pending_on_direction_change() -> None:
    scheduler = CommandScheduler(min_interval_s=0.1, ttl_s=0.05)

    scheduler.submit(_output(1, 0, frame_id=1), now_s=1.0)
    scheduler.submit(_output(2, 0, frame_id=1), now_s=1.01)
    replaced = scheduler.submit(_output(-2, 0, frame_id=1), now_s=1.02)

    assert replaced.output is None
    assert replaced.metadata["cancel_reason"] == "DIRECTION_CHANGE"
    assert scheduler.status(now_s=1.02)["pending_dx"] == -2


def test_command_scheduler_splits_large_command_into_steps() -> None:
    scheduler = CommandScheduler(
        min_interval_s=0.0,
        ttl_s=0.05,
        max_step_x=20,
        max_step_y=20,
    )

    ready = scheduler.submit(_output(60, 0, frame_id=1), now_s=1.0)
    status = scheduler.status(now_s=1.0)

    assert ready.output is not None
    assert ready.output.dx == 20
    assert ready.output.dy == 0
    assert ready.metadata["split_steps_total"] == 3
    assert ready.metadata["pending_steps"] == 2
    assert status["pending_dx"] == 40
    assert status["pending_steps"] == 2


def test_command_scheduler_splits_axes_independently_without_subminimum_cross_axis_steps() -> None:
    scheduler = CommandScheduler(
        min_interval_s=0.0,
        ttl_s=0.05,
        max_step_x=32,
        max_step_y=32,
    )

    first = scheduler.submit(_output(64, 16, frame_id=1), now_s=1.0)
    second = scheduler.tick(now_s=1.01)

    assert first.output is not None
    assert (first.output.dx, first.output.dy) == (32, 16)
    assert second.output is not None
    assert (second.output.dx, second.output.dy) == (32, 0)


def test_runtime_scheduler_raises_legacy_step_limit_to_twice_device_minimum() -> None:
    config = RuntimeConfig()
    config.control.scheduler_step_counts_x = 20
    config.control.scheduler_step_counts_y = 20
    config.hardware.min_effective_move_counts_x = 16
    config.hardware.min_effective_move_counts_y = 16
    registry = ExecutorRegistry.from_config(config)
    assert registry.scheduler is not None

    ready = registry.scheduler.submit(_output(30, 0, frame_id=1), now_s=1.0)

    assert registry.scheduler.status(now_s=1.0)["max_step_x"] == 32
    assert ready.output is not None
    assert ready.output.dx == 30
    assert ready.metadata["split_steps_total"] == 1


def test_command_scheduler_holds_full_split_queue_when_throttled() -> None:
    scheduler = CommandScheduler(
        min_interval_s=0.1,
        ttl_s=0.05,
        max_step_x=20,
        max_step_y=20,
    )

    scheduler.submit(_output(1, 0, frame_id=1), now_s=1.0)
    held = scheduler.submit(_output(60, 0, frame_id=1), now_s=1.01)
    status = scheduler.status(now_s=1.01)

    assert held.output is None
    assert held.metadata["pending_steps"] == 3
    assert held.metadata["pending_dx"] == 60
    assert status["pending_steps"] == 3
    assert status["pending_dx"] == 60


def test_command_scheduler_expires_pending_and_uses_predicted_ttl() -> None:
    scheduler = CommandScheduler(min_interval_s=0.1, ttl_s=0.05, predicted_ttl_s=0.02)

    scheduler.submit(_output(1, 0, frame_id=1), now_s=1.0)
    scheduler.submit(_output(2, 0, frame_id=1, predicted=True), now_s=1.01)
    after_expiry = scheduler.submit(_output(3, 0, frame_id=1), now_s=1.04)

    assert after_expiry.metadata["expired_reason"] == "EXPIRED"
    assert scheduler.status(now_s=1.04)["expired_pending"] == 1


def test_command_scheduler_enters_cooldown_after_device_error() -> None:
    scheduler = CommandScheduler(min_interval_s=0.0, ttl_s=0.05, device_error_cooldown_s=0.1)

    ready = scheduler.submit(_output(1, 0), now_s=1.0)
    error = scheduler.record_execution_result(sent=False, message="driver failed", now_s=1.0)
    during_cooldown = scheduler.submit(_output(2, 0), now_s=1.05)

    assert ready.output is not None
    assert error["command_status"] == "error"
    assert during_cooldown.output is None
    assert during_cooldown.metadata["command_status"] == "cooldown"


def test_kmnet_executor_sends_calibrated_counts_without_device_y_flip(monkeypatch) -> None:
    class FakeKmNetDriver:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[int, ...]]] = []

        def move_beizer(self, *args: int) -> int:
            self.calls.append(("move_beizer", tuple(args)))
            return 0

    driver = FakeKmNetDriver()
    monkeypatch.setattr(
        "novasight.executors.kmnet.load_kmnet_driver",
        lambda: KmNetLoadResult(
            module=driver,
            available=True,
            source="test",
            reason="",
            platform="test",
            machine="test",
            python_tag="test",
        ),
    )
    executor = KmNetExecutor()
    executor.connected = True

    result = executor.execute(
        ControlOutput(
            dx=5,
            dy=7,
            action="move",
            confidence=1.0,
            source_id="test",
            accepted=True,
            clipped=False,
            reason="test",
            move_kind="bezier",
            move_ms=12,
            bezier_ctrl=(1, 2, 3, 4),
        )
    )

    assert result.sent is True
    assert result.metadata["driver_dx"] == 5
    assert result.metadata["driver_dy"] == 7
    assert "flipped_dy" not in result.metadata
    assert driver.calls == [("move_beizer", (5, 7, 12, 1, 2, 3, 4))]
