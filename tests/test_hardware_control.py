"""Tests for hardware heartbeat, the active angle strategy, and command dispatch.

Each test focuses on one observable contract:
- Heartbeat suspends after a stale window.
- ExperimentalAnglePidStrategy consumes the compensated target contract
  and leaves trigger gating to RuntimeService.
- ExecutorRegistry must route production output through CommandScheduler
  before any device call.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from novasight.contracts import ControlIntent, Detection
from novasight.control import (
    AngularErrorMapper,
    AngularPDController,
    CommandScheduler,
    ControlOutput,
    ExperimentalAnglePidStrategy,
)
from novasight.executors import ExecutorRegistry
from novasight.executors.kmnet import KmNetExecutor
from novasight.executors.kmnet_loader import KmNetLoadResult
from novasight.hardware import BoxInputState, HardwareHeartbeat
from novasight.runtime import CONTROL_FRAME_FIELDS, run_replay_acceptance


def _target(
    x: float = 100, y: float = 100, w: float = 0, h: float = 0, score: float = 0.9
) -> Detection:
    return Detection(cls=0, score=score, x=x, y=y, w=w, h=h)


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
    )


def _compensated_target_raw(
    *,
    roi_width: int = 320,
    roi_height: int = 320,
    capture_width: int = 1920,
    capture_height: int = 1080,
    roi_offset_x: int = 800,
    roi_offset_y: int = 380,
    roi_x: float = 170.0,
    roi_y: float = 170.0,
) -> dict:
    return {
        "roi_width": roi_width,
        "roi_height": roi_height,
        "capture_width": capture_width,
        "capture_height": capture_height,
        "roi_offset_x": roi_offset_x,
        "roi_offset_y": roi_offset_y,
        "frame_id": 7,
        "target_key": "track:1",
        "compensated_target": {
            "track_id": 1,
            "source_frame_id": 7,
            "capture_ts_ns": 1_000_000,
            "state_ts_ns": 1_000_000,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "control_x": roi_offset_x + roi_x,
            "control_y": roi_offset_y + roi_y,
            "raw_x": roi_x,
            "raw_y": roi_y,
            "smoothed_x": roi_x,
            "smoothed_y": roi_y,
            "delta_x": 0.0,
            "delta_y": 0.0,
            "applied": False,
            "reason": "TEST",
            "measurement_age_ms": 0.0,
            "compensation_ms": 0.0,
            "prediction_confidence": 1.0,
            "predicted_source": False,
            "control_allowed": True,
        },
    }


def test_hardware_heartbeat_suspends_after_timeout() -> None:
    heartbeat = HardwareHeartbeat(timeout_s=0.05)

    assert heartbeat.should_suspend(now_s=1.0) is True
    heartbeat.mark_seen(now_s=1.0)

    assert heartbeat.should_suspend(now_s=1.02) is False
    assert heartbeat.should_suspend(now_s=1.10) is True


def test_legacy_pixel_strategy_modules_are_not_present() -> None:
    strategy_module = importlib.import_module("novasight.control.strategy")

    for symbol in (
        "PIDStrategy",
        "StraightStrategy",
        "PredictiveStrategy",
        "ProportionalStrategy",
        "ControlCommandCoalescer",
    ):
        assert not hasattr(strategy_module, symbol)

    for module_name in (
        "novasight.control.dynamic_pid",
        "novasight.control.isolated_mouse",
    ):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            assert exc.name == module_name
        else:
            raise AssertionError(f"legacy module still importable: {module_name}")


def test_current_hardware_public_api_exposes_only_kmnet_runtime_adapter() -> None:
    hardware_module = importlib.import_module("novasight.hardware")

    assert "KmboxNetAdapter" in hardware_module.__all__
    assert "create_hardware_box" in hardware_module.__all__
    assert "MAKCUAdapter" not in hardware_module.__all__
    assert not hasattr(hardware_module, "MAKCUAdapter")


def test_experimental_angle_strategy_uses_compensated_target_contract() -> None:
    strategy = ExperimentalAnglePidStrategy(
        kp_x=1.0,
        kp_y=1.0,
        kalman_enabled=False,
        max_step_counts=80,
    )

    command = strategy.calculate(
        _target(160, 160, 20, 20),
        (160, 160),
        BoxInputState(raw=_compensated_target_raw()),
    )

    assert command.dx > 0
    assert command.dy > 0
    assert command.confidence == 0.9
    assert command.debug["unit_pipeline"] == "compensated_control_px_to_angle_rad_to_counts"
    assert command.debug["capture_size_source"] == "frame_metadata"
    assert command.debug["comp_x"] == 970
    assert command.debug["comp_y"] == 550


def test_experimental_angle_strategy_composes_angular_control_contract() -> None:
    strategy = ExperimentalAnglePidStrategy(kalman_enabled=False)

    assert isinstance(strategy.error_mapper, AngularErrorMapper)
    assert isinstance(strategy.angular_controller, AngularPDController)


def test_experimental_angle_strategy_does_not_gate_hardware_trigger() -> None:
    strategy = ExperimentalAnglePidStrategy(
        kp_x=1.0,
        kp_y=1.0,
        kalman_enabled=False,
        max_step_counts=80,
    )

    command = strategy.calculate(
        _target(160, 160, 20, 20),
        (160, 160),
        BoxInputState(left=False, right=False, raw=_compensated_target_raw()),
    )

    assert command.dx > 0
    assert command.dy > 0
    assert command.debug["control_allowed"] is True
    assert "trigger" not in command.reason.lower()


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
    strategy_source = (root / "novasight" / "control" / "strategy.py").read_text(encoding="utf-8")

    assert "CompensatedTarget(Control px)->AngularErrorMapper->AngularPDController->CommandScheduler->kmNet" in service_source
    assert "ExperimentalAnglePidStrategy(" in service_source
    assert "self.executors.execute(intent)" in service_source
    assert "ControlOutput(" not in service_source
    assert "diagnostic_move" not in service_source

    assert "COMPENSATED_TARGET_REQUIRED" in strategy_source
    assert "AngularErrorMapper" in strategy_source
    assert "AngularPDController" in strategy_source

    submit_index = executor_source.index("decision = self.scheduler.submit(bounded)")
    send_index = executor_source.index("self.executors[self.selected].execute(bounded)")
    assert submit_index < send_index
    assert "command scheduler required" in executor_source


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


def test_angle_strategy_does_not_recreate_target_selection_or_bbox_aim_cache() -> None:
    root = Path(__file__).resolve().parents[1]
    strategy_source = (root / "novasight" / "control" / "strategy.py").read_text(encoding="utf-8")

    assert "COMPENSATED_TARGET_REQUIRED" in strategy_source
    for forbidden in (
        "def aim_point(",
        "_stable_aim_point",
        "_KalmanCenterTrack",
        "_KalmanAxis",
        "_last_target_box",
        "_cached_target",
        "bbox_cache",
        "last_detection",
    ):
        assert forbidden not in strategy_source


def test_replay_acceptance_and_recorder_schema_do_not_reintroduce_legacy_control_paths() -> None:
    report = run_replay_acceptance()

    assert report.passed is True
    required_fields = {
        "candidate_count",
        "track_id",
        "x",
        "raw_aim_x",
        "error_x_rad",
        "final_output_x_counts",
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
    executor = KmNetExecutor(flip_dy=True)
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
