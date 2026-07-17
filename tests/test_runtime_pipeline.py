"""Tests for core runtime primitives and pipeline behavior."""

import copy
import math
import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest

from novasight.capture.pipeline import build_appsink_candidates
from novasight.capture.session import CaptureSession
from novasight.capture.source import CapturedFrame, GstAppSinkFrameSource
from novasight.capture.state import CaptureProfile, CaptureRuntimeState
from novasight.config import RuntimeConfig
from novasight.contracts import ControlIntent, Detection, DetectionBatch, FrameContext, Track
from novasight.executors import BoxInputState
from novasight.executors import ExecutionResult, ExecutorRegistry
from novasight.inference.contracts import InferenceResult
from novasight.runtime import (
    FailFastHandler,
    FrameHandle,
    FreshnessGate,
    LatestFrameExchange,
    LatestFrameBroker,
    RuntimeConfigStore,
    RuntimePipeline,
    RuntimeService,
    configure_logging,
)
from novasight.runtime.state import RuntimeFrameResult


def _frame(frame_id: int) -> CapturedFrame:
    return CapturedFrame(
        frame_id=frame_id,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=1_000_000_000 + frame_id * 1_000,
        capture_wait_ms=1.0,
        image=None,
    )


def _handle(generation: int) -> FrameHandle:
    return FrameHandle(
        generation=generation,
        frame_id=generation,
        source_sequence=generation,
        capture_ts_ns=time.monotonic_ns(),
        clock_domain="monotonic",
        pipeline_running_time_ns=None,
        width=640,
        height=640,
        format="NV12",
        resource=object(),
    )


def test_box_input_freshness_expires_stalled_button_sample() -> None:
    config = RuntimeConfig()
    config.control.scheduler_interval_ms = 4.0
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )
    now_ns = 1_000_000_000

    fresh = BoxInputState(
        left=True,
        raw={"sample_ts_ns": now_ns - 49_000_000},
    )
    stale = BoxInputState(
        left=True,
        raw={"sample_ts_ns": now_ns - 51_000_000},
    )

    assert service._box_input_is_fresh(fresh, now_ns) is True
    assert service._box_input_is_fresh(stale, now_ns) is False


def test_class_aim_y_hot_update_changes_the_next_control_observation() -> None:
    config = RuntimeConfig()
    config.inference.detection_class_profile = "default"
    config.inference.detection_class_profiles = {"default": ["body", "head"]}
    config.control.aim.class_roles = {"default": {"0": "body", "1": "head"}}
    config.control.aim.role_y_ratios.head = 0.15
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )
    service.last_inference_status.update(
        {
            "source_geometry_trusted": True,
            "source_width": 640,
            "source_height": 640,
            "roi_offset_x": 0,
            "roi_offset_y": 0,
        }
    )
    target = Track(
        track_id=7,
        cls=1,
        score=0.9,
        x=200.0,
        y=100.0,
        w=80.0,
        h=100.0,
    )
    context = FrameContext(
        frame_id=1,
        width=640,
        height=640,
        capture_ts_ns=1_000_000_000,
    )

    first = service._mouse_observation_metadata(
        context=context,
        target=target,
        control_metadata={
            "control_width": 640,
            "control_height": 640,
            "capture_geometry_trusted": True,
        },
        selector_debug={},
        control_now_ts_ns=1_001_000_000,
        measurement_dt_s=None,
        left_trigger_active=False,
        left_trigger_hold_ms=0.0,
    )

    updated = copy.deepcopy(config)
    updated.control.aim.role_y_ratios.head = 0.75
    service.update_targeting_config(updated, reset_control_history=True)
    second = service._mouse_observation_metadata(
        context=context,
        target=target,
        control_metadata={
            "control_width": 640,
            "control_height": 640,
            "capture_geometry_trusted": True,
        },
        selector_debug={},
        control_now_ts_ns=1_002_000_000,
        measurement_dt_s=None,
        left_trigger_active=False,
        left_trigger_hold_ms=0.0,
    )

    assert first["mouse_observation_debug"]["raw_aim"]["y_ratio"] == pytest.approx(0.15)
    assert first["mouse_observation"].observed_y_px == pytest.approx(115.0)
    assert second["mouse_observation_debug"]["raw_aim"]["y_ratio"] == pytest.approx(0.75)
    assert second["mouse_observation"].observed_y_px == pytest.approx(175.0)


def test_failed_device_send_clears_fixed_recoil_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "calibrated_angular"
    executors = ExecutorRegistry.from_config(config)
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    controller = service._active_mouse_controller()
    controller.recoil._residual_counts_y = 0.5
    controller.state.residual_y_counts = 0.25
    controller.state.output_history_valid = True
    intent = ControlIntent(
        dx=0.0,
        dy=1.0,
        action=None,
        confidence=1.0,
        reason="test",
        source_id="test",
    )
    monkeypatch.setattr(service, "_control_intent_from_context", lambda _context: intent)
    monkeypatch.setattr(
        executors,
        "execute",
        lambda output: ExecutionResult(
            executor_id="kmnet",
            sent=False,
            intent=output,
            message="device send failed",
            metadata={"stage": "device"},
        ),
    )

    service.process_frame(FrameContext(frame_id=1, width=640, height=640))

    assert controller.recoil._residual_counts_y == 0.0
    assert controller.state.residual_y_counts == 0.0
    assert controller.state.output_history_valid is False


def test_latest_replace_tick_drops_recoil_when_button_sample_becomes_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    config.control.shared.recoil_enabled = True
    executors = ExecutorRegistry.from_config(config)
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    controller = service._active_robust_predictive_controller()
    controller._recoil._residual_counts_y = 0.5
    service.last_control = {"pipeline": {"recoil_active": True}}
    stale_sample = time.monotonic_ns() - 1_000_000_000
    monkeypatch.setattr(
        service,
        "_box_input_state",
        lambda: BoxInputState(
            left=True,
            raw={"sample_ts_ns": stale_sample},
        ),
    )
    tick_called = False

    def tick_pending() -> ExecutionResult:
        nonlocal tick_called
        tick_called = True
        raise AssertionError("stale recoil command must be cleared before device send")

    monkeypatch.setattr(executors, "tick_pending", tick_pending)

    result = service.process_control_tick()

    assert result.execution_results == []
    assert tick_called is False
    assert controller._recoil._residual_counts_y == 0.0


def test_latest_replace_device_failure_resets_v2_output_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    config.control.shared.recoil_enabled = True
    executors = ExecutorRegistry.from_config(config)
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    controller = service._active_robust_predictive_controller()
    controller._recoil._residual_counts_y = 0.5
    controller._quantizer_y.accumulator = 0.25
    service.last_control = {"pipeline": {"recoil_active": True}}
    now_ns = time.monotonic_ns()
    monkeypatch.setattr(
        service,
        "_box_input_state",
        lambda: BoxInputState(
            left=True,
            raw={"sample_ts_ns": now_ns},
        ),
    )
    intent = ControlIntent(
        dx=0.0,
        dy=1.0,
        action=None,
        confidence=1.0,
        reason="test",
        source_id="test",
    )
    monkeypatch.setattr(
        executors,
        "tick_pending",
        lambda: ExecutionResult(
            executor_id="kmnet",
            sent=False,
            intent=intent,
            message="device send failed",
            metadata={"stage": "device"},
        ),
    )

    result = service.process_control_tick()

    assert result.execution_results[0].sent is False
    assert controller._recoil._residual_counts_y == 0.0
    assert controller._quantizer_y.accumulator == 0.0


def test_latest_replace_superseded_tick_preserves_newer_recoil_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    config.control.shared.recoil_enabled = True
    executors = ExecutorRegistry.from_config(config)
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    controller = service._active_robust_predictive_controller()
    controller._recoil._residual_counts_y = 0.5
    service.last_control = {"pipeline": {"recoil_active": True}}
    now_ns = time.monotonic_ns()
    monkeypatch.setattr(
        service,
        "_box_input_state",
        lambda: BoxInputState(
            left=True,
            raw={"sample_ts_ns": now_ns},
        ),
    )
    intent = ControlIntent(
        dx=0.0,
        dy=1.0,
        action=None,
        confidence=1.0,
        reason="test",
        source_id="test",
    )
    monkeypatch.setattr(
        executors,
        "tick_pending",
        lambda: ExecutionResult(
            executor_id="kmnet",
            sent=False,
            intent=intent,
            message="scheduled command discarded after executor reconfiguration",
            metadata={"stage": "scheduler", "action": "scheduler_superseded"},
        ),
    )

    result = service.process_control_tick()

    assert result.execution_results[0].sent is False
    assert controller._recoil._residual_counts_y == pytest.approx(0.5)


class _TriggeredKmNet:
    executor_id = "kmnet"

    def __init__(self) -> None:
        self.outputs = []

    def available(self) -> bool:
        return True

    def read_buttons(self) -> dict[str, object]:
        return {
            "available": True,
            "left": True,
            "right": False,
            "reason": "",
            "raw": {"source": "integration-test"},
            "sample_ts_ns": time.monotonic_ns(),
            "poll_ts_ns": time.monotonic_ns(),
        }

    def status(self, **_kwargs) -> dict[str, object]:
        return {"available": True, "connected": True, "button_left": True}

    def execute(self, output) -> ExecutionResult:
        self.outputs.append(output)
        return ExecutionResult(
            executor_id="kmnet",
            sent=True,
            intent=output,
            message="sent",
            metadata={"driver_dx": int(output.dx), "driver_dy": int(output.dy)},
        )


class _UnavailableButtonKmNet(_TriggeredKmNet):
    def read_buttons(self) -> dict[str, object]:
        return {
            "available": False,
            "left": False,
            "right": False,
            "reason": "isdown_left failed rc=-1",
            "raw": {},
            "sample_ts_ns": 0,
            "poll_ts_ns": time.monotonic_ns(),
        }


class _HeldLeftKmNet(_TriggeredKmNet):
    def read_buttons(self) -> dict[str, object]:
        now_ns = time.monotonic_ns()
        return {
            "available": True,
            "left": True,
            "right": False,
            "reason": "",
            "left_pressed_since_ts_ns": now_ns - 100_000_000,
            "right_pressed_since_ts_ns": 0,
            "sample_ts_ns": now_ns,
            "poll_ts_ns": now_ns,
        }


def test_latest_frame_exchange_name_is_capacity_one_latest_mailbox() -> None:
    exchange = LatestFrameExchange()

    for generation in range(1, 5):
        exchange.publish(_handle(generation))

    latest = exchange.acquire_latest(after_generation=-1, timeout_s=0.0)

    assert latest is not None
    assert latest.generation == 4
    assert exchange.status()["max_pending_depth"] == 1
    assert exchange.status()["overwritten_frames"] == 3


def test_freshness_gate_uses_strictest_positive_threshold() -> None:
    now_ns = 1_000_000_000_000
    gate = FreshnessGate.strictest(55, 10, 0, None)

    reason = gate.stale_reason(
        capture_ts_ns=now_ns - 12_000_000,
        now_ns=now_ns,
        template="stale {age_ms:.1f}>{threshold_ms:.1f}",
    )

    assert reason == "stale 12.0>10.0"


def test_gst_cpu_latest_pipeline_uses_single_frame_leaky_appsink() -> None:
    profile = CaptureProfile(
        device="/dev/video0",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=120,
        preference="manual",
        selection_reason="test",
    )

    candidates = build_appsink_candidates(profile, roi_size=640)
    pipeline = candidates[0].pipeline

    assert candidates[0].label.startswith("gst_cpu_latest:")
    assert "v4l2src device=/dev/video0 io-mode=2 do-timestamp=true" in pipeline
    assert "image/jpeg,width=1920,height=1080,framerate=120/1" in pipeline
    assert pipeline.count("queue max-size-buffers=1") >= 2
    assert "leaky=downstream" in pipeline
    assert "jpegparse" in pipeline
    assert "nvv4l2decoder mjpeg=1" in pipeline
    assert "nvvidconv left=640 right=1280 top=220 bottom=860" in pipeline
    assert "video/x-raw,format=BGRx,width=640,height=640" in pipeline
    assert "appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false" in pipeline
    assert any(
        "video/x-raw,format=RGBA,width=640,height=640" in item.pipeline for item in candidates
    )
    assert any(
        "video/x-raw,format=RGB,width=640,height=640" in item.pipeline for item in candidates
    )
    assert any(
        "video/x-raw,format=I420,width=640,height=640" in item.pipeline for item in candidates
    )


def test_cpu_compatible_capture_backend_is_public_appsink_bridge() -> None:
    from novasight.capture import CpuCompatibleCaptureBackend

    assert CpuCompatibleCaptureBackend is GstAppSinkFrameSource


def test_latest_frame_broker_releases_overwritten_and_cleared_pending_handles() -> None:
    released: list[str] = []
    broker = LatestFrameBroker()
    first = FrameHandle(
        generation=1,
        frame_id=1,
        source_sequence=1,
        capture_ts_ns=time.monotonic_ns(),
        clock_domain="monotonic",
        pipeline_running_time_ns=None,
        width=640,
        height=640,
        format="NV12",
        resource="first",
        release_callback=lambda resource: released.append(resource),
    )
    second = FrameHandle(
        generation=2,
        frame_id=2,
        source_sequence=2,
        capture_ts_ns=time.monotonic_ns(),
        clock_domain="monotonic",
        pipeline_running_time_ns=None,
        width=640,
        height=640,
        format="NV12",
        resource="second",
        release_callback=lambda resource: released.append(resource),
    )

    broker.publish(first)
    broker.publish(second)
    broker.clear()

    assert released == ["first", "second"]


def test_latest_frame_broker_overwrites_pending_frames() -> None:
    broker = LatestFrameBroker()

    broker.publish(_handle(100))
    first = broker.acquire_latest(after_generation=-1, timeout_s=0.0)
    for generation in range(101, 105):
        broker.publish(_handle(generation))
    second = broker.acquire_latest(after_generation=100, timeout_s=0.0)
    for generation in range(105, 109):
        broker.publish(_handle(generation))
    third = broker.acquire_latest(after_generation=104, timeout_s=0.0)

    assert [first.generation, second.generation, third.generation] == [100, 104, 108]
    status = broker.status()
    assert status["pending_depth"] == 0
    assert status["max_pending_depth"] == 1
    assert status["overwritten_frames"] == 6
    assert status["latest_overwrite_count"] == 6
    assert status["busy_drop_count"] == 6


def test_latest_frame_broker_exposes_proven_latest_frame_metadata() -> None:
    broker = LatestFrameBroker()
    capture_ts_ns = time.monotonic_ns()
    broker.publish(
        FrameHandle(
            generation=7,
            frame_id=11,
            source_sequence=11,
            capture_ts_ns=capture_ts_ns,
            clock_domain="monotonic",
            pipeline_running_time_ns=None,
            width=480,
            height=480,
            format="BGRx",
            resource=object(),
            metadata={
                "resource_memory": "system",
                "capture_ts_source": "userspace_monotonic_receive",
                "content_validation_status": "not_integrated",
                "content_validation_reason": "frame content has not been measured",
            },
        )
    )

    status = broker.status()

    assert status["published_generation"] == 7
    assert status["published_frame_id"] == 11
    assert status["published_capture_ts_ns"] == capture_ts_ns
    assert status["published_frame_age_ms"] >= 0.0
    assert status["published_width"] == 480
    assert status["published_height"] == 480
    assert status["published_format"] == "BGRx"
    assert status["published_resource_memory"] == "system"
    assert status["published_capture_ts_source"] == "userspace_monotonic_receive"
    assert status["published_content_validation_status"] == "not_integrated"
    assert status["published_content_validation_reason"] == "frame content has not been measured"


def test_latest_frame_broker_rejects_stale_generation_publish() -> None:
    released: list[str] = []
    broker = LatestFrameBroker()
    broker.publish(
        FrameHandle(
            generation=2,
            frame_id=2,
            source_sequence=2,
            capture_ts_ns=time.monotonic_ns(),
            clock_domain="monotonic",
            pipeline_running_time_ns=None,
            width=640,
            height=640,
            format="BGR",
            resource="newest",
            release_callback=lambda resource: released.append(resource),
        )
    )
    newest = broker.acquire_latest(after_generation=-1, timeout_s=0.0)
    assert newest is not None
    newest.release()

    broker.publish(
        FrameHandle(
            generation=1,
            frame_id=1,
            source_sequence=1,
            capture_ts_ns=time.monotonic_ns(),
            clock_domain="monotonic",
            pipeline_running_time_ns=None,
            width=640,
            height=640,
            format="BGR",
            resource="stale",
            release_callback=lambda resource: released.append(resource),
        )
    )

    stale = broker.acquire_latest(after_generation=-1, timeout_s=0.0)

    assert stale is None
    assert released == ["newest", "stale"]
    assert broker.status()["stale_published_frames"] == 1
    assert broker.status()["published_generation"] == 2


def test_latest_frame_broker_rejects_stale_frame_id_publish() -> None:
    released: list[str] = []
    broker = LatestFrameBroker()
    broker.publish(
        FrameHandle(
            generation=2,
            frame_id=20,
            source_sequence=20,
            capture_ts_ns=time.monotonic_ns(),
            clock_domain="monotonic",
            pipeline_running_time_ns=None,
            width=640,
            height=640,
            format="BGR",
            resource="newest",
            release_callback=lambda resource: released.append(resource),
        )
    )
    newest = broker.acquire_latest(after_generation=-1, timeout_s=0.0)
    assert newest is not None
    newest.release()

    broker.publish(
        FrameHandle(
            generation=3,
            frame_id=19,
            source_sequence=19,
            capture_ts_ns=time.monotonic_ns(),
            clock_domain="monotonic",
            pipeline_running_time_ns=None,
            width=640,
            height=640,
            format="BGR",
            resource="stale-frame-id",
            release_callback=lambda resource: released.append(resource),
        )
    )

    stale = broker.acquire_latest(after_generation=-1, timeout_s=0.0)
    status = broker.status()

    assert stale is None
    assert released == ["newest", "stale-frame-id"]
    assert status["stale_published_frames"] == 1
    assert status["published_generation"] == 2
    assert status["published_frame_id"] == 20


def test_latest_frame_broker_pending_depth_never_exceeds_one() -> None:
    broker = LatestFrameBroker()

    for generation in range(100):
        broker.publish(_handle(generation))
        assert broker.status()["pending_depth"] == 1

    latest = broker.acquire_latest(after_generation=-1, timeout_s=0.0)

    assert latest.generation == 99
    assert broker.status()["pending_depth"] == 0


def test_capture_session_publishes_captured_frames_to_latest_frame_broker() -> None:
    class OneFrameSource:
        backend_label = "test:one-frame"

        def __init__(self) -> None:
            self.closed = False
            self.reads = 0

        def read(self) -> CapturedFrame | None:
            if self.closed:
                return None
            self.reads += 1
            if self.reads > 1:
                time.sleep(0.01)
                return None
            return _frame(7)

        def close(self) -> None:
            self.closed = True

    source = OneFrameSource()
    profile = CaptureProfile(
        device="/dev/test",
        pixel_format="BGR",
        width=2,
        height=2,
        fps=120,
        preference="manual",
        selection_reason="broker test",
    )
    capture = CaptureSession(source_factory=lambda _profile: source)

    try:
        capture.start(profile)
        handle = capture.latest_frame_broker.acquire_latest(
            after_generation=-1,
            timeout_s=1.0,
        )
    finally:
        capture.stop("test complete")

    assert handle is not None
    assert handle.generation == 1
    assert handle.frame_id == 7
    assert handle.source_sequence == 7
    assert handle.metadata["captured_frame"].frame_id == 7
    assert handle.clock_domain == "monotonic"


def test_runtime_config_store_returns_isolated_snapshots() -> None:
    cfg = RuntimeConfig()
    store = RuntimeConfigStore(cfg)

    snap = store.snapshot()
    snap.capture.device = "/dev/changed"

    assert store.snapshot().capture.device == "/dev/video0"
    assert store.status()["version"] == 0
    assert store.status()["roi_size"] == 640


def test_failfast_writes_crash_log_before_exit(tmp_path) -> None:
    exits: list[int] = []
    fatal: list[tuple[str, str]] = []

    def exit_fn(code: int):
        exits.append(code)
        raise SystemExit(code)

    handler = FailFastHandler(
        log_dir=tmp_path,
        exit_fn=exit_fn,
        on_fatal=lambda name, exc, path: fatal.append((name, str(path))),
    )

    with pytest.raises(SystemExit):
        handler.run("capture", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    assert exits == [1]
    assert fatal == [("capture", str(tmp_path / "crash_capture.log"))]
    assert "boom" in (tmp_path / "crash_capture.log").read_text(encoding="utf-8")


def test_configure_logging_writes_to_configured_directory(tmp_path) -> None:
    cfg = RuntimeConfig()
    cfg.logging.dir = str(tmp_path)

    path = configure_logging(cfg)

    assert path == tmp_path / "novasight.log"
    assert path.parent.exists()


def test_runtime_pipeline_requires_running_capture_session_and_does_not_configure() -> None:
    capture = SimpleNamespace(
        source=None,
        state=SimpleNamespace(available=False),
        session=SimpleNamespace(running=False),
        config=SimpleNamespace(device="/dev/video0"),
        configure=lambda *_args, **_kwargs: pytest.fail("runtime must not configure capture"),
    )
    runtime = SimpleNamespace(
        running=False,
        process_captured_frame=lambda frame, *, acquired_generation=None: None,
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    with pytest.raises(RuntimeError, match="采集未启动，无法运行推理链路。"):
        pipeline.start()

    assert runtime.running is False
    assert pipeline.running is False


def test_runtime_pipeline_rejects_unloaded_model_before_threads_start() -> None:
    cfg = RuntimeConfig()
    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
    )
    runtime = SimpleNamespace(
        running=False,
        config=cfg,
        inference=SimpleNamespace(
            status=lambda: {
                "available": True,
                "loaded": False,
                "reason": "no active deployment",
            }
        ),
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    with pytest.raises(RuntimeError, match="推理模型未加载"):
        pipeline.start()

    assert runtime.running is False
    assert pipeline.running is False
    assert "no active deployment" in str(pipeline.stats.last_error)


def test_runtime_pipeline_requires_gpu_bridge_for_nvmm_latest_inference() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "nvmm"
    cfg.inference.backend = "nvmm_latest"
    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        config=cfg.capture,
        wait_preview_frame=lambda *, after_frame_id=None, timeout_s=0.0: None,
    )
    runtime = SimpleNamespace(
        running=False,
        config=cfg,
        status=lambda: {
            "gpu_preprocessor": {
                "available": False,
                "reason": "JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE",
                "detail": "native backend unavailable",
                "native_status": {
                    "available": False,
                    "reason": "native_backend_unavailable",
                },
            }
        },
        process_captured_frame=lambda frame, *, acquired_generation=None: pytest.fail(
            "inference must not start"
        ),
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    with pytest.raises(RuntimeError, match="NVMM TensorRT GPU preprocess is not ready"):
        pipeline.start()

    assert runtime.running is False
    assert pipeline.running is False
    assert "native backend unavailable" in (pipeline.stats.last_error or "")


def test_runtime_pipeline_allows_nvmm_capture_when_inference_is_disabled() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "nvmm"
    cfg.inference.enabled = False
    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        config=cfg.capture,
        wait_preview_frame=lambda *, after_frame_id=None, timeout_s=0.0: None,
    )
    runtime = SimpleNamespace(
        running=False,
        config=cfg,
        status=lambda: {
            "gpu_preprocessor": {
                "available": False,
                "reason": "JETSON_GPU_RESOURCE_BRIDGE_UNAVAILABLE",
            }
        },
        process_captured_frame=lambda frame, *, acquired_generation=None: pytest.fail(
            "inference is disabled"
        ),
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    pipeline.stop()

    assert runtime.running is False


def test_runtime_pipeline_consumes_latest_frames_without_read_frame() -> None:
    frames = [_frame(1), _frame(2)]
    wait_calls: list[tuple[int | None, float]] = []
    processed: list[int] = []
    processed_two = threading.Event()

    def wait_preview_frame(*, after_frame_id: int | None = None, timeout_s: float = 0.0):
        wait_calls.append((after_frame_id, timeout_s))
        for frame in frames:
            if after_frame_id is None or frame.frame_id > after_frame_id:
                return frame
        return None

    def process_captured_frame(
        frame: CapturedFrame, *, acquired_generation: int | None = None
    ) -> None:
        processed.append(frame.frame_id)
        if len(processed) >= 2:
            processed_two.set()

    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        wait_preview_frame=wait_preview_frame,
    )
    runtime = SimpleNamespace(running=False, process_captured_frame=process_captured_frame)
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    assert processed_two.wait(1.0)
    pipeline.stop()

    assert processed == [1, 2]
    assert wait_calls[:2] == [(-1, 0.1), (1, 0.1)]
    status = pipeline.status()
    assert status["consumed_frames"] == 2
    assert "queue" not in status
    assert "capture_frames" not in status


def test_runtime_pipeline_prefers_latest_frame_broker_over_preview_wait() -> None:
    broker = LatestFrameBroker()
    frame = CapturedFrame(
        frame_id=11,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=time.monotonic_ns(),
        capture_wait_ms=1.0,
        image=None,
    )
    broker.publish(
        FrameHandle(
            generation=1,
            frame_id=11,
            source_sequence=11,
            capture_ts_ns=frame.capture_ts_ns,
            clock_domain="monotonic",
            pipeline_running_time_ns=None,
            width=2,
            height=2,
            format="BGR",
            resource=frame,
            metadata={"captured_frame": frame},
        )
    )
    processed: list[int] = []
    processed_one = threading.Event()
    cfg = RuntimeConfig()
    cfg.capture.memory = "cpu"
    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        latest_frame_broker=broker,
        wait_preview_frame=lambda **_kwargs: pytest.fail(
            "broker path must not poll preview frames"
        ),
    )
    runtime = SimpleNamespace(
        running=False,
        config=cfg,
        process_captured_frame=lambda item, *, acquired_generation=None: (
            processed.append(item.frame_id),
            processed_one.set(),
            RuntimeFrameResult(control_intents=[], execution_results=[], observation_updated=False),
        )[-1],
        process_control_tick=lambda: None,
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    assert processed_one.wait(1.0)
    pipeline.stop()

    assert processed == [11]
    assert pipeline.status()["latest_frame_broker"]["acquired_generation"] == 1


def test_runtime_pipeline_skips_stale_frame_before_inference() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "cpu"
    cfg.inference.inference_input_deadline_ms = 10.0
    stale = CapturedFrame(
        frame_id=1,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=time.monotonic_ns() - 100_000_000,
        capture_wait_ms=1.0,
        image=None,
    )
    fresh = CapturedFrame(
        frame_id=2,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=time.monotonic_ns(),
        capture_wait_ms=1.0,
        image=None,
    )
    frames = [stale, fresh]
    processed: list[int] = []
    processed_fresh = threading.Event()

    def wait_preview_frame(*, after_frame_id: int | None = None, timeout_s: float = 0.0):
        del timeout_s
        for frame in frames:
            if after_frame_id is None or frame.frame_id > after_frame_id:
                if frame.frame_id == 2:
                    return CapturedFrame(
                        frame_id=2,
                        width=2,
                        height=2,
                        pixel_format="BGR",
                        ts_ns=time.monotonic_ns(),
                        capture_wait_ms=1.0,
                        image=None,
                    )
                return frame
        return None

    def process_captured_frame(
        frame: CapturedFrame, *, acquired_generation: int | None = None
    ) -> RuntimeFrameResult:
        processed.append(frame.frame_id)
        processed_fresh.set()
        return RuntimeFrameResult(
            control_intents=[], execution_results=[], observation_updated=False
        )

    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        wait_preview_frame=wait_preview_frame,
    )
    runtime = SimpleNamespace(
        running=False,
        config=cfg,
        process_captured_frame=process_captured_frame,
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    assert processed_fresh.wait(1.0)
    pipeline.stop()

    assert processed == [2]
    assert pipeline.stats.skipped_frames >= 1
    assert "input frame age exceeds deadline" in (pipeline.stats.last_error or "")


def test_runtime_service_maps_deepstream_stage_and_drop_statistics() -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream_nvinfer"
    cfg.capture.backend = "deepstream_nvinfer"
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
        ),
        capture=SimpleNamespace(state=CaptureRuntimeState()),
        inference=SimpleNamespace(status=lambda: {"available": True}),
    )
    service.pipeline = SimpleNamespace(
        stats=SimpleNamespace(processed_frames=8, control_observations=7),
        status=lambda: {
            "deepstream": {
                "available": True,
                "running": True,
                "capture_frames": 12,
                "capture_fps": 121.0,
                "input_frames": 10,
                "input_fps": 120.0,
                "output_buffers": 9,
                "output_fps": 116.0,
                "published_batches": 8,
                "published_fps": 110.0,
                "stale_dropped_batches": 1,
                "timestamp_rejected_batches": 2,
                "non_monotonic_dropped_batches": 3,
                "timestamp_source": "first_probe_offset_pts",
                "latest_frame_age_ms": 6.0,
                "last_batch_age_ms": 8.0,
                "inference_input_age_ms_stats": {"p50": 2.1},
                "nvinfer_stage_ms_stats": {"p50": 5.5},
                "nvinfer_timing_scope": "sink_to_src_including_parser",
                "batch_age_ms_stats": {"p50": 9.0},
                "detection_batch_build_ms_stats": {"p50": 0.4},
                "parser": {"decode_ms": 0.3},
                "detection_batch_mailbox": {"overwritten_batches": 4},
            }
        },
    )

    statistics = service.state().statistics

    assert statistics["capture_counter"] == 12
    assert statistics["capture_fps"] == 121.0
    assert statistics["nvinfer_input_counter"] == 10
    assert statistics["nvinfer_input_fps"] == 120.0
    assert statistics["inference_counter"] == 9
    assert statistics["detection_batch_counter"] == 8
    assert statistics["detection_batch_consumed_counter"] == 8
    assert statistics["inference_fps"] == 116.0
    assert statistics["detection_batch_fps"] == 110.0
    assert statistics["skipped_counter"] == 10
    assert statistics["timestamp_rejected_batches"] == 2
    assert statistics["timestamp_source"] == "first_probe_offset_pts"
    assert statistics["batch_age_ms"] == 8.0
    assert statistics["stage_ingress_ms"] == 2.1
    assert statistics["stage_engine_ms"] == 5.5
    assert statistics["stage_engine_scope"] == "sink_to_src_including_parser"
    assert statistics["stage_decode_ms"] == 0.3
    assert statistics["stage_postprocess_ms"] == 0.3
    assert statistics["e2e_latency"] == 9.0


def test_detection_batch_latency_stages_form_one_non_overlapping_timeline() -> None:
    service = object.__new__(RuntimeService)
    capture_ts_ns = 1_000_000_000
    batch = DetectionBatch(
        frame_id=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 10_000_000,
        inference_end_ts_ns=capture_ts_ns + 15_000_000,
        publish_ts_ns=capture_ts_ns + 16_000_000,
        detections=[],
        classes=["target"],
        coordinate_space="roi",
        metadata={"source": "deepstream_nvinfer"},
    )

    service._set_detection_batch_pipeline_timings(
        batch,
        total_start_ns=capture_ts_ns + 16_500_000,
        control_start_ns=capture_ts_ns + 18_000_000,
        done_ns=capture_ts_ns + 19_000_000,
    )

    timings = service.last_pipeline_timings
    assert timings["ingress_ms"] == pytest.approx(10.0)
    assert timings["engine_ms"] == pytest.approx(5.0)
    assert timings["batch_build_ms"] == pytest.approx(1.0)
    assert timings["publish_age_ms"] == pytest.approx(16.0)
    assert timings["control_wait_ms"] == pytest.approx(2.0)
    assert timings["control_ms"] == pytest.approx(1.0)
    assert timings["accounted_ms"] == pytest.approx(19.0)
    assert timings["total_ms"] == pytest.approx(19.0)
    assert timings["unattributed_ms"] == pytest.approx(0.0)


def test_executed_control_activity_aggregates_all_windows_in_one_result() -> None:
    service = object.__new__(RuntimeService)
    service.config = SimpleNamespace(
        control=SimpleNamespace(configured_actuation_delay_s=0.004)
    )
    service._executed_control_samples = deque(
        [
            (945_000_000, 1, 0),
            (965_000_000, 2, -1),
            (985_000_000, -3, 4),
        ]
    )

    activity = service._executed_control_activity(
        control_now_ts_ns=1_000_000_000,
        capture_ts_ns=990_000_000,
        measurement_dt_s=0.030,
    )

    assert activity["executed_counts_last_20ms_x"] == -3
    assert activity["executed_counts_last_40ms_x"] == -1
    assert activity["executed_counts_last_60ms_x"] == 0
    assert activity["executed_counts_since_previous_observation_x"] == -1
    assert activity["executed_counts_since_previous_observation_y"] == 3
    assert activity["latest_successful_send_x_ts_ns"] == 985_000_000
    assert activity["latest_successful_send_y_ts_ns"] == 985_000_000


def test_runtime_service_accepts_batch_when_newer_generation_arrives_after_acquire() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "system"
    cfg.control.latency_reject_if_age_exceeds_ms = 55.0

    class Broker:
        def status(self) -> dict[str, int]:
            return {"published_generation": 2, "published_frame_id": 2}

    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("empty detections should not execute"),
        ),
        capture=SimpleNamespace(latest_frame_broker=Broker(), state=CaptureRuntimeState()),
        inference=SimpleNamespace(
            status=lambda: {"available": True, "loaded": True, "selected": "tensorrt"},
            infer=lambda _frame: InferenceResult(
                available=True,
                detections=[],
                classes=["target"],
                debug={"timings": {}, "preprocess": {"model_width": 2, "model_height": 2}},
            ),
        ),
    )
    service.running = True
    frame = CapturedFrame(
        frame_id=1,
        generation=1,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=time.monotonic_ns(),
        capture_wait_ms=1.0,
        image=None,
    )

    result = service.process_captured_frame(frame, acquired_generation=1)
    state = service.state()

    assert result.observation_updated is True
    assert service.last_frame_context is not None
    assert service.last_inference_status["available"] is True
    assert service.last_inference_status.get("stale_rejected") is not True
    assert service.last_inference_status["acquired_generation"] == 1
    assert service.last_inference_status["latest_generation"] == 1
    assert service.last_inference_status["broker_published_generation"] == 2
    assert service.last_inference_status["generation_lag"] == 1
    assert service.last_inference_status["published_since_acquire"] == 1
    assert state.statistics["stale_drop_count"] == 0
    assert "control_observe_fps" in state.statistics
    assert "inference_ms" in state.statistics
    assert "postprocess_ms" in state.statistics


def test_target_pipeline_diagnostics_explain_zero_decode_candidates(caplog) -> None:
    cfg = RuntimeConfig()

    class Broker:
        def status(self) -> dict[str, int]:
            return {"published_generation": 1, "published_frame_id": 1}

    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("zero detections must not execute"),
        ),
        capture=SimpleNamespace(latest_frame_broker=Broker(), state=CaptureRuntimeState()),
        inference=SimpleNamespace(
            status=lambda: {"available": True, "loaded": True, "selected": "tensorrt"},
            infer=lambda _frame: InferenceResult(
                available=True,
                detections=[],
                classes=["target"],
                debug={
                    "timings": {},
                    "preprocess": {"model_width": 320, "model_height": 320},
                    "decode": {
                        "raw_candidates": 8400,
                        "max_score": 0.18,
                        "threshold_candidates": 0,
                        "nms_detections": 0,
                    },
                },
            ),
        ),
    )
    service.running = True
    frame = CapturedFrame(
        frame_id=1,
        generation=1,
        width=640,
        height=640,
        pixel_format="BGR",
        ts_ns=time.monotonic_ns(),
        capture_wait_ms=1.0,
        image=None,
    )

    with caplog.at_level("INFO", logger="novasight.runtime.service"):
        service.process_captured_frame(frame, acquired_generation=1)

    diagnostics = service.state().vision["target_pipeline"]
    assert diagnostics["code"] == "CONFIDENCE_THRESHOLD_REJECTED"
    assert diagnostics["stage"] == "inference_decode"
    assert diagnostics["counts"]["decode_raw_candidates"] == 8400
    assert diagnostics["counts"]["threshold_candidates"] == 0
    assert "target pipeline blocked" in caplog.text


def test_target_pipeline_diagnostics_explain_selection_fov_rejection() -> None:
    cfg = RuntimeConfig()
    cfg.control.target_fov_radius_px = 50.0
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("rejected target must not execute"),
        ),
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.9, x1=0, y1=280, x2=40, y2=380)],
        classes=["target"],
        coordinate_space="roi",
    )

    service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=640,
        roi_offset_y=220,
    )

    diagnostics = service.state().vision["target_pipeline"]
    control = service.state().vision["control"]
    candidate_filter = control["candidate_filter"]
    rejected = candidate_filter["rejected"][0]
    assert diagnostics["code"] == "OUTSIDE_TARGET_FOV"
    assert diagnostics["stage"] == "association_filter"
    assert diagnostics["counts"]["mapped_detections"] == 1
    assert diagnostics["counts"]["association_candidates"] == 0
    assert diagnostics["counts"]["tracker_active"] == 0
    assert diagnostics["counts"]["inside_fov"] == 0
    assert diagnostics["rejection_reasons"] == ["selection_fov"]
    assert control["global_state"] == "TARGET_UNAVAILABLE"
    assert candidate_filter["selection_center_px"] == {"x": 320.0, "y": 320.0}
    assert candidate_filter["selection_radius_px"] == pytest.approx(50.0)
    assert rejected["aim_x"] == pytest.approx(20.0)
    assert rejected["aim_y"] == pytest.approx(302.0)
    assert rejected["distance_px"] == pytest.approx(math.hypot(300.0, 18.0))


def test_target_pipeline_diagnostics_expose_effective_class_filter_and_rejected_classes() -> None:
    cfg = RuntimeConfig()
    cfg.inference.detection_class_filter = "1"
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("class-filtered target must not execute"),
        ),
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns()
    service.process_detection_batch(
        DetectionBatch(
            frame_id=1,
            generation=1,
            capture_ts_ns=capture_ts_ns,
            inference_start_ts_ns=capture_ts_ns + 1_000,
            inference_end_ts_ns=capture_ts_ns + 2_000,
            detections=[
                Detection(cls=0, score=0.90, x1=260, y1=250, x2=320, y2=390),
                Detection(cls=2, score=0.85, x1=330, y1=250, x2=390, y2=390),
            ],
            classes=["body", "head", "other"],
            coordinate_space="roi",
        ),
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=640,
        roi_offset_y=220,
    )

    vision = service.state().vision
    candidate_filter = vision["control"]["candidate_filter"]
    diagnostics = vision["target_pipeline"]

    assert diagnostics["code"] == "BASIC_CANDIDATE_REJECTED"
    assert candidate_filter["effective_class_filter"] == "1"
    assert candidate_filter["basic"]["rejected_class_ids"] == [0, 2]


def test_control_and_button_state_logs_only_on_trigger_state_changes(caplog, monkeypatch) -> None:
    monotonic_s = [100.0]
    monkeypatch.setattr("novasight.runtime.service.time.monotonic", lambda: monotonic_s[0])
    service = RuntimeService(
        RuntimeConfig(),
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="kmnet",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
        ),
    )
    context = FrameContext(
        frame_id=1,
        width=640,
        height=640,
        capture_ts_ns=time.monotonic_ns(),
        detections=[Detection(cls=0, score=0.9, x1=280, y1=240, x2=360, y2=400)],
    )
    target = Track(
        track_id=7,
        cls=0,
        score=0.9,
        box=context.detections[0].box,
    )
    command = SimpleNamespace(dx=4.0, dy=0.0, reason="test")

    with caplog.at_level("DEBUG", logger="novasight.runtime.service"):
        service._log_box_input_state(BoxInputState(), "hardware")
        service._log_box_input_state(BoxInputState(), "hardware")
        service._log_box_input_state(BoxInputState(left=True), "hardware")
        monotonic_s[0] += 2.0
        service._log_box_input_state(BoxInputState(left=True), "hardware")

        service._log_control_decision(
            context=context,
            target=target,
            command=command,
            can_emit=False,
            trigger_raw={"source": "kmnet_executor", "left": False, "right": False},
            output_mode="kmnet",
            hardware_kind="kmnet",
            trigger_mode="hardware",
        )
        monotonic_s[0] += 2.0
        service._log_control_decision(
            context=context,
            target=target,
            command=command,
            can_emit=False,
            trigger_raw={"source": "kmnet_executor", "left": False, "right": False},
            output_mode="kmnet",
            hardware_kind="kmnet",
            trigger_mode="hardware",
        )
        service._log_control_decision(
            context=context,
            target=target,
            command=command,
            can_emit=True,
            trigger_raw={"source": "kmnet_executor", "left": True, "right": False},
            output_mode="kmnet",
            hardware_kind="kmnet",
            trigger_mode="hardware",
        )

    box_logs = [
        record for record in caplog.records if record.getMessage().startswith("box input source=")
    ]
    decision_logs = [
        record for record in caplog.records if record.getMessage().startswith("control decision")
    ]
    assert len(box_logs) == 1
    assert "active=True" in box_logs[0].getMessage()
    assert len(decision_logs) == 2
    assert "emit=False" in decision_logs[0].getMessage()
    assert "emit=True" in decision_logs[1].getMessage()


def test_detection_batch_with_hardware_trigger_reaches_mouse_controller_scheduler_and_kmnet() -> (
    None
):
    config = RuntimeConfig()
    config.control.trigger_mode = "hardware"
    config.control.mode = "universal_saturated"
    kmnet = _TriggeredKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        # The ROI is shifted 40 px left. Full-screen center maps to ROI x=360,
        # so this aim at x=410 is a real +50 px control error.
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    observation_result = service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    send_result = service.process_control_tick()

    assert len(observation_result.control_intents) == 1
    assert service.last_control is not None
    assert service.last_control["pipeline"]["control_allowed"] is True
    assert service.last_control["trigger_active"] is True
    assert service.last_control["will_emit"] is True
    assert service.last_control["selector_debug"]["control_center_roi_px"] == {
        "x": 360.0,
        "y": 320.0,
    }
    assert service.last_control["mouse_observation"]["predicted_aim_x_roi_px"] == pytest.approx(
        410.0
    )
    assert service.last_control["pipeline"]["predicted_error_x_px"] == pytest.approx(50.0)
    assert service.last_control["dx"] == 10
    assert service.last_control["dy"] == 0
    assert len(send_result.execution_results) == 1
    assert send_result.execution_results[0].sent is True
    assert len(kmnet.outputs) == 1
    assert kmnet.outputs[0].dx != 0 or kmnet.outputs[0].dy != 0


def test_default_target_driven_mode_sends_one_detection_without_kmnet_button_sample() -> None:
    config = RuntimeConfig()
    config.control.mode = "universal_saturated"
    kmnet = _UnavailableButtonKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    observation_result = service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    send_result = service.process_control_tick()

    assert config.control.trigger_mode == "always"
    assert len(observation_result.control_intents) == 1
    assert service.last_control is not None
    assert service.last_control["will_emit"] is True
    assert len(send_result.execution_results) == 1
    assert send_result.execution_results[0].sent is True
    assert len(kmnet.outputs) == 1


def test_scheduler_disabled_sends_detection_budget_in_observation_call() -> None:
    config = RuntimeConfig()
    config.control.mode = "universal_saturated"
    config.control.scheduler_enabled = False
    kmnet = _UnavailableButtonKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    result = service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )

    assert len(result.execution_results) == 1
    assert result.execution_results[0].sent is True
    assert result.execution_results[0].metadata["stage"] == "direct_output"
    assert [(output.dx, output.dy) for output in kmnet.outputs] == [(10, 0)]
    assert executors.scheduler is None


def test_dual_phase_algorithm_delivers_latest_observation_on_control_tick() -> None:
    algorithm_id = "dual_phase_atan_robust_predictive_v2"
    config = RuntimeConfig()
    config.control.active_algorithm = algorithm_id
    config.control.trigger_mode = "always"
    # V2 uses latest-replace even when this legacy switch is enabled.
    config.control.scheduler_enabled = True
    kmnet = _UnavailableButtonKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns() - 10_000_000
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    observation_result = service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    latest_result = service.process_detection_batch(
        DetectionBatch(
            frame_id=2,
            generation=2,
            capture_ts_ns=capture_ts_ns + 5_000_000,
            inference_start_ts_ns=capture_ts_ns + 5_001_000,
            inference_end_ts_ns=capture_ts_ns + 5_002_000,
            detections=[Detection(cls=0, score=0.95, x1=334, y1=250, x2=494, y2=568)],
            classes=["target"],
            coordinate_space="roi",
        ),
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    tick_result = service.process_control_tick()

    assert executors.scheduler is not None
    assert executors.single_command_per_observation is False
    assert executors.latest_replace is True
    assert len(observation_result.control_intents) == 1
    assert len(observation_result.execution_results) == 1
    assert observation_result.execution_results[0].sent is False
    assert observation_result.execution_results[0].metadata["stage"] == "scheduler"
    assert observation_result.execution_results[0].metadata["action"] == "replace_plan"
    assert latest_result.execution_results[0].metadata["source_frame_id"] == 2
    assert len(tick_result.execution_results) == 1
    assert tick_result.execution_results[0].sent is True
    assert tick_result.execution_results[0].intent.source_frame_id == 2
    assert len(kmnet.outputs) == 1
    assert int(kmnet.outputs[0].dx) != 0
    assert service.last_control is not None
    assert service.last_control["pipeline"]["scheduler_used"] is True
    assert service.last_control["pipeline"]["algorithm"] == algorithm_id
    assert service.last_control["pipeline"]["executor_success"] is True


def test_dual_phase_far_command_above_127_reaches_kmnet_unchanged() -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    precise = config.control.dual_phase_atan_robust_predictive_v2
    precise.atan.scale_counts = 1024.0
    precise.atan.far.kp = 0.90
    precise.atan.far.max_counts_per_update = 600.0
    kmnet = _UnavailableButtonKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns() - 10_000_000
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=390, y1=250, x2=550, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    tick_result = service.process_control_tick()

    assert tick_result.execution_results[0].sent is True
    assert len(kmnet.outputs) == 1
    assert kmnet.outputs[0].dx > 127
    assert kmnet.outputs[0].dx == service.last_control["dx"]


def test_latest_replace_discards_popped_command_when_newer_observation_arrives() -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    kmnet = _TriggeredKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    expires_ts_ns = time.monotonic_ns() + 1_000_000_000

    def intent(frame_id: int, dx: int) -> ControlIntent:
        return ControlIntent(
            dx=dx,
            dy=0,
            action="move",
            confidence=1.0,
            reason="test",
            source_id="test",
            source_frame_id=frame_id,
            source_track_id=1,
            trajectory_generation=frame_id,
            trigger_required=True,
            trigger_active=True,
            command_expires_ts_ns=expires_ts_ns,
        )

    executors.execute(intent(1, 10))
    executors._executor_lock.acquire()
    tick_results: list[ExecutionResult] = []
    thread = threading.Thread(target=lambda: tick_results.append(executors.tick_pending()))
    try:
        thread.start()
        deadline = time.monotonic() + 1.0
        while executors.status()["scheduler"]["has_pending"]:
            assert time.monotonic() < deadline
            time.sleep(0.001)
        executors.execute(intent(2, 20))
    finally:
        executors._executor_lock.release()
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert tick_results[0].sent is False
    assert tick_results[0].metadata["action"] == "scheduler_superseded"
    assert kmnet.outputs == []

    latest = executors.tick_pending(now_s=time.monotonic() + 0.010)
    assert latest.sent is True
    assert [(output.source_frame_id, output.dx) for output in kmnet.outputs] == [(2, 20)]


def test_dual_phase_recoil_reads_real_left_trigger_in_always_mode() -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    config.control.trigger_mode = "always"
    config.control.shared.recoil_enabled = True
    config.control.shared.recoil_start_delay_ms = 0.0
    config.control.shared.recoil_y_counts_per_observation = 1.0
    kmnet = _HeldLeftKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    base_capture_ts_ns = time.monotonic_ns() - 30_000_000

    for index in range(2):
        capture_ts_ns = base_capture_ts_ns + index * 10_000_000
        service.process_detection_batch(
            DetectionBatch(
                frame_id=index + 1,
                generation=index + 1,
                capture_ts_ns=capture_ts_ns,
                inference_start_ts_ns=capture_ts_ns + 1_000,
                inference_end_ts_ns=capture_ts_ns + 2_000,
                detections=[Detection(cls=0, score=0.95, x=340, y=298, w=40, h=100)],
                classes=["target"],
                coordinate_space="roi",
            ),
            width=640,
            height=640,
            source_width=1920,
            source_height=1080,
            roi_offset_x=600,
            roi_offset_y=220,
        )

    tick_result = service.process_control_tick()

    assert len(tick_result.execution_results) == 1
    assert tick_result.execution_results[0].sent is True
    assert len(kmnet.outputs) == 1
    assert (kmnet.outputs[0].dx, kmnet.outputs[0].dy) == (0, 1)
    assert service.last_control is not None
    assert service.last_control["pipeline"]["recoil_active"] is True
    assert service.last_control["pipeline"]["recoil_y_counts_float"] == pytest.approx(1.0)


@pytest.mark.parametrize("terminal_state", ["stopped", "cancelled", "fatal"])
def test_dual_phase_rejects_late_batch_after_terminal_runtime_state(
    terminal_state: str,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    config.control.trigger_mode = "always"
    kmnet = _UnavailableButtonKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    if terminal_state == "stopped":
        service.running = False
    elif terminal_state == "cancelled":
        service.cancel_control("RUNTIME_STOPPED")
    else:
        service.record_fatal_error("inference", RuntimeError("boom"), "crash.log")
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    result = service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    assert result.control_intents == []
    assert result.execution_results == []
    assert kmnet.outputs == []
    assert service._accepted_batch_generation == -1
    assert service._accepted_batch_capture_ts_ns == 0
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["terminal_rejected"] is True
    assert service.last_inference_status["reason"] == (
        "RUNTIME_FATAL_ERROR" if terminal_state == "fatal" else "RUNTIME_STOPPED"
    )


def test_runtime_session_reset_clears_observation_telemetry_immediately() -> None:
    config = RuntimeConfig()
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )
    service.running = True
    service.last_target = {"track_id": 7}
    service.last_execution = {"sent": True}
    service.last_inference_status = {
        "ran": True,
        "available": True,
        "frame_age_ms": 8.0,
        "inference_ms": 2.0,
    }
    service.last_pipeline_timings = {"control_ms": 1.5}
    service.stale_drop_count = 4

    service.reset_runtime_session("RUNTIME_STOPPED")

    assert service.running is False
    assert service.last_target is None
    assert service.last_execution is None
    assert service.last_pipeline_timings == {}
    assert service.stale_drop_count == 0
    assert service.last_inference_status == {
        "ran": False,
        "available": False,
        "reason": "RUNTIME_STOPPED",
        "terminal_rejected": True,
    }


def test_terminal_state_remains_authoritative_over_concurrent_batch_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )
    service.running = True

    def stop_during_freshness(_batch: DetectionBatch) -> str:
        service.cancel_control("RUNTIME_STOPPED")
        return "synthetic generation rollback"

    monkeypatch.setattr(
        service,
        "_detection_batch_freshness_reason",
        stop_during_freshness,
    )
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[],
        classes=["target"],
        coordinate_space="roi",
    )

    result = service.process_detection_batch(batch, width=640, height=640)

    assert result.execution_results == []
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["terminal_rejected"] is True
    assert service.last_inference_status["reason"] == "RUNTIME_STOPPED"


def test_dual_phase_preserves_zero_generation_through_latest_replace_path() -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    config.control.trigger_mode = "always"
    kmnet = _UnavailableButtonKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=42,
        generation=0,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    result = service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    tick_result = service.process_control_tick()

    assert service.last_frame_context is not None
    assert service.last_frame_context.generation == 0
    assert result.control_intents[0].trajectory_generation == 0
    assert result.execution_results[0].metadata["trajectory_generation"] == 0
    assert tick_result.execution_results[0].intent.trajectory_generation == 0
    assert service.last_control is not None
    assert service.last_control["trajectory_generation"] == 0
    assert service.last_control["pipeline"]["generation"] == 0


def test_dual_phase_control_tick_observes_trigger_release_without_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RuntimeConfig()
    config.control.active_algorithm = "dual_phase_atan_robust_predictive_v2"
    config.control.trigger_mode = "hardware"
    kmnet = _UnavailableButtonKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True
    release_calls: list[bool] = []
    robust_controller = service.control_algorithms.active_controller
    original_release = robust_controller.release_trigger

    def record_release() -> None:
        release_calls.append(True)
        original_release()

    monkeypatch.setattr(robust_controller, "release_trigger", record_release)

    result = service.process_control_tick()

    assert release_calls == [True]
    assert result.execution_results == []
    assert kmnet.outputs == []


def test_cancel_control_resets_detection_batch_generation_cursor() -> None:
    config = RuntimeConfig()
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=ExecutorRegistry.from_config(config),
    )
    service._accepted_batch_generation = 10
    service._accepted_batch_capture_ts_ns = 2_000_000_000

    service.cancel_control("RUNTIME_STOPPED")

    assert service._accepted_batch_generation == -1
    assert service._accepted_batch_capture_ts_ns == 0


def test_hot_switch_from_calibrated_to_universal_still_sends_to_kmnet() -> None:
    config = RuntimeConfig()
    config.control.trigger_mode = "hardware"
    config.control.mode = "calibrated_angular"
    kmnet = _TriggeredKmNet()
    executors = ExecutorRegistry.from_config(config)
    executors.executors["kmnet"] = kmnet
    service = RuntimeService(
        config,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=executors,
    )
    service.running = True

    updated = copy.deepcopy(config)
    updated.control.mode = "universal_saturated"
    service.update_config(updated)

    capture_ts_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=1,
        generation=1,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.95, x1=330, y1=250, x2=490, y2=568)],
        classes=["target"],
        coordinate_space="roi",
    )

    observation_result = service.process_detection_batch(
        batch,
        width=640,
        height=640,
        source_width=1920,
        source_height=1080,
        roi_offset_x=600,
        roi_offset_y=220,
    )
    send_result = service.process_control_tick()

    assert service.control_algorithms.active_controller.mode == "universal_saturated"
    assert service.last_control is not None
    assert service.last_control["pipeline"]["control_mode"] == "universal_saturated"
    assert service.last_control["dx"] == 10
    assert len(observation_result.control_intents) == 1
    assert len(send_result.execution_results) == 1
    assert send_result.execution_results[0].sent is True
    assert kmnet.outputs[-1].dx == 5


def test_runtime_service_records_generation_lag_without_rejecting_in_flight_batch() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "system"
    cfg.control.latency_reject_if_age_exceeds_ms = 55.0

    class Broker:
        def __init__(self) -> None:
            self.published_generation = 1
            self.published_frame_id = 1

        def status(self) -> dict[str, int]:
            return {
                "published_generation": self.published_generation,
                "published_frame_id": self.published_frame_id,
            }

    broker = Broker()

    def infer(_frame):
        broker.published_generation = 2
        broker.published_frame_id = 2
        return InferenceResult(
            available=True,
            detections=[],
            classes=["target"],
            debug={"timings": {}, "preprocess": {"model_width": 2, "model_height": 2}},
        )

    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("empty detections should not execute"),
        ),
        capture=SimpleNamespace(latest_frame_broker=broker, state=CaptureRuntimeState()),
        inference=SimpleNamespace(
            status=lambda: {"available": True, "loaded": True, "selected": "tensorrt"},
            infer=infer,
        ),
    )
    service.running = True
    frame = CapturedFrame(
        frame_id=1,
        generation=1,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=time.monotonic_ns(),
        capture_wait_ms=1.0,
        image=None,
    )

    result = service.process_captured_frame(frame, acquired_generation=1)

    assert result.observation_updated is True
    assert service.last_frame_context is not None
    assert service.last_inference_status["available"] is True
    assert service.last_inference_status.get("stale_rejected") is not True
    assert service.last_inference_status["acquired_generation"] == 1
    assert service.last_inference_status["latest_generation"] == 1
    assert service.last_inference_status["broker_published_generation"] == 2
    assert service.last_inference_status["generation_lag"] == 1
    assert service.last_inference_status["published_since_acquire"] == 1
    assert service.stale_drop_count == 0


def test_runtime_service_rejects_batch_older_than_acquired_generation() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "system"
    cfg.control.latency_reject_if_age_exceeds_ms = 55.0

    class Broker:
        def status(self) -> dict[str, int]:
            return {"published_generation": 2, "published_frame_id": 2}

    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("old acquired DetectionBatch must not execute"),
        ),
        capture=SimpleNamespace(latest_frame_broker=Broker(), state=CaptureRuntimeState()),
        inference=SimpleNamespace(
            status=lambda: {"available": True, "loaded": True, "selected": "tensorrt"},
            infer=lambda _frame: InferenceResult(
                available=True,
                detections=[],
                classes=["target"],
                debug={"timings": {}, "preprocess": {"model_width": 2, "model_height": 2}},
            ),
        ),
    )
    frame = CapturedFrame(
        frame_id=1,
        generation=1,
        width=2,
        height=2,
        pixel_format="BGR",
        ts_ns=time.monotonic_ns(),
        capture_wait_ms=1.0,
        image=None,
    )

    result = service.process_captured_frame(frame, acquired_generation=2)

    assert result.observation_updated is False
    assert service.last_frame_context is None
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["stale_rejected"] is True
    assert service.last_inference_status["acquired_generation"] == 2
    assert service.last_inference_status["latest_generation"] == 2
    assert "latest generation" in service.last_inference_status["reason"]


def test_runtime_pipeline_resets_frame_cursor_when_restarted() -> None:
    frames = [_frame(10)]
    processed: list[int] = []
    processed_one = threading.Event()

    def wait_preview_frame(*, after_frame_id: int | None = None, timeout_s: float = 0.0):
        del timeout_s
        for frame in frames:
            if after_frame_id is None or frame.frame_id > after_frame_id:
                return frame
        return None

    def process_captured_frame(
        frame: CapturedFrame, *, acquired_generation: int | None = None
    ) -> None:
        processed.append(frame.frame_id)
        processed_one.set()

    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        wait_preview_frame=wait_preview_frame,
    )
    runtime = SimpleNamespace(running=False, process_captured_frame=process_captured_frame)
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    assert processed_one.wait(1.0)
    pipeline.stop()

    frames[:] = [_frame(1)]
    processed_one.clear()

    pipeline.start()
    assert processed_one.wait(1.0)
    pipeline.stop()

    assert processed == [10, 1]


def test_runtime_pipeline_stops_when_capture_becomes_unavailable() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "cpu"
    observed = threading.Event()
    capture_state = SimpleNamespace(available=True, last_error=None)
    capture = SimpleNamespace(
        source=object(),
        state=capture_state,
        session=SimpleNamespace(running=True),
    )

    def wait_preview_frame(*, after_frame_id: int | None = None, timeout_s: float = 0.0):
        del after_frame_id, timeout_s
        capture_state.available = False
        capture_state.last_error = "capture read failed: boom"
        observed.set()
        return None

    capture.wait_preview_frame = wait_preview_frame
    runtime = SimpleNamespace(
        running=False,
        config=cfg,
        process_captured_frame=lambda _frame, *, acquired_generation=None: pytest.fail(
            "unavailable capture must not process inference"
        ),
        process_control_tick=lambda: None,
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    try:
        assert observed.wait(1.0)
        deadline = threading.Event()
        assert deadline.wait(0.05) is False
        assert runtime.running is False
        assert pipeline.running is False
        assert "capture read failed" in (pipeline.stats.last_error or "")
    finally:
        pipeline.stop()


def test_runtime_pipeline_consumes_frames_from_capture_session_thread() -> None:
    class ThreadedSource:
        backend_label = "test:threaded"

        def __init__(self) -> None:
            self.closed = False
            self.frame_id = 0

        def read(self) -> CapturedFrame | None:
            if self.closed:
                return None
            time_sleep.wait(0.005)
            self.frame_id += 1
            return CapturedFrame(
                frame_id=self.frame_id,
                width=2,
                height=2,
                pixel_format="BGR",
                ts_ns=1_000_000_000 + self.frame_id * 1_000_000,
                capture_wait_ms=0.1,
                image=None,
            )

        def close(self) -> None:
            self.closed = True

    time_sleep = threading.Event()
    source = ThreadedSource()
    profile = CaptureProfile(
        device="/dev/test",
        pixel_format="BGR",
        width=2,
        height=2,
        fps=120,
        preference="manual",
        selection_reason="threaded test",
    )
    capture = CaptureSession(source_factory=lambda _profile: source)
    capture.start(profile)
    processed: list[int] = []
    processed_two = threading.Event()

    def process_captured_frame(
        frame: CapturedFrame, *, acquired_generation: int | None = None
    ) -> None:
        processed.append(frame.frame_id)
        if len(processed) >= 2:
            processed_two.set()

    cfg = RuntimeConfig()
    cfg.capture.memory = "cpu"
    runtime = SimpleNamespace(
        running=False,
        config=cfg,
        process_captured_frame=process_captured_frame,
        process_control_tick=lambda: None,
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    try:
        pipeline.start()
        assert processed_two.wait(1.0)
        assert runtime.running is True
        assert pipeline.status()["processed_frames"] >= 2
        assert processed == sorted(processed)
    finally:
        pipeline.stop()
        capture.stop("test complete")


def test_detection_batch_copies_mutable_inputs() -> None:
    detections = [Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)]
    classes = ["body"]
    metadata = {"source": "custom_tensorrt"}

    batch = DetectionBatch(
        frame_id=7,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=detections,
        classes=classes,
        coordinate_space="roi",
        metadata=metadata,
    )
    detections.clear()
    classes[0] = "changed"
    metadata["source"] = "changed"

    assert len(batch.detections) == 1
    assert batch.classes == ["body"]
    assert batch.metadata == {"source": "custom_tensorrt"}


def test_runtime_service_process_detection_batch_uses_roi_contract() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail(
                "default hardware trigger should not emit in this seam test"
            ),
        ),
    )
    service.running = True
    batch = DetectionBatch(
        frame_id=7,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="roi",
    )

    service.process_detection_batch(
        batch,
        width=480,
        height=480,
        source_width=1920,
        source_height=1080,
        roi_offset_x=720,
        roi_offset_y=300,
    )

    assert service.last_frame_context is not None
    assert service.last_frame_context.frame_id == 7
    assert service.last_frame_context.width == 480
    assert service.last_frame_context.height == 480
    assert len(service.last_frame_context.detections) == 1
    assert service.last_inference_status["available"] is True
    assert service.last_inference_status["source"] == "detection_batch"
    assert service.last_inference_status["mapped_detections"] == 1
    assert service.last_inference_status["latency_source"] == "custom_tensorrt_done"
    assert service.last_inference_status["inference_ms"] == pytest.approx(0.001)
    assert service.last_inference_status["source_width"] == 1920
    assert service.last_inference_status["source_height"] == 1080
    assert service.last_inference_status["source_geometry_source"] == "detection_batch"


def test_runtime_service_rejects_stale_detection_batch_before_control() -> None:
    cfg = RuntimeConfig()
    cfg.runtime.freshness_threshold_ms = 10.0
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("stale DetectionBatch must not execute"),
        ),
    )
    service.running = True
    capture_ts_ns = time.monotonic_ns() - 50_000_000
    batch = DetectionBatch(
        frame_id=17,
        capture_ts_ns=capture_ts_ns,
        inference_start_ts_ns=capture_ts_ns + 1_000,
        inference_end_ts_ns=capture_ts_ns + 2_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="roi",
        metadata={"source": "deepstream", "timestamp_source": "gst_clock_base_time_pts"},
    )

    result = service.process_detection_batch(
        batch,
        width=480,
        height=480,
        source_width=1920,
        source_height=1080,
        roi_offset_x=720,
        roi_offset_y=300,
    )

    assert result.control_intents == []
    assert service.last_frame_context is None
    assert service.last_control is not None
    assert service.last_control["control_allowed"] is False
    assert service.last_control["runtime_reset_reason"] == "DETECTION_BATCH_STALE"
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["mapped_detections"] == 0
    assert service.last_inference_status["stale_rejected"] is True
    assert service.last_inference_status["detection_batch_metadata"]["source"] == "deepstream"
    assert "frame age exceeds" in service.last_inference_status["reason"]
    assert service.last_inference_status["source_geometry_trusted"] is True
    assert service.last_inference_status["roi_offset_x"] == 720
    assert service.last_inference_status["roi_offset_y"] == 300
    assert service.last_inference_status["roi_region"] == {"x": 720, "y": 300, "w": 480, "h": 480}
    assert service.last_pipeline_timings["engine_ms"] == pytest.approx(0.001)
    assert service.last_pipeline_timings["handoff_ms"] >= 0.0
    assert service.last_pipeline_timings["postprocess_ms"] == pytest.approx(0.0)


def test_runtime_service_rejects_detection_batch_marked_stale() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("stale DetectionBatch must not execute"),
        ),
    )
    service.running = True
    now_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=21,
        generation=21,
        capture_ts_ns=now_ns,
        inference_start_ts_ns=now_ns + 1_000,
        inference_end_ts_ns=now_ns + 2_000,
        detections=[],
        classes=["0"],
        coordinate_space="roi",
        is_stale=True,
        clock_domain="monotonic",
    )

    result = service.process_detection_batch(
        batch,
        width=480,
        height=480,
        source_width=1920,
        source_height=1080,
        roi_offset_x=720,
        roi_offset_y=300,
    )

    assert result.control_intents == []
    assert service.last_frame_context is None
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["latest_rejected"] is True
    assert service.last_inference_status["is_stale"] is True
    assert "marked stale" in service.last_inference_status["reason"]


def test_runtime_service_rejects_detection_batch_generation_and_capture_rollback() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("empty detections should not emit"),
        ),
    )
    service.running = True
    now_ns = time.monotonic_ns()
    first = DetectionBatch(
        frame_id=30,
        generation=30,
        capture_ts_ns=now_ns,
        inference_start_ts_ns=now_ns + 1_000,
        inference_end_ts_ns=now_ns + 2_000,
        detections=[],
        classes=["0"],
        coordinate_space="roi",
    )
    generation_rollback = DetectionBatch(
        frame_id=31,
        generation=29,
        capture_ts_ns=now_ns + 10_000,
        inference_start_ts_ns=now_ns + 11_000,
        inference_end_ts_ns=now_ns + 12_000,
        detections=[],
        classes=["0"],
        coordinate_space="roi",
    )
    capture_rollback = DetectionBatch(
        frame_id=32,
        generation=31,
        capture_ts_ns=now_ns,
        inference_start_ts_ns=now_ns + 13_000,
        inference_end_ts_ns=now_ns + 14_000,
        detections=[],
        classes=["0"],
        coordinate_space="roi",
    )

    accepted = service.process_detection_batch(
        first,
        width=480,
        height=480,
        source_width=1920,
        source_height=1080,
        roi_offset_x=720,
        roi_offset_y=300,
    )
    rejected_generation = service.process_detection_batch(
        generation_rollback,
        width=480,
        height=480,
        source_width=1920,
        source_height=1080,
        roi_offset_x=720,
        roi_offset_y=300,
    )
    generation_reason = service.last_inference_status["reason"]
    generation_latest_rejected = service.last_inference_status["latest_rejected"]
    rejected_capture = service.process_detection_batch(
        capture_rollback,
        width=480,
        height=480,
        source_width=1920,
        source_height=1080,
        roi_offset_x=720,
        roi_offset_y=300,
    )

    assert accepted.observation_updated is True
    assert rejected_generation.control_intents == []
    assert "generation must increase" in generation_reason
    assert generation_latest_rejected is True
    assert rejected_capture.control_intents == []
    assert "capture_ts_ns must increase" in service.last_inference_status["reason"]


def test_runtime_service_state_exposes_detection_batch_fps_from_pipeline_stats() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
        ),
    )
    service.pipeline = SimpleNamespace(
        stats=SimpleNamespace(
            window_processed_frames=12,
            skipped_frames=3,
            inference_fps=118.5,
            detection_batch_fps=117.8,
            window_control_observations=10,
            control_observation_fps=116.6,
            queue_latency_ms=0.4,
            inference_latency_ms=1.6,
            e2e_latency_ms=8.2,
        ),
        status=lambda: {},
    )

    state = service.state()

    assert state.statistics["inference_fps"] == pytest.approx(118.5)
    assert state.statistics["detection_batch_fps"] == pytest.approx(117.8)
    assert state.statistics["control_observation_counter"] == 10
    assert state.statistics["control_observation_fps"] == pytest.approx(116.6)
    assert state.statistics["skipped_counter"] == 3


def test_runtime_service_state_samples_pipeline_once() -> None:
    cfg = RuntimeConfig()
    cfg.inference.backend = "deepstream_nvinfer"
    calls = 0

    def pipeline_status() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"deepstream": {"running": True, "available": True}}

    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
        ),
        inference=SimpleNamespace(status=lambda: {}),
    )
    service.pipeline = SimpleNamespace(stats=None, status=pipeline_status)

    state = service.state()

    assert calls == 1
    assert state.inference["running"] is True


def test_runtime_service_rejects_non_roi_detection_batch() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("invalid DetectionBatch must not execute"),
        ),
    )
    service.running = True
    batch = DetectionBatch(
        frame_id=7,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="model",
    )

    result = service.process_detection_batch(batch, width=480, height=480)

    assert result.control_intents == []
    assert service.last_frame_context is None
    assert service.last_pipeline_timings["engine_ms"] == pytest.approx(0.001)
    assert service.last_pipeline_timings["control_ms"] == 0.0
    assert service.last_control is not None
    assert service.last_control["control_allowed"] is False
    assert (
        service.last_control["runtime_reset_reason"] == "DETECTION_BATCH_COORDINATE_SPACE_INVALID"
    )
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["source"] == "detection_batch"
    assert service.last_inference_status["reason"] == "DetectionBatch coordinate_space must be roi"
    assert service.last_inference_status["detection_coordinate_space"] == "model"
    assert service.last_inference_status["inference_ms"] == pytest.approx(0.001)
    assert service.last_inference_status["latency_source"] == "custom_tensorrt_done"
    assert service.last_inference_status["input_width"] == 480
    assert service.last_inference_status["input_height"] == 480


def test_runtime_service_rejects_invalid_roi_detection_batch() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("invalid DetectionBatch must not execute"),
        ),
    )
    service.running = True
    batch = DetectionBatch(
        frame_id=8,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[Detection(cls=0, score=0.9, x1=520, y1=20, x2=620, y2=80)],
        classes=["0"],
        coordinate_space="roi",
    )

    result = service.process_detection_batch(batch, width=480, height=480)

    assert result.control_intents == []
    assert service.last_frame_context is None
    assert service.last_pipeline_timings["engine_ms"] == pytest.approx(0.001)
    assert service.last_pipeline_timings["control_ms"] == 0.0
    assert service.last_control is not None
    assert service.last_control["control_allowed"] is False
    assert service.last_control["runtime_reset_reason"] == "DETECTION_BATCH_ROI_CONTRACT_INVALID"
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["source"] == "detection_batch"
    assert "center must be inside ROI" in service.last_inference_status["reason"]
    assert service.last_inference_status["detection_coordinate_space"] == "roi"
    assert service.last_inference_status["raw_detections"] == 1
    assert service.last_inference_status["mapped_detections"] == 0
    assert service.last_inference_status["inference_ms"] == pytest.approx(0.001)
    assert service.last_inference_status["latency_source"] == "custom_tensorrt_done"
