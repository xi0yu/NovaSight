"""Tests for core runtime primitives and pipeline behavior."""
import threading
from types import SimpleNamespace

import pytest

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.contracts import Detection, DetectionBatch
from novasight.runtime import (
    FailFastHandler,
    RuntimeConfigStore,
    RuntimePipeline,
    RuntimeService,
    configure_logging,
)


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
    runtime = SimpleNamespace(running=False, process_captured_frame=lambda frame: None)
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    with pytest.raises(RuntimeError, match="采集未启动，无法运行推理链路。"):
        pipeline.start()

    assert runtime.running is False
    assert pipeline.running is False


def test_runtime_pipeline_requires_ready_gpu_bridge_for_nvmm_inference() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "nvmm"
    capture = SimpleNamespace(
        source=object(),
        state=SimpleNamespace(available=True),
        session=SimpleNamespace(running=True),
        config=cfg.capture,
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
        process_captured_frame=lambda frame: pytest.fail("inference must not start"),
    )
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    with pytest.raises(RuntimeError, match="capture.memory=nvmm 需要可用的 Jetson native bridge"):
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
        process_captured_frame=lambda frame: pytest.fail("inference is disabled"),
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

    def process_captured_frame(frame: CapturedFrame) -> None:
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
    assert wait_calls[:2] == [(0, 0.1), (1, 0.1)]
    status = pipeline.status()
    assert status["consumed_frames"] == 2
    assert "queue" not in status
    assert "capture_frames" not in status


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

    def process_captured_frame(frame: CapturedFrame) -> None:
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


def test_runtime_service_process_detection_batch_uses_roi_contract() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("default hardware trigger should not emit in this seam test"),
        ),
    )
    batch = DetectionBatch(
        frame_id=7,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="roi",
    )

    service.process_detection_batch(batch, width=480, height=480)

    assert service.last_frame_context is not None
    assert service.last_frame_context.frame_id == 7
    assert service.last_frame_context.width == 480
    assert service.last_frame_context.height == 480
    assert len(service.last_frame_context.detections) == 1


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
