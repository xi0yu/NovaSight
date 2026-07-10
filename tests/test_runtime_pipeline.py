"""Tests for core runtime primitives and pipeline behavior."""
import threading
import time
from types import SimpleNamespace

import pytest

from novasight.capture.pipeline import build_appsink_candidates
from novasight.capture.session import CaptureSession
from novasight.capture.source import CapturedFrame, GstAppSinkFrameSource
from novasight.capture.state import CaptureProfile, CaptureRuntimeState
from novasight.config import RuntimeConfig
from novasight.contracts import Detection, DetectionBatch, FrameContext
from novasight.inference.contracts import InferenceResult
from novasight.runtime import (
    DetectionBatchMailbox,
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
    assert any("video/x-raw,format=RGBA,width=640,height=640" in item.pipeline for item in candidates)
    assert any("video/x-raw,format=RGB,width=640,height=640" in item.pipeline for item in candidates)
    assert any("video/x-raw,format=I420,width=640,height=640" in item.pipeline for item in candidates)


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


def test_detection_batch_mailbox_keeps_only_latest_generation() -> None:
    mailbox = DetectionBatchMailbox()

    for generation in range(100, 109):
        now_ns = time.monotonic_ns()
        mailbox.publish(
            DetectionBatch(
                frame_id=generation,
                generation=generation,
                capture_ts_ns=now_ns,
                inference_start_ts_ns=now_ns + 1_000,
                inference_end_ts_ns=now_ns + 2_000,
                detections=[],
                classes=[],
                coordinate_space="roi",
            )
        )

    latest = mailbox.acquire_latest(after_generation=100, timeout_s=0.0)
    status = mailbox.status()

    assert latest is not None
    assert latest.generation == 108
    assert status["pending_depth"] == 1
    assert status["max_pending_depth"] == 1
    assert status["overwritten_batches"] == 8


def test_detection_batch_mailbox_rejects_stale_generation_publish() -> None:
    mailbox = DetectionBatchMailbox()
    now_ns = time.monotonic_ns()
    mailbox.publish(
        DetectionBatch(
            frame_id=10,
            generation=10,
            capture_ts_ns=now_ns,
            inference_start_ts_ns=now_ns + 1_000,
            inference_end_ts_ns=now_ns + 2_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        )
    )
    mailbox.publish(
        DetectionBatch(
            frame_id=9,
            generation=9,
            capture_ts_ns=now_ns + 3_000,
            inference_start_ts_ns=now_ns + 4_000,
            inference_end_ts_ns=now_ns + 5_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        )
    )

    latest = mailbox.acquire_latest(after_generation=-1, timeout_s=0.0)
    status = mailbox.status()

    assert latest is not None
    assert latest.generation == 10
    assert latest.frame_id == 10
    assert status["latest_generation"] == 10
    assert status["latest_frame_id"] == 10
    assert status["published_batches"] == 1
    assert status["stale_published_batches"] == 1


def test_detection_batch_mailbox_rejects_stale_frame_id_publish() -> None:
    mailbox = DetectionBatchMailbox()
    now_ns = time.monotonic_ns()
    mailbox.publish(
        DetectionBatch(
            frame_id=20,
            generation=20,
            capture_ts_ns=now_ns,
            inference_start_ts_ns=now_ns + 1_000,
            inference_end_ts_ns=now_ns + 2_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        )
    )
    mailbox.publish(
        DetectionBatch(
            frame_id=19,
            generation=21,
            capture_ts_ns=now_ns + 3_000,
            inference_start_ts_ns=now_ns + 4_000,
            inference_end_ts_ns=now_ns + 5_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        )
    )

    latest = mailbox.acquire_latest(after_generation=-1, timeout_s=0.0)
    status = mailbox.status()

    assert latest is not None
    assert latest.generation == 20
    assert latest.frame_id == 20
    assert status["latest_generation"] == 20
    assert status["latest_frame_id"] == 20
    assert status["published_batches"] == 1
    assert status["stale_published_batches"] == 1


def test_detection_batch_mailbox_rejects_capture_timestamp_rollback() -> None:
    mailbox = DetectionBatchMailbox()
    now_ns = time.monotonic_ns()
    mailbox.publish(
        DetectionBatch(
            frame_id=20,
            generation=20,
            capture_ts_ns=now_ns,
            inference_start_ts_ns=now_ns + 1_000,
            inference_end_ts_ns=now_ns + 2_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        )
    )
    mailbox.publish(
        DetectionBatch(
            frame_id=21,
            generation=21,
            capture_ts_ns=now_ns - 1_000,
            inference_start_ts_ns=now_ns + 3_000,
            inference_end_ts_ns=now_ns + 4_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        )
    )

    latest = mailbox.acquire_latest(after_generation=-1, timeout_s=0.0)
    status = mailbox.status()

    assert latest is not None
    assert latest.generation == 20
    assert latest.frame_id == 20
    assert latest.capture_ts_ns == now_ns
    assert status["latest_generation"] == 20
    assert status["latest_frame_id"] == 20
    assert status["latest_capture_ts_ns"] == now_ns
    assert status["published_batches"] == 1
    assert status["stale_published_batches"] == 1


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
        process_captured_frame=lambda frame, *, acquired_generation=None: pytest.fail("inference must not start"),
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
        process_captured_frame=lambda frame, *, acquired_generation=None: pytest.fail("inference is disabled"),
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

    def process_captured_frame(frame: CapturedFrame, *, acquired_generation: int | None = None) -> None:
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
        wait_preview_frame=lambda **_kwargs: pytest.fail("broker path must not poll preview frames"),
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

    def process_captured_frame(frame: CapturedFrame, *, acquired_generation: int | None = None) -> RuntimeFrameResult:
        processed.append(frame.frame_id)
        processed_fresh.set()
        return RuntimeFrameResult(control_intents=[], execution_results=[], observation_updated=False)

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


def test_runtime_service_drops_detection_when_newer_generation_arrives_during_inference() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "system"
    cfg.control.latency_reject_if_age_exceeds_ms = 55.0

    class Broker:
        def status(self) -> dict[str, int]:
            return {"published_generation": 2}

    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("expired DetectionBatch must not execute"),
        ),
        capture=SimpleNamespace(latest_frame_broker=Broker(), state=CaptureRuntimeState()),
        inference=SimpleNamespace(
            status=lambda: {"available": True, "loaded": True, "selected": "tensorrt"},
            infer=lambda _frame: InferenceResult(
                available=True,
                detections=[],
                classes=["target"],
                debug={"timings": {}, "preprocess": {"model_width": 2, "model_height": 2}},
            )
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
    state = service.state()

    assert result.observation_updated is False
    assert service.last_frame_context is None
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["stale_rejected"] is True
    assert service.last_inference_status["latest_generation"] == 2
    assert "latest generation" in service.last_inference_status["reason"]
    assert state.statistics["stale_drop_count"] == 1
    assert "control_observe_fps" in state.statistics
    assert "inference_ms" in state.statistics
    assert "postprocess_ms" in state.statistics


def test_runtime_service_drops_detection_when_newer_frame_id_arrives_during_inference() -> None:
    cfg = RuntimeConfig()
    cfg.capture.memory = "system"
    cfg.control.latency_reject_if_age_exceeds_ms = 55.0

    class Broker:
        def status(self) -> dict[str, int]:
            return {"published_generation": 1, "published_frame_id": 2}

    service = RuntimeService(
        cfg,
        models=SimpleNamespace(get_active_deployment=lambda: None),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("expired DetectionBatch must not execute"),
        ),
        capture=SimpleNamespace(latest_frame_broker=Broker(), state=CaptureRuntimeState()),
        inference=SimpleNamespace(
            status=lambda: {"available": True, "loaded": True, "selected": "tensorrt"},
            infer=lambda _frame: InferenceResult(
                available=True,
                detections=[],
                classes=["target"],
                debug={"timings": {}, "preprocess": {"model_width": 2, "model_height": 2}},
            )
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
    assert service.last_inference_status["latest_generation"] == 2
    assert service.last_inference_status["latest_frame_id"] == 2
    assert ("latest frame_id" in service.last_inference_status["reason"]
            or "latest generation" in service.last_inference_status["reason"])


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

    def process_captured_frame(frame: CapturedFrame, *, acquired_generation: int | None = None) -> None:
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
        process_captured_frame=lambda _frame, *, acquired_generation=None: pytest.fail("unavailable capture must not process inference"),
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

    def process_captured_frame(frame: CapturedFrame, *, acquired_generation: int | None = None) -> None:
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


@pytest.mark.skip(reason="legacy full DeepStream detection_source path removed")
def test_runtime_pipeline_consumes_detection_batches_from_deepstream_source() -> None:
    batch = DetectionBatch(
        frame_id=42,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="roi",
    )
    processed: list[tuple[int, int, int, int | None, int | None, int | None, int | None]] = []
    processed_one = threading.Event()

    class DetectionSource:
        roi_width = 480
        roi_height = 480
        pipeline_config = SimpleNamespace(
            capture_width=1920,
            capture_height=1080,
            roi_left=720,
            roi_top=300,
        )

        def __init__(self) -> None:
            self.running = False
            self.started = False
            self.stopped = False

        def start(self) -> None:
            self.running = True
            self.started = True

        def stop(self) -> None:
            self.running = False
            self.stopped = True

        def latest_result(self, *, after_frame_id: int | None = None):
            if after_frame_id is None or batch.frame_id > after_frame_id:
                return batch
            return None

        def status(self) -> dict[str, object]:
            return {
                "selected": "deepstream",
                "available": True,
                "running": self.running,
                "last_error": "",
                "published_batches": 1,
                "last_frame_id": batch.frame_id,
                "pipeline": "large gst pipeline string",
            }

    def process_detection_batch(
        detection_batch: DetectionBatch,
        *,
        width: int,
        height: int,
        source_width: int | None = None,
        source_height: int | None = None,
        roi_offset_x: int | None = None,
        roi_offset_y: int | None = None,
    ) -> None:
        processed.append((
            detection_batch.frame_id,
            width,
            height,
            source_width,
            source_height,
            roi_offset_x,
            roi_offset_y,
        ))
        processed_one.set()

    source = DetectionSource()
    runtime = SimpleNamespace(
        running=False,
        config=RuntimeConfig(),
        process_detection_batch=process_detection_batch,
        process_control_tick=lambda: None,
    )
    pipeline = RuntimePipeline(
        capture=SimpleNamespace(),
        runtime=runtime,
        detection_source=source,
    )

    pipeline.start()
    assert processed_one.wait(1.0)
    pipeline.stop()

    assert source.started is True
    assert source.stopped is True
    assert processed == [(42, 480, 480, 1920, 1080, 720, 300)]
    status = pipeline.status()
    assert status["consumed_detection_batches"] == 1
    assert status["last_frame_id"] == 42
    assert status["skipped_frames"] == 0
    assert status["detection_batch_fps"] >= 0.0
    assert status["detection_source"]["selected"] == "deepstream"
    assert status["detection_source"]["published_batches"] == 1
    assert "pipeline" not in status["detection_source"]


@pytest.mark.skip(reason="legacy full DeepStream detection_source path removed")
def test_runtime_pipeline_resets_stats_between_detection_source_restarts() -> None:
    batches = [
        DetectionBatch(
            frame_id=0,
            capture_ts_ns=1_000_000_000,
            inference_start_ts_ns=1_000_001_000,
            inference_end_ts_ns=1_000_002_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        ),
        DetectionBatch(
            frame_id=1,
            capture_ts_ns=1_000_100_000,
            inference_start_ts_ns=1_000_101_000,
            inference_end_ts_ns=1_000_102_000,
            detections=[],
            classes=[],
            coordinate_space="roi",
        ),
    ]
    processed: list[int] = []
    processed_event = threading.Event()

    class DetectionSource:
        roi_width = 480
        roi_height = 480

        def __init__(self) -> None:
            self.running = False
            self.index = 0
            self.delivered_this_run = False

        def start(self) -> None:
            self.running = True
            self.delivered_this_run = False

        def stop(self) -> None:
            self.running = False

        def latest_result(self, *, after_frame_id: int | None = None):
            if self.delivered_this_run or self.index >= len(batches):
                return None
            batch = batches[self.index]
            if after_frame_id is None or batch.frame_id > after_frame_id:
                self.index += 1
                self.delivered_this_run = True
                return batch
            return None

        def status(self) -> dict[str, object]:
            return {"available": True, "running": self.running}

    def process_detection_batch(detection_batch: DetectionBatch, **_kwargs) -> None:
        processed.append(detection_batch.frame_id)
        processed_event.set()

    source = DetectionSource()
    runtime = SimpleNamespace(
        running=False,
        config=RuntimeConfig(),
        process_detection_batch=process_detection_batch,
        process_control_tick=lambda: None,
    )
    pipeline = RuntimePipeline(
        capture=SimpleNamespace(),
        runtime=runtime,
        detection_source=source,
    )

    pipeline.start()
    assert processed_event.wait(1.0)
    pipeline.stop()
    first_status = pipeline.status()
    assert first_status["consumed_detection_batches"] == 1
    assert first_status["last_frame_id"] == 0

    processed_event.clear()
    pipeline.start()
    assert processed_event.wait(1.0)
    pipeline.stop()
    second_status = pipeline.status()

    assert processed == [0, 1]
    assert second_status["consumed_detection_batches"] == 1
    assert second_status["processed_frames"] == 1
    assert second_status["last_frame_id"] == 1


@pytest.mark.skip(reason="legacy full DeepStream detection_source path removed")
def test_runtime_pipeline_does_not_count_rejected_batch_as_control_observation() -> None:
    batch = DetectionBatch(
        frame_id=5,
        capture_ts_ns=1_000_000_000,
        inference_start_ts_ns=1_000_001_000,
        inference_end_ts_ns=1_000_002_000,
        detections=[],
        classes=[],
        coordinate_space="roi",
    )
    processed_event = threading.Event()

    class DetectionSource:
        roi_width = 480
        roi_height = 480

        def __init__(self) -> None:
            self.running = False
            self.delivered = False

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.running = False

        def latest_result(self, *, after_frame_id: int | None = None):
            if self.delivered or (after_frame_id is not None and batch.frame_id <= after_frame_id):
                return None
            self.delivered = True
            return batch

        def status(self) -> dict[str, object]:
            return {"available": True, "running": self.running}

    def process_detection_batch(_detection_batch: DetectionBatch, **_kwargs) -> RuntimeFrameResult:
        processed_event.set()
        return RuntimeFrameResult(
            control_intents=[],
            execution_results=[],
            observation_updated=False,
        )

    source = DetectionSource()
    runtime = SimpleNamespace(
        running=False,
        config=RuntimeConfig(),
        process_detection_batch=process_detection_batch,
        process_control_tick=lambda: None,
    )
    pipeline = RuntimePipeline(
        capture=SimpleNamespace(),
        runtime=runtime,
        detection_source=source,
    )

    pipeline.start()
    assert processed_event.wait(1.0)
    pipeline.stop()
    status = pipeline.status()

    assert status["consumed_detection_batches"] == 1
    assert status["processed_frames"] == 1
    assert status["window_control_observations"] == 0
    assert status["detection_batch_fps"] >= 0.0
    assert status["control_observation_fps"] == 0.0


@pytest.mark.skip(reason="legacy full DeepStream detection_source path removed")
def test_runtime_pipeline_stops_deepstream_source_when_it_becomes_unavailable() -> None:
    stopped = threading.Event()

    class DetectionSource:
        roi_width = 480
        roi_height = 480

        def __init__(self) -> None:
            self.running = False
            self.stop_count = 0

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.running = False
            self.stop_count += 1
            stopped.set()

        def latest_result(self, *, after_frame_id: int | None = None):
            del after_frame_id
            return None

        def status(self) -> dict[str, object]:
            if self.running:
                self.running = False
                return {
                    "selected": "deepstream",
                    "available": True,
                    "running": False,
                    "last_error": "nvinfer pipeline error",
                }
            return {
                "selected": "deepstream",
                "available": True,
                "running": False,
                "last_error": "nvinfer pipeline error",
            }

    source = DetectionSource()
    runtime = SimpleNamespace(
        running=False,
        config=RuntimeConfig(),
        process_detection_batch=lambda _batch, **_kwargs: pytest.fail("stopped source must not produce batches"),
        process_control_tick=lambda: None,
    )
    pipeline = RuntimePipeline(
        capture=SimpleNamespace(),
        runtime=runtime,
        detection_source=source,
    )

    pipeline.start()
    try:
        assert stopped.wait(1.0)
        assert runtime.running is False
        assert "nvinfer pipeline error" in (pipeline.stats.last_error or "")
        assert source.stop_count >= 1
    finally:
        pipeline.stop()


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
    cfg.control.latency_reject_if_age_exceeds_ms = 10.0
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


@pytest.mark.skip(reason="legacy DeepStream timestamp-source rejection removed")
def test_runtime_service_rejects_deepstream_untrusted_timestamp_source() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("untrusted timestamp batch must not execute"),
        ),
    )
    now_ns = time.monotonic_ns()
    batch = DetectionBatch(
        frame_id=18,
        capture_ts_ns=now_ns,
        inference_start_ts_ns=now_ns + 1_000,
        inference_end_ts_ns=now_ns + 2_000,
        detections=[Detection(cls=0, score=0.9, x1=10, y1=20, x2=40, y2=80)],
        classes=["0"],
        coordinate_space="roi",
        metadata={"source": "deepstream", "timestamp_source": "first_probe_offset_pts"},
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
    assert service.last_control["runtime_reset_reason"] == "DETECTION_BATCH_TIMESTAMP_SOURCE_INVALID"
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["mapped_detections"] == 0
    assert service.last_inference_status["timestamp_source_invalid"] is True
    assert service.last_inference_status["timestamp_source"] == "first_probe_offset_pts"
    assert "timestamp_source must be gst_clock_base_time_pts" in service.last_inference_status["reason"]
    assert service.last_inference_status["detection_batch_metadata"]["source"] == "deepstream"
    assert service.last_pipeline_timings["engine_ms"] == pytest.approx(0.001)
    assert service.last_pipeline_timings["control_ms"] == 0.0


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


@pytest.mark.skip(reason="legacy DeepStream TensorMeta contract removed")
def test_runtime_service_rejects_deepstream_missing_tensor_meta_batch() -> None:
    cfg = RuntimeConfig()
    service = RuntimeService(
        cfg,
        models=SimpleNamespace(),
        executors=SimpleNamespace(
            selected="noop",
            status=lambda: {},
            update_runtime_config=lambda _cfg: None,
            execute=lambda _intent: pytest.fail("missing tensor meta batch must not execute"),
        ),
    )
    service.last_frame_context = FrameContext(
        frame_id=6,
        width=480,
        height=480,
        detections=[Detection(cls=0, score=0.9, x1=200, y1=200, x2=260, y2=280)],
        classes=["0"],
        capture_ts_ns=1_000_000,
    )
    batch = DetectionBatch(
        frame_id=7,
        capture_ts_ns=time.monotonic_ns(),
        inference_start_ts_ns=time.monotonic_ns() + 1_000,
        inference_end_ts_ns=time.monotonic_ns() + 2_000,
        detections=[],
        classes=["0"],
        coordinate_space="roi",
        metadata={
            "source": "deepstream",
            "timestamp_source": "gst_clock_base_time_pts",
            "empty_reason": "missing_tensor_meta",
        },
    )

    result = service.process_detection_batch(batch, width=480, height=480)

    assert result.control_intents == []
    assert service.last_frame_context is None
    assert service.last_control is not None
    assert service.last_control["control_allowed"] is False
    assert service.last_control["runtime_reset_reason"] == "DETECTION_BATCH_MISSING_TENSOR_META"
    assert service.last_inference_status["available"] is False
    assert service.last_inference_status["missing_tensor_meta"] is True
    assert service.last_inference_status["mapped_detections"] == 0
    assert service.last_inference_status["detection_batch_metadata"]["empty_reason"] == (
        "missing_tensor_meta"
    )


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
    assert service.last_pipeline_timings["engine_ms"] == pytest.approx(0.001)
    assert service.last_pipeline_timings["control_ms"] == 0.0
    assert service.last_control is not None
    assert service.last_control["control_allowed"] is False
    assert service.last_control["runtime_reset_reason"] == "DETECTION_BATCH_COORDINATE_SPACE_INVALID"
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
