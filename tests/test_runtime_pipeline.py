"""Tests for the core runtime primitives: latest-frame queue, config
snapshots, the fail-fast crash handler, and logging setup.
"""
import threading
from types import SimpleNamespace

import pytest

from novasight.capture.source import CapturedFrame
from novasight.config import RuntimeConfig
from novasight.runtime import (
    FailFastHandler,
    LatestFrameQueue,
    RuntimeConfigStore,
    RuntimePipeline,
    configure_logging,
)


def test_latest_frame_queue_overwrites_old_frame() -> None:
    queue = LatestFrameQueue[int]()

    queue.put(1)
    queue.put(2)

    assert queue.get(timeout=0) == 2
    assert queue.status()["capacity"] == 1
    assert queue.status()["dropped"] == 1


def test_runtime_config_store_returns_isolated_snapshots() -> None:
    cfg = RuntimeConfig()
    store = RuntimeConfigStore(cfg)

    snap = store.snapshot()
    snap.capture.device = "/dev/changed"

    assert store.snapshot().capture.device == "/dev/video0"
    assert store.status()["version"] == 0


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

    with pytest.raises(RuntimeError, match="采集未启动，无法启动推理控制。"):
        pipeline.start()

    assert runtime.running is False
    assert pipeline.running is False


def test_runtime_pipeline_consumes_latest_frames_without_read_frame() -> None:
    frames = [
        CapturedFrame(
            frame_id=1,
            width=2,
            height=2,
            pixel_format="BGR",
            ts_ns=1_000_000_000,
            capture_wait_ms=1.0,
            image=None,
        ),
        CapturedFrame(
            frame_id=2,
            width=2,
            height=2,
            pixel_format="BGR",
            ts_ns=1_000_001_000,
            capture_wait_ms=1.0,
            image=None,
        ),
    ]
    wait_calls: list[tuple[int | None, float]] = []
    read_frame_calls = 0
    processed: list[int] = []
    processed_two = threading.Event()

    def wait_preview_frame(*, after_frame_id: int | None = None, timeout_s: float = 0.0):
        wait_calls.append((after_frame_id, timeout_s))
        for frame in frames:
            if after_frame_id is None or frame.frame_id > after_frame_id:
                return frame
        return None

    def read_frame():
        nonlocal read_frame_calls
        read_frame_calls += 1
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
        read_frame=read_frame,
    )
    runtime = SimpleNamespace(running=False, process_captured_frame=process_captured_frame)
    pipeline = RuntimePipeline(capture=capture, runtime=runtime)

    pipeline.start()
    assert processed_two.wait(1.0)
    pipeline.stop()

    assert processed == [1, 2]
    assert read_frame_calls == 0
    assert wait_calls[:2] == [(0, 0.1), (1, 0.1)]
