"""Tests for the core runtime primitives: latest-frame queue, config
snapshots, the fail-fast crash handler, and logging setup.
"""
import pytest

from novasight.config import RuntimeConfig
from novasight.runtime import (
    FailFastHandler,
    LatestFrameQueue,
    RuntimeConfigStore,
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
