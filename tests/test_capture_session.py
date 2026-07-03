import threading
import time
from collections.abc import Callable

import pytest

from novasight.capture.session import CaptureSession
from novasight.capture.source import CapturedFrame
from novasight.capture.state import CaptureProfile


class CountingSource:
    backend_label = "gst-appsink:test"

    def __init__(self) -> None:
        self.count = 0
        self.closed = False

    def read(self):
        self.count += 1
        return CapturedFrame(
            frame_id=self.count,
            width=320,
            height=320,
            pixel_format="NV12",
            ts_ns=time.monotonic_ns(),
            capture_wait_ms=0.1,
            image=None,
        )

    def close(self) -> None:
        self.closed = True


class BlockingSource:
    backend_label = "gst-appsink:blocking"

    def __init__(self) -> None:
        self.closed = False
        self.read_entered = threading.Event()
        self.release_read = threading.Event()

    def read(self):
        self.read_entered.set()
        self.release_read.wait()
        return None

    def close(self) -> None:
        self.closed = True


class ExplodingSource:
    backend_label = "gst-appsink:exploding"

    def __init__(self) -> None:
        self.closed = False

    def read(self):
        raise RuntimeError("camera disconnected")

    def close(self) -> None:
        self.closed = True


class ControlledSource:
    backend_label = "gst-appsink:controlled"

    def __init__(self) -> None:
        self.closed = False
        self._frames: list[CapturedFrame] = []
        self._condition = threading.Condition()

    def emit(self, frame_id: int) -> None:
        with self._condition:
            self._frames.append(
                CapturedFrame(
                    frame_id=frame_id,
                    width=320,
                    height=320,
                    pixel_format="NV12",
                    ts_ns=time.monotonic_ns(),
                    capture_wait_ms=0.1,
                    image=None,
                )
            )
            self._condition.notify_all()

    def read(self):
        with self._condition:
            if not self._frames:
                self._condition.wait(0.01)
            if not self._frames:
                return None
            return self._frames.pop(0)

    def close(self) -> None:
        self.closed = True
        with self._condition:
            self._condition.notify_all()


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 0.2) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _assert_event_not_set(event: threading.Event, timeout_s: float = 0.05) -> None:
    assert event.wait(timeout_s) is False


def _profile() -> CaptureProfile:
    return CaptureProfile(
        device="/dev/video0",
        pixel_format="MJPG",
        width=1920,
        height=1080,
        fps=120,
        preference="auto_high_fps",
        selection_reason="test",
    )


def test_capture_session_starts_thread_and_publishes_latest_frame() -> None:
    source = CountingSource()
    session = CaptureSession(source_factory=lambda profile: source)

    state = session.start(_profile())
    frame = session.latest_frame(after_frame_id=0, timeout_s=0.2)
    stopped = session.stop("test complete")

    assert state.available is True
    assert frame is not None
    assert frame.frame_id >= 1
    assert session.running is False
    assert stopped.available is False
    assert source.closed is True


def test_capture_session_updates_statistics_for_published_frames() -> None:
    source = CountingSource()
    session = CaptureSession(source_factory=lambda profile: source)

    session.start(_profile())
    frame = session.latest_frame(after_frame_id=0, timeout_s=0.2)
    state = session.stop("test complete")

    assert frame is not None
    assert state.statistics.capture_counter >= 1
    assert state.statistics.capture_fps >= 0
    assert state.statistics.dropped_counter == 0
    assert state.statistics.skipped_counter == 0


def test_capture_session_uses_sliding_window_fps_not_instant_period() -> None:
    source = CountingSource()
    session = CaptureSession(source_factory=lambda profile: source)
    stop_event = threading.Event()
    session.state.profile = _profile()
    session._source = source

    for index, ts_ns in enumerate(
        [
            1_000_000_000,
            1_200_000_000,
            1_500_000_000,
            1_900_000_000,
            2_000_000_000,
        ],
        start=1,
    ):
        session._publish_frame(
            CapturedFrame(
                frame_id=index,
                width=320,
                height=320,
                pixel_format="NV12",
                ts_ns=ts_ns,
                capture_wait_ms=0.1,
                image=None,
            ),
            source=source,
            stop_event=stop_event,
        )

    assert session.state.frame_period_ms == 100.0
    assert session.state.statistics.capture_counter == 5
    assert session.state.statistics.capture_fps == pytest.approx(4.0)
    assert session.state.fps_capture == pytest.approx(4.0)


def test_capture_session_reconfigure_closes_previous_source() -> None:
    sources: list[CountingSource] = []

    def factory(profile):
        source = CountingSource()
        sources.append(source)
        return source

    session = CaptureSession(source_factory=factory)
    session.start(_profile())
    session.reconfigure(_profile())
    session.stop("done")

    assert len(sources) == 2
    assert sources[0].closed is True
    assert sources[1].closed is True


def test_capture_session_failed_reconfigure_closes_old_source_and_reports_error() -> None:
    first_source = BlockingSource()
    factory_calls = 0

    def factory(profile):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return first_source
        raise RuntimeError("new pipeline failed")

    session = CaptureSession(
        source_factory=factory,
        empty_read_sleep_s=0,
    )
    session.start(_profile())
    assert first_source.read_entered.wait(0.2)
    first_source.release_read.set()

    with pytest.raises(RuntimeError, match="new pipeline failed"):
        session.reconfigure(_profile())

    assert first_source.closed is True
    assert session.source is None
    assert session.running is False
    assert session.state.available is False
    assert session.state.last_error == "new pipeline failed"


def test_capture_session_stop_waits_for_read_loop_before_closing_source() -> None:
    source = BlockingSource()
    session = CaptureSession(
        source_factory=lambda profile: source,
        empty_read_sleep_s=0,
    )
    session.start(_profile())
    assert source.read_entered.wait(0.2)

    stopped: list[object] = []
    stopper = threading.Thread(
        target=lambda: (stopped.append(session.stop("done")), stopped_event.set()),
        daemon=True,
    )
    stopped_event = threading.Event()
    stopper.start()

    _assert_event_not_set(stopped_event)
    assert stopped == []
    assert stopper.is_alive()
    assert source.closed is False
    assert session.running is True

    source.release_read.set()
    stopper.join(0.2)

    assert stopped_event.is_set()
    assert stopped
    assert source.closed is True
    assert session.running is False


def test_capture_session_stop_timeout_keeps_wedged_source_owned() -> None:
    source = BlockingSource()
    session = CaptureSession(
        source_factory=lambda profile: source,
        empty_read_sleep_s=0,
        stop_timeout_s=0.02,
    )
    session.start(_profile())
    assert source.read_entered.wait(0.2)

    stopped: list[object] = []
    stopped_event = threading.Event()
    stopper = threading.Thread(
        target=lambda: (stopped.append(session.stop("done")), stopped_event.set()),
        daemon=True,
    )
    stopper.start()

    assert stopped_event.wait(0.2)
    assert stopped
    assert source.closed is False
    assert session.source is source
    assert session.running is True
    assert "did not stop" in str(session.state.last_error)

    source.release_read.set()
    stopper.join(0.2)
    stopped_again = session.stop("cleanup")

    assert source.closed is True
    assert stopped_again.available is False
    assert session.running is False


def test_capture_session_reconfigure_waits_for_previous_read_loop_to_exit() -> None:
    first_source = BlockingSource()
    sources: list[object] = []

    def factory(profile):
        if not sources:
            sources.append(first_source)
            return first_source
        source = CountingSource()
        sources.append(source)
        return source

    session = CaptureSession(
        source_factory=factory,
        empty_read_sleep_s=0,
    )
    session.start(_profile())
    assert first_source.read_entered.wait(0.2)

    reconfigured: list[object] = []
    reconfigurer = threading.Thread(
        target=lambda: (
            reconfigured.append(session.reconfigure(_profile())),
            reconfigured_event.set(),
        ),
        daemon=True,
    )
    reconfigured_event = threading.Event()
    reconfigurer.start()

    _assert_event_not_set(reconfigured_event)
    assert reconfigured == []
    assert reconfigurer.is_alive()
    assert sources == [first_source]
    assert first_source.closed is False
    assert session.source is first_source

    first_source.release_read.set()
    reconfigurer.join(0.2)

    assert reconfigured_event.is_set()
    assert reconfigured
    assert len(sources) == 2
    assert first_source.closed is True
    assert session.source is sources[1]

    session.stop("done")


def test_capture_session_reconfigure_timeout_does_not_open_replacement() -> None:
    first_source = BlockingSource()
    sources: list[BlockingSource | CountingSource] = []

    def factory(profile):
        if not sources:
            sources.append(first_source)
            return first_source
        source = CountingSource()
        sources.append(source)
        return source

    session = CaptureSession(
        source_factory=factory,
        empty_read_sleep_s=0,
        stop_timeout_s=0.02,
    )
    session.start(_profile())
    assert first_source.read_entered.wait(0.2)

    reconfigured: list[object] = []
    reconfigured_event = threading.Event()

    def reconfigure() -> None:
        try:
            reconfigured.append(session.reconfigure(_profile()))
        except RuntimeError as exc:
            reconfigured.append(exc)
        finally:
            reconfigured_event.set()

    reconfigurer = threading.Thread(target=reconfigure, daemon=True)
    reconfigurer.start()

    assert reconfigured_event.wait(0.2)
    assert len(reconfigured) == 1
    assert "did not stop" in str(reconfigured[0])
    assert len(sources) == 1
    assert first_source.closed is False
    assert session.source is first_source
    assert session.running is True
    assert "did not stop" in str(session.state.last_error)

    first_source.release_read.set()
    reconfigurer.join(0.2)
    session.stop("cleanup")


def test_capture_session_serializes_concurrent_start_calls() -> None:
    sources: list[BlockingSource] = []
    factory_can_return = threading.Event()

    def factory(profile):
        source = BlockingSource()
        sources.append(source)
        factory_can_return.wait()
        return source

    session = CaptureSession(
        source_factory=factory,
        empty_read_sleep_s=0,
    )
    second_started = threading.Event()
    first = threading.Thread(
        target=lambda: session.start(_profile()),
        daemon=True,
    )
    second = threading.Thread(
        target=lambda: (second_started.set(), session.start(_profile())),
        daemon=True,
    )

    first.start()
    assert _wait_until(lambda: len(sources) == 1)
    second.start()
    assert second_started.wait(0.2)

    _assert_event_not_set(factory_can_return)
    assert len(sources) == 1

    factory_can_return.set()
    first.join(0.2)
    assert sources[0].read_entered.wait(0.2)
    sources[0].release_read.set()
    second.join(0.2)

    assert len(sources) == 2
    assert sources[0].closed is True
    assert session.source is sources[1]

    sources[1].release_read.set()
    session.stop("done")


def test_capture_session_stop_without_reason_preserves_read_error() -> None:
    source = ExplodingSource()
    session = CaptureSession(
        source_factory=lambda profile: source,
        empty_read_sleep_s=0,
    )
    session.start(_profile())
    assert _wait_until(lambda: session.state.last_error is not None)

    read_error = session.state.last_error
    stopped = session.stop()

    assert source.closed is True
    assert read_error == "capture read failed: camera disconnected"
    assert stopped.last_error == read_error


def test_capture_session_latest_frame_returns_only_newer_frames() -> None:
    source = ControlledSource()
    session = CaptureSession(
        source_factory=lambda profile: source,
        empty_read_sleep_s=0,
    )
    session.start(_profile())
    source.emit(1)
    first = session.latest_frame(0, 0.2)
    assert first is not None

    stale = session.latest_frame(after_frame_id=first.frame_id, timeout_s=0)
    source.emit(2)
    second = session.latest_frame(after_frame_id=first.frame_id, timeout_s=0.2)
    session.stop("done")

    assert stale is None
    assert second is not None
    assert second.frame_id == 2
