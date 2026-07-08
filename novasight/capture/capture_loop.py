from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from novasight.contracts import FrameContext

from .pipeline_planner import PipelinePlan


@dataclass(frozen=True)
class CapturedBufferSlot:
    buffer: Any
    context: FrameContext
    dequeue_timestamp_ns: int
    owner: Any | None = field(default=None, repr=False, compare=False)


class LatestFrameBuffer:
    """Single-slot latest-frame buffer for non-copying GStreamer handoff."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._slot: CapturedBufferSlot | None = None
        self._version = 0

    def put(
        self,
        buffer: Any,
        ctx: FrameContext,
        *,
        dequeue_timestamp_ns: int | None = None,
        owner: Any | None = None,
    ) -> None:
        slot = CapturedBufferSlot(
            buffer=buffer,
            context=ctx,
            dequeue_timestamp_ns=(
                int(dequeue_timestamp_ns)
                if dequeue_timestamp_ns is not None
                else time.monotonic_ns()
            ),
            owner=owner,
        )
        with self._condition:
            self._slot = slot
            self._version += 1
            self._condition.notify_all()

    def get(self, *, after_version: int | None = None, timeout_s: float | None = None) -> tuple[Any, FrameContext] | None:
        slot = self.get_slot(after_version=after_version, timeout_s=timeout_s)
        if slot is None:
            return None
        return slot.buffer, slot.context

    def get_slot(
        self,
        *,
        after_version: int | None = None,
        timeout_s: float | None = None,
    ) -> CapturedBufferSlot | None:
        deadline = None if timeout_s is None else time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while self._slot is None or (
                after_version is not None and self._version <= int(after_version)
            ):
                if timeout_s == 0:
                    return None
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._slot

    @property
    def version(self) -> int:
        with self._condition:
            return self._version


class CaptureLoop(threading.Thread):
    def __init__(self, pipeline_plan: PipelinePlan, buffer: LatestFrameBuffer):
        super().__init__(name="novasight-capture-loop", daemon=True)
        self.plan = pipeline_plan
        self.buffer = buffer
        self.pipeline: Any | None = None
        self._loop: Any | None = None
        self._Gst: Any | None = None
        self._stop_event = threading.Event()
        self._frame_id = 0
        self._last_error = ""

    @property
    def last_error(self) -> str:
        return self._last_error

    def run(self) -> None:
        try:
            Gst, GLib = _import_gst()
            self._Gst = Gst
            if not Gst.is_initialized():
                Gst.init(None)
            self.pipeline = Gst.parse_launch(_appsink_pipeline(self.plan))
            appsink = self.pipeline.get_by_name("sink")
            if appsink is None:
                raise RuntimeError("capture loop appsink element not found")
            appsink.set_property("emit-signals", True)
            appsink.connect("new-sample", self._on_new_sample)
            bus = self.pipeline.get_bus()
            if bus is not None:
                bus.add_signal_watch()
                bus.connect("message", self._on_bus_message)
            self._loop = GLib.MainLoop()
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("capture loop pipeline failed to enter PLAYING")
            self._loop.run()
        except Exception as exc:
            self._last_error = str(exc)
        finally:
            self._stop_pipeline()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop_event.set()
        pipeline = self.pipeline
        if pipeline is not None and self._Gst is not None:
            try:
                pipeline.send_event(self._Gst.Event.new_eos())
            except Exception:
                pass
        loop = self._loop
        if loop is not None:
            try:
                loop.quit()
            except Exception:
                pass
        if self.is_alive():
            self.join(timeout=max(0.0, float(timeout_s)))

    def _on_new_sample(self, appsink: Any) -> Any:
        Gst = self._Gst
        if Gst is None:
            return None
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buffer = sample.get_buffer()
        if buffer is None:
            return Gst.FlowReturn.OK
        dequeue_ts_ns = time.monotonic_ns()
        width, height = _sample_size(sample)
        capture_ts_ns = _buffer_timestamp_ns(buffer, Gst) or dequeue_ts_ns
        self._frame_id += 1
        ctx = FrameContext(
            frame_id=self._frame_id,
            width=width,
            height=height,
            capture_ts_ns=capture_ts_ns,
        )
        self.buffer.put(
            buffer,
            ctx,
            dequeue_timestamp_ns=dequeue_ts_ns,
            owner=sample,
        )
        return Gst.FlowReturn.OK

    def _on_bus_message(self, _bus: Any, message: Any) -> None:
        Gst = self._Gst
        if Gst is None:
            return
        message_type = getattr(message, "type", None)
        if message_type == Gst.MessageType.ERROR:
            self._last_error = _message_error_text(message)
            if self._loop is not None:
                self._loop.quit()
        elif message_type == Gst.MessageType.EOS:
            if self._loop is not None:
                self._loop.quit()

    def _stop_pipeline(self) -> None:
        pipeline = self.pipeline
        Gst = self._Gst
        if pipeline is not None and Gst is not None:
            try:
                pipeline.set_state(Gst.State.NULL)
                pipeline.get_state(2 * Gst.SECOND)
            except Exception:
                pass


def _appsink_pipeline(plan: PipelinePlan) -> str:
    pipeline = plan.generate_gst_launch_string()
    sink = "appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
    return pipeline.rsplit("!", 1)[0].strip() + f" ! {sink}"


def _import_gst() -> tuple[Any, Any]:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst
    except Exception as exc:
        raise RuntimeError(f"PyGObject Gst unavailable: {exc}") from exc
    return Gst, GLib


def _sample_size(sample: Any) -> tuple[int, int]:
    caps = sample.get_caps()
    structure = caps.get_structure(0) if caps is not None and caps.get_size() > 0 else None
    if structure is None:
        return 0, 0
    ok_width, width = structure.get_int("width")
    ok_height, height = structure.get_int("height")
    return (int(width) if ok_width else 0, int(height) if ok_height else 0)


def _buffer_timestamp_ns(buffer: Any, Gst: Any) -> int | None:
    none_value = getattr(Gst, "CLOCK_TIME_NONE", None)
    for attr in ("pts", "dts"):
        value = getattr(buffer, attr, None)
        if not isinstance(value, int) or value < 0:
            continue
        if none_value is not None and value == none_value:
            continue
        return int(value)
    return None


def _message_error_text(message: Any) -> str:
    try:
        error, debug = message.parse_error()
    except Exception:
        return str(message)
    text = getattr(error, "message", str(error))
    return f"{text} ({debug})" if debug else text
