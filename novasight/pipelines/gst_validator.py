from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class GstValidationResult:
    pipeline: str
    available: bool
    ok: bool
    checked_buffers: int = 0
    nvmm_buffers: int = 0
    memory_types: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    reason: str = ""


def is_nvmm_buffer(buffer: Any) -> bool:
    text = " ".join(buffer_memory_types(buffer)).lower()
    return "nvmm" in text or "gstnvbuf" in text or "nvbuf" in text


def buffer_memory_types(buffer: Any) -> list[str]:
    try:
        count = int(buffer.n_memory())
    except Exception:
        return []
    memory_types: list[str] = []
    for index in range(max(0, count)):
        try:
            memory = buffer.peek_memory(index)
        except Exception:
            continue
        detected: list[str] = []
        for name in ("NVMM", "GstNvBufMemory", "DMABuf", "DmaBuf", "CUDA", "GLMemory"):
            try:
                if memory.is_type(name):
                    detected.append(name)
            except Exception:
                continue
        memory_types.append(",".join(detected) if detected else type(memory).__name__)
    return memory_types


def build_capture_nvmm_pipeline(
    *,
    device: str,
    pixel_format: str,
    width: int,
    height: int,
    fps: int | Fraction,
    sink_name: str = "validator_sink",
) -> str:
    framerate = _fraction_text(Fraction(fps))
    normalized = str(pixel_format or "").upper()
    queue = "queue max-size-buffers=1 leaky=downstream"
    if normalized in {"MJPG", "MJPEG"}:
        caps = f"image/jpeg,width={width},height={height},framerate={framerate}"
        decode = f"{queue} ! jpegparse ! nvv4l2decoder mjpeg=1 ! "
    else:
        gst_format = "YUY2" if normalized == "YUYV" else normalized
        caps = f"video/x-raw,format={gst_format},width={width},height={height},framerate={framerate}"
        decode = ""
    return (
        f"v4l2src device={_gst_property_value(device)} do-timestamp=true ! "
        f"{caps} ! {decode}{queue} ! nvvidconv ! "
        "video/x-raw(memory:NVMM),format=NV12 ! "
        f"fakesink name={sink_name} sync=false"
    )


def validate_nvmm_pipeline(
    pipeline_description: str,
    *,
    timeout_s: float = 5.0,
    sink_name: str = "validator_sink",
) -> GstValidationResult:
    try:
        Gst, GLib = _import_gst()
    except RuntimeError as exc:
        return GstValidationResult(
            pipeline=pipeline_description,
            available=False,
            ok=False,
            reason=str(exc),
        )

    if not Gst.is_initialized():
        Gst.init(None)

    state = {
        "checked_buffers": 0,
        "nvmm_buffers": 0,
        "memory_types": [],
        "errors": [],
    }
    pipeline = None
    loop = None
    timeout_id = None
    try:
        pipeline = Gst.parse_launch(pipeline_description)
        sink = pipeline.get_by_name(sink_name) or pipeline.get_by_name("sink")
        if sink is None:
            raise RuntimeError(f"pipeline missing fakesink/appsink named {sink_name!r} or 'sink'")
        sink_pad = sink.get_static_pad("sink")
        if sink_pad is None:
            raise RuntimeError("validation sink has no sink pad")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, _probe_buffer, (state, Gst))
        bus = pipeline.get_bus()
        if bus is not None:
            bus.add_signal_watch()
        loop = GLib.MainLoop()
        if bus is not None:
            bus.connect("message", _on_bus_message, (state, loop, Gst))
        timeout_id = GLib.timeout_add(int(max(0.1, timeout_s) * 1000), _quit_loop, loop)
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("pipeline failed to enter PLAYING")
        loop.run()
    except Exception as exc:
        state["errors"].append(str(exc))
    finally:
        if timeout_id is not None and loop is not None:
            try:
                GLib.source_remove(timeout_id)
            except Exception:
                pass
        if pipeline is not None:
            try:
                pipeline.set_state(Gst.State.NULL)
                pipeline.get_state(2 * Gst.SECOND)
            except Exception:
                pass

    checked = int(state["checked_buffers"])
    nvmm = int(state["nvmm_buffers"])
    errors = [str(item) for item in state["errors"]]
    return GstValidationResult(
        pipeline=pipeline_description,
        available=True,
        ok=checked > 0 and checked == nvmm and not errors,
        checked_buffers=checked,
        nvmm_buffers=nvmm,
        memory_types=sorted(set(str(item) for item in state["memory_types"])),
        errors=errors,
        reason="" if checked else "no buffers reached validation probe",
    )


def _probe_buffer(_pad: Any, info: Any, user_data: tuple[dict[str, Any], Any]) -> Any:
    state, Gst = user_data
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    memory_types = buffer_memory_types(buffer)
    state["checked_buffers"] += 1
    state["memory_types"].extend(memory_types)
    if is_nvmm_buffer(buffer):
        state["nvmm_buffers"] += 1
    return Gst.PadProbeReturn.OK


def _on_bus_message(_bus: Any, message: Any, user_data: tuple[dict[str, Any], Any, Any]) -> None:
    state, loop, Gst = user_data
    message_type = getattr(message, "type", None)
    if message_type == Gst.MessageType.ERROR:
        state["errors"].append(_message_error_text(message))
        loop.quit()
    elif message_type == Gst.MessageType.EOS:
        loop.quit()


def _quit_loop(loop: Any) -> bool:
    loop.quit()
    return False


def _import_gst() -> tuple[Any, Any]:
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst
    except Exception as exc:
        raise RuntimeError(f"PyGObject Gst unavailable: {exc}") from exc
    return Gst, GLib


def _message_error_text(message: Any) -> str:
    try:
        error, debug = message.parse_error()
    except Exception:
        return str(message)
    text = getattr(error, "message", str(error))
    return f"{text} ({debug})" if debug else text


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _gst_property_value(value: object) -> str:
    text = str(value)
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._:-")
    if text and all(ch in safe_chars for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
