from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from novasight.capture.pipeline_planner import PipelinePlan
from novasight.model_registry.manifest import ModelManifest


PadProbeCallback = Callable[[Any, Any, Any], Any]


class DeepStreamPipeline:
    def __init__(
        self,
        plan: PipelinePlan,
        model_manifest: ModelManifest,
        *,
        nvinfer_config_path: str | Path | None = None,
        batched_push_timeout_us: int = 0,
        tracker_config_path: str | Path | None = None,
    ) -> None:
        self.plan = plan
        self.model_manifest = model_manifest
        self.nvinfer_config_path = Path(nvinfer_config_path or "deepstream.ini")
        self.batched_push_timeout_us = max(0, int(batched_push_timeout_us))
        self.tracker_config_path = Path(tracker_config_path) if tracker_config_path else None
        self.pipeline_description = self._build_pipeline_description()
        self.pipeline: Any | None = None
        self._Gst: Any | None = None

    def start(self) -> None:
        Gst = self._gst()
        self.stop()
        pipeline = Gst.parse_launch(self.pipeline_description)
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("DeepStream pipeline failed to enter PLAYING")
        self.pipeline = pipeline

    def stop(self) -> None:
        if self.pipeline is None:
            return
        Gst = self._gst()
        try:
            self.pipeline.send_event(Gst.Event.new_eos())
        except Exception:
            pass
        self.pipeline.set_state(Gst.State.NULL)
        try:
            self.pipeline.get_state(2 * Gst.SECOND)
        except Exception:
            pass
        self.pipeline = None

    def attach_probe(
        self,
        pad: Any,
        callback: PadProbeCallback,
        user_data: Any = None,
    ) -> int:
        Gst = self._gst()
        target_pad = self._resolve_pad(pad)
        return int(target_pad.add_probe(Gst.PadProbeType.BUFFER, callback, user_data))

    def _build_pipeline_description(self) -> str:
        queue = "queue max-size-buffers=1 leaky=downstream"
        source = self._source_segment(queue)
        crop = self._crop_segment()
        mux_width = int(self.plan.inference_width)
        mux_height = int(self.plan.inference_height)
        segments = [
            source,
            queue,
            crop,
            (
                "video/x-raw(memory:NVMM),format=NV12,"
                f"width={mux_width},height={mux_height},pixel-aspect-ratio=1/1"
            ),
            queue,
            "mux.sink_0",
            "nvstreammux name=mux batch-size=1 live-source=1 nvbuf-memory-type=3 "
            f"width={mux_width} height={mux_height} "
            f"batched-push-timeout={self.batched_push_timeout_us}",
            (
                "nvinfer name=primary-infer "
                f"config-file-path={_gst_property_value(self.nvinfer_config_path.resolve(strict=False))}"
            ),
        ]
        if self.tracker_config_path is not None:
            segments.append(
                "nvtracker name=tracker "
                f"ll-config-file={_gst_property_value(self.tracker_config_path.resolve(strict=False))}"
            )
        segments.append("fakesink name=deepstream_sink sync=false")
        return " ! ".join(segments)

    def _source_segment(self, queue: str) -> str:
        fps = _fraction_text(self.plan.capture_fps)
        fmt = self.plan.capture_format.upper()
        if fmt in {"MJPG", "MJPEG"}:
            caps = (
                f"image/jpeg,width={self.plan.capture_width},"
                f"height={self.plan.capture_height},framerate={fps}"
            )
            decode = f"{queue} ! jpegparse ! nvv4l2decoder mjpeg=1"
        else:
            gst_format = "YUY2" if fmt == "YUYV" else fmt
            caps = (
                f"video/x-raw,format={gst_format},width={self.plan.capture_width},"
                f"height={self.plan.capture_height},framerate={fps}"
            )
            decode = ""
        parts = [
            f"v4l2src device={_gst_property_value(self.plan.device)} do-timestamp=true",
            caps,
        ]
        if decode:
            parts.append(decode)
        return " ! ".join(parts)

    def _crop_segment(self) -> str:
        crop = self.plan.crop_rect
        if crop is None:
            return "nvvidconv"
        return (
            "nvvidconv "
            f"left={crop.left} right={crop.right} "
            f"top={crop.top} bottom={crop.bottom}"
        )

    def _resolve_pad(self, pad: Any) -> Any:
        if hasattr(pad, "add_probe"):
            return pad
        if self.pipeline is None:
            raise RuntimeError("DeepStream pipeline is not started")
        text = str(pad)
        element_name, _, pad_name = text.partition(":")
        if not element_name or not pad_name:
            raise ValueError("pad must be a Gst.Pad or 'element:pad-name'")
        element = self.pipeline.get_by_name(element_name)
        if element is None:
            raise ValueError(f"pipeline element not found: {element_name}")
        target_pad = element.get_static_pad(pad_name)
        if target_pad is None:
            raise ValueError(f"element {element_name!r} has no pad {pad_name!r}")
        return target_pad

    def _gst(self) -> Any:
        if self._Gst is not None:
            return self._Gst
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError(f"PyGObject Gst unavailable: {exc}") from exc
        if not Gst.is_initialized():
            Gst.init(None)
        self._Gst = Gst
        return Gst


def _fraction_text(value: Any) -> str:
    return f"{value.numerator}/{value.denominator}"


def _gst_property_value(value: object) -> str:
    text = str(value)
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._:-")
    if text and all(ch in safe_chars for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
