from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeepStreamPipelineConfig:
    device: str
    capture_width: int
    capture_height: int
    fps: int
    roi_left: int
    roi_top: int
    roi_size: int
    model_width: int
    model_height: int
    nvinfer_config_path: Path
    pixel_format: str = "MJPG"
    io_mode: int = 2
    batched_push_timeout_us: int = 0
    tracker_config_path: Path | None = None

    @property
    def roi_right(self) -> int:
        return int(self.roi_left) + int(self.roi_size)

    @property
    def roi_bottom(self) -> int:
        return int(self.roi_top) + int(self.roi_size)


def build_deepstream_pipeline(config: DeepStreamPipelineConfig) -> str:
    _validate_config(config)
    config_path = Path(config.nvinfer_config_path).resolve(strict=False)
    tracker_config_path = (
        Path(config.tracker_config_path).resolve(strict=False)
        if config.tracker_config_path is not None
        else None
    )
    queue = "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream"
    inference_segments = [
        f"nvinfer name=primary-infer config-file-path={_gst_property_value(config_path)}",
    ]
    if tracker_config_path is not None:
        inference_segments.extend(
            [
                "!",
                f"nvtracker name=tracker ll-config-file={_gst_property_value(tracker_config_path)}",
            ]
        )
    return " ".join(
        [
            f"v4l2src device={_gst_property_value(config.device)} io-mode={config.io_mode} do-timestamp=true",
            "!",
            (
                f"image/jpeg,width={config.capture_width},height={config.capture_height},"
                f"framerate={config.fps}/1"
            ),
            "!",
            queue,
            "!",
            "jpegparse",
            "!",
            "nvv4l2decoder mjpeg=1",
            "!",
            "video/x-raw(memory:NVMM),format=I420",
            "!",
            queue,
            "!",
            (
                "nvvidconv "
                f"left={config.roi_left} right={config.roi_right} "
                f"top={config.roi_top} bottom={config.roi_bottom}"
            ),
            "!",
            (
                "video/x-raw(memory:NVMM),format=NV12,"
                f"width={config.model_width},height={config.model_height},"
                "pixel-aspect-ratio=1/1"
            ),
            "!",
            queue,
            "!",
            "mux.sink_0",
            "nvstreammux name=mux",
            "batch-size=1",
            f"width={config.model_width}",
            f"height={config.model_height}",
            "live-source=1",
            "sync-inputs=0",
            f"batched-push-timeout={config.batched_push_timeout_us}",
            "!",
            *inference_segments,
            "!",
            "fakesink sync=false",
        ]
    )


def validate_deepstream_capture_pixel_format(pixel_format: str) -> None:
    normalized = str(pixel_format or "").strip().upper()
    if normalized and normalized not in {"MJPG", "MJPEG"}:
        raise ValueError(
            "DeepStream backend currently expects MJPEG capture "
            f"(capture.pixel_format={normalized})"
        )


def _validate_config(config: DeepStreamPipelineConfig) -> None:
    positive_fields = {
        "capture_width": config.capture_width,
        "capture_height": config.capture_height,
        "fps": config.fps,
        "roi_size": config.roi_size,
        "model_width": config.model_width,
        "model_height": config.model_height,
    }
    for field, value in positive_fields.items():
        if int(value) <= 0:
            raise ValueError(f"{field} must be positive")
    validate_deepstream_capture_pixel_format(config.pixel_format)
    if int(config.io_mode) < 0:
        raise ValueError("io_mode must be >= 0")
    if int(config.batched_push_timeout_us) < 0:
        raise ValueError("batched_push_timeout_us must be >= 0")
    if config.roi_left < 0 or config.roi_top < 0:
        raise ValueError("ROI left/top must be >= 0")
    if config.roi_right > config.capture_width or config.roi_bottom > config.capture_height:
        raise ValueError("ROI crop must stay inside capture frame")


def _gst_property_value(value: object) -> str:
    text = str(value)
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._:-")
    if text and all(ch in safe_chars for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
