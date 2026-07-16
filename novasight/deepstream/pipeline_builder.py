from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


LATEST_ONLY_QUEUE = (
    "queue max-size-buffers=1 max-size-bytes=0 "
    "max-size-time=0 leaky=downstream"
)


@dataclass(frozen=True, slots=True)
class DeepStreamPipelineConfig:
    device: str
    capture_width: int
    capture_height: int
    fps: int
    roi_left: int
    roi_top: int
    roi_width: int
    roi_height: int
    model_width: int
    model_height: int
    nvinfer_config_path: Path
    pixel_format: str = "MJPG"
    io_mode: int = 2
    batched_push_timeout_us: int = 0
    preview_enabled: bool = True
    preview_fps: int = 30
    crosshair_enabled: bool = False
    crosshair_size: int = 96
    crosshair_fps: int = 10

    @property
    def roi_right(self) -> int:
        return int(self.roi_left) + int(self.roi_width)

    @property
    def roi_bottom(self) -> int:
        return int(self.roi_top) + int(self.roi_height)


def build_deepstream_pipeline(config: DeepStreamPipelineConfig) -> str:
    _validate_config(config)
    nvinfer_config = _gst_property_value(
        Path(config.nvinfer_config_path).expanduser().resolve(strict=False)
    )
    elements = [
            (
                f"v4l2src name=capture-source device={_gst_property_value(config.device)} "
                f"io-mode={int(config.io_mode)} do-timestamp=true"
            ),
            "!",
            (
                f"image/jpeg,width={int(config.capture_width)},"
                f"height={int(config.capture_height)},framerate={int(config.fps)}/1"
            ),
            "!",
            LATEST_ONLY_QUEUE,
            "!",
            "jpegparse",
            "!",
            "nvv4l2decoder mjpeg=1",
            "!",
            "video/x-raw(memory:NVMM),format=I420",
            "!",
            LATEST_ONLY_QUEUE,
            "!",
            (
                "nvvidconv "
                f"left={int(config.roi_left)} right={int(config.roi_right)} "
                f"top={int(config.roi_top)} bottom={int(config.roi_bottom)}"
            ),
            "!",
            (
                "video/x-raw(memory:NVMM),format=NV12,"
                f"width={int(config.roi_width)},height={int(config.roi_height)},"
                "pixel-aspect-ratio=1/1"
            ),
            "!",
            "tee name=novasight_roi_split",
            "novasight_roi_split.",
            "!",
            LATEST_ONLY_QUEUE,
        ]
    if (
        int(config.roi_width) != int(config.model_width)
        or int(config.roi_height) != int(config.model_height)
    ):
        elements.extend(
            [
                "!",
                "nvvidconv",
                "!",
                (
                    "video/x-raw(memory:NVMM),format=NV12,"
                    f"width={int(config.model_width)},height={int(config.model_height)},"
                    "pixel-aspect-ratio=1/1"
                ),
                "!",
                LATEST_ONLY_QUEUE,
            ]
        )
    elements.extend(["!", "mux.sink_0"])
    if config.preview_enabled:
        elements.extend(
            [
                "novasight_roi_split.",
                "!",
                LATEST_ONLY_QUEUE,
                "!",
                "valve name=preview-valve drop=false",
                "!",
                f"videorate drop-only=true max-rate={int(config.preview_fps)}",
                "!",
                (
                    "video/x-raw(memory:NVMM),format=NV12,"
                    f"width={int(config.roi_width)},height={int(config.roi_height)},"
                    f"framerate={int(config.preview_fps)}/1"
                ),
                "!",
                "nvjpegenc name=preview-encoder",
                "!",
                "appsink name=preview_sink emit-signals=false max-buffers=1 drop=true sync=false",
            ]
        )
    if config.crosshair_enabled:
        crosshair_left = (int(config.roi_width) - int(config.crosshair_size)) // 2
        crosshair_top = (int(config.roi_height) - int(config.crosshair_size)) // 2
        crosshair_right = crosshair_left + int(config.crosshair_size)
        crosshair_bottom = crosshair_top + int(config.crosshair_size)
        elements.extend(
            [
                "novasight_roi_split.",
                "!",
                LATEST_ONLY_QUEUE,
                "!",
                f"videorate drop-only=true max-rate={int(config.crosshair_fps)}",
                "!",
                (
                    "nvvidconv name=crosshair-crop "
                    f"left={crosshair_left} right={crosshair_right} "
                    f"top={crosshair_top} bottom={crosshair_bottom}"
                ),
                "!",
                (
                    "video/x-raw(memory:NVMM),format=NV12,"
                    f"width={int(config.crosshair_size)},"
                    f"height={int(config.crosshair_size)},"
                    f"framerate={int(config.crosshair_fps)}/1"
                ),
                "!",
                "nvjpegenc name=crosshair-encoder",
                "!",
                (
                    "appsink name=crosshair_sink emit-signals=false "
                    "max-buffers=1 drop=true sync=false"
                ),
            ]
        )
    elements.extend(
        [
            "nvstreammux name=mux",
            "batch-size=1",
            "live-source=1",
            f"width={int(config.model_width)}",
            f"height={int(config.model_height)}",
            "sync-inputs=0",
            f"batched-push-timeout={int(config.batched_push_timeout_us)}",
            "!",
            f"nvinfer name=primary-infer config-file-path={nvinfer_config} batch-size=1",
            "!",
            "fakesink name=deepstream-sink sync=false async=false qos=false",
        ]
    )
    return " ".join(elements)


def _validate_config(config: DeepStreamPipelineConfig) -> None:
    positive = {
        "capture_width": config.capture_width,
        "capture_height": config.capture_height,
        "fps": config.fps,
        "roi_width": config.roi_width,
        "roi_height": config.roi_height,
        "model_width": config.model_width,
        "model_height": config.model_height,
    }
    for name, value in positive.items():
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    pixel_format = str(config.pixel_format or "").strip().upper()
    if pixel_format not in {"", "MJPG", "MJPEG"}:
        raise ValueError("deepstream_nvinfer currently requires MJPEG capture")
    if int(config.io_mode) < 0:
        raise ValueError("io_mode must be >= 0")
    if int(config.batched_push_timeout_us) < 0:
        raise ValueError("batched_push_timeout_us must be >= 0")
    if int(config.preview_fps) <= 0:
        raise ValueError("preview_fps must be positive")
    if int(config.crosshair_fps) <= 0:
        raise ValueError("crosshair_fps must be positive")
    if (
        int(config.crosshair_size) < 32
        or int(config.crosshair_size) > min(int(config.roi_width), int(config.roi_height))
        or int(config.crosshair_size) % 2 != 0
    ):
        raise ValueError("crosshair_size must be even and inside the ROI")
    if int(config.roi_left) < 0 or int(config.roi_top) < 0:
        raise ValueError("ROI left/top must be >= 0")
    if config.roi_right > int(config.capture_width) or config.roi_bottom > int(
        config.capture_height
    ):
        raise ValueError("ROI crop must stay inside capture frame")


def _gst_property_value(value: object) -> str:
    text = str(value)
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._:-")
    if text and all(char in safe for char in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


__all__ = [
    "DeepStreamPipelineConfig",
    "LATEST_ONLY_QUEUE",
    "build_deepstream_pipeline",
]
