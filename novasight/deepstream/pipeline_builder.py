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
    io_mode: int = 2
    batched_push_timeout_us: int = 0

    @property
    def roi_right(self) -> int:
        return int(self.roi_left) + int(self.roi_size)

    @property
    def roi_bottom(self) -> int:
        return int(self.roi_top) + int(self.roi_size)


def build_deepstream_pipeline(config: DeepStreamPipelineConfig) -> str:
    _validate_config(config)
    config_path = Path(config.nvinfer_config_path).resolve(strict=False)
    queue = "queue max-size-buffers=1 leaky=downstream"
    return " ".join(
        [
            f"v4l2src device={config.device} io-mode={config.io_mode} do-timestamp=true",
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
            f"batched-push-timeout={config.batched_push_timeout_us}",
            "!",
            f"nvinfer name=primary-infer config-file-path={config_path}",
            "!",
            "fakesink sync=false",
        ]
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
    if config.roi_left < 0 or config.roi_top < 0:
        raise ValueError("ROI left/top must be >= 0")
    if config.roi_right > config.capture_width or config.roi_bottom > config.capture_height:
        raise ValueError("ROI crop must stay inside capture frame")
