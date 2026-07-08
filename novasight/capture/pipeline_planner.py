from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

from novasight.roi import center_roi_region

from .device_probe import DeviceCapability, PixelFormatCaps, ResolutionCaps


class InfeasibleConfiguration(ValueError):
    pass


@dataclass(frozen=True)
class RoiConfig:
    size: int | None = None
    offset_x: int = 0
    offset_y: int = 0


@dataclass(frozen=True)
class RoiRect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class PipelinePlan:
    source_type: str
    device: str
    capture_format: str
    capture_width: int
    capture_height: int
    capture_fps: Fraction
    decode_backend: str
    memory_domain: str
    crop_rect: RoiRect | None
    inference_width: int
    inference_height: int
    model_profile: str
    sink_type: str
    latency_policy: str
    gst_template: str

    def generate_gst_launch_string(self) -> str:
        fps = _fraction_caps(self.capture_fps)
        queue = "queue max-size-buffers=1 leaky=downstream"
        sink = "fakesink sync=false" if self.sink_type == "fakesink" else self.sink_type
        if self.capture_format in {"MJPG", "MJPEG"}:
            caps = (
                f"image/jpeg,width={self.capture_width},height={self.capture_height},"
                f"framerate={fps}"
            )
            decode = f"{queue} ! jpegparse ! nvv4l2decoder mjpeg=1 ! "
        else:
            gst_format = "YUY2" if self.capture_format == "YUYV" else self.capture_format
            caps = (
                f"video/x-raw,format={gst_format},width={self.capture_width},"
                f"height={self.capture_height},framerate={fps}"
            )
            decode = ""

        convert = "nvvidconv"
        if self.crop_rect is not None:
            convert = (
                f"{convert} left={self.crop_rect.left} right={self.crop_rect.right} "
                f"top={self.crop_rect.top} bottom={self.crop_rect.bottom}"
            )
        output_caps = (
            "video/x-raw(memory:NVMM),format=NV12,"
            f"width={self.inference_width},height={self.inference_height}"
        )
        return (
            f"v4l2src device={_gst_property_value(self.device)} do-timestamp=true ! "
            f"{caps} ! {decode}{queue} ! {convert} ! {output_caps} ! {sink}"
        )


PlanValidator = Callable[[PipelinePlan], bool]


class PipelinePlanner:
    @staticmethod
    def plan(
        capabilities: list[DeviceCapability],
        device: str,
        target_res: tuple[int, int],
        target_fps: Fraction,
        model_input_size: tuple[int, int],
        model_hash: str,
        roi_config: RoiConfig,
        *,
        validator: PlanValidator | None = None,
        cache_dir: str | Path | None = "/var/cache/novasight/pipeline_plans",
    ) -> PipelinePlan:
        key = _cache_key(device, target_res, target_fps, model_input_size, model_hash, roi_config)
        cache_path = Path(cache_dir) / f"{key}.json" if cache_dir is not None else None
        if cache_path is not None:
            cached = _load_plan(cache_path)
            if cached is not None:
                return cached

        candidates = _candidate_plans(
            capabilities=capabilities,
            device=device,
            target_res=target_res,
            target_fps=target_fps,
            model_input_size=model_input_size,
            model_hash=model_hash,
            roi_config=roi_config,
        )
        if not candidates:
            raise InfeasibleConfiguration(
                f"no capture format supports {device} {target_res[0]}x{target_res[1]} "
                f"at {_fraction_caps(target_fps)}"
            )

        for plan in candidates:
            if validator is not None and not validator(plan):
                continue
            if cache_path is not None:
                _store_plan(cache_path, plan)
            return plan

        raise InfeasibleConfiguration("all matching capture candidates failed validation")


def _candidate_plans(
    *,
    capabilities: list[DeviceCapability],
    device: str,
    target_res: tuple[int, int],
    target_fps: Fraction,
    model_input_size: tuple[int, int],
    model_hash: str,
    roi_config: RoiConfig,
) -> list[PipelinePlan]:
    matched_device = next((cap for cap in capabilities if cap.device_path == device), None)
    if matched_device is None:
        return []

    plans: list[tuple[int, PipelinePlan]] = []
    for pixel_format in matched_device.pixel_formats:
        for resolution in pixel_format.resolutions:
            if not _resolution_matches(resolution, target_res, target_fps):
                continue
            plan = _build_plan(
                device=device,
                pixel_format=pixel_format,
                resolution=resolution,
                target_fps=target_fps,
                model_input_size=model_input_size,
                model_hash=model_hash,
                roi_config=roi_config,
            )
            plans.append((_score_format(pixel_format.format), plan))
    return [plan for _, plan in sorted(plans, key=lambda item: item[0], reverse=True)]


def _build_plan(
    *,
    device: str,
    pixel_format: PixelFormatCaps,
    resolution: ResolutionCaps,
    target_fps: Fraction,
    model_input_size: tuple[int, int],
    model_hash: str,
    roi_config: RoiConfig,
) -> PipelinePlan:
    roi_rect: RoiRect | None = None
    if roi_config.size is not None:
        left, top, size = center_roi_region(
            source_width=resolution.width,
            source_height=resolution.height,
            requested_size=roi_config.size,
            offset_x=roi_config.offset_x,
            offset_y=roi_config.offset_y,
        )
        roi_rect = RoiRect(left=left, top=top, width=size, height=size)

    normalized_format = pixel_format.format.upper()
    decode_backend = "nvv4l2decoder" if normalized_format in {"MJPG", "MJPEG"} else "none"
    return PipelinePlan(
        source_type="v4l2",
        device=device,
        capture_format=normalized_format,
        capture_width=resolution.width,
        capture_height=resolution.height,
        capture_fps=target_fps,
        decode_backend=decode_backend,
        memory_domain="NVMM",
        crop_rect=roi_rect,
        inference_width=int(model_input_size[0]),
        inference_height=int(model_input_size[1]),
        model_profile=model_hash,
        sink_type="fakesink",
        latency_policy="drop_old",
        gst_template="v4l2src_nvmm_single_frame_latest",
    )


def _resolution_matches(
    resolution: ResolutionCaps,
    target_res: tuple[int, int],
    target_fps: Fraction,
) -> bool:
    return (
        resolution.width == int(target_res[0])
        and resolution.height == int(target_res[1])
        and target_fps in resolution.fps_list
    )


def _score_format(pixel_format: str) -> int:
    return {"NV12": 100, "MJPG": 90, "MJPEG": 90, "YUYV": 50, "YUY2": 50}.get(
        pixel_format.upper(),
        10,
    )


def _cache_key(
    device: str,
    target_res: tuple[int, int],
    target_fps: Fraction,
    model_input_size: tuple[int, int],
    model_hash: str,
    roi_config: RoiConfig,
) -> str:
    payload = {
        "device": device,
        "target_res": list(target_res),
        "target_fps": _fraction_caps(target_fps),
        "model_input_size": list(model_input_size),
        "model_hash": model_hash,
        "roi_config": asdict(roi_config),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_plan(path: Path) -> PipelinePlan | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _plan_from_json(payload)
    except Exception:
        return None


def _store_plan(path: Path, plan: PipelinePlan) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_plan_to_json(plan), sort_keys=True, indent=2), encoding="utf-8")
    except OSError:
        pass


def _plan_to_json(plan: PipelinePlan) -> dict[str, object]:
    payload = asdict(plan)
    payload["capture_fps"] = _fraction_caps(plan.capture_fps)
    return payload


def _plan_from_json(payload: dict[str, object]) -> PipelinePlan:
    crop_payload = payload.get("crop_rect")
    crop_rect = RoiRect(**crop_payload) if isinstance(crop_payload, dict) else None
    return PipelinePlan(
        source_type=str(payload["source_type"]),
        device=str(payload["device"]),
        capture_format=str(payload["capture_format"]),
        capture_width=int(payload["capture_width"]),
        capture_height=int(payload["capture_height"]),
        capture_fps=_parse_fraction(str(payload["capture_fps"])),
        decode_backend=str(payload["decode_backend"]),
        memory_domain=str(payload["memory_domain"]),
        crop_rect=crop_rect,
        inference_width=int(payload["inference_width"]),
        inference_height=int(payload["inference_height"]),
        model_profile=str(payload["model_profile"]),
        sink_type=str(payload["sink_type"]),
        latency_policy=str(payload["latency_policy"]),
        gst_template=str(payload["gst_template"]),
    )


def _fraction_caps(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _parse_fraction(value: str) -> Fraction:
    numerator, denominator = value.split("/", 1)
    return Fraction(int(numerator), int(denominator))


def _gst_property_value(value: object) -> str:
    text = str(value)
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._:-")
    if text and all(ch in safe_chars for ch in text):
        return text
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
