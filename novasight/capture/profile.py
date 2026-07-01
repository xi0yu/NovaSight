from __future__ import annotations

from .state import CaptureCapability, CapturePreference, CaptureProfile


_VALID_PREFERENCES = {
    "auto_high_fps",
    "auto_low_latency",
    "auto_balanced",
    "manual",
}


def _expanded(caps: list[CaptureCapability]) -> list[tuple[str, int, int, int]]:
    return [
        (cap.pixel_format.upper(), cap.width, cap.height, fps)
        for cap in caps
        for fps in cap.fps_list
    ]


def _format_rank_high_fps(pixel_format: str, fps: int) -> int:
    if pixel_format == "MJPG" and fps >= 120:
        return 0
    if pixel_format == "NV12" and fps >= 60:
        return 1
    if pixel_format == "YUYV":
        return 2
    if pixel_format == "MJPG":
        return 3
    return 4


def _format_rank_low_latency(pixel_format: str) -> int:
    return {"NV12": 0, "YUYV": 1, "MJPG": 2}.get(pixel_format, 3)


def select_capture_profile(
    device: str,
    capabilities: list[CaptureCapability],
    preference: CapturePreference = "auto_high_fps",
    *,
    pixel_format: str | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> CaptureProfile:
    if preference not in _VALID_PREFERENCES:
        raise ValueError(f"unknown capture preference: {preference}")

    choices = _expanded(capabilities)
    if not choices:
        raise ValueError(f"no capture capabilities available for {device}")

    if preference == "manual":
        required = (str(pixel_format).upper(), width, height, fps)
        for choice in choices:
            if choice == required:
                return CaptureProfile(
                    device=device,
                    pixel_format=choice[0],
                    width=choice[1],
                    height=choice[2],
                    fps=choice[3],
                    preference=preference,
                    selection_reason="manual profile matched device capabilities",
                )
        raise ValueError(f"unsupported capture profile for {device}: {required}")

    if preference == "auto_low_latency":
        choice = sorted(
            choices,
            key=lambda item: (
                _format_rank_low_latency(item[0]),
                -item[3],
                -(item[1] * item[2]),
            ),
        )[0]
        reason = "auto_low_latency selected raw-friendly profile"
    elif preference == "auto_balanced":
        choice = sorted(
            choices,
            key=lambda item: (
                abs((item[1] * item[2]) - (1920 * 1080)),
                -item[3],
                _format_rank_high_fps(item[0], item[3]),
            ),
        )[0]
        reason = "auto_balanced selected profile closest to 1080p with high fps"
    else:
        choice = sorted(
            choices,
            key=lambda item: (
                -item[3],
                _format_rank_high_fps(item[0], item[3]),
                -(item[1] * item[2]),
            ),
        )[0]
        reason = "auto_high_fps selected highest fps profile"

    return CaptureProfile(
        device=device,
        pixel_format=choice[0],
        width=choice[1],
        height=choice[2],
        fps=choice[3],
        preference=preference,
        selection_reason=reason,
    )
