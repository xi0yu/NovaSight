from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable

from .state import CaptureCapabilities, CaptureCapability


_FORMAT_RE = re.compile(r"^\s*\[\d+\]:\s*'([A-Za-z0-9]+)'")
_SIZE_RE = re.compile(r"^\s*Size:\s*Discrete\s+(\d+)x(\d+)")
_FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")


def run_v4l2_ctl(device: str) -> str | None:
    if shutil.which("v4l2-ctl") is None:
        return None
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_v4l2_formats(text: str) -> list[CaptureCapability]:
    capabilities: list[CaptureCapability] = []
    current_format: str | None = None
    current_width: int | None = None
    current_height: int | None = None
    current_fps: list[int] = []

    def flush() -> None:
        nonlocal current_width, current_height, current_fps
        if current_format and current_width and current_height and current_fps:
            capabilities.append(
                CaptureCapability(
                    pixel_format=current_format.upper(),
                    width=current_width,
                    height=current_height,
                    fps_list=sorted(set(current_fps), reverse=True),
                )
            )
        current_width = None
        current_height = None
        current_fps = []

    for line in text.splitlines():
        if match := _FORMAT_RE.match(line):
            flush()
            current_format = match.group(1).upper()
            continue
        if match := _SIZE_RE.match(line):
            flush()
            current_width = int(match.group(1))
            current_height = int(match.group(2))
            continue
        if match := _FPS_RE.search(line):
            current_fps.append(int(round(float(match.group(1)))))

    flush()
    return capabilities


def query_capabilities(
    device: str = "/dev/video0",
    runner: Callable[[str], str | None] = run_v4l2_ctl,
) -> CaptureCapabilities:
    text = runner(device)
    if text is None:
        return CaptureCapabilities(
            available=False,
            device=device,
            reason="v4l2-ctl unavailable, device missing, or command failed",
        )
    capabilities = parse_v4l2_formats(text)
    return CaptureCapabilities(
        available=bool(capabilities),
        device=device,
        capabilities=capabilities,
        reason="" if capabilities else "no discrete V4L2 formats found",
    )
