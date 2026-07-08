from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction


@dataclass(frozen=True)
class ResolutionCaps:
    width: int
    height: int
    fps_list: list[Fraction] = field(default_factory=list)


@dataclass(frozen=True)
class PixelFormatCaps:
    format: str
    description: str
    resolutions: list[ResolutionCaps] = field(default_factory=list)


@dataclass(frozen=True)
class DeviceCapability:
    device_path: str
    driver: str
    card_name: str
    bus_info: str
    pixel_formats: list[PixelFormatCaps] = field(default_factory=list)


_FORMAT_RE = re.compile(r"^\s*\[\d+\]:\s*'([^']+)'\s*\((.*)\)\s*$")
_SIZE_RE = re.compile(r"^\s*Size:\s*Discrete\s+(\d+)x(\d+)\s*$")
_FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")
_DEVICE_PATH_RE = re.compile(r"^\s*(/dev/video\d+)\s*$")


class DeviceProbe:
    @staticmethod
    def probe_all() -> list[DeviceCapability]:
        try:
            output = DeviceProbe._run_command(["v4l2-ctl", "--list-devices"])
        except Exception:
            return []

        capabilities: list[DeviceCapability] = []
        for device_path in _parse_device_paths(output):
            try:
                capabilities.append(DeviceProbe.probe_device(device_path))
            except Exception:
                continue
        return capabilities

    @staticmethod
    def probe_device(device_path: str) -> DeviceCapability:
        all_output = DeviceProbe._run_command(["v4l2-ctl", "-d", device_path, "--all"])
        formats_output = DeviceProbe._run_command(
            ["v4l2-ctl", "-d", device_path, "--list-formats-ext"]
        )
        return DeviceProbe.parse_device_outputs(
            device_path,
            all_output=all_output,
            formats_output=formats_output,
        )

    @staticmethod
    def parse_device_outputs(
        device_path: str,
        *,
        all_output: str,
        formats_output: str,
    ) -> DeviceCapability:
        metadata = _parse_device_metadata(all_output)
        return DeviceCapability(
            device_path=device_path,
            driver=metadata.get("driver", ""),
            card_name=metadata.get("card_name", ""),
            bus_info=metadata.get("bus_info", ""),
            pixel_formats=_parse_formats_ext(formats_output),
        )

    @staticmethod
    def _run_command(command: list[str]) -> str:
        if shutil.which(command[0]) is None:
            raise RuntimeError(f"{command[0]} unavailable")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(detail or f"{command[0]} failed with {result.returncode}")
        return result.stdout


def _parse_device_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = _DEVICE_PATH_RE.match(line)
        if not match:
            continue
        path = match.group(1)
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def _parse_device_metadata(text: str) -> dict[str, str]:
    fields = {
        "Driver name": "driver",
        "Card type": "card_name",
        "Bus info": "bus_info",
    }
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip()
        mapped = fields.get(normalized_key)
        if mapped:
            metadata[mapped] = value.strip()
    return metadata


def _parse_formats_ext(text: str) -> list[PixelFormatCaps]:
    formats: list[PixelFormatCaps] = []
    current_format: str | None = None
    current_description = ""
    current_resolutions: list[ResolutionCaps] = []
    current_width: int | None = None
    current_height: int | None = None
    current_fps: list[Fraction] = []

    def flush_resolution() -> None:
        nonlocal current_width, current_height, current_fps
        if current_width is not None and current_height is not None:
            current_resolutions.append(
                ResolutionCaps(
                    width=current_width,
                    height=current_height,
                    fps_list=sorted(set(current_fps), reverse=True),
                )
            )
        current_width = None
        current_height = None
        current_fps = []

    def flush_format() -> None:
        nonlocal current_format, current_description, current_resolutions
        flush_resolution()
        if current_format is not None:
            formats.append(
                PixelFormatCaps(
                    format=current_format.upper(),
                    description=current_description,
                    resolutions=current_resolutions,
                )
            )
        current_format = None
        current_description = ""
        current_resolutions = []

    for line in text.splitlines():
        if match := _FORMAT_RE.match(line):
            flush_format()
            current_format = match.group(1).upper()
            current_description = match.group(2).strip()
            continue
        if match := _SIZE_RE.match(line):
            flush_resolution()
            current_width = int(match.group(1))
            current_height = int(match.group(2))
            continue
        if match := _FPS_RE.search(line):
            current_fps.append(_fps_fraction(match.group(1)))

    flush_format()
    return formats


def _fps_fraction(value: str) -> Fraction:
    fraction = Fraction(value).limit_denominator(1001)
    if fraction.denominator == 1:
        return fraction
    rounded = round(float(fraction))
    if abs(float(fraction) - rounded) < 0.001:
        return Fraction(rounded, 1)
    return fraction
