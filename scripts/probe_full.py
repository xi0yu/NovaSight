#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from novasight.capture.device_probe import DeviceCapability, DeviceProbe, ResolutionCaps


_FPS_RE = re.compile(r"(?:average|current):\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe NovaSight capture hardware and write Phase 0 baseline JSON files."
    )
    parser.add_argument("--output-dir", default=".", help="Directory for JSON reports.")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per gst test.")
    parser.add_argument(
        "--resolution",
        action="append",
        default=[],
        help="Target resolution as WIDTHxHEIGHT. May be repeated. Defaults to 1080p and 1440p.",
    )
    parser.add_argument(
        "--formats",
        default="MJPG,NV12,YUYV",
        help="Comma-separated capture formats to benchmark.",
    )
    parser.add_argument(
        "--run-gst-tests",
        action="store_true",
        help="Run gst-launch stability tests. Without this, only capabilities are probed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capabilities = DeviceProbe.probe_all()
    device_payload = {
        "generated_at": _now_iso(),
        "devices": [_device_to_json(capability) for capability in capabilities],
        "summary": _capability_summary(capabilities),
    }
    (output_dir / "device_capabilities.json").write_text(
        json.dumps(device_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    baseline_payload = {
        "generated_at": _now_iso(),
        "duration_s": float(args.duration),
        "gst_tests_enabled": bool(args.run_gst_tests),
        "results": [],
    }
    if args.run_gst_tests:
        target_resolutions = _parse_resolutions(args.resolution)
        target_formats = [item.strip().upper() for item in args.formats.split(",") if item.strip()]
        baseline_payload["results"] = [
            _run_format_baseline(
                capability=capability,
                resolution=resolution,
                pixel_format=pixel_format,
                duration_s=args.duration,
            )
            for capability in capabilities
            for resolution in target_resolutions
            for pixel_format in target_formats
        ]
    else:
        baseline_payload["skip_reason"] = "pass --run-gst-tests to execute gst-launch baselines"

    (output_dir / "baseline_fps.json").write_text(
        json.dumps(baseline_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'device_capabilities.json'}")
    print(f"wrote {output_dir / 'baseline_fps.json'}")
    return 0


def _device_to_json(capability: DeviceCapability) -> dict[str, Any]:
    payload = asdict(capability)
    for pixel_format in payload["pixel_formats"]:
        for resolution in pixel_format["resolutions"]:
            resolution["fps_list"] = [
                _fraction_to_json(Fraction(fps)) for fps in resolution["fps_list"]
            ]
    return payload


def _capability_summary(capabilities: list[DeviceCapability]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for capability in capabilities:
        for pixel_format in capability.pixel_formats:
            for resolution in pixel_format.resolutions:
                max_fps = max(resolution.fps_list) if resolution.fps_list else None
                rows.append(
                    {
                        "device_path": capability.device_path,
                        "card_name": capability.card_name,
                        "format": pixel_format.format,
                        "width": resolution.width,
                        "height": resolution.height,
                        "max_fps": _fraction_to_json(max_fps) if max_fps else None,
                    }
                )
    return rows


def _run_format_baseline(
    *,
    capability: DeviceCapability,
    resolution: tuple[int, int],
    pixel_format: str,
    duration_s: float,
) -> dict[str, Any]:
    matching_resolution = _find_resolution(capability, resolution, pixel_format)
    result: dict[str, Any] = {
        "device_path": capability.device_path,
        "card_name": capability.card_name,
        "format": pixel_format,
        "width": resolution[0],
        "height": resolution[1],
        "duration_s": float(duration_s),
        "supported": matching_resolution is not None,
        "command": None,
        "returncode": None,
        "stable": False,
        "actual_fps": None,
        "stderr_tail": "",
        "tegrastats_tail": "",
    }
    if matching_resolution is None:
        result["skip_reason"] = "format/resolution not advertised by v4l2-ctl"
        return result
    if shutil.which("gst-launch-1.0") is None:
        result["skip_reason"] = "gst-launch-1.0 unavailable"
        return result

    target_fps = max(matching_resolution.fps_list) if matching_resolution.fps_list else Fraction(30, 1)
    command = _gst_command(
        device=capability.device_path,
        pixel_format=pixel_format,
        width=resolution[0],
        height=resolution[1],
        fps=target_fps,
        duration_s=duration_s,
    )
    result["command"] = " ".join(command)
    tegrastats = _start_tegrastats()
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(2.0, duration_s + 5.0),
            check=False,
        )
        elapsed_s = time.monotonic() - started
        output = "\n".join([completed.stdout, completed.stderr])
        actual_fps = _parse_reported_fps(output)
        result.update(
            {
                "returncode": completed.returncode,
                "elapsed_s": elapsed_s,
                "stable": completed.returncode == 0,
                "actual_fps": actual_fps,
                "target_fps": _fraction_to_json(target_fps),
                "stderr_tail": _tail(completed.stderr),
            }
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "returncode": None,
                "elapsed_s": time.monotonic() - started,
                "stable": False,
                "target_fps": _fraction_to_json(target_fps),
                "stderr_tail": _tail(str(exc)),
                "error": "gst-launch timed out",
            }
        )
    finally:
        result["tegrastats_tail"] = _stop_tegrastats(tegrastats)
    return result


def _find_resolution(
    capability: DeviceCapability,
    resolution: tuple[int, int],
    pixel_format: str,
) -> ResolutionCaps | None:
    normalized = "MJPG" if pixel_format == "MJPEG" else pixel_format
    for candidate_format in capability.pixel_formats:
        if candidate_format.format.upper() != normalized:
            continue
        for candidate_resolution in candidate_format.resolutions:
            if (
                candidate_resolution.width == resolution[0]
                and candidate_resolution.height == resolution[1]
            ):
                return candidate_resolution
    return None


def _gst_command(
    *,
    device: str,
    pixel_format: str,
    width: int,
    height: int,
    fps: Fraction,
    duration_s: float,
) -> list[str]:
    frame_count = max(1, int(float(fps) * duration_s))
    framerate = f"{fps.numerator}/{fps.denominator}"
    if pixel_format in {"MJPG", "MJPEG"}:
        caps = f"image/jpeg,width={width},height={height},framerate={framerate}"
        decode = ["jpegparse", "!", "nvv4l2decoder", "mjpeg=1", "!"]
    else:
        gst_format = "YUY2" if pixel_format == "YUYV" else pixel_format
        caps = f"video/x-raw,format={gst_format},width={width},height={height},framerate={framerate}"
        decode = []
    return [
        "gst-launch-1.0",
        "v4l2src",
        f"device={device}",
        "do-timestamp=true",
        f"num-buffers={frame_count}",
        "!",
        caps,
        "!",
        *decode,
        "fpsdisplaysink",
        "video-sink=fakesink",
        "text-overlay=false",
        "sync=false",
    ]


def _start_tegrastats() -> subprocess.Popen[str] | None:
    if shutil.which("tegrastats") is None:
        return None
    try:
        return subprocess.Popen(
            ["tegrastats", "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return None


def _stop_tegrastats(process: subprocess.Popen[str] | None) -> str:
    if process is None:
        return ""
    try:
        process.terminate()
        output, _ = process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate(timeout=2)
    return _tail(output or "")


def _parse_resolutions(values: list[str]) -> list[tuple[int, int]]:
    if not values:
        return [(1920, 1080), (2560, 1440)]
    resolutions: list[tuple[int, int]] = []
    for value in values:
        width, height = value.lower().split("x", 1)
        resolutions.append((int(width), int(height)))
    return resolutions


def _parse_reported_fps(output: str) -> float | None:
    matches = [float(match.group(1)) for match in _FPS_RE.finditer(output)]
    return matches[-1] if matches else None


def _fraction_to_json(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _tail(text: str, *, lines: int = 20) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
