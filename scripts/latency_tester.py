#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


_TRACER_LATENCY_NS_RE = re.compile(r"(?:time|ts|latency)=\(guint64\)(\d+)")
_TRACER_SIMPLE_NS_RE = re.compile(r"latency[^\n\r]*?(\d{5,})\s*ns", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure NovaSight Phase 0.3 baseline latency and write baseline_latency.json."
    )
    parser.add_argument("--output", default="baseline_latency.json")
    parser.add_argument(
        "--recording-csv",
        default=None,
        help="Control-frame CSV from NovaSight recording. Preferred when capture+inference+HID ran.",
    )
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--pixel-format", default="MJPG", choices=["MJPG", "MJPEG", "NV12", "YUYV"])
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--sink",
        default="fakesink",
        choices=["fakesink", "autovideosink"],
        help="Use autovideosink for capture-to-display fallback on a Jetson desktop.",
    )
    parser.add_argument(
        "--allow-videotest",
        action="store_true",
        help="Use videotestsrc when /dev/video* or gst-launch is unavailable only for script dry-runs.",
    )
    parser.add_argument(
        "--manual-sample-ms",
        action="append",
        type=float,
        default=[],
        help="Manually supplied LED/screen end-to-end sample in ms. May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recording_samples = _samples_from_recording(Path(args.recording_csv)) if args.recording_csv else []
    manual_samples = [float(value) for value in args.manual_sample_ms if float(value) >= 0.0]

    if recording_samples:
        mode = "control_recording_csv"
        samples = recording_samples
        source = str(Path(args.recording_csv))
        gst_result = None
    elif manual_samples:
        mode = "manual_led_or_screen_samples"
        samples = manual_samples
        source = "manual_sample_ms"
        gst_result = None
    else:
        mode = "gst_latency_tracer"
        gst_result = _run_gst_latency(args)
        samples = list(gst_result.get("latency_ms_samples", []))
        source = str(gst_result.get("command", ""))

    payload = {
        "generated_at": _now_iso(),
        "mode": mode,
        "source": source,
        "sample_count": len(samples),
        "latency_ms_stats": _stats(samples),
        "recording_csv": str(Path(args.recording_csv)) if args.recording_csv else "",
        "gst": gst_result,
        "notes": _notes(mode=mode, sample_count=len(samples), gst_result=gst_result),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output_path}")
    if not samples:
        print("warning: no latency samples collected; see notes in JSON")
    return 0


def _samples_from_recording(path: Path) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(f"recording CSV not found: {path}")
    samples: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample = _row_latency_ms(row)
            if sample is not None and sample >= 0.0:
                samples.append(sample)
    return samples


def _row_latency_ms(row: dict[str, str]) -> float | None:
    for field in (
        "e2e_latency_ms",
        "end_to_end_latency_ms",
        "measurement_age_ms",
        "capture_to_hid_ms",
    ):
        value = _float(row.get(field))
        if value is not None:
            return value
    inference_latency = _float(row.get("inference_latency_ms"))
    if inference_latency is not None:
        return inference_latency
    capture_ts_ns = _int(row.get("capture_ts_ns"))
    expires_ts_ns = _int(row.get("expires_ts_ns"))
    if capture_ts_ns is not None and expires_ts_ns is not None and expires_ts_ns >= capture_ts_ns:
        return (expires_ts_ns - capture_ts_ns) / 1e6
    return None


def _run_gst_latency(args: argparse.Namespace) -> dict[str, Any]:
    if shutil.which("gst-launch-1.0") is None:
        if not args.allow_videotest:
            return {
                "available": False,
                "skip_reason": "gst-launch-1.0 unavailable",
                "latency_ms_samples": [],
            }
    source = _gst_source(args)
    if source is None:
        return {
            "available": False,
            "skip_reason": f"capture device unavailable: {args.device}",
            "latency_ms_samples": [],
        }
    command = _gst_command(args, source=source)
    env = dict(os.environ)
    env["GST_TRACERS"] = "latency(flags=pipeline+element)"
    env["GST_DEBUG"] = "GST_TRACER:7"
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(2.0, float(args.duration) + 5.0),
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "command": " ".join(command),
            "returncode": None,
            "elapsed_s": time.monotonic() - started,
            "stable": False,
            "latency_ms_samples": [],
            "stderr_tail": _tail(str(exc)),
            "error": "gst-launch latency tracer timed out",
        }
    output = "\n".join([completed.stdout, completed.stderr])
    samples = _parse_gst_latency_ms(output)
    return {
        "available": True,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "elapsed_s": time.monotonic() - started,
        "stable": completed.returncode == 0,
        "latency_ms_samples": samples,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
    }


def _gst_source(args: argparse.Namespace) -> list[str] | None:
    if Path(args.device).exists():
        return [
            "v4l2src",
            f"device={args.device}",
            "do-timestamp=true",
            f"num-buffers={max(1, int(args.duration * args.fps))}",
        ]
    if args.allow_videotest and shutil.which("gst-launch-1.0") is not None:
        return [
            "videotestsrc",
            "is-live=true",
            f"num-buffers={max(1, int(args.duration * args.fps))}",
        ]
    return None


def _gst_command(args: argparse.Namespace, *, source: list[str]) -> list[str]:
    pixel_format = str(args.pixel_format).upper()
    if pixel_format in {"MJPG", "MJPEG"} and source[0] == "v4l2src":
        caps = f"image/jpeg,width={args.width},height={args.height},framerate={args.fps}/1"
        decode = ["jpegparse", "!", "jpegdec", "!"]
    else:
        gst_format = "YUY2" if pixel_format == "YUYV" else "NV12" if source[0] == "v4l2src" else "I420"
        caps = f"video/x-raw,format={gst_format},width={args.width},height={args.height},framerate={args.fps}/1"
        decode = []
    sink = ["fakesink", "sync=false"] if args.sink == "fakesink" else ["autovideosink", "sync=false"]
    return ["gst-launch-1.0", *source, "!", caps, "!", *decode, "videoconvert", "!", *sink]


def _parse_gst_latency_ms(output: str) -> list[float]:
    samples_ns: list[int] = []
    samples_ns.extend(int(match.group(1)) for match in _TRACER_LATENCY_NS_RE.finditer(output))
    samples_ns.extend(int(match.group(1)) for match in _TRACER_SIMPLE_NS_RE.finditer(output))
    return [value / 1e6 for value in samples_ns if value >= 0]


def _stats(samples: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in samples if value >= 0.0)
    if not ordered:
        return {"count": 0, "avg": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(ordered),
        "avg": mean(ordered),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _notes(*, mode: str, sample_count: int, gst_result: dict[str, Any] | None) -> list[str]:
    notes: list[str] = []
    if mode == "control_recording_csv":
        notes.append("Latency samples came from NovaSight control-frame recording rows.")
    elif mode == "manual_led_or_screen_samples":
        notes.append("Latency samples were manually supplied from LED/screen measurement.")
    else:
        notes.append("Latency samples came from GStreamer latency tracer output.")
    if sample_count == 0:
        notes.append("No p50/p95/p99 can be computed until samples are available.")
    if gst_result is not None and gst_result.get("skip_reason"):
        notes.append(str(gst_result["skip_reason"]))
    return notes


def _float(value: object) -> float | None:
    try:
        text = str(value if value is not None else "").strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _int(value: object) -> int | None:
    try:
        text = str(value if value is not None else "").strip()
        return int(float(text)) if text else None
    except (TypeError, ValueError):
        return None


def _tail(text: str, *, lines: int = 30) -> str:
    return "\n".join(str(text or "").splitlines()[-lines:])


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
