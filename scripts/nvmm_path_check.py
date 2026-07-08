#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Any

from novasight.pipelines.gst_validator import (
    build_capture_nvmm_pipeline,
    validate_nvmm_pipeline,
)


@dataclass(frozen=True)
class NvmmCheck:
    device: str
    pixel_format: str
    width: int
    height: int
    fps: str
    pipeline: str
    validated: bool
    result: dict[str, Any] | None
    passed: bool


@dataclass(frozen=True)
class NvmmCheckSummary:
    checks: list[NvmmCheck]
    passed: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 2 capture pipelines produce NVMM buffers.")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--resolution", action="append", default=[], help="WIDTHxHEIGHT. May be repeated.")
    parser.add_argument("--formats", default="MJPG,NV12")
    parser.add_argument("--fps", default="60")
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="Only render pipeline descriptions.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    summary = run_nvmm_checks(
        device=args.device,
        resolutions=_parse_resolutions(args.resolution),
        pixel_formats=[item.strip().upper() for item in args.formats.split(",") if item.strip()],
        fps=_parse_fraction(args.fps),
        timeout_s=args.timeout_s,
        dry_run=args.dry_run,
    )
    payload = json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if summary.passed else 2


def run_nvmm_checks(
    *,
    device: str,
    resolutions: list[tuple[int, int]],
    pixel_formats: list[str],
    fps: Fraction,
    timeout_s: float,
    dry_run: bool,
) -> NvmmCheckSummary:
    checks: list[NvmmCheck] = []
    for width, height in resolutions:
        for pixel_format in pixel_formats:
            pipeline = build_capture_nvmm_pipeline(
                device=device,
                pixel_format=pixel_format,
                width=width,
                height=height,
                fps=fps,
            )
            result = None if dry_run else asdict(validate_nvmm_pipeline(pipeline, timeout_s=timeout_s))
            passed = dry_run or bool(result and result.get("ok") is True)
            checks.append(
                NvmmCheck(
                    device=device,
                    pixel_format=pixel_format,
                    width=width,
                    height=height,
                    fps=_fraction_text(fps),
                    pipeline=pipeline,
                    validated=not dry_run,
                    result=result,
                    passed=passed,
                )
            )
    return NvmmCheckSummary(checks=checks, passed=all(check.passed for check in checks))


def _parse_resolutions(values: list[str]) -> list[tuple[int, int]]:
    if not values:
        return [(1920, 1080), (2560, 1440)]
    return [_parse_resolution(value) for value in values]


def _parse_resolution(value: str) -> tuple[int, int]:
    width, sep, height = value.lower().partition("x")
    if not sep:
        raise ValueError(f"resolution must be WIDTHxHEIGHT: {value}")
    return int(width), int(height)


def _parse_fraction(value: str) -> Fraction:
    text = value.strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(int(text), 1)


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


if __name__ == "__main__":
    raise SystemExit(main())
