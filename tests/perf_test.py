#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import statistics
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class PerfSummary:
    samples: int
    duration_s: float
    e2e_latency_ms: dict[str, float]
    capture_fps: dict[str, float]
    inference_fps: dict[str, float]
    dropped_frames: int
    covered_frames: int
    stale_detection_samples: int
    gates: dict[str, dict[str, Any]]
    passed: bool
    errors: list[str]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample NovaSight telemetry and summarize latency.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/api/runtime/state")
    parser.add_argument("--duration-s", type=float, default=3600.0)
    parser.add_argument("--interval-s", type=float, default=0.05)
    parser.add_argument("--header", action="append", default=[], help="Extra HTTP header as Name: Value.")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-p95-ms", type=float, default=20.0)
    parser.add_argument("--target-fps", type=float, default=0.0)
    parser.add_argument("--fps-tolerance-pct", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=1)
    args = parser.parse_args()

    summary = run_perf_probe(
        base_url=args.base_url,
        endpoint=args.endpoint,
        duration_s=args.duration_s,
        interval_s=args.interval_s,
        headers=_parse_headers(args.header),
        max_p95_ms=args.max_p95_ms,
        target_fps=args.target_fps,
        fps_tolerance_pct=args.fps_tolerance_pct,
        min_samples=args.min_samples,
    )
    payload = json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    print(payload)
    return 0 if summary.passed else 2


def run_perf_probe(
    *,
    base_url: str,
    endpoint: str,
    duration_s: float,
    interval_s: float,
    headers: dict[str, str] | None = None,
    max_p95_ms: float = 20.0,
    target_fps: float = 0.0,
    fps_tolerance_pct: float = 1.0,
    min_samples: int = 1,
) -> PerfSummary:
    start = time.monotonic()
    deadline = start + max(0.0, duration_s)
    interval = max(0.01, interval_s)
    samples: list[dict[str, Any]] = []
    errors: list[str] = []
    while time.monotonic() < deadline:
        try:
            samples.append(_fetch_json(f"{base_url.rstrip('/')}{endpoint}", headers=headers or {}))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors.append(str(exc))
        time.sleep(interval)
    elapsed = time.monotonic() - start
    stats = [_statistics(sample) for sample in samples]
    e2e = [_stat_number(stat, "e2e_latency_ms", "e2e_latency", "end_to_end_latency_ms") for stat in stats]
    capture_fps = [_stat_number(stat, "capture_fps", "tensor_meta_fps") for stat in stats]
    inference_fps = [_stat_number(stat, "inference_fps", "detection_batch_fps", "postprocess_fps") for stat in stats]
    stale_detection_samples = sum(
        1
        for stat in stats
        if _stat_number(stat, "last_frame_age_ms") > max(1000.0 * interval * 2.0, 100.0)
    )
    dropped_frames = int(
        max(
            (
                _stat_number(stat, "dropped_counter", "frames_dropped", "skipped_counter")
                for stat in stats
            ),
            default=0.0,
        )
    )
    covered_frames = int(
        max(
            (
                _stat_number(
                    stat,
                    "control_observation_counter",
                    "inference_counter",
                    "capture_counter",
                    "published_batches",
                )
                for stat in stats
            ),
            default=0.0,
        )
    )
    latency = _percentiles(e2e)
    capture = _percentiles(capture_fps)
    inference = _percentiles(inference_fps)
    gates = _evaluate_gates(
        samples=len(samples),
        errors=errors,
        e2e_latency_ms=latency,
        capture_fps=capture,
        max_p95_ms=max_p95_ms,
        target_fps=target_fps,
        fps_tolerance_pct=fps_tolerance_pct,
        min_samples=min_samples,
    )
    passed = all(gate["passed"] for gate in gates.values())
    return PerfSummary(
        samples=len(samples),
        duration_s=round(elapsed, 3),
        e2e_latency_ms=latency,
        capture_fps=capture,
        inference_fps=inference,
        dropped_frames=dropped_frames,
        covered_frames=covered_frames,
        stale_detection_samples=stale_detection_samples,
        gates=gates,
        passed=passed,
        errors=errors[-20:],
    )


def _evaluate_gates(
    *,
    samples: int,
    errors: list[str],
    e2e_latency_ms: dict[str, float],
    capture_fps: dict[str, float],
    max_p95_ms: float,
    target_fps: float,
    fps_tolerance_pct: float,
    min_samples: int,
) -> dict[str, dict[str, Any]]:
    gates: dict[str, dict[str, Any]] = {}
    gates["samples"] = _gate(
        samples >= max(1, min_samples),
        value=samples,
        limit=max(1, min_samples),
        message="sample count meets minimum",
    )
    gates["errors"] = _gate(
        not errors,
        value=len(errors),
        limit=0,
        message="telemetry requests completed without errors",
    )
    gates["e2e_latency_p95_ms"] = _gate(
        e2e_latency_ms["p95"] <= max_p95_ms,
        value=e2e_latency_ms["p95"],
        limit=max_p95_ms,
        message="p95 end-to-end latency is within budget",
    )
    if target_fps > 0.0:
        tolerance = abs(target_fps) * max(0.0, fps_tolerance_pct) / 100.0
        lower = target_fps - tolerance
        upper = target_fps + tolerance
        value = capture_fps["p50"]
        gates["capture_fps_p50"] = {
            "passed": lower <= value <= upper,
            "value": value,
            "target": target_fps,
            "tolerance_pct": fps_tolerance_pct,
            "lower": round(lower, 3),
            "upper": round(upper, 3),
            "message": "capture FPS p50 is within target tolerance",
        }
    else:
        gates["capture_fps_p50"] = {
            "passed": True,
            "value": capture_fps["p50"],
            "target": None,
            "tolerance_pct": fps_tolerance_pct,
            "message": "capture FPS gate skipped because --target-fps was not set",
        }
    return gates


def _gate(passed: bool, *, value: Any, limit: Any, message: str) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "value": value,
        "limit": limit,
        "message": message,
    }


def _fetch_json(url: str, *, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(url, headers=headers)
    with urlopen(request, timeout=2.0) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("telemetry endpoint did not return a JSON object")
    return payload


def _statistics(sample: dict[str, Any]) -> dict[str, Any]:
    value = sample.get("statistics", {})
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _stat_number(stat: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in stat:
            return _number(stat.get(key))
    return 0.0


def _percentiles(values: list[float]) -> dict[str, float]:
    values = [value for value in values if value >= 0.0]
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    sorted_values = sorted(values)
    return {
        "p50": round(statistics.median(sorted_values), 3),
        "p95": round(_percentile(sorted_values, 0.95), 3),
        "p99": round(_percentile(sorted_values, 0.99), 3),
        "min": round(sorted_values[0], 3),
        "max": round(sorted_values[-1], 3),
    }


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * quantile))))
    return sorted_values[index]


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, sep, header_value = value.partition(":")
        if sep and name.strip():
            headers[name.strip()] = header_value.strip()
    return headers


if __name__ == "__main__":
    raise SystemExit(main())
