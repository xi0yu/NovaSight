#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from novasight.api import create_app
from novasight.config import load_runtime_config
from novasight.runtime.pipeline_factory import create_runtime_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NovaSight DeepStream object-meta freshness gate."
    )
    parser.add_argument("--config", default="config/novasight.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--max-p50-batch-age-ms", type=float, default=15.0)
    parser.add_argument("--max-p95-batch-age-ms", type=float, default=30.0)
    parser.add_argument("--max-age-drift-ms", type=float, default=10.0)
    parser.add_argument("--report-json", default="/tmp/novasight-deepstream-60s.json")
    parser.add_argument("--allow-short", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seconds < 60.0 and not args.allow_short:
        print("reason: verification_requires_at_least_60_seconds", file=sys.stderr)
        return 2
    config = load_runtime_config(args.config)
    if config.inference.backend != "deepstream_nvinfer":
        print("reason: inference.backend must be deepstream_nvinfer", file=sys.stderr)
        return 2
    config.hardware.auto_connect = False
    config.control.trigger_mode = "hardware"
    app = create_app(data_dir=args.data_dir, config_path=args.config, config=config)
    runtime = app.state.runtime
    pipeline = create_runtime_pipeline(capture=app.state.capture, runtime=runtime)
    runtime.pipeline = pipeline
    samples: list[dict[str, Any]] = []
    started_ns = time.monotonic_ns()
    try:
        pipeline.start()
        deadline = time.monotonic() + max(0.0, args.seconds)
        while time.monotonic() < deadline:
            time.sleep(max(0.05, args.sample_interval))
            status = pipeline.status()
            deepstream = dict(status.get("deepstream", {}))
            samples.append(
                {
                    "elapsed_s": (time.monotonic_ns() - started_ns) / 1e9,
                    "running": bool(status.get("running")),
                    "published_batches": int(deepstream.get("published_batches", 0)),
                    "last_frame_id": int(deepstream.get("last_frame_id", -1)),
                    "last_batch_age_ms": float(deepstream.get("last_batch_age_ms", 0.0)),
                    "timestamp_source": str(deepstream.get("timestamp_source", "")),
                    "parser": dict(deepstream.get("parser", {})),
                    "mailbox": dict(deepstream.get("detection_batch_mailbox", {})),
                    "batch_age_ms_stats": dict(deepstream.get("batch_age_ms_stats", {})),
                    "last_error": str(deepstream.get("last_error", "")),
                }
            )
            if not status.get("running"):
                break
    except Exception as exc:
        samples.append({"elapsed_s": 0.0, "running": False, "last_error": str(exc)})
    finally:
        pipeline.stop()
    report = evaluate(samples, args)
    report["duration_s"] = (time.monotonic_ns() - started_ns) / 1e9
    report["samples"] = samples
    report["measurement_gaps"] = [
        "TensorRT kernel time is not exposed per frame by nvinfer",
        "DeepStream cluster-mode=2 NMS time is not exposed per frame",
    ]
    target = Path(args.report_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    print(f"report_json: {target}")
    return 0 if report["passed"] else 2


def evaluate(samples: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    if not samples:
        failures.append("no telemetry samples")
        return {"passed": False, "failures": failures}
    final = samples[-1]
    parser = final.get("parser", {})
    mailbox = final.get("mailbox", {})
    age_stats = final.get("batch_age_ms_stats", {})
    if int(final.get("published_batches", 0)) <= 0:
        failures.append("no DetectionBatch was published")
    if final.get("timestamp_source") != "gst_clock_base_time_pts":
        failures.append("capture timestamp is not mapped from Gst clock/base-time PTS")
    if parser.get("available") is not True or int(parser.get("decode_calls", 0)) <= 0:
        failures.append("native parser did not report successful decode calls")
    if int(parser.get("parse_failures", 0)) > 0:
        failures.append(
            "native parser reported output-contract failures "
            f"(last_error_code={int(parser.get('last_error_code', 0))})"
        )
    if int(mailbox.get("max_pending_depth", -1)) != 1:
        failures.append("DetectionBatch mailbox is not capacity one")
    p50 = float(age_stats.get("p50", 0.0))
    p95 = float(age_stats.get("p95", 0.0))
    if p50 > args.max_p50_batch_age_ms:
        failures.append(f"batch_age p50 {p50:.2f}ms exceeds {args.max_p50_batch_age_ms:.2f}ms")
    if p95 > args.max_p95_batch_age_ms:
        failures.append(f"batch_age p95 {p95:.2f}ms exceeds {args.max_p95_batch_age_ms:.2f}ms")
    ages = [
        float(sample["last_batch_age_ms"])
        for sample in samples
        if int(sample.get("published_batches", 0)) > 0
    ]
    drift = _window_drift(ages)
    if drift > args.max_age_drift_ms:
        failures.append(f"batch_age drift {drift:.2f}ms exceeds {args.max_age_drift_ms:.2f}ms")
    return {
        "passed": not failures,
        "failures": failures,
        "published_batches": int(final.get("published_batches", 0)),
        "last_frame_id": int(final.get("last_frame_id", -1)),
        "timestamp_source": str(final.get("timestamp_source", "")),
        "batch_age_ms": {"p50": p50, "p95": p95, "drift": drift},
        "parser": parser,
        "mailbox": mailbox,
    }


def _window_drift(values: list[float]) -> float:
    if len(values) < 4:
        return 0.0
    window = max(1, min(5, len(values) // 2))
    return max(0.0, statistics.mean(values[-window:]) - statistics.mean(values[:window]))


if __name__ == "__main__":
    raise SystemExit(main())
