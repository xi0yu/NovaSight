#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import math
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ProbeResult:
    name: str
    passed: bool
    detail: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NovaSight Phase 9 fault-injection probes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--header", action="append", default=[], help="Extra HTTP header as Name: Value.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bad_model = subparsers.add_parser("bad-model", help="Submit an invalid ONNX and verify service health survives.")
    bad_model.add_argument("--model-id", default="fault_bad_model")
    bad_model.add_argument("--source-path", default="")

    capture_loss = subparsers.add_parser("capture-loss", help="Poll runtime while the capture device is unplugged.")
    capture_loss.add_argument("--duration-s", type=float, default=30.0)
    capture_loss.add_argument("--interval-s", type=float, default=0.5)

    overload = subparsers.add_parser("overload", help="Run a load command while probing telemetry gates.")
    overload.add_argument("--duration-s", type=float, default=120.0)
    overload.add_argument("--interval-s", type=float, default=0.05)
    overload.add_argument("--max-p95-ms", type=float, default=20.0)
    overload.add_argument("--target-fps", type=float, default=0.0)
    overload.add_argument("--fps-tolerance-pct", type=float, default=1.0)
    overload.add_argument("--min-samples", type=int, default=10)
    overload.add_argument(
        "--stress-command",
        default="",
        help="Optional load command. Defaults to stress-ng if installed.",
    )

    args = parser.parse_args()
    headers = _parse_headers(args.header)
    if args.command == "bad-model":
        result = run_bad_model_probe(
            base_url=args.base_url,
            headers=headers,
            model_id=args.model_id,
            source_path=args.source_path,
        )
    elif args.command == "capture-loss":
        result = run_capture_loss_probe(
            base_url=args.base_url,
            headers=headers,
            duration_s=args.duration_s,
            interval_s=args.interval_s,
        )
    elif args.command == "overload":
        result = run_overload_probe(
            base_url=args.base_url,
            headers=headers,
            duration_s=args.duration_s,
            interval_s=args.interval_s,
            max_p95_ms=args.max_p95_ms,
            target_fps=args.target_fps,
            fps_tolerance_pct=args.fps_tolerance_pct,
            min_samples=args.min_samples,
            stress_command=args.stress_command,
        )
    else:
        parser.error(f"unknown command: {args.command}")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.passed else 2


def run_bad_model_probe(
    *,
    base_url: str,
    headers: dict[str, str],
    model_id: str,
    source_path: str,
) -> ProbeResult:
    created_fixture = not source_path
    invalid_path = (
        Path(source_path)
        if source_path
        else _write_invalid_onnx(model_id=model_id)
    )
    try:
        models_root = Path("models").resolve(strict=False)
        try:
            relative_path = invalid_path.resolve(strict=False).relative_to(models_root).as_posix()
        except ValueError:
            relative_path = str(invalid_path)
        registration_response = _request_json(
            base_url,
            "/api/models/catalog/register",
            method="POST",
            headers={**headers, "Content-Type": "application/json"},
            body={"relative_path": relative_path},
            allow_http_error=True,
        )
        health_response = _request_json(
            base_url,
            "/healthz",
            headers=headers,
            allow_http_error=True,
        )
        runtime_response = _request_json(
            base_url,
            "/api/runtime/state",
            headers=headers,
            allow_http_error=True,
        )
    finally:
        if created_fixture:
            invalid_path.unlink(missing_ok=True)

    registration_body = registration_response.get("body") or {}
    registration_detail = (
        str(registration_body.get("detail", ""))
        if isinstance(registration_body, dict)
        else str(registration_body)
    )
    model_rejected = (
        int(registration_response.get("status_code", 0)) == 400
        and "TensorRT .engine files only" in registration_detail
    )
    health_ok = bool((health_response.get("body") or {}).get("ok")) and int(health_response.get("status_code", 0)) == 200
    runtime_serving = int(runtime_response.get("status_code", 0)) in {200, 401}
    return ProbeResult(
        name="bad_model_isolated_failure",
        passed=model_rejected and health_ok and runtime_serving,
        detail={
            "invalid_source_path": str(invalid_path),
            "registration_status_code": registration_response.get("status_code"),
            "registration_body": registration_response.get("body"),
            "health_status_code": health_response.get("status_code"),
            "runtime_status_code": runtime_response.get("status_code"),
            "expectation": (
                "unsupported model registration returns the expected 400 validation error "
                "while health/runtime endpoints still serve"
            ),
        },
    )


def run_capture_loss_probe(
    *,
    base_url: str,
    headers: dict[str, str],
    duration_s: float,
    interval_s: float,
) -> ProbeResult:
    deadline = time.monotonic() + max(0.0, duration_s)
    interval = max(0.05, interval_s)
    samples = 0
    unavailable_samples = 0
    last_capture: dict[str, Any] = {}
    last_status_code = 0
    while time.monotonic() < deadline:
        response = _request_json(base_url, "/api/runtime/state", headers=headers, allow_http_error=True)
        last_status_code = int(response.get("status_code", 0))
        body = response.get("body")
        if isinstance(body, dict):
            capture = body.get("capture")
            if isinstance(capture, dict):
                samples += 1
                last_capture = capture
                if capture.get("available") is False:
                    unavailable_samples += 1
        time.sleep(interval)
    return ProbeResult(
        name="capture_loss_observed",
        passed=samples > 0 and unavailable_samples > 0,
        detail={
            "samples": samples,
            "unavailable_samples": unavailable_samples,
            "last_status_code": last_status_code,
            "last_capture": last_capture,
            "expectation": "unplug capture device during the window; runtime state should report capture.available=false",
        },
    )


def run_overload_probe(
    *,
    base_url: str,
    headers: dict[str, str],
    duration_s: float,
    interval_s: float,
    max_p95_ms: float,
    target_fps: float,
    fps_tolerance_pct: float,
    min_samples: int,
    stress_command: str,
) -> ProbeResult:
    _ensure_repo_root_importable()
    from tests.perf_test import run_perf_probe

    command = _load_command(stress_command, duration_s)
    process = _start_load_process(command)
    try:
        summary = run_perf_probe(
            base_url=base_url,
            endpoint="/api/runtime/state",
            duration_s=duration_s,
            interval_s=interval_s,
            headers=headers,
            max_p95_ms=max_p95_ms,
            target_fps=target_fps,
            fps_tolerance_pct=fps_tolerance_pct,
            min_samples=min_samples,
        )
    finally:
        _stop_load_process(process)
    detail = {
        "load_command": command,
        "load_started": process is not None,
        "load_exit_code": process.poll() if process is not None else None,
        "perf": asdict(summary),
        "expectation": "runtime keeps serving telemetry under load and latency gates remain bounded",
    }
    return ProbeResult(
        name="inference_overload_bounded_latency",
        passed=summary.passed,
        detail=detail,
    )


def _ensure_repo_root_importable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)


def _load_command(value: str, duration_s: float) -> list[str]:
    if value.strip():
        return shlex.split(value)
    stress_ng = shutil.which("stress-ng")
    if stress_ng is None:
        return []
    timeout_s = max(1, int(math.ceil(duration_s)))
    return [stress_ng, "--cpu", "0", "--timeout", f"{timeout_s}s"]


def _start_load_process(command: list[str]) -> subprocess.Popen[bytes] | None:
    if not command:
        return None
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_load_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
    allow_http_error: bool = False,
) -> dict[str, Any]:
    encoded_body = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=encoded_body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=5.0) as response:
            return {
                "status_code": int(response.status),
                "body": _decode_json(response.read()),
            }
    except HTTPError as exc:
        if not allow_http_error:
            raise
        return {
            "status_code": int(exc.code),
            "body": _decode_json(exc.read()),
        }
    except (TimeoutError, URLError) as exc:
        return {
            "status_code": 0,
            "body": {"detail": str(exc)},
        }


def _decode_json(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _write_invalid_onnx(*, model_id: str) -> Path:
    models_root = Path("models")
    models_root.mkdir(parents=True, exist_ok=True)
    safe_model_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in model_id
    ).strip("_") or "fault_bad_model"
    handle, raw_path = tempfile.mkstemp(
        prefix=f"{safe_model_id}.",
        suffix=".onnx",
        dir=models_root,
    )
    path = Path(raw_path)
    with open(handle, "wb", closefd=True) as output:
        output.write(b"not an onnx model\n")
    return path


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, sep, header_value = value.partition(":")
        if sep and name.strip():
            headers[name.strip()] = header_value.strip()
    return headers


if __name__ == "__main__":
    raise SystemExit(main())
