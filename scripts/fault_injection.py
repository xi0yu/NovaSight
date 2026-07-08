#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
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
    invalid_path = Path(source_path) if source_path else _write_invalid_onnx()
    payload = {
        "source_path": str(invalid_path),
        "model_id": model_id,
    }
    import_response = _request_json(
        base_url,
        "/api/v1/models/import",
        method="POST",
        headers={**headers, "Content-Type": "application/json"},
        body=payload,
        allow_http_error=True,
    )
    health_response = _request_json(base_url, "/healthz", headers=headers, allow_http_error=True)
    runtime_response = _request_json(base_url, "/api/runtime/state", headers=headers, allow_http_error=True)
    import_rejected = int(import_response.get("status_code", 0)) >= 400
    health_ok = bool((health_response.get("body") or {}).get("ok")) and int(health_response.get("status_code", 0)) == 200
    runtime_serving = int(runtime_response.get("status_code", 0)) in {200, 401}
    return ProbeResult(
        name="bad_model_isolated_failure",
        passed=import_rejected and health_ok and runtime_serving,
        detail={
            "invalid_source_path": str(invalid_path),
            "import_status_code": import_response.get("status_code"),
            "import_body": import_response.get("body"),
            "health_status_code": health_response.get("status_code"),
            "runtime_status_code": runtime_response.get("status_code"),
            "expectation": "invalid model import is rejected while health/runtime endpoints still serve",
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


def _write_invalid_onnx() -> Path:
    path = Path(tempfile.gettempdir()) / "novasight_fault_invalid.onnx"
    path.write_bytes(b"not an onnx model\n")
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
