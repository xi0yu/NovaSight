#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: dict[str, Any]


@dataclass(frozen=True)
class DeploymentCheckSummary:
    unit_path: str
    base_url: str
    checks: list[CheckResult]
    passed: bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the NovaSight Rust daemon deployment.")
    parser.add_argument("--unit", default="deploy/novasight.service")
    parser.add_argument("--base-url", default="http://127.0.0.1:5174")
    parser.add_argument("--release-root", type=Path, help="Validate a staged release tree.")
    parser.add_argument("--header", action="append", default=[], help="Extra HTTP header as Name: Value.")
    parser.add_argument("--skip-api", action="store_true", help="Only check the systemd unit file.")
    args = parser.parse_args()

    summary = run_deployment_check(
        unit_path=Path(args.unit),
        base_url=args.base_url,
        headers=_parse_headers(args.header),
        skip_api=args.skip_api,
        release_root=args.release_root,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary.passed else 2


def run_deployment_check(
    *,
    unit_path: Path,
    base_url: str,
    headers: dict[str, str],
    skip_api: bool,
    release_root: Path | None = None,
) -> DeploymentCheckSummary:
    unit = _read_unit(unit_path)
    checks = [
        _unit_check(unit, "Service", "Type", "simple"),
        _unit_check(unit, "Service", "Restart", "on-failure"),
        _unit_check(unit, "Service", "StandardOutput", "journal"),
        _unit_check(unit, "Service", "StandardError", "journal"),
        _unit_check(unit, "Service", "RuntimeDirectory", "novasight"),
        _unit_check(unit, "Service", "LogsDirectory", "novasight"),
        _environment_check(
            unit,
            expected="NOVASIGHT_INSTANCE_LOCK=/run/novasight/instance.lock",
        ),
        _unit_check(
            unit,
            "Service",
            "EnvironmentFile",
            "-/etc/novasight/novasight.env",
        ),
        _exec_start_pre_check(unit),
        _exec_start_check(unit),
    ]
    if not skip_api:
        checks.extend(_api_checks(base_url=base_url, headers=headers))
    if release_root is not None:
        checks.extend(_release_checks(release_root))
    return DeploymentCheckSummary(
        unit_path=str(unit_path),
        base_url=base_url,
        checks=checks,
        passed=all(check.passed for check in checks),
    )


def _release_checks(root: Path) -> list[CheckResult]:
    required = (
        "bin/novasightd",
        "bin/novasightctl",
        "lib/libnovasight_deepstream_bridge.so",
        "lib/libnovasight_parser.so",
        "scripts/model_ingress_job.py",
        "deploy/novasight.service",
        "share/novasight/novasight.production.yaml",
        "novasight/__init__.py",
        "novasight/executors/kmnet_host.py",
        "novasight/executors/kmnet_loader.py",
    )
    checks = [
        CheckResult(
            name=f"release.{relative}.present",
            passed=(root / relative).is_file(),
            detail={"path": str(root / relative)},
        )
        for relative in required
    ]
    config = root / "share/novasight/novasight.production.yaml"
    contents = config.read_text(encoding="utf-8") if config.is_file() else ""
    expected_paths = (
        "deepstream_parser_library: /opt/novasight/lib/libnovasight_parser.so",
        "deepstream_nvinfer_config: /var/lib/novasight/runtime/deepstream/active-nvinfer.ini",
        "data_dir: /var/lib/novasight",
        "database: /var/lib/novasight/novasight.db",
        "license: /var/lib/novasight/license.json",
        "python_executable: /usr/bin/python3",
    )
    checks.append(
        CheckResult(
            name="release.production_config.absolute_paths",
            passed=all(value in contents for value in expected_paths),
            detail={"path": str(config), "required_values": list(expected_paths)},
        )
    )
    checks.append(_kmnet_helper_check(root))
    return checks


def _kmnet_helper_check(root: Path) -> CheckResult:
    request = '{"id":1,"op":"hello","protocol":1}\n{"id":2,"op":"shutdown"}\n'
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-m", "novasight.executors.kmnet_host"],
            cwd=root,
            input=request,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return CheckResult(
            name="release.python_helper.handshake",
            passed=False,
            detail={"error": str(error)},
        )
    lines = result.stdout.splitlines()
    hello = _decode_json(lines[0].encode()) if lines else None
    stopped = _decode_json(lines[1].encode()) if len(lines) > 1 else None
    hello_result = hello.get("result") if isinstance(hello, dict) else None
    passed = (
        result.returncode == 0
        and isinstance(hello, dict)
        and hello.get("ok") is True
        and isinstance(hello_result, dict)
        and hello_result.get("protocol") == 1
        and isinstance(stopped, dict)
        and stopped.get("ok") is True
    )
    return CheckResult(
        name="release.python_helper.handshake",
        passed=passed,
        detail={
            "returncode": result.returncode,
            "hello": hello,
            "shutdown": stopped,
            "stderr": result.stderr,
        },
    )


def _read_unit(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str  # type: ignore[method-assign]
    with path.open("r", encoding="utf-8") as handle:
        parser.read_file(handle)
    return parser


def _unit_check(
    unit: configparser.ConfigParser,
    section: str,
    key: str,
    expected: str,
) -> CheckResult:
    actual = unit.get(section, key, fallback="")
    return CheckResult(
        name=f"unit.{section}.{key}",
        passed=actual == expected,
        detail={
            "actual": actual,
            "expected": expected,
        },
    )


def _environment_check(unit: configparser.ConfigParser, *, expected: str) -> CheckResult:
    values = unit.get("Service", "Environment", fallback="")
    actual = values if isinstance(values, str) else ""
    return CheckResult(
        name="unit.Service.Environment.instance_lock",
        passed=expected in actual,
        detail={
            "actual": actual,
            "expected_contains": expected,
        },
    )


def _exec_start_check(unit: configparser.ConfigParser) -> CheckResult:
    actual = unit.get("Service", "ExecStart", fallback="")
    expected = "/opt/novasight/bin/novasightd"
    return CheckResult(
        name="unit.Service.ExecStart.novasightd",
        passed=actual == expected or actual.startswith(f"{expected} "),
        detail={
            "actual": actual,
            "expected_executable": expected,
        },
    )


def _exec_start_pre_check(unit: configparser.ConfigParser) -> CheckResult:
    actual = unit.get("Service", "ExecStartPre", fallback="")
    executable = "/opt/novasight/bin/novasightd"
    required = (
        "--config /etc/novasight/novasight.yaml",
        "--model-job-script /opt/novasight/scripts/model_ingress_job.py",
        "--check",
    )
    return CheckResult(
        name="unit.Service.ExecStartPre.production_preflight",
        passed=actual.startswith(f"{executable} ") and all(value in actual for value in required),
        detail={
            "actual": actual,
            "expected_executable": executable,
            "required_arguments": list(required),
        },
    )


def _api_checks(*, base_url: str, headers: dict[str, str]) -> list[CheckResult]:
    health = _request_json(base_url, "/healthz", headers=headers)
    status = _request_json(base_url, "/api/v1/status", headers=headers)
    status_body = status.get("body") if isinstance(status.get("body"), dict) else {}
    return [
        CheckResult(
            name="api.healthz",
            passed=health.get("status_code") == 200 and bool((health.get("body") or {}).get("ok")),
            detail=health,
        ),
        CheckResult(
            name="api.v1.status.runtime_snapshot",
            passed=(
                status.get("status_code") == 200
                and isinstance(status_body.get("daemon"), dict)
                and isinstance(status_body.get("pipeline"), dict)
                and isinstance(status_body.get("subsystems"), dict)
            ),
            detail=status,
        ),
    ]


def _request_json(base_url: str, path: str, *, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(f"{base_url.rstrip('/')}{path}", headers=headers)
    try:
        with urlopen(request, timeout=5.0) as response:
            return {
                "status_code": int(response.status),
                "body": _decode_json(response.read()),
            }
    except HTTPError as exc:
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


def _parse_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for value in values:
        name, sep, header_value = value.partition(":")
        if sep and name.strip():
            headers[name.strip()] = header_value.strip()
    return headers


if __name__ == "__main__":
    raise SystemExit(main())
